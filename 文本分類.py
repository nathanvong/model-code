import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, TabularDataset, BucketIterator

# 1. 讀取數據
df = pd.read_csv('data.csv')  # 請將 'data.csv' 替換為你的CSV檔名

# 2. 定義字段
TEXT = Field(tokenize='spacy', include_lengths=True)
LABEL = Field(sequential=False)

# 3. 創建數據集
datafields = [('text', TEXT), ('label', LABEL)]
train_data, valid_data = TabularDataset.splits(
    path='.',
    train='train.csv',  # 請根據需要創建訓練和驗證資料集
    validation='valid.csv',
    format='csv',
    fields=datafields
)

# 4. 建立詞彙表
TEXT.build_vocab(train_data, max_size=10000)
LABEL.build_vocab(train_data)

# 5. 創建迭代器
train_iterator, valid_iterator = BucketIterator.splits(
    (train_data, valid_data),
    batch_size=32,
    sort_within_batch=True,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# 6. 定義模型
class SentimentModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.rnn(embedded)
        return self.softmax(self.fc(hidden[-1]))

# 7. 設定模型參數
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = len(LABEL.vocab)

model = SentimentModel(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
optimizer = optim.Adam(model.parameters())
criterion = nn.NLLLoss()

# 8. 訓練模型
model.train()
for epoch in range(5):  # 訓練5個epoch
    for batch in train_iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.text
        predictions = model(text).squeeze(1)
        loss = criterion(predictions, batch.label)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# 9. 評估模型
model.eval()
# 評估代碼可以根據需要添加