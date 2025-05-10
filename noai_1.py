import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression #因為是分類題目，分成投中和投不中
df = pd.read_csv("./data_train.csv")
x, y = df.iloc[:, 0:2], df.iloc[:, 4] # x : feature, y : label
x = x.astype(float)
# print(x)
# print(y)
print(x.iloc[:, 0].min(), x.iloc[:, 0].max())
loc_x_min = x.iloc[:, 0].min()
loc_x_max = x.iloc[:, 0].max()
x.iloc[:, 0] -= x.iloc[:, 0].min() # 將所有數字右移變正數
x.iloc[:, 0] /= x.iloc[:, 0].max() #化身0至1之間的數字，稱為歸一化。
x.iloc[:, 1] -= x.iloc[:, 1].min()
x.iloc[:, 1] /= x.iloc[:, 1].max()
# 高斯分佈用標準化，線性分佈用歸一化
#plt.scatter(x.iloc[:, 0], x.iloc[:, 1], c=y) # c = y 看你y有多種顏色，系統自己幫你決定
#plt.show()
model = LogisticRegression()
model.fit(x, y)
print(model.score(x, y))