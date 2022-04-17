import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler#归一化，标准化
from sklearn.model_selection import train_test_split

data=pd.read_csv("iris.arff.csv",header=0)
print(data.sample(10))
data["class"] = data["class"].map({"Iris-versicolor":0,"Iris-setosa":1,"Iris-virginica":2}) # 类别名称映射为数字

print(len(data))
if data.duplicated().any(): # 重复值
    data.drop_duplicates(inplace=True) #删除重复值
    print(len(data))
data["class"].value_counts()  # 查看各个类别的鸢尾花记录

y = pd.concat([data.iloc[:,:-1]], axis=0)
x= pd.concat([data.iloc[:,-1]], axis=0)
y_train, y_test,x_train, x_test = train_test_split( y, x,random_state=12)

transfer = StandardScaler(with_mean=False)
y_train = transfer.fit_transform(y_train)
y_test = transfer.transform(y_test)

knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(y_train,x_train)
y_predict=knn.predict(y_test)
score=knn.score(y_test,x_test)
print(score)










    