import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier#调用KNN包
from time import time#调用time包测试程序运行时间

t1 = time()#开始时间
np.set_printoptions(threshold=2000)
data_train=pd.read_csv("semeion_train.csv",encoding="utf-8",header=None)
x_train=data_train[0]

Images_train = np.zeros([1115, 256])
Labels_train = np.zeros([1115, 10])
nums_train=np.zeros([1115],dtype=int)

for i in range(1115):#处理行数据进行分割，分别分割到Image和Labels数组中
    x_train[i]=x_train[i].rstrip(' \n')
    currentLine=x_train[i].split(' ')
    Images_train[i][:]=currentLine[:256]
    Labels_train[i][:]=currentLine[256:]

for i in range(1115):#存放具体数值
    for j in range(10):
        if Labels_train[i][j]==1:
            nums_train[i]=j
            break

#print(nums_train)  
data_test=pd.read_csv("semeion_test.csv",encoding="utf-8",header=None)
x_test=data_test[0]

Images_test = np.zeros([478, 256])
Labels_test = np.zeros([478, 10])
nums_test=np.zeros([478],dtype=int)

for i in range(478):#处理行数据进行分割，分别分割到Image和Labels数组中
    x_test[i]=x_test[i].rstrip(' \n')
    currentLine=x_test[i].split(' ')
    Images_test[i][:]=currentLine[:256]
    Labels_test[i][:]=currentLine[256:]

for i in range(478):#存放具体数值
    for j in range(10):
        if Labels_test[i][j]==1:
            nums_test[i]=j
            break

#print(nums_test)


knn1=KNeighborsClassifier(n_neighbors=1)#调用KNN包进行学习
print(nums_train)
knn1.fit(Images_train,nums_train)
Images_predict1=knn1.predict(Images_test)
score1 = knn1.score(Images_test, nums_test)


knn2=KNeighborsClassifier(n_neighbors=3)
knn2.fit(Images_train,nums_train)
Images_predict2=knn2.predict(Images_test)
score2 = knn2.score(Images_test, nums_test)


knn3=KNeighborsClassifier(n_neighbors=5)
knn3.fit(Images_train,nums_train)
Images_predict3=knn1.predict(Images_test)
score3 = knn3.score(Images_test, nums_test)



print("n_neignbors=1时准确率为：", score1)
print("n_neignbors=3时准确率为：", score2)
print("n_neignbors=5时准确率为：", score3)


t2=time()#结束时间
print("消耗时间为：",t2-t1,"s")#输出程序运行时间