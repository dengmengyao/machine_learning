import numpy as np
import pandas as pd
from time import time

t1 = time()#开始时间
np.set_printoptions(threshold=2000)
data_train=pd.read_csv("semeion_train.csv",encoding="utf-8",header=None)
#print(data_train)
x_train=data_train[0]
#print(x_train)

Images_train = np.zeros([1115, 256])
Labels_train = np.zeros([1115, 10])
nums_train=np.zeros([1115],dtype=int)

for i in range(1115):#处理行数据进行分割，分别分割到Image和Labels数组中
    x_train[i]=x_train[i].rstrip(' ')
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
#print(nums_train)
Labels_test = np.zeros([478, 10])
nums_test=np.zeros([478],dtype=int)

for i in range(478):#处理行数据进行分割，分别分割到Image和Labels数组中
    x_test[i]=x_test[i].rstrip(' ')
    currentLine=x_test[i].split(' ')
    Images_test[i][:]=currentLine[:256]
    Labels_test[i][:]=currentLine[256:]

for i in range(478):#存放具体数值
    for j in range(10):
        if Labels_test[i][j]==1:
            nums_test[i]=j
            break

#print(nums_test)
class KNN:#使用KNN进行算法实现
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = np.asarray(X) #转换为ndarray类型
        self.y = np.asarray(y)

    def predict(self, X):
        X = np.asarray(X)
        result = []
        for x in X:
            dis = np.sqrt(np.sum((x-self.X)**2, axis=1)) # 对于测试机的每隔一个样本，一次与训练集的所有数据求欧氏距离
            index = dis.argsort()# 返回排序结果的下标
            index = index[:self.k] # 截取前K个数据，k为近零样本的数量
            count = np.bincount(self.y[index]) # 返回数组中每个整数元素出现次数，元素必须是非负整数(统计那个数字出现最多的索引)
            result.append(count.argmax()) # 返回ndarray中值最大的元素所对应的索引，就是出现次数最多的索引，也就是我们判定的类别（最终结果）
        return np.asarray(result)


knn1=KNN(k=1)
knn1.fit(Images_train,nums_train)
Images_predict1=knn1.predict(Images_test)
accuracy1=0
for i in range(478):
    if Images_predict1[i]==nums_test[i]:
        accuracy1+=1

print("n_neignbors=1时准确率为：",accuracy1/478)


knn2=KNN(k=3)
knn2.fit(Images_train,nums_train)
Images_predict2=knn2.predict(Images_test)
accuracy2=0
for i in range(478):
    if Images_predict2[i]==nums_test[i]:
        accuracy2+=1

print("n_neignbors=3时准确率为：",accuracy2/478)


knn3=KNN(k=5)
knn3.fit(Images_train,nums_train)
Images_predict3=knn1.predict(Images_test)
accuracy3=0
for i in range(478):
    if Images_predict3[i]==nums_test[i]:
        accuracy3+=1

print("n_neignbors=5时准确率为：",accuracy3/478)
t2=time()#结束时间
print("消耗时间为：",t2-t1,"s")
