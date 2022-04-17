import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut#留一法分割

data=pd.read_csv("semeion_train.csv",encoding="utf-8",header=-1)
data1=

Images = np.zeros([1593, 256])#存放前256位表示图像
Labels = np.zeros([1593, 10])#存放后十位表示具体数值
nums=np.zeros([1593],dtype=int)#存数值

for i in range(1593):#处理行数据进行分割，分别分割到Image和Labels数组中
    arrayOfLines[i]=arrayOfLines[i].rstrip(' \n')
    currentLine=arrayOfLines[i].split(' ')
    Images[i][:]=currentLine[:256]
    Labels[i][:]=currentLine[256:]

for i in range(1593):#存放具体数值
    for j in range(10):
        if Labels[i][j]==1:
            nums[1]=j
            break


class KNN:
    '''使用KNN实现K近邻算法实现分类'''
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
            index = index[:self.k] # 截取前K个
            count = np.bincount(self.y[index]) # 返回数组中每个整数元素出现次数，元素必须是非负整数
            result.append(count.argmax()) # 返回ndarray中值最大的元素所对应的索引，就是出现次数最多的索引，也就是我们判定的类别
        return np.asarray(result)


    def predict2(self, X):
        X = np.asarray(X)
        result = []
        for x in X:
            dis = np.sqrt(np.sum((x-self.X)**2, axis=1)) # 对于测试机的每隔一个样本，一次与训练集的所有数据求欧氏距离
            index = dis.argsort()# 返回排序结果的下标
            index = index[:self.k] # 截取前K个
            count = np.bincount(self.y[index], weights=1/dis[index]) # 返回数组中每个整数元素出现次数，元素必须是非负整数
            result.append(count.argmax()) # 返回ndarray中值最大的元素所对应的索引，就是出现次数最多的索引，也就是我们判定的类别
        return np.asarray(result)





loo1=LeaveOneOut()
knn1 = KNN(k=1)
accuracy1=0
for train,test in loo1.split(Images):
    knn1.fit(Images[train],nums[train])
    nums_p=knn1.predict(Images[test])
    if nums_p==nums[test]:
        accuracy1+=1
print("n_neignbors=1时准确率为：",accuracy1 / np.shape(Images)[0])


loo2=LeaveOneOut()
knn2 = KNN(k=3)
accuracy2=0
for train,test in loo2.split(Images):
    knn2.fit(Images[train],nums[train])
    nums_p=knn2.predict(Images[test])
    if nums_p==nums[test]:
        accuracy2+=1
print("n_neignbors=3时准确率为：",accuracy2 / np.shape(Images)[0])


loo3=LeaveOneOut()
knn3 = KNN(k=5)
accuracy3=0
for train,test in loo3.split(Images):
    knn3.fit(Images[train],nums[train])
    nums_p=knn3.predict(Images[test])
    if nums_p==nums[test]:
        accuracy3+=1
print("n_neignbors=5时准确率为：",accuracy3 / np.shape(Images)[0])


#labels = ['正确','错误']
#sizes = [accuracy1,np.shape(Images)[0]-accuracy1]
#explode = (0,0)
#plt.pie(sizes,explode=explode,labels=labels,autopct='%1.1f%%',shadow=False,startangle=150)
#plt.title("饼图示例")
#plt.show()  