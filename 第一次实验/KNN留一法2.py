import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier #KNN机器学习包
from sklearn.model_selection import LeaveOneOut#留一法分割

filename="semeion.data"
fr=open(filename,'r',encoding="utf-8")
arrayOfLines=fr.readlines()#按行读取对应的数据

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

loo1=LeaveOneOut()
knn1 = KNeighborsClassifier(n_neighbors=1)
accuracy1=0
for train,test in loo1.split(Images):
    knn1.fit(Images[train],nums[train])
    nums_p=knn1.predict(Images[test])
    if(nums_p==nums[test]):
        accuracy1+=1
print(accuracy1 / np.shape(Images)[0])



loo2=LeaveOneOut()
knn2 = KNeighborsClassifier(n_neighbors=3)
accuracy2=0
for train,test in loo2.split(Images):
    knn2.fit(Images[train],nums[train])
    nums_p=knn2.predict(Images[test])
    if(nums_p==nums[test]):
        accuracy2+=1
print(accuracy2 / np.shape(Images)[0])



loo3=LeaveOneOut()
knn3 = KNeighborsClassifier(n_neighbors=5)
accuracy3=0
for train,test in loo3.split(Images):
    knn3.fit(Images[train],nums[train])
    nums_p=knn3.predict(Images[test])
    if(nums_p==nums[test]):
        accuracy3+=1
print(accuracy3 / np.shape(Images)[0])