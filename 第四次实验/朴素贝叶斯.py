import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

datas=[]#对wine.data进行处理，一共有三类数据
data=open('wine.data')
for line in data.readlines():
    datas.append(line.replace('\n','').split(','))
datas=[[eval(s) for s in d] for d in datas]
    
for d in datas:
    if(len(d)<14):
        print("error")

data.close()
print("{}中总共记载了{}个样本:".format('wine.data',len(datas)))
#print(datas)

types=[[],[],[]]#按照数据的首标号对数据进行分类，并去除掉首标签
for i in datas:
    types[i[0]-1].append(i[1:])
#print(types)


def bayes_classfy(train,test,types):#朴素贝叶斯分类器
    data_num=0
    for i in range(3):
        data_num+=len(types[i])
    means=[np.mean(train[i],axis=0) for i in range(3)]#均值
    stds=[np.std(train[i],axis=0) for i in range(3)]#标准差
    wrong=0
    for i in range(3):
        for t in test[i]:
            cmap0=[]
            for j in range(3):
                #由于数据集中所有的属性都是连续值，
                #连续值的似然估计可以按照高斯分布来计算：
                cmap=np.exp(-np.power(t-means[j],2)/(2*np.power(stds[j],2)))/((2*math.pi) ** 0.5 * stds[j])
                cmap=np.prod(cmap)#将每一项高斯分布值相乘
                cmap*=len(types[j])/data_num
                cmap0.append(cmap)#保存预测结果
            cmapmax=cmap0.index(max(cmap0))
            if cmapmax!=i:
                wrong+=1
    return wrong

def crosscheck(types,num):#进行十次交叉验证
    test_data=[[],[],[]]#测试集
    train_data=[[],[],[]]#训练集
    test_len=[round(len(types[i])/num) for i in range(3)]
    datas=sum([len(types[i]) for i in range(3)])
    wrongs=0
    for i in range(num):#进行七折交叉验证,将每个数据集分成7份，每一份分别作为测试集，其余作为训练集进行训练。
        for j in range(3):#按照分层采样的方式将数据分割训练集与测试集
            if i==num-1:
                test_data[j]=np.mat(types[j][i*test_len[j]:])
                train_data[j]=np.mat(types[j][:i*test_len[j]])
            else:
                test_data[j]=np.mat(types[j][i*test_len[j]:(i+1)*test_len[j]])
                train_data[j]=np.mat(types[j][:i*test_len[j]]+types[j][(i+1)*test_len[j]:])
        wrongs+=bayes_classfy(train_data,test_data,types)
    print("当进行{}折交叉验证时准确率为:".format(num)+str(1-wrongs/datas))#正确率

for i in range(10):
   crosscheck(types,i+3)#输出不同折交叉验证下准确率

