import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']#保证plt的中文输出
mpl.rcParams['axes.unicode_minus'] = False

datas=[]#对wine.data进行处理，一共有三类数据
data=open('wine.data')
for line in data.readlines():
    datas.append(line.replace('\n','').split(','))
datas=[[eval(s) for s in d] for d in datas]
    
for d in datas:
    if(len(d)<14):
        print("error")

data.close()

types=[[],[],[]]#对数据分类
for i in datas:
    types[i[0]-1].append(i[1:])
#print(types)


def crosscheck(types):#进行十次交叉验证
    test_data=[[],[],[]]
    train_data=[[],[],[]]
    test_y=[]
    pre_y=[]
    test_len=[round(len(types[i])/10) for i in range(3)]
    data_num=0
    for i in range(3):
        data_num+=len([types])
    for i in range(10):#七折交叉验证
        for j in range(3):
            if i==9:
                test_data[j]=np.mat(types[j][i*test_len[j]:])
                train_data[j]=np.mat(types[j][:i*test_len[j]])
            else:
                test_data[j]=np.mat(types[j][i*test_len[j]:(i+1)*test_len[j]])
                train_data[j]=np.mat(types[j][:i*test_len[j]]+types[j][(i+1)*test_len[j]:])
        means=[np.mean(train_data[i],axis=0) for i in range(3)]
        stds=[np.std(train_data[i],axis=0) for i in range(3)]

        for i in range(3):
           for t in test_data[i]:
                cmap0=[]
                for j in range(3):
                #由于数据集中所有的属性都是连续值，
                #连续值的似然估计可以按照高斯分布来计算：
                   cmap=np.exp(-np.power(t-means[j],2)/(2*np.power(stds[j],2)))/((2*math.pi) ** 0.5 * stds[j])
                   cmap=np.prod(cmap)#将每一项高斯分布值相乘
                   cmap*=len(types[j])/data_num
                   cmap0.append(cmap)#保存预测结果
                cmapmax=cmap0.index(max(cmap0))
                pre_y.append(cmapmax)#记录预测值
                test_y.append(i)#记录实际值

    return test_y,pre_y


def confusion(test_y,pre_y):
    label=['1','2','3']
    conf = []
    for i in range(3):
        conf.append([0] * 3)
    for i in range(len(test_y)):
        conf[test_y[i]][pre_y[i]] += 1
    conf=np.array(conf)
    print("混淆矩阵:\n",conf)
    precision=[]
    recall=[]
    F1_score=[]
    for i in range(3):
       precision.append(round(conf[i][i]/np.sum(conf,axis=0)[i],2))
       recall.append(round(conf[i][i]/np.sum(conf,axis=1)[i],2))
       F1_score.append(round(2*precision[i]*recall[i]/(precision[i]+recall[i]),2))
    print("     precision ")
    for i in range(3):
        print("{}    {}  ".format(i+1,precision[i]))
    print("     recall")
    for i in range(3):
        print("{}    {}  ".format(i+1,recall[i]))
    print("     F1_score ")
    for i in range(3):
        print("{}    {}  ".format(i+1,F1_score[i]))
    sns.heatmap(conf,annot=True,fmt='d',xticklabels=label,yticklabels=label)#混淆矩阵
    plt.ylabel('实际结果', fontsize=18)
    plt.xlabel('预测结果', fontsize=18)
    plt.show()


test_y,pre_y=crosscheck(types)
confusion(test_y,pre_y)

