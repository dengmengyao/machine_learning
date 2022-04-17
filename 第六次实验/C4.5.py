import numpy as np
import pandas as pd
import math
from collections import Counter, defaultdict

def DataSet(DataNum, Fea):#按照给定特征划分数据集
    Split=defaultdict(list)
    num=0.0
    nu=0
    for d in DataNum:
        if type(d[Fea]) != float:
            Split[d[Fea]].append(np.delete(d, Fea))
        else:
            num+=d[Fea]
            nu+=1
    for d in DataNum:
        if type(d[Fea])==float:
            if(d[Fea])>(round(num/nu,3)):
                more="大于"+str(round(num/nu,3))
                Split[more].append(np.delete(d,Fea))
            else:
                less="小于"+str(round(num/nu,3))
                Split[less].append(np.delete(d,Fea))
    return list(Split.keys()),list(Split.values())

def CalculateInfomationEntropy(columnIndex, DataNum):
     #数据集总项数
    n=len(DataNum) #统计数据的总数
     #标签计数对象初始化
    DataNum=np.array(DataNum)
    nums=Counter(DataNum[:, columnIndex])
    entropy=0.0#初始化信息熵
    for u, num in nums.items():
        p=num/float(n)
        entropy-=p*math.log(p, 2)#香农公式
    return entropy

def ChooseBestfeatureToSplit(DataNum):#选择信息增益最大的特征对数据进行切分
    n=len(DataNum[0])-1#总特征
    #计算切划分前的信息熵
    entropy=CalculateInfomationEntropy(-1,DataNum)#初始计算香农熵
    bestGain=0.0
    Fea=-1
    for i in range(n):
        #print("%s4343",i)
        k,Split=DataSet(DataNum, i)
        featEntropy = CalculateInfomationEntropy(i, DataNum)
        #print(featEntropy)
        entropyed=0.0
        for da in Split:#计算划分后的信息熵
            pe=len(da)/float(len(DataNum))
            entropyed+=pe*CalculateInfomationEntropy(-1,da)#对熵进行加权求和
        Gain=entropy-entropyed #计算信息增益
        if float(featEntropy)!=0.0 :
            infoGain = Gain/float(featEntropy)
        if infoGain >bestGain or float(featEntropy)==0.0: #比较求最大熵
            bestGain=infoGain
            Fea=i
    return Fea

def DecisionTree(DataNum, Fea):#生成决策树
    DataNum=np.array(DataNum)
    counter=Counter(DataNum[:, -1])
    if len(counter) == 1:#如果数据集样本属于同一类，说明该叶子结点划分完毕
        return DataNum[0, -1]
    if len(DataNum[0]) == 1: #如果数据集样本只有一列，则说明所有特征都划分完毕
        return counter.most_common(1)[0][0]
    featureindex=ChooseBestfeatureToSplit(DataNum)#选择信息增益最大的特征对数据进行切分
    f=Fea[featureindex]
    Tree={f: {}}
    Fea.remove(f)
    key,Split=DataSet(DataNum, featureindex)
    for k,d in zip(key, Split):#如果还可以划分则继续进行递归调用
        Tree[f][k]=DecisionTree(d, Fea[:])
    return Tree

def classifyC45(Tree, Fea, DataNum):#使用决策树，对待分类样本进行分类
    Prediction=None
    u=list(Tree.keys())[0]#获取决策树节点的特征
    Tree=Tree[u]
    featureindex=Fea.index(u)# 将标签字符串转换为索引
    for key in Tree:
        if DataNum[featureindex] == key:#找到了对应的分叉
            #如果再往下还有子树，则继续递归
            if isinstance(Tree[key], dict):
                Prediction=classifyC45(Tree[key], Fea, DataNum)
            else:
                Prediction=Tree[key]
    if Prediction is None: #没有找到对应的分叉
        for key in Tree:
            if not isinstance(Tree[key], dict):
                Prediction=Tree[key]
                break
    return Prediction

train_feature=list(pd.read_csv("Watermelon-train2.csv",encoding="gbk").columns.values)[1:-1]
test_feature=list(pd.read_csv("Watermelon-test2.csv",encoding="gbk").columns.values)[1:-1]
train_data=np.array(pd.read_csv("Watermelon-train2.csv",encoding="gbk"))[:,1:]
test_data=np.array(pd.read_csv("Watermelon-test2.csv",encoding="gbk"))[:,1:]
Decision_Tree=DecisionTree(train_data,train_feature)
num=0
for d in test_data:
    Prediction=classifyC45(Decision_Tree, test_feature, d[0:-1])
    if d[-1] == Prediction:
        num+=1
correct=num/test_data.shape[0]
print("决策树为:{}".format(Decision_Tree))
print("准确率为:{}".format(correct))

