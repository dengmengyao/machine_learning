import numpy as np
import pandas as pd
import math
import copy

def CreateData(mean,cov,num,label):#根据已知条件创建数据集
    x,y,z=np.random.multivariate_normal(mean,cov,num).T
    Labels=np.ones(num)*label
    X=np.array([x,y,z,Labels])
    return X.T


def Distance(x1,x2,dimen,p=2):#计算两点之间的距离，p=2时为欧式距离
    dis=0
    for i in range(dimen):
        dis+=math.pow(x1[i]-x2[i],p)
    return math.sqrt(dis)

def Single_Linkage(sin1,sin2,dimen,p=2):#最短距离 / 单连接 (single linkage)
    Min=float("-inf")
    for i in range(len(sin1)):
        for j in range(len(sin2)):
            d=Distance(sin1[i],sin2[j],dimen,p)
            if Min>d:
                Min=d
    return Min

def Complete_Linkage(com1,com2,dimen,p=2):#最⻓距离 / 全连接 (complete linkage)
    Max=float("-inf")
    for i in range(len(com1)):
        for j in range(len(com2)):
            d=Distance(com1[i],com2[j],dimen,p)
            if Max<d:
                Max=d
    return Max

def Average_Linkage(ave1,ave2,dimen,p=2):#平均距离 (average linkage)
    d=0
    for i in range(len(ave1)):
        for j in range(len(ave2)):
            d+=Distance(ave1[i],ave2[j],dimen,p)
    ans=d/(len(ave1)*len(ave2))
    return ans

def dismin(dis):#找到类集合中距离最近的两个类
    Min=float("inf")
    rs=[0,0]
    for i in range(len(dis)):
        for j in range(len(dis)):
            if i!=j and dis[i][j]<Min:
                Min=dis[i][j]
                rs=[i,j]
    return rs#返回距离最近两个类的坐标


def mix(clu,res):#res为两个要合并的类的标号
    #将b合并到a中，然后用remove移除b,实现类的合并
    a=clu[res[0]]
    b=clu[res[1]]
    a.extend(b)
    clu.remove(b)

def dataset(clu):
    #返回三个类中分别含有三类别数量
    count=[[0,0,0],[0,0,0],[0,0,0]]
    for i in range(3):
        for j in range(len(clu[i])):
            count[i][int(clu[i][j][3]-1)]+=1
    count=np.array(count)
    #数量最多的类作为这一类的标签
    label=count.argmax(axis=1)+1
    return count,label


def Make_cluster(clu,dimen,u,p):#生成类距离的矩阵
    v=len(clu)
    if u==0:#Single_Linkage
        temp=np.zeros((v,v))
        for i in range(v):
            for j in range(v):
                temp[i][j]=Single_Linkage(clu[i],clu[j],dimen,p)
    elif u==1:#Complete_Linkage
        temp=np.zeros((v,v))
        for i in range(v):
            for j in range(v):
                temp[i][j]=Complete_Linkage(clu[i],clu[j],dimen,p)
    elif u==2:#Average_Linkage
        temp=np.zeros((v,v))
        for i in range(v):
            for j in range(v):
                temp[i][j]=Average_Linkage(clu[i],clu[j],dimen,p)
    return temp

def mixclu(clu,dimen,u,p=2):
    if u==0:
        print("Single_Linkage")
    elif u==1:
        print("Complete_Linkage")
    elif u==2:
        print("Average_Linkage")
    while(len(clu)>3):
        dis=Make_cluster(clu,dimen,u,p)#每次的迭代会重新产生类矩阵
        res=dismin(dis)#res为最近两个类的下标
        mix(clu,res)#合并

mean=np.array([[1,1,1],[2,2,2],[3,3,3]])
cov = [[0.4,0,0],[0,0.4,0],[0,0,0.4]]
n=20
p=1/3
nums=[round(n*p) for i in range(3)]
total=0
clu1=[]
for i in range(3):
    total+=nums[i]
    clu1.extend(CreateData(mean[i],cov,nums[i],i+1))

clu=[[list(c)]for c in clu1]

print("共产生了{}个样本".format(total))
for i in range(3):
    cor=0
    cluster=copy.deepcopy(clu)
    #print(cluster)
    print(cluster[1][1])
    mixclu(cluster,3,i)
    
    count,label=dataset(cluster)

    for i in range(3):
        print("cluster[{}]标签为{}".format(i,label[i]))
    print("构成与错误率")
    print("cluster\t类\t类1\t类2\t类3\t错误率\n")
    for i in range(3):
        print("  ",i,"\t",label[i],end="")
        for j in range(3):
            print("\t{}".format(count[i][j],j+1),end="")
        cor+=count[i][label[i]-1]
        print("\t{}\n".format(1-count[i][label[i]-1]/sum(count[i])),end="")
    print("综上:总的错误率为{}".format(1-cor/total))

