import numpy as np
from numpy.core.fromnumeric import argmax
import pandas as pd
import matplotlib.pyplot as plt
fig = plt.figure()#画图
ax1=fig.add_subplot(2,1,1)
ax2=fig.add_subplot(2,1,2)
m=np.array([[1,1],[4,4],[8,1]])#均值矢量
p1=[1/3,1/3,1/3]
p2=[0.6,0.3,0.1]
cov=2*np.eye(2)
dot_num = 1000
#画图
mm1=np.random.multivariate_normal(m[0],cov,int(dot_num/3))#划分不同点位分布概率
mm2=np.random.multivariate_normal(m[1],cov,int(dot_num/3))
mm3=np.random.multivariate_normal(m[2],cov,int(dot_num/3))
mn1=np.random.multivariate_normal(m[0],cov,int(dot_num*0.6))
mn2=np.random.multivariate_normal(m[1],cov,int(dot_num*0.3))
mn3=np.random.multivariate_normal(m[2],cov,int(dot_num*0.1))

ax1.scatter(mm1[:,0],mm1[:,1],c='r',marker='.')
ax1.scatter(mm2[:,0],mm2[:,1],c='g',marker='.')
ax1.scatter(mm3[:,0],mm3[:,1],c='b',marker='.')
ax2.scatter(mn1[:,0],mn1[:,1],c='r',marker='.')
ax2.scatter(mn2[:,0],mn2[:,1],c='g',marker='.')
ax2.scatter(mn3[:,0],mn3[:,1],c='b',marker='.')

plt.show()

def Gauss_function(x,m,cov):
    '''
       计算2维样本数据的概率p(x|w)，参数m是均值向量
       cov是已知的协方差矩阵，x是样本数据
       计算公式为：
       p = 1/(2*np.pi*np.sqrt(d_cov))*np.exp(-0.5*np.dot(np.dot((x - m),i_cov), (x - m)))
    '''
    d_cov=np.linalg.det(cov) #计算协方差矩阵行列式
    i_cov=np.linalg.det(cov) #计算协方差矩阵的逆
    px = 1/(2*np.pi*np.sqrt(d_cov))*np.exp(-0.5*np.dot(np.dot((x - m),i_cov), (x - m)))
    return px

def samples(m,cov,p,label):
    #生成单种类别数据
    num_array=round(1000*p)
    #生成一个符合正态分布矩阵，x,y符合正态分布
    x,y=np.random.multivariate_normal(m, cov, num_array).T#生成随机变量
    #z标明类别
    z=np.ones(num_array)*label
    X=np.array([x,y,z])
    return X.T

def Data_Make(m,cov,p):
    #根据先验概率生成数集,生成所有概率类别的数据
    X=[]
    label=0
    for i in range(3):
        #将对应数据添加到已有数据
        X.extend(samples(m[i],cov,p[i],label))
        label+=1
        #i+=1
    return X

def Max_Posterior(X,m,cov,p):
    num=np.array(X).shape[0]#数据总数
    label_num=m.shape[0]#均值数
    Error=0
    for i in range(num):
        p_temp=np.zeros(3)
        for j in range(label_num):
            #计算样本后验概率
            p_temp[j]=Gauss_function(X[i][0:2],m[j],cov)*p[j]
        p0=np.argmax(p_temp)
        #跟样本真实类别进行比对
        if  p0!=X[i][2]:
            Error+=1
    return Error/num

errors1=np.zeros(20)#多次记录样本错误率
errors2=np.zeros(20)
print("多次计算最大后验概率:")
for i in range(20):
    X1=Data_Make(m,cov,p1)
    X2=Data_Make(m,cov,p2)
    errors1[i]=round(Max_Posterior(X1,m,cov,p1),3)#统计样本错误率
    errors2[i]=round(Max_Posterior(X2,m,cov,p2),3)
    print("数据集X第{0}次:".format(i+1),errors1[i],"数据集X'第{0}次:".format(i+1),errors2[i])
print("数据集X 20次计算平均数为:",round(np.mean(errors1),5))
print("数据集X' 20次计算平均数为:",round(np.mean(errors2),5))

   
    





