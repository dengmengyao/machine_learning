import numpy as np
from numpy.core.fromnumeric import argmax
import pandas as pd
import matplotlib.pyplot as plt
import math
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



def Gauss_function(x,X,h,j,p):
   num=np.array(X).shape[0]#数据总数
   z0=0
   for i in range(num):
       if X[i][2]==j:
          u=X[i][0]
          #print(u)
          v=X[i][1]
          z0+=(1/((2*np.pi*h*h)**(1/2)))*np.exp(-0.5*((x[0]-u)**2+(x[1]-v)**2)/(h**2))
   return z0


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
    return X
        

def Gauss_nuclear_function(X,x,y,h):
   num=np.array(X).shape[0]#数据总数
   z0=0
   for i in range(num):
       u=X[i][0]
       #print(u)
       v=X[i][1]
       z0+=((1/(2*math.pi*h))*math.exp(-0.5*((x-u)**2+(y-v)**2)/(h**2)))/num

   return z0


def Likelihood(X,X0,m,h,p):
    num=np.array(X).shape[0]#数据总数
    label_num=m.shape[0]#均值数
    Error=0
    for i in range(num):
        p_temp=np.zeros(3)

        for j in range(label_num):
            #计算样本似然概率
               p_temp[j]=Gauss_function(X[i][0:2],X0,h,j,p[j])/(1000*p[j])
        p0=np.argmax(p_temp)
           #跟样本真实类别进行比对
        if  p0!=X[i][2]:
               Error+=1

 
        
    return Error/num


def draw(ax,k,X,label):#画图函数
    x = np.arange(0, 10, 0.25)
    y = np.arange(0, 10, 0.25)
    x, y = np.meshgrid(x, y)
    z=np.zeros((40,40))

    for i in range(40):
       for j in range(40):
           z[i][j]=Gauss_nuclear_function(X,x[i][j],y[i][j],k)
    #print(z[30][30])
    ax.plot_surface(x, y, z, rstride = 1, cstride = 1, cmap=plt.get_cmap('gist_rainbow')) # row 行步长, colum 列步长# 渐变颜色
    ax.contourf(x, y, z,zdir='z', offset=-2, cmap=plt.get_cmap('gist_rainbow'))# 使用数据方向, 填充投影轮廓位置
    ax.set_zlim(0,0.025)

    ax.set_title('h= {0} ,数据集{1}'.format(k,label),fontproperties="SimHei")





X0=Data_Make(m,cov,p1)
X00=Data_Make(m,cov,p1)
X1=Data_Make(m,cov,p2)
X11=Data_Make(m,cov,p2)
label=1
k=[0.1,0.5,1,1.5,2]
for i in range(5):
   print("当h={0},".format(k[i]),"数据集1错误率为:",round(Likelihood(X0,X0,m,k[i],p1),6))#统计样本错误率
   print("当h={0},".format(k[i]),"数据集2错误率为:",round(Likelihood(X1,X1,m,k[i],p2),6))
fig = plt.figure(figsize=(25,10))
ax1=fig.add_subplot(2,5,1,projection='3d')
ax2=fig.add_subplot(2,5,2,projection='3d')
ax3=fig.add_subplot(2,5,3,projection='3d')
ax4=fig.add_subplot(2,5,4,projection='3d')
ax5=fig.add_subplot(2,5,5,projection='3d')
ax6=fig.add_subplot(2,5,6,projection='3d')
ax7=fig.add_subplot(2,5,7,projection='3d')
ax8=fig.add_subplot(2,5,8,projection='3d')
ax9=fig.add_subplot(2,5,9,projection='3d')
ax10=fig.add_subplot(2,5,10,projection='3d')

draw(ax1,k[0],X0,label)
draw(ax2,k[1],X0,label)
draw(ax3,k[2],X0,label)
draw(ax4,k[3],X0,label)
draw(ax5,k[4],X0,label)

label=2

draw(ax6,k[0],X1,label)
draw(ax7,k[1],X1,label)
draw(ax8,k[2],X1,label)
draw(ax9,k[3],X1,label)
draw(ax10,k[4],X1,label)

plt.show()
#综上个，h=0.5时最合适