import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv('winequality-white.csv')


lr = 0.0000001# 学习率alpha，我们采用10倍的方式进行学习率取值尝试

nums=[0,0,0,0,0,0,0,0,0,0,0,0]#多元线性回归系数

x_data=data.iloc[:,0:11]#自变量
y_data=data.iloc[:,11]#因变量

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=12,test_size=0.2)
#划分训练集与测试集

max=15#最大迭代次数

def h_predict(nums, x_data, i):#最小二乘目标函数
    h_pre=nums[0]
    for j in range(0,len(nums)-1):
        h_pre+=nums[j+1]*x_data.iloc[i,j]
    return h_pre

# 最小二乘法,计算MSE
def MSE(nums, x_data, y_data):
    totalError = 0
    for i in range(0,len(x_data)):
        totalError += (h_predict(nums, x_data, i)-y_data.iloc[i]) ** 2
    return totalError / len(x_data)

def gradient_descent_runner(x_data, y_data, nums, lr, epochs):#我们采用批量梯度下降的方法构造线性回归模型
    # 计算总数据量
    mse=[]
    m = float(len(x_data))
    # 循环epochs次
    for i in range(epochs):
        nums_grad= [0,0,0,0,0,0,0,0,0,0,0,0]
        # 求偏导
        for j in range(0, len(x_data)):#计算出回归表达式各个系数
            nums_grad[0] += (1/m) * (h_predict(nums, x_data, j) - y_data.iloc[j])
            for h in range(len(nums_grad)-1):
                nums_grad[h+1]+=(1/m) * x_data.iloc[j,h] * (h_predict(nums, x_data, j) - y_data.iloc[j])
        # 更新b和k
        for h in range(len(nums)):
            nums[h]=nums[h]-(lr*nums_grad[h])
        mse.append(MSE(nums, x_data, y_data))
    return nums,mse

print("After {0} iterations:".format(max))

fig = plt.figure()#画图
ax1=fig.add_subplot(3,2,1)
ax2=fig.add_subplot(3,2,2)
ax3=fig.add_subplot(3,2,3)
ax4=fig.add_subplot(3,2,4)
ax5=fig.add_subplot(3,2,5)
ax6=fig.add_subplot(3,2,6)

nums=[0,0,0,0,0,0,0,0,0,0,0,0]
nums,mse = gradient_descent_runner(x_train, y_train, nums, lr, max)
print("学习率=",lr,"MSE_train=",MSE(nums,x_train,y_train),"MSE_test=",MSE(nums,x_test,y_test))
ax1.plot(range(len(mse)),mse,'b',label="mse")
ax1.set_title('alpha=0.0000001')


lr*=10
nums=[0,0,0,0,0,0,0,0,0,0,0,0]
nums,mse = gradient_descent_runner(x_train, y_train, nums, lr, max)
print("学习率=",lr,"MSE_train=",MSE(nums,x_train,y_train),"MSE_test=",MSE(nums,x_test,y_test))
ax2.plot(range(len(mse)),mse,'b',label="mse")
ax2.set_title('alpha=0.000001')

lr*=10
nums=[0,0,0,0,0,0,0,0,0,0,0,0]
nums,mse = gradient_descent_runner(x_train, y_train, nums, lr, max)
print("学习率=",lr,"MSE_train=",MSE(nums,x_train,y_train),"MSE_test=",MSE(nums,x_test,y_test))
ax3.plot(range(len(mse)),mse,'b',label="mse")
ax3.set_title('alpha=0.00001')

lr*=10
nums=[0,0,0,0,0,0,0,0,0,0,0,0]
nums,mse = gradient_descent_runner(x_train, y_train, nums, lr, max)
print("学习率=",lr,"MSE_train=",MSE(nums,x_train,y_train),"MSE_test=",MSE(nums,x_test,y_test))
ax4.plot(range(len(mse)),mse,'b',label="mse")
ax4.set_title('alpha=0.0001')

lr*=10
nums=[0,0,0,0,0,0,0,0,0,0,0,0]
nums,mse = gradient_descent_runner(x_train, y_train, nums, lr, max)
print("学习率=",lr,"MSE_train=",MSE(nums,x_train,y_train),"MSE_test=",MSE(nums,x_test,y_test))
ax5.plot(range(len(mse)),mse,'b',label="mse")
ax5.set_title('alpha=0.001')

lr*=10
nums=[0,0,0,0,0,0,0,0,0,0,0,0]
nums,mse = gradient_descent_runner(x_train, y_train, nums, lr, max)
print("学习率=",lr,"MSE_train=",MSE(nums,x_train,y_train),"MSE_test=",MSE(nums,x_test,y_test))
ax6.plot(range(len(mse)),mse,'b',label="mse")
ax6.set_title('alpha=0.01')

plt.show()
#for i in range(len(nums)):
#    print("nums",i,"=",nums[i])

#print("MSE_train=",MSE(nums,x_train,y_train))
#print("MSE_test=",MSE(nums,x_test,y_test))

#for j in range(20):
#   mean=nums[0]
#   for i in range(len(nums)-1):
#     mean+=nums[i+1]*x_data.iloc[j,i]
#   print(mean)