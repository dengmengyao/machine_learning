import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("dataset_regression.csv")
x=data["x"]
y=data["y"]

plt.scatter(x,y)
plt.show()

def computer_cost(w,b,x,y):#均方误差MSE计算公式
    total_cost=0
    m=len(x)

    for i in range(m):
        total_cost+=(y[i]-w*x[i]-b)**2
    
    return total_cost/m

def ave(da):#平均数
    sum=0
    num=len(da)
    for i in range(num):
        sum+=da[i]
    return sum/num

def fit(x,y):#进行回归分析，利用最小二乘法计算θ1，θ0
    m=len(x)
    x_ave=ave(x)

    sum_xy=0
    sum_x2=0
    sum_delta=0

    for i in range(m):
        sum_xy+=y[i]*(x[i]-x_ave)
        sum_x2+=x[i]**2
    
    w=sum_xy/(sum_x2-m*(x_ave**2))

    for i in range(m):
        sum_delta+=(y[i]-w*x[i])

    b=sum_delta/m
    return w,b

w,b=fit(x,y)

print('θ1 is : ', w)
print('θ0 is : ', b)

cost=computer_cost(w,b,x,y)

print('训练误差为 : ', cost)

plt.scatter(x,y)
predict_y=w*x+b

plt.plot(x,predict_y,c='r')
plt.show()

m=[-1.25,-0.75,-0.25,0.25,0.75,1.25]

for i in range(len(m)):#预测值
    print(m[i],w*m[i]+b)
