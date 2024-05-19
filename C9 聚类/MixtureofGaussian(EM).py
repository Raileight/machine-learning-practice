import numpy as np
import math
import random
import matplotlib.pyplot as plt

#实现 原型聚类-高斯混合聚类算法
'''
具体使用em算法

初始化高斯混合分布的各个参数

利用参数确定各个标签的后验概率

利用极大似然估计确定新的参数

反复迭代
'''

def gauss(x,sigma,u):#计算高斯分布概率值的函数
    det_of_sigma = np.linalg.det(sigma)
    inverse_of_sigma = np.linalg.inv(sigma)

    return math.exp(-0.5 * ((x - u) @ inverse_of_sigma @ (x - u).T)) / ((2 * math.pi) ** (x.shape[0]/2)) * (det_of_sigma ** (1/2))

def gamma_calc(x,sigma,u,a,k):#更新gamma后验概率的函数
    ans = np.zeros([x.shape[0],k])
    for j in range(0,x.shape[0]):
        for i in range(0,k):
            ans[j,i] = a[i] * gauss(x[j],sigma[i],u[i])
            sum = 0
            for l in range(0,k):
                sum = sum + a[l] * gauss(x[j],sigma[l],u[l])
            ans[j,i] = ans[j,i] / sum

    return ans


def MoG(x,k,T):#输入数据、聚类数、迭代次数
    #随机初始化均值
    u = np.zeros([k,x.shape[1]])
    '''
    for i in range(0,k):
        rd = random.randint(int(x.shape[0]/k) * i,int(x.shape[0]/k) * i+9)
        u[i] = x[rd]
    '''
    u[0] = x[5]
    u[1] = x[21]
    u[2] = x[26]
    
    #初始化协方差矩阵
    temp = np.zeros([x.shape[1]]) + 0.1
    sigma1 = np.diag(temp)#一个2阶对角矩阵

    sigma = np.zeros([k,2,2])
    for i in range(0,k):
        sigma[i] = sigma1

    #初始化混合系数
    a = np.zeros([k]) + 1/k

    #开始迭代
    while(T >= 0):
        #计算后验概率gamma[j,i] = P(z_j = i|x_j) E步
        gamma = gamma_calc(x,sigma,u,a,k)

        #更新均值向量 M步
        for i in range(0,k):
            sum1 = 0
            for j in range(0,x.shape[0]):
                sum1 = sum1 + gamma[j,i] * x[j]
            u[i] = sum1 / np.sum(gamma[...,i])

        #更新协方差矩阵 M步
        for i in range(0,k):
            sum2 = np.zeros([2,2])
            for j in range(0,x.shape[0]):
                sum2 = sum2 + gamma[j,i] * np.matrix(x[j] - u[i]).reshape(2,1) @ np.matrix(x[j] - u[i]).reshape(1,2)
            sigma[i] = sum2 / np.sum(gamma[...,i])

        #更新混合系数 M步
        for i in range(0,k):
            a[i] = np.sum(gamma[...,i]) / x.shape[0]
        
        T = T - 1

    #判断标签
    gamma = gamma_calc(x,sigma,u,a,k)
    label = np.zeros([x.shape[0]]) - 1
    for j in range(x.shape[0]):
        max = -1
        for i in range(0,k):
            if gamma[j,i] > max:
                max = gamma[j,i]
                tag = i
        label[j] = tag

    return label
    





fp = open(r'.\waterlemon4.0.txt','r',encoding='UTF-8')

data = []

for line in fp:
    line = line.strip('\n')
    data.append(line.split(' '))
fp.close()#读入

data = np.array(data,dtype = float)

label = MoG(data,3,5)

print(label)

a = np.array(['or','ob','oy','og','ok'])

for i in range(0,3):
    plt.plot(data[label == i,0],data[label == i,1],a[i])

plt.show()


