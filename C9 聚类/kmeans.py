import numpy as np
import math
import matplotlib.pyplot as plt

#实现 原型聚类-k均值算法（kmeans）
'''
初始化均值向量

计算距离，并将样本划入最近的均值向量一蔟

计算新的均值

更新均值
'''

def dist(x1,x2):#计算欧氏距离
    ans = 0
    for i in range(0,x1.shape[0]):
        ans = ans + (x1[i] - x2[i]) ** 2
    return ans

def kmeans(x,k,T):#数据x，以及聚类种类数k,迭代次数T
    label = np.zeros([x.shape[0]]) - 1#维护一个数组，存储每一个样本的蔟
    
    means = x[0:k,...]#初始化均值 直接取前k个

    for i in range(0,k):
            label[i] = i#给取出的均值定义标签
    
    while(T >= 0):

        #计算新的均值
        for i in range(0,k):
            means[i] = np.average(x[label == i],axis = 0)

        #开始对样本点划分
        for i in range(0,x.shape[0]):
            min = 1e6#足够大
            for j in range(0,k):  
                if dist(x[i],means[j]) <= min:
                    min = dist(x[i],means[j])
                    tag = j
            label[i] = tag
        
        T = T - 1

    return label,means
    
fp = open(r'.\waterlemon4.0.txt','r',encoding='UTF-8')

data = []

for line in fp:
    line = line.strip('\n')
    data.append(line.split(' '))
fp.close()#读入

data = np.array(data,dtype = float)

label,means = kmeans(data,3,5)

a = np.array(['or','ob','oy','og'])

for i in range(0,3):
    plt.plot(data[label == i,0],data[label == i,1],a[i])

plt.show()