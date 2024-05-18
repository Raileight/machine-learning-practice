import numpy as np
import math
import matplotlib.pyplot as plt
import random

#实现原型聚类-学习向量量化（LVQ） 带标签的
'''
初始化原型向量

随机选择样本计算距离

对距离最近的原型，根据标签是否一致，按照一定的学习率学习
'''
def dist(x1,x2):#计算欧氏距离
    ans = 0
    for i in range(0,x1.shape[0]):
        ans = ans + (x1[i] - x2[i]) ** 2
    return ans

def LVQ(x,y,T):#数据x,标签y,聚类数k(2),迭代次数T

    n = 0.1

    #初始化原型
    P = np.zeros([5,3])
    P[0,0:2] = x[4]
    P[1,0:2] = x[11]
    P[2,0:2] = x[17]
    P[3,0:2] = x[22]
    P[4,0:2] = x[28]
    P[0,2] = 1
    P[1,2] = -1
    P[2,2] = -1
    P[3,2] = 1
    P[4,2] = 1

    #随机选择样本进行学习
    while(T >= 0):
        rd = random.randint(0,29)
        min = 1e6
        for i in range(0,5):
            if dist(x[rd],P[i,0:2]) <= min:
                min = dist(x[rd],P[i,0:2])
                tag = i
        if y[rd] == P[tag,2]:
            P[tag,0:2] = P[tag,0:2] + n * (x[rd] - P[tag,0:2])
        else:
            P[tag,0:2] = P[tag,0:2] - n * (x[rd] - P[tag,0:2])
        T = T - 1

    label = np.zeros([30])
    for i in range(0,x.shape[0]):
        min = 1e6
        for j in range(0,5):
            if dist(x[i],P[j,0:2]) <= min:
                min = dist(x[i],P[j,0:2])
                tag = j
        label[i] = tag
    return label

fp = open(r'.\waterlemon4.0.txt','r',encoding='UTF-8')

data = []

for line in fp:
    line = line.strip('\n')
    data.append(line.split(' '))
fp.close()#读入

data = np.array(data,dtype = float)
y = np.zeros([30]) + 1
y[8:21] = -1

label = LVQ(data,y,400)

print(label)

a = np.array(['or','ob','oy','og','ok'])

for i in range(0,5):
    plt.plot(data[label == i,0],data[label == i,1],a[i])

plt.show()
