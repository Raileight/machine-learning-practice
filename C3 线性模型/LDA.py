import array as arr
import numpy as np
import math

fp = open(r'..\data.txt','r',encoding='utf-16')

data = []

for line in fp:
    line = line.strip('\n')
    data.append(line.split(' '))

fp.close()

data = np.array(data,dtype = float)

x = data[:,0:7].reshape(900,7)
y = data[:,7].reshape(900,1)

#计算类平均
ave0 = np.zeros([1,7])
ave1 = np.zeros([1,7])

for i in range(0,450):
    ave0 += x[i]
    ave1 += x[i + 450]
ave0 = ave0/450
ave1 = ave1/450


#类内散度矩阵
s0 = np.zeros([7,7])
s1 = np.zeros([7,7])
for i in range(0,450):
    s0 += np.dot((x[i] - ave0).T,x[i] - ave0)
    s1 += np.dot((x[i + 450] - ave1).T,x[i + 450] - ave1)
sw = np.matrix(s0 + s1)

#奇异值分解
#计算闭式解

m = np.dot(sw.I,(ave0-ave1).T)


#召回验证

cor = 0

ave0p = np.dot(m.reshape(1,7),ave0.reshape(7,1))
ave1p = np.dot(m.reshape(1,7),ave1.reshape(7,1))


for i in range(0,900):
    
    xp = np.dot(m.reshape(1,7),x[i].reshape(7,1))
    if (abs(xp - ave1p) >= abs(xp - ave0p) and y[i] == 0) or (abs(xp - ave1p) < abs(xp - ave0p) and y[i]== 1): cor += 1

print(cor/900)

#empty不算初始化




