import numpy as np
import math
import queue
import matplotlib.pyplot as plt

'''
实现密度聚类中的DBSCAN算法 找到基于密度可达关系导出的最大密度相连集合

主要思想是DFS,使用队列来进行搜索

需要实现：

确定邻域内的元素个数

维护一个队列进行DFS

维护一个未访问元素集合hash

确定邻域内的元素
'''
q = queue.Queue()

e = 0.11

minpts = 5

hash = np.zeros([100])

coreobject = np.zeros([100])

def dist(x_1,x_2):
    return np.linalg.norm(x_1 - x_2)

def N(x_j,data):
    ans = 0
    obset = np.zeros([100])
    for i in range(0,data.shape[0]):#遍历数据
        if(dist(x_j,data[i,...]) <= e):#距离小于指定值
            obset[ans] = i
            ans = ans + 1#计数
    return ans,obset

fp = open(r'.\waterlemon4.0.txt','r',encoding='UTF-8')

data = []

for line in fp:
    line = line.strip('\n')
    data.append(line.split(' '))
fp.close()#读入

data = np.array(data,dtype = float)

datatag = np.zeros([100])

#初始化核心对象
k = 0
for i in range(0,data.shape[0]):
    ans,obset = N(data[i],data)
    if(ans >= minpts):
        coreobject[k] = i
        k = k + 1

#DFS
tag = 0
for i in range(0,k):
    if(hash[int(coreobject[i])] == 1):
        continue
    tag = tag + 1
    q.put(coreobject[i])#选出没有访问过的一个核心对象
    while(not q.empty()):#开始DFS
        temp = int(q.get())
        hash[temp] = 1
        datatag[temp] = tag#记录标签
        ans,obset = N(data[temp],data)
        if(ans >= minpts):#符合邻域条件，将其可达的元素加入队列
            for j in range(0,obset.shape[0]):
                if(hash[int(obset[j])] == 0):
                    datatag[int(obset[j])] = tag#未访问过的可达对象 记录标签
                    hash[int(obset[j])] = 1#已访问
                    q.put(obset[j])#将新对象加入队列
    

print(datatag)
        
a = np.array(['or','ob','oy','og','ok'])

for i in range(0,data.shape[0]):
    plt.plot(data[i,0],data[i,1],a[int(datatag[i])])
    plt.text(data[i,0]+0.01,data[i,1],i)
plt.show()

