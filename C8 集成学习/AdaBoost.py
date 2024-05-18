import numpy as np
import math

#实现了集成学习中Boosting-AdaBoost算法
#以未剪枝的决策树作为基学习器 采用指数损失函数 采用相对多数投票结合策略
#采用数据集 waterlemon(alpha).txt

'''
针对数据集训练模型h_t
计算错误率
更新权重
更新分布（使用带权样本）
'''

#单层决策树-决策树桩
'''
使用信息增益计算最大信息增益的属性，划分后直接按照数量最多的类型进行划分
'''
def log2(x):
    if x != 0:
        return math.log2(x)
    else:
        return 0
    
def sgn(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    if x == 0:
        return 0
    
def ent(y,D):#D为数据分布 y为标签
    p1 = np.sum(D[y == 1])/np.sum(D)
    p0 = np.sum(D[y == -1])/np.sum(D)

    return -p0 * log2(p0)-p1 * log2(p1)

def gain(x):#计算信息增益并确定最大的属性和划分点

    max = 0#最大的信息增益
    flag = -1#信息增益最大的属性
    for i in range(0,2):
        #取出属性 并排序
        temp = x
        temp = temp[np.argsort(temp[:,i])]
        #print(temp)
        #计算gain(D,a,t) 并找出最大值max = gain(D,a)
        temp_max = 0
        for j in range(0,x.shape[0] - 1):#一共16个划分点
            gain_j = ent(x[...,2],x[...,3]) - np.sum(temp[0:j + 1,3]) * ent(temp[0:j + 1,2],temp[0:j+1,3]) - (1 - np.sum(temp[0:j + 1,3])) * ent(temp[j + 1:17,2],temp[j+1:17,3])
            #print(gain_j," ",(temp[j,i] + temp[j + 1,i]) / 2," ",temp[j,i])
            if gain_j > temp_max:
                temp_max = gain_j#找到最大值
                bound_temp = (temp[j,i] + temp[j + 1,i]) / 2
        if temp_max > max:#该属性的信息增益更大
            max = temp_max#更新最大值
            flag = i#更新属性位置
            bound = bound_temp#更新对应的划分点
            testdata = x[...,2][x[...,flag] <= bound]
            if np.sum(testdata == 1) <= np.sum(testdata == -1):
                label = -1
            else:
                label = 1
    return flag,bound,label

fp = open(r'..\waterlemon(alpha).txt','r',encoding='UTF-8')

data = []

for line in fp:
    line = line.strip('\n')
    data.append(line.split(' '))
fp.close()#读入

data = np.array(data,dtype = float)

D = np.zeros([17,1]) + 1/17#给数据引入权重，初始权重相同

T = 10#训练数目
h = np.zeros([T,4])#存储训练好的基学习器 存储：属性、划分点、标签、权重



for i in range(0,T):
    tempdata = np.append(data,D,axis = 1)
    h[i,0],h[i,1],h[i,2] = gain(tempdata)#训练一个学习器
    correct = 0
    err = 0
    for j in range(0,17):#计算错误率
        if data[j,int(h[i,0])] <= h[i,1] and data[j,2] == h[i,2]:
            correct = correct + 1
            continue
        if data[j,int(h[i,0])] > h[i,1] and data[j,2] != h[i,2]:
            correct = correct + 1
            continue
        err = err + 1
    eps = err/(err+correct)
    if eps > 0.5:
        break
    else:#错误率低于0.5则继续训练
        h[i,3] = 1/2 * math.log((1 - eps) / eps)#计算对应的权重
        #以下更新分布D
        for j in range(0,17):
            if (data[j,int(h[i,0])] <= h[i,1] and data[j,2] == h[i,2]) or (data[j,int(h[i,0])] > h[i,1] and data[j,2] != h[i,2]):
                D[j] = D[j] * math.exp(-h[i,3])
            else:
                D[j] = D[j] * math.exp(h[i,3])
        #归一
        D = D / np.sum(D)

#合并学习器进行预测
print(h)

for j in range(0,17):
    ans = 0
    for i in range(0,T):
        if data[j,int(h[i,0])] <= h[i,1]:#满足决策条件
            ans = ans + h[i,3] * h[i,2]#标签为h[i,2]
        else:
            ans = ans - h[i,3] * h[i,2]
    print(sgn(ans) == data[j,2])
    ans = 0


        







