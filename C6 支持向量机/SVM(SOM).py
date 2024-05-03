import numpy as np
import math
import matplotlib.pyplot as plt

'''
使用SOM算法对SVM进行求解

包含如下步骤

选取需要更新的参数

计算新参数，并更新

更新偏移量b

在计算新参数时需要用到这些函数 计算预测量的函数g 计算内积的核函数K 剪切时用到的HL 剪切函数 以及程序入口
'''
'''
以下为内积函数（核函数）根据传入的mode参数可以使用不同的核
'''
def K(x1,x2,mode,p = 5,q = -1):
    if mode == 'linear':#线性核
        return np.dot(x1,x2)
    if mode == 'poly':#多项式核 多项式次数p默认为1
        return np.dot(x1,x2) ** p
    if mode == 'gauss' or mode == 'RBF':#高斯核 带宽p默认为1
        temp = np.dot(x1 - x2,x1 - x2)
        return math.exp(-temp / (2 * p ** 2))
    if mode == 'laplace':#拉普拉斯核 带宽p默认为1
        temp = np.linalg.norm(x1 - x2)
        return math.exp(-temp / p)
    if mode == 'sigmoid':#sigmoid核 beta(p)默认为1 theta(q)默认为-1
        return math.tanh(p * np.dot(x1,x2) + q)
    
'''
以下为使用参数alpha计算预测值的函数g

传入参数alpha 偏移量b 以及对应的数据x和标签y 核函数mode

事实上可以先打表计算出全部E值以节约计算时间，此处并未实现
'''
def g(x_,x,y,a,b,mode):
    temp = 0
    for i in range(0,x.shape[0]):
        temp += a[i] * K(x_,x[i,...],mode) * y[i]
    return temp + b

'''
以下为剪切函数 将a_unc剪切为符合约束条件的a_new

传入需要剪切的参数序号以及参数和软间隔常数C,返回两个被剪切好的函数
'''

def cut(i,j,aj,a,y,C):
    if y[i] == y[j]:
        L = max(0,a[i] + a[j] - C)
        H = min(C,a[i] + a[j])
    else:
        L = max(0,a[j] - a[i])
        H = min(C,C + a[j] - a[i])
    if aj > H:
        aj_new = H
        ai_new = a[i] + y[i] * y[j] * (a[j] - aj_new)
        return ai_new,aj_new
    else:
        if aj < L:
            aj_new = L
            ai_new = a[i] + y[i] * y[j] * (a[j] - aj_new)
            return ai_new,aj_new
        else:
            aj_new = aj
            ai_new = a[i] + y[i] * y[j] * (a[j] - aj_new)
            return ai_new,aj_new

'''
偏移量b更新函数

计算新的b

使用更健壮的计算方式，使用全部a进行计算
'''

def b_calc(a,x,y,mode):
    b_new = 0
    s = 0
    for i in range(0,a.shape[0]):
        if a[i] != 0:
            s = s + 1
            b_new += 1/y[i]
            for j in range(0,a.shape[0]):
                b_new -= a[j] * y[j] * K(x[i,...],x[j,...],mode)
    return b_new/s

'''
以下为选择下一对更新变量的函数

第一个变量选择违反KKT条件的变量

第二个变量选择E值相差最大的变量//这里为了方便选择遍历其他所有变量
'''

def select1(a,b,x,y,C,mode):
    for i in range(0,a.shape[0]):
        if a[i] > 0 and a[i] < C:
            if y[i] * g(x[i,...],x,y,a,b,mode) != 1:
                return i
    for i in range(0,a.shape[0]):
        if a[i] == 0:
            if y[i] * g(x[i,...],x,y,a,b,mode) < 1:
                return i
        if a[i] == C:
            if y[i] * g(x[i,...],x,y,a,b,mode) > 1:
                return i
    return -1

'''
以下为主程序，进行计算新的参数并更新
'''

if '__main__' == __name__:

    fp = open(r'.\waterlemon(alpha).txt','r',encoding='UTF-8')

    data = []

    for line in fp:
        line = line.strip('\n')
        data.append(line.split(' '))
    fp.close()#读入

    x = np.array(data)[...,0:2]
    y = np.array(data)[...,2]

    y = np.array(y,dtype = float)
    x = np.array(x,dtype = float)

    y[y == 1] = 1
    y[y == 0] = -1

    a = np.zeros(x.shape[0])#初始化拉格朗日系数

    a[0] = 0.5
    a[8] = 0.5

    b = 0#初始化偏移量

    C = 10000#软间隔上界

    mode = 'linear'

    flag = select1(a,b,x,y,C,mode)#选择第一个变量

    MAX = 100

    while flag != -1 and MAX >= 0:
        
        for i in range(0,a.shape[0]):
            if i == flag: continue

            a2_unc = a[i] + y[i]/(K(x[flag,...],x[flag,...],mode) + K(x[i,...],x[i,...],mode) - 2 * K(x[flag,...],x[i,...],mode)) * (g(x[flag,...],x,y,a,b,mode) - y[flag] - g(x[i,...],x,y,a,b,mode) + y[i])
            a1_new,a2_new = cut(flag,i,a2_unc,a,y,C)#计算unc并剪切获得新的参数

            a[flag] = a1_new
            a[i] = a2_new#更新参数a

            #print(a)

            b = b_calc(a,x,y,mode)#更新参数b
        
        flag = select1(a,b,x,y,C,mode)#选择第一个变量
        MAX -= 1

    for i in range(0,x.shape[0]):
        print(g(x[i,...],x,y,a,b,'linear'))
    



    








                           







