import numpy as np
import math


def log(x,y):
    if x == 0: return 0
    else:
        return math.log(x,y)

def pre(x,y,typese,type):#提前打表计算好各个属性的种类中的正例反例数目占比，typese控制每个特质有哪些类别，type控制连续型（1）或离散型（0）
    temp_pos = 0
    temp_neg = 0
    temp_tol = 0
    ans = np.zeros([x.shape[1],x.shape[0],5,5])#初始化索引，索引1：特质 索引2：第几个取值 索引3：正例（1）反例（0）

    for i in range(0,x.shape[1]):
        if type[i] == 0:
            for j in range(0,typese[i].shape[0]):
                for k in range(0,x[...,i].shape[0]):
                    if x[k,i] == typese[i,j]:
                        temp_tol += 1
                        if y[k] == 1: temp_pos += 1
                        else: temp_neg += 1
                if temp_tol != 0:
                    ans[i,j,0,0] = temp_neg/temp_tol
                    ans[i,j,1,0] = temp_pos/temp_tol
                    temp_pos = 0
                    temp_neg = 0
                    temp_tol = 0
                else:
                    temp_pos = 0
                    temp_neg = 0
                    temp_tol = 0
        else:#type == 1
            temp = np.c_[x,y][np.c_[x,y][:,i].argsort()]
            for j in range(0,x[...,i].shape[0] - 1):
                temp_pos = np.sum(temp[0 : j + 1,x.shape[1]] == '1')
                temp_neg = np.sum(temp[0 : j + 1,x.shape[1]] == '0')

                ans[i,j,0,0] = temp_neg/(j + 1)
                ans[i,j,1,0] = temp_pos/(j + 1)
                temp_pos = 0
                temp_neg = 0

                temp_pos = np.sum(temp[j + 1 : x.shape[0],x.shape[1]] == '1')
                temp_neg = np.sum(temp[j + 1 : x.shape[0],x.shape[1]] == '0')
                ans[i,j,0,1] = temp_neg/(x.shape[0] - j - 1)
                ans[i,j,1,1] = temp_pos/(x.shape[0] - j - 1)
                temp_pos = 0
                temp_neg = 0
    return ans

def gain(x,y,type,typese,pre):#计算每个属性的信息增益 pre是打表值
    ans = np.zeros([x.shape[1],5])
    entD = -(np.sum(y == 0)/y.shape[0])*(log(np.sum(y == 0)/y.shape[0],2))-(np.sum(y == 1)/y.shape[0])*(log(np.sum(y == 1)/y.shape[0],2))
    for i in range(0,x.shape[1]):
        if type[i] == 0:#离散型计算
            ans[i,0] = entD
            for j in range(0,typese[i].shape[0]):
                ans[i,0] -= ent(x,y,i,j,pre)*np.sum(x[...,i] == typese[i,j])/y.shape[0]
        if type[i] == 1:#连续型计算
            max = 0
            flag = -1
            
            for j in range(0,x.shape[0] - 1):
                ent_neg = -(pre[i,j,0,0]*log(pre[i,j,0,0],2)) - (pre[i,j,1,0]*log(pre[i,j,1,0],2))
                ent_pos = -(pre[i,j,0,1]*log(pre[i,j,0,1],2)) - (pre[i,j,1,1]*log(pre[i,j,1,1],2))
                k = pre[i,j,0,1]*log(pre[i,j,0,1],2)
                temp = entD - ent_neg*(j+1)/y.shape[0] - ent_pos*(y.shape[0] - j - 1)/y.shape[0]
                if temp >= max :
                    max = temp
                    flag = j
            ans[i,0] = max
            sort_temp = np.array(np.c_[x,y][np.c_[x,y][:,i].argsort()][...,i],dtype = float)
            ans[i,1] = (sort_temp[flag]+sort_temp[flag+1])/2
    return ans#第一列给出每个属性的信息增益，第二列是连续属性的最大划分点

def ent(x,y,i,j,pre):#计算离散型ent的函数
    ent = -pre[i,j,0,0] * log(pre[i,j,0,0],2) - pre[i,j,1,0] * log(pre[i,j,1,0],2)
    return ent

#主程序

def decitree(x,y,typese,type,A,flag_,N,tree,s,t):


    if flag_ != -1 and type[flag_] == 0:#离散属性 取出Dv
        bo = x[...,flag_] != typese[flag_,N]
        x = np.delete(x,bo,0)
        y = np.delete(y,bo,0)

    else:#连续属性取出Dv
        if flag_ != -1:
            temp = gain(x,y,type,typese,pre(x,y,typese,type))
            a = np.array(x[...,flag_],dtype = float)
            if typese[flag_,N] == '1':
                bo = a <= temp[flag_,1]
                x = np.delete(x,bo,0)
                y = np.delete(y,bo,0)
            else:
                bo = a >= temp[flag_,1]
                x = np.delete(x,bo,0)
                y = np.delete(y,bo,0)

    if np.sum(A == 1) == 0 or np.sum(x == x[0]) == np.size(x):#属性被用光或者无法区别
        if np.sum(y == 0) >= np.sum(y == 1): 
            tree[s,t,0] = '-999'
            return tree
        else:
            tree[s,t,0] = '-1000'
            return tree
    if np.sum(y == 1) == y.shape[0] or np.sum(y == 0) == y.shape[0]:#全是反例或正例，不需要继续划分
        tree[s,t,0] = -999-y[0]
        if type[flag_] == 1:
            tree[s, t, 1] = temp[flag_,1]
        return tree
    else:#能够继续区别
        ans = gain(x,y,type,typese,pre(x,y,typese,type))
        flag = np.array(np.where(ans[...,0] == np.max(ans[...,0])))#信息增益最大的位置

        for p in range(0,flag.shape[1]):
            if A[flag[0,p]] != -1:
                tree[s,t,0] = flag[0,p]
                point = p
        A[flag[0,point]] = -1
        if flag[0,point] <= 4:
            tree = decitree(x,y,typese,type,A,flag[0,point],0,tree,s + 1,3 * t + 0)
            tree = decitree(x,y,typese,type,A,flag[0,point],1,tree,s + 1,3 * t + 1)
            tree = decitree(x,y,typese,type,A,flag[0,point],2,tree,s + 1,3 * t + 2)
            return tree
        else:
            if flag[0,point] >= 5:
                tree = decitree(x,y,typese,type,A,flag[0,point],0,tree,s + 1,3 * t + 0)

                tree = decitree(x,y,typese,type,A,flag[0,point],1,tree,s + 1,3 * t + 1)
                return tree


def test(x,y,tree,typese,type):#应用决策树
    i = 0
    j = 0
    while 1:
        if tree[i,j,0] == -999 or tree[i,j,0] == -1000:
            return -999 - tree[i,j,0]
        else:
            a = tree[i,j,0]
            a = int(a)
            if type[a] == 0:
                if x[a] == typese[a,0]: 
                    j = 3 * i + 0 
                if x[a] == typese[a,1]:
                    j = 3 * i + 1 
                if x[a] == typese[a,2]:
                    j = 3 * i + 2
            else:
                if float(x[a][0]) <= tree[i,j,1]: 
                    j += 3 * i + 0 
                else:
                    j += 3 * i + 1
            i = i + 1


fp = open(r'..\waterlemon.txt','r',encoding='UTF-8')

data = []

for line in fp:
    line = line.strip('\n')
    data.append(line.split(' '))
fp.close()#读入

x = np.array(data)[...,0:8]
y = np.array(data)[...,8]


y = np.array(y,dtype = int)
x[...,6] = np.array(x[...,6],dtype = float)
x[...,7] = np.array(x[...,7],dtype = float)#转换类型


type = np.array([0,0,0,0,0,0,1,1])
typese = np.array([['乌黑','青绿','浅白'],['蜷缩','稍蜷','硬挺'],['浊响','沉闷','清脆'],['清晰','稍糊','模糊'],['凹陷','稍凹','平坦'],['硬滑','软粘',''],[-1,1,0],[-1,1,0]])
#类型控制 后续可引入新函数自动识别类型和属性

A = np.array([1,1,1,1,1,1,1,1])

#以下为决策树主体(基于信息熵和信息增益)

#ans = gain(x,y,type,typese,pre(x,y,typese,type))

tree = -np.ones([8,24,2])#使用一个二维表存储决策树，设置为三叉树
'''
例如
3
6 5 -999
-999 -1000 -1 -999 -1000 -1

则对应了P85的决策树

tree的第三维度存储着连续性属性的划分点
'''

tree = decitree(x,y,typese,type,A,-1,-1,tree,0,0)



#print(tree)

for i in range(0,x.shape[0]):
    print(y[i] == test(x[i],y[i],tree,typese,type))

#误差计算（召回）




"""

已实现：

信息熵的决策树

存储决策树

计算正确率的函数

未实现：

不同剪枝类型有不同的决策树生成方式

使用queue改良

dfs和bfs哪种更好?

gini指数型决策树

多变量决策树

"""