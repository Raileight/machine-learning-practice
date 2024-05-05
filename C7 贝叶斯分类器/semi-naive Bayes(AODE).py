import numpy as np
import math

'''
为了简化，这里将连续性属性去除
半朴素贝叶斯分类器(AODE)

1 计算各个先验概率 含离散型和连续性 离散型即直接计数 连续性则假设为正态分布计算均值和方差 其中需要使用拉普拉斯修正使未观测的属性不影响最后的结果
p(c|x) = pc_x
p(x|c,x) = px_cx

2 查询 利用1中计算的先验概率表计算后验概率 使用概率最高的类别作为查询结果

需要函数

先验概率计算 prior_calc(数据 标签 属性类别) 输出一个先验概率表 pc[i] pxc[i][j]
查询函数 query 输出最佳标签
'''

'''
以下为先验概率计算函数
'''

def prior_calc(x,y,typese):
    #计算p(c,x_i)
    pcx = np.zeros([2,6,3])#直接固定为两类 后续扩展需要更改
    for i in range(0,x.shape[1]):
        if i < 5:
            for j in range(0,3):
                pcx[0,i,j] = (np.sum(np.logical_and(y == 0,x[...,i] == typese[i,j])) + 1) / (x.shape[0] + 2 * 3)#拉普拉斯修正
                pcx[1,i,j] = (np.sum(np.logical_and(y == 1,x[...,i] == typese[i,j])) + 1) / (x.shape[0] + 2 * 3)
                continue
        if i == 5:
            for j in range(0,2):
                pcx[0,i,j] = (np.sum(np.logical_and(y == 0,x[...,i] == typese[i,j])) + 1) / (x.shape[0] + 2 * 2)#拉普拉斯修正
                pcx[1,i,j] = (np.sum(np.logical_and(y == 1,x[...,i] == typese[i,j])) + 1) / (x.shape[0] + 2 * 2)
                continue

    #计算p(x_j|c,x_i)
    px_cx = np.zeros([6,3,2,6,3])
    for i in range(0,x.shape[1]):
        for j in range(0,x.shape[1]):
            if j <= 4:#离散型 且有三个取值
                for k in range(0,3):
                    for t in range(0,3):#p(x_j == k|x_i == t,c)
                        px_cx[j,k,0,i,t] = (np.sum(np.logical_and(x[...,i] == typese[i,t],y == 0,x[...,j] == typese[j,k])) + 1) / (np.sum(y == 0) + 3)
                        px_cx[j,k,1,i,t] = (np.sum(np.logical_and(x[...,i] == typese[i,t],y == 1,x[...,j] == typese[j,k])) + 1) / (np.sum(y == 1) + 3)
            if j == 5:
                for k in range(0,3):
                    for t in range(0,3):#p(x_j == k|x_i == t,c)
                        px_cx[j,k,0,i,t] = (np.sum(np.logical_and(x[...,i] == typese[i,t],y == 0,x[...,j] == typese[j,k])) + 1) / (np.sum(y == 0) + 2)
                        px_cx[j,k,1,i,t] = (np.sum(np.logical_and(x[...,i] == typese[i,t],y == 1,x[...,j] == typese[j,k])) + 1) / (np.sum(y == 1) + 2)
    return pcx,px_cx

'''
以下为查询函数 朴素贝叶斯主体
'''

def query(x,typese,pcx,px_cx):
    #计算好瓜的后验概率
    good = 0
    bad = 0
    for i in range(0,6):
        flag_i = np.where(typese[i,...] == x[i])[0][0]
        temp = pcx[0,i,flag_i]
        for j in range(0,6):
            flag_j = np.where(typese[j,...] == x[j])[0][0]
            temp = temp * px_cx[j,flag_j,0,i,flag_i]
        bad += temp

        temp = pcx[1,i,flag_i]
        for j in range(0,6):
            flag_j = np.where(typese[j,...] == x[j])[0][0]
            temp = temp * px_cx[j,flag_j,1,i,flag_i]
        good += temp

    if bad > good:
        return 0
    else:
        return 1
    
'''
验证
'''

if '__main__' == __name__:

    fp = open(r'D:\桌面\南大LAMDA\代码\西瓜书\第七章贝叶斯分类器\waterlemon.txt','r',encoding='UTF-8')

    data = []

    for line in fp:
        line = line.strip('\n')
        data.append(line.split(' '))
    fp.close()#读入

    x = np.array(data)[...,0:6]
    y = np.array(data)[...,8]

    typese = np.array([['乌黑','青绿','浅白'],['蜷缩','稍蜷','硬挺'],['浊响','沉闷','清脆'],['清晰','稍糊','模糊'],['凹陷','稍凹','平坦'],['硬滑','软粘','']])

    for j in range(0,6):
        for i in range(0,17):
            for k in range(0,3):
                if x[i,j] == typese[j,k]:
                    x[i,j] = k
    x = np.array(x,dtype = float)
    y = np.array(y,dtype = float)

    typese = np.array([[0,1,2],[0,1,2],[0,1,2],[0,1,2],[0,1,2],[0,1,2]])

    pc,pxc = prior_calc(x,y,typese)
  
    test = np.array([1,0,0,0,0,0])

    print(query(test,typese,pc,pxc))#验证





    



