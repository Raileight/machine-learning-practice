import numpy as np
import math

'''
朴素贝叶斯分类器含两步

1 计算各个先验概率 含离散型和连续性 离散型即直接计数 连续性则假设为正态分布计算均值和方差 其中需要使用拉普拉斯修正使未观测的属性不影响最后的结果

2 查询 利用1中计算的先验概率表计算后验概率 使用概率最高的类别作为查询结果

需要函数

先验概率计算 prior_calc(数据 标签 属性类型(离散0 连续1) 属性类别) 输出一个先验概率表 pc[i] pxc[i][j]
查询函数 query 输出最佳标签
'''

'''
以下为先验概率计算函数
'''

def prior_calc(x,y,typese):
    #计算p(c)
    good = np.sum(y == 1)
    bad = np.sum(y == 0)
    pc = np.zeros(2)#直接固定为两类 后续扩展需要更改
    pc[0] = (bad + 1) / (y.shape[0] + 2)#拉普拉斯修正
    pc[1] = (good + 1) / (y.shape[0] + 2)

    #计算pxc
    pxc = np.zeros([8,3,2])
    for i in range(0,x.shape[1]):
        if i <= 4:#离散型 且有三个取值
            for j in range(0,3):
                pxc[i,j,0] = (np.sum(np.logical_and(x[...,i] == typese[i,j],y == 0)) + 1) / (np.sum(y == 0) + 3)
                pxc[i,j,1] = (np.sum(np.logical_and(x[...,i] == typese[i,j],y == 1)) + 1) / (np.sum(y == 1) + 3)
            continue
        if i == 5:
            for j in range(0,2):
                pxc[i,j,0] = (np.sum(np.logical_and(x[...,i] == typese[i,j],y == 0)) + 1) / (np.sum(y == 0) + 3)
                pxc[i,j,1] = (np.sum(np.logical_and(x[...,i] == typese[i,j],y == 1)) + 1) / (np.sum(y == 1) + 3)
            continue
        if i > 5:
            pxc[i,0,0] = np.mean(x[...,i][y == 0])
            pxc[i,1,0] = np.std(x[...,i][y == 0])#存储均值和方差用于计算
            pxc[i,0,1] = np.mean(x[...,i][y == 1])
            pxc[i,1,1] = np.std(x[...,i][y == 1])#存储均值和方差用于计算
    return pc,pxc

'''
以下为查询函数 朴素贝叶斯主体
'''

def query(x,typese,pc,pxc):
    #计算好瓜的后验概率
    good = math.log(pc[1])
    bad = math.log(pc[0])
    for i in range(0,x.shape[0]):
        if i < 5:
            flag = np.where(typese[i,...] == x[i])[0][0]#x的第i个属性是typese里的哪一个
            bad = bad + math.log(pxc[i][flag][0])
            good = good + math.log(pxc[i][flag][0])
        else:
            bad = bad + math.log(1 / (math.sqrt(2 * math.pi) * pxc[i,1,0]) * math.exp(-(x[i] - pxc[i,0,0]) ** 2 / (2 * pxc[i,1,0] ** 2)))
            good = good + math.log(1 / (math.sqrt(2 * math.pi) * pxc[i,1,1]) * math.exp(-(x[i] - pxc[i,0,1]) ** 2 / (2 * pxc[i,1,1] ** 2)))

    if bad > good:
        return 0
    else:
        return 1
    
'''
验证
'''

if '__main__' == __name__:

    fp = open(r'..\waterlemon.txt','r',encoding='UTF-8')

    data = []

    for line in fp:
        line = line.strip('\n')
        data.append(line.split(' '))
    fp.close()#读入

    x = np.array(data)[...,0:8]
    y = np.array(data)[...,8]

    typese = np.array([['乌黑','青绿','浅白'],['蜷缩','稍蜷','硬挺'],['浊响','沉闷','清脆'],['清晰','稍糊','模糊'],['凹陷','稍凹','平坦'],['硬滑','软粘',''],[0,0,0],[0,0,0]])

    for j in range(0,6):
        for i in range(0,17):
            for k in range(0,3):
                if x[i,j] == typese[j,k]:
                    x[i,j] = k
    x = np.array(x,dtype = float)
    y = np.array(y,dtype = float)

    typese = np.array([[0,1,2],[0,1,2],[0,1,2],[0,1,2],[0,1,2],[0,1,2],[0,0,0],[0,0,0]])

    pc,pxc = prior_calc(x,y,typese)
  
    test = np.array([1,0,0,0,0,0.697,0.460])

    print(query(test,typese,pc,pxc))#验证





    



