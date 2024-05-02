import numpy as np
import math


fp = open(r'\waterlemon.txt','r',encoding='UTF-8')

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
typese = np.array([['乌黑','青绿','浅白'],['蜷缩','稍蜷','硬挺'],['浊响','沉闷','清脆'],['清晰','稍糊','模糊'],['凹陷','稍凹','平坦'],['硬滑','软粘',''],[0,0,0],[0,0,0]])

#编码

y_encode = np.zeros([17,2])

for j in range(0,6):
    for i in range(0,17):
        for k in range(0,3):
            if x[i,j] == typese[j,k]:
                x[i,j] = k
x = np.array(x,dtype = float)
for i in range(0,17):
    if y[i] == 0:
        y_encode[i] = [1,0]#坏
    else:
        y_encode[i] = [0,1]#好


#对特征进行编码
'''
'乌黑','青绿','浅白'
被替换为
0,1,2

好瓜1
被替换为
[0 1]
使得第一个输出神经元确定坏瓜，第二个神经元确定好瓜

事实上此处可以使用one-hot编码，处理无序关系的特征，相对的各个权重需要增长为原来的三倍数量
'''

'''
初始化阈值和权值 = 0.5

有8个输入神经元，单隐层也设置为8个，输出层设置为2个

单隐层神经元接受八个输入，于是各有八个权重以及一个阈值，共计9*8 = 72个值
输出层接受八个输入，有9*2 = 18个值
'''
v = np.random.rand(8,9)#随机初始化各个权重
w = np.random.rand(2,9) 

n = 0.1 #学习率

'''
计算单隐层的输出

接受输入值并输出八个神经元的值f(x_i - v[i,8])

f是sigmoid函数
'''

def f(x):
    return 1/(1 + (math.exp(-x)))

def hidden_layer(x):
    ans = np.zeros([8])
    for i in range(0,8):#神经元
        for j in range(0,8):#属性
            ans[i] += v[i,j] * x[j]#将每个属性j与对应的权重j乘，求和
        ans[i] -= v[i,8]#减去阈值
        ans[i] = f(ans[i])#激活

    return ans

'''
计算输出层的结果

接受单隐层的八个输出，激活并输出两个数值
'''
def output_layer(x):
    ans = np.zeros([2])
    for i in range(0,2):#神经元
        for j in range(0,8):#单隐层数量
            ans[i] += w[i,j] * x[j]#将每个属性j与对应的权重j乘，求和
        ans[i] -= w[i,8]#减去阈值
        ans[i] = f(ans[i])#激活

    return ans

'''
标准BP算法

计算输出层的均方误差并沿负梯度方向调整各个权重

首先计算梯度值
'''

def gradient(ans_hid,ans_out,x,y):#接受输出层结果和单隐层结果,实际标签,输入的数据
    global w
    global v
    
    #计算输出层权重和阈值的修正量,18个结果
    out = np.zeros([2,9])
    for i in range(0,2):#神经元
        for j in range(0,8):#前八个是权重
            out[i,j] = -(ans_out[i] * (1 - ans_out[i]) * (y[i] - ans_out[i])) * ans_hid[j]
        out[i,8] = ans_out[i] * (1 - ans_out[i]) * (y[i] - ans_out[i])
    
    #计算单隐层权重和阈值的修正量
    hid = np.zeros([8,9])
    for i in range(0,8):#神经元
        for j in range(0,8):#前八个是权重
            hid[i,j] = -ans_hid[i] * (1 - ans_hid[i]) * (w[0,i] * out[0,8] + w[1,i] * out[1,8]) * x[j]
        hid[i,8] = ans_hid[i] * (1 - ans_hid[i]) * (w[0,i] * out[0,8] + w[1,i] * out[1,8])
    
    #更新权重和阈值
    w = w - n * out
    v = v - n * hid
    return

'''
更新权重后计算下一个数据，反复更新
以下为神经网络主程序
'''

def neural_standardBP(x,y,MAX):#接受原始数据和标签以及训练轮数
    for j in range(0,MAX):
        for i in range(0,17):#一轮训练
            ans_hid = hidden_layer(x[i,...])
            ans_out = output_layer(ans_hid)
            gradient(ans_hid,ans_out,x[i,...],y[i,...])
            #print(w)

    for i in range(0,17):#计算均方误差
        ans_hid = hidden_layer(x[i,...])
        ans_out = output_layer(ans_hid)

        if (ans_out[0] - ans_out[1]) * (y[i,0] - y[i,1]) >= 0:
            print("正确！")
        else:
            print("错误！")
    return

neural_standardBP(x,y_encode,1000)


            


    



    


            





            






