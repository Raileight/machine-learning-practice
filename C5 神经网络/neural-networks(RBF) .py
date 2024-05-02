import numpy as np
import math

'''
异或的真值表
'''
x = np.array([[1,1],[0,0],[0,1],[1,0]])
y = np.array([0,0,1,1])

'''
使用高斯函数作为径向基函数
随机初始化参数beta

单隐层有两个神经元，于是输出层有两个权重，高斯函数具有两个权重
'''
b = np.random.rand(8) 
w = np.random.rand(8) 

n = 0.003 #学习率

'''
确定样本中心，此处为了方便直接确定样本中心
'''

c = np.random.rand(8,2)

'''
计算单隐层的输出
f是高斯径向基函数
'''

def f(x,i):
    return math.exp(-b[i] * np.linalg.norm(x - c[i]) ** 2)

def hidden_layer(x):
    ans = np.zeros([8])
    for i in range(0,8):
        ans[i] = f(x,i)#计算径向基函数
    return ans

'''
计算输出层的结果

接受单隐层的输出
'''
def output_layer(ans_hid):
    ans = 0
    for i in range(0,8):
        ans += w[i] * ans_hid[i]

    return ans

'''
标准BP算法

计算输出层的均方误差并沿负梯度方向调整各个权重

首先计算梯度值
'''

def gradient(ans_hid,ans_out,x,y):#接受输出层结果和单隐层结果,实际标签,输入的数据
    global w
    global b
    
    #计算输出层权重的修正量
    out = np.zeros([8])
    for i in range(0,8):
        out[i] = ans_hid[i] * (ans_out - y)
    
    #计算单隐层径向基函数参数的修正量
    hid = np.zeros([8])
    for i in range(0,8):
        hid[i] = -(ans_out - y) * w[i] * (np.linalg.norm(x - c[i]) ** 2) * ans_hid[i]
    
    #更新权重和参数
    w = w - n * out
    b = b - n * hid
    return

'''
更新权重后计算下一个数据，反复更新
以下为神经网络主程序
'''

def neural_RBF(x,y,MAX):#接受原始数据和标签以及训练轮数
    for j in range(0,MAX):
        for i in range(0,4):#一轮训练
            ans_hid = hidden_layer(x[i])
            ans_out = output_layer(ans_hid)
            gradient(ans_hid,ans_out,x[i],y[i])
            #print(E/4)

    for i in range(0,4):
        ans_hid = hidden_layer(x[i])
        ans_out = output_layer(ans_hid)
        if (ans_out > 0.5 and y[i] == 1) or (ans_out < 0.5 and y[i] == 0):
            print("正确！")
        else:
            print("错误！")
        

    return

neural_RBF(x,y,10000)


            


    



    


            





            






