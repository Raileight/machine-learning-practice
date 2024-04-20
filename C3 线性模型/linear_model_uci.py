import array as arr
import numpy as np
import math

#线性模型+牛顿法+uci+十次十折+留一

fp = open(r'..\data.txt','r',encoding='utf-16')

data = []

for line in fp:
    line = line.strip('\n')
    data.append(line.split(' '))

fp.close()

data = np.array(data,dtype = float)

def l(x,y,beta):
    res = 0
    for i in range(0,810):
        res += -y[i]*np.dot(beta,x[i,...])+math.log(1 + math.exp(np.dot(beta,x[i,...])))
    return res

def p1(x_i,beta):
    return math.exp(np.dot(beta,x_i))/(1 + math.exp(np.dot(beta,x_i)))

def l1(x,y,beta):
    res = 0

    for i in range(0,810):
        res +=  x[i,...]*(y[i]-p1(x[i,...],beta))
    return -res

def l3(x,y,beta):
    res = 0
    for i in range(0,810):
        res += np.dot(x[i,...].reshape(7,1),x[i,...].reshape(1,7))*(p1(x[i,...],beta)*(1 - p1(x[i,...],beta)))
    return res

x = data[:,0:7].reshape(900,7)
y = data[:,7].reshape(900,1)

def train(N):
    x_del = np.delete(x,slice(45*N,45*N+45),0)
    x_del = np.delete(x_del,slice(45*N+450,45*N+45+450),0)
    y_del = np.delete(y,slice(45*N,45*N+45),0)
    y_del = np.delete(y_del,slice(45*N+450,45*N+45+450),0)
    beta = np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1])

    for i in range(0,31):
        beta = beta-np.dot(np.matrix(l3(x_del,y_del,beta)).I,l1(x_del,y_del,beta))
    Y = np.empty([810,1])
    cor = 0
    for i in range(0,45):
        Y[i] = 1 / (1 + math.exp(-(np.dot(x[45 * N + i],beta.reshape(7,1)))))
        if (Y[i] >= 0.5 and y[45 * N + i] == 1) or (Y[i] < 0.5 and y[45 * N + i] == 0) : cor += 1
    for i in range(0,45):
        Y[i] = 1 / (1 + math.exp(-(np.dot(x[45 * N + 450 + i],beta.reshape(7,1)))))
        if (Y[i] >= 0.5 and y[45 * N + 450 + i] == 1) or (Y[i] < 0.5 and y[45 * N + 450 + i] == 0) : cor += 1
    return cor/90

def train3(N):
    x_del = np.delete(x,N,0)

    y_del = np.delete(y,N,0)

    beta = np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1])

    for i in range(0,31):
        beta = beta-np.dot(np.matrix(l3(x_del,y_del,beta)).I,l1(x_del,y_del,beta))
    Y = np.empty([899,1])
    cor = 0
    Y = 1 / (1 + math.exp(-(np.dot(x[N],beta.reshape(7,1)))))
    if (Y >= 0.5 and y[N] == 1) or (Y < 0.5 and y[N] == 0): return 1
    else: return 0

ans = 0
for i in range(0,10):
    ans += train(i)
print(ans/10)

ans = 0
for i in range(0,900):
    ans += train3(i)
    print(i)
print(ans/900)




    






    





