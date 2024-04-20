import array as arr
import numpy as np
import math

#牛顿法+对率回归

def l(x,y,beta):
    res = 0
    for i in range(0,16):
        res += -y[i]*np.dot(beta,x[i,...])+math.log(1+math.exp(np.dot(beta,x[i,...])))
    return res

def p1(x_i,beta):
    return math.exp(np.dot(beta,x_i))/(1 + math.exp(np.dot(beta,x_i)))

def l1(x,y,beta):
    res = 0
    for i in range(0,16):
        res +=  x[i,...]*(y[i]-p1(x[i,...],beta))
    return -res

def l3(x,y,beta):
    res = 0
    for i in range(0,16):
        res += np.dot(x[i,...].reshape(3,1),x[i,...].reshape(1,3))*(p1(x[i,...],beta)*(1 - p1(x[i,...],beta)))
    return res

a = [0.697,0.460,
0.774,0.376,
0.634,0.264,
0.608,0.318,
0.556,0.215,
0.403,0.237,
0.481,0.149,
0.437,0.211,
0.666,0.091,
0.243,0.267,
0.245,0.057,
0.343,0.099,
0.639,0.161,
0.657,0.198,
0.360,0.370,
0.593,0.042,
0.719,0.103]



x = np.array(a)

x = x.reshape(17,2)

one = np.ones([17,1])

x = np.c_[x , one]

y = np.r_[np.ones([8,1]),np.zeros([9,1])]

beta = np.array([1,1,1])

for i in range(0,50):
    beta = beta-np.dot(np.matrix(l3(x,y,beta)).I,l1(x,y,beta))

Y = np.empty([17,1])

for i in range(0,16):
    Y[i] = 1 / (1 + math.exp(-(np.dot(x[i],beta.reshape(3,1)))))
    if (Y[i] >= 0.5 and y[i] == 1) or (Y[i] < 0.5 and y[i] == 0):
        print("correct")
    else: 
        print("error")


