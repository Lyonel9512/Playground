
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 20:56:04 2019

@author: Lyonel
"""

from time import time
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 设定M，A，C



def lcg(seed, mi, ma, num):
    m = 2**32
    a = 663608941
    c = 0
    delta = ma-mi
    rdls = []
    for i in range(num):
        seed = (a * seed + c) % m
        rdls.append(((delta*seed/(m-1)) + mi)/delta)
    return rdls
    


def random(n):
    num = n
    mi = 0
    ma = 100
    seed = time()
    rd = lcg(seed, mi, ma, num)
    return rd


random(100)

# runs test


def runs_test(rint): #游程检验
    ls = rint
    ls = pd.DataFrame(ls)
    ls['bool'] = ((ls - ls.shift(1)) > 0).values.astype(int)
    R = 1
    for i in range(2, len(ls)):
        if ls.iloc[i, 1] == ls.iloc[i-1, 1]:
            R = R
        else:
            R = R + 1
    ER = (2*(len(ls)) - 1)/3
    VR = (3*(len(ls)) - 5)/18
    if ((R - ER)/math.sqrt(VR)) < 3.29:
        out = str('independent')
    else:
        out = str('non-independent')
    print(out)

runs_test(random(100))

'''
=============卡方检验======================
'''
rdls = random(100)
rdls_df = pd.DataFrame(rdls)
chimat = pd.DataFrame(index = rdls_df.index,columns = rdls_df.index)
for i in range(len(rdls)):
    for j in range(len(rdls)):
        chimat.iloc[i,j] = [rdls[i], rdls[j]]
A1 = A2 = A3 = A4 = 0
for i in range(len(rdls)):
    for j in range(len(rdls)):
        if chimat.iloc[i,j][0] < 0.5 and chimat.iloc[i,j][1] > 0.5:
            A1 = A1 + 1
        elif chimat.iloc[i,j][0] > 0.5 and chimat.iloc[i,j][1] > 0.5:
            A2 = A2 + 1
        elif chimat.iloc[i,j][0] < 0.5 and chimat.iloc[i,j][1] < 0.5:
            A3 = A3 + 1
        elif chimat.iloc[i,j][0] > 0.5 and chimat.iloc[i,j][1] < 0.5:
            A4 = A4 + 1
chi2 = (((A1 - 2500)**2/2500) + ((A2 - 2500)**2/2500) + ((A3 - 2500)**2/2500) + ((A4 - 2500)**2/2500))

'''
===========随机概率分布生成=============
'''
def pareto(alpha, k, n):
    temprd = random(n)
    pareto_ind = k/((1 - np.array(temprd)) ** (1/alpha))
    return pareto_ind


def edis(lamda, n):
    tempedis = random(n)
    edis_ind = np.log(np.array(tempedis))/(-lamda)
    return edis_ind

'''
==============蒙特卡洛价格模拟===========
'''
def GenP(n):
    sigma = 0.2
    r = 0.02
    rdls = random(n)
    edls = edis(2,n)
    theta = 2 * np.pi * np.array(rdls)
    X = np.sqrt(edls)*np.cos(theta)
    ei = X*sigma/np.sqrt(n)
    Ei = ei + (r/n) - ((sigma**2)/(2*n))
    Ei = Ei.T
    S0 = np.zeros(n)
    S0[0] = 100
    for i in range(len(S0)-1):
        S0[i+1] = S0[i] * np.exp(Ei[i])
    return S0



for i in range(100):
    S = GenP(1000)
    plt.plot(np.arange(1000), S)
    
'''
============simcev生成============
'''
    
def simcev(n, r, sigma, So, T, gam):
    Yt = np.ones([n,1])*(So**(1-gam))/(1-gam)
    y = Yt
    dt = T/1000
    c1 = r*(1-gam)/sigma
    c2 = gam*sigma/(2*(1-gam))
    dw = np.random.randn(n, 1000) * np.sqrt(np.sqrt(dt))
    for i in range(1000):
        z = np.argwhere(Yt == 0)
        v = np.nonzero(Yt)
        Yt = np.maximum(0, Yt[v] + (c1 * Yt[v] - c2/Yt[v])*dt + dw[v,i][0])
        try:
            y = np.column_stack((y, Yt))
        except ValueError:
            for zi in list(z[:,0]):
                Yt = np.insert(Yt, zi, values=0, axis=0)
            y = np.column_stack((y, Yt))
    s = ((1 -gam)*np.maximum(y,0)) ** (1/(1-gam))
    return s

S = simcev(1000, 0.05, 0.2, 10, 0.25, 0.8)

St = S[:, 1000]
'''
=================美式期权===============
'''
def reg(y, X):
    m = np.linalg.inv(np.dot(np.transpose(X), X))
    param = np.dot(np.dot(m, np.transpose(X)), y)
    return param

st = [1.34, 1.54, 1.03, 0.92, 1.52, 0.90, 1.01, 1.34]
s2 = [1.08, 1.26, 1.07, 0.97, 1.56, 0.77, 0.84, 1.22]
s1 = [1.09, 1.16, 1.22, 0.93, 1.11, 0.76, 0.92, 0.88]
s0 = [1]*8
mat = pd.DataFrame(data = [s0,s1,s2,st])
mat = mat.T
  
def itter(x, y, k, rf):
    s = []
    for i in x:
        if i > k:
            s.append(0)
        else:
            s.append(i)
    Y = y * math.e ** (-rf)
    B = np.column_stack((np.array(s),np.array(s)**2))
    est = sm.OLS(Y, sm.add_constant(B)).fit()
    cont = np.array([max(i,0) for i in est.predict()])
    exer = np.array([max(k - i,0) for i in x])
    s2 = []
    for i in range(len(exer)):
        if exer[i] > cont[i]:
            s2.append(exer[i])
        else:
            s2.append(0)
    s2 = np.array(s2)
    return s2 #itter(s2, st_e, 1.1, 0.06)

def lsm(mat,k,rf): 
    smat = pd.DataFrame(np.zeros([len(mat.index),len(mat.columns)-1]))
    st = mat.iloc[:,-1]
    st_k = np.array([max(k - i,0) for i in st])
    smat.iloc[:,-1] = st_k
    for i in range(len(smat.columns)-1,-1,-1):
        if i != 0:
            st_k2 = itter(mat.iloc[:,i],st_k,k=k,rf=rf)
            smat.iloc[:,i-1] = st_k2
            st_k = st_k2
        else:
            se = smat.iloc[:,0]
            ind = se[se != 0].index
            smat.iloc[ind,1:len(smat.columns)] = 0
            for i in range(1, len(smat.columns)):
                smat.iloc[:,0] = smat.iloc[:,0] + smat.iloc[:,i]*(math.e ** (-0.06*i))
            price = smat.iloc[:,0].mean()
    return smat, price

lsm(mat, 1.1, 0.06)