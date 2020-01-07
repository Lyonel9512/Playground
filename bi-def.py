# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 11:49:33 2019

@author: Lyonel
"""

import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.utils.data as Data
import torchvision.transforms as transforms
import sys
import random
import math
sys.path.append("E:/")
import d2lzh as d2l
import torch.nn.functional as F
import operator
import matplotlib.pyplot as plt
import matplotlib
import time
from pandas import Series
from pandas import DataFrame
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb



'''
'''
data = pd.read_csv('E:/data/test.csv')
data_trans = pd.get_dummies(data)
data_trans['Shop_1'] = ''
data_trans['Shop_2'] = ''
data_trans['Shop_3'] = ''
data_trans['Shop_4'] = ''
data_trans['Shop_5'] = ''
data_trans['Shop_1'] = data_trans['Shop'].map(lambda x: 1 if x == 1 else 0)
data_trans['Shop_2'] = data_trans['Shop'].map(lambda x: 1 if x == 2 else 0)
data_trans['Shop_3'] = data_trans['Shop'].map(lambda x: 1 if x == 3 else 0)
data_trans['Shop_4'] = data_trans['Shop'].map(lambda x: 1 if x == 4 else 0)
data_trans['Shop_5'] = data_trans['Shop'].map(lambda x: 1 if x == 5 else 0)
data_trans.drop(columns = 'Shop', inplace = True)
data_trans = data_trans.astype(float)

def normalize(x):
    xmean = x.mean()
    xstd = x.std()
    for i in range(len(x)):
        x[i] = (x[i] - xmean)/xstd
    return x
data_trans['Volume'] = normalize(data_trans['Volume'])
data_trans['Volume'] = normalize(data_trans['Volume'])
data_trans['Collect'] = normalize(data_trans['Collect'])
data_trans['Price'] = normalize(data_trans['Price'])
from sklearn.model_selection import train_test_split
X, Y = data_trans.drop(columns = 'Hot'), data_trans['Hot']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=33)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)

'''
先用LGB试试
'''
auc_mean_lgb = 0
y_pred_lgb = pd.DataFrame()
for k, (train_index, valid_index) in enumerate(skf.split(X_train, Y_train)):
    x_train, x_valid, y_train, y_valid = X_train.iloc[train_index], X_train.iloc[valid_index], Y_train.iloc[train_index], Y_train.iloc[valid_index]

    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_valid = lgb.Dataset(x_valid, y_valid, reference=lgb_train)

    params = {'bagging_fraction': 0.9986494501909666,
              'feature_fraction': 0.9865305012812717,
              'lambda_l1': 0.008271160906292652,
              'lambda_l2': 2.8796980614904624,
              'num_leaves': 10,
              'boosting_type': 'gbdt',
              'objective': 'binary',
              'metric': {'auc'},
              'max_depth': 4,
              'min_child_weight': 6,
              #   'num_leaves': 16,
              'learning_rate': 0.02,  # 0.05
              # 'feature_fraction': 0.7,
              # 'bagging_fraction': 0.7,
              'bagging_freq': 5,
              # 'lambda_l1':0.25,
              # 'lambda_l2':0.5,
              # 'scale_pos_weight':10.0/1.0, #14309.0 / 691.0, #不设置
              # 'num_threads':4,
              }
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=2000,
                    valid_sets=lgb_valid,
                    early_stopping_rounds=100,
                    verbose_eval=100)

    cv_y_pred = gbm.predict(x_valid, num_iteration=gbm.best_iteration)
    test_y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    auc = roc_auc_score(y_valid, cv_y_pred)

    auc_mean_lgb += auc
    y_pred_lgb[k] = test_y_pred
    print("AUC Score : %f" % auc)

print('auc_mean_lgb:', auc_mean_lgb/5)

'''
随便用个MLP试试
'''
num_inputs, num_hiddens, num_outputs = 13, 5, 2
net = nn.Sequential(nn.Linear(num_inputs, num_hiddens), nn.ReLU(), nn.Linear(num_hiddens, num_outputs))
loss = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr = 0.01)
batch_size = 10
num_epochs = 100
for k, (train_index, valid_index) in enumerate(skf.split(X_train, Y_train)):
    x_train, x_valid, y_train, y_valid = X_train.iloc[train_index], X_train.iloc[valid_index], Y_train.iloc[train_index], Y_train.iloc[valid_index]
    dataset = Data.TensorDataset(torch.tensor(x_train.values, dtype = torch.float), torch.tensor(y_train.values, dtype = torch.float))
    train_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
    net = net.float()
    train_ls = []
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X).float(), y.long())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_ls.append(l.item())

y_pred = net(torch.tensor(X_test.values, dtype = torch.float))
acc = ((y_pred.argmax(axis = 1) == torch.tensor(Y_test.values, dtype = torch.float)).float().sum().item())/70