from glob import glob
import pandas as pd
import numpy as np
import math
from scipy import stats
import datetime


def getp(a, b, p, delt, r):
    price = math.e**(-delt*r) * (p*a + (1 - p)*b)
    return price


def crr(sigma, step, delt, S0, K, r, type):
    mat = np.zeros([step+1, step+1])
    mat = pd.DataFrame(mat)
    mat.iloc[0, 0] = 1
    u = math.e**(sigma * np.sqrt(delt))
    d = math.e**(-sigma * np.sqrt(delt))
    p = (math.e**(r * delt) - d)/(u - d)
    option_type = type.split('_')[1]
    buy_type = type.split('_')[0]
    for i in range(1, step+1):
        for j in range(i+1):
            mat.iloc[j, i] = u**(i-j)*d**(j)
    S = S0 * mat
    op_call = S - K
    op_pull = K - S
    op_call[op_call < 0] = 0
    op_pull[op_pull < 0] = 0
    if option_type == 'call':
        op2 = op_call.copy()
        for i in range(step, -1, -1):
            for j in range(i):
                op2.iloc[j, i-1] = getp(a=op2.iloc[j, i],
                                        b=op2.iloc[j+1, i], p=p, delt=delt, r=r)
                if buy_type == 'American':
                    op2.iloc[j, i-1] = max(op2.iloc[j, i-1],
                                           op_call.iloc[j, i-1])
                elif buy_type == 'Euro':
                    continue
    if option_type == 'put':
        op2 = op_pull.copy()
        for i in range(step, -1, -1):
            for j in range(i):
                op2.iloc[j, i-1] = getp(a=op2.iloc[j, i],
                                        b=op2.iloc[j+1, i], p=p, delt=delt, r=r)
                if buy_type == 'American':
                    op2.iloc[j, i-1] = max(op2.iloc[j, i-1],
                                           op_pull.iloc[j, i-1])
                elif buy_type == 'Euro':
                    continue
    return op2.iloc[0, 0]

#test = crr(sigma = 0.4068, step = 6, delt = 1/12, S0 = 50, K = 52.08, r = 0.04, type = 'Euro_call')


def BS(sigma, S0, K, r, t):
    d1 = (math.log(S0/K) + (r + 0.5 * (sigma ** 2)) * t)/(sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    c = S0*stats.norm.cdf(d1, 0, 1) - K * \
        math.e**(-r * t)*stats.norm.cdf(d2, 0, 1)
    return c


'''
def N_m(f, x0):
    while np.abs(f.subs('x', x0)) > 0.0001:
        x = Symbol("x")
        dx0 = diff(f,x)
        if dx0.subs('x', x0) == 0:
            print("you cannot use this x0 as start")
            break
        else:
            x1 = x0 - f.subs('x', x0)/dx0.subs('x', x0)
            x0 = x1
    return x0

def diff(f, x):
    fx = f(x)
    x1 = x + 0.0001
    fx1 = f(x1)
    dff = (fx1 - fx) / 0.0001
    return dff
'''


def N_m(S0, K, r, t, c):
    x0 = 0
    while abs(BS(x0, S0, K, r, t) - c) > 1e-8:
        # for i in range(10000):
        x01 = x0 + 0.01
        dx0 = BS(x01, S0, K, r, t)/0.01 - BS(x0, S0, K, r, t)/0.01
        if dx0 == 0:
            print("you cannot use this x0 as start")
            break
        else:
            x1 = x0 - (BS(x0, S0, K, r, t) - c)/dx0
            x0 = x1
    return x0


def B_m(S0, K, r, t, c):
    x0 = 0
    x1 = 25
    x2 = (x0 + x1)/2
    while abs(BS((x0 + x1)/2, S0, K, r, t) - c) > 0.0001:
        # for i in range(100):
        x2 = (x0 + x1)/2
        if BS(x2, S0, K, r, t) - c < 0:
            x0 = x2
        elif BS(x2, S0, K, r, t) - c > 0:
            x1 = x2
    return x2


def inter(x1, x2, t, T):  # 插值法，x1表示t1值，x2表示t2值，t表示插入值位置，T表示t1，t2距离
    y = (x2 - x1) * (t - 1) / (T - 1) + x1
    return y


data = pd.read_excel("E:/SHIBOR.xls")
data.drop(index=[0, 2804, 2805], inplace=True)
data["指标名称"] = pd.to_datetime(data["指标名称"])

'''
def rint(t, t0): #选取2006-10-09作为起始日
    t0 = datetime.datetime.strptime(t0, "%Y-%m-%d")
    #t = datetime.datetime.strptime(t, "%Y-%m-%d")
    #delta = t
    #deltaday = int(delta.days)
    b = data.loc[data['指标名称'] == t0].index.values[0] - 1
    deltaday = t
    if deltaday < 8 and deltaday >= 1:
        x1 = data.iloc[b,1]
        x2 = data.iloc[b,2]
        outdata = inter(x1, x2, deltaday, 7)
        return outdata
    elif deltaday < 15 and deltaday >= 7:
        x1 = data.iloc[b,2]
        x2 = data.iloc[b,3]
        outdata = inter(x1, x2, deltaday - 6, 8)
        return outdata
    elif deltaday < 31 and deltaday >= 14:
        x1 = data.iloc[b,3]
        x2 = data.iloc[b,4]
        outdata = inter(x1, x2, deltaday - 13, 17)
        return outdata
    elif deltaday < 91 and deltaday >= 30:
        x1 = data.iloc[b,4]
        x2 = data.iloc[b,5]
        outdata = inter(x1, x2, deltaday - 29, 61)
        return outdata
    elif deltaday < 181 and deltaday >= 90:
        x1 = data.iloc[b,5]
        x2 = data.iloc[b,6]
        outdata = inter(x1, x2, deltaday - 89, 91)
        return outdata
    elif deltaday < 271 and deltaday >= 180:
        x1 = data.iloc[b,6]
        x2 = data.iloc[b,7]
        outdata = inter(x1, x2, deltaday - 179, 91)
        return outdata
    elif deltaday < 366 and deltaday >= 270:
        x1 = data.iloc[b,7]
        x2 = data.iloc[b,8]
        outdata = inter(x1, x2, deltaday - 269, 96)
        return outdata

'''


def rint(t, t0):  # t0选取2006-10-09作为起始日,t代表第几天
    t0 = datetime.datetime.strptime(t0, "%Y-%m-%d")
    b = data.loc[data['指标名称'] == t0].index.values[0] - 1
    days = [1, 7, 14, 30, 90, 180, 270, 365]
    for i in range(len(days)):
        if t < (days[i+1] + 1) and t >= days[i]:
            x1 = data.iloc[b, i + 1]
            x2 = data.iloc[b, i + 2]
            outdata = inter(x1, x2, t - days[i] + 1, days[i+1] - days[i] + 1)
            return outdata
        else:
            i = i + 1


'''示例：
ls = []
for j in range(1, 366):
    r = rint(j, "2010-10-09")
    ls.append(r)
'''

edata = pd.read_csv('E:/50ETF.csv')  # 50ETF数据读入并做处理
edata['Unnamed: 0'] = pd.to_datetime(edata['Unnamed: 0'])

info = pd.read_csv("E:/OptInfo.csv")  # 期权信息数据读入并做处理
info = info[['code', 'name', 'exercise_price', 'list_date', 'expire_date']]


OptionFile = 'E:/OptPrice2/*.csv'

option_info = []

for file in glob(OptionFile):
    option_info.append(pd.read_csv(file))


option_data = pd.concat(option_info)
option_data['date'] = pd.to_datetime(option_data['date'])  # 得到期权信息总表

info['list_date'] = pd.to_datetime(info['list_date'])
info['expire_date'] = pd.to_datetime(info['expire_date'])
info['CoP'] = (info.name.apply(lambda x: x[5]) == '购').values.astype(int)
info = info.loc[info['CoP'] == 1]

t = '2015-12-01'  # 设置日期
td = datetime.datetime.strptime(t, "%Y-%m-%d")  # 日期转为时间戳

info_filt = info[info['list_date'] <
                 td][info['expire_date'] > td]  # 筛选设置日期处于交易日之间的期权
code_filt = info_filt['code'].drop_duplicates().to_list()  # 得到筛选后的期权列表

option_filt = option_data[option_data['code'].isin(code_filt)]  # 过滤得到期权信息
option_filt.dropna(inplace=True)  # 去除nan值

last_data = pd.DataFrame({"code": [], "t0": [], "t": [], "S0": [], "k": [
], "T": [], "c": [], "vol": [], "rf": []})  # 新建dataframe用于存储数据
data_list = []

for i in code_filt:  # 期权循环
    ind_data = option_filt.loc[option_filt['code'] == i]
    ind_data = ind_data[ind_data['date'] >= td]
    ind_data.index = ind_data['date']  # 筛选出交易日大于所设置天数的期权交易信息
    j = ind_data['date'][-1]  # 对天数循环
    true_T = (j - td).days  # 真实天数用于计算无风险利率
    delta_T = (ind_data.iloc[-1]['Unnamed: 0'] -
               ind_data.iloc[0]['Unnamed: 0'])/252  # 交易天数用于计算期权波动率
    rf = rint(true_T, t) * 0.01
    s = edata.loc[edata['Unnamed: 0'] == td].iloc[0, 1]
    k = ind_data.loc[ind_data['date'] == td].iloc[0, 3]
    c = ind_data.loc[ind_data['date'] == td].iloc[0, 4]
    try:
        vol = B_m(s, k, rf, delta_T, c)
    except ZeroDivisionError:
        vol = np.nan
    day_data = pd.DataFrame({"code": [i], "t0": [td], "t": [j], "S0": [s], "k": [
                            k], "T": [delta_T], "c": [c], "vol": [vol], "rf": [rf]})
    data_list.append(day_data)

last_data = pd.concat(data_list)
