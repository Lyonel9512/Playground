# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 19:45:07 2019

@author: Lyonel
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


old_sample = pd.read_csv('E:/data/min_data.csv')
ad_columns = old_sample.columns.to_list()
data = pd.read_csv('E:/data/000300.csv')
data['time'] = ''
data = data.reindex(columns=ad_columns)
data.time = pd.to_datetime(data['Unnamed: 0'])
#code_ls = old_sample.code.drop_duplicates().to_list()
#df = old_sample[old_sample.code == code_ls[0]]

#df.index = pd.to_datetime(df.time)
#df = df['2019-07-22':'2019-10-25']
df = data
df.index = pd.to_datetime(df.time)


'''函数包'''


def bucket(x):
    ls = []
    for i in range(len(x)):
        if sum(x[0:i+1]) < vbs:
            ls.append(x[i])
        else:
            res = vbs - sum(x[0:i])
            ls.append(res)
            break
    return ls


def phi(delta_p, x):
    sigma_p = np.std(delta_p)
    phi = stats.norm.cdf(delta_p[x]/sigma_p)
    return phi


def diff_dict(df_day, a, b):
    a_b = []
    temp = []
    for i in df_day.index:
        temp = [df_day.loc[i, a], df_day.loc[i, b]]
        a_b.append(temp)
    tdict = {key: value for (key, value) in a_b}
    return tdict


def round_vpin(x):
    return x.sum()/(50 * vbs)


df_day = df['2015-07-01':]
sample_day = df['2015-01-05':'2015-06-30']
day_volume = sample_day['volume'].resample('d').sum()
vbs = int(day_volume.sum()/((len(df['2015-01-05':'2015-06-30'])/240) * 50))

'''单日股票vpin计算'''


def vpin_cal(df_day):
    vpin_df = pd.DataFrame(
        columns=['time', 'delta_p', 'tb', 'v', 'Bucket', 'phi', 'vb', 'vs'])
    ol_df = pd.DataFrame(
        columns=['Bucket', 'vb_agg', 'vs_agg', 'oi', 'initial_time', 'final_time'])

    df_day['chg'] = df_day['close'] - df_day['open']
    df_day['chg'] = df_day['chg'].fillna(0)
    df_day.time = pd.to_datetime(df_day.time)
    # for i in range(df_day.shape[0]):
    #df_day.iloc[i, 0] = df_day.iloc[:i+1, 6].sum()

    v = df_day.volume.to_list()

    tbls = []
    while v != [] and v != [0]:
        if v[0] % vbs != 0:
            ls = bucket(v)
            tbls.append(ls)
            v[len(ls) - 1] = v[len(ls) - 1] - ls[-1]
            v = v[len(ls) - 1:]
        else:
            if v[0] == vbs:
                tbls.append([vbs])
                v = v[1:]
            elif v[0] == 0:
                ls = bucket(v)
                tbls.append(ls)
                v[len(ls) - 1] = v[len(ls) - 1] - ls[-1]
                v = v[len(ls) - 1:]
            else:
                tbls.append([vbs])
                v[0] = v[0] - vbs
    '''
    时间传入
    '''
    total_ls = []
    for i in tbls:
        for j in i:
            total_ls.append(j)

    vpin_df.tb = total_ls

    v = df_day.volume.to_list()
    trade_time = df_day.index.to_list()
    trade_time_vpin = []
    num_trade_time_vpin = 0

    while num_trade_time_vpin < len(vpin_df):
        if total_ls[0] == v[0]:
            trade_time_vpin.append(trade_time[0])
            total_ls.remove(total_ls[0])
            v.remove(v[0])
            trade_time.remove(trade_time[0])
            num_trade_time_vpin += 1
        else:
            for belong in range(len(total_ls)):
                if sum(total_ls[0: belong + 1]) == v[0]:
                    trade_time_vpin.append([trade_time[0]] * (belong + 1))
                    break
                else:
                    belong += 1
            for remove in range(belong + 1):
                total_ls.remove(total_ls[0])
            v.remove(v[0])
            trade_time.remove(trade_time[0])
            num_trade_time_vpin += belong + 1

    time_vpin_ls = []
    for i in trade_time_vpin:
        if type(i) == pd._libs.tslibs.timestamps.Timestamp:
            time_vpin_ls.append(i)
        else:
            for j in i:
                time_vpin_ls.append(j)

    vpin_df.time = time_vpin_ls

    '''
    通过时间传入其他量
    '''

    chg_dict = diff_dict(df_day, 'time', 'chg')

    vpin_df.index = vpin_df.time

    for i in vpin_df.index:
        vpin_df.loc[i, 'delta_p'] = chg_dict[i]

    v_ls = []
    for i in tbls:
        for j in range(len(i)):
            v_ls_i = sum(i[0:j+1])
            v_ls.append(v_ls_i)

    vpin_df.v = v_ls

    bkt_ls = []
    for i in range(len(tbls)):
        for j in range(len(tbls[i])):
            bkt = i + 1
            bkt_ls.append(bkt)

    vpin_df.Bucket = bkt_ls

    phi_ls = [phi(vpin_df.delta_p.to_list(), i) for i in range(len(vpin_df))]

    vpin_df.phi = phi_ls

    vpin_df.vb = vpin_df.tb.multiply(vpin_df.phi).astype('int')

    vpin_df.vs = vpin_df.tb - vpin_df.vb

    ol_df = pd.DataFrame(
        columns=['Bucket', 'vb_agg', 'vs_agg', 'oi', 'initial_time', 'final_time'])
    ol_df.Bucket = list(set(bkt_ls))

    vb_agg_ls = [vpin_df[vpin_df.Bucket == i].vb.sum()
                 for i in list(set(bkt_ls))]
    vs_agg_ls = [vpin_df[vpin_df.Bucket == i].vs.sum()
                 for i in list(set(bkt_ls))]
    ol_df.vb_agg, ol_df.vs_agg = vb_agg_ls, vs_agg_ls
    ol_df.oi = abs(ol_df.vb_agg - ol_df.vs_agg)

    ini_time_ls = [vpin_df[vpin_df.Bucket == i].time[0]
                   for i in list(set(bkt_ls))]
    fin_time_ls = [vpin_df[vpin_df.Bucket == i].time[-1]
                   for i in list(set(bkt_ls))]
    ol_df.initial_time, ol_df.final_time = ini_time_ls, fin_time_ls

    return vpin_df, ol_df


'''真正的计算开始咯'''

oi_df = vpin_cal(df_day)[1]

#vpin = oi_df.oi.sum()/(len(oi_df) * vbs)

vpin_ls_df = pd.DataFrame(
    columns=['ID', 'Vpin', 'Initial_Bucket', 'Final_Bucket'])
ini_bkt_ls = oi_df.Bucket[:len(oi_df)-49]
fin_bkt_ls = ini_bkt_ls + 49
vpin_ls_df.Initial_Bucket, vpin_ls_df.Final_Bucket = ini_bkt_ls, fin_bkt_ls

for i in range(len(vpin_ls_df)):
    vpin_ls_df.iloc[i, 1] = round_vpin(oi_df.iloc[i:i+50, 3])

vpin_ls_df.ID = vpin_ls_df.index + 1

vpin_cal(df_day)[0].to_csv('E:/vpin_1.csv')
oi_df.to_csv('E:/vpin_2.csv')
vpin_ls_df.to_csv('E:/vpin_3.csv')

'''对oi_df做年份统计'''
years = ['2015', '2016', '2017', '2018', '2019']


def year(x):
    oi_copy = oi_df.copy()
    oi_copy.index = oi_copy.initial_time
    oi_1 = oi_copy[x:x]
    oi_1.index = oi_1.final_time
    oi_year = oi_1[x:x]
    bucket = oi_year.Bucket[-1]
    return bucket


year_bucket = [year(i) for i in years]  # [4068, 7732, 11488, 14805, 18166]
#vpin_ls_df[0: 4068-49].Vpin.mean(), vpin_ls_df[0: 4068-49].Vpin.std()
#0.3778, 0.1320; 0.1629,0.0625;0.1292, 0.0403; 0.1645, 0.0489; 0.2012, 0.0862
'''画图'''
d = 0.01
plot_array = np.array(vpin_ls_df.Vpin, dtype=float)
num_bin = np.arange(min(plot_array), max(plot_array), d)

plt.figure(figsize=(20, 8), dpi=80)

plt.hist(plot_array, num_bin)
plt.xticks(num_bin, rotation=45)
plt.xlabel('vpin')
plt.ylabel('num')
plt.title('vpin_distribution')
plt.grid(alpha=0.01)


plt.show()
plt.hist(plot_array, cumulative=True, density=True, bins=num_bin)
plot_array.mean()  # 0.211  0.2280
plot_array.skew()  # 1.5968 1.00917
plot_array.kurt()  # 2.2966  1.05630
plot_array.std()  # 0.1235 0.06757
len(plot_array)*(plot_array.skew()**2/6 +
                 (plot_array.kurt()-3)**2/24)  # 8072.5681 5927.027

'''RV'''
df_rv = pd.DataFrame(columns=['up', 'down', 'and', 'or', 'up_down', 'rv'])
data1 = pd.read_csv('E:/vpin/vpin_1.csv')
data2 = pd.read_csv('E:/vpin/vpin_2.csv')
data3 = pd.read_csv('E:/vpin/vpin_3.csv')
data4 = pd.read_csv('E:/vpin/close.csv')
data4.set_index(data4.time, inplace=True)


def bucket_return(start, end):
    bucket_info = data4[start:end]
    ret = np.log(bucket_info.iloc[-1, 4]/bucket_info.iloc[0, 3])
    return ret


ret_ls = []
it, ft = data2.initial_time.to_list(), data2.final_time.to_list()
zipped = zip(it, ft)
for (i, j) in zipped:
    ret = bucket_return(i, j)
    ret_ls.append(ret)
ret_ls_2 = np.array(ret_ls)**2

rv = []
for i in range(len(ret_ls_2)-49):
    rv.append(np.mean(ret_ls_2[i:i+50]))
meana = pd.Series(ret_ls).mean()
sigmaa = pd.Series(ret_ls).std()
d_range = meana - 3 * sigmaa
up_range = meana + 3 * sigmaa

vpin = [[], [], [], [], [], [], [], [], [], []]
for i in range(len(data3.Vpin)):
    for j in range(10):
        if j/10 < data3.Vpin[i] < (j+1)/10:
            vpin[j].append(data3.iloc[i, 1])
        else:
            continue
df = pd.DataFrame(columns=['up', 'down', 'and', 'or', 'up_down', 'rv'], index=[
                  '(0,1)', '(1,2)', '(2,3)', '(3,4)', '(4,5)', '(5,6)', '(6,7)', '(7,8)', '(8,9)', '(9,10)'])
for row in range(len(df)):
    up, down, andls, orls, rv_mean = 0, 0, 0, 0, 0
    for i in vpin[row]:
        if [d_range] * 50 <= ret_ls[i-1: i+50] <= [up_range] * 50:
            continue
        else:
            if ret_ls[i-1: i+50] > [up_range] * 50 and ret_ls[i-1: i+50] < [d_range] * 50:
                andls += 1
            elif ret_ls[i-1: i+50] < [up_range] * 50 or ret_ls[i-1: i+50] < [d_range] * 50:
                orls += 1
            elif ret_ls[i-1: i+50] < [d_range] * 50:
                down += 1
            elif ret_ls[i-1: i+50] > [up_range] * 50:
                up += 1
        rv_mean += rv[i - 1]
    try:
        df.iloc[row, 0], df.iloc[row, 1], df.iloc[row, 2], df.iloc[row, 3], df.iloc[row, 5] = up / \
            len(vpin[row]), down/len(vpin[row]), andls/len(vpin[row]
                                                           ), orls/len(vpin[row]), rv_mean * 10 ** 3 / len(vpin[row])
    except ZeroDivisionError:
        df.iloc[row, 0], df.iloc[row, 1], df.iloc[row,
                                                  2], df.iloc[row, 3], df.iloc[row, 5] = 0, 0, 0, 0, 0
