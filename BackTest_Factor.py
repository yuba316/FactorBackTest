# -*- coding: utf-8 -*-
"""
Created on Fri May 29 20:35:08 2020

@author: yuba316
"""

import copy
import math
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import tushare as ts
pro = ts.pro_api('da949c80ceb5513dcc45b50ba0b0dec1bc518132101bec0dfb19da56')

stock = pd.read_csv(r"D:\work\back_test_system\DataBase\Stock\Stock.csv")
stock['trade_date'] = stock['trade_date'].apply(str)

#%%

def RankInvest(temp,pct,long_short,weight):
    
    df = copy.deepcopy(temp)
    df['rank'] = df['score'].rank(method='min',na_option='keep',ascending=False)
    n = len(df)
    df['invest'] = df.apply(lambda x: 1 if x['rank']<=math.floor(n*pct) else 0,axis=1)
    if long_short:
        df['invest'] = df.apply(lambda x: -1 if x['rank']>=math.ceil(n-n*pct) else x['invest'],axis=1)
    m,l = len(df[df['invest']==1]),len(df[df['invest']==-1])
    if weight:
        cumsum_m,cumsum_l = m*(1+m)/2,l*(1+l)/2
        df['weight'] = df.apply(lambda x: (m-x['rank']+1)/cumsum_m if x['invest']==1 else 0,axis=1)
        df['weight'] = df.apply(lambda x: (l-(n-x['rank']+1)+1)/cumsum_l if x['invest']==-1 else x['weight'],axis=1)
    else:
        df['weight'] = df.apply(lambda x: 1/m if x['invest']==1 else 0,axis=1)
        df['weight'] = df.apply(lambda x: 1/l if x['invest']==-1 else x['weight'],axis=1)
    
    return df[['invest','weight']]

#%%

def DailyInvest(temp,last_list,origin):
    
    df = copy.deepcopy(temp)
    df = pd.merge(df,origin,how='left',on='ts_code')
    df['close_0'].fillna(0,inplace=True)
    stock_list = list(df[df['invest']!=0]['ts_code'])
    new_stock = list(set(stock_list)-set(last_list))
    del_stock = list(set(last_list)-set(stock_list))
    last_list = stock_list
    df['type'] = df['ts_code'].apply(lambda x: 1 if x in new_stock else (2 if x in del_stock else 0))
    df['close_0'] = df.apply(lambda x: x['close'] if x['type']==1 else x['close_0'],axis=1)
    df['profit'] = df.apply(lambda x: x['close']-x['close_0'] if x['close_0']!=0 else 0,axis=1)
    df['close_0'] = df.apply(lambda x: 0 if x['type']==2 else x['close_0'],axis=1)
    origin = df[['ts_code','close_0']]
    
    return df[['profit','close_0']],last_list,origin

#%%

def DailyProfit(temp,last,capital,left,basic):
    
    df = copy.deepcopy(temp)
    df = pd.merge(df,last,how='left',on='ts_code')
    df['last_weight'].fillna(0,inplace=True)
    df['last_volume'].fillna(0,inplace=True)
    ratio = (df['last_weight']*df['last_close']).sum()/(df['weight']*df['close']).sum()
    if pd.isnull(ratio):
        ratio = 0
    df['iscover'] = (ratio*df['weight']<=df['last_weight'])&(df['last_weight']>0) # 今天需要减仓的股票
    df['volume'] = df.apply(lambda x: ratio*x['last_volume']*x['weight']/x['last_weight'] if x['iscover'] else x['last_volume'],axis=1)
    df['volume'].fillna(0,inplace=True)
    df['volume'] = df['volume'].apply(int)
    
    df['cover'] = df.apply(lambda x: x['last_volume']-x['volume'] if x['iscover'] else 0,axis=1)
    cover_cost = (df['cover']*df['close']).sum()
    available_cap = left+cover_cost
    df['buy'] = df.apply(lambda x: 0 if x['iscover'] else ratio*x['weight']-x['last_weight'],axis=1)
    new_total = (df['buy']*df['close']).sum()
    df['buy'] = df.apply(lambda x: 0 if x['buy']==0 else int(x['buy']*available_cap/new_total),axis=1)
    df['volume'] = df['volume']+df['buy']
    
    profit = (df['last_volume']*df['profit']*df['invest']).sum()
    capital = basic+profit+left
    left = available_cap-(df['buy']*df['close']).sum()
    basic = (df['volume']*df['origin_close']).sum()
    last = copy.deepcopy(df[['ts_code','weight','volume','origin_close']])
    last.rename(columns={'weight':'last_weight','volume':'last_volume','origin_close':'last_close'},inplace=True)
    
    return last,capital,left,basic

#%%

def FactorBT(df,pct=0.1,long_short=True,weight=False,capital=1000000):
    
    # input:
    # factor[DataFrame]: [trade_date,stock,close,score]记录每天每只股票的因子得分
    # pct[float]: 选择进行投资的组别比例
    # long_short[bool]: 是否计算多空组合收益
    # weight[bool]: 是否按排名分权重进行投资
    
    factor = copy.deepcopy(df)
    factor['invest'],factor['weight'] = 0,0
    trade_date = list(np.sort(factor['trade_date'].unique()))
    n = len(trade_date)
    for i in range(n):
        temp = factor[factor['trade_date']==trade_date[i]]
        df = RankInvest(temp,pct,long_short,weight)
        index = factor[factor['trade_date']==trade_date[i]].index
        factor.loc[index,'invest'],factor.loc[index,'weight'] = list(df['invest']),list(df['weight'])
    factor.drop(['score'],axis=1,inplace=True)
    
    factor['profit'],factor['origin_close'] = 0,0
    origin = copy.deepcopy(factor[factor['trade_date']==trade_date[0]])
    origin['close_0'] = origin.apply(lambda x: x['close'] if x['invest']!=0 else 0,axis=1)
    origin = origin[['ts_code','close_0']]
    last_list = []
    for i in range(n):
        temp = factor[factor['trade_date']==trade_date[i]]
        df,last_list,origin = DailyInvest(temp,last_list,origin)
        index = factor[factor['trade_date']==trade_date[i]].index
        factor.loc[index,'profit'],factor.loc[index,'origin_close'] = list(df['profit']),list(df['close_0'])
    
    strategy,base,save,factor['volume'] = [],[],[],0
    temp = copy.deepcopy(factor[factor['trade_date']==trade_date[0]])
    total = (temp['weight']*temp['close']).sum()
    temp['volume'] = (capital/total*temp['weight']).apply(int)
    last = copy.deepcopy(temp[['ts_code','weight','volume','origin_close']])
    last.rename(columns={'weight':'last_weight','volume':'last_volume','origin_close':'last_close'},inplace=True)
    strategy.append(capital)
    left = capital-(temp['volume']*temp['close']).sum()
    save.append(left)
    basic = capital-left
    base.append(basic)
    for i in range(1,n,1):
        temp = factor[factor['trade_date']==trade_date[i]]
        last,capital,left,basic = DailyProfit(temp,last,capital,left,basic)
        strategy.append(capital)
        save.append(left)
        base.append(basic)
        if capital<=0:
            break
    
    n = len(strategy)
    result = pd.DataFrame({'trade_date':trade_date[:n],'strategy':strategy,'capital_left':save,'capital_base':base})
    result['trade_date'] = result['trade_date'].apply(lambda x: datetime.datetime.strptime(x,'%Y%m%d'))
    result['strategy_pct'] = result['strategy']/strategy[0]-1
    
    return result

#%% test

factor = stock[['trade_date','ts_code','close','pre_close']]
del stock
factor['score'] = factor['close']/factor['pre_close']-1
factor = factor[factor['trade_date']>='20200101']
factor.reset_index(drop=True,inplace=True)

#%%

result = FactorBT(factor,0.1,True,True)
index = pro.index_daily(ts_code='399300.SZ',start_date='20200101', end_date='20200525',fields='trade_date,close')
index.sort_values(by='trade_date',inplace=True)
index.reset_index(drop=True,inplace=True)
index['trade_date'] = index['trade_date'].apply(lambda x: datetime.datetime.strptime(x,'%Y%m%d'))
index['index_pct'] = index['close']/index['close'].iloc[0]-1
plt.figure(figsize=(12,4))
plt.title('回测结果')
plt.plot(result['trade_date'],result['strategy_pct'],label='因子选股策略')
plt.plot(index['trade_date'],index['index_pct'],label='沪深300指数')
plt.xticks(rotation=60)
plt.legend(loc='upper left')

#%%
'''
factor = stock[stock['trade_date']>='20120101'][['trade_date','ts_code','close']]
del stock
factor['last_month_close'] = factor.groupby(['ts_code'])['close'].apply(lambda x: x.shift(21))
factor.dropna(axis=0,inplace=True)
factor['score'] = factor['close']/factor['last_month_close']-1
factor.reset_index(drop=True,inplace=True)
factor['month'] = factor['trade_date'].apply(lambda x: int(int(x)/100))
month = list(factor.groupby(['month']).first()['trade_date'])
factor = factor[factor['trade_date'].apply(lambda x: x in month)]
factor.reset_index(drop=True,inplace=True)
factor.drop(['last_month_close','month'],axis=1,inplace=True)
result = FactorBT(factor)
plt.plot(result['trade_date'],result['strategy_pct'])
'''