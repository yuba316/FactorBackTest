# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 11:01:17 2020

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
factor = stock[stock['trade_date']>='20200101'][['trade_date','ts_code','close','pre_close']]
factor['score'] = factor['close']/factor['pre_close']-1

#%%

def FactorBT(df,freq=5,pct=0.1,long_short=True,weight=False,capital=1000000):
    
    stock = copy.deepcopy(df)
    trade_date = list(np.sort(stock['trade_date'].unique()))
    n = len(trade_date)
    trade_date = [trade_date[i] for i in range(0,n,freq)]
    factor = copy.deepcopy(stock[stock['trade_date'].apply(lambda x: x in trade_date)])
    
    factor['rank'] = factor.groupby('trade_date')['score'].rank(method='min',na_option='keep',ascending=False)
    factor['invest'] = factor.groupby('trade_date')['rank'].apply(lambda x: x<=math.floor(x.count()*pct)).apply(int)
    if long_short:
        factor['invest'] = factor['invest']-factor.groupby('trade_date')['rank'].apply(lambda x: x>=math.ceil(x.count()*(1-pct))).apply(int)
    if weight:
        factor['weight'] = factor[factor['invest']==1].groupby('trade_date')['rank'].apply(lambda x: (x.count()-x+1)/x.sum())
        factor['weight'].fillna(0,inplace=True)
        if long_short:
            factor['weight_1'] = factor[factor['invest']==-1].groupby('trade_date')['rank'].apply(lambda x: (x.count()-(x.max()-x))/((1+x.count())*x.count()/2))
            factor['weight_1'].fillna(0,inplace=True)
            factor['weight'] = factor['weight']+factor['weight_1']
            factor.drop(['weight_1'],axis=1,inplace=True)
    else:
        factor['weight'] = factor[factor['invest']==1].groupby('trade_date')['invest'].apply(lambda x: x/x.count())
        factor['weight'].fillna(0,inplace=True)
        if long_short:
            factor['weight_1'] = factor[factor['invest']==-1].groupby('trade_date')['invest'].apply(lambda x: -1*x/x.count())
            factor['weight_1'].fillna(0,inplace=True)
            factor['weight'] = factor['weight']+factor['weight_1']
            factor.drop(['weight_1'],axis=1,inplace=True)
    
    Profit,Capital,Volume = [0],[capital],pd.Series([])
    factor['next_close'] = factor.groupby('ts_code')['close'].shift(-1)
    for i in trade_date:
        temp = copy.deepcopy(factor[factor['trade_date']==i])
        temp['volume'] = ((Capital[-1]*temp['weight'])/(temp['weight']*temp['close']).sum()).apply(int)
        Profit.append((temp['invest']*(temp['next_close']-temp['close'])*temp['volume']).sum())
        Capital.append(Capital[-1]+Profit[-1])
        Volume = Volume.append(temp['volume'])
        if Capital[-1]<=0:
            break
    factor['origin_close'],factor['volume'] = factor['close'],Volume
    factor['volume'].fillna(0,inplace=True)
    n = min(len(Capital),len(trade_date))
    result = pd.DataFrame({'trade_date':trade_date[:n],'profit':Profit[:n],'capital':Capital[:n]})
    
    stock[['invest','origin_close','volume']] = factor[['invest','origin_close','volume']]
    stock[['invest','origin_close','volume']] = stock.groupby('ts_code')[['invest','origin_close','volume']].fillna(method='ffill')
    stock['profit'] = stock['invest']*(stock['close']-stock['origin_close'])*stock['volume']
    strategy = stock.groupby('trade_date')['profit'].sum()
    strategy = pd.DataFrame({'trade_date':strategy.index,'profit':list(strategy)})
    result = pd.merge(strategy,result,how='left',on='trade_date')
    result['profit_y'].fillna(0,inplace=True)
    result['profit'] = result['profit_x']+result['profit_y']
    result['capital'].fillna(method='ffill',inplace=True)
    result['strategy'] = result['capital']+result['profit_x']
    result.drop(['profit_x','profit_y'],axis=1,inplace=True)
    result['strategy_pct'] = result['strategy']/result['strategy'].iloc[0]-1
    result['trade_date'] = result['trade_date'].apply(lambda x: datetime.datetime.strptime(x,'%Y%m%d'))
    
    return result

#%%

result = FactorBT(factor,5,0.1,False,False)
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