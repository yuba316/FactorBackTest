# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 09:35:29 2020

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
factor = stock[['trade_date','ts_code','close','pre_close']]
del stock
factor['score'] = factor['close']/factor['pre_close']-1
factor.drop(['pre_close'],axis=1,inplace=True)
factor.reset_index(drop=True,inplace=True)

#%%

def FactorTest(df,freq=5,pct=0.1):
    
    factor = copy.deepcopy(df)
    trade_date = list(np.sort(factor['trade_date'].unique()))
    n = len(trade_date)
    trade_date = [trade_date[i] for i in range(0,n,5)]
    factor = factor[factor['trade_date'].apply(lambda x: x in trade_date)]
    
    factor['next_close'] = factor.groupby('ts_code')['close'].shift(-1)
    factor['profit'] = factor['next_close']/factor['close']-1
    factor['rank'] = factor.groupby('trade_date')['score'].rank(method='min',na_option='keep',ascending=True)
    factor['p_rank'] = factor.groupby('trade_date')['profit'].rank(method='min',na_option='keep',ascending=True)
    factor.dropna(inplace=True)
    
    result = {}
    temp = factor.groupby('trade_date')[['score','profit']].corr()
    temp.index.names = ['trade_date','key']
    result['IC'] = list(temp.query('key==\'profit\'')['score'])
    temp = factor.groupby('trade_date')[['rank','p_rank']].corr()
    temp.index.names = ['trade_date','key']
    result['Rank_IC'] = list(temp.query('key==\'p_rank\'')['rank'])
    
    temp = factor.groupby('trade_date')['ts_code'].count()
    temp = pd.DataFrame({'trade_date':temp.index,'group':list(temp)})
    factor = pd.merge(factor,temp,how='left',on='trade_date')
    factor['long'] = factor['rank']>(1-pct)*factor['group']
    factor['short'] = factor['rank']<pct*factor['group']
    temp = factor[factor['long']].groupby('trade_date')[['score','profit']].corr()
    temp.index.names = ['trade_date','key']
    result['long_IC'] = list(temp.query('key==\'profit\'')['score'])
    temp = factor[factor['long']].groupby('trade_date')[['rank','p_rank']].corr()
    temp.index.names = ['trade_date','key']
    result['long_Rank_IC'] = list(temp.query('key==\'p_rank\'')['rank'])
    result['long_short_profit'] = list(factor[factor['long']].groupby('trade_date')['profit'].mean()-\
                                       factor[factor['short']].groupby('trade_date')['profit'].mean())
    
    result = pd.DataFrame(result)
    res = result.mean()
    n = len(result)
    res['IR'] = result['IC'].mean()/result['IC'].std()*np.sqrt(252/n)
    res['Rank_IR'] = result['Rank_IC'].mean()/result['Rank_IC'].std()*np.sqrt(252/n)
    res['long_IR'] = result['long_IC'].mean()/result['long_IC'].std()*np.sqrt(252/n)
    res['long_Rank_IR'] = result['long_Rank_IC'].mean()/result['long_Rank_IC'].std()*np.sqrt(252/n)
    
    return res

#%% test

result = FactorTest(factor,5,0.1)