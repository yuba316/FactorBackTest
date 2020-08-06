# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 15:59:46 2020

@author: yuba316
"""

import copy
import math
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from pyfinance.ols import PandasRollingOLS
'''
stock = pd.read_csv(r"D:\work\back_test_system\DataBase\Stock\Stock.csv")
stock['trade_date'] = stock['trade_date'].apply(str)
factor = stock[stock['trade_date']>='20200101']
'''
#%%

def plus(factor,x,y,name):
    if type(y)==int:
        factor[name]=factor[x]+y
    else:
        factor[name]=factor[x]+factor[y]
    return factor

def minus(factor,x,y,name):
    if type(y)==int:
        factor[name]=factor[x]-y
    else:
        factor[name]=factor[x]-factor[y]
    return factor

def multiple(factor,x,y,name):
    if type(y)==int:
        factor[name]=factor[x]*y
    else:
        factor[name]=factor[x]*factor[y]
    return factor

def divide(factor,x,y,name):
    if type(y)==int:
        factor[name]=factor[x]/y
    else:
        factor[name]=factor[x]/factor[y]
    return factor

def log(factor,x,name,num=None):
    if num!=None:
        factor[name]=factor[x].apply(lambda x: math.log(x,num))
    else:
        factor[name]=np.log(factor[x])
    return factor

def exp(factor,x,name,num=None):
    if num!=None:
        factor[name]=np.power(num,factor[x])
    else:
        factor[name]=np.exp(factor[x])
    return factor

def sqrt(factor,x,name,num=None):
    if num!=None:
        factor[name]=np.power(factor[x],1/num)
    else:
        factor[name]=np.sqrt(factor[x])
    return factor

def square(factor,x,name,num=None):
    if num!=None:
        factor[name]=np.power(factor[x],num)
    else:
        factor[name]=np.square(factor[x])
    return factor

def reci(factor,x,name):
    factor[name]=np.power(factor[x],-1)
    return factor

def oppo(factor,x,name):
    factor[name]=-1*factor[x]
    return factor

def absl(factor,x,name):
    factor[name]=abs(factor[x])
    return factor

#%%

def TsMax(factor,x,num,name):
    temp = factor.groupby('ts_code')[x].rolling(num).max()
    temp.index = temp.index.droplevel()
    factor[name] = temp
    return factor

def TsMin(factor,x,num,name):
    temp = factor.groupby('ts_code')[x].rolling(num).min()
    temp.index = temp.index.droplevel()
    factor[name] = temp
    return factor

def TsMid(factor,x,num,name):
    temp = factor.groupby('ts_code')[x].rolling(num).median()
    temp.index = temp.index.droplevel()
    factor[name] = temp
    return factor

def SMA(factor,x,num,name):
    temp = factor.groupby('ts_code')[x].rolling(num).mean()
    temp.index = temp.index.droplevel()
    factor[name] = temp
    return factor

def WMA(factor,x,num,name):
    day = np.arange(1,num+1,1)
    temp = factor.groupby('ts_code')[x].rolling(num).apply(lambda x: (x*day/day.sum()).sum(),raw=True)
    temp.index = temp.index.droplevel()
    factor[name] = temp
    return factor

def std(factor,x,num,name):
    temp = factor.groupby('ts_code')[x].rolling(num).std()
    temp.index = temp.index.droplevel()
    factor[name] = temp
    return factor

def skew(factor,x,num,name):
    temp = factor.groupby('ts_code')[x].rolling(num).skew()
    temp.index = temp.index.droplevel()
    factor[name] = temp
    return factor

def kurt(factor,x,num,name):
    temp = factor.groupby('ts_code')[x].rolling(num).kurt()
    temp.index = temp.index.droplevel()
    factor[name] = temp
    return factor

def norm(factor,x,num,name):
    gb = factor.groupby('ts_code')[x].rolling(num)
    mean,std = gb.mean(),gb.std()
    mean.index,std.index = mean.index.droplevel(),std.index.droplevel()
    temp = (factor[x]-mean)/std
    factor[name] = temp
    return factor

def normMaxMin(factor,x,num,name):
    gb = factor.groupby('ts_code')[x].rolling(num)
    Min,Max = gb.min(),gb.max()
    Min.index,Max.index = Min.index.droplevel(),Max.index.droplevel()
    temp = (factor[x]-Min)/(Max-Min)
    factor[name] = temp
    return factor

def TsRank(factor,x,num,name):
    temp = factor.groupby('ts_code')[x].rolling(num).apply(lambda x: rankdata(x)[-1],raw=True)
    temp.index = temp.index.droplevel()
    factor[name] = temp
    return factor

def TsToMax(factor,x,num,name):
    temp = factor.groupby('ts_code')[x].rolling(num).apply(lambda x: num-np.argmax(x)-1,raw=True)
    temp.index = temp.index.droplevel()
    factor[name] = temp
    return factor

def TsToMin(factor,x,num,name):
    temp = factor.groupby('ts_code')[x].rolling(num).apply(lambda x: num-np.argmin(x)-1,raw=True)
    temp.index = temp.index.droplevel()
    factor[name] = temp
    return factor

def Corr(factor,x,y,num,name):
    temp = factor.groupby('ts_code')[[x,y]].rolling(num).corr()
    temp.index = temp.index.droplevel()
    temp.index.names = ['index','key']
    temp = temp.query('key==\''+x+'\'')[y]
    temp.index = temp.index.droplevel('key')
    factor[name] = temp
    return factor

def Cov(factor,x,y,num,name):
    temp = factor.groupby('ts_code')[[x,y]].rolling(num).cov()
    temp.index = temp.index.droplevel()
    temp.index.names = ['index','key']
    temp = temp.query('key==\''+x+'\'')[y]
    temp.index = temp.index.droplevel('key')
    factor[name] = temp
    return factor

def Sum(factor,x,num,name):
    temp = factor.groupby('ts_code')[x].rolling(num).sum()
    temp.index = temp.index.droplevel()
    factor[name] = temp
    return factor

def Prod(factor,x,num,name):
    temp = factor.groupby('ts_code')[x].rolling(num).apply(lambda x: x.prod(),raw=True)
    temp.index = temp.index.droplevel()
    factor[name] = temp
    return factor

def delay(factor,x,num,name):
    temp = factor.groupby('ts_code')[x].shift(num)
    factor[name] = temp
    return factor

def delta(factor,x,num,name):
    delay = factor.groupby('ts_code')[x].shift(num)
    temp = factor[x]-delay
    factor[name] = temp
    return factor

def delta_pct(factor,x,num,name):
    delay = factor.groupby('ts_code')[x].shift(num)
    temp = (factor[x]-delay)/delay
    factor[name] = temp
    return factor

def RegAlpha(factor,x,y,num,name):
    temp = copy.deepcopy(factor[['trade_date','ts_code',x,y]])
    temp.sort_values(by=['ts_code','trade_date'],inplace=True)
    res = PandasRollingOLS(temp[x],temp[y],num)
    factor[name] = res.alpha
    index = factor.groupby('ts_code').head(num-1).index
    factor.loc[index,name] = np.nan
    return factor

def RegBeta(factor,x,y,num,name):
    temp = copy.deepcopy(factor[['trade_date','ts_code',x,y]])
    temp.sort_values(by=['ts_code','trade_date'],inplace=True)
    res = PandasRollingOLS(temp[x],temp[y],num)
    factor[name] = res.beta
    index = factor.groupby('ts_code').head(num-1).index
    factor.loc[index,name] = np.nan
    return factor

def RegResi(factor,x,y,num,name):
    temp = copy.deepcopy(factor[['trade_date','ts_code',x,y]])
    temp.sort_values(by=['ts_code','trade_date'],inplace=True)
    res = PandasRollingOLS(temp[x],temp[y],num)
    temp['alpha'],temp['beta'] = res.beta,res.beta
    temp['resi'] = temp[y]-temp['alpha']-temp[x]*temp['beta']
    factor[name] = temp['resi']
    index = factor.groupby('ts_code').head(num-1).index
    factor.loc[index,name] = np.nan
    return factor

def compareMax(factor,x,y,name):
    factor[name] = factor[[x,y]].max(axis=1)
    return factor

def compareMin(factor,x,y,name):
    factor[name] = factor[[x,y]].min(axis=1)
    return factor

def compareIf(factor,z,x,y,name):
    factor[name] = factor[x]*(factor[z]>0)+factor[y]*(factor[z]<=0)
    return factor

#%%

def SecRank(factor,x,name):
    temp = factor.groupby('trade_date')[x].rank(method='min',na_option='keep',ascending=True)
    factor[name] = temp
    return factor

def SecNorm(factor,x,name):
    gb = factor.groupby('trade_date')[x]
    temp = gb.apply(lambda x: (x-x.mean())/x.std())
    factor[name] = temp
    return factor

def SecNormMaxMin(factor,x,name):
    gb = factor.groupby('trade_date')[x]
    temp = gb.apply(lambda x: (x-x.min())/(x.max()-x.min()))
    factor[name] = temp
    return factor

def SecOne(factor,x,name):
    gb = factor.groupby('trade_date')[x]
    temp = gb.apply(lambda x: x/x.sum())
    factor[name] = temp
    return factor

def SecDeMean(factor,x,name):
    gb = factor.groupby('trade_date')[x]
    temp = gb.apply(lambda x: x-x.mean())
    factor[name] = temp
    return factor