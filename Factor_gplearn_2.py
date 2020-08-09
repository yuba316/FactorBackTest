# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 10:30:11 2020

@author: yuba316
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from pyfinance.ols import PandasRollingOLS
import datetime

import graphviz
import pickle
from gplearn import genetic
from gplearn.functions import make_function
from gplearn.genetic import SymbolicTransformer, SymbolicRegressor
from gplearn.fitness import make_fitness
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split

import jqdatasdk as jq
jq.auth('18903041915', 'iamaman369')

#%% 1. 数据读入

stock_list = jq.get_index_stocks('000300.XSHG')
start_date = '2020-01-01'
end_date = '2020-06-30'
fields = ['open', 'close', 'low', 'high', 'volume', 'avg', 'pre_close']
stock_price = jq.get_price(stock_list, start_date=start_date, end_date=end_date, fq=None, fields=fields)
stock_price = stock_price.sort_values(by=['code','time']).reset_index(drop=True)
stock_price['time'] = stock_price['time'].apply(lambda x: datetime.datetime.strftime(x,'%Y%m%d'))
stock_price['profit'] = stock_price.groupby('code')['close'].shift(-20)/stock_price['close']-1
stock_price['5'],stock_price['10'],stock_price['15'],stock_price['20'],stock_price['30'],\
    stock_price['60'],stock_price['100'],stock_price['120'],stock_price['200'] = 5,10,15,20,30,60,100,120,200
fields = fields+['5','10','15','20','30','60','100','120','200']
'''
trade_date_list = jq.get_trade_days(start_date, end_date)
trade_date_list = [datetime.datetime.strftime(i,'%Y%m%d') for i in trade_date_list]

def SpreadQuote(df,quote,trade_date_list,stock_list):
    new_df = pd.DataFrame({'time':trade_date_list})
    for i in stock_list:
        temp_df = df[df['code']==i][['time',quote]]
        new_df = pd.merge(new_df,temp_df,how='left',on='time')
        new_df.rename(columns={quote:i},inplace=True)
    return new_df

stock_quote = {}
for i in fields:
    stock_quote[i] = SpreadQuote(stock_price, i, trade_date_list, stock_list)
'''
#%% 2. 算子自定义

# 发现gplearn支持fit的数据集只能是二维的array-like矩阵，所以还是只能用stock_price再结合groupby去自定义函数
# 但是由于输入的array本身没有groupby的功能，所以要在函数里自行添加

stock_price.fillna(0,inplace=True)
trade_date,stock_code,target = stock_price['time'],stock_price['code'],stock_price['profit'].values
stock_price.drop(['time','code','profit'],axis=1,inplace=True)
stock_price = stock_price.values

init_function = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min', 'sin', 'cos', 'tan']

def _exp(data): # 指数运算
    return np.exp(data)

def _square(data): # 平方运算
    return np.square(data)

def _ts_max(data,window): # 历史滚动最大
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(data))
    df = pd.DataFrame({'0':data})
    df['time'] = trade_date
    df['code'] = stock_code
    value = df.groupby('code')['0'].rolling(window).max()
    
    return np.nan_to_num(value.values)

def _ts_min(data,window): # 历史滚动最小
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(data))
    df = pd.DataFrame({'0':data})
    df['time'] = trade_date
    df['code'] = stock_code
    value = df.groupby('code')['0'].rolling(window).min()
    
    return np.nan_to_num(value.values)

def _ts_mid(data,window): # 历史滚动中位数
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(data))
    df = pd.DataFrame({'0':data})
    df['time'] = trade_date
    df['code'] = stock_code
    value = df.groupby('code')['0'].rolling(window).median()
    
    return np.nan_to_num(value.values)

def _ts_mean(data,window): # 历史滚动平均
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(data))
    df = pd.DataFrame({'0':data})
    df['time'] = trade_date
    df['code'] = stock_code
    value = df.groupby('code')['0'].rolling(window).mean()
    
    return np.nan_to_num(value.values)

def _ts_wma(data,window): # 历史滚动加权平均
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(data))
    df = pd.DataFrame({'0':data})
    df['time'] = trade_date
    df['code'] = stock_code
    day = np.arange(1,window+1,1)
    value = df.groupby('code')['0'].rolling(window).apply(lambda x: (x*day/day.sum()).sum(),raw=True)
    
    return np.nan_to_num(value.values)

def _ts_std(data,window): # 历史滚动标准差
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(data))
    df = pd.DataFrame({'0':data})
    df['time'] = trade_date
    df['code'] = stock_code
    value = df.groupby('code')['0'].rolling(window).std()
    
    return np.nan_to_num(value.values)

def _ts_skew(data,window): # 历史滚动偏度
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(data))
    df = pd.DataFrame({'0':data})
    df['time'] = trade_date
    df['code'] = stock_code
    value = df.groupby('code')['0'].rolling(window).skew()
    
    return np.nan_to_num(value.values)

def _ts_kurt(data,window): # 历史滚动峰度
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(data))
    df = pd.DataFrame({'0':data})
    df['time'] = trade_date
    df['code'] = stock_code
    value = df.groupby('code')['0'].rolling(window).kurt()
    
    return np.nan_to_num(value.values)

def _ts_norm(data,window): # 历史滚动标准化
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(data))
    df = pd.DataFrame({'0':data})
    df['time'] = trade_date
    df['code'] = stock_code
    gb = df.groupby('code')['0'].rolling(window)
    mean,std = gb.mean().values,gb.std().values
    value = (df['0'].values-mean)/std
    
    return np.nan_to_num(value.values)

def _ts_normMaxMin(data,window): # 历史滚动极值标准化
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(data))
    df = pd.DataFrame({'0':data})
    df['time'] = trade_date
    df['code'] = stock_code
    gb = df.groupby('code')['0'].rolling(window)
    Min,Max = gb.min().values,gb.max().values
    value = (df['0'].values-Min)/(Max-Min)
    
    return np.nan_to_num(value.values)

def _ts_rank(data,window): # 历史滚动排名
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(data))
    df = pd.DataFrame({'0':data})
    df['time'] = trade_date
    df['code'] = stock_code
    value = df.groupby('code')['0'].rolling(window).apply(lambda x: rankdata(x)[-1],raw=True)
    
    return np.nan_to_num(value.values)

def _ts_argmax(data,window): # 历史滚动最大值距离
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(data))
    df = pd.DataFrame({'0':data})
    df['time'] = trade_date
    df['code'] = stock_code
    value = df.groupby('code')['0'].rolling(window).apply(lambda x: window-np.argmax(x)-1,raw=True)
    
    return np.nan_to_num(value.values)

def _ts_argmin(data,window): # 历史滚动最小值距离
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(data))
    df = pd.DataFrame({'0':data})
    df['time'] = trade_date
    df['code'] = stock_code
    value = df.groupby('code')['0'].rolling(window).apply(lambda x: window-np.argmin(x)-1,raw=True)
    
    return np.nan_to_num(value.values)

def _ts_corr(df1,df2,window): # 历史滚动相关系数
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(df1))
    df = pd.DataFrame({'0':df1,'1':df2})
    df['time'] = trade_date
    df['code'] = stock_code
    value = df.groupby('code')[['0','1']].rolling(window).corr()
    value.index.names = ['code','index','key']
    value = value.query('key==\'0\'')['1']
    
    return np.nan_to_num(value.values)

def _ts_cov(df1,df2,window): # 历史滚动协方差
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(df1))
    df = pd.DataFrame({'0':df1,'1':df2})
    df['time'] = trade_date
    df['code'] = stock_code
    value = df.groupby('code')[['0','1']].rolling(window).cov()
    value.index.names = ['code','index','key']
    value = value.query('key==\'0\'')['1']
    
    return np.nan_to_num(value.values)

def _ts_sum(data,window): # 历史滚动求和
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(data))
    df = pd.DataFrame({'0':data})
    df['time'] = trade_date
    df['code'] = stock_code
    value = df.groupby('code')['0'].rolling(window).sum()
    
    return np.nan_to_num(value.values)

def _ts_prod(data,window): # 历史滚动累乘
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(data))
    df = pd.DataFrame({'0':data})
    df['time'] = trade_date
    df['code'] = stock_code
    value = df.groupby('code')['0'].rolling(window).apply(lambda x: x.prod(),raw=True)
    
    return np.nan_to_num(value.values)

def _ts_delay(data,window): # 滞后
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(data))
    df = pd.DataFrame({'0':data})
    df['time'] = trade_date
    df['code'] = stock_code
    value = df.groupby('code')['0'].shift(window)
    
    return np.nan_to_num(value.values)

def _ts_delta(data,window): # 滞后差值
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(data))
    df = pd.DataFrame({'0':data})
    df['time'] = trade_date
    df['code'] = stock_code
    delay = df.groupby('code')['0'].shift(window).values
    value = df['0'].values-delay
    
    return np.nan_to_num(value.values)

def _ts_deltaPct(data,window): # 滞后差值比例
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(data))
    df = pd.DataFrame({'0':data})
    df['time'] = trade_date
    df['code'] = stock_code
    delay = df.groupby('code')['0'].shift(window).values
    value = (df['0'].values-delay)/delay
    
    return np.nan_to_num(value.values)

def _ts_alpha(df1,df2,window): # 历史滚动回归常数
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(df1))
    df = pd.DataFrame({'0':df1,'1':df2})
    df['time'] = trade_date
    df['code'] = stock_code
    res = PandasRollingOLS(df['0'],df['1'],window)
    df['res'] = res.alpha
    index = df.groupby('code').head(window-1).index
    df.loc[index,'res'] = np.nan
    value = df['res']
    
    return np.nan_to_num(value.values)

def _ts_beta(df1,df2,window): # 历史滚动回归系数
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(df1))
    df = pd.DataFrame({'0':df1,'1':df2})
    df['time'] = trade_date
    df['code'] = stock_code
    res = PandasRollingOLS(df['0'],df['1'],window)
    df['res'] = res.beta
    index = df.groupby('code').head(window-1).index
    df.loc[index,'res'] = np.nan
    value = df['res']
    
    return np.nan_to_num(value.values)

def _ts_resi(df1,df2,window): # 历史滚动回归残差
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(df1))
    df = pd.DataFrame({'0':df1,'1':df2})
    df['time'] = trade_date
    df['code'] = stock_code
    res = PandasRollingOLS(df['0'],df['1'],window)
    df['alpha'] = res.alpha
    df['beta'] = res.beta
    df['resi'] = df['1']-df['alpha']-df['beta']*df['0']
    index = df.groupby('code').head(window-1).index
    df.loc[index,'resi'] = np.nan
    value = df['resi']
    
    return np.nan_to_num(value.values)

def _sec_max(df1,df2): # 两值取大
    df = pd.DataFrame({'0':df1,'1':df2})
    df['time'] = trade_date
    df['code'] = stock_code
    value = df[['0','1']].max(axis=1)
    
    return np.nan_to_num(value.values)

def _sec_min(df1,df2): # 两值取小
    df = pd.DataFrame({'0':df1,'1':df2})
    df['time'] = trade_date
    df['code'] = stock_code
    value = df[['0','1']].min(axis=1)
    
    return np.nan_to_num(value.values)

def _sec_compareIf(df1,df2,df3): # df2 if df1>0 else df3
    df = pd.DataFrame({'0':df1,'1':df2,'2':df3})
    df['time'] = trade_date
    df['code'] = stock_code
    value = df['1']*(df['0']>0)+df['2']*(df['0']<=0)
    
    return np.nan_to_num(value.values)

def _sec_rank(df1): # 截面排名
    df = pd.DataFrame({'0':df1})
    df['time'] = trade_date
    df['code'] = stock_code
    value = df.groupby('time')['0'].rank(method='min',na_option='keep',ascending=True)
    
    return np.nan_to_num(value.values)

def _sec_norm(df1): # 截面标准化
    df = pd.DataFrame({'0':df1})
    df['time'] = trade_date
    df['code'] = stock_code
    gb = df.groupby('time')['0']
    value = gb.apply(lambda x: (x-x.mean())/x.std())
    
    return np.nan_to_num(value.values)

def _sec_normMaxMin(df1): # 截面极值标准化
    df = pd.DataFrame({'0':df1})
    df['time'] = trade_date
    df['code'] = stock_code
    gb = df.groupby('time')['0']
    value = gb.apply(lambda x: (x-x.min())/(x.max()-x.min()))
    
    return np.nan_to_num(value.values)

def _sec_one(df1): # 截面归一化
    df = pd.DataFrame({'0':df1})
    df['time'] = trade_date
    df['code'] = stock_code
    gb = df.groupby('time')['0']
    value = gb.apply(lambda x: x/x.sum())
    
    return np.nan_to_num(value.values)

def _sec_demean(df1): # 截面去均值
    df = pd.DataFrame({'0':df1})
    df['time'] = trade_date
    df['code'] = stock_code
    gb = df.groupby('time')['0']
    value = gb.apply(lambda x: x-x.mean())
    
    return np.nan_to_num(value.values)

exp = make_function(function=_exp, name='exp', arity=1)
square = make_function(function=_square, name='square', arity=1)
ts_max = make_function(function=_ts_max, name='ts_max', arity=2)
ts_min = make_function(function=_ts_min, name='ts_min', arity=2)
ts_mid = make_function(function=_ts_mid, name='ts_mid', arity=2)
ts_mean = make_function(function=_ts_mean, name='ts_mean', arity=2)
ts_wma = make_function(function=_ts_wma, name='ts_wma', arity=2)
ts_std = make_function(function=_ts_std, name='ts_std', arity=2)
ts_skew = make_function(function=_ts_skew, name='ts_skew', arity=2)
ts_kurt = make_function(function=_ts_kurt, name='ts_kurt', arity=2)
ts_norm = make_function(function=_ts_norm, name='tsnorm', arity=2)
ts_normMaxMin = make_function(function=_ts_normMaxMin, name='ts_normMaxMin', arity=2)
ts_rank = make_function(function=_ts_rank, name='ts_rank', arity=2)
ts_argmax = make_function(function=_ts_argmax, name='ts_argmax', arity=2)
ts_argmin = make_function(function=_ts_argmin, name='ts_argmin', arity=2)
ts_corr = make_function(function=_ts_corr, name='ts_corr', arity=3)
ts_cov = make_function(function=_ts_cov, name='ts_cov', arity=3)
ts_sum = make_function(function=_ts_sum, name='ts_sum', arity=2)
ts_prod = make_function(function=_ts_prod, name='ts_prod', arity=2)
ts_delay = make_function(function=_ts_delay, name='ts_delay', arity=2)
ts_delta = make_function(function=_ts_delta, name='ts_delta', arity=2)
ts_delta_pct = make_function(function=_ts_deltaPct, name='ts_delta_pct', arity=2)
ts_alpha = make_function(function=_ts_alpha, name='ts_alpha', arity=3)
ts_beta = make_function(function=_ts_beta, name='ts_beta', arity=3)
ts_resi = make_function(function=_ts_resi, name='ts_resi', arity=3)
sec_max = make_function(function=_sec_max, name='sec_max', arity=2)
sec_min = make_function(function=_sec_min, name='sec_min', arity=2)
sec_compareIf = make_function(function=_sec_compareIf, name='sec_compareIf', arity=3)
sec_rank = make_function(function=_sec_rank, name='sec_rank', arity=1)
sec_norm = make_function(function=_sec_norm, name='sec_norm', arity=1)
sec_normMaxMin = make_function(function=_sec_normMaxMin, name='sec_normMaxMin', arity=1)
sec_one = make_function(function=_sec_one, name='sec_one', arity=1)
sec_demean = make_function(function=_sec_demean, name='sec_demean', arity=1)

user_function = [exp, square, ts_max, ts_min, ts_mid, ts_mean, ts_wma, ts_std, ts_skew, \
                 ts_kurt, ts_norm, ts_normMaxMin, ts_rank, ts_argmax, ts_argmin, ts_corr, \
                 ts_cov, ts_sum, ts_prod, ts_delay, ts_delta, ts_delta_pct, ts_alpha, \
                 ts_beta, ts_resi, sec_max, sec_min, sec_compareIf, sec_rank, sec_norm, \
                 sec_normMaxMin, sec_one, sec_demean]

#%% 3. 适应度定义

# 事实上y_pred反映的就是个股的因子得分情况，将其排名与y（收益率）做相关系数的计算就可以得到RankIC值
def _rankIC_metric(y,y_pred,sample_weight):
    df = pd.DataFrame({'profit':y,'score':y_pred})
    df['time'] = trade_date
    df['code'] = stock_code
    df['rank'] = df.groupby('time')['score'].rank(method='min',na_option='keep',ascending=True)
    df['p_rank'] = df.groupby('time')['profit'].rank(method='min',na_option='keep',ascending=True)
    res = df.groupby('time')[['rank','p_rank']].corr()
    res.index.names = ['time','key']
    res = np.mean(res.query('key==\'p_rank\'')['rank'])
    
    return res
rankIC_metric = make_fitness(function=_rankIC_metric, greater_is_better=True)

#%% 4. 开始遗传进化

generations = 2 # 进化世代数
population_size = 500 # 每一代中的公式数量
tournament_size = 10 # 每一代中被随机选中计算适应度的公式数
const_range = (0.0,10.0)
function_set = init_function+user_function # 函数算子
metric = rankIC_metric # 目标函数作为适应度
random_state = 2008091 # 设置随机种子
factor_gp = SymbolicTransformer(feature_names=fields, 
                                function_set=function_set, 
                                generations=generations, 
                                metric=metric, 
                                population_size=population_size, 
                                tournament_size=tournament_size, 
                                const_range=const_range, 
                                random_state=random_state)
factor_gp.fit(stock_price, target)

with open(r'D:\work\back_test_system\FactorBackTest\gp_model.pkl', 'wb') as f:
    pickle.dump(factor_gp, f)

best_programs = factor_gp._best_programs
best_programs_dict = {}
for p in best_programs:
    factor_name = 'alpha_'+str(best_programs.index(p)+1)
    best_programs_dict[factor_name] = {'fitness':p.fitness_, 'expression':str(p), 'depth':p.depth_, 'length':p.length_}
best_programs_dict = pd.DataFrame(best_programs_dict).T
best_programs_dict = best_programs_dict.sort_values(by='fitness')