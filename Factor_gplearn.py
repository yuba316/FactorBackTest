# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 17:19:45 2020

@author: yuba316
"""

import numpy as np
import pandas as pd
from pyfinance.ols import PandasRollingOLS
import graphviz
from scipy.stats import rankdata
import pickle
from gplearn import genetic
from gplearn.functions import make_function
from gplearn.genetic import SymbolicTransformer, SymbolicRegressor
from gplearn.fitness import make_fitness

from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split

import jqdatasdk as jq
import jqfactor_analyzer as ja

jq.auth('18903041915', 'iamaman369')

import warnings
warnings.filterwarnings("ignore")

#%% 设置表达式组成元素与数据集划分

start_date = '2019-07-01'
end_date = '2020-06-30'
fields = ['open', 'close', 'low', 'high', 'volume', 'avg', 'pre_close']
stock_price = jq.get_price('000300.XSHG', start_date=start_date, end_date=end_date, fq=None, fields=fields)
stock_price['5'],stock_price['10'],stock_price['15'],stock_price['20'],stock_price['30'],\
    stock_price['60'],stock_price['100'],stock_price['120'],stock_price['200'] = 5,10,15,20,30,60,100,120,200

stock_price['rtn'] = stock_price['close'].shift(-1)/stock_price['open'].shift(-1)-1
stock_price['rtn'].iloc[0] = 0
fields = fields+['5','10','15','20','30','60','100','120','200']
data = stock_price[fields].values
target = stock_price['rtn'].values
test_size = 0.2
test_num = int(len(data)*test_size)
X_train = data[:-test_num]
X_test = data[-test_num:]
y_train = np.nan_to_num(target[:-test_num])
y_test = np.nan_to_num(target[-test_num:])

#%% 设置计算函数

init_function = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min', 'sin', 'cos', 'tan']

def _exp(data): # new
    return np.exp(data)

def _square(data): # new
    return np.square(data)

def _ts_max(data,window):
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(data))
    value = np.array(pd.Series(data.flatten()).rolling(window).max().tolist())
    value = np.nan_to_num(value)
    return value

def _ts_min(data,window):
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(data))
    value = np.array(pd.Series(data.flatten()).rolling(window).min().tolist())
    value = np.nan_to_num(value)
    return value

def _ts_mid(data,window): # new
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(data))
    value = np.array(pd.Series(data.flatten()).rolling(window).median().tolist())
    value = np.nan_to_num(value)
    return value

def _sma(data,window):
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(data))
    value = np.array(pd.Series(data.flatten()).rolling(window).mean().tolist())
    value = np.nan_to_num(value)
    return value

def _wma(data,window): # new
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(data))
    day = np.arange(1,window+1,1)
    value = np.array(pd.Series(data.flatten()).rolling(window).apply(lambda x: (x*day/day.sum()).sum(),raw=True).tolist())
    value = np.nan_to_num(value)
    return value

def _stddev(data,window):
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(data))
    value = np.array(pd.Series(data.flatten()).rolling(window).std().tolist())
    value = np.nan_to_num(value)
    return value

def _skew(data,window): # new
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(data))
    value = np.array(pd.Series(data.flatten()).rolling(window).skew().tolist())
    value = np.nan_to_num(value)
    return value

def _kurt(data,window): # new
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(data))
    value = np.array(pd.Series(data.flatten()).rolling(window).kurt().tolist())
    value = np.nan_to_num(value)
    return value

def _norm(data,window): # new
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(data))
    data = pd.Series(data.flatten())
    rolling = data.rolling(window)
    value = np.array(((data-rolling.mean())/rolling.std()).tolist())
    value = np.nan_to_num(value)
    return value

def _normMaxMin(data,window): # new
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(data))
    data = pd.Series(data.flatten())
    rolling = data.rolling(window)
    MAX,MIN = rolling.max(),rolling.min()
    value = np.array(((data-MIN)/(MAX-MIN)).tolist())
    value = np.nan_to_num(value)
    return value

def _rolling_rank(data):
    value = rankdata(data)[-1]
    return value

def _ts_rank(data,window):
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(data))
    value = np.array(pd.Series(data.flatten()).rolling(window).apply(_rolling_rank).tolist())
    value = np.nan_to_num(value)
    return value

def _ts_argmax(data,window):
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(data))
    value = window-pd.Series(data.flatten()).rolling(window).apply(np.argmax)-1
    value = np.nan_to_num(value)
    return value

def _ts_argmin(data,window):
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(data))
    value = window-pd.Series(data.flatten()).rolling(window).apply(np.argmin)-1
    value = np.nan_to_num(value)
    return value

def _corr(data_x,data_y,window): # new
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(data_x))
    data = pd.DataFrame({'x':data_x.flatten(),'y':data_y.flatten()})
    data = data.rolling(window).corr()
    data.index.names = ['index','key']
    data = data.query('key==\'x\'')['y']
    data.index = data.index.droplevel('key')
    value = np.array(data.tolist())
    value = np.nan_to_num(value)
    return value

def _cov(data_x,data_y,window): # new
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(data_x))
    data = pd.DataFrame({'x':data_x.flatten(),'y':data_y.flatten()})
    data = data.rolling(window).cov()
    data.index.names = ['index','key']
    data = data.query('key==\'x\'')['y']
    data.index = data.index.droplevel('key')
    value = np.array(data.tolist())
    value = np.nan_to_num(value)
    return value

def _ts_sum(data,window):
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(data))
    value = np.array(pd.Series(data.flatten()).rolling(window).sum().tolist())
    value = np.nan_to_num(value)
    return value

def _rolling_prod(data):
    return np.prod(data)

def _product(data,window):
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(data))
    value = np.array(pd.Series(data.flatten()).rolling(window).apply(_rolling_prod).tolist())
    value = np.nan_to_num(value)
    return value

def _delay(data,window):
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(data))
    value = pd.Series(data.flatten()).shift(window)
    value = np.nan_to_num(value)
    return value

def _delta(data,window):
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(data))
    data = pd.Series(data.flatten())
    value = np.array((data-data.shift(window)).tolist())
    value = np.nan_to_num(value)
    return value

def _delta_pct(data,window): # new
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(data))
    data = pd.Series(data.flatten())
    delay = data.shift(window)
    value = np.array(((data-delay)/delay).tolist())
    value = np.nan_to_num(value)
    return value

def _reg_alpha(data_x,data_y,window):
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(data_x))
    data = pd.DataFrame({'x':data_x.flatten(),'y':data_y.flatten()})
    res = PandasRollingOLS(data['x'],data['y'],window)
    data['alpha'] = res.alpha
    value = np.array(data['alpha'].tolist())
    value = np.nan_to_num(value)
    return value

def _reg_beta(data_x,data_y,window):
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(data_x))
    data = pd.DataFrame({'x':data_x.flatten(),'y':data_y.flatten()})
    res = PandasRollingOLS(data['x'],data['y'],window)
    data['beta'] = res.beta
    value = np.array(data['beta'].tolist())
    value = np.nan_to_num(value)
    return value

def _reg_resi(data_x,data_y,window):
    window = window[0]
    if type(window)!=int:
        return np.zeros(len(data_x))
    data = pd.DataFrame({'x':data_x.flatten(),'y':data_y.flatten()})
    res = PandasRollingOLS(data['x'],data['y'],window)
    data['alpha'],data['beta'] = res.alpha,res.beta
    data['resi'] = data['y']-data['alpha']-data['x']*data['beta']
    value = np.array(data['resi'].tolist())
    value = np.nan_to_num(value)
    return value

def _rank(data):
    value = np.array(pd.Series(data.flatten()).rank().tolist())
    value = np.nan_to_num(value)
    return value

def _scale(data):
    k=1
    data = pd.Series(data.flatten())
    value = data.mul(k).div(np.abs(data).sum())
    value = np.nan_to_num(value)
    return value

exp = make_function(function=_exp, name='exp', arity=1)
square = make_function(function=_square, name='square', arity=1)
ts_max = make_function(function=_ts_max, name='ts_max', arity=2)
ts_min = make_function(function=_ts_min, name='ts_min', arity=2)
ts_mid = make_function(function=_ts_mid, name='ts_mid', arity=2)
sma = make_function(function=_sma, name='sma', arity=2)
wma = make_function(function=_wma, name='wma', arity=2)
stddev = make_function(function=_stddev, name='stddev', arity=2)
skew = make_function(function=_skew, name='skew', arity=2)
kurt = make_function(function=_kurt, name='kurt', arity=2)
norm = make_function(function=_norm, name='norm', arity=2)
normMaxMin = make_function(function=_normMaxMin, name='norm_MaxMin', arity=2)
ts_rank = make_function(function=_ts_rank, name='ts_rank', arity=2)
ts_argmax = make_function(function=_ts_argmax, name='ts_argmax', arity=2)
ts_argmin = make_function(function=_ts_argmin, name='ts_argmin', arity=2)
corr = make_function(function=_corr, name='corr', arity=3)
cov = make_function(function=_cov, name='cov', arity=3)
ts_sum = make_function(function=_ts_sum, name='ts_sum', arity=2)
product = make_function(function=_product, name='product', arity=2)
delay = make_function(function=_delay, name='delay', arity=2)
delta = make_function(function=_delta, name='delta', arity=2)
delta_pct = make_function(function=_delta_pct, name='delta_pct', arity=2)
reg_alpha = make_function(function=_reg_alpha, name='reg_alpha', arity=3)
reg_beta = make_function(function=_reg_beta, name='reg_beta', arity=3)
reg_resi = make_function(function=_reg_resi, name='reg_resi', arity=3)
rank = make_function(function=_rank, name='rank', arity=1)
scale = make_function(function=_scale, name='scale', arity=1)

user_function = [exp, square, ts_mid, wma, skew, kurt, norm, normMaxMin, \
                 corr, cov, delta_pct, reg_alpha, reg_beta, reg_resi, \
                     delta, delay, rank, scale, sma, stddev, product, \
                         ts_rank, ts_min, ts_max, ts_argmax, ts_argmin, ts_sum]

#%% 设置目标函数

def _my_metric(y, y_pred, w):
    value = np.sum(y+y_pred)
    return value
my_metric = make_fitness(function=_my_metric, greater_is_better=True)

#%% 生成表达式

generations = 3 # 进化世代数
population_size = 1000 # 每一代中的公式数量
tournament_size = 20 # 每一代中被随机选中计算适应度的公式数
const_range = (0.0,10.0)
function_set = init_function+user_function # 函数算子
metric = my_metric # 目标函数作为适应度
random_state = 316 # 设置随机种子
est_gp = SymbolicTransformer(feature_names=fields, 
                             function_set=function_set, 
                             generations=generations, 
                             metric=metric, 
                             population_size=population_size, 
                             tournament_size=tournament_size, 
                             const_range=const_range, 
                             random_state=random_state)
est_gp.fit(X_train, y_train)

with open(r'D:\work\back_test_system\FactorBackTest\gp_model.pkl', 'wb') as f:
    pickle.dump(est_gp, f)

best_programs = est_gp._best_programs
best_programs_dict = {}
for p in best_programs:
    factor_name = 'alpha_'+str(best_programs.index(p)+1)
    best_programs_dict[factor_name] = {'fitness':p.fitness_, 'expression':str(p), 'depth':p.depth_, 'length':p.length_}

best_programs_dict = pd.DataFrame(best_programs_dict).T
best_programs_dict = best_programs_dict.sort_values(by='fitness')
'''
#%% 单因子分析——alpha2: delay(sma(product(neg(tan(low)))))

def alpha_2(df):
    return _delay(_sma(_product(-np.tan(np.array(df['low'].tolist())))))

stock_price['alpha_2'] = alpha_2(stock_price)

# 设置起止时间
start_date = '2019-07-01'
end_date = '2020-06-30'

# 设置调仓周期
periods=(1, 5, 20)

# 设置分层数量
quantiles=5

# 设置股票池
scu='000300.XSHG'
securities = jq.get_index_stocks(scu)

# 获取需要分析的数据
factor_data = pd.DataFrame(columns=securities, index=stock_price.index)
for security in securities:
    stock_price = jq.get_price(security, start_date=start_date, end_date=end_date, fq='post', fields=fields)
    stock_price['rtn'] = stock_price['close'].shift(-1)/stock_price['open'].shift(-1)-1
    stock_price['alpha_2'] = alpha_2(stock_price)
    factor_data[security] = stock_price['alpha_2']
    
# 使用获取的因子值进行单因子分析
far = ja.analyze_factor(factor=factor_data, 
                        weight_method='avg', 
                        industry='jq_l1', 
                        quantiles=quantiles, 
                        periods=periods, 
                        max_loss=0.25)

# 生成统计图表
far.create_full_tear_sheet(demeaned=False, group_adjust=False, by_group=False, 
                           turnover_periods=None, avgretplot=(5, 15), std_bar=False)
'''