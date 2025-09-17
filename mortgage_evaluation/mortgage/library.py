#-*- coding: utf-8 -*-
import pandas as pd
from functools import partial
import numpy as np
import json
import multiprocessing
from multiprocessing import Process
import scipy
import statsmodels.tsa.api as smt
import statsmodels.tsa.stattools
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.api import qqplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import *
import random
import os.path
from scipy.optimize import brent,fmin,minimize
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.interpolate import *
from numpy.polynomial.polynomial import Polynomial
from statsmodels.tsa.stattools import adfuller

from scipy.signal import argrelextrema as ag
from scipy import optimize as opt
from scipy.fftpack import fft,fftfreq

from scipy.optimize import brent, fmin, minimize
import warnings
warnings.filterwarnings("ignore")
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima
from pmdarima.arima import ADFTest
import math
import re
import random
from sklearn.metrics import *
from sklearn import svm
import scipy.stats as st
from statsmodels.tsa.arima_model import ARIMA
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import *
from sklearn.pipeline import *
import keras
import keras.preprocessing
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers.preprocessing import *
from sklearn.model_selection import train_test_split
from itertools import product
import pulp
import scikeras
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
from keras.layers import Reshape
import datetime
from datetime import timedelta
from dateutil.relativedelta import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KernelDensity




def loop_val(list):
    res_loop_val_rnn = []
    for n in product(*list):
        res_loop_val_rnn.append(n)
        print(n)
    ds_params_ocmbination = pd.Series(res_loop_val_rnn)
    return ds_params_ocmbination

def get_sort_index(df,col_name):
    """change the timestamp object into datetime object"""
    time_idx = pd.Series(df[col_name].apply(lambda x:datetime.datetime.strptime(x,'%Y-%m')),index=df.index)
    df['time_idx']= time_idx
    df_tar = df.set_index('time_idx').sort_index(ascending=True)
    return df_tar

def index_process(ds_mask,col,path,name):
    """This is the index_process function
    input should be series object get from query or csv
    outputs include the quarterly arranged data with fill_in, the same data type without fill_in, differ,
    and the combined dataframe and csv file in the specific path"""
    Date_class = Date_Class()
    Fillin = fillin()
    df_time = pd.Series(ds_mask.index)
    quarter = Date_class.get_allquarter(df_time.iloc[0],
                                        df_time.iloc[-1])[0]
    df_quarter = ds_mask.reindex(index=quarter)
    """at least four points should be deployed in the interp1d"""
    df_quarter_fillin = Fillin.fill_in(
        pd.DataFrame(df_quarter).reset_index(),
        col,
        'interp',
        2,
        'cubic',
        get_path(
            path + '\\intermid\\fillin_quarter\\'+name+'\\'
        )
    )

    df_quarter_fillin_ = df_quarter_fillin.set_index(0)[col]
    """start the interpolate process
    change date format from timestamp into datetime object
    and stst should be one date object ahead"""

    df_idx_month = data_freq_change(
        pd.DataFrame(df_quarter_fillin_),
        col
    )

    """测试不填充缺失值直接平滑"""
    df_idx_month_2 = data_freq_change(
        pd.DataFrame(ds_mask).dropna(),
        col
    )
    differ_month = df_idx_month_2 - df_idx_month
    df_idx = pd.concat([
        df_idx_month,
        df_idx_month_2,
        differ_month
    ], axis=1)
    df_idx.columns = [name+'_month_fillin',
                            name+'_month',
                            'differ']
    df_idx.to_csv(
        get_path(
            path + '\\intermid\\month\\'+name+'\\'
        )
        + name+'_idx_monthCombined.csv'
    )
    return df_idx,df_quarter_fillin

def get_standard_data(df,col,path,name,col_name):
    """This is the function for normalization process
    input should be time series dataframe with multiple columns and datetime object index
    output including the standarded normalized data displayed in series type of pandas
    col_name should be like name_month_fillin  OR name_month
    if not fillin exists, results of month_filin and month should be the same
    output:
    df_scaled is raw data scaled to [0,1]
    df_mask is the index_process outcome with fillin/simple mod
    indx is the complete outcome of index_process including df_idx(month outcome) and quarterly data after fillin function """
    df_mask = index_process(df,col,path,name)[0][col_name]
    indx = index_process(df,col,path,name)
    index_time = index_process(df,col,path,name)[0].index
    scaler = MinMaxScaler(feature_range=(0.001,1.001),copy=True)
    scaled_data = scaler.fit_transform(df_mask.values.reshape(-1, 1))
    df_scaled = pd.DataFrame(list(scaled_data.reshape(len(scaled_data))),index=index_time)
    return df_scaled, df_mask,indx


def get_standard_data_2(df):
    """This is only the normalization process
    CHANGE THE FEATURE RANGE INTO (0.001,1.001)"""
    scaler = MinMaxScaler(feature_range=(0.001,1.001),copy=True)
    scaled = scaler.fit_transform(df.values.reshape(-1,1))
    df_scaled = pd.DataFrame(list(scaled.reshape(len(df))),index=df.index)
    return df_scaled

def get_z_score(df):
    """This is the z_score transformer function,
    inputs should be the same as that of get_standard_data_2"""
    mu = df.mean()[0]
    sigma = df.std()[0]
    df_temp = df.apply(lambda x:(x-mu)/sigma)
    return df_temp



def data_freq_change(df,col):
    """there is only one column in df and we need to change the x,y first
    df should be one column dataframe with datetime object index"""
    idx_mask = pd.Series(list(range(0,len(df))),index=df.index)
    df['idx'] = idx_mask

    """定义实例再调用实例"""
    Date_class = Date_Class()
    """修改插值逻辑，将整体插值改为每两个季度之间插值"""
    month_list = list(df.index)
    month_list_mask = [(month_list[i],month_list[i+1]) for i in range(0,len(month_list)-1)]
    df_new = pd.DataFrame()
    for n in range(len(month_list_mask)):
        month_list_mask_n = month_list_mask[n]
        start = month_list_mask_n[0]
        end = month_list_mask_n[1]
        month_nums = (end.year-start.year)*12+end.month-start.month
        frqq = month_nums+1
        time_series_n = Date_class.accum_date(
            month_list_mask_n[0]+relativedelta(months=-1),
            frqq,'month')[0]
        y_new_n = pd.Series(list(range(0,len(time_series_n))),index=time_series_n)
        """interpolate process in interp1d"""
        f = interp1d([y_new_n.iloc[0],y_new_n.iloc[-1]],
                     [df[col].loc[time_series_n[0]],df[col].loc[time_series_n[-1]]]
        )
        df_new_n = pd.DataFrame(f(y_new_n),index=time_series_n)
        df_new = pd.concat([df_new,df_new_n],axis=0)
    test = df_new.reset_index()
    df_new_test = test.drop_duplicates('index',keep='first')
    df_new_mask = df_new_test.set_index('index')
    return df_new_mask




def get_differ(df, col_source):
    res_differ = pd.DataFrame()
    for col in df.columns:
        ds = df[col]
        ds_differ = ds - col_source
        res_differ[col] = ds_differ
    return res_differ

def jdg_sequ(df_1, col_1, col_2, df_2):
    # 输出连续缺失值的时间段,df_1 类似raw
    nums = df_1[df_1[col_1].isnull()].index.to_list()  # 第一个dataframe中的空值所在的索引位置
    # 输出连续的位置索引
    ranges = sum((list(t) for t in zip(nums, nums[1:]) if t[0] + 1 != t[1]), [])
    iranges = iter(nums[0:1] + ranges + nums[-1:])
    test = [(n, next(iranges)) for n in iranges]  # 输出连续空值所在的位置索引
    # 将连续的位置索引转换为时间
    res = sum((list([[df_1[col_2].loc[t[0]], df_1[col_2].loc[t[1]]]]) for t in test), [])   # 输出空值(包含散点和连续值)的时间段
    # 输出该连续空值的时间在目标df_2（绘图目标）的位置索引
    prds = []
    for i in range(len(res)):
        prds_i = res[i]
        num_1 = df_2.index.get_loc(prds_i[0])
        num_2 = df_2.index.get_loc(prds_i[1])
        num_i = [num_1, num_2]
        prds.insert(i, num_i)
    return res, test, prds

def get_fig_simple(df,cols_list,xlabel,tick_spacing,path,save_name):
    """the simple fig painting function"""
    fig = plt.figure(figsize=(18, 16))
    ax = fig.add_subplot(111)

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    for i in range(len(cols_list)):
        col_i = cols_list[i]
        ds_i = df[col_i]
        ax.plot(ds_i.index,ds_i)
        ax.set(xlabel=xlabel)
    ax.legend(cols_list,loc='best')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.savefig(
        path+save_name+'.pdf'
    )
    return " end the simple fig painting process"

def get_fig(df, fig_type, x_label, tick_spacing, periods, path, save_name):
    # 以0/1区分图表类型，0代表scatter，1代表plot,
    """df index should be string type
    numbers of columns should be less than 10"""
    fig = plt.figure(figsize=(18, 16))
    cols = pd.Series(df.columns, index=range(1, len(df.columns) + 1))  # 按照第一个参数的列名循环
    length = len(cols)
    for i in range(1, length + 1):
        col_i = cols[i]  # 列名
        print(col_i)
        ax_i = fig.add_subplot(int(str(length) + str(1) + str(i)))
        fig_type_i = fig_type[i-1]
        if fig_type_i == 0:
            ax_i.scatter(df.index, df[col_i], s=75, alpha=0.5)
            # 散点图由第一个参数绘制
            ax_i.set(xlabel=x_label, ylabel=col_i)
            ax_i.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            if len(periods)>0:
                for i in range(len(periods)):
                    period_i = periods[i]
                    strt_i = period_i[0]
                    end_i = period_i[1]
                    ax_i.axvspan(strt_i, end_i, label="missing data", color="green", alpha=0.3)
        if fig_type_i == 1:
            ax_i.plot(df.index, df[col_i], color='blue')
            ax_i.set(xlabel=x_label, ylabel=col_i)
            ax_i.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            if len(periods)>0:
                for i in range(len(periods)):
                    period_i = periods[i]
                    strt_i = period_i[0]
                    end_i = period_i[1]
                    ax_i.axvspan(strt_i, end_i, label="missing data", color="green", alpha=0.3)
    plt.savefig(path + save_name + '.pdf')

def get_best_param(df,col):
    best_param_mae = df[df['mae'] == df['mae'].min()].index
    best_param_mse = df[df['mse'] == df['mse'].min()].index
    best_param_rmse = df[df['rmse'] == df['rmse'].min()].index
    best_param_nrmse = df[df['nrmse'] == df['nrmse'].min()].index
    bpmae = best_param_mae[0]
    bpmse = best_param_mse[0]
    bpr = best_param_rmse[0]
    bpnr = best_param_nrmse[0]

    list_temp = [bpmae,bpmse,bpr,bpnr]
    temp = pd.DataFrame(
        pd.Series(list_temp),
        columns=[col]
    )
    temp['index']=pd.Series(['bpmae','bpmse','bpr','bpnr'])
    temp = temp.set_index('index')
    return temp

def increase():
    n=9
    while n<=10000:
        n+=1
        yield n

def counter():
    it = increase()
    return next(it)

# def get_path(path):
#     path_to_get = path
#     if not os.path.exists(path_to_get):
#         os.makedirs(path_to_get)
#         return path_to_get
#     else:
#         return path_to_get

def get_path(path):
    path_to_get = path.replace('\\','/')
    if not os.path.exists(path_to_get):
        os.makedirs(path_to_get)
        return path_to_get
    else:
        return path_to_get

def to_ds(list):
    length = len(list)+1
    ds = pd.Series(list,
                   index=range(length))
    return ds

def series_to_supervised(data, n_in, n_out=1, dropnan=True):

    n_vars = 1 if type(data) is list else len(pd.DataFrame(data).columns)
    df = pd.DataFrame(data)
    cols = list()
    name = list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))  # 将不同shift的值填入cols
        name += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            name += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            name += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # 拼接数据
        agg = pd.concat(cols, axis=1)
        agg.columns = name
        # 把null值转换为0
        if dropnan:
            agg.dropna(inplace=True)
        # print(agg)
        return agg

def precision_matrix(y_true, y_predict):
    mae = pd.DataFrame([mean_absolute_error(y_true=y_true,
                                            y_pred=y_predict,
                                            sample_weight=None,
                                            multioutput='uniform_average')],
                       columns=['mae'])
    mse = pd.DataFrame([mean_squared_error(y_true=y_true,
                                           y_pred=y_predict,
                                           sample_weight=None,
                                           multioutput='uniform_average')],
                       columns=['mse']
                       )
    rmse = pd.DataFrame([np.sqrt(mean_squared_error
                                 (y_true=y_true,
                                  y_pred=y_predict,
                                  sample_weight=None,
                                  multioutput='uniform_average'))],
                        columns=['rmse'])
    nrmse = pd.DataFrame([(np.sqrt(mean_squared_error
                                   (y_true=y_true,
                                    y_pred=y_predict,
                                    sample_weight=None,
                                    multioutput='uniform_average'))) / (
                                  float(y_predict.max()) - float(y_predict.min()))],
                         columns=['nrmse'])
    precision_matrix = pd.concat([mae, mse, rmse, nrmse], axis=1)
    return precision_matrix


def fourier(ds,k):
    """This is the function to calculate data cycle based on fourier transform"""
    fft_series = fft(ds.values).reshape(len(ds))  # fourier transform - plurals
    power = np.abs(fft_series)  # amplitude
    sample_freq = fftfreq(fft_series.size)  # frequency
    pop_mask = np.where(sample_freq>0)
    freqs = sample_freq[pop_mask]  # positive frequency
    powers = power[pop_mask]  # positive amplitude
    top_k_seasons = k
    top_k_index = np.argpartition(powers,-top_k_seasons)[-top_k_seasons:]
    top_k_power = powers[top_k_index]
    top_k_freq = freqs[top_k_index]
    fft_periods = (1/top_k_freq).astype(int)
    max_periods = max(fft_periods[np.where(fft_periods < len(ds))])
    print('data circles in %s' % max_periods)
    return top_k_power,fft_periods,max_periods



def get_num(path_file):
    f_num = open(path_file,'r+')
    a = f_num.read()
    a = int(a)+1
    f_num.seek(0)
    f_num.truncate()
    f_num.write(str(a))
    f_num.close()
    return str(a)


def func_linear(x,a,b):
    return a*x+b

def func_exponent(x,a,b,c):
    return a*np.exp(b*x)+c

def func_poli(x,a,b,c,d,e):
    return a * np.log(x) + b / x + c * x ** 2 + d * x + e

def func_poli_2(x,a,b,c,d):
    return a/x+b*x**2+c*x+d

def curve_fit_test(mod,df,df_new,time_index):
    """THIS IS MAIN FUNCTION OF CURVE_FIT,
    @@@
    df & df_new should be data after scale transformation
    input should be data frame with time series index and only one column
    mod==0 exponent function
    mod==1 linear function
    df is the input to calculate the co-efficiency
    df_new is the input dataframe for the outcome dependent variable
    """
    x_data = np.array(list(range(1,len(df)+1)))
    y_data = df[0].values
    x_new = 1
    x_new_series = pd.Series(list(range(1,len(df_new))))
    y_new = df_new.iloc[0][0]
    if mod ==0:
        params = opt.curve_fit(func_linear,x_data,y_data)[0][0]
        b = y_new-x_new*params
        y_series_new = x_new_series.apply(lambda x:params*x+b)
        y_series_new_2 = pd.concat([pd.DataFrame(df_new.iloc[0].values),y_series_new])
        y_series_new_2['time_index']=time_index
        y_series_new_3 = y_series_new_2.set_index('time_index')
        y_series_new_scaled = get_standard_data_2(y_series_new_3)
        # y_series_new_scaled = get_standard_data_2(pd.DataFrame(y_series_new))

        return y_series_new_scaled,y_series_new_3
    if mod ==1:
        params=opt.curve_fit(func_exponent,x_data,y_data)[0]
        c=y_new-params[0]*np.exp(params[1]*x_new)
        y_series_new = x_new_series.apply(lambda x:params[0]*np.exp(params[1]*x)+c)
        y_series_new_2 = pd.concat([pd.DataFrame(df_new.iloc[0].values),y_series_new])
        y_series_new_2['time_index']=time_index
        y_series_new_3 = y_series_new_2.set_index('time_index')
        y_series_new_scaled = get_standard_data_2(y_series_new_3)
        return y_series_new_scaled,y_series_new_3
    if mod == 2:
        params = opt.curve_fit(func_poli,x_data,y_data)[0]
        e = y_new - params[0]*np.log(x_new) - params[1]/x_new - params[2]*x_new**2 - params[3]*x_new
        y_series_new = x_new_series.apply(
            lambda x:params[0]*np.log(x)+params[1]/x+params[2]*x**2+params[3]*x+e
        )
        y_series_new_2 = pd.concat([pd.DataFrame(df_new.iloc[0].values),y_series_new])
        y_series_new_2['time_index']=time_index
        y_series_new_3 = y_series_new_2.set_index('time_index')
        y_series_new_scaled = get_standard_data_2(y_series_new_3)
        return y_series_new_scaled, y_series_new_3
    if mod == 3:
        params = opt.curve_fit(func_poli_2,x_data,y_data)[0]
        e = y_new - params[0]*np.log(x_new)+params[1]/x_new-params[2]*x_new**2-params[3]*x_new
        y_series_new = x_new_series.apply(
            lambda x:params[0]*np.log(x)+params[1]/x+params[2]*x**2+params[3]*x+e
        )
        y_series_new_2 = pd.concat([pd.DataFrame(df_new.iloc[0].values),y_series_new])
        y_series_new_2['time_index']=time_index
        y_series_new_3 = y_series_new_2.set_index('time_index')
        y_series_new_scaled = get_standard_data_2(y_series_new_3)
        return y_series_new_scaled,y_series_new_3

class monte_carlo():
    """THIS IS THE MONTE CARLO PROCESS IN THRESHOLD PROJECT,
    INCLUDING KERNEL DENSITY ESTIMATION PROCESS
    """

    def count(self,data):
        """test"""
        lis = np.array(data)
        lis_unique = np.unique(lis)
        x=[]
        y=[]
        for k in lis_unique:
            mask = (lis==k)


    def kernel_density_log(self,df,col,bandwd,cv_test):
        """df is the target dataframe with datetime object index,bandwd is the bandwidth grid
        cv_test is the cross validation parameter"""
        log_test = df[col].apply(lambda x:np.log(x))
        x_input = log_test.values[:,np.newaxis]
        grid = GridSearchCV(KernelDensity(kernel='gaussian'),bandwd,cv=cv_test)
        grid.fit(x_input)
        best_bandw = grid.best_estimator_['bandwidth']


class weighted_average():
    """This is the weight average repeating process in multi-porcess and simple loop"""

    def simple_average(self,df_tar,save_path,df_weight_i,df_temp,m):
        """input df_weight_i should be one slice of weight dataframe,
        weight_dataframe is the weight sorted by columns--tier and index-- the range from 0 to repeating
        df_temp is the reference dataframe of tier
        @@ exg:

        df_weight_i = weight_list.iloc[:1,].T
        df_tar_mask : the target input dataframe to calculate the weighted average of the index
        df_temp: the df_temp_mask
        """

        df_after_weight = pd.DataFrame()
        for i in range(len(df_tar.columns)):
            col_i = df_tar.columns[i]
            tier_i = df_temp[df_temp['class']==col_i]['tier'].iloc[0]
            tier_denomi = len(df_temp[df_temp['tier']==tier_i])
            weight_i_mask = df_weight_i.loc[tier_i]/tier_denomi
            dire = df_temp[df_temp['class']==col_i]['direction'].iloc[0]
            weight_i = weight_i_mask*dire
            ds = df_tar[col_i]
            ds_after_w =ds.apply(lambda x:x*weight_i)
            df_after_weight = pd.concat([df_after_weight,ds_after_w],axis=1)

        df_aw_sum = df_after_weight.sum(axis=1)
        # df_after_weight.to_csv(
        #     get_path(
        #         save_path+'\\weighted_res\\'
        #     )
        #     +str(m)+'_weight_detail_macro.csv'
        # )
        # df_aw_sum.to_csv(
        #     get_path(
        #         save_path+'\\weighted_res\\'
        #     )
        #     +str(m)+'_weight_sum.csv'
        # )

        return df_aw_sum,df_after_weight


    def weight_adjust(self,X,df_1,df_2,df_weight,df_temp_):
        """THIS IS THE WEIGHT ADJUSTING FUNCTION
        matrix_weight: the matrix of weight with weight variables,
        df_2: the index of macro_index,
        df_1: the factor components of provident fund
        df_weight: the raw pf weight matrix generated from dirichlet distribution with
        initial weights set from relation power,
        df_temp: the temp dataframe to record the relation power with class and tier,
        matrix_weight should be in the same structure with df_weight
        @@@
        ALL INDEX SHOULD BE DATETIME OBJECT
        """
        df_weight_res = pd.DataFrame()
        matrix_weight = pd.DataFrame(X,columns=[df_temp_.columns[0]])
        matrix_weight['tier'] = pd.Series(['tier_1','tier_2','tier_3'],index=matrix_weight.index)
        for cols in df_1.columns:
            ds_pf = df_1[cols]
            class_i = df_temp_[df_temp_['class']==cols]['tier'].iloc[0]
            n_denomi = len(df_temp_[df_temp_['tier'] == class_i])
            initial_weight_ds = df_weight[class_i]/n_denomi
            matrix_ds = matrix_weight[matrix_weight['tier']==class_i]['class'].iloc[0]
            initial_matrix = initial_weight_ds.values+matrix_ds
            """THE MATRIX ADJUSTED"""
            tar_ds = ds_pf*initial_matrix
            df_weight_res = pd.concat([df_weight_res,tar_ds],axis=1)
        df_sum = df_weight_res.fillna(0).sum(axis=1)
        df_combined = pd.concat([df_2,df_sum],axis=1)
        df_combined.columns=['macro','pf']
        df_I = df_combined['macro']-df_combined['pf']
        adf_p = statsmodels.tsa.stattools.adfuller(df_I)[1]
        return adf_p

    def func_tar_adf(self, X,df_1,df_2,df_weight,df_temp_):
        weight_adj = self.weight_adjust(X,df_1,df_2,df_weight,df_temp_)
        xIni = np.array([-2,2])
        xopt = fmin(weight_adj,xIni)
        return xopt


    def weight_average_loop(self,df_tar, save_path, weight_list,df_temp):
        start = datetime.datetime.now()
        print(start)
        print("start the weight average loop process ")
        for i in range(len(weight_list)):
            weight_i = weight_list.iloc[i]
            df_sum = self.simple_average(df_tar,save_path,weight_i,df_temp,i)[0]
            df_detail = self.simple_average(df_tar, save_path, weight_i,df_temp,i)[1]
            print(df_sum)
            print(df_detail)
        print("end the weight average loop process")
        end = datetime.datetime.now()
        period = end-start
        print(period)
        print("close the loop process in weighted average function")
        return df_sum

    def weighted_average_multiprocess(self,weight_list, df_tar, save_path):
        start = datetime.datetime.now()
        print(start)
        print("start the weight_average ")
        pool_num = multiprocessing.cpu_count()
        Pool_ = multiprocessing.Pool(processes=pool_num)
        func = partial(self.simple_average, df_tar, save_path)
        temp = Pool_.map(func, weight_list)
        Pool_.join()
        Pool_.close()
        end = datetime.datetime.now()
        period = end-start
        print(period)

        return "close the weighted_average multiprocess"

class pf_process():
    """This is the provident fund data process function class
    including classifying data function,
    """
    def get_typed_data(self,df,save_path,type_A,type_B,B_1,B_2,time_index_col):
        """Type_A is the classification A list, should be columns name of df
        Type_B is the second classification list, should be columns name of df"""
        df['time_index'] = df[time_index_col].apply(lambda x:datetime.datetime.strptime(str(x),"%Y%m"))
        list_a = df[type_A].unique()
        for a in range(len(list_a)):
            obj_a = list_a[a]
            df_a = df[df[type_A]==obj_a].set_index('time_index')
            if type_B !=():
                df_a_b_1 = df_a[df_a[type_B]==B_1]
                df_a_b_2 = df_a[df_a[type_B]==B_2]
                df_a_b_1.to_csv(
                    get_path(
                        save_path+B_1+'\\'
                    )
                    +obj_a+'_'+B_1+'.csv'
                )
                df_a_b_2.to_csv(
                    get_path(
                        save_path+B_2+'\\'
                    )
                    +obj_a+'_'+B_2+'.csv'
                )
            if type_B == ():
                df_a.to_csv(
                    get_path(
                        save_path
                    )
                    +obj_a+'.csv'
                )
        return "close the get_typed data process"


class fillin():
    """This is the fillin function in three methods
    simple_linear, interpolate, and lagrange"""

    def func(self,k, x, b):
        return k * x + b

    def Simple_linear(self,A, B, x_tar):  # A/B 分别是数组
        """定义实例"""
        func_ = self.func(A,x_tar,B)
        params = opt.curve_fit(func_, A, B)
        k = params[0][0]
        b = params[0][1]
        y_tar = k * x_tar + b
        return y_tar

    def fill_in(self,df_m, col, fill_in_method, k, interp_method, save_path):
        # 假设前提：插值无法预测，需要先判断数据的最后一个是否为空
        # 找目标列的非空值

        df_m_notnu = df_m[col].dropna()
        notnul_index = list(df_m_notnu.index)  # 目标列非空值所在的index
        # 目标列空值所在的index
        nul_index = df_m[df_m[col].isnull()].index.to_list()

        lagr_moc = pd.DataFrame()
        interp_moc = pd.DataFrame()
        SimpleLinear_moc = pd.DataFrame()
        knn_moc = pd.DataFrame()

        for i in range(len(nul_index)):

            ind_i = nul_index[i]
            # 基本假设：在缺失值的位置前后取值k位，并保证去掉空值后还能取到K位
            # k 是前后取值的距离
            # 取数
            new = notnul_index.copy()
            new.insert(0, ind_i)
            new.sort()
            posi = new.index(ind_i)
            # 在not_null中取数，取到的必然是非空数据

            a = new[max(posi - k, 0)]
            b = new[min(posi + k, len(new) - 1)]
            x_i = df_m_notnu.loc[a:b].index.to_list()
            w_i = df_m_notnu.loc[a:b].to_list()
            """lagrange interpolate"""
            if fill_in_method == 'lag':
                print("The lagrange fillin method_"+col+'_'+str(ind_i))
                y = Polynomial(lagrange(x_i, w_i).coef)
                lagr_i = pd.DataFrame([y(ind_i), df_m['index'].loc[ind_i]],
                                      columns=pd.Series(ind_i),
                                      index=['moc_data', 'date_index']).T  # 输出拉格朗日插值
                df_m[col].loc[ind_i] = lagr_i['moc_data']
                lagr_moc = pd.concat([lagr_moc, lagr_i])  # 输出模拟结果

                df_m.to_csv(get_path(save_path + 'lag\\') + 'total_moc_lag.csv')
                lagr_moc.to_csv(get_path(save_path + 'lag\\') + 'moc_lag.csv')

            """scipy.interpolate"""
            if fill_in_method == 'interp':

                interp_mod = interp_method
                print("The interpid method_"+col+'_'+interp_mod+'_'+str(ind_i))
                # f = interp1d(x_i,
                #              w_i,
                #              kind=interp_mod)
                # x_new = df_m.index[a:b]
                y = float(interp1d(x_i,
                                   w_i,
                                   kind=interp_mod,
                                   fill_value="extrapolate").__call__(ind_i))
                interp_i = pd.DataFrame([y, df_m[0].loc[ind_i]], columns=pd.Series(ind_i),
                                        index=['moc_data', 'date_index']).T
                df_m[col].loc[ind_i] = interp_i['moc_data']
                interp_moc = pd.concat([interp_moc, interp_i])
                interp_moc.to_csv(get_path(save_path + 'interp\\') + 'interp_moc.csv')
                df_m.to_csv(get_path(save_path + 'interp\\') + 'total_moc_interp1d.csv')

            # 输出简单线性插值法,修改条件为仅2个点求插值
            if fill_in_method == 'simple_line':
                print("The simple linear method_"+col+'_'+str(ind_i))
                #  非最后一个点缺失
                a_SL = new[max(posi - 1, 0)]
                b_SL = new[min(posi + 1, len(new) - 1)]

                try:
                    x_i_SL = np.array(df_m_notnu.loc[a_SL:b_SL].index)
                    w_i_SL = np.array(df_m_notnu.loc[a_SL:b_SL])
                    Simple_Linear = self.Simple_linear(x_i_SL, w_i_SL, ind_i)
                    # y = Simple_Linear(x_i_SL, w_i_SL, ind_i)
                    y = Simple_Linear()
                    SimpL_i = pd.DataFrame([y, df_m['index'].loc[ind_i]],
                                           columns=pd.Series(ind_i),
                                           index=['moc_data', 'date_index']).T
                    df_m[col].loc[ind_i] = SimpL_i['moc_data']
                    SimpleLinear_moc = pd.concat([SimpleLinear_moc, SimpL_i])
                # 最后一个值缺失
                except:
                    x_i_sl = np.array(df_m_notnu.loc[a_SL - 1:a_SL].index)
                    w_i_sl = np.array(df_m_notnu.loc[a_SL - 1:a_SL])
                    Simple_Linear = self.Simple_linear(x_i_sl, w_i_sl, ind_i)
                    y_sl = Simple_Linear()
                    SimpL_i = pd.DataFrame([y_sl, df_m['index'].loc[ind_i]],
                                           columns=pd.Series(ind_i),
                                           index=['moc_data', 'date_index']).T
                    df_m[col].loc[ind_i] = SimpL_i['moc_data']
                    SimpleLinear_moc = pd.concat([SimpleLinear_moc, SimpL_i])
                print(SimpleLinear_moc)
                SimpleLinear_moc.to_csv(get_path(save_path + 'SL\\') + 'simpleline_moc.csv')
                df_m.to_csv(get_path(save_path + 'SL\\') + 'total_moc_Sl.csv')
        return df_m

    def fill_in_2(self,df_m, col, fill_in_method, k, interp_method, save_path):
        """THIS IS THE FILL IN METHOD WITH ROLLING WINDOW WHEN SELECTING NOT_NULL DATA
        df_m should be indexed with range data"""
        # 假设前提：插值无法预测，需要先判断数据的最后一个是否为空
        # 找目标列的非空值

        df_m_notnu = df_m[col].dropna()
        notnul_index = list(df_m_notnu.index)  # 目标列非空值所在的index
        # 目标列空值所在的index
        nul_index = df_m[df_m[col].isnull()].index.to_list()

        new = notnul_index.copy()

        for i in range(len(nul_index)):
            ind_i = nul_index[i]
            # 基本假设：在缺失值的位置前后取值k位，并保证去掉空值后还能取到K位
            """保证取数的窗口会随着缺失值的序位增加而后移"""
            # k 是前后取值的距离
            # 取数
            new_temp = notnul_index.copy()  # 非空数集
            new.insert(0, ind_i)
            new.sort()

            new_temp.insert(0,ind_i)
            new_temp.sort()
            posi = new.index(ind_i)
            # 在not_null中取数，取到的必然是非空数据
            """修改原逻辑，将a,b从数位改为序位"""
            ds_new_temp = pd.Series(new_temp)
            index_new_temp_i = ds_new_temp[ds_new_temp==ind_i].index[0]

            a = ds_new_temp.loc[max(index_new_temp_i - k, 0)]
            b = ds_new_temp.loc[min(index_new_temp_i + k, len(new_temp) - 1)]
            x_i = df_m_notnu.loc[a:b].index.to_list()
            w_i = df_m_notnu.loc[a:b].to_list()
            """lagrange interpolate"""
            if fill_in_method == 'lag':
                print("The lagrange fillin method_"+col+'_'+str(ind_i))
                y = Polynomial(lagrange(x_i, w_i).coef)
                lagr_i = pd.DataFrame([y(ind_i), df_m['index'].loc[ind_i]],
                                      columns=pd.Series(ind_i),
                                      index=['moc_data', 'date_index']).T  # 输出拉格朗日插值
                df_m[col].loc[ind_i] = lagr_i['moc_data']


                df_m.to_csv(get_path(save_path + 'lag\\') + 'total_moc_lag.csv')


            """scipy.interpolate"""
            if fill_in_method == 'interp':

                interp_mod = interp_method
                print("The interpid method_"+col+'_'+interp_mod+'_'+str(ind_i))
                # f = interp1d(x_i,
                #              w_i,
                #              kind=interp_mod)
                # x_new = df_m.index[a:b]
                y = float(interp1d(x_i,
                                   w_i,
                                   kind=interp_mod,
                                   fill_value="extrapolate").__call__(ind_i))
                interp_i = pd.DataFrame([y, df_m[0].loc[ind_i]], columns=pd.Series(ind_i),
                                        index=['moc_data', 'date_index']).T
                df_m[col].loc[ind_i] = interp_i['moc_data']
                df_m.to_csv(get_path(save_path + 'interp\\') + 'total_moc_interp1d.csv')

            """simple linear interpolate"""
            # 输出简单线性插值法,修改条件为仅2个点求插值
            if fill_in_method == 'simple_line':
                print("The simple linear method_"+col+'_'+str(ind_i))
                #  非最后一个点缺失
                a_SL = new_temp[max(posi - 1, 0)]
                b_SL = new_temp[min(posi + 1, len(new_temp) - 1)]

                try:
                    x_i_SL = np.array(df_m_notnu.loc[a_SL:b_SL].index)
                    w_i_SL = np.array(df_m_notnu.loc[a_SL:b_SL])
                    Simple_Linear = self.Simple_linear(x_i_SL, w_i_SL, ind_i)
                    # y = Simple_Linear(x_i_SL, w_i_SL, ind_i)
                    y = Simple_Linear()
                    SimpL_i = pd.DataFrame([y, df_m['index'].loc[ind_i]],
                                           columns=pd.Series(ind_i),
                                           index=['moc_data', 'date_index']).T
                    df_m[col].loc[ind_i] = SimpL_i['moc_data']
                    # SimpleLinear_moc = pd.concat([SimpleLinear_moc, SimpL_i])
                # 最后一个值缺失
                except:
                    x_i_sl = np.array(df_m_notnu.loc[a_SL - 1:a_SL].index)
                    w_i_sl = np.array(df_m_notnu.loc[a_SL - 1:a_SL])
                    Simple_Linear = self.Simple_linear(x_i_sl, w_i_sl, ind_i)
                    y_sl = Simple_Linear()
                    SimpL_i = pd.DataFrame([y_sl, df_m['index'].loc[ind_i]],
                                           columns=pd.Series(ind_i),
                                           index=['moc_data', 'date_index']).T
                    df_m[col].loc[ind_i] = SimpL_i['moc_data']
                    # SimpleLinear_moc = pd.concat([SimpleLinear_moc, SimpL_i])

            if fill_in_method == 'knn':
                print("The k nearest neighbor method_"+col+'_'+str(ind_i))

                k_mean = sum(w_i)/len(w_i)
                k_mean_i = pd.DataFrame([k_mean, df_m['index'].loc[ind_i]],
                                      columns=pd.Series(ind_i),
                                      index=['moc_data', 'date_index']).T  # 输出均值
                df_m[col].loc[ind_i] = k_mean_i['moc_data']
                # k_mean_moc = pd.concat([knn_moc, k_mean_i])  # 输出模拟结果
                # k_mean_moc.to_csv(get_path(save_path + 'kmean\\') + 'moc_kmean.csv')
                df_m.to_csv(get_path(save_path + 'kmean\\') + 'total_moc_kmean.csv')

        return df_m





class Date_Class():
    """需要标准日期格式为yyyy-mm-01"""
    def get_alldate(self,start_date,end_date):
        """find the datetime between a specific start and end，
        start_date & end_date should both be datetime object"""
        date_list=[]
        date_list.insert(0, start_date)
        while start_date < end_date:
            start_date += datetime.timedelta(days=1)
            date_list.append(start_date)
        ds_date = pd.Series(date_list, index=range(len(date_list)))
        return ds_date, date_list

    def get_allmonth(self,start_date,end_date):
        """find the month between a specific start and end
        start & end should both be datetime object
        including start &end"""
        month_list = []
        month_list.insert(0,start_date)
        while start_date < end_date:
            start_date += relativedelta(months=1)
            month_list.append(start_date)

        ds_month = pd.Series(month_list,index=range(len(month_list)))
        return month_list,ds_month
    def get_allmonth_nums(self,start_date,nums):
        month_list=[]
        month_list.insert(0,start_date)
        for i in range(1,nums):
            start_date += relativedelta(months=1)
            month_list.append(start_date)
        ds_month = pd.Series(month_list,index=range(nums))
        return month_list,ds_month



    def get_allquarter(self,start_date,end_date):
        """find all the quarter month in the date range
        start & end date should both be datetime object
        including the end_date if it is the last quarter"""

        get_month = self.get_allmonth(start_date, end_date)
        df_month = pd.DataFrame(get_month[1])
        df_month['month']=df_month[0].apply(lambda x:datetime.datetime.strftime(x,'%Y-%m-%d')[5:7])
        num_list = [3,6,9,12]
        str_list = [str(x).zfill(2) for x in num_list if len(str(x))==1]
        str_list.append(str(12))
        df_qurter = df_month[df_month['month'].isin(str_list)]
        return df_qurter



    def index_type_swith(self,ds,mod):
        """index_type switch from datetime object to string and reserve
        input_format:series or dataframe with only one column"""
        ds_tar = pd.DataFrame()
        if mod == 0:
            # from string to datetime
            ds_mask = ds.reset_index()
            ds_mask['idx_tar'] = ds_mask.iloc[:, 0].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))
            ds_mask = ds_mask.set_index('idx_tar')
            ds_tar = ds_mask.drop(ds_mask.columns[0], axis=1)
        if mod == 1:
            # from datetime to string
            ds_mask = ds.reset_index()
            ds_mask['idx_level'] = ds_mask.iloc[:, 0].apply(lambda x: datetime.datetime.strftime(x, "%Y-%m-%d"))
            ds_mask = ds_mask.set_index('idx_level')
            ds_tar = ds_mask.drop(ds_mask.columns[0], axis=1)
        if mod ==2:
            # from int to datetime
            ds_mask = ds.reset_index()
            ds_mask['idx_tar'] = ds_mask.iloc[:,0].apply(lambda x:
                                                datetime.datetime.strptime(str(x),"%Y%m"))
            ds_mask = ds_mask.set_index('idx_tar')
            ds_tar  =ds_mask.drop(ds_mask.columns[0],axis=1)
        if mod ==3:
            # from int to string
            ds_mask = ds.reset_index()
            ds_mask['idx_tar'] = ds_mask.iloc[:,0].apply(
                lambda x:str(x)
            )
            ds_mask = ds_mask.set_index('idx_tar')
            ds_tar = ds_mask.drop(ds_mask.columns[0],axis=1)
        return ds_tar

    def accum_date(self,start,nums_,mod):
        """accumulatingly adding datetime, month mod and day mod
        input start should be datetime object,
        Output is tuple with first element the datetime list and the second element the string object
        excluding start month and start day"""
        date_list = list()
        str_date_list = list()
        nums = nums_
        start_date = start
        if mod == 'month':
            for i in range(1, nums + 1):
                date_i = start_date + relativedelta(months=i)
                str_date_i = datetime.datetime.strftime(date_i, "%Y-%m-%d")
                date_list.insert(i, date_i)
                str_date_list.insert(i, str_date_i)
        if mod == 'day':
            for i in range(1, nums + 1):
                date_i = start_date + timedelta(days=i)
                str_date_i = datetime.datetime.strftime(date_i, "%Y-%m-%d")
                date_list.insert(i, date_i)
                str_date_list.insert(i, str_date_i)
        return date_list, str_date_list



class drop_extreme(Date_Class):
    def __init__(self):
        print("The drop extreme")
    def drop_extre(self,df_,param):
        """drop extreme using median method
        the base of other smoothing function,
        with only one parameter --the higher bound and the lower bound the same """
        df = df_
        n = param
        res = pd.DataFrame()
        for cols in df.columns:
            ds = df[cols]
            med = ds.quantile(0.5)
            ds_med = abs(ds - med).quantile(0.5)
            max = ds_med + n * ds_med
            min = ds_med - n * ds_med
            ds_tar = ds.clip(min, max)
            res[cols] = ds_tar
        return res
    def drop_extreme_med(self,df_,n_lower,n_higher):
        """THIS IS THE DROP EXTREME FUNCTION FOR RATE MOD,
        USED AS DATA SHRINKAGE FUNCTION
        df_ is dataframe,
        n_lower int/float, the lowest bound
        n_high int/float, the highest bound
        """
        res = pd.DataFrame()
        res_max = pd.DataFrame()
        res_min = pd.DataFrame()
        for cols in df_.columns:
            ds = df_[cols]
            med = ds.quantile(0.5)
            # ds_med = abs(ds-med).quantile(0.5)
            ds_med = abs(ds-med).quantile(0.5)
            max_ = med+n_higher*ds_med
            min_ = med-n_lower*ds_med
            ds_tar = ds.clip(min_,max_)
            res = pd.concat([res,ds_tar],axis=1)
            res_max = pd.concat([res_max, pd.DataFrame([max_], columns=[cols])], axis=1)
            res_min = pd.concat([res_min, pd.DataFrame([min_], columns=[cols])], axis=1)
        return res,res_max,res_min

    def drop_extreme_2(self,df_,n_lower,n_higher):
        """n_lower is the lower bound parameter
        n_higher is the higher bound parameter
        output is a tuple consists of the final result of the raw dataframe, the maximum bound, and the minimum bound"""
        res = pd.DataFrame()
        res_max = pd.DataFrame()
        res_min = pd.DataFrame()
        for cols in df_.columns:
            ds = df_[cols]
            med = ds.quantile(0.5)
            ds_med = abs(ds - med).quantile(0.5)
            max = ds_med + n_higher * ds_med
            min = ds_med - n_lower * ds_med
            ds_tar = ds.clip(min, max)
            res[cols] = ds_tar
            res_max = pd.concat([res_max,pd.DataFrame([max],columns=[cols])],axis=1)
            res_min = pd.concat([res_min,pd.DataFrame([min],columns=[cols])],axis=1)
        return res, res_max, res_min

    def drop_Extreme_rate(self,df_rate_0,df_rate_1,df_raw):
        """
        THIS IS THE FORWARD REVERSE FUNCTION OF THE RATE DROPPING METHOD.
        df_rate_0 is the rate dataframe before dropping extreme
        df_rate_1 is the rate dataframe after dropping extreme
        df_raw is the raw data(not rate)
        using backward number to reverse to raw 前值累乘还原
        input should be dataframe object
        """

        differ = df_rate_0 - df_rate_1
        raw_res = pd.DataFrame()
        for cols in differ.columns:
            ds_raw = pd.DataFrame(df_raw[cols])  # 未去极值的原始累加数据
            ds_rate_1 = df_rate_1[cols]  # 去极值后的日增长率
            # ds是差别矩阵
            ds = pd.DataFrame(
                differ[cols]).reset_index()

            ds['temp'] = pd.Series(range(1, len(ds) + 1), index=ds.index)  # 加入顺序判断。顺序从1开始
            df_differ = ds[ds[cols] != 0]  # 需要去极值的时间和顺序
            temp_loc = df_differ['temp'] - 1  # 需要去极值的前值的顺序
            # 前值
            temp_raw = ds_raw.iloc[temp_loc].reset_index()
            # 去极值后的增长率,ds_rate_1是differ 重置索引之后的数据

            ds_mask = pd.DataFrame(differ[cols])
            df_differ_mask = ds_mask[ds_mask[cols]!=0]
            temp_rate = ds_rate_1[ds_rate_1.index.isin(df_differ_mask.index)].reset_index()
            temp_rate.columns = ['index_date', 'rate']

            df_res_temp = pd.concat([temp_raw[cols], temp_rate], axis=1)
            df_res_temp['replace'] = df_res_temp[cols] * (df_res_temp['rate'] + 1)
            df_res_temp = df_res_temp.set_index('index_date')
            replace = pd.DataFrame(pd.Series(df_res_temp['replace']))
            replace.columns = [cols]
            temp_ds_raw = ds_raw[~ds_raw.index.isin(df_res_temp.index)]

            res_final_0 = pd.concat([temp_ds_raw, replace], axis=0).reset_index()
            res_final_0['index_date'] = res_final_0['index'].apply(
                lambda x: pd.to_datetime(x)
            )
            res_final_0 = res_final_0.set_index('index_date').sort_index(ascending=True)

            res_final_1 = res_final_0.reset_index().set_index('index')
            raw_res = pd.concat([raw_res, res_final_1[cols]], axis=1)
        return raw_res

    def drop_extreme_reverse_2(self,df_rate_0,df_rate_1,df_raw,col):
        """THIS IS THE REVERSE FUNCTION WITH RAW EXTREME VALUES REPLACED BY VALUES OUT OF NEW RATE
        df_rate_0 is the raw rate dataframe
        df_rate_1 is the rate dataframe after drop_extreme function
        df_raw is the raw data without any process
        """
        differ = df_rate_0-df_rate_1
        differ_mask = differ[differ != 0].dropna()
        differ_mask_index = differ_mask.index.tolist()  # date of data need to be replaced
        # df_raw_mask = df_raw.reset_index()
        df_raw_mask = df_raw.reset_index().reset_index()
        """find the order of replace
        order index of timed data needed to be replaced
        """
        temp_tar = df_raw_mask[df_raw_mask['index'].isin(differ_mask_index)]
        # order_mask = df_raw_mask[df_raw_mask.iloc[:, 0].isin(differ_mask_index)].index.tolist()
        order_mask = temp_tar['level_0'].tolist()

        # index_tar = list(map(lambda x: x - 1, order_mask))
        # rep = df_raw_mask[df_raw_mask.index.isin(index_tar)].iloc[:, 1]
        for idx in range(len(temp_tar)):
            date_idx = temp_tar['index'].iloc[idx]
            rate_1_idx = df_rate_1.loc[date_idx][col]
            order_idx = temp_tar['level_0'].iloc[idx]-1
            raw_value = df_raw_mask.loc[order_idx][col]
            new_value = raw_value*(rate_1_idx+1)
            df_raw_mask.iat[temp_tar['level_0'].iloc[idx],2]=new_value
        df_tar = df_raw_mask.set_index('index')
        df_tar = df_tar.drop('level_0',axis=1)
        return df_tar,df_raw_mask

    def reverse_rate_mask(self,df_rate_0,df_rate_1,df_raw):
        """
        df_rate_0 is the raw change rate without dropping extreme values
        df_rate_1 is the change rate after dropping extreme values using drop_extreme_2
        df_raw is raw data
        way of rate calculation pct_change(-1)
        input should be series type or slice of dataframe by columns
        """
        differ = df_rate_0 - df_rate_1
        differ_mask = differ[differ != 0].dropna()
        """time index of timed data needed to be replaced"""
        differ_mask_index = differ_mask.index.tolist()
        """find the order of replace"""
        df_raw_mask = df_raw.reset_index()
        """order index of timed data needed to be replaced"""
        order_mask = df_raw_mask[df_raw_mask.iloc[:, 0].isin(differ_mask_index)].index.tolist()

        index_tar = list(map(lambda x: x + 1, order_mask))
        rep = df_raw_mask[df_raw_mask.index.isin(index_tar)].iloc[:, 1]

        rate_tar = df_rate_1[df_rate_1.index.isin(differ_mask_index)] + 1
        rate_tar_ary = rate_tar.values.reshape(len(rate_tar))
        replace = rep.values * rate_tar_ary
        list_replace = list(replace)
        ds_replace_mask = pd.Series(list_replace, index=differ_mask_index)
        ds_1_mask = df_raw[~df_raw.index.isin(differ_mask_index)]

        ds_1 = Date_Class.index_type_swith(self,ds_1_mask, 0)
        ds_replace = Date_Class.index_type_swith(self,ds_replace_mask, 0)

        ds_mask = pd.concat([ds_1.iloc[:, 0], ds_replace.iloc[:, 0]], axis=0)
        ds_tar_mask = ds_mask.sort_index(ascending=True)
        ds_tar = Date_Class.index_type_swith(self,ds_tar_mask, 1)
        return ds_tar

class RNN_Class(Date_Class):
    def __init__(self):
        print('THIS IS THE RNN MODEL CLASS')
    def RNN_HYPER(self,ds,col_i,result,denomi,a,ds_params):
        """The RNN_HYPER function is RNN model with overlap and in compliance with a multiprocess calculation
        input should be each column of the target dataframe
        col_i should be column name
        denomi is the fixed parameter for splitting train set and test set
        ds_param should be the parameter combination generated from loop_val.
        This is not a fourier cycle function"""
        print("start the RNN hyper parameter searching in multiprocess")

        if ds.iloc[0] == 0:
            ds_1 = ds.drop(ds.index[0])
        else:
            ds_1 = ds
        # 数据归一化,将scaler范围调整为（0，1）
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_fit_transform = scaler.fit_transform(np.array(list(ds_1)).reshape(-1, 1))
        df_temp = pd.DataFrame(scaled_fit_transform,index=ds_1.index).reset_index()
        # dice the scaled_fit_form into test data and extro-forecasting data

        overlap = ds_params[-1]
        # 将时间序列数据转化为监督数据
        # 最节省数据量的测试方法是直接后推12+overlap

        ds_supervised_ = series_to_supervised(scaled_fit_transform, 12 + overlap)

        ds_supervised = ds_supervised_.values

        # 划分测试集和训练集,因为train_test_split是随即分割，故此处不采取此函数
        length = len(ds_supervised)
        num = length - math.floor(length / denomi)
        # ds_supervised的第一列是X，ds_supervised的最后一列是Y
        # 后期测试集的true data 需要后挪一位index
        x_train = ds_supervised[:, 0][0:num]
        x_predict = ds_supervised[:, 0][num:]
        y_train = ds_supervised[:, -1][0:num]
        y_true = ds_supervised[:, -1][num:]

        # 22.11.14 从scaled_fit_transform从后向前取数（12+overlap）

        test_predict = scaled_fit_transform[::-1][overlap:12 + 2 * overlap][::-1]

        # 将测试集数据转化为三维数据
        x_train_3D = x_train.reshape((x_train.shape[0], 1, 1))
        x_predict_3D = x_predict.reshape((x_predict.shape[0], 1, 1))

        test_predict_3D = test_predict.reshape(test_predict.shape[0], 1, 1)

        # df_PM = pd.DataFrame()
        print(
            "units is %s" % ds_params[0]
            + " ,dropout_rate %s" % ds_params[1]
            + " ,epoch %s" % ds_params[2]
            + " ,batch_size %s" % ds_params[3]
            + " and overlap is %s" % ds_params[4]
        )
        units = ds_params[0]
        dropout_rate = ds_params[1]
        epoch = ds_params[2]
        batch_size = ds_params[3]
        model_i = Sequential()
        model_i.add(
            keras.layers.SimpleRNN(
                units=units,
                return_sequences=True
            )
        )
        model_i.add(Dropout(dropout_rate))
        model_i.add(
            keras.layers.SimpleRNN(
                units=units
            )
        )
        model_i.add(Activation('relu'))
        model_i.add(Dropout(dropout_rate))
        model_i.add(Dense(1))
        model_i.compile(
            loss='mae',
            optimizer='adam'
        )
        # 22.10.18 修改validation_data为(x_predict_3D,y_true)
        res_i = model_i.fit(
            x_train_3D,
            y_train,
            epochs=epoch,
            batch_size=batch_size,
            verbose=0,
            validation_data=(x_predict_3D, y_true),
            shuffle=False
        )
        history = res_i.history
        train_loss = history['loss']
        test_loss = history['val_loss']
        # 损失函数
        df_history = pd.DataFrame([train_loss, test_loss], index=['train_loss', 'test_loss']).T

        test_predict_i = model_i.predict(test_predict_3D)

        # 找到预测值所在的index,预测值是按照1-6来计算

        index_num = df_temp[::-1][0:overlap][::-1]['index'].iloc[0]


        date_start = datetime.datetime.strptime(index_num, "%Y-%m-%d")

        str_date_list = []
        for i in range(0, len(test_predict)):
            date_i = date_start + relativedelta(months=i)
            str_date_i = datetime.datetime.strftime(date_i, "%Y-%m-%d")
            str_date_list.insert(i, str_date_i)

        # y_predict_temp是overlap的部分
        y_predict = pd.DataFrame(scaler.inverse_transform(test_predict_i), index=str_date_list)
        y_predict_temp = pd.merge(
            ds_1,
            y_predict,
            left_index=True,
            right_index=True,

        )
        y_predict_temp.columns = ['true_value', 'prediction']

        y_predict.to_csv(
            get_path(
                result + 'RNN\\loop_'+a+'\\prediction_data\\' + col_i + '\\prediction\\' + str(ds_params) + '\\'
            )
            + str(ds_params) + '_prediction_data.csv'
        )
        df_history.to_csv(
            get_path(
                result + 'RNN\\loop_'+a+'\\prediction_data\\' + col_i + '\\loss\\' + str(ds_params) + '\\'
            )
            + str(ds_params) + '_prediction_loss_data.csv'
        )

        df_pm = precision_matrix(y_true=y_predict_temp['true_value'],
                                 y_predict=y_predict_temp['prediction'])
        df_pm['param_comb'] = str(ds_params)

        df_pm.to_csv(
            get_path(
                result + 'RNN\\loop_'+a+'\\precision_matrix_detail\\' + col_i + '\\' + str(ds_params) + '\\'
            )
            + str(ds_params) + '_precision_matrix.csv'
        )
        return df_pm

    def LSTM_HYPER(self,ds,col_i,result,denomi,a,ds_params):
        """The LSTM_HYPER is a multiprocess function based on LSTM neuron system with overlap.
        Other parameters are the same as that of RNN_HYPER
        """
        print("THIS IS THE LSTM PROCESS")
        if ds.iloc[0] == 0:
            ds_1 = ds.drop(ds.index[0])
        else:
            ds_1 = ds
        # 数据归一化,输入的参数为去掉首位数为0的ds
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_fit_transform = scaler.fit_transform(np.array(list(ds_1)).reshape(-1, 1))
        df_temp = pd.DataFrame(scaled_fit_transform, index=ds_1.index).reset_index()
        overlap = ds_params[-1]
        # 将时间序列数据转化为监督数据,第一列是input_ds.shift(1)之后去掉nan，第二列是去掉了第一个数的input_ds
        ds_supervised_ = series_to_supervised(scaled_fit_transform, 12+overlap)
        ds_supervised = ds_supervised_.values

        # 划分测试集和训练集,因为train_test_split是随机分割，故此处不采取此函数
        length = len(ds_supervised)
        num = length - math.floor(length / denomi)
        # ds_supervised的第一列是X，ds_supervised的第二列是Y
        # 后期测试集的true data 需要后挪一位index
        x_train = ds_supervised[:, 0][0:num]
        x_test = ds_supervised[:, 0][num:]
        y_train = ds_supervised[:, -1][0:num]
        y_true = ds_supervised[:, -1][num:]

        test_predict = scaled_fit_transform[::-1][overlap:12 + 2 * overlap][::-1]

        # 将测试集数据转化为三维数据
        x_train_3D = x_train.reshape((x_train.shape[0], 1, 1))
        x_test_3D = x_test.reshape((x_test.shape[0], 1, 1))
        test_predict_3D = test_predict.reshape(test_predict.shape[0], 1, 1)
        unit = ds_params[0]
        dropout_rate = ds_params[1]
        epoch = ds_params[2]
        batch_size = ds_params[3]
        print(
            "unit is %s" % unit
            + ",dropout_rate is %s" % dropout_rate
            + ",epoch is %s" % epoch
            + ",and batch_size is %s" % batch_size
            + ", and overlap is %s" % overlap
        )
        model_i = Sequential()
        model_i.add(LSTM(units=unit,

                           input_shape=(x_train_3D.shape[1], x_train_3D.shape[2]),

                           return_sequences=True
                           ))
        model_i.add(Dropout(dropout_rate))

        model_i.add(LSTM(units=unit))
        model_i.add(Dense(1))
        model_i.compile(loss='mae', optimizer='adam')

        res = model_i.fit(x_train_3D,
                                    y_train,
                                    epochs =epoch,
                                    batch_size = batch_size,
                                    validation_data=(x_test_3D, y_true),
                                    verbose=0,
                                    shuffle=False)
        history = res.history
        train_loss = history['loss']
        test_loss = history['val_loss']
        # 损失函数
        df_history = pd.DataFrame([train_loss, test_loss], index=['train_loss', 'test_loss']).T
        # 预测值

        test_predict_i = model_i.predict(test_predict_3D)
        index_num = df_temp[::-1][0:overlap][::-1]['index'].iloc[0]
        date_start = datetime.datetime.strptime(index_num,"%Y-%m-%d")
        str_date_list=[]
        for i in range(0, len(test_predict)):
            date_i = date_start + relativedelta(months=i)
            str_date_i = datetime.datetime.strftime(date_i, "%Y-%m-%d")
            str_date_list.insert(i, str_date_i)
        y_predict = pd.DataFrame(scaler.inverse_transform(test_predict_i),index=str_date_list)
        y_predict_temp = pd.merge(
            ds_1,
            y_predict,
            left_index=True,
            right_index=True,

        )
        y_predict_temp.columns=['true_value','prediction']
        y_predict.to_csv(
            get_path(
                result + 'LSTM\\loop_'+a+'\\prediction_data\\' + col_i + '\\prediction\\' + str(ds_params) + '\\'
            )
            + str(ds_params) + '_prediction_data.csv'
        )
        df_history.to_csv(
            get_path(
                result + 'LSTM\\loop_'+a+'\\prediction_data\\' + col_i + '\\loss\\' + str(ds_params) + '\\'
            )
            + str(ds_params) + '_prediction_loss_data.csv'
        )


        df_pm = precision_matrix(y_true=y_predict_temp['true_value'],
                                 y_predict=y_predict_temp['prediction'])
        df_pm['param_comb'] = str(ds_params)

        df_pm.to_csv(
            get_path(
                result+'LSTM\\loop_'+a+'\\precision_matrix_detail\\'+col_i+'\\'+str(ds_params)+'\\'
            )
            + str(ds_params)+'_precision_matrix.csv'
        )
        return df_pm

    def RNN_HYPER_NO_OVERLAP(self,ds,col_i,result,denomi,lag,a, ds_params):
        """The RNN model without overlap"""
        if ds.iloc[0] == 0:
            ds_1 = ds.drop(ds.index[0])
        else:
            ds_1 = ds
        # 数据归一化,输入的参数为去掉首位数为0的ds

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_fit_transform = scaler.fit_transform(np.array(list(ds_1)).reshape(-1, 1))
        df_temp = pd.DataFrame(scaled_fit_transform, index=ds_1.index).reset_index()
        df_temp_shift = series_to_supervised(df_temp,lag-1).reset_index()
        mask_shift = df_temp_shift[df_temp_shift.columns[-2:]]

        """scaled_fit_transform with time order"""

        """using lag in ds_supervised"""
        ds_supervised_ = series_to_supervised(scaled_fit_transform,lag-1)
        ds_supervised = ds_supervised_.values
        length = len(ds_supervised)
        num = length - math.floor(length / denomi)
        # ds_supervised的第一列是X，ds_supervised的第二列是Y

        x_train = ds_supervised[:, 0][0:num]
        x_test = ds_supervised[:, 0][num:]
        y_train = ds_supervised[:, -1][0:num]
        y_true = ds_supervised[:, -1][num:]
        # x_predict是在12个月向外预测的基础上的第一个预测数的时间向前推lag个时间单位
        x_predict_order_1 = df_temp.index[-1] + 1 - (lag - 1)
        x_predict_order_2 = x_predict_order_1 + 12 - 1
        df_predict = df_temp.loc[x_predict_order_1:x_predict_order_2][0].values

        x_predict = df_predict.reshape(len(df_predict), 1)
        """change the index search method into relativedelta"""
        x_time_train = [df_temp['index'].iloc[0], df_temp['index'].iloc[num - 1]]
        x_time_test = [df_temp['index'].iloc[num],
                       datetime.datetime.strftime(
                           datetime.datetime.strptime(df_temp['index'].iloc[num], '%Y-%m-%d') + relativedelta(
                               months=len(x_test) - 1),
                           '%Y-%m-%d')
                       ]
        y_time_train = [mask_shift['var1(t)'].iloc[0], mask_shift['var1(t)'].iloc[num - 1]]
        y_time_test = [mask_shift['var1(t)'].iloc[num],
                       datetime.datetime.strftime(
                           datetime.datetime.strptime(mask_shift['var1(t)'].iloc[num], '%Y-%m-%d') + relativedelta(
                               months=len(y_true) - 1),
                           '%Y-%m-%d')
                       ]
        time_frame = pd.DataFrame(
            [
                [x_time_train[0], x_time_train[1]],
                [y_time_train[0], y_time_train[1]],
                [x_time_test[0], x_time_test[1]],
                [y_time_test[0], y_time_test[1]]
            ],
            index=['x_train_time', 'y_train_time', 'x_test_time', 'y_test_time'],
            columns=['start_time', 'end_time']
        )

        x_train_3D = x_train.reshape(len(x_train), 1, 1)
        test_predict_3D = x_predict.reshape(len(x_predict), 1, 1)
        x_test_3D = x_test.reshape(len(x_test), 1, 1)
        """considering dropping overlap, and using the accuracy matrix of the test set instead"""
        unit = ds_params[0]
        dropout_rate = ds_params[1]
        epoch = ds_params[2]
        batch_size = ds_params[3]
        print(
            "unit is %s" % unit
            + ",dropout_rate is %s" % dropout_rate
            + ",epoch is %s" % epoch
            + ",and batch_size is %s" % batch_size
        )
        model_i = Sequential()
        model_i.add(
            keras.layers.SimpleRNN(
                units=unit,
                return_sequences=True
            )
        )
        model_i.add(Dropout(dropout_rate))

        model_i.add(keras.layers.SimpleRNN(units=unit))
        model_i.add(Activation('relu'))
        model_i.add(Dropout(dropout_rate))
        model_i.add(Dense(1))
        model_i.compile(loss='mae', optimizer='adam')

        res = model_i.fit(x_train_3D,
                          y_train,
                          epochs=epoch,
                          batch_size=batch_size,
                          validation_data=(x_test_3D, y_true),
                          verbose=0,
                          shuffle=False)
        history = res.history
        train_loss = history['loss']
        test_loss = history['val_loss']
        """loss function"""
        df_history = pd.DataFrame([train_loss, test_loss], index=['train_loss', 'test_loss']).T
        test_predict_i = model_i.predict(test_predict_3D)

        time_idx_0 = pd.to_datetime(df_temp.iloc[-1]['index'])
        time_idx_list_mask = pd.Series(Date_Class.accum_date(self,time_idx_0, 12, 'month')[0])

        time_idx_list = time_idx_list_mask.apply(lambda x: datetime.datetime.strftime(x, '%Y-%m-%d'))
        """real prediction part"""
        y_predict = pd.DataFrame(scaler.inverse_transform(test_predict_i), index=time_idx_list)

        y_time_idx = datetime.datetime.strptime(y_time_test[0], '%Y-%m-%d') - relativedelta(months=1)
        idx_list_mask = Date_Class.accum_date(self,y_time_idx, len(y_true), 'month')[0]

        """calculating precision part"""
        y_predict_mask = pd.DataFrame(
            scaler.inverse_transform(
                model_i.predict(x_test_3D)
            ),
            index=idx_list_mask)

        y_predict_mask.to_csv(
            get_path(
                result + 'LSTM_tar_' + a + '\\loop\\prediction_data\\' + col_i + '\\prediction\\' + str(
                    ds_params) + '\\'
            )
            + str(ds_params) + '_mask_predict.csv'
        )
        df_1_temp = pd.DataFrame(Date_Class.index_type_swith(self,ds_1.loc[y_time_test[0]:y_time_test[1]], 0))
        y_precision = pd.concat([y_predict_mask, df_1_temp], axis=1)
        y_precision.columns = ['prediction', 'true_value']

        """precision matrix"""
        df_pm = precision_matrix(y_true=y_precision['true_value'],
                                 y_predict=y_precision['prediction'])
        df_pm['param_comb'] = str(ds_params)
        y_precision.to_csv()
        y_predict.to_csv(
            get_path(
                result + 'LSTM_tar_' + a + '\\loop\\prediction_data\\' + col_i + '\\prediction\\' + str(
                    ds_params) + '\\'
            )
            + str(ds_params) + '_prediction_data.csv'
        )
        df_history.to_csv(
            get_path(
                result + 'LSTM_tar_' + a + '\\loop\\prediction_data\\' + col_i + '\\loss\\' + str(ds_params) + '\\'
            )
            + str(ds_params) + '_prediction_loss_data.csv'
        )
        df_pm.to_csv(
            get_path(
                result + 'LSTM_tar_' + a + '\\loop\\precision_matrix_detail\\' + col_i + '\\' + str(
                    ds_params) + '\\'
            )
            + str(ds_params) + '_precision_matrix.csv'
        )
        time_frame.to_csv(
            get_path(
                result + 'LSTM_tar_' + a + '\\loop\\time_frame\\' + col_i + '\\'
            )
            + '_time_frame_all.csv'
        )
        return df_pm

    def LSTM_HYPER_NO_OVERLAP(self,ds,col_i,result,denomi,lag,a, ds_params):
        """LSTM multiprocess model without overlap"""

        if ds.iloc[0] == 0:
            ds_1 = ds.drop(ds.index[0])
        else:
            ds_1 = ds
        # 数据归一化,输入的参数为去掉首位数为0的ds

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_fit_transform = scaler.fit_transform(np.array(list(ds_1)).reshape(-1, 1))
        df_temp = pd.DataFrame(scaled_fit_transform, index=ds_1.index).reset_index()
        df_temp_shift = series_to_supervised(df_temp,lag-1).reset_index()
        mask_shift = df_temp_shift[df_temp_shift.columns[-2:]]

        """scaled_fit_transform with time order"""


        """using lag in ds_supervised"""
        ds_supervised_ = series_to_supervised(scaled_fit_transform,lag-1)
        ds_supervised = ds_supervised_.values
        length = len(ds_supervised)
        num = length - math.floor(length / denomi)

        # ds_supervised的第一列是X，ds_supervised的第二列是Y

        x_train = ds_supervised[:, 0][0:num]
        x_test = ds_supervised[:, 0][num:]
        y_train = ds_supervised[:, -1][0:num]
        y_true = ds_supervised[:, -1][num:]
        # x_predict是在12个月向外预测的基础上的第一个预测数的时间向前推lag个时间单位
        x_predict_order_1 = df_temp.index[-1]+1-(lag-1)
        x_predict_order_2 = x_predict_order_1+12-1
        df_predict = df_temp.loc[x_predict_order_1:x_predict_order_2][0].values

        x_predict = df_predict.reshape(len(df_predict),1)


        """change the time frame location function"""

        x_time_train = [df_temp['index'].iloc[0],df_temp['index'].iloc[num-1]]
        x_time_test = [df_temp['index'].iloc[num],
                       datetime.datetime.strftime(
                           datetime.datetime.strptime(df_temp['index'].iloc[num],'%Y-%m-%d')+relativedelta(months=len(x_test)-1),
                           '%Y-%m-%d')
                       ]
        y_time_train = [mask_shift['var1(t)'].iloc[0],mask_shift['var1(t)'].iloc[num-1]]
        y_time_test = [mask_shift['var1(t)'].iloc[num],
                       datetime.datetime.strftime(
                           datetime.datetime.strptime(mask_shift['var1(t)'].iloc[num],'%Y-%m-%d')+relativedelta(months=len(y_true)-1),
                           '%Y-%m-%d')
                       ]

        time_frame = pd.DataFrame(
            [
                [x_time_train[0],x_time_train[1]],
                [y_time_train[0],y_time_train[1]],
                [x_time_test[0],x_time_test[1]],
                [y_time_test[0],y_time_test[1]]
            ],
            index=['x_train_time','y_train_time','x_test_time','y_test_time'],
            columns=['start_time','end_time']
        )

        x_train_3D = x_train.reshape(len(x_train),1,1)

        test_predict_3D = x_predict.reshape(len(x_predict),1,1)
        x_test_3D = x_test.reshape(len(x_test),1,1)
        """considering dropping overlap, and using the accuracy matrix of the test set instead"""
        unit = ds_params[0]
        dropout_rate = ds_params[1]
        epoch = ds_params[2]
        batch_size = ds_params[3]
        print(
            "unit is %s" % unit
            + ",dropout_rate is %s" % dropout_rate
            + ",epoch is %s" % epoch
            + ",and batch_size is %s" % batch_size
        )
        model_i = Sequential()
        model_i.add(
            LSTM(
                units=unit,
                input_shape=(x_train_3D.shape[1], x_train_3D.shape[2]),
                return_sequences=True
            )
        )
        model_i.add(Dropout(dropout_rate))

        model_i.add(LSTM(units=unit))
        model_i.add(Dense(1))
        model_i.compile(loss='mae', optimizer='adam')

        res = model_i.fit(x_train_3D,
                                    y_train,
                                    epochs =epoch,
                                    batch_size = batch_size,
                                    validation_data=(x_test_3D, y_true),
                                    verbose=0,
                                    shuffle=False)
        history = res.history
        train_loss = history['loss']
        test_loss = history['val_loss']
        """loss function"""
        df_history = pd.DataFrame([train_loss, test_loss], index=['train_loss', 'test_loss']).T
        test_predict_i = model_i.predict(test_predict_3D)


        time_idx_0 = pd.to_datetime(df_temp.iloc[-1]['index'])
        time_idx_list_mask = pd.Series(Date_Class.accum_date(self,time_idx_0,12,'month')[0])

        time_idx_list = time_idx_list_mask.apply(lambda x:datetime.datetime.strftime(x,'%Y-%m-%d'))
        """real prediction part"""
        y_predict = pd.DataFrame(scaler.inverse_transform(test_predict_i),index=time_idx_list)


        y_time_idx = datetime.datetime.strptime(y_time_test[0],'%Y-%m-%d') - relativedelta(months=1)
        idx_list_mask = Date_Class.accum_date(self,y_time_idx,len(y_true),'month')[0]

        """calculating precision part"""
        y_predict_mask = pd.DataFrame(
            scaler.inverse_transform(
                model_i.predict(x_test_3D)
            ),
            index=idx_list_mask)
        y_predict_mask.to_csv(
            get_path(
                result+'LSTM_tar_'+a+'\\loop\\prediction_data\\' + col_i + '\\prediction\\' +str(ds_params)+'\\'
            )
            +str(ds_params)+'_mask_predict.csv'
        )
        df_1_temp = pd.DataFrame(Date_Class.index_type_swith(self,ds_1.loc[y_time_test[0]:y_time_test[1]],0))
        y_precision = pd.concat([y_predict_mask,df_1_temp],axis=1)
        y_precision.columns=['prediction','true_value']

        """precision matrix"""
        df_pm = precision_matrix(y_true=y_precision['true_value'],
                                 y_predict=y_precision['prediction'])
        df_pm['param_comb'] = str(ds_params)
        y_precision.to_csv()
        y_predict.to_csv(
            get_path(
                result + 'LSTM_tar_'+a+'\\loop\\prediction_data\\' + col_i + '\\prediction\\' + str(ds_params) + '\\'
            )
            + str(ds_params) + '_prediction_data.csv'
        )
        df_history.to_csv(
            get_path(
                result + 'LSTM_tar_'+a+'\\loop\\prediction_data\\' + col_i + '\\loss\\' + str(ds_params) + '\\'
            )
            + str(ds_params) + '_prediction_loss_data.csv'
        )
        df_pm.to_csv(
            get_path(
                result+'LSTM_tar_'+a+'\\loop\\precision_matrix_detail\\'+col_i+'\\'+str(ds_params)+'\\'
            )
            + str(ds_params)+'_precision_matrix.csv'
        )
        time_frame.to_csv(
            get_path(
                result+'LSTM_tar_'+a+'\\loop\\time_frame\\'+col_i+'\\'
            )
            +'_time_frame_all.csv'
        )
        return df_pm

class Best_Hyparams(Date_Class):
    def __init__(self):
        print("THIS IS THE BEST_HYPERPARAM CLASS")
    def best_hyperparam_1(self,df_bp, method, figsize, df_raw, df_raw_test, denomi,
                        result, history_path, predict_path,
                        save_path_loss, save_path_predict, save_path_accuracy):
        """The function is to find the prediction and the loss of RNN and LSTM models above,
        best_hyparams is for RNN and LSTM with overlap.
        df_bp is the best parameter dataframe,
        method is chosen among bpmae,bpmse,bpr,and bpnr,
        figsize is the figure size parameter,
        df_raw is the raw data, data without any process, truncated the last 12 data
        df_raw_test is the whole raw data,
        history path is the file path where loss data is
        predict path is the file path where prediction data is
        """
        df_method = pd.DataFrame(df_bp.loc[method])
        # 修改ds_cols的index_range，ds_cols即为原始列名
        ds_cols = pd.Series(df_bp.columns,index=(range(1,len(df_bp.columns)+1)))
        # 给df_method加入标志序数
        df_method['temp'] = pd.Series(range(1, len(ds_cols) + 1), index=df_method.index)
        for i in range(1,len(ds_cols)+1):
            col_i = ds_cols[i]  # 列名
            # time_frame_path_ = result + time_frame_path + col_i + '\\'
            # time_frame = pd.read_csv(time_frame_path_+'_time_frame_all.csv',index_col=0)

            ds_method = df_raw[col_i]  # 原始数据集的，相当于moc_data_precs[col_i]

            fig_loss_i = plt.figure(figsize=figsize)
            fig_predict_i = plt.figure(figsize=figsize)


            comb = df_method[df_method['temp'] == i][method][0]

            if ds_method.iloc[0] == 0:  # ds_method是原始数据
                ds_ = ds_method.drop(ds_method.index[0])
            else:
                ds_ = ds_method
            length = len(ds_)
            num = length - math.floor(length / denomi)
            raw_y_train = ds_.iloc[0:num + 1]
            raw_y_test = ds_.iloc[num:]
            # 需要将储存csv文件的路径格式固定
            history_path_ = result+history_path+col_i+'\\loss\\'+comb+'\\'

            predict_path_ = result+predict_path+col_i+'\\prediction\\'+comb+'\\'
            history = pd.read_csv(
                history_path_
                + comb
                + '_prediction_loss_data.csv',
                index_col=0
            )
            predict = pd.read_csv(
                predict_path_
                + comb
                + '_prediction_data.csv',
                index_col=0,

            )
            """adding index conversion"""


            # 损失函数绘图

            # calculating the accuracy
            accu = pd.concat(
                [predict,df_raw_test[col_i].loc[predict.index[0]:]]
                ,axis=1
            )
            accu.columns=['predict','true']
            accu['accuracy']=1-abs(accu['predict']-accu['true'])/accu['true']
            average_accu = pd.Series(accu['accuracy'].mean(),index=['average accuracy'])
            accu_ = pd.concat([accu,average_accu],axis=0)


            accu_.to_csv(
                get_path(save_path_accuracy+col_i+'\\')
                +col_i
                +'_accuracy.csv'
            )

            ax_loss_i = fig_loss_i.add_subplot(111)
            ax_loss_i.plot(history['train_loss'], label='train_loss')
            ax_loss_i.plot(history['test_loss'], label='test_loss')
            ax_loss_i.legend(loc='best')

            ax_loss_i.xaxis.set_major_locator(ticker.MultipleLocator(20))
            ax_loss_i.set_title(col_i + '_' + comb + '_loss_function')

            ax_predict_i = fig_predict_i.add_subplot(111)
            ax_predict_i.plot(raw_y_train)
            ax_predict_i.plot(raw_y_test,label='true')

            ax_predict_i.plot(predict, label='prediction')
            ax_predict_i.legend(loc='best')
            ax_predict_i.xaxis.set_major_locator(ticker.MultipleLocator(20))

            ax_predict_i.set_title(col_i + '_' + comb + '_prediction')


            fig_loss_i.savefig(
                get_path(save_path_loss+col_i+'\\')
                +'loss.pdf'
            )
            fig_predict_i.savefig(
                get_path(save_path_predict+col_i+'\\')
                +'prediction.pdf'
            )


        res = 'finish hyper parameter best combination search and figure'
        return res

    def best_hyperparam_2(self,df_bp,method,figsize,df_raw_test,
                    result,history_path,predict_path,df_fourier,
                    save_path_loss,save_path_predict,save_path_accuracy):
        """This is the best hyper parameter searching and figure painting function for RNN/LSTM without overlap
        df_fourier is the fourier transform result dataframe"""
        df_method = pd.DataFrame(df_bp.loc[method])
        # 修改ds_cols的index_range，ds_cols即为原始列名
        ds_cols = pd.Series(df_bp.columns, index=(range(1, len(df_bp.columns) + 1)))
        # 给df_method加入标志序数
        df_method['temp'] = pd.Series(range(1, len(ds_cols) + 1), index=df_method.index)
        for i in range(1, len(ds_cols) + 1):
            col_i = ds_cols[i]  # 列名
            lag_i = df_fourier[col_i]['target']

            # ds_method = df_raw[col_i]  # 原始数据集的，相当于moc_data_precs[col_i]

            fig_loss_i = plt.figure(figsize=figsize)
            fig_predict_i = plt.figure(figsize=figsize)

            comb = df_method[df_method['temp'] == i][method][0]


            # 需要将储存csv文件的路径格式固定
            history_path_ = result + history_path + col_i + '\\loss\\' + comb + '\\'

            predict_path_ = result + predict_path + col_i + '\\prediction\\' + comb + '\\'


            history = pd.read_csv(
                history_path_
                + comb
                + '_prediction_loss_data.csv',
                index_col=0
            )

            predict_extro = pd.read_csv(
                predict_path_
                + comb
                + '_prediction_data.csv',
                index_col=0,
            )
            predict_mask = pd.read_csv(
                predict_path_ +
                comb +
                '_mask_predict.csv',
                index_col=0
            )

            accu = pd.concat(
                [predict_extro, df_raw_test[col_i].loc[predict_extro.index[0]:]]
                , axis=1
            )
            accu.columns = ['predict', 'true']
            accu['accuracy'] = 1 - abs(accu['predict'] - accu['true']) / accu['true']
            average_accu = pd.Series(accu['accuracy'].mean(), index=['average accuracy'])
            lag_ = pd.Series([lag_i], index=['fourier circle'])
            accu_ = pd.concat([accu, average_accu, lag_], axis=0)
            accu_.to_csv(
                get_path(save_path_accuracy + col_i + '\\')
                + col_i
                + '_accuracy.csv'
            )

            ax_loss_i = fig_loss_i.add_subplot(111)
            ax_loss_i.plot(history['train_loss'], label='train_loss')
            ax_loss_i.plot(history['test_loss'], label='test_loss')
            ax_loss_i.legend(loc='best')

            ax_loss_i.xaxis.set_major_locator(ticker.MultipleLocator(20))
            ax_loss_i.set_title(col_i + '_' + comb + '_loss_function')
            """prepare the prediction figure"""
            comb_predict = pd.concat([df_raw_test[col_i], predict_mask, predict_extro], axis=1)
            comb_predict.columns = ['data_all', 'predict_test', 'predict_extrov']

            ax_predict_i = fig_predict_i.add_subplot(111)
            ax_predict_i.plot(comb_predict['data_all'], label='true_all')
            ax_predict_i.plot(comb_predict['predict_test'], label='predict_test')

            ax_predict_i.plot(comb_predict['predict_extrov'], label='predict_target')
            ax_predict_i.legend(loc='best')
            ax_predict_i.xaxis.set_major_locator(ticker.MultipleLocator(20))

            ax_predict_i.set_title(col_i + '_' + comb + '_prediction')

            fig_loss_i.savefig(
                get_path(save_path_loss + col_i + '\\')
                + 'loss.pdf'
            )
            fig_predict_i.savefig(
                get_path(save_path_predict + col_i + '\\')
                + 'prediction.pdf'
            )

        res = 'finish hyper parameter best combination search and figure'
        return res