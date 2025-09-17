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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from sklearn.model_selection import TimeSeriesSplit
from keras.layers import Reshape
import datetime
from datetime import timedelta
from dateutil.relativedelta import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KernelDensity







def get_path(path):
    path_to_get = path.replace('\\','/')
    if not os.path.exists(path_to_get):
        os.makedirs(path_to_get)
        return path_to_get
    else:
        return path_to_get

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
        # month_list.insert(-1,end_date)
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
    def drop_extreme(self,df_,param):
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


class RNN_Class(Date_Class):
    """THIS IS THE RNN TEST WITHOUT INTERMEDIATE FILES"""
    def __init__(self):
        print('THIS IS THE RNN MODEL CLASS')

    def RNN_HYPER(self,ds,denomi,ds_params):
        """
        The RNN_HYPER function is RNN model with overlap and in compliance with a multiprocess calculation
        input: should be each column of the target dataframe
        col_i: should be column name
        denomi: the fixed parameter for splitting train set and test set
        ds_param: should be the parameter combination generated from loop_val.
        This is not a fourier cycle function
        ds : should be datetime indexed
        """
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

        # test_predict_3D是从scaled_fit_transform从后向前取数（12+overlap）
        test_predict_i = model_i.predict(test_predict_3D)

        # 找到预测值所在的index,预测值是按照1-6来计算

        index_num = df_temp[::-1][0:overlap][::-1]['index'].iloc[0]



        date_index =Date_Class.get_allmonth_nums(self,index_num,len(test_predict))



        # y_predict_temp是overlap的部分
        """23.08.17修改y_predict为series"""
        y_predict = pd.Series(scaler.inverse_transform(test_predict_i).reshape(len(test_predict_i)),
                              index=date_index[0])


        y_predict_temp = pd.merge(
            ds_1,
            pd.DataFrame(y_predict),
            left_index=True,
            right_index=True,

        )
        y_predict_temp.columns = ['true_value', 'prediction']



        df_pm = precision_matrix(y_true=y_predict_temp['true_value'],
                                 y_predict=y_predict_temp['prediction'])
        df_pm['param_comb'] = str(ds_params)


        return df_pm,y_predict,df_history

    def LSTM_HYPER(self,ds,denomi,ds_params):
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
        date_index =Date_Class.get_allmonth_nums(self,index_num,len(test_predict))



        # y_predict_temp是overlap的部分
        """23.08.17修改y_predict为series"""
        y_predict = pd.Series(scaler.inverse_transform(test_predict_i).reshape(len(test_predict_i)),
                              index=date_index[0])


        y_predict_temp = pd.merge(
            ds_1,
            pd.DataFrame(y_predict),
            left_index=True,
            right_index=True,

        )
        y_predict_temp.columns=['true_value','prediction']



        df_pm = precision_matrix(y_true=y_predict_temp['true_value'],
                                 y_predict=y_predict_temp['prediction'])
        df_pm['param_comb'] = str(ds_params)


        return df_pm,y_predict,df_history


# class Best_Hyparams(Date_Class):
#     def __init__(self):
#         print("THIS IS THE BEST_HYPERPARAM CLASS")
#     def best_hyperparam_1(self,df_bp, method, figsize, df_raw, df_raw_test, denomi,
#                         result, history_path, predict_path,
#                         save_path_loss, save_path_predict, save_path_accuracy):
#         """The function is to find the prediction and the loss of RNN and LSTM models above,
#         best_hyparams is for RNN and LSTM with overlap.
#         df_bp is the best parameter dataframe,
#         method is chosen among bpmae,bpmse,bpr,and bpnr,
#         figsize is the figure size parameter,
#         df_raw is the raw data, data without any process, truncated the last 12 data
#         df_raw_test is the whole raw data,
#         history path is the file path where loss data is
#         predict path is the file path where prediction data is
#         """
#         df_method = pd.DataFrame(df_bp.loc[method])
#         # 修改ds_cols的index_range，ds_cols即为原始列名
#         ds_cols = pd.Series(df_bp.columns,index=(range(1,len(df_bp.columns)+1)))
#         # 给df_method加入标志序数
#         df_method['temp'] = pd.Series(range(1, len(ds_cols) + 1), index=df_method.index)
#         for i in range(1,len(ds_cols)+1):
#             col_i = ds_cols[i]  # 列名
#             # time_frame_path_ = result + time_frame_path + col_i + '\\'
#             # time_frame = pd.read_csv(time_frame_path_+'_time_frame_all.csv',index_col=0)
#
#             ds_method = df_raw[col_i]  # 原始数据集的，相当于moc_data_precs[col_i]
#
#             fig_loss_i = plt.figure(figsize=figsize)
#             fig_predict_i = plt.figure(figsize=figsize)
#
#
#             comb = df_method[df_method['temp'] == i][method][0]
#
#             if ds_method.iloc[0] == 0:  # ds_method是原始数据
#                 ds_ = ds_method.drop(ds_method.index[0])
#             else:
#                 ds_ = ds_method
#             length = len(ds_)
#             num = length - math.floor(length / denomi)
#             raw_y_train = ds_.iloc[0:num + 1]
#             raw_y_test = ds_.iloc[num:]
#             # 需要将储存csv文件的路径格式固定
#             history_path_ = result+history_path+col_i+'\\loss\\'+comb+'\\'
#
#             predict_path_ = result+predict_path+col_i+'\\prediction\\'+comb+'\\'
#             history = pd.read_csv(
#                 history_path_
#                 + comb
#                 + '_prediction_loss_data.csv',
#                 index_col=0
#             )
#             predict = pd.read_csv(
#                 predict_path_
#                 + comb
#                 + '_prediction_data.csv',
#                 index_col=0,
#
#             )
#             """adding index conversion"""
#
#
#             # 损失函数绘图
#
#             # calculating the accuracy
#             accu = pd.concat(
#                 [predict,df_raw_test[col_i].loc[predict.index[0]:]]
#                 ,axis=1
#             )
#             accu.columns=['predict','true']
#             accu['accuracy']=1-abs(accu['predict']-accu['true'])/accu['true']
#             average_accu = pd.Series(accu['accuracy'].mean(),index=['average accuracy'])
#             accu_ = pd.concat([accu,average_accu],axis=0)
#
#
#             accu_.to_csv(
#                 get_path(save_path_accuracy+col_i+'\\')
#                 +col_i
#                 +'_accuracy.csv'
#             )
#
#             ax_loss_i = fig_loss_i.add_subplot(111)
#             ax_loss_i.plot(history['train_loss'], label='train_loss')
#             ax_loss_i.plot(history['test_loss'], label='test_loss')
#             ax_loss_i.legend(loc='best')
#
#             ax_loss_i.xaxis.set_major_locator(ticker.MultipleLocator(20))
#             ax_loss_i.set_title(col_i + '_' + comb + '_loss_function')
#
#             ax_predict_i = fig_predict_i.add_subplot(111)
#             ax_predict_i.plot(raw_y_train)
#             ax_predict_i.plot(raw_y_test,label='true')
#
#             ax_predict_i.plot(predict, label='prediction')
#             ax_predict_i.legend(loc='best')
#             ax_predict_i.xaxis.set_major_locator(ticker.MultipleLocator(20))
#
#             ax_predict_i.set_title(col_i + '_' + comb + '_prediction')
#
#
#             fig_loss_i.savefig(
#                 get_path(save_path_loss+col_i+'\\')
#                 +'loss.pdf'
#             )
#             fig_predict_i.savefig(
#                 get_path(save_path_predict+col_i+'\\')
#                 +'prediction.pdf'
#             )
#
#
#         res = 'finish hyper parameter best combination search and figure'
#         return res
#
#     def best_hyperparam_2(self,df_bp,method,figsize,df_raw_test,
#                     result,history_path,predict_path,df_fourier,
#                     save_path_loss,save_path_predict,save_path_accuracy):
#         """This is the best hyper parameter searching and figure painting function for RNN/LSTM without overlap
#         df_fourier is the fourier transform result dataframe"""
#         df_method = pd.DataFrame(df_bp.loc[method])
#         # 修改ds_cols的index_range，ds_cols即为原始列名
#         ds_cols = pd.Series(df_bp.columns, index=(range(1, len(df_bp.columns) + 1)))
#         # 给df_method加入标志序数
#         df_method['temp'] = pd.Series(range(1, len(ds_cols) + 1), index=df_method.index)
#         for i in range(1, len(ds_cols) + 1):
#             col_i = ds_cols[i]  # 列名
#             lag_i = df_fourier[col_i]['target']
#
#             # ds_method = df_raw[col_i]  # 原始数据集的，相当于moc_data_precs[col_i]
#
#             fig_loss_i = plt.figure(figsize=figsize)
#             fig_predict_i = plt.figure(figsize=figsize)
#
#             comb = df_method[df_method['temp'] == i][method][0]
#
#
#             # 需要将储存csv文件的路径格式固定
#             history_path_ = result + history_path + col_i + '\\loss\\' + comb + '\\'
#
#             predict_path_ = result + predict_path + col_i + '\\prediction\\' + comb + '\\'
#
#
#             history = pd.read_csv(
#                 history_path_
#                 + comb
#                 + '_prediction_loss_data.csv',
#                 index_col=0
#             )
#
#             predict_extro = pd.read_csv(
#                 predict_path_
#                 + comb
#                 + '_prediction_data.csv',
#                 index_col=0,
#             )
#             predict_mask = pd.read_csv(
#                 predict_path_ +
#                 comb +
#                 '_mask_predict.csv',
#                 index_col=0
#             )
#
#             accu = pd.concat(
#                 [predict_extro, df_raw_test[col_i].loc[predict_extro.index[0]:]]
#                 , axis=1
#             )
#             accu.columns = ['predict', 'true']
#             accu['accuracy'] = 1 - abs(accu['predict'] - accu['true']) / accu['true']
#             average_accu = pd.Series(accu['accuracy'].mean(), index=['average accuracy'])
#             lag_ = pd.Series([lag_i], index=['fourier circle'])
#             accu_ = pd.concat([accu, average_accu, lag_], axis=0)
#             accu_.to_csv(
#                 get_path(save_path_accuracy + col_i + '\\')
#                 + col_i
#                 + '_accuracy.csv'
#             )
#
#             ax_loss_i = fig_loss_i.add_subplot(111)
#             ax_loss_i.plot(history['train_loss'], label='train_loss')
#             ax_loss_i.plot(history['test_loss'], label='test_loss')
#             ax_loss_i.legend(loc='best')
#
#             ax_loss_i.xaxis.set_major_locator(ticker.MultipleLocator(20))
#             ax_loss_i.set_title(col_i + '_' + comb + '_loss_function')
#             """prepare the prediction figure"""
#             comb_predict = pd.concat([df_raw_test[col_i], predict_mask, predict_extro], axis=1)
#             comb_predict.columns = ['data_all', 'predict_test', 'predict_extrov']
#
#             ax_predict_i = fig_predict_i.add_subplot(111)
#             ax_predict_i.plot(comb_predict['data_all'], label='true_all')
#             ax_predict_i.plot(comb_predict['predict_test'], label='predict_test')
#
#             ax_predict_i.plot(comb_predict['predict_extrov'], label='predict_target')
#             ax_predict_i.legend(loc='best')
#             ax_predict_i.xaxis.set_major_locator(ticker.MultipleLocator(20))
#
#             ax_predict_i.set_title(col_i + '_' + comb + '_prediction')
#
#             fig_loss_i.savefig(
#                 get_path(save_path_loss + col_i + '\\')
#                 + 'loss.pdf'
#             )
#             fig_predict_i.savefig(
#                 get_path(save_path_predict + col_i + '\\')
#                 + 'prediction.pdf'
#             )
#
#         res = 'finish hyper parameter best combination search and figure'
#         return res