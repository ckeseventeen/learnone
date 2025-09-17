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
from library import *
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
from pandas.testing import assert_frame_equal
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
import math
import library
import jieba
import cpca
import jieba.posseg as pseg
import jieba
import paddle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

DA = library.Date_Class()
scaler = MinMaxScaler(feature_range=(0.001, 1.001), copy=True)
wa = library.weighted_average()

# def get_path(path):
#     path_to_get = path
#     if not os.path.exists(path_to_get):
#         os.makedirs(path_to_get)
#         return path_to_get
#     else:
#         return path_to_get


def get_dupli_count(col_1,col_2,df,name_1):
    """THIS IS THE FUNCTION FOR CHECKING DUPLICATED
    col_1: The counting key for nunique and unique
    col_2: The counting key for counting the presence of col_2
    name_1: The key for groupby in df
    """
    func = {
        col_1:['nunique','unique'],
        col_2:'count'
    }
    res = df.groupby(name_1).agg(func)
    return res

def replace_item(ds_1,ds_2):
    """THIS IS THE FUNCTION FOR REPLACE ANY ITEM THAT IS NOT WANTED(EG. ZERO/NULL)
    ds_1: The target series type object which includes zero/null items that are not wanted,
    ds_2: The substitute series
    both series should be datetime indexed
    """
    not_0_d = ds_1[ds_1 != 0]
    is_0_d = ds_1[ds_1 == 0]  # the index which needs to be changed
    replace_d = ds_2.loc[is_0_d.index]
    combind_psm = pd.concat([not_0_d, replace_d])
    combind_psm_fin = combind_psm.sort_index(ascending=True)
    return combind_psm_fin

def get_dupli(ds,col):
    """THIS IS THE FUNCTION TO FILTER OUT DUPLICATED ITEMS IN SERIES/LIST/DATAFRAMW
    ds should be the target to be filtered
    ds should be series typed object/dataframe
    col is the target col name to be filtered in the dataframe of ds"""
    ds_unique = ds.unique()
    if len(ds)>len(ds_unique):
        df = pd.DataFrame(ds)
        df['range'] = pd.Series(1,index=df.index)
        df_count=pd.DataFrame(df['range'].groupby(df[col]).count())
        df_count.columns=['count']
        dupli = df_count[df_count['count']>1]
        return dupli
    else:
        pass


def get_date_different(x):
    """THIS IS THE FUNCTION TO CHANGE A SERIES OF STR INTO DATE WITH DIFFERENT STRING TYPE IN SAME SERIES"""
    if len(str(x)) == 23:
        date_x = datetime.datetime.strptime(str(x)[0:10],"%Y-%m-%d")
        return date_x
    if len(str(x)) == 10:
        date_x = datetime.datetime.strptime(str(x)[0:10],"%Y-%m-%d")
        return date_x



def test_merge(df_1,df_2,df_comb,col_key_1,col_key_2):
    """THIS IS THE FUNCITON TO TEST THE RESULT OF pd.merge
    ASSUMPTION:
    No duplicating column names in df_comb
    df_1: the first element to merge
    df_2: the second element to merge
    df_comb: the result after merge
    col_key_1: the key to filter in df_1
    col_key_2: the key to filter in df_2
    """
    df_test_1 = df_comb.loc[:,df_1.columns]
    df_test_2 = df_comb.loc[:,df_2.columns]
    df_1_temp = df_1[df_1[col_key_1].isin(df_test_1[col_key_1])].reset_index()
    df_1_temp_temp = df_1_temp.drop(df_1_temp.columns[0],axis=1)
    df_2_temp = df_2[df_2[col_key_2].isin(df_test_2[col_key_2])].reset_index()
    df_2_temp_temp = df_2_temp.drop(df_2_temp.columns[0],axis=1)
    res_1 = (df_test_1.any()==df_1_temp_temp.any())
    res_2 = (df_test_2.any()==df_2_temp_temp.any())
    if (len(res_1[res_1==False])==0) and(len(res_2[res_2==False])==0):
        t="PASS THE MERGE TEST"
        print(t)
        return t
    else:
        p="NEEDS CHECKING"
        print(p)
        return p


class scaled_factor(Date_Class):
    """THIS IS THE CLASS FOR CALCULATING SCALE FACTORS AND SCALE BASIS"""

    def scaled_factor_DCF(self,pv,I,N,real_rmaBal,mod,x,start_month,dkqs,date_now):
        """THIS IS FUNCTION TO CALCULATE SCALED FACTOR USING DISCOUNTED CASH FLOW METHOD IN LOAN VALUE PROCESS
        ASSUMPTION:
        1. loan valuation process are based on asset time value theories
        2. all loans should be measured by fixed interest rate
        3. loans could be measured by equal repayment(Linear) method or equal principal repayment method(Annuity)
        4. do not take commercial loan into consideration
        5. the first month to repay is ont month after signing contract
        PARAMETER DESCRIPTION
        pv: the present value of the total loan
        fv: the final value of the loan when is due
        I: annually interest rate
        N: numbers of periods of repayment
        real_rmaBal: the real remaining balance of the loan,should be dataframe object,should be the last of df_annu/df_leaner
        like pd.DataFrame([remaining_bal],index=[len(df_annu)+1])
        x: the number counter
        start_month: the issue date, should be datetime object
        date_now: the spot time, should be datetime object, should be month-typed datetime object

        mod: 1 for equal repayment,concerning parameters pv,fv,I,and N
        mod: 2 for equal principal repayment, concerning parameters I,N,pv
        s_date: the start date of repayment, based on assumption, should be the signing date of the contract
        e_date: the end date of repayment, calculated by DA function
        """
        print(x)
        intst = (I/12)/100  # monthly interest rates
        if mod == 1:
            """equal repayment method -- Annuity"""
            print("THIS IS THE ANNUITY MOD")
            start_month_repay = start_month+relativedelta(months=1)
            month_repay = Date_Class.get_allmonth_nums(self,start_month_repay,dkqs)
            end_month_repay = month_repay[0][-1]
            df_annu = pd.DataFrame(index=range(1,N+1))
            df_annu['intst'] = 1+intst
            df_annu['N_periods'] = pd.Series(range(1,N+1),index=df_annu.index)
            df_annu['discount_factor'] = df_annu['N_periods'].apply(
                lambda x:math.pow(1+intst,-x)
            )
            df_annu['time_to_repay'] = month_repay[0]
            df_annu['time_to_repay_month'] = df_annu['time_to_repay'].apply(lambda x:
                                                                            datetime.datetime.strptime(
                                                                            str(x)[0:7],
                                                                            "%Y-%m")
            )
            pmt = pv/df_annu['discount_factor'].sum()
            total_repayment = pmt*N
            df_annu['pmt'] = pd.Series(pmt,index=df_annu.index)
            df_annu['principal'] = pd.Series(pv,index=df_annu.index)
            df_annu['total_repayment'] = pd.Series(total_repayment,index=df_annu.index)
            df_annu['total_left'] = df_annu['total_repayment'] - df_annu['pmt']*df_annu['N_periods']



            if (date_now-start_month_repay).days<=0:
                """ the first repay date is later than now, the repay process is not started yet"""
                res_fin = 1
                res_fin_mom = 0
                return res_fin, res_fin_mom
            if ((date_now-start_month_repay).days >0) and ((date_now-end_month_repay).days<0):
                df_annu['total_left'].loc[df_annu['time_to_repay_month']==date_now]=real_rmaBal
                res_scaled = scaler.fit_transform(df_annu['total_left'].values.reshape(-1,1))
                df_res_scaled = pd.DataFrame(list(res_scaled.reshape(len(res_scaled))),index=list(
                    list(df_annu['time_to_repay_month'])
                ))
                date_last = date_now-relativedelta(months=1)
                res_fin_last_2 = df_res_scaled.loc[date_last][0]
                res_fin_2 = df_res_scaled.loc[date_now][0]
                res_fin_mom_2 = (res_fin_2-res_fin_last_2)/res_fin_last_2
                return res_fin_2, res_fin_mom_2

            if (date_now-end_month_repay).days > 0:
                res_fin_3 = 0
                res_fin_mom_3 =0
                return res_fin_3,res_fin_mom_3

        if mod ==2:
            print("THIS IS THE LINEAR MOD")
            start_month_repay = start_month + relativedelta(months=1)
            month_repay = Date_Class.get_allmonth_nums(self, start_month_repay, dkqs)
            end_month_repay = month_repay[0][-1]
            df_linear=pd.DataFrame(index=range(1,N+1))
            df_linear['range'] = pd.Series(range(1,N+1),index=df_linear.index)
            df_linear['monthlyPrinc_repay'] = pd.Series(pv/N,index=df_linear.index)
            df_linear['accumPrinc_repaid'] = df_linear['monthlyPrinc_repay']*df_linear['range']
            df_linear['princ_left'] = pv-df_linear['accumPrinc_repaid']
            df_linear['insts_month'] = df_linear['princ_left']*intst
            df_linear['total_repayment'] = df_linear['monthlyPrinc_repay']+df_linear['insts_month']
            df_linear['total_accum_repayment'] = df_linear['range'].apply(
                lambda x:df_linear['total_repayment'].loc[1:x].sum()
            )
            df_linear['time_to_repay'] = month_repay[0]
            df_linear['time_to_repay_month'] = df_linear['time_to_repay'].apply(lambda x:
                                                                            datetime.datetime.strptime(
                                                                            str(x)[0:7],
                                                                            "%Y-%m")
            )

            df_linear['total_left'] = df_linear['total_repayment'].sum()-df_linear['total_accum_repayment']
            if (date_now-start_month_repay).days <= 0:
                res_fin_linear = 1
                res_fin_linear_mom = 0
                return res_fin_linear,res_fin_linear_mom

            if ((date_now - start_month_repay).days > 0) and ((date_now - end_month_repay).days < 0):
                df_linear['total_left'].loc[df_linear['time_to_repay_month']==date_now]=real_rmaBal
                res_linear_scaled = scaler.fit_transform(df_linear['total_left'].values.reshape(-1,1))
                df_res_linear_scaled = pd.DataFrame(list(res_linear_scaled.reshape(len(res_linear_scaled))),
                                                    index=list(
                    list(df_linear['time_to_repay_month'])
                ))
                date_last_2 = date_now-relativedelta(months=1)
                res_fin_last_linear = df_res_linear_scaled.loc[date_last_2][0]
                res_fin_linear_2 = df_res_linear_scaled.loc[date_now][0]
                res_fin_linear_mom_2 = (res_fin_linear_2-res_fin_last_linear)/res_fin_last_linear
                return res_fin_linear_2, res_fin_linear_mom_2

            if (date_now-end_month_repay).days > 0:
                res_fin_linear_3 = 0
                res_fin_linear_mom_3 = 0
                return res_fin_linear_3,res_fin_linear_mom_3




class houseAge_analysis():
    """THIS IS THE HOUSE AGE ANALYSIS FOR ADJUSTING"""
    def IQR_test(self,df,w_dirichlet,col):
        """THIS IS THE IQR test to analyze house age
        ASSUMPTION:
        1. Classification are based on the IQR standard and the sample house age
        2. Weight adjust factors are based on classification principles and the concerning dirichlet distribution
        3. Fixed five categories
        df: the house age input dataframe with LOANCONTRCODE/JKHTBH as main key, any type of index is ok
        w_list: the list of dirichlet parameter, fixed parameter in mortgage.py,not in parameter
        col: the column name of IQR input
        """

        ds = df[col]
        df['scaled_ds_hag'] = pd.Series(scaler.fit_transform(ds.values.reshape(-1,1)).reshape(len(ds)),index=df.index)
        df_iqr = pd.DataFrame(
            {
             "iqr_min":df['scaled_ds_hag'].min(),
             "iqr_25":df['scaled_ds_hag'].quantile(0.25),
             "iqr_50":df['scaled_ds_hag'].mean(),
             "iqr_75":df['scaled_ds_hag'].quantile(0.75),
             "iqr_max":df['scaled_ds_hag'].max()
             },
            index=['quantile_info']
        )
        # the temp file to define the initial dirichlet parameters

        df_temp_houseage = pd.DataFrame(
            [['tier_1','tier_2','tier_3','tier_4'],
            w_dirichlet,
            [-1,-1,-1,-1]],index=['tier','weight','direction']
        ).T

        df['tier_hag'] = pd.Series(index=df.index)
        df['tier_hag'].loc[(df['scaled_ds_hag']>=df_iqr['iqr_min'][0])&
                     (df['scaled_ds_hag']<df_iqr['iqr_25'][0])]='tier_1'
        df['tier_hag'].loc[(df['scaled_ds_hag']>=df_iqr['iqr_25'][0])&
                                 (df['scaled_ds_hag']<df_iqr['iqr_50'][0])]='tier_2'
        df['tier_hag'].loc[(df['scaled_ds_hag']>=df_iqr['iqr_50'][0])&
                                 (df['scaled_ds_hag']<df_iqr['iqr_75'][0])]='tier_3'
        df['tier_hag'].loc[(df['scaled_ds_hag']>=df_iqr['iqr_75'][0])&
                                 (df['scaled_ds_hag']<df_iqr['iqr_max'][0])]='tier_4'

        df['weight_hag'] = pd.Series(index=df.index)
        df['weight_hag'].loc[df['tier_hag']=='tier_1']=(df_temp_houseage[df_temp_houseage
        ['tier']=='tier_1']['weight'].values[0])/100*(-1)
        df['weight_hag'].loc[df['tier_hag']=='tier_2']=(df_temp_houseage[df_temp_houseage
        ['tier']=='tier_2']['weight'].values[0])/100*(-1)
        df['weight_hag'].loc[df['tier_hag']=='tier_3']=(df_temp_houseage[df_temp_houseage
        ['tier']=='tier_3']['weight'].values[0])/100*(-1)
        df['weight_hag'].loc[df['tier_hag']=='tier_4']=(df_temp_houseage[df_temp_houseage
        ['tier']=='tier_4']['weight'].values[0])/100*(-1)
        return df

    def IQR_G(self,df,col,name,res_list):
        """THIS IS THE GENERAL MODEL FOR SIMPLE IQR TEST
        df: the input dataframe
        col: the target column name to input
        name: the target name to put into the new column in df
        res_list: the res list to put into the new df, length of the res_list should be in aligned with iqr level
        """
        ds = df[col]
        df['scaled_'+name] = pd.Series(scaler.fit_transform(ds.values.reshape(-1,1)).reshape(len(ds)),index=df.index)
        df_iqr = pd.DataFrame(
            {
             "iqr_min":df['scaled_'+name].min(),
             "iqr_25":df['scaled_'+name].quantile(0.25),
             "iqr_50":df['scaled_'+name].mean(),
             "iqr_75":df['scaled_'+name].quantile(0.75),
             "iqr_max":df['scaled_'+name].max()
             },
            index=['quantile_info']
        )
        df['tier_'+name] = pd.Series(index=df.index)
        df['tier_'+name].loc[(df['scaled_'+name] >= df_iqr['iqr_min'][0]) &
                     (df['scaled_'+name] < df_iqr['iqr_25'][0])] = res_list[0]
        df['tier_'+name].loc[(df['scaled_'+name] >= df_iqr['iqr_25'][0]) &
                                 (df['scaled_'+name] < df_iqr['iqr_50'][0])] = res_list[1]
        df['tier_'+name].loc[(df['scaled_'+name] >= df_iqr['iqr_50'][0]) &
                                 (df['scaled_'+name] < df_iqr['iqr_75'][0])] = res_list[2]
        df['tier_'+name].loc[(df['scaled_'+name] >= df_iqr['iqr_75'][0]) &
                                 (df['scaled_'+name] <= df_iqr['iqr_max'][0])] = res_list[3]
        return df

    def iqr_test_G(self,df_in,col,name,df_temp):
        """THIS IS THE IQR TEST GENERAL VERSION"""
        ds = df_in[col]
        df = pd.DataFrame(scaler.fit_transform(ds.values.reshape(-1,1)).reshape(len(ds)),index=df_in.index)
        df.columns=['scaled_ds_'+name]
        df_iqr = pd.DataFrame(
            {
             "iqr_min":df['scaled_ds_'+name].min(),
             "iqr_25":df['scaled_ds_'+name].quantile(0.25),
             "iqr_50":df['scaled_ds_'+name].mean(),
             "iqr_75":df['scaled_ds_'+name].quantile(0.75),
             "iqr_max":df['scaled_ds_'+name].max()
             },
            index=['quantile_info']
        )


        df['tier_'+name] = pd.Series(index=df.index)
        df['tier_'+name].loc[(df['scaled_ds_'+name]>=df_iqr['iqr_min'][0])&
                     (df['scaled_ds_'+name]<df_iqr['iqr_25'][0])]='tier_1'
        df['tier_'+name].loc[(df['scaled_ds_'+name]>=df_iqr['iqr_25'][0])&
                                 (df['scaled_ds_'+name]<df_iqr['iqr_50'][0])]='tier_2'
        df['tier_'+name].loc[(df['scaled_ds_'+name]>=df_iqr['iqr_50'][0])&
                                 (df['scaled_ds_'+name]<df_iqr['iqr_75'][0])]='tier_3'
        df['tier_'+name].loc[(df['scaled_ds_'+name]>=df_iqr['iqr_75'][0])&
                                 (df['scaled_ds_'+name]<df_iqr['iqr_max'][0])]='tier_4'

        df['weight_'+name] = pd.Series(index=df.index)
        df['weight_'+name].loc[df['tier_'+name]=='tier_1']=(df_temp[df_temp
        ['tier']=='tier_1']['weight'].values[0])/100*(-1)
        df['weight_'+name].loc[df['tier_'+name]=='tier_2']=(df_temp[df_temp
        ['tier']=='tier_2']['weight'].values[0])/100*(-1)
        df['weight_'+name].loc[df['tier_'+name]=='tier_3']=(df_temp[df_temp
        ['tier']=='tier_3']['weight'].values[0])/100*(-1)
        df['weight_'+name].loc[df['tier_'+name]=='tier_4']=(df_temp[df_temp
        ['tier']=='tier_4']['weight'].values[0])/100*(-1)



        return df


class address():
    """THIS IS THE CLASS FOR ADDRESS CLASSIFICATION"""
    # def loop_kws(self,kws_list,first_wrd):
    #     """THIS IS TEH FUNCTION TO LOOP THE KEY WORDS IN ORDER TO FORM RE REGULARIZATION
    #     kws_list: the key words list
    #     first_wrd: the first word to append"""
    #
    #     wrs = '([^'+first_wrd+']'+'+'+first_wrd
    #     for i in range(1,len(kws_list)):
    #         str_i = '|.+' + kws_list[i]
    #         wrs = wrs+str_i
    #     wrs_res = wrs+')?(.*)'
    #     return wrs_res
    #
    # def loop_kws_level_1(self,kws_list,first_wrd):
    #     """THIS IS THE ADDRESS REGULARIZATION FORM OF THE FIRST LEVEL ADDRESS"""
    #     wrs_level1 = r'([^(\d|\s)$'+first_wrd+']+'+first_wrd
    #     for i in range(1,len(kws_list)):
    #         str_i = '|.+[' + kws_list[i]+']'
    #         wrs_level1 = wrs_level1+str_i
    #     wrs_level1_res = wrs_level1+')?(.*)'
    #     return wrs_level1_res

    def loop_kws_level_all(self,kws_list,first_word):
        """THIS IS THE ADDRESS REGULARIZATION FORMED TO GET THE CORRECT ADDRESS CLASSIFICATION
        kws_list: the key words list
        first_wrd: the first word to append"
        """
        wrs = r'^.*'+first_word
        for i in range(1,len(kws_list)):
            str_i = '|^.*'+ kws_list[i]
            wrs = wrs+str_i

        return wrs



    def re_regularization(self,str_in,key_words_level_3,key_words_level_2,key_words_level_1):
        """THIS IS THE RE REGULARIZATION TO DIVIDE ADDRESS STRING,
        str_in: the input address in string type which should be like 'XX路xx号/xx街xx小区xxxx栋....'
        key_words_level_3: the key words source concerning community name to divide the apartment complex(community)
        key_words_level_2: the num./street info list
        key_words_level_1: the Rd. info list
        """
        try:
            if (str_in==None)==False:
                addr_pattern_1 = self.loop_kws_level_all(key_words_level_1,key_words_level_1[0])
                addr_level_1 = re.findall(addr_pattern_1, str_in)
                df_add_1 = pd.DataFrame([addr_level_1[0]]) if len(addr_level_1)>0 \
                    else pd.DataFrame([np.NaN])

                addr_pattern_2 = self.loop_kws_level_all(key_words_level_2,key_words_level_2[0])
                len_filter_2 = len(addr_level_1[0]) if len(pd.DataFrame(addr_level_1)) >0 else 0
                addr_level_2 = re.findall(addr_pattern_2,
                                          str_in[len_filter_2:])
                df_add_2 = pd.DataFrame([addr_level_2[0]]) if len(addr_level_2)>0 \
                    else pd.DataFrame([np.NaN])


                addr_pattern_3 = self.loop_kws_level_all(key_words_level_3,key_words_level_3[0])
                len_filter_3 = len(addr_level_2[0]) if len(pd.DataFrame(addr_level_2)) >0 else 0
                addr_level_3 = re.findall(addr_pattern_3,
                                          str_in[len_filter_2+len_filter_3:])
                df_add_3 = pd.DataFrame([addr_level_3[0]]) if len(addr_level_3)>0 \
                    else pd.DataFrame([np.NaN])

                len_filter_4 = len(addr_level_3[0]) if len(pd.DataFrame(addr_level_3)) >0 else 0
                df_add_rest = pd.DataFrame([str_in[len_filter_2+len_filter_3+len_filter_4:]])

                addr_res = pd.concat([df_add_1,df_add_2,df_add_3,df_add_rest]).T
                addr_res.columns=['路/巷','号/街','小区','其它']
                addr_l_l1 = addr_level_1[0] if len(pd.DataFrame(addr_level_1)) >0 else ''
                addr_l_l2 = addr_level_2[0] if len(pd.DataFrame(addr_level_2)) >0 else ''
                addr_l_l3 = addr_level_3[0] if len(pd.DataFrame(addr_level_3)) >0 else ''
                add_res_list=[addr_l_l1,addr_l_l2,addr_l_l3,
                              str_in[len_filter_2+len_filter_3+len_filter_4:]]
                print(addr_res)
                return addr_res,add_res_list
            else:
                res_none = pd.DataFrame([np.NaN,np.NaN,np.NaN,np.NaN],index=['路/巷','号/街','小区','其它']).T
                res_none_list = [np.NaN,np.NaN,np.NaN,np.NaN]
                return res_none,res_none_list
        except:
            res_except = [np.NaN,np.NaN,np.NaN,np.NaN]
            return res_except

    def get_address_cpca(self,df_FWZL,col,key_words_level_3,key_words_level_2,key_words_level_1):
        """THIS IS THE ORIGIN FUNCTION TO GET ADDRESS FROM RAW FWZL DATA,USING RE REGULARIZATION TO CUT ADDRESS
        df_FWZL: the raw FWZL data with JKHTBH/LOANCONTRCODE
        col: the target column name for input
        key_words: the key words used to form the regularization matrix
        """
        df_FWZL['raw_address'] = pd.DataFrame(cpca.transform(df_FWZL[col])['地址'])
        df_FWZL['raw_address'].loc[df_FWZL['raw_address'].isnull()]=\
            df_FWZL['FWZL'].loc[df_FWZL['raw_address'].isnull()]

        df_FWZL['range'] = pd.Series(range(len(df_FWZL)), index=df_FWZL.index)
        """对每一笔贷款的FWZL先提取‘路’信息，然后是‘号’/‘街’"""
        # test =df_FWZL.head(100)

        df_FWZL['adr_l1'] =df_FWZL['range'].apply(lambda x:
                                                  self.re_regularization
                                                      (
                                                      df_FWZL['raw_address'].iloc[x],
                                                      key_words_level_3,
                                                      key_words_level_2,
                                                      key_words_level_1
                                                  )[1][0]
                                                  )
        df_FWZL['adr_l2'] =df_FWZL['range'].apply(lambda x:
                                                  self.re_regularization
                                                      (
                                                      df_FWZL['raw_address'].iloc[x],
                                                      key_words_level_3,
                                                      key_words_level_2,
                                                      key_words_level_1
                                                  )[1][1]
                                                  )
        df_FWZL['adr_l3'] =df_FWZL['range'].apply(lambda x:
                                                  self.re_regularization
                                                      (
                                                      df_FWZL['raw_address'].iloc[x],
                                                      key_words_level_3,
                                                      key_words_level_2,
                                                      key_words_level_1
                                                  )[1][2]
                                                  )
        df_FWZL['adr_l4'] =df_FWZL['range'].apply(lambda x:
                                                  self.re_regularization
                                                      (
                                                      df_FWZL['raw_address'].iloc[x],
                                                      key_words_level_3,
                                                      key_words_level_2,
                                                      key_words_level_1
                                                  )[1][3]
                                                  )
        return df_FWZL

    def get_addr_clusstering(self,df_addr,col_list,count_nums_l3,count_nums_l2,count_nums_l1):
        """THIS IS THE NEIGHBORHOOD COMMUNITY NAME CLUSTERING USING SKLEARN TfidfVectorizer() OR SIMPLE WORD COUNTING
        df_addr: the feature address input
        col_list: the target col in df_addr
        count_nums_l3: the level_3 threshold on which clustering process is based
        count_num_l2: the level_2 threshold on which clustering process is based
        count_num_l1: the level_1 threshold on which clustering process is based

        OUTPUT:
        1. The final outcome of the clustering process
        2. The details of final outcome
        3 and after. The details of different cluster layer
        """

        test_input = df_addr[col_list]

        clus_l3 = test_input[test_input['adr_l3'].notnull()]

        tar_words = clus_l3.drop(clus_l3[clus_l3['adr_l3']==''].index)

        """FREQUENCY ESTIMATION"""

        """THE FIRST LAYER"""
        rep_addr_l3 = pd.DataFrame(tar_words['adr_l3'].groupby(tar_words['adr_l3']).count())
        rep_addr_l3.columns=['addr_l3_count']
        rep_addr_l3['signal_l3'] = pd.Series(1,index=rep_addr_l3.index)
        rep_addr_l3['signal_l3'].loc[rep_addr_l3['addr_l3_count'] < count_nums_l3] = np.NaN
        rep_addr_l3_temp = rep_addr_l3.reset_index()

        rep_sig_l3 = rep_addr_l3_temp[rep_addr_l3_temp['signal_l3'].notnull()]
        rep_sig_l3['range_sig_l3'] = pd.Series(range(len(rep_sig_l3)),index=rep_sig_l3.index)
        rep_sig_temp = rep_sig_l3[['adr_l3','addr_l3_count','range_sig_l3']]   # THE CLUSTERING COMMUNITY RESULT

        combin_l3 = pd.merge(clus_l3,rep_sig_temp,how='outer',on='adr_l3')
        valid_nums_l3 = len(rep_sig_l3)  # THE TARGET VALID L3 CLUSTERED NUMBER
        valid_addr_l3 = combin_l3[combin_l3['range_sig_l3'].notnull()]



        """THE SECOND ROUND OF ADDRESS CLUSTERING, USING ADDRESS OF LEVEL 2"""
        """THE SECOND LAYER"""

        non_after_l3 = combin_l3[combin_l3['range_sig_l3'].isnull()]

        clus_l2 = non_after_l3[non_after_l3['adr_l2']!='']  # adr_l2去空
        rep_addr_l2 = pd.DataFrame(clus_l2['adr_l2'].groupby(clus_l2['adr_l2']).count())
        rep_addr_l2.columns=['addr_l2_count']
        rep_addr_l2['signal_l2']= pd.Series(1,index=rep_addr_l2.index)
        rep_addr_l2['signal_l2'].loc[rep_addr_l2['addr_l2_count'] < count_nums_l2] = np.NaN
        rep_addr_l2_temp = rep_addr_l2.reset_index()

        rep_sig_l2 = rep_addr_l2_temp[rep_addr_l2_temp['signal_l2'].notnull()]
        rep_sig_l2['range_sig_l2'] = pd.Series(range(valid_nums_l3,valid_nums_l3+len(rep_sig_l2)),
                                               index=rep_sig_l2.index)
        rep_sig_temp_2 = rep_sig_l2[['adr_l2','addr_l2_count','range_sig_l2']]

        """change the logic of merge"""
        non_after_l3_merge = pd.merge(non_after_l3,rep_sig_temp_2,how='outer',on='adr_l2')
        tar_l3_merge = non_after_l3_merge[['JKHTBH','addr_l2_count','range_sig_l2']]
        combin_l3_l2 = pd.merge(combin_l3, tar_l3_merge, how='outer', on='JKHTBH')

        combin_l3_l2['concat_l2_l3'] = pd.Series(index=combin_l3_l2.index)

        combin_l3_l2['concat_l2_l3'].loc[combin_l3_l2['range_sig_l2'].notnull()]=combin_l3_l2['range_sig_l2'].loc[
            combin_l3_l2['range_sig_l2'].notnull()
        ]

        combin_l3_l2['concat_l2_l3'].loc[combin_l3_l2['range_sig_l3'].notnull()]=combin_l3_l2['range_sig_l3'].loc[
            combin_l3_l2['range_sig_l3'].notnull()]

        valid_nums_l2 = len(rep_sig_l2)
        valid_addr_l2 = combin_l3_l2[combin_l3_l2['range_sig_l2'].notnull()]

        """THE THIRD ROUND OF ADDRESS CLUSTERING, USING ADDRESS OF LEVEL 1"""
        """THE THIRD LAYER"""
        non_after_l2 = combin_l3_l2[combin_l3_l2['concat_l2_l3'].isnull()]
        clus_l1 = non_after_l2[non_after_l2['adr_l1']!='']  # adr_l1去空
        rep_addr_l1 = pd.DataFrame(clus_l1['adr_l1'].groupby(clus_l1['adr_l1']).count())
        rep_addr_l1.columns=['addr_l1_count']
        rep_addr_l1['signal_l1']= pd.Series(1,index=rep_addr_l1.index)
        rep_addr_l1['signal_l1'].loc[rep_addr_l1['addr_l1_count'] < count_nums_l1] = np.NaN
        rep_addr_l1_temp = rep_addr_l1.reset_index()

        rep_sig_l1 = rep_addr_l1_temp[rep_addr_l1_temp['signal_l1'].notnull()]
        rep_sig_l1['range_sig_l1'] = pd.Series(range(valid_nums_l3+valid_nums_l2,
                                                     valid_nums_l3+valid_nums_l2+len(rep_sig_l1)),
                                               index=rep_sig_l1.index)

        rep_sig_temp_1 = rep_sig_l1[['adr_l1', 'addr_l1_count', 'range_sig_l1']]
        non_after_l2_merge = pd.merge(non_after_l2,rep_sig_temp_1,how='outer',on='adr_l1')
        tar_l2_merge = non_after_l2_merge[['JKHTBH', 'addr_l1_count', 'range_sig_l1']]



        combin_l3_l2_l1 = pd.merge(combin_l3_l2,tar_l2_merge,how='outer',on='JKHTBH')
        valid_nums_l1 = len(rep_sig_l1)
        valid_addr_l1 = combin_l3_l2_l1[combin_l3_l2_l1['range_sig_l1'].notnull()]


        """concat l1,l2,l3"""
        combin_l3_l2_l1['concat_l2_l3_l1'] = pd.Series(index=combin_l3_l2.index)

        combin_l3_l2_l1['concat_l2_l3_l1'].loc[combin_l3_l2_l1['range_sig_l1'].notnull()] = \
            combin_l3_l2_l1['range_sig_l1'].loc[
            combin_l3_l2_l1['range_sig_l1'].notnull()]
        combin_l3_l2_l1['concat_l2_l3_l1'].loc[combin_l3_l2_l1['range_sig_l2'].notnull()] = \
            combin_l3_l2_l1['range_sig_l2'].loc[
            combin_l3_l2_l1['range_sig_l2'].notnull()
        ]
        combin_l3_l2_l1['concat_l2_l3_l1'].loc[combin_l3_l2_l1['range_sig_l3'].notnull()] = \
            combin_l3_l2_l1['range_sig_l3'].loc[
            combin_l3_l2_l1['range_sig_l3'].notnull()
        ]


        """THE FORTH ROUND OF ADDRESS CLUSTERING"""
        """THE FORTH LAYER"""
        non_after_l1 = combin_l3_l2_l1[combin_l3_l2_l1['concat_l2_l3_l1'].isnull()]

        rep_addr_l0 = pd.DataFrame(non_after_l1['INSTNAME'].groupby(non_after_l1['INSTNAME']).count())
        rep_addr_l0.columns=['addr_l0_count']
        rep_addr_l0['signal_l0'] = pd.Series(1, index=rep_addr_l0.index)

        rep_addr_l0_temp = rep_addr_l0.reset_index()

        rep_sig_l0 = rep_addr_l0_temp[rep_addr_l0_temp['signal_l0'].notnull()]
        rep_sig_l0['range_sig_l0'] = pd.Series(range(valid_nums_l3 + valid_nums_l2+valid_nums_l1,
                                                     valid_nums_l3+valid_nums_l2+valid_nums_l1+len(rep_addr_l0_temp)),
                                               index=rep_sig_l0.index)

        rep_sig_temp_0 = rep_sig_l0[['INSTNAME', 'addr_l0_count', 'range_sig_l0']]
        non_after_l1_merge = pd.merge(non_after_l1,rep_sig_temp_0,how='outer',on='INSTNAME')
        tar_l1_merge = non_after_l1_merge[['JKHTBH','addr_l0_count', 'range_sig_l0']]
        combin_l3_l2_l1_l0 = pd.merge(combin_l3_l2_l1,tar_l1_merge,how='outer',on='JKHTBH')


        """concat l1,l2,l3"""
        combin_l3_l2_l1_l0['concat_l3_l2_l1_l0'] = pd.Series(index=combin_l3_l2_l1_l0.index)

        combin_l3_l2_l1_l0['concat_l3_l2_l1_l0'].loc[combin_l3_l2_l1_l0['range_sig_l0'].notnull()] = \
            combin_l3_l2_l1_l0['range_sig_l0'].loc[
            combin_l3_l2_l1_l0['range_sig_l0'].notnull()]
        combin_l3_l2_l1_l0['concat_l3_l2_l1_l0'].loc[combin_l3_l2_l1_l0['range_sig_l1'].notnull()] = \
            combin_l3_l2_l1_l0['range_sig_l1'].loc[
            combin_l3_l2_l1_l0['range_sig_l1'].notnull()]
        combin_l3_l2_l1_l0['concat_l3_l2_l1_l0'].loc[combin_l3_l2_l1_l0['range_sig_l2'].notnull()] = \
            combin_l3_l2_l1_l0['range_sig_l2'].loc[
            combin_l3_l2_l1_l0['range_sig_l2'].notnull()]
        combin_l3_l2_l1_l0['concat_l3_l2_l1_l0'].loc[combin_l3_l2_l1_l0['range_sig_l3'].notnull()] = \
            combin_l3_l2_l1_l0['range_sig_l3'].loc[
            combin_l3_l2_l1_l0['range_sig_l3'].notnull()]
        valid_nums_l0 = len(rep_sig_l0)
        valid_addr_l0 = combin_l3_l2_l1_l0[combin_l3_l2_l1_l0['range_sig_l0'].notnull()]
        valid_nums = pd.DataFrame([valid_nums_l3,valid_nums_l2,valid_nums_l1,valid_nums_l0],
                                  index=['level_3','level_2','level_1','level_0'])
        tar_list_total = ['JKHTBH','INSTNAME','adr_l1','adr_l2','adr_l3','concat_l3_l2_l1_l0']
        df_final_res = combin_l3_l2_l1_l0[tar_list_total]
        return df_final_res,combin_l3_l2_l1_l0, valid_nums,valid_addr_l3,valid_addr_l2,valid_addr_l1,valid_addr_l0


def get_path(path):
    path_to_get = path.replace('\\','/')
    if not os.path.exists(path_to_get):
        os.makedirs(path_to_get)
        return path_to_get
    else:
        return path_to_get
def get_mortgage_factors(df_input,monthly_path,area,list_tar,path_weight,df_temp_m_wa):
    """THIS IS THE FUNCTION TO CALCULATE THE MORTGAGE ANALYSIS FACTORS
    df_input: the input dataframe from last step
    monthly_path & area: the string to save results
    list_tar: the column list to calculate real estate comprehensive index
    path_weight: the path of weight to read the weight matrix
    columns_list: the list of columns in the weight matrix
    df_temp_m_wa: the temp dataframe to calculate the weighted average comprehensive index
    Return:
    df_spot_HcompreIndex: the spot house comprehensive index after MinMaxscaled, with string index
    weight_detail: the time series of different factors after weight adjusting, with string index
    weight_fin: the scaled time series of total HCMI, with string index
    weight_sum: the time series of total HCMI, with string index

    """
    


    """TRADING VOLUME"""

    trading_vol_m = df_input['JKHTBH'].groupby(df_input['month']).count()

    """TRADING PRICE TOTAL"""

    avg_trading_amt_total_m = df_input['FWZJ'].groupby(df_input['month']).sum() / trading_vol_m

    med_trading_amt_m = df_input['FWZJ'].groupby(df_input['month']).median()

    """TRADING AREA-GROSS"""

    avg_trading_garea_m = df_input['FWJZMJ'].groupby(df_input['month']).sum() / trading_vol_m
    med_trading_garea_m = df_input['FWJZMJ'].groupby(df_input['month']).median()

    """TRADING AREA CONSTRUCTION"""

    avg_trading_carea_m = df_input['FWTNMJ'].groupby(df_input['month']).sum() / trading_vol_m
    med_trading_carea_m = df_input['FWTNMJ'].groupby(df_input['month']).median()

    """TRADING PRICE PER SQUARED METER -- CREA--FWTNMJ"""


    fwtnmj_m = replace_item(avg_trading_carea_m, avg_trading_garea_m)
    avg_trading_amt_perm_m = avg_trading_amt_total_m / fwtnmj_m

    """adding average price per squared meter,using FWJZMJ instead"""

    DF_COMBINED_PPSM = replace_item(df_input['FWJZMJ'], df_input['FWTNMJ'])
    df_input['avg_price_psm_d'] = df_input['FWZJ'] / DF_COMBINED_PPSM

    avg_price_psm_m = df_input['avg_price_psm_d'].groupby(
        df_input['month']).sum() / df_input['JKHTBH'].groupby(
        df_input['month']).count()





    """MONTHLY DATA COMBINED"""
    ratio_combined_m = pd.concat([trading_vol_m,
                                  avg_trading_amt_total_m, med_trading_amt_m,
                                  avg_trading_garea_m, med_trading_garea_m,
                                  avg_trading_carea_m, med_trading_carea_m,
                                  avg_trading_amt_perm_m, avg_price_psm_m

                                  ], axis=1)
    ratio_combined_m.columns = ['vol_m',
                                'amt_avg_m', 'amt_med_m',
                                'garea_avg_m', 'garea_med_m',
                                'carea_avg_m', 'carea_med_m',
                                'perm_amt_m', 'avg_ppsm_m']
    scaled_m = ratio_combined_m.apply(lambda x:
                                      pd.Series(
                                          list(MinMaxScaler(feature_range=(0.001, 1.001)).fit_transform(
                                              x.values.reshape(len(ratio_combined_m), 1)
                                          ).reshape(len(ratio_combined_m))),
                                          index=ratio_combined_m.index),
                                      axis=0)

    # ratio_combined_m.to_csv(
    #     get_path(monthly_path+'ratio_combined\\'+area+'\\')
    #     +'ratio_combined_monthly.csv'
    # )
    # scaled_m.to_csv(
    #     get_path(
    #         monthly_path+'scaled_ratio_combined\\'+area+'\\'
    #     )
    #     +'scaled_ratio_combined_monthly.csv'
    # )
    df_scaled_tar = scaled_m[list_tar]
    weight_matrix = pd.read_csv(path_weight,index_col=0)

    weight_test = wa.simple_average(df_scaled_tar,
                                    get_path(monthly_path +
                                             'weighted_avg_ratio_combined\\' + area + '\\'),
                                    weight_matrix.T,
                                    df_temp_m_wa,
                                    0)
    weight_sum = weight_test[0]
    weight_detail = weight_test[1]
    weight_detail.columns = list_tar
    weight_fin = pd.DataFrame(
        MinMaxScaler(feature_range=(0.001, 1.001)).fit_transform(
            weight_sum.values.reshape(-1, 1)),index=weight_detail.index)
    """weight_fin is the comprehensive index after MinMaxScaled(feature=(0.001,1.001))"""
    weight_fin.columns = [area]
    spot_house_compreIndex = weight_fin.iloc[-1]
    df_spot_HcompreIndex = pd.DataFrame([spot_house_compreIndex],index=[weight_test[0].index[-1]])

    # weight_fin.to_csv(
    #     get_path(
    #         monthly_path+'weighted_avg_ratio_combined\\'+area+'\\'
    #     )
    #     +'weighted_fin.csv'
    # )
    return df_spot_HcompreIndex,weight_detail,weight_fin,weight_sum,ratio_combined_m
































