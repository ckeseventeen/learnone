import datetime
import json
import smtpd

import sys
import library_rnn
from library_rnn import *
import traceback

from pandas.testing import assert_frame_equal
import multiprocessing
import os

import numpy as np
import pulp

import pandas as pd
import statsmodels.tsa.stattools

from scipy import interpolate
from scipy.interpolate import *
import library as lib
from library import *
from scipy import stats
from scipy.stats import t
from scipy import optimize as opt
import pickle
from dateutil import relativedelta
from dateutil import rrule
from library_mort import *
import cpca
import jieba
import pandas._libs.tslibs.base
import jieba

# import sys
# import traceback
# from pyspark.sql import SparkSession
# from pyspark.sql.types import StringType,StructType,IntegerType,DoubleType,StructField





def get_num(txt_name):
    f_num = open(txt_name,'r+')
    a = f_num.read()
    a = int(a)+1
    f_num.seek(0)
    f_num.truncate()
    f_num.write(str(a))
    f_num.close()
    return str(a)

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



class mortgage():
    def __init__(self):
        with open('parameters.json','r',encoding='UTF-8') as f:
            self.parameters = json.load(f)

    # def start(self):
    #     print("THIS IS THE START OF THE MORTGAGE VALUATION MODEL")
    #     today = datetime.datetime.now()
    #     data = self.parameters['data']
    #     gb301 = pd.read_csv(data+"raw\\RAW_DATA\\"+'GB301.csv')  # for loan fwtnmj/fwjzmj
    #     ln773 = pd.read_csv(data+"raw\\RAW_DATA\\"+'LN773.csv')  # for loan fwtnmj/fwjzmj
    #     ln016 = pd.read_csv(data+'raw\\RAW_DATA\\'+'LN016.csv')
    #     ln003 = pd.read_csv(data+'raw\\RAW_DATA\\'+'LN003.csv')  # for pledge rate
    #     ln014 = pd.read_csv(data+'raw\\RAW_DATA\\'+'LN014.csv')  # for estate code
    #
    #     data = self.parameters['data']
    #     result = self.parameters['result']
    #     da = Date_Class()
    #     wa = weighted_average()
    #     fi = fillin()
    #     ha = houseAge_analysis()
    #     sf =scaled_factor()
    #     addr= address()
    #
    #     nominal = self.parameters['nominal']
    #     w_1 = self.parameters['tier']['tier_1']*nominal
    #     w_2 = self.parameters['tier']['tier_2']*nominal
    #     weight_tuple = (w_1,w_2)
    #     weight_matrix = pd.DataFrame(np.random.dirichlet(weight_tuple, 1).transpose().T)
    #     weight_matrix.columns=['tier_1','tier_2']
    #
    #
    #     raw_path = data + 'raw_from_db\\'
    #     raw_origin_cols = pd.read_csv(raw_path+'raw_frmdb.csv')
    #     raw_test_temp = raw_origin_cols
    #     # raw_test_temp = pd.read_csv(raw_path+'raw_frmdb.csv')
    #     raw_test_temp.columns=['LOANCONTRCODE','JKHTBH','LOANCONTRSTATE',
    #                            'DKQS','FWZL','FWJZMJ','FWTNMJ','FWXZ','FWZJ','LOANTYPE',
    #                            'GFSFK','HTDKJE','DKLL','DKDBLX','MONTHREPAYAMT','REMAINBAL',
    #                            'INSTCODE','INSTAREA','HOUSEAGE','INSTNAME','JKHTQDRQ',
    #                            'DYWPGJZ','ZYWPGJZ','DKYHBXH','COMMLOANAMT','LNREPAYMONTHAMT',
    #                            'COMMLOANTERM']
    #     raw_test=raw_test_temp[(raw_test_temp['FWJZMJ']<=300)&(raw_test_temp['DYWPGJZ']!=0)]
    #
    #     """dealing with date
    #     JKHTQDRQ date/month
    #     calculate the total contracts signed in a specific time period"""
    #
    #     date = raw_test['JKHTQDRQ'].apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d"))
    #     raw_test['index_date'] = date
    #
    #     month = raw_test['index_date'].apply(lambda x: datetime.datetime.strptime(
    #         str(x)[0:7], "%Y-%m"))
    #     raw_test['month'] = month
    #
    #     """
    #     to get the total house age
    #     """
    #     raw_test['yrsutil'] = raw_test['index_date'].apply(lambda x:
    #                                                        rrule.rrule(rrule.YEARLY, dtstart=x, until=today).count()-1)
    #     raw_test['tolyrsutil'] = raw_test['yrsutil']+raw_test['HOUSEAGE']
    #     raw_temp_2 = raw_test.reset_index().set_index('index_date').sort_index(ascending=True)
    #     raw_temp = raw_temp_2.drop(columns=['FWJZMJ','FWTNMJ'])
    #
    #     """DEALING WITH FWJZMJ/FWTNMJ
    #     focus on raw_temp;
    #     gb301_jkhtbh do not present duplicating jkhtbh
    #     """
    #
    #     jkhtbh_list = list(raw_temp['JKHTBH'].unique())  #  valid jkhtbh from raw_temp
    #     gb301_part = gb301.loc[:,['JKHTBH','FWJZMJ','FWTNMJ','YDFKRQ','DKHKFS']]
    #
    #     gb301_part_jkhtbh = gb301_part[gb301_part['JKHTBH'].isin(jkhtbh_list)]
    #
    #     test_combin = pd.merge(raw_temp.reset_index(),gb301_part_jkhtbh,how='inner',on='JKHTBH')   # 65128
    #     """TEST IF pd.merge IS GOING WRONG"""
    #
    #     test_merge_rawtemp = test_merge(raw_temp.reset_index(),gb301_part_jkhtbh,test_combin,'JKHTBH','JKHTBH')
    #
    #     FWMJ_from_testcombin = test_combin.loc[:,['JKHTBH','FWJZMJ','FWTNMJ','YDFKRQ','DKHKFS']]
    #
    #     ln773_part = ln773.loc[:, ['JKHTBH', 'FWJZMJ', 'FWTNMJ', 'YDFKRQ', 'DKHKFS']]
    #     """TEST ANOTHER METHOD TO FILTER FWJZMJ/FWTNMJ FROM LN773"""
    #     jkhtbh_all = pd.DataFrame(raw_temp['JKHTBH'].unique(),columns=['JKHTBH'])
    #     jkhtbh_outGB301 = jkhtbh_all[~(jkhtbh_all['JKHTBH'].isin(list(FWMJ_from_testcombin['JKHTBH'].unique())))]  # 23436
    #     jkhtbh_fromLN773 = ln773_part[ln773_part['JKHTBH'].isin(jkhtbh_outGB301['valid_jkhtbh_all'].unique())]  # 15593
    #
    #
    #
    #
    #
    #     """filtering out both (FWJZMJ==0 & FWTNMJ==0)"""
    #     ln773_part_drop_0 = ln773_part[~((ln773_part['FWJZMJ']==0)&(ln773_part['FWTNMJ']==0))]
    #     jkhtbh_fromLN773_2 = pd.merge(jkhtbh_outGB301, ln773_part_drop_0, how='inner', on='JKHTBH')  # 15082
    #
    #
    #     """LOCATING THE DIFFERENT JKHTBH"""
    #     jkhtbh_locate = list(set(list(jkhtbh_outGB301['JKHTBH'].unique())) ^ set(list(jkhtbh_fromLN773_2['JKHTBH'].unique())))
    #     df_jkhtbh_locate = pd.DataFrame(jkhtbh_locate)
    #     df_jkhtbh_locate.to_csv('jkhtbh_locate.csv')
    #
    #
    #     """checking jkhtbh duplication"""
    #     res_dupli_ln773_jkhtbh = get_dupli(ln773_part_drop_0['JKHTBH'],'JKHTBH')
    #
    #     """filtering in valid jkhtbh"""
    #
    #     ln773_part_jkhtbh = ln773_part_drop_0[(ln773_part_drop_0['JKHTBH'].isin(jkhtbh_list))]
    #     ln773_part_jkhtbh_aftertest_combin = ln773_part_jkhtbh[~(ln773_part_jkhtbh['JKHTBH'].isin(
    #         list(test_combin['JKHTBH'].unique())
    #     ))]
    #
    #     """combined FWJZMJ/FWTNMJ with key as jkhtbh from ln773 and gb301"""
    #     FWMJ_combin = pd.concat([FWMJ_from_testcombin,ln773_part_jkhtbh_aftertest_combin])
    #     raw_temp_fwmj = pd.merge(raw_temp.reset_index(),FWMJ_combin,how='inner',on='JKHTBH')
    #     test_merge_raw_fwmj = test_merge(raw_temp.reset_index(),
    #                                      FWMJ_combin,
    #                                      raw_temp_fwmj,
    #                                      'JKHTBH',
    #                                      'JKHTBH')
    #     # raw_temp_fwmj.to_csv(get_path(result+'data_check\\')+'raw_from_origin.csv')
    #
    #
    #
    #     raw_temp_fwmj['issue_date']=raw_temp_fwmj['YDFKRQ'].apply(
    #         lambda x:get_date_different(x)
    #     )
    #     raw_temp_fwmj['issue_date'].loc[raw_temp_fwmj['YDFKRQ'].isnull()] = \
    #         raw_temp_fwmj['index_date'].loc[raw_temp_fwmj['YDFKRQ'].isnull()]
    #
    #     """calculating mortgage rate"""
    #     raw_temp_fwmj['total_loanamt'] = raw_temp_fwmj['COMMLOANAMT'].fillna(0)+raw_temp_fwmj['HTDKJE']
    #     """DEALING WITH MORTGAGE--MORTGAGE RATE SHOULD BE APPLIED WITH EVERY LOANCONTRCODE"""
    #     # raw_temp_fwmj['mortgage_rate'] = raw_temp_fwmj['total_loanamt']/(raw_temp_fwmj['DYWPGJZ']+raw_temp_fwmj['ZYWPGJZ'])
    #     raw_temp_fwmj['mortgage_rate'] = raw_temp_fwmj['total_loanamt'] / (
    #                 raw_temp_fwmj['DYWPGJZ'])
    #
    #
    #
    #     raw_temp_fwmj.to_csv(
    #         get_path(
    #             result+'intermid_input\\'
    #         )
    #         +'raw_temp_fwmj.csv'
    #     )
    #
    #
    #     area_list = raw_temp_fwmj['INSTNAME'].unique()
    #
    #
    #
    #     """DEALING WITH HOUSE AGE ADJUSTING FACTOR"""
    #     w_list_houseAge=[10,20,30,40]
    #     df_houseAge = raw_temp_fwmj.loc[:,['tolyrsutil','LOANCONTRCODE','JKHTBH']]
    #     w_dirichlet_ha = list(np.random.dirichlet(tuple(w_list_houseAge), 1).transpose().reshape(
    #         len(w_list_houseAge)))
    #
    #
    #     """THE INPUT DATAFRAME TO CALCULATE SCALE"""
    #     """raw_temp_fwmj will be covered with tolyrsutil automatically"""
    #     raw_temp_fwmj_2 = ha.IQR_test(raw_temp_fwmj,w_dirichlet_ha,'tolyrsutil')
    #
    #     raw_temp_fwmj['range'] = pd.Series(range(len(raw_temp_fwmj)),
    #                                         index=raw_temp_fwmj.index)
    #     raw_temp_fwmj['end_month'] = raw_temp_fwmj['range'].apply(
    #         lambda x:raw_temp_fwmj['issue_date'].iloc[x] + relativedelta(
    #             months=raw_temp_fwmj['DKQS'].iloc[x]+1)
    #     )
    #     raw_temp_fwmj['first_repay'] = raw_temp_fwmj['issue_date'].apply(
    #         lambda x:x+relativedelta(months=1)
    #     )
    #
    #     """
    #     ZYWPGJZ
    #     """
    #
    #     raw_temp_fwmj['scaled_zywpgjz'] = pd.Series(
    #         list(MinMaxScaler(feature_range=(0, 1)).fit_transform(
    #             raw_temp_fwmj['ZYWPGJZ'].fillna(0).values.reshape(-1, 1)).reshape(len(raw_temp_fwmj))
    #              ), index=raw_temp_fwmj.index)
    #
    #
    #     """DEALING WITH THE ADDRESS PROBLEM"""
    #     df_address = raw_temp_fwmj.loc[:, ['INSTNAME','JKHTBH', 'LOANCONTRCODE', 'FWZL']]
    #
    #     """address key_words level 1"""
    #     addr_level_1 = self.parameters['address']['level_1']
    #     addr_level_2 = self.parameters['address']['level_2']
    #
    #     add_f = open(data+'address\\addresssource.txt',"r",encoding='UTF-8')
    #     add_words = []
    #     for line in add_f:
    #         add_words.append(line.strip())
    #
    #     # df_test = df_address.head(500)
    #
    #     test = addr.get_address_cpca(df_address,'FWZL',
    #                                  add_words,
    #                                  addr_level_2,
    #                                  addr_level_1)
    #
    #
    #     count_nums_l3 = self.parameters['address']['count_nums_l3']
    #     count_nums_l2 = self.parameters['address']['count_nums_l2']
    #     count_nums_l1 = self.parameters['address']['count_nums_l1']
    #
    #     tu_res_test = addr.get_addr_clusstering(test,
    #                                          ['JKHTBH','INSTNAME','FWZL','raw_address',
    #                                           'adr_l1','adr_l2','adr_l3','adr_l4'],
    #                                          count_nums_l3,
    #                                          count_nums_l2,
    #                                          count_nums_l1
    #                                          )
    #     """MERGE TO FORM THE FINAL INPUT DATAFRAME"""
    #
    #     df_input_final = pd.merge(raw_temp_fwmj, tu_res_test[0], how='outer', on='JKHTBH')
    #     df_input_final.to_csv(
    #         get_path(
    #             data+'final_input\\'
    #         )
    #         +'final_input.csv'
    #     )
    #
    #     print('...')

    def get_weight(self):
        """THIS IS THE DIRICHLET PROCESS TO GET THE DIFFERENT WEIGHT
        THIS FUNCTION WILL ONLY RUN FOR ONCE
        AND ALL WEIGHT DATA SOURCE COMES FROM THE SAME DIRICHLET PARAMETER"""
        data=self.parameters['data']
        w_1 = self.parameters['tier']['tier_1']
        w_2 = self.parameters['tier']['tier_2']
        nominal = self.parameters['nominal']
        weight_tuple = (w_1*nominal,w_2*nominal)
        weight_matrix = pd.DataFrame(np.random.dirichlet(weight_tuple, 1).transpose(),index=['tier_1','tier_2']).T
        weight_matrix.to_csv(
            get_path(data+'weight\\')
            +'weight_final.csv'
        )
        print('FINISH THE WEIGHT GET PROCESS')







        print('...')



    def mort_analysis(self):
        print("THIS IS THE MORTGAGE MODEL ANALYSIS USING SQL DATA FROM DBEAVER")
        data = self.parameters['data']
        result = self.parameters['result']
        date_now_input = datetime.datetime.strptime(self.parameters['end_date'],"%Y-%m-%d")
        ha = houseAge_analysis()
        sf = scaled_factor()
        input_path = get_path(data+'final_input\\')
        df_input_final = pd.read_csv(input_path+'final_input.csv',index_col=0)
        df_input_final['issue_date']= df_input_final['issue_date'].apply(
            lambda x:datetime.datetime.strptime(x,"%Y-%m-%d")
        )
        df_temp_m = pd.DataFrame([
            ['vol_m',
             'amt_avg_m', 'amt_med_m',
             'garea_avg_m', 'garea_med_m',
             'carea_avg_m', 'carea_med_m',
             'perm_amt_m','avg_ppsm_m','house_age_m'],
            ['tier_1',
             'tier_1','tier_1',
             'tier_2','tier_2',
             'tier_2','tier_2',
             'tier_1','tier_1','tier_2'],
            [1,1,1,1,1,1,1,1,1,-1]
        ],
            columns=range(10),
            index=['class','tier','direction']).T
        weight_matrix = pd.read_csv(get_path(data+'weight\\')+'weight_final.csv',index_col=0)
        path_weight_in = get_path(data+'weight\\')+'weight_final.csv'



        w_dirichlet_mr = list(np.random.dirichlet((10,20,30,40), 1).transpose().reshape(4))
        """mortgage_rate_adjusting_factor"""
        df_temp_mr= pd.DataFrame(
            [['tier_1','tier_2','tier_3','tier_4'],
            w_dirichlet_mr,
            [-1,-1,-1,-1]],index=['tier','weight','direction']
        ).T


        area_list = df_input_final['INSTNAME_x'].unique()
        list_tar = self.parameters['proxy']['monthly']
        df_temp_tar = df_temp_m[df_temp_m['class'].isin(list_tar)]
        time_span = self.parameters['time_span']
        res_all = pd.DataFrame()
        spot_hcmi_inst = pd.DataFrame()
        df_temp_output = df_temp_m[df_temp_m['class'].isin(list_tar)]
        weight_2 = weight_matrix.T.reset_index()
        weight_2.columns = ['tier','weight']
        df_temp_output_2 = pd.merge(df_temp_output,weight_2,on='tier',how='outer')
        df_weight_output = df_temp_output_2[['class','weight']]
        for i in range(len(area_list)):
            area_i = area_list[i]
            print(area_i)
            df_area_i = df_input_final[df_input_final['INSTNAME_x']==area_i]
            monthly_path_i = get_path(result+'monthly\\')

            spot_index = get_mortgage_factors(df_area_i,
                                                     get_path(monthly_path_i),
                                                     area_i,
                                                     list_tar,
                                                     path_weight_in,
                                                     df_temp_tar)
            anti_scaled_i = spot_index[4][list_tar]
            spot_HCompreIndex = spot_index[0]

            mom = spot_index[1].pct_change()
            factor_mom = mom.loc[spot_HCompreIndex.index]
            weight_detail = spot_index[1]
            weight_fin = spot_index[2]
            df_spot_hcmi_mom = weight_fin.pct_change()
            spot_hcmi_mom = df_spot_hcmi_mom.loc[spot_HCompreIndex.index]
            weight_sum = spot_index[3]
            spot_hcmi_inst = pd.concat([spot_hcmi_inst,weight_fin],axis=1)
            # weight_sum.to_csv(
            #     get_path(
            #         result+'instname\\weight\\'
            #     )
            #     +area_i+'_weight_sum.csv'
            # )

            categ_classi_addr = list(np.sort(df_area_i['concat_l3_l2_l1_l0'].unique()))

            spot_time_i = spot_HCompreIndex.index[0]
            res_cc_fin_i = pd.DataFrame()
            """DEALING WITH SPOT HOUSE COMPREHENSIVE INDEX"""
            for cc in range(len(categ_classi_addr)):
                caty = categ_classi_addr[cc]
                area_No_ = str(int(caty))
                print("THIS IS THE COMMUNITY CLASSIFICATION PROCESS WITH AREA No."+area_No_ +" IN INSTITUDE "+area_i)
                df_cc = df_area_i[df_area_i['concat_l3_l2_l1_l0']==caty]
                monthly_path_cc = result+'monthly\\'+area_No_+'\\'

                RES_res_cc = get_mortgage_factors(df_cc,
                                              monthly_path_cc,
                                              area_No_,
                                              list_tar,
                                            path_weight_in,
                                              df_temp_tar)

                anti_scaled_cc = RES_res_cc[4][list_tar]
                res_cc = RES_res_cc[0]
                mom_cc = RES_res_cc[1].pct_change()

                factor_mom_cc = mom_cc.loc[res_cc.index]
                detail_res_cc = RES_res_cc[1]
                weight_fin_cc = RES_res_cc[2]
                df_spot_hcmi_mom_cc = weight_fin_cc.pct_change()
                spot_hcmi_mom_cc = df_spot_hcmi_mom_cc.loc[res_cc.index]
                time_cc = res_cc.index[0]
                real_time_span = abs(datetime.datetime.strptime(spot_time_i,"%Y-%m-%d") -\
                                     datetime.datetime.strptime(time_cc,"%Y-%m-%d")).days
                if real_time_span < time_span:
                    res_temp = pd.DataFrame([res_cc[res_cc.columns[0]][0]],columns=['HCompreIndex'])
                    res_temp['last_transaction'] = pd.Series([res_cc.index[0]])
                    res_temp['sig_commu'] = 'yes'
                    res_temp['area_No'] = caty
                    res_temp['area_i'] = caty
                    for lt in range(len(list_tar)):
                        list_lt_cc = list_tar[lt]
                        list_lt_cc_mom = list_tar[lt]+'_mom'
                        list_lt_cc_weight = list_tar[lt]+'_weight'
                        list_lt_cc_antiS = list_tar[lt]+'_anti_scale'
                        res_temp[list_lt_cc] = detail_res_cc[list_lt_cc][-1]
                        res_temp[list_lt_cc_mom] = factor_mom[list_lt_cc][-1]
                        res_temp[list_lt_cc_weight] = \
                            df_weight_output[df_weight_output['class']==list_lt_cc]['weight'].iloc[0]
                        res_temp[list_lt_cc_antiS] = anti_scaled_cc[list_lt_cc].iloc[-1]
                    res_temp['spot_HCMI_mom'] = spot_hcmi_mom_cc.iloc[0][0]
                    res_cc_fin_i = pd.concat([res_cc_fin_i,res_temp],axis=0)


                else:
                    res_temp_2 = pd.DataFrame([spot_HCompreIndex[spot_HCompreIndex.columns[0]][0]],
                                              columns=['HCompreIndex'])
                    res_temp_2['last_transaction'] = pd.Series([spot_HCompreIndex.index[0]])
                    res_temp_2['sig_commu'] ='no'
                    res_temp_2['area_No'] = caty
                    res_temp_2['area_i'] = area_i
                    for lt in range(len(list_tar)):
                        list_lt_cc_2 = list_tar[lt]
                        list_lt_cc_mom_2 = list_tar[lt] + '_mom'
                        list_lt_cc_weight_2 = list_tar[lt]+'_weight'
                        list_lt_cc_antiS_2 = list_tar[lt]+'_anti_scale'
                        res_temp_2[list_lt_cc_2] = weight_detail[list_lt_cc_2][-1]
                        res_temp_2[list_lt_cc_mom_2] = factor_mom_cc[list_lt_cc_2][-1]
                        res_temp_2[list_lt_cc_weight_2] = \
                        df_weight_output[df_weight_output['class'] == list_lt_cc_2]['weight'].iloc[0]
                        res_temp_2[list_lt_cc_antiS_2] = anti_scaled_i[list_lt_cc_2].iloc[-1]
                    res_temp_2['spot_HCMI_mom'] = spot_hcmi_mom.iloc[0][0]
                    res_cc_fin_i = pd.concat([res_cc_fin_i,res_temp_2],axis=0)
            # res_cc_fin_i.to_csv(
            #     get_path(monthly_path_i+area_i+'\\total\\')
            #     +'res_HCompInx_fin_i.csv'
            # )
            res_cc_fin_i['concat_l3_l2_l1_l0'] = res_cc_fin_i['area_No']
            """THE DATAFRAME AFTER HOUSE VALUE COMPREHENSIVE INDEX"""
            df_HCompreInx_all = pd.merge(res_cc_fin_i,df_area_i,how='outer',on='concat_l3_l2_l1_l0')
            res_all = pd.concat([res_all, df_HCompreInx_all], axis=0)
        # res_all.to_csv(
        #     get_path(
        #         result+'intermid\\'
        #     )
        #     +'res_all_before_loanleft.csv'
        # )
        print('END THE RES_ALL CALCULATION')
        """CALCULATE THE SCALED REMAINED LOAN PRINCIPLE PLUS INTEREST"""
        print("AND START THE DCF PROCESS NOW")

        res_all['range_1'] = pd.Series(range(len(res_all)),index=res_all.index)

        res_temp_loanleft = res_all['range_1'].apply(
            lambda x:sf.scaled_factor_DCF(
                res_all['HTDKJE'].iloc[x],
                res_all['DKLL'].iloc[x],
                res_all['DKQS'].iloc[x],
                res_all['REMAINBAL'].iloc[x],
                int(res_all['DKHKFS'].iloc[x]),
                x,
                res_all['issue_date'].iloc[x],
                res_all['DKQS'].iloc[x],
                date_now_input
            )
        )

        res_all['scaled_totLoanLeft'] = res_temp_loanleft.apply(lambda x:x[0])
        # res_all['scaled_totLoanLeft'] = res_all['range_1'].apply(
        #     lambda x:sf.scaled_factor_DCF(
        #         res_all['HTDKJE'].iloc[x],
        #         res_all['DKLL'].iloc[x],
        #         res_all['DKQS'].iloc[x],
        #         res_all['REMAINBAL'].iloc[x],
        #         int(res_all['DKHKFS'].iloc[x]),
        #         x,
        #         res_all['issue_date'].iloc[x],
        #         res_all['DKQS'].iloc[x],
        #         date_now_input
        #     )[0]
        # )

        res_all['spot_house_compreIndex_R'] = res_all['HCompreIndex']
        mort_rate_Ajd_factor = ha.iqr_test_G(res_all,'mortgage_rate','mort_rate_adjusting',df_temp_mr)
        res_all['mort_rate_adjusting'] = mort_rate_Ajd_factor['weight_mort_rate_adjusting']

        res_all['spot_house_compreIndex_A'] = res_all['spot_house_compreIndex_R']* \
                                              (1+res_all['mort_rate_adjusting'])+\
                                                        res_all['scaled_zywpgjz']
        # res_all['spot_house_compreIndex_A'] = res_all['spot_house_compreIndex_R']*\
        #                                                 res_all['mortgage_rate']+\
        #                                                 res_all['scaled_zywpgjz']
        res_all['spot_house_compreInde_afterhagAdju'] = res_all['spot_house_compreIndex_A']* \
                                                                  (res_all['weight_hag']+1)
        res_all['risk_res'] = pd.Series(index=res_all.index)
        res_all['risk_res'].loc[
            res_all['scaled_totLoanLeft']<res_all['spot_house_compreInde_afterhagAdju']
        ]='NoRiskWarning'
        res_all['risk_res'].loc[
            res_all['scaled_totLoanLeft']>res_all['spot_house_compreInde_afterhagAdju']
        ]='RiskWarning'


        res_all['loan_left_mom_abs'] = res_temp_loanleft.apply(lambda x:abs(x[1]))

        # res_all['loan_left_mom'] = res_all['range_1'].apply(
        #     lambda x:sf.scaled_factor_DCF(
        #         res_all['HTDKJE'].iloc[x],
        #         res_all['DKLL'].iloc[x],
        #         res_all['DKQS'].iloc[x],
        #         res_all['REMAINBAL'].iloc[x],
        #         int(res_all['DKHKFS'].iloc[x]),
        #         x,
        #         res_all['issue_date'].iloc[x],
        #         res_all['DKQS'].iloc[x],
        #         date_now_input
        #     )[1]
        # )



        """CONCAT ALL DATA TO FORM THE FINAL COMPLETE RES DATAFRAME"""
        df_risk = res_all[res_all['risk_res']=='RiskWarning']
        df_risk['IdxDiff'] = df_risk['scaled_totLoanLeft'] - df_risk['spot_house_compreInde_afterhagAdju']
        df_after_risklevel = ha.IQR_G(df_risk,
                                      'IdxDiff',
                                      'risk_level',
                                      ['low risk','low to median risk','median to high risk','high risk'])
        df_risk['vol_diagnose'] = pd.Series(index=df_risk.index)
        df_risk['vol_diagnose'].loc[df_risk['vol_m_mom']<0]='lower'
        df_risk['amt_diagnose'] = pd.Series(index=df_risk.index)
        df_risk['amt_diagnose'].loc[df_risk['amt_avg_m_mom'] <0]='lower'
        df_risk['carea_diagnose'] = pd.Series(index=df_risk.index)
        df_risk['carea_diagnose'].loc[df_risk['carea_avg_m_mom'] <0]='lower'
        df_risk['perm_diagnose'] = pd.Series(index=df_risk.index)
        df_risk['perm_diagnose'].loc[df_risk['perm_amt_m_mom'] <0]='lower'


        list_tar_antiS = [i+'_anti_scale' for i in list_tar]
        list_tar_mom = [i+'_mom' for i in list_tar]
        list_tar_weight = [i+'_weight' for i in list_tar]
        list_diagnose = ['vol_diagnose','amt_diagnose','carea_diagnose','perm_diagnose']
        tar_list_show = self.parameters['show_items']+\
                        list_tar+list_tar_antiS+\
                        list_tar_mom+\
                        list_tar_weight+\
                        list_diagnose+['spot_HCMI_mom','loan_left_mom_abs']

        final_output = df_after_risklevel[tar_list_show]
        final_output.to_csv(
            get_path(
                result+'final_output\\'
            )
            +'final_output_risk.csv'
        )
        spot_hcmi_inst.to_csv(
            get_path(
                result+'HCMI\\'
            )
            +'spot_hcmi_instname.csv'
        )

        col_nums = len(tar_list_show)
        try:
            ##spark写hive表
            # 定义hive表的结构体
            schema = []
            for cn in range(col_nums):
                schema_list = StructField("column"+ str(cn), IntegerType(), True)
                schema.append(schema_list)
            schema = StructType([
                    schema
            ])
            # 创建sparkDataFrame (对于非表结构的需要按照这个方式创建新的dataframe)
            # sdf = spark.createDataFrame(df,schema=schema)
            # 通过sparkDataFrame写入hive表
            final_output.write.saveAsTable('rl.guar_worth_predict', format='Hive', mode='append')

        except:
            error_detail = traceback.format_exc()
            print("ERROR:\n")
            print(error_detail)
            sys.exit(-1)




        print("....")

    def data_hive_test(self):
        ha = houseAge_analysis()
        addr = address()
        data = self.parameters['data']
        # result = self.parameters['result']
        today = datetime.datetime.now()

        table_name = 'hzfl.fl_guar_worth_predict'
        spark = SparkSession \
            .builder \
            .appName("PySpark") \
            .config("spark.sql.execution.arrow.enabled", "true") \
            .config("spark.default.parallelism", 50) \
            .enableHiveSupport() \
            .getOrCreate()
        #执行sparksql，并且获取数据到dataframe中
        df = spark.sql("select * from "+table_name)
        #数据展示
        df.show()





        # 获取返回信息


        test_data = df

        # data = self.parameters['data']
        # result = self.parameters['result']
        # path = data+'data_hive_test\\'
        # test_data = pd.read_csv(path+'data_from_hive.csv')
        #
        # raw_path = data + 'raw_from_db\\'
        # raw_origin_cols = pd.read_csv(raw_path+'raw_frmdb.csv')
        # col_name_test = pd.DataFrame(test_data.columns,columns=['hive_test'])
        # col_name_origin_cols = pd.DataFrame(raw_origin_cols.columns,columns=['raw_chinese'])
        # """THIS IS THE FINAL INPUT"""
        # raw_from_origin = pd.read_csv(result+'data_check\\raw_from_origin.csv',index_col=0)
        # col_raw_origin = pd.DataFrame(raw_from_origin.columns,columns=['origin_data'])
        # # col_tar = ['LOANCONTRCODE','JKHTBH','LOANCONTRSTATE','DKQS','FWZL',
        # #            'FWXZ','FWJZ','LOANTYPE','GFSFK','HTDKJE','DKLL','DKDBLX',
        # #            'MONTHREPAYAMT','REMAINBAL','INSTCODE','INSTAREA','HOUSEAGE',
        # #            'INSTNAME','JKHTQDRQ','DYWPGJZ','ZYWPGJZ','DKYHBXH',
        # #            'COMMLOANAMT','LNREPAYMONTHAMT','COMMLOANTERM','FWJZMJ',
        # #            'FWTNMJ','YDFKRQ']


        test_data_rename = test_data.rename(
            columns={'loan_contr_state':'LOANCONTRSTATE',
                     'gb_loan_contr_code':'JKHTBH',
                     'loan_contr_code':'LOANCONTRCODE',
                     'loan_period':'DKQS',
                     'inst_code':'INSTCODE',
                     'inst_name':'INSTNAME',
                     'the_house_area':'INSTAREA',
                     'prop_age':'HOUSEAGE',
                     'contr_sign_date':'JKHTQDRQ',
                     'prop_addr':'FWZL',
                     'prop_nature':'FWXZ',
                     'prop_buid_area':'FWJZMJ',
                     'prop_inside_area':'FWTNMJ',
                     'agreed_disb_date':'YDFKRQ',
                     'prop_total_worth':'FWZJ',
                     'loan_type':'DKLX',
                     'down_payment':'GFSFK',
                     'loan_amt':'HTDKJE',
                     'loan_rate':'DKLL',
                     'loan_guar_type':'DKDBLX',
                     'month_repay_amt':'MONTHREPAYAMT',
                     'loan_balance':'REMAINBAL',
                     'collat_estimate_worth':'DYWPGJZ',
                     'ple_worth':'ZYWPGJZ',
                     'loan_repay_total_prin_int':'DKYHBXH',
                     'comm_loan_amt':'COMMLOANAMT',
                     'loan_repay_month_amt':'LNREPAYMONTHAMT',
                     'comm_loan_ter':'COMMLOANTERM',
                     'repay_method':'DKHKFS'
                     }
        )


        raw_test = test_data_rename[(test_data_rename['FWJZMJ'] <= 300) & (test_data_rename['DYWPGJZ'] != 0)]
        """dealing with date
        JKHTQDRQ date/month 
        calculate the total contracts signed in a specific time period"""

        date = raw_test['JKHTQDRQ'].apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d"))
        raw_test['index_date_2'] = date
        raw_test['index_date'] = date

        month = raw_test['index_date'].apply(lambda x: datetime.datetime.strptime(
            str(x)[0:7], "%Y-%m"))
        raw_test['month'] = month
        """
        to get the total house age
        """
        raw_test['yrsutil'] = raw_test['index_date'].apply(lambda x:
                                                           rrule.rrule(rrule.YEARLY,
                                                                       dtstart=x,
                                                                       until=today).count()-1)
        raw_test['tolyrsutil'] = raw_test['yrsutil']+raw_test['HOUSEAGE']

        raw_temp_fwmj = raw_test.reset_index().set_index('index_date_2').sort_index(ascending=True)

        raw_temp_fwmj['issue_date'] = raw_temp_fwmj['YDFKRQ'].apply(
            lambda x: get_date_different(x)
        )
        raw_temp_fwmj['issue_date'].loc[raw_temp_fwmj['YDFKRQ'].isnull()] = \
            raw_temp_fwmj['index_date'].loc[raw_temp_fwmj['YDFKRQ'].isnull()]

        """calculating mortgage rate"""
        raw_temp_fwmj['total_loanamt'] = raw_temp_fwmj['COMMLOANAMT'].fillna(0) + raw_temp_fwmj['HTDKJE']
        """DEALING WITH MORTGAGE--MORTGAGE RATE SHOULD BE APPLIED WITH EVERY LOANCONTRCODE"""
        raw_temp_fwmj['mortgage_rate'] = raw_temp_fwmj['total_loanamt'] / (
                    raw_temp_fwmj['DYWPGJZ'])




        # raw_temp_fwmj.to_csv(
        #         get_path(
        #           result+'intermid_input\\'
        #     )
        #     + 'raw_temp_fwmj.csv'
        # )

        area_list = raw_temp_fwmj['INSTNAME'].unique()

        """DEALING WITH HOUSE AGE ADJUSTING FACTOR"""
        w_list_houseAge = [10, 20, 30, 40]
        df_houseAge = raw_temp_fwmj.loc[:, ['tolyrsutil', 'LOANCONTRCODE', 'JKHTBH']]
        w_dirichlet_ha = list(np.random.dirichlet(tuple(w_list_houseAge), 1).transpose().reshape(
            len(w_list_houseAge)))

        """THE INPUT DATAFRAME TO CALCULATE SCALE"""
        """raw_temp_fwmj will be covered with tolyrsutil automatically"""
        raw_temp_fwmj_2 = ha.IQR_test(raw_temp_fwmj, w_dirichlet_ha, 'tolyrsutil')

        raw_temp_fwmj['range'] = pd.Series(range(len(raw_temp_fwmj)),
                                           index=raw_temp_fwmj.index)
        raw_temp_fwmj['end_month'] = raw_temp_fwmj['range'].apply(
            lambda x: raw_temp_fwmj['issue_date'].iloc[x] + relativedelta(
                months=raw_temp_fwmj['DKQS'].iloc[x] + 1)
        )
        raw_temp_fwmj['first_repay'] = raw_temp_fwmj['issue_date'].apply(
            lambda x: x + relativedelta(months=1)
        )

        """
        ZYWPGJZ
        """

        raw_temp_fwmj['scaled_zywpgjz'] = pd.Series(
            list(MinMaxScaler(feature_range=(0, 1)).fit_transform(
                raw_temp_fwmj['ZYWPGJZ'].fillna(0).values.reshape(-1, 1)).reshape(len(raw_temp_fwmj))
                 ), index=raw_temp_fwmj.index)

        """DEALING WITH THE ADDRESS PROBLEM"""
        df_address = raw_temp_fwmj.loc[:, ['INSTNAME', 'JKHTBH', 'LOANCONTRCODE', 'FWZL']]

        """address key_words level 1"""
        addr_level_1 = self.parameters['address']['level_1']
        addr_level_2 = self.parameters['address']['level_2']

        add_f = open(get_path(data+'address\\addresssource.txt'), "r", encoding='UTF-8')
        add_words = []
        for line in add_f:
            add_words.append(line.strip())



        test = addr.get_address_cpca(df_address, 'FWZL',
                                     add_words,
                                     addr_level_2,
                                     addr_level_1)

        count_nums_l3 = self.parameters['address']['count_nums_l3']
        count_nums_l2 = self.parameters['address']['count_nums_l2']
        count_nums_l1 = self.parameters['address']['count_nums_l1']

        tu_res_test = addr.get_addr_clusstering(test,
                                                ['JKHTBH', 'INSTNAME', 'FWZL', 'raw_address',
                                                 'adr_l1', 'adr_l2', 'adr_l3', 'adr_l4'],
                                                count_nums_l3,
                                                count_nums_l2,
                                                count_nums_l1
                                                )
        """MERGE TO FORM THE FINAL INPUT DATAFRAME"""

        df_input_final = pd.merge(raw_temp_fwmj, tu_res_test[0], how='outer', on='JKHTBH')
        df_input_final.to_csv(
            get_path(data+
                 'final_input\\'
            )
            + 'final_input.csv'
        )






        print('...')





    def RNNwithoutMultiprocess(self):
        print("start the RNN forecasting in multiprocessing WITHOUT MULTIPROCESS.POOL")
        start = datetime.datetime.now()
        print(start)
        de = library_rnn.drop_extreme()
        rnn_class = library_rnn.RNN_Class()
        deeplearning_mod =0
        result = self.parameters['result']
        data = self.parameters['data']
        date_mod='month'


        denomi = 5
        # LSTM 参数组合
        units_list_lstm = [15,20,25]
        dr_list_lstm = [0.01,0.05]
        epoch_list_lstm = [5,10]
        batch_size_lstm = [2,5,10]
        overlap_lstm = [3,5,6]
        loop_lstm = [units_list_lstm, dr_list_lstm, epoch_list_lstm, batch_size_lstm,overlap_lstm]
        ds_params_LSTM = loop_val(loop_lstm)

        # RNN 参数组合
        units_list_rnn = [5,10,15,20,25,30,35,40]
        dr_list_rnn = [0.01,0.05,0.1]
        epoch_list_rnn = [5,10]
        bs_list_rnn = [2,6]
        overlap = [3,5,6]
        loop_val_ = [units_list_rnn, dr_list_rnn, epoch_list_rnn, bs_list_rnn, overlap]
        ds_params_RNN = loop_val(loop_val_)



        path_tar = result+'HCMI\\'
        file_list = os.listdir(path_tar)





        for f in range(len(file_list)):
            file_f = file_list[f]
            data_f_2 = pd.read_csv(path_tar+file_f,index_col=0)
            data_f_2 = pd.DataFrame(data_f_2['惠城管理部'])
            data_f_ = Date_Class.index_type_swith(self,data_f_2['惠城管理部'],0)
            columns_f = data_f_.columns
            test_f = data_f_.reset_index()
            test_f.columns = ['index', columns_f[0]]
            data_f = test_f.set_index('index')  # data_f是最终读取进来的值



            if date_mod =='month':

                EX =6
                rate = data_f.pct_change().dropna()
                rate_noEXt = de.drop_extreme(rate,EX)
                # test_2是原始值而非差分值
                test_2 = de.drop_Extreme_rate(rate,rate_noEXt,data_f)

                moc_data_precs = test_2.iloc[0:len(test_2)-12]



                for cols in moc_data_precs.columns:
                    ds = moc_data_precs[cols]
                    print(cols)

                    # 去掉pool_num的for循环

                    if deeplearning_mod == 0:
                        print("start the RNN multiprocess")
                        res_rnn = pd.DataFrame()
                        res_pm = pd.DataFrame()
                        res_history = pd.DataFrame()
                        for param_num in range(len(ds_params_RNN)):
                            param_n = ds_params_RNN[param_num]

                            rnn_total_pn=rnn_class.RNN_HYPER(ds,denomi,param_n)
                            y_predict_pn = pd.DataFrame(rnn_total_pn[1])
                            y_predict_pn.columns=[str(param_n)]
                            res_rnn = pd.concat([res_rnn,y_predict_pn],axis=1)

                            df_pm_pn_2 = pd.DataFrame(rnn_total_pn[0])
                            df_pm_pn = df_pm_pn_2.set_index('param_comb')
                            res_pm = pd.concat([res_pm,df_pm_pn],axis=0)
                            history_2 = pd.DataFrame([[list(rnn_total_pn[2]['train_loss']),list(rnn_total_pn[2]['test_loss'])]])
                            history_2['param_com'] = pd.Series(str(param_n))
                            history = history_2.set_index('param_com')
                            res_history = pd.concat([res_history,history],axis=0)



                        res_pm.columns=['mae','mse','rmse','nrmse']

                        best_param = get_best_param(res_pm,cols)


                        res_history.to_csv(get_path(result+'history\\')+'his_all.csv')
                        method = 'bpnr'
                        best_hyperparam = best_param.loc[method]
                        best_prediction = res_rnn[best_hyperparam]
                        # df_best_prediction = pd.concat([ds.loc[best_prediction.index],best_prediction],axis=1)
                        best_prediction.to_csv(get_path(result+'best_prediction\\'+'best_prediction.csv'))


                    if deeplearning_mod == 1:
                        print("start the lstm multiprocess")
                        res_lstm = pd.DataFrame()
                        res_pm_lstm = pd.DataFrame()
                        res_history_lstm = pd.DataFrame()
                        for pl in range(len(ds_params_LSTM)):
                            param_l = ds_params_LSTM[pl]
                            lstm_total= rnn_class.LSTM_HYPER(ds, denomi, param_l)
                            y_predict_lstm = pd.DataFrame(lstm_total[1])
                            y_predict_lstm.columns = [str(param_l)]
                            res_lstm = pd.concat([res_lstm, y_predict_lstm], axis=1)

                            df_pm_lstm_2 = pd.DataFrame(lstm_total[0])
                            df_pm_lstm = df_pm_lstm_2.set_index('param_comb')
                            res_pm_lstm = pd.concat([res_pm_lstm, df_pm_lstm], axis=0)
                            history_2_lstm = pd.DataFrame(
                                [[list(lstm_total[2]['train_loss']), list(lstm_total[2]['test_loss'])]])
                            history_2_lstm['param_comb'] = pd.Series(str(param_l))
                            history_lstm = history_2_lstm.set_index('param_comb')



                            res_history_lstm = pd.concat([res_history_lstm, history_lstm], axis=0)

                        res_pm_lstm.columns = ['mae', 'mse', 'rmse', 'nrmse']

                        best_param_lstm = get_best_param(res_pm_lstm, cols)


                        res_history_lstm.to_csv(get_path(result + 'history_lstm\\') + 'his_all.csv')
                        method_lstm = 'bpnr'
                        best_hyperparam_lstm = best_param_lstm.loc[method_lstm]
                        best_prediction_lstm = res_lstm[best_hyperparam_lstm]
                        # df_best_prediction_lstm = pd.concat([ds.loc[best_prediction_lstm.index], best_prediction_lstm], axis=1)
                        best_prediction_lstm.to_csv(get_path(result + 'best_prediction_lstm\\' + 'best_predcition.csv'))

        end = datetime.datetime.now()
        time_need = end-start
        print("The time need for multiprocess RNN is %s" % str(time_need))


