# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 17:34:37 2024

@author: yue0004
"""

# This is the Python script to calculate the QL, NSE, RMSE, and RPSS
# Last Modified on August 9, 2024
#########################################################################################

#%% import packages
#import pdb

import pandas as pd
# import warnings
# import a customized module
import sys
sys.path.append('E:/USBR_Snow_Forecast/Fcst_Model')
import Utils
# import seaborn as sns
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import matplotlib.lines as mlines
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# from scipy.stats import norm
# from scipy.stats import gumbel_r
# from mpl_toolkits.axes_grid1 import make_axes_locatable


#%% set global parameters
save_test_resutls = True
src_dir = "E:/USBR_Snow_Forecast/Fcst_Model"


#%% get target/observed value from train set
# Train set
metadata_path =  src_dir + '/data/metadata.csv'
metadata = pd.read_csv(metadata_path)
# Train set
target_tr_path = src_dir + '/data/train_1982_2022.csv'
target_df = pd.read_csv(target_tr_path)
target_df = target_df.rename(columns={'year': 'WY'})
target_df['site_id_short'] = target_df['site_id'].apply(
    lambda x: x[0].upper() + x.split('_')[1][0].upper() + x.split('_')[-1][0].upper() if isinstance(x, str) and len(
        x) > 0 else None)

#%% create a DF to store dataframes of results by site id
# Group the DataFrame by 'site_id'
# Create a dictionary to store DataFrames by site_id_short
df_by_sites = {}
# Group by 'site_id_short'
grouped_dataframes = target_df.groupby('site_id_short')
# Loop through each group
for site_id_short, group_df in grouped_dataframes:
    # Store the DataFrame in df_by_sites
    df_by_sites[site_id_short] = group_df
    # Drop NaN values in-place
    df_by_sites[site_id_short].dropna(inplace=True)
    
    
#%% define site id for interests   
site_id_short = target_df['site_id_short'].unique()  # site_id_short is an array
site_id_short = site_id_short[site_id_short != 'DLI']
monthly_NSF_tr_path = src_dir + '/data/train_monthly_naturalized_flow_1982_2022.csv'
# monthly_NSF_test_path = 'data/test_monthly_naturalized_flow.csv'
site_id_in_monthlyNSF = pd.read_csv(monthly_NSF_tr_path)['site_id'].unique()
site_id_short_in_monthlyNSF = [Utils.get_site_short_id(x, metadata_path) for x in site_id_in_monthlyNSF]
site_id_short_not_in_monthlyNSF = list(set(target_df['site_id_short'].unique()) - set(site_id_short_in_monthlyNSF))
site_id_not_in_monthlyNSF = list(set(target_df['site_id'].unique()) - set(site_id_in_monthlyNSF))   
# manully remove one site
site_id_short_in_monthlyNSF.remove('SRA')
    
#%% define forecast issue dates
# forecast date
forecast_date = []
days = ['01', '08', '15', '22']
# days = ['01']
months = ['01', '02', '03', '04', '05', '06', '07']
for month in months:
    for day in days:
        forecast_date.append(month + '-' + day)

#%% define training and test period (1982-2021)

start_year = 1982
end_year = 2021
year_list = list(range(start_year, end_year + 1))
quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]



#%%

#%% define data frame to save results
###################################
# DF for test results with SWE
df_test_result_withswe = pd.DataFrame(
    columns=['site_id', 'WY', 'value_10', 'value_30', 'value_50', 'value_70', 'value_90'])
# DF for mean QL with SWE
mean_QL_df_withswe = pd.DataFrame(columns=['site_id', 'issue_date', 'QL'])
mean_QL_df_withswe['issue_date'] = forecast_date
# DF for normalized QL with SWE
mean_NQL_df_withswe = pd.DataFrame(columns=['site_id', 'issue_date', 'NQL'])
mean_NQL_df_withswe['issue_date'] = forecast_date
# DF for mean NSE with SWE
mean_NSE_df_withswe = pd.DataFrame(columns=['site_id', 'issue_date', 'NSE'])
mean_NSE_df_withswe['issue_date'] = forecast_date
# DF for mean RMSE with SWE
mean_RMSE_df_withswe = pd.DataFrame(columns=['site_id', 'issue_date', 'RMSE'])
mean_RMSE_df_withswe['issue_date'] = forecast_date
# DF for mean NRMSE with SWE
mean_NRMSE_df_withswe = pd.DataFrame(columns=['site_id', 'issue_date', 'NRMSE'])
mean_NRMSE_df_withswe['issue_date'] = forecast_date
# DF for mean RPSS with SWE
mean_RPSS_df_withswe = pd.DataFrame(columns=['site_id', 'issue_date', 'RPSS'])
mean_RPSS_df_withswe['issue_date'] = forecast_date
# DF for interval coverage with SWE
mean_IC_df_withswe = pd.DataFrame(columns=['site_id', 'issue_date', 'IC'])
mean_IC_df_withswe['issue_date'] = forecast_date
# define the DF for use in for loop
mean_df_withswe = pd.DataFrame(
    columns=['site_id', 'issue_date', 'value_10', 'value_30', 'value_50', 'value_70', 'value_90'])
mean_df_withswe['issue_date'] = forecast_date

########################################
# DF for test results without SWE
df_test_result_woswe = pd.DataFrame(
    columns=['site_id', 'WY', 'value_10', 'value_30', 'value_50', 'value_70', 'value_90'])
# DF for mean QL without SWE
mean_QL_df_woswe = pd.DataFrame(columns=['site_id', 'issue_date', 'QL'])
mean_QL_df_woswe['issue_date'] = forecast_date
# DF for normalized QL without SWE
mean_NQL_df_woswe = pd.DataFrame(columns=['site_id', 'issue_date', 'NQL'])
mean_NQL_df_woswe['issue_date'] = forecast_date
# DF for mean NSE without SWE
mean_NSE_df_woswe = pd.DataFrame(columns=['site_id', 'issue_date', 'NSE'])
mean_NSE_df_woswe['issue_date'] = forecast_date
# DF for mean RMSE without SWE
mean_RMSE_df_woswe = pd.DataFrame(columns=['site_id', 'issue_date', 'RMSE'])
mean_RMSE_df_woswe['issue_date'] = forecast_date
# DF for mean NRMSE without SWE
mean_NRMSE_df_woswe = pd.DataFrame(columns=['site_id', 'issue_date', 'NRMSE'])
mean_NRMSE_df_woswe['issue_date'] = forecast_date
# DF for mean RPSS without SWE
mean_RPSS_df_woswe = pd.DataFrame(columns=['site_id', 'issue_date', 'RPSS'])
mean_RPSS_df_woswe['issue_date'] = forecast_date
# DF for interval coverage without SWE
mean_IC_df_woswe = pd.DataFrame(columns=['site_id', 'issue_date', 'IC'])
mean_IC_df_woswe['issue_date'] = forecast_date
# define the DF for use in for loop
mean_df_woswe = pd.DataFrame(
    columns=['site_id', 'issue_date', 'value_10', 'value_30', 'value_50', 'value_70', 'value_90'])
mean_df_woswe['issue_date'] = forecast_date


#########################################
# DF for observed/target values
target_all_df = pd.DataFrame(
    columns=['site_id', 'issue_date', 'volume'])
target_all_df['issue_date'] = forecast_date

# DF for target, with and without SWE 
target_all_df_list = []
target_allyears_df_list = []
mean_QL_df_withswe_list = []
mean_QL_df_woswe_list = []
mean_NQL_df_withswe_list = []
mean_NQL_df_woswe_list = []
mean_NSE_df_withswe_list = []
mean_NSE_df_woswe_list = []
mean_RMSE_df_withswe_list = []
mean_RMSE_df_woswe_list = []
mean_NRMSE_df_withswe_list = []
mean_NRMSE_df_woswe_list = []
mean_RPSS_df_withswe_list = []
mean_RPSS_df_woswe_list = []
mean_IC_df_withswe_list = []
mean_IC_df_woswe_list = []
mean_df_withswe_list = []
mean_df_woswe_list = []
result_df_withswe_list = []
result_df_woswe_list = []
target_train_df_list = []
site_id_short = site_id_short[site_id_short != 'DLI']


#%% Extract results by forecast site and organize the data 
# Remove DLI as it ends in June.
# Remove SRR and LRI as they are part in Canada.
site_id_short_in_monthlyNSF = [site for site in site_id_short_in_monthlyNSF if site not in ['DLI', 'SRR', 'LRI']]
train_year = list(range(1982, 2022))
test_year = list(range(1982, 2022))
# loop by forecast site
for site in site_id_short_in_monthlyNSF:
    # assign site id variable to DF
    # DF for QL
    mean_QL_df_withswe['site_id'] = site
    mean_QL_df_woswe['site_id'] = site
    # DF for NQL
    mean_NQL_df_withswe['site_id'] = site
    mean_NQL_df_woswe['site_id'] = site
    # DF for NSE
    mean_NSE_df_withswe['site_id'] = site
    mean_NSE_df_woswe['site_id'] = site
    # DF for RMSE
    mean_RMSE_df_withswe['site_id'] = site
    mean_RMSE_df_woswe['site_id'] = site
    # DF for NRMSE
    mean_NRMSE_df_withswe['site_id'] = site
    mean_NRMSE_df_woswe['site_id'] = site
    # DF for RPSS
    mean_RPSS_df_withswe['site_id'] = site
    mean_RPSS_df_woswe['site_id'] = site
    # define variable needed for RPSS
    rps_withswe = 0
    rps_woswe = 0
    rps_climo = 0
    # extract results only for train year
    target_train_df = df_by_sites[site][df_by_sites[site]['WY'].isin(train_year)].reset_index(drop=True)
    # append to the list
    target_train_df_list.append(target_train_df)
    # loop by forecast issue date
    for date in forecast_date:
        # import GBR results wih SWE
        result_df_withswe = pd.read_csv('E:/USBR_Snow_Forecast/DATA/GBR/4fold_combined_with_SWE/%s_%s.csv' % (site, date))
        result_df_withswe['issue_date'] = date
        # import GBR results without SWE
        result_df_woswe = pd.read_csv('E:/USBR_Snow_Forecast/DATA/GBR/4fold_combined_without_SWE/%s_%s.csv' % (site, date))
        result_df_woswe['issue_date'] = date
        # import reforecasts based on climatology
        result_df_refcst = pd.read_csv('E:/USBR_Snow_Forecast/DATA/GBR/4fold_combined_Climate/%s_%s.csv' % (site, date))
        result_df_refcst['issue_date'] = date
        # organize the reference data
        NSF_df = Utils.slice_df(df_by_sites[site], 1982, 2021)
        target_df = NSF_df[NSF_df['WY'].isin(test_year)].reset_index(drop=True)
        target_df_temp = target_df.copy()
        target_df_temp['issue_date'] = date
        target_all_df.loc[target_all_df['issue_date'] == date, 'volume'] = target_df['volume'].mean()
        target_all_df.loc[target_all_df['issue_date'] == date, 'site_id'] = target_df['site_id_short'][0]
        # organize GBR results
        result_df_withswe_list.append(result_df_withswe)
        result_df_woswe_list.append(result_df_woswe)
        target_allyears_df_list.append(target_df_temp)

        #################################################
        # Calculate mean QL
        #################################################
        # mean QL loss with swe (mean of 5 quantiles and 10 test years)
        mean_QL = Utils.mean_quantile_loss_eval(target_df['volume'],
                                                result_df_withswe[
                                                    ['value_10', 'value_30', 'value_50', 'value_70', 'value_90']],
                                                quantiles)

        mean_QL_df_withswe.loc[mean_QL_df_withswe['issue_date'] == date, 'QL'] = mean_QL


        # mean QL loss without swe (mean of 5 quantiles and 10 test years)
        mean_QL = Utils.mean_quantile_loss_eval(target_df['volume'],
                                                result_df_woswe[
                                                    ['value_10', 'value_30', 'value_50', 'value_70', 'value_90']],
                                                quantiles)

        mean_QL_df_woswe.loc[mean_QL_df_woswe['issue_date'] == date, 'QL'] = mean_QL
        
        #######################################################
        # Calculate mean QL relative to the observation, normalized QL
        #######################################################
        # mean normalized QL loss with swe 
        mean_QL = Utils.mean_quantile_loss_eval(target_df['volume'],
                                                result_df_withswe[
                                                    ['value_10', 'value_30', 'value_50', 'value_70', 'value_90']],
                                                quantiles)

        mean_NQL_df_withswe.loc[mean_NQL_df_withswe['issue_date'] == date, 'NQL'] = mean_QL/target_df['volume'].mean() * 100
        # mean normalized loss without swe
        mean_QL = Utils.mean_quantile_loss_eval(target_df['volume'],
                                                result_df_woswe[
                                                    ['value_10', 'value_30', 'value_50', 'value_70', 'value_90']],
                                                quantiles)

        mean_NQL_df_woswe.loc[mean_NQL_df_woswe['issue_date'] == date, 'NQL'] = mean_QL/target_df['volume'].mean() * 100

        #######################################################
        # calculate NSE
        #######################################################
        # NSE with SWE
        NSE_withswe = Utils.compute_NS(target_df.volume, result_df_withswe['value_50'])
        mean_NSE_df_withswe.loc[mean_NSE_df_withswe['issue_date'] == date, 'NSE'] = NSE_withswe
        
        # NSE without SWE
        NSE_woswe = Utils.compute_NS(target_df.volume, result_df_woswe['value_50'])
        mean_NSE_df_woswe.loc[mean_NSE_df_woswe['issue_date'] == date, 'NSE'] = NSE_woswe

        ########################################################
        # calculate RMSE
        ########################################################
        # RMSE with SWE
        RMSE_withswe = Utils.compute_RMSE(target_df.volume, result_df_withswe['value_50'])
        mean_RMSE_df_withswe.loc[mean_RMSE_df_withswe['issue_date'] == date, 'RMSE'] = RMSE_withswe
        
        # RMSE without SWE
        RMSE_woswe = Utils.compute_RMSE(target_df.volume, result_df_woswe['value_50'])
        mean_RMSE_df_woswe.loc[mean_RMSE_df_woswe['issue_date'] == date, 'RMSE'] = RMSE_woswe
        ##########################################################
        # calculate NRMSE
        ###########################################################
        # NRMSE with SWE
        NRMSE_withswe = Utils.compute_NRMSE(target_df.volume, result_df_withswe['value_50'])
        mean_NRMSE_df_withswe.loc[mean_NRMSE_df_withswe['issue_date'] == date, 'NRMSE'] = NRMSE_withswe
        
        # RMSE without SWE
        NRMSE_woswe = Utils.compute_NRMSE(target_df.volume, result_df_woswe['value_50'])
        mean_NRMSE_df_woswe.loc[mean_NRMSE_df_woswe['issue_date'] == date, 'NRMSE'] = NRMSE_woswe
        
        ##########################################################
        # calculate RPSS
        ##########################################################
        cdf_values_obs_withswe = Utils.calculate_cdf_values(result_df_withswe, quantiles, target_df['volume'])
        cdf_values_obs_woswe = Utils.calculate_cdf_values(result_df_woswe, quantiles, target_df['volume'])
        rps_withswe += np.sum((np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.0]) - cdf_values_obs_withswe) ** 2)
        rps_woswe += np.sum((np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.0]) - cdf_values_obs_woswe) ** 2)
    
        # Compute the RPS_climo
        # cdf_values_obs = Utils.calculate_cdf_climo(target_train_df['volume'], target_df['volume'])
        cdf_values_obs = Utils.calculate_cdf_values(result_df_refcst, quantiles, target_df['volume'])
        rps_climo += np.sum((np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.0]) - cdf_values_obs) ** 2)
        # calculate RPSS
        RPSS_withswe = 1 - rps_withswe / rps_climo
        RPSS_woswe = 1 - rps_woswe / rps_climo
        
        mean_RPSS_df_withswe.loc[mean_RPSS_df_withswe['issue_date'] == date, 'RPSS'] = RPSS_withswe
        mean_RPSS_df_woswe.loc[mean_RPSS_df_woswe['issue_date'] == date, 'RPSS'] = RPSS_woswe
       
        ########################################################################
        # Calculate 80% confidence interval converage
        ########################################################################
        # with SWE
        IC_withswe = Utils.calculate_interval_coverage(target_df.volume, result_df_withswe['value_10'], result_df_withswe['value_90'])
        mean_IC_df_withswe.loc[mean_IC_df_withswe.loc['issue_date'] == date, 'IC'] = IC_withswe
        # without SWE
        IC_woswe = Utils.calculate_interval_coverage(target_df.volume, result_df_woswe['value_10'], result_df_woswe['value_90'])
        mean_IC_df_woswe.loc[mean_IC_df_woswe.loc['issue_date'] == date, 'IC'] = IC_woswe
        
        #################### time series ##############################
        mean_withswe_series = result_df_withswe.select_dtypes(include=['number']).mean()
        mean_withswe_df = pd.DataFrame(mean_withswe_series, columns=['Mean']).transpose().drop('WY', axis=1)
        mean_withswe_df.insert(0, 'site_id', site)
        mean_withswe_df.insert(1, 'issue_date', date)
        mean_df_withswe.loc[
            mean_df_withswe['issue_date'] == date, ['site_id', 'value_10', 'value_30', 'value_50', 'value_70',
                                                    'value_90']] = \
            mean_withswe_df[['site_id', 'value_10', 'value_30', 'value_50', 'value_70', 'value_90']].values

        mean_woswe_series = result_df_woswe.select_dtypes(include=['number']).mean()
        mean_woswe_df = pd.DataFrame(mean_woswe_series, columns=['Mean']).transpose().drop('WY', axis=1)
        mean_woswe_df.insert(0, 'site_id', site)
        mean_woswe_df.insert(1, 'issue_date', date)
        mean_df_woswe.loc[
            mean_df_woswe['issue_date'] == date, ['site_id', 'value_10', 'value_30', 'value_50', 'value_70',
                                                  'value_90']] = \
            mean_woswe_df[['site_id', 'value_10', 'value_30', 'value_50', 'value_70', 'value_90']].values
    ####################################################################
    # append all results together
    ####################################################################
    target_all_df_list.append(target_all_df.copy())
    # for results with SWE
    mean_QL_df_withswe_list.append(mean_QL_df_withswe.copy())
    mean_NQL_df_withswe_list.append(mean_NQL_df_withswe.copy())
    mean_NSE_df_withswe_list.append(mean_NSE_df_withswe.copy())
    mean_RMSE_df_withswe_list.append(mean_RMSE_df_withswe.copy())
    mean_NRMSE_df_withswe_list.append(mean_NRMSE_df_withswe.copy())
    mean_RPSS_df_withswe_list.append(mean_RPSS_df_withswe.copy())
    mean_IC_df_withswe_list.append(mean_IC_df_withswe.copy())
    # for results without SWE
    mean_QL_df_woswe_list.append(mean_QL_df_woswe.copy())
    mean_NQL_df_woswe_list.append(mean_NQL_df_woswe.copy())
    mean_NSE_df_woswe_list.append(mean_NSE_df_woswe.copy())
    mean_RMSE_df_woswe_list.append(mean_RMSE_df_woswe.copy())
    mean_NRMSE_df_woswe_list.append(mean_NRMSE_df_woswe.copy())
    mean_RPSS_df_woswe_list.append(mean_RPSS_df_woswe.copy())
    mean_IC_df_woswe_list.append(mean_IC_df_woswe.copy())

    mean_df_withswe_list.append(mean_df_withswe.copy())
    mean_df_woswe_list.append(mean_df_woswe.copy())

#%% save evaluation to local
# observed AMJJ total Q
target_df_all = pd.concat(target_all_df_list, axis=0, ignore_index=True)

pred_df_all_withswe = pd.concat(result_df_withswe_list, axis=0, ignore_index=True)
pred_df_all_woswe = pd.concat(result_df_woswe_list, axis=0, ignore_index=True)
target_df_allyears = pd.concat(target_allyears_df_list, axis=0, ignore_index=True)
target_train_df_allyears = pd.concat(target_train_df_list, axis=0, ignore_index=True)
# save to local
# QL with SWE
mean_QL_df_all_withswe = pd.concat(mean_QL_df_withswe_list, axis=0, ignore_index=True)
mean_QL_df_all_withswe.to_csv('E:/USBR_Snow_Forecast/Results/mean_QL_withSWE_all.csv', index=False)
# NQL with SWE
mean_NQL_df_all_withswe = pd.concat(mean_NQL_df_withswe_list, axis=0, ignore_index=True)
mean_NQL_df_all_withswe.to_csv('E:/USBR_Snow_Forecast/Results/mean_NQL_withSWE_all.csv', index=False)
# NSE with SWE
mean_NSE_df_all_withswe = pd.concat(mean_NSE_df_withswe_list, axis=0, ignore_index=True)
mean_NSE_df_all_withswe.to_csv('E:/USBR_Snow_Forecast/Results/mean_NSE_withSWE_all.csv', index=False)
# RMSE with SWE
mean_RMSE_df_all_withswe = pd.concat(mean_RMSE_df_withswe_list, axis=0, ignore_index=True)
mean_RMSE_df_all_withswe.to_csv('E:/USBR_Snow_Forecast/Results/mean_RMSE_withSWE_all.csv', index=False)
# NRMSE with SWE
mean_NRMSE_df_all_withswe = pd.concat(mean_NRMSE_df_withswe_list, axis=0, ignore_index=True)
mean_NRMSE_df_all_withswe.to_csv('E:/USBR_Snow_Forecast/Results/mean_NRMSE_withSWE_all.csv', index=False)
# RPSS with SWE
mean_RPSS_df_all_withswe = pd.concat(mean_RPSS_df_withswe_list, axis=0, ignore_index=True)
mean_RPSS_df_all_withswe.to_csv('E:/USBR_Snow_Forecast/Results/mean_RPSS_withSWE_all.csv', index=False)
# Interval coverage with SWE
mean_IC_df_all_withswe = pd.concat(mean_IC_df_withswe_list, axis=0, ignore_index=True)
mean_IC_df_all_withswe.to_csv('E:/USBR_Snow_Forecast/Results/mean_IC_withSWE_all.csv', index=False)

# QL without SWE
mean_QL_df_all_woswe = pd.concat(mean_QL_df_woswe_list, axis=0, ignore_index=True)
mean_QL_df_all_woswe.to_csv('E:/USBR_Snow_Forecast/Results/mean_QL_withoutSWE_all.csv', index=False)
# NQL without SWE
mean_NQL_df_all_woswe = pd.concat(mean_NQL_df_woswe_list, axis=0, ignore_index=True)
mean_NQL_df_all_woswe.to_csv('E:/USBR_Snow_Forecast/Results/mean_NQL_withoutSWE_all.csv', index=False)
# NSE without SWE
mean_NSE_df_all_woswe = pd.concat(mean_NSE_df_woswe_list, axis=0, ignore_index=True)
mean_NSE_df_all_woswe.to_csv('E:/USBR_Snow_Forecast/Results/mean_NSE_withoutSWE_all.csv', index=False)
# RMSE without SWE
mean_RMSE_df_all_woswe = pd.concat(mean_RMSE_df_woswe_list, axis=0, ignore_index=True)
mean_RMSE_df_all_woswe.to_csv('E:/USBR_Snow_Forecast/Results/mean_RMSE_withoutSWE_all.csv', index=False)
# NRMSE without SWE
mean_NRMSE_df_all_woswe = pd.concat(mean_NRMSE_df_woswe_list, axis=0, ignore_index=True)
mean_NRMSE_df_all_woswe.to_csv('E:/USBR_Snow_Forecast/Results/mean_NRMSE_withoutSWE_all.csv', index=False)
# RPSS without SWE
mean_RPSS_df_all_woswe = pd.concat(mean_RPSS_df_woswe_list, axis=0, ignore_index=True)
mean_RPSS_df_all_woswe.to_csv('E:/USBR_Snow_Forecast/Results/mean_RPSS_withoutSWE_all.csv', index=False)


# mean_all_withswe = pd.concat(mean_df_withswe_list, axis=0, ignore_index=True)
# mean_all_woswe = pd.concat(mean_df_woswe_list, axis=0, ignore_index=True)