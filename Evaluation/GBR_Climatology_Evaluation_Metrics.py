# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 20:22:50 2025

@author: yue0004
"""

# This is the Python script to evaluate the accuracy of quantiles from climatology
# The climatology is based on the same folds as verification
# Last Modified on January 24, 2025
###############################################################################

#%% import packages
import pdb
import pandas as pd
import warnings
# import a customized module
import sys
sys.path.append('E:/USBR_Snow_Forecast/Fcst_Model')
import Utils
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import norm
from scipy.stats import gumbel_r
from mpl_toolkits.axes_grid1 import make_axes_locatable


#%% set global parameters
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

###############################################################################
###############################################################################
#%% define data frame to save results
###################################
# DF for test results with SWE
df_test_result_climo = pd.DataFrame(
    columns=['site_id', 'WY', 'value_10', 'value_30', 'value_50', 'value_70', 'value_90'])
# DF for mean QL with SWE
mean_QL_df_climo = pd.DataFrame(columns=['site_id', 'issue_date', 'QL'])
mean_QL_df_climo['issue_date'] = forecast_date


# DF for mean QL with SWE
mean_AllQL_df_climo = pd.DataFrame(columns=['site_id', 'issue_date', 'QL10', 'QL30', 'QL50', 'QL70', 'QL90'])
mean_AllQL_df_climo['issue_date'] = forecast_date

# DF for mean NSE with SWE
mean_NSE_df_climo = pd.DataFrame(columns=['site_id', 'issue_date', 'NSE'])
mean_NSE_df_climo['issue_date'] = forecast_date
# DF for mean RMSE with SWE
mean_RMSE_df_climo = pd.DataFrame(columns=['site_id', 'issue_date', 'RMSE'])
mean_RMSE_df_climo['issue_date'] = forecast_date
# DF for mean NRMSE with SWE
mean_NRMSE_df_climo = pd.DataFrame(columns=['site_id', 'issue_date', 'NRMSE'])
mean_NRMSE_df_climo['issue_date'] = forecast_date
# DF for interval coverage with SWE
mean_IC_df_climo = pd.DataFrame(columns=['site_id', 'issue_date', 'IC'])
mean_IC_df_climo['issue_date'] = forecast_date
# # DF for mean RPSS with SWE
# mean_RPSS_df_climo = pd.DataFrame(columns=['site_id', 'issue_date', 'RPSS'])
# mean_RPSS_df_climo['issue_date'] = forecast_date
# define the DF for use in for loop
mean_df_climo = pd.DataFrame(
    columns=['site_id', 'issue_date', 'value_10', 'value_30', 'value_50', 'value_70', 'value_90'])
mean_df_climo['issue_date'] = forecast_date


#########################################
# DF for observed/target values
target_all_df = pd.DataFrame(
    columns=['site_id', 'issue_date', 'volume'])
target_all_df['issue_date'] = forecast_date

# DF for target, climo
target_all_df_list = []
target_allyears_df_list = []
# QL
mean_QL_df_climo_list = []
# QL by quantiles
mean_AllQL_df_climo_list = []
# mean_QL_df_woswe_list = []
mean_NSE_df_climo_list = []
# mean_NSE_df_woswe_list = []
mean_RMSE_df_climo_list = []
# mean_RMSE_df_woswe_list = []
mean_NRMSE_df_climo_list = []
# mean_NRMSE_df_woswe_list = []
mean_IC_df_climo_list = []
# mean_IC_df_woswe_list = []
# mean_RPSS_df_climo_list = []
# mean_RPSS_df_woswe_list = []
mean_df_climo_list = []
# mean_df_woswe_list = []
result_df_climo_list = []
# result_df_woswe_list = []
target_train_df_list = []
site_id_short = site_id_short[site_id_short != 'DLI']

#%% Define funciton to calculate Quantile Loss (mathmetially same as function in Utilis)
def pinball_loss(y_true, y_pred, quantile):
    errors = y_true - y_pred
    # y true > y predicted
    # Compute loss using vectorized operations
    loss = np.where(errors >= 0, quantile * errors, (1 - quantile) * np.abs(errors))
    
    return np.mean(loss) 

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
    mean_QL_df_climo['site_id'] = site
    # mean_QL_df_woswe['site_id'] = site
    # DF for NSE
    mean_NSE_df_climo['site_id'] = site
    # mean_NSE_df_woswe['site_id'] = site
    # DF for RMSE
    mean_RMSE_df_climo['site_id'] = site
    # mean_RMSE_df_woswe['site_id'] = site
    # DF for NRMSE
    mean_NRMSE_df_climo['site_id'] = site
    # mean_NRMSE_df_woswe['site_id'] = site
    # DF for Interval coverage
    mean_IC_df_climo['site_id'] = site
    # mean_IC_df_woswe['site_id'] = site
    # # DF for RPSS
    # mean_RPSS_df_climo['site_id'] = site
    # mean_RPSS_df_woswe['site_id'] = site
    # # define variable needed for RPSS
    # rps_climo = 0
    # rps_woswe = 0
    # rps_climo = 0
    # extract results only for train year
    target_train_df = df_by_sites[site][df_by_sites[site]['WY'].isin(train_year)].reset_index(drop=True)
    # append to the list
    target_train_df_list.append(target_train_df)
    # loop by forecast issue date
    for date in forecast_date:
        # # import GBR results
        # # import GBR with SWE
        # result_df_climo = pd.read_csv('E:/USBR_Snow_Forecast/DATA/GBR/4fold_combined_with_SWE/%s_%s.csv' % (site, date))
        # result_df_climo['issue_date'] = date
        # # import GBR without SWE
        # result_df_woswe = pd.read_csv('E:/USBR_Snow_Forecast/DATA/GBR/4fold_combined_without_SWE/%s_%s.csv' % (site, date))
        # result_df_woswe['issue_date'] = date
        # import reforecasts based on climatology
        result_df_climo = pd.read_csv('E:/USBR_Snow_Forecast/DATA/GBR/4fold_combined_Climate/%s_%s.csv' % (site, date))
        result_df_climo['issue_date'] = date
        
        # organize the reference data
        NSF_df = Utils.slice_df(df_by_sites[site], 1982, 2021)
        target_df = NSF_df[NSF_df['WY'].isin(test_year)].reset_index(drop=True)
        target_df_temp = target_df.copy()
        target_df_temp['issue_date'] = date
        target_all_df.loc[target_all_df['issue_date'] == date, 'volume'] = target_df['volume'].mean()
        target_all_df.loc[target_all_df['issue_date'] == date, 'site_id'] = target_df['site_id_short'][0]
        # organize GBR results
        result_df_climo_list.append(result_df_climo)
        # result_df_woswe_list.append(result_df_woswe)
        target_allyears_df_list.append(target_df_temp)

        #################################################
        # Calculate mean QL
        #################################################
        # mean QL loss with swe (mean of 5 quantiles and 10 test years)
        mean_QL = Utils.mean_quantile_loss_eval(target_df['volume'],
                                                result_df_climo[
                                                    ['value_10', 'value_30', 'value_50', 'value_70', 'value_90']],
                                                quantiles)

        mean_QL_df_climo.loc[mean_QL_df_climo['issue_date'] == date, 'QL'] = mean_QL


        #################################################
        # Calculate mean QL by each quantile
        #################################################
        # mean QL loss with swe (mean of 40 test years)
        # QL for Q10
        mean_QL10 = pinball_loss(target_df['volume'],result_df_climo['value_10'],0.1)
        # QL for Q30
        mean_QL30 = pinball_loss(target_df['volume'],result_df_climo['value_30'],0.3)
        # QL for Q50
        mean_QL50 = pinball_loss(target_df['volume'],result_df_climo['value_50'],0.5)
        # QL for Q70
        mean_QL70 = pinball_loss(target_df['volume'],result_df_climo['value_70'],0.7)
        # QL for Q90
        mean_QL90 = pinball_loss(target_df['volume'],result_df_climo['value_90'],0.9)
        # save to df
        mean_AllQL_df_climo.loc[mean_AllQL_df_climo['issue_date'] == date, 'QL10'] = mean_QL10
        mean_AllQL_df_climo.loc[mean_AllQL_df_climo['issue_date'] == date, 'QL30'] = mean_QL30
        mean_AllQL_df_climo.loc[mean_AllQL_df_climo['issue_date'] == date, 'QL50'] = mean_QL50
        mean_AllQL_df_climo.loc[mean_AllQL_df_climo['issue_date'] == date, 'QL70'] = mean_QL70
        mean_AllQL_df_climo.loc[mean_AllQL_df_climo['issue_date'] == date, 'QL90'] = mean_QL90

        #######################################################
        # calculate NSE
        #######################################################
        # NSE with SWE
        NSE_climo = Utils.compute_NS(target_df.volume, result_df_climo['value_50'])
        mean_NSE_df_climo.loc[mean_NSE_df_climo['issue_date'] == date, 'NSE'] = NSE_climo
        
        # # NSE without SWE
        # NSE_woswe = Utils.compute_NS(target_df.volume, result_df_woswe['value_50'])
        # mean_NSE_df_woswe.loc[mean_NSE_df_woswe['issue_date'] == date, 'NSE'] = NSE_woswe

        ########################################################
        # calculate RMSE
        ########################################################
        # RMSE with SWE
        RMSE_climo = Utils.compute_RMSE(target_df.volume, result_df_climo['value_50'])
        mean_RMSE_df_climo.loc[mean_RMSE_df_climo['issue_date'] == date, 'RMSE'] = RMSE_climo
        
        # # RMSE without SWE
        # RMSE_woswe = Utils.compute_RMSE(target_df.volume, result_df_woswe['value_50'])
        # mean_RMSE_df_woswe.loc[mean_RMSE_df_woswe['issue_date'] == date, 'RMSE'] = RMSE_woswe
        ##########################################################
        # calculate NRMSE
        ###########################################################
        # NRMSE with SWE
        NRMSE_climo = Utils.compute_NRMSE(target_df.volume, result_df_climo['value_50'])
        mean_NRMSE_df_climo.loc[mean_NRMSE_df_climo['issue_date'] == date, 'NRMSE'] = NRMSE_climo
        
        # # RMSE without SWE
        # NRMSE_woswe = Utils.compute_NRMSE(target_df.volume, result_df_woswe['value_50'])
        # mean_NRMSE_df_woswe.loc[mean_NRMSE_df_woswe['issue_date'] == date, 'NRMSE'] = NRMSE_woswe
        
        ########################################################################
        # Calculate 80% confidence interval converage
        ########################################################################
        # with SWE
        IC_climo = Utils.calculate_interval_coverage(target_df.volume, result_df_climo['value_10'], result_df_climo['value_90'])
        mean_IC_df_climo.loc[mean_IC_df_climo['issue_date'] == date, 'IC'] = IC_climo
        # # without SWE
        # IC_woswe = Utils.calculate_interval_coverage(target_df.volume, result_df_woswe['value_10'], result_df_woswe['value_90'])
        # mean_IC_df_woswe.loc[mean_IC_df_woswe['issue_date'] == date, 'IC'] = IC_woswe
        
       
        #################### time series ##############################
        mean_climo_series = result_df_climo.select_dtypes(include=['number']).mean()
        mean_climo_df = pd.DataFrame(mean_climo_series, columns=['Mean']).transpose().drop('WY', axis=1)
        mean_climo_df.insert(0, 'site_id', site)
        mean_climo_df.insert(1, 'issue_date', date)
        mean_df_climo.loc[
            mean_df_climo['issue_date'] == date, ['site_id', 'value_10', 'value_30', 'value_50', 'value_70',
                                                    'value_90']] = \
            mean_climo_df[['site_id', 'value_10', 'value_30', 'value_50', 'value_70', 'value_90']].values

        # mean_woswe_series = result_df_woswe.select_dtypes(include=['number']).mean()
        # mean_woswe_df = pd.DataFrame(mean_woswe_series, columns=['Mean']).transpose().drop('WY', axis=1)
        # mean_woswe_df.insert(0, 'site_id', site)
        # mean_woswe_df.insert(1, 'issue_date', date)
        # mean_df_woswe.loc[
        #     mean_df_woswe['issue_date'] == date, ['site_id', 'value_10', 'value_30', 'value_50', 'value_70',
        #                                           'value_90']] = \
        #     mean_woswe_df[['site_id', 'value_10', 'value_30', 'value_50', 'value_70', 'value_90']].values
    # append all results togethee=r
    target_all_df_list.append(target_all_df.copy())
    # for results with climo
    mean_QL_df_climo_list.append(mean_QL_df_climo.copy())
    mean_AllQL_df_climo_list.append(mean_AllQL_df_climo.copy())
    mean_NSE_df_climo_list.append(mean_NSE_df_climo.copy())
    mean_RMSE_df_climo_list.append(mean_RMSE_df_climo.copy())
    mean_NRMSE_df_climo_list.append(mean_NRMSE_df_climo.copy())
    mean_IC_df_climo_list.append(mean_IC_df_climo.copy())
    # mean_RPSS_df_climo_list.append(mean_RPSS_df_climo.copy())
    # for results without SWE
    # mean_QL_df_woswe_list.append(mean_QL_df_woswe.copy())
    # mean_NSE_df_woswe_list.append(mean_NSE_df_woswe.copy())
    # mean_RMSE_df_woswe_list.append(mean_RMSE_df_woswe.copy())
    # mean_NRMSE_df_woswe_list.append(mean_NRMSE_df_woswe.copy())
    # mean_IC_df_woswe_list.append(mean_IC_df_woswe.copy())
    # mean_RPSS_df_woswe_list.append(mean_RPSS_df_woswe.copy())

    mean_df_climo_list.append(mean_df_climo.copy())
    # mean_df_woswe_list.append(mean_df_woswe.copy())

#%% save evaluation to local
# observed AMJJ total Q
target_df_all = pd.concat(target_all_df_list, axis=0, ignore_index=True)

pred_df_all_climo = pd.concat(result_df_climo_list, axis=0, ignore_index=True)
# pred_df_all_woswe = pd.concat(result_df_woswe_list, axis=0, ignore_index=True)
target_df_allyears = pd.concat(target_allyears_df_list, axis=0, ignore_index=True)
target_train_df_allyears = pd.concat(target_train_df_list, axis=0, ignore_index=True)
# save to local
# QL with climo
mean_QL_df_all_climo = pd.concat(mean_QL_df_climo_list, axis=0, ignore_index=True)
mean_QL_df_all_climo.to_csv(src_dir +'/results_single_basin_1982_2021/mean_QL_climo_all.csv', index=False)
# QL, all quantiles, with climo
mean_AllQL_df_all_climo = pd.concat(mean_AllQL_df_climo_list, axis=0, ignore_index=True)
mean_AllQL_df_all_climo.to_csv(src_dir +'/results_single_basin_1982_2021/mean_QL_AllQ_climo_all.csv', index=False)
# NSE with climo
mean_NSE_df_all_climo = pd.concat(mean_NSE_df_climo_list, axis=0, ignore_index=True)
mean_NSE_df_all_climo.to_csv(src_dir +'/results_single_basin_1982_2021/mean_NSE_climo_all.csv', index=False)
# RMSE with climo
mean_RMSE_df_all_climo = pd.concat(mean_RMSE_df_climo_list, axis=0, ignore_index=True)
mean_RMSE_df_all_climo.to_csv(src_dir +'/results_single_basin_1982_2021/mean_RMSE_climo_all.csv', index=False)
# NRMSE with climo
mean_NRMSE_df_all_climo = pd.concat(mean_NRMSE_df_climo_list, axis=0, ignore_index=True)
mean_NRMSE_df_all_climo.to_csv(src_dir +'/results_single_basin_1982_2021/mean_NRMSE_climo_all.csv', index=False)
# Interval coverage with climo
mean_IC_df_all_climo = pd.concat(mean_IC_df_climo_list, axis=0, ignore_index=True)
mean_IC_df_all_climo.to_csv(src_dir +'/results_single_basin_1982_2021/mean_IC_climo_all.csv', index=False)
# # RPSS with SWE
# mean_RPSS_df_all_climo = pd.concat(mean_RPSS_df_climo_list, axis=0, ignore_index=True)
# mean_RPSS_df_all_climo.to_csv(src_dir +'/results_single_basin_1982_2021/mean_RPSS_climo_all.csv', index=False)









