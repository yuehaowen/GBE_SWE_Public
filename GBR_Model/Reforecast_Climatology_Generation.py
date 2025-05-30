# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 19:17:17 2024

@author: yue0004
"""

# This is the Python script to generate ensemble forecasts based on historical records
# Last Modified on August 12, 2024
#########################################################################

#%% import libraries
import pdb
import pandas as pd
import warnings
# import a customized module
import sys
sys.path.append('E:/USBR_Snow_Forecast/Fcst_Model')
import Utils
import seaborn as sns
import numpy as np

#%% set global variable
save_test_resutls = True
src_dir = "E:/USBR_Snow_Forecast/Fcst_Model"

#%% import complete training data set
forecast_date = []
days = ['01', '08', '15', '22']
# days = ['01']
months = ['01', '02', '03', '04', '05', '06', '07']
for month in months:
    for day in days:
        forecast_date.append(month + '-' + day)
        
# define the spatial cases dates for May, June, and July
May_dates = ['05-01', '05-08', '05-15', '05-22']
June_dates = ['06-01', '06-08', '06-15', '06-22']
July_dates = ['07-01', '07-08', '07-15', '07-22']


#%% define training period and validation period
folds = [
    ((1992, 2021, 1982, 1991)),
    (((1982, 1991), (2002, 2021), 1992, 2001)),
    (((1982, 2001), (2012, 2021), 2002, 2011)),
    ((1982, 2011, 2012, 2021))
]



#%% Set train set parameters
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
# Remove DLI as it ends in June.
# Remove SRR and LRI as they are part in Canada.
site_id_short_in_monthlyNSF = [site for site in site_id_short_in_monthlyNSF if site not in ['DLI', 'SRR', 'LRI']]

#%%
# get monthly NSF for April, May, June and July
# import
Monthly_NSF_AMJJ = pd.read_csv('E:/USBR_Snow_Forecast/DATA/Naturalized_Flow/Train_Monthly_AMJJ_NSF_HY.csv')
# get short site id
Monthly_NSF_AMJJ['site_short_ID']=[Utils.get_site_short_id(x, metadata_path) for x in Monthly_NSF_AMJJ['site_id']]

#%% define data frame to save results

df_refcst_result = pd.DataFrame(columns=['site_id', 'WY', 'value_10', 'value_30', 'value_50', 'value_70', 'value_90'])

#%% generate reforecast based on climatology
# loop by forecast site
for site in site_id_short_in_monthlyNSF:
    print('==== ', site)

    # loop by forecast date
    for date in forecast_date:
        print(f'============= forecast date: {date} ===============')
        # define dara frame to save results
        df_refcst_result_list = [] 
        for fold in folds: 
            #############################################
            # initial data frame to save results
            # extract the observed value by site
            NSF_df = Utils.slice_df(df_by_sites[site], 1982, 2021)
            # seperate train and test data set by pre-defined folds
            train_year, test_year, NSF_train_df, NSF_test_df = Utils.get_fold_data_by_year(NSF_df, fold)
            # initial data frame
            df_refcst_result=pd.DataFrame(columns=['site_id', 'WY', 'value_10', 'value_30', 'value_50', 'value_70', 'value_90'])
            # add value to data frame
            df_refcst_result['site_id'] = np.repeat(NSF_df['site_id'].unique(),10)
            df_refcst_result['WY'] = test_year
            ##############################################
            # Extract monthly value for each month for the test period (10 years)
            # Extract monthly NSF for April (test period)
            Site_NSF_Apr_test = Monthly_NSF_AMJJ[(Monthly_NSF_AMJJ['site_short_ID'] == site) & (Monthly_NSF_AMJJ['WY'].isin(test_year)) & (Monthly_NSF_AMJJ['RefMonth'] == 4)]
            Site_NSF_Apr_test = Site_NSF_Apr_test.reset_index(drop=True)
            # Extract monthly NSF for May (test period)
            Site_NSF_May_test = Monthly_NSF_AMJJ[(Monthly_NSF_AMJJ['site_short_ID'] == site) & (Monthly_NSF_AMJJ['WY'].isin(test_year)) & (Monthly_NSF_AMJJ['RefMonth'] == 5)]
            Site_NSF_May_test = Site_NSF_May_test.reset_index(drop=True)
            # Extract montly NSF for June (test period)
            Site_NSF_Jun_test = Monthly_NSF_AMJJ[(Monthly_NSF_AMJJ['site_short_ID'] == site) & (Monthly_NSF_AMJJ['WY'].isin(test_year)) & (Monthly_NSF_AMJJ['RefMonth'] == 6)]
            Site_NSF_Jun_test = Site_NSF_Jun_test.reset_index(drop=True)
            ###############################################
            # Extract monthly value for each month for the train period (30 years)
            # Extract monthly NSF for May (train period)
            Site_NSF_May_train = Monthly_NSF_AMJJ[(Monthly_NSF_AMJJ['site_short_ID'] == site) & (Monthly_NSF_AMJJ['WY'].isin(train_year)) & (Monthly_NSF_AMJJ['RefMonth'] == 5)]
            Site_NSF_May_train = Site_NSF_May_train.reset_index(drop=True)
            # Extract montly NSF for June (train period)
            Site_NSF_Jun_train = Monthly_NSF_AMJJ[(Monthly_NSF_AMJJ['site_short_ID'] == site) & (Monthly_NSF_AMJJ['WY'].isin(train_year)) & (Monthly_NSF_AMJJ['RefMonth'] == 6)]
            Site_NSF_Jun_train = Site_NSF_Jun_train.reset_index(drop=True)
            # Extract monthly NSF for July (train period)
            Site_NSF_Jul_train = Monthly_NSF_AMJJ[(Monthly_NSF_AMJJ['site_short_ID'] == site) & (Monthly_NSF_AMJJ['WY'].isin(train_year)) & (Monthly_NSF_AMJJ['RefMonth'] == 7)]
            Site_NSF_Jul_train = Site_NSF_Jul_train.reset_index(drop=True)
            
            # if issue date is in May
            if date in May_dates:
                # get the total Q of May, Jun and Jul
                TotalQ_Fcst = Site_NSF_May_train['volume'] + Site_NSF_Jun_train['volume'] + Site_NSF_Jul_train['volume']
                # get WSF by adding observed Apr and May, Jun, and Jul
                df_refcst_result['value_10'] = np.repeat(TotalQ_Fcst.quantile([0.1]).values,10) + Site_NSF_Apr_test['volume']
                df_refcst_result['value_30'] = np.repeat(TotalQ_Fcst.quantile([0.3]).values,10) + Site_NSF_Apr_test['volume']
                df_refcst_result['value_50'] = np.repeat(TotalQ_Fcst.quantile([0.5]).values,10) + Site_NSF_Apr_test['volume']
                df_refcst_result['value_70'] = np.repeat(TotalQ_Fcst.quantile([0.7]).values,10) + Site_NSF_Apr_test['volume']
                df_refcst_result['value_90'] = np.repeat(TotalQ_Fcst.quantile([0.9]).values,10) + Site_NSF_Apr_test['volume']
            
            # if issue date is in June
            elif date in June_dates:
                # get the total Q of Apr + May
                TotalQ_Obs = Site_NSF_Apr_test['volume'] + Site_NSF_May_test['volume']
                # get the total Q of Jun and Jul
                TotalQ_Fcst = Site_NSF_Jun_train['volume'] + Site_NSF_Jul_train['volume']
                # get WSF by adding observed Apr and May, Jun, and Jul
                df_refcst_result['value_10'] = np.repeat(TotalQ_Fcst.quantile([0.1]).values,10) + TotalQ_Obs
                df_refcst_result['value_30'] = np.repeat(TotalQ_Fcst.quantile([0.3]).values,10) + TotalQ_Obs
                df_refcst_result['value_50'] = np.repeat(TotalQ_Fcst.quantile([0.5]).values,10) + TotalQ_Obs
                df_refcst_result['value_70'] = np.repeat(TotalQ_Fcst.quantile([0.7]).values,10) + TotalQ_Obs
                df_refcst_result['value_90'] = np.repeat(TotalQ_Fcst.quantile([0.9]).values,10) + TotalQ_Obs
                
            # if issue date is in July
            elif date in July_dates:
                # get the total Q of May, Jun and Jul
                TotalQ_Obs = Site_NSF_Apr_test['volume'] + Site_NSF_May_test['volume'] + Site_NSF_Jun_test['volume']
                TotalQ_Fcst = Site_NSF_Jul_train['volume']
                # get WSF by adding observed Apr and May, Jun, and Jul
                df_refcst_result['value_10'] = np.repeat(TotalQ_Fcst.quantile([0.1]).values,10) + TotalQ_Obs
                df_refcst_result['value_30'] = np.repeat(TotalQ_Fcst.quantile([0.3]).values,10) + TotalQ_Obs
                df_refcst_result['value_50'] = np.repeat(TotalQ_Fcst.quantile([0.5]).values,10) + TotalQ_Obs
                df_refcst_result['value_70'] = np.repeat(TotalQ_Fcst.quantile([0.7]).values,10) + TotalQ_Obs
                df_refcst_result['value_90'] = np.repeat(TotalQ_Fcst.quantile([0.9]).values,10) + TotalQ_Obs
                
            else: 

                df_refcst_result['value_10'] = np.repeat(NSF_train_df['volume'].quantile([0.1]).values,10)
                df_refcst_result['value_30'] = np.repeat(NSF_train_df['volume'].quantile([0.3]).values,10)
                df_refcst_result['value_50'] = np.repeat(NSF_train_df['volume'].quantile([0.5]).values,10)
                df_refcst_result['value_70'] = np.repeat(NSF_train_df['volume'].quantile([0.7]).values,10)
                df_refcst_result['value_90'] = np.repeat(NSF_train_df['volume'].quantile([0.9]).values,10)
            
            # append data frame to the list
            df_refcst_result_list.append(df_refcst_result)
        
        
        pred_df_all_clim = pd.concat(df_refcst_result_list, axis=0, ignore_index=True)

        # Save the combined dataframe to a new CSV file
        combined_filename ='E:/USBR_Snow_Forecast/DATA/GBR/4fold_combined_Climate'+'/%s_%s.csv' % (site, date)
        pred_df_all_clim.to_csv(combined_filename, index=False)







