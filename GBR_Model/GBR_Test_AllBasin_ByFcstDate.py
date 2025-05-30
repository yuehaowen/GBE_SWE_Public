# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 16:41:52 2025

@author: yue0004
"""

# This is the Python script to test the results from training using all basins together by forecast dates
# Last Modified on January 15, 2024
###############################################################################

#%% import packages part 1
import glob
import os
import sys
import pandas as pd
import warnings

#%% import packages part 2
# import a customized module
import sys
sys.path.append('E:/USBR_Snow_Forecast/Fcst_Model')
import Utils

#%% import packages part 3
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from datetime import datetime
import pdb
import joblib
from joblib import load

#%% set global parameters
warnings.filterwarnings('ignore')
random_state = 42
save_test_results = True
src_dir = r"E:\OneDrive\OneDrive - University of Oklahoma\2023-2024 Forecasting"
local_dir = 'E:/USBR_Snow_Forecast/Fcst_Model'
#%% import target variable
# **************************************************************************** #
# Read in naturalized streamflow (target variable)
# **************************************************************************** #
# Train set
metadata_path = os.path.join(src_dir, 'data', 'metadata.csv')

# if not os.path.exists(metadata_path):
#     raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
metadata = pd.read_csv(metadata_path)

# Train set
target_tr_path = os.path.join(src_dir, 'data', 'train_1982_2022.csv')
target_df = pd.read_csv(target_tr_path)
target_df = target_df.rename(columns={'year': 'WY'})
target_df['site_id_short'] = target_df['site_id'].apply(
    lambda x: x[0].upper() + x.split('_')[1][0].upper() + x.split('_')[-1][0].upper() if isinstance(x, str) and len(
        x) > 0 else None)


#%% Create dictionary 
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


#%% Get the site ID
# get all ids from target df
site_id_short = target_df['site_id_short'].unique()  # site_id_short is an array
site_id_short = site_id_short[site_id_short != 'DLI']
# get monthly natrualized streamflow for training
monthly_NSF_tr_path = os.path.join(src_dir, 'data', 'train_monthly_naturalized_flow_1982_2022.csv')

# monthly_NSF_test_path = 'data/test_monthly_naturalized_flow.csv'
site_id_in_monthlyNSF = pd.read_csv(monthly_NSF_tr_path)['site_id'].unique()
site_id_short_in_monthlyNSF = [Utils.get_site_short_id(x, metadata_path) for x in site_id_in_monthlyNSF]
site_id_short_not_in_monthlyNSF = list(set(target_df['site_id_short'].unique()) - set(site_id_short_in_monthlyNSF))
site_id_not_in_monthlyNSF = list(set(target_df['site_id'].unique()) - set(site_id_in_monthlyNSF))
# remove unwanted sites
site_id_short_in_monthlyNSF.remove('SRA')

# Here we should have 19 forecast sites
# ['MRT','BRI','PRI','PRP','DRI','RRI','TPI','FRI','YRM','ARD','VRV','WRO','GRD','HHI','CRF','SRS','SRH','ORD','BRB']

#%% Define forecast dates
# forecast dates: four days per month
forecast_date = []
days = ['01', '08', '15', '22']
# days = ['01']
months = ['01', '02', '03', '04', '05', '06', '07']
for month in months:
    for day in days:
        forecast_date.append(month + '-' + day)

#%% Define training and testing period
# **************************************************************************** #
# Define training and testing period
# **************************************************************************** #
start_year = 1982
end_year = 2021
year_list = list(range(start_year, end_year + 1))
quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]


#%% Load predictorsn and train model by sites
# **************************************************************************** #
# Read in and process predictors [PPT_acc, SWE_acc, Tmax_mean, Tmin_mean]
# **************************************************************************** #
mean_QL_df = pd.DataFrame(columns=['issue_date', 'GBR'])
IC_df = pd.DataFrame(columns=['issue_date', 'GBR'])
mean_QL_df['issue_date'] = forecast_date
IC_df['issue_date'] = forecast_date

# Write the prediction for the current issue date
pred_df_GBR = pd.DataFrame(columns=['site_id', 'WY', 'issue_date', 'volume_10', 'volume_50', 'volume_90'])
pred_df_GBR_dict = {}

Data_folder = 'data/Updated'
concatenated_PPT = []
concatenated_SWE = []
concatenated_Tmax = []
concatenated_Tmin = []
concatenated_Tmax_var = []
concatenated_Tmin_var = []
concatenated_NSF = []
concatenated_NSF_test = []
concatenated_monthly_NSF = []
concatenated_drainage_area = []
df_test_result = pd.DataFrame(columns=['site_id', 'WY', 'value_10', 'value_30', 'value_50', 'value_70', 'value_90'])
# site_id_list = [[element] for element in site_id_short]
SWE_flags = ['SWE', "noSWE"]
folds = [
    ("f1", (1992, 2021, 1982, 1991)),
    ("f2", ((1982, 1991), (2002, 2021), 1992, 2001)),
    ("f3", ((1982, 2001), (2012, 2021), 2002, 2011)),
    ("f4", (1982, 2011, 2012, 2021))
]

#%% Test the prediction by sites

for date in forecast_date:
    print(f'============= forecast date: {date} ===============')
    for site in site_id_short_in_monthlyNSF:  # TODO 22 sites
        print('==== ', site)
        # loop by folds
        for fold_name, fold in folds:

            # loop by swe or noswe
            for SWE_ind in SWE_flags:
                if SWE_ind == 'SWE':
                    model_10_path = os.path.join('E:/USBR_Snow_Forecast/Fcst_Model/Results_Temp/all_basin_uaswe/with_SWE/',
                                                 f"AllSite_{date}_0.1_{fold_name}_model.dat")
                    model_30_path = os.path.join('E:/USBR_Snow_Forecast/Fcst_Model/Results_Temp/all_basin_uaswe/with_SWE/',
                                                 f"AllSite_{date}_0.3_{fold_name}_model.dat")
                    model_50_path = os.path.join('E:/USBR_Snow_Forecast/Fcst_Model/Results_Temp/all_basin_uaswe/with_SWE/',
                                                 f"AllSite_{date}_0.5_{fold_name}_model.dat")
                    model_70_path = os.path.join('E:/USBR_Snow_Forecast/Fcst_Model/Results_Temp/all_basin_uaswe/with_SWE/',
                                                 f"AllSite_{date}_0.7_{fold_name}_model.dat")
                    model_90_path = os.path.join('E:/USBR_Snow_Forecast/Fcst_Model/Results_Temp/all_basin_uaswe/with_SWE/',
                                                 f"AllSite_{date}_0.9_{fold_name}_model.dat")
                elif SWE_ind == 'noSWE':
                    model_10_path = os.path.join('E:/USBR_Snow_Forecast/Fcst_Model/Results_Temp/all_basin_uaswe/without_SWE/',
                                                 f"AllSite_{date}_0.1_{fold_name}_model.dat")
                    model_30_path = os.path.join('E:/USBR_Snow_Forecast/Fcst_Model/Results_Temp/all_basin_uaswe/without_SWE/',
                                                 f"AllSite_{date}_0.3_{fold_name}_model.dat")
                    model_50_path = os.path.join('E:/USBR_Snow_Forecast/Fcst_Model/Results_Temp/all_basin_uaswe/without_SWE/',
                                                 f"AllSite_{date}_0.5_{fold_name}_model.dat")
                    model_70_path = os.path.join('E:/USBR_Snow_Forecast/Fcst_Model/Results_Temp/all_basin_uaswe/without_SWE/',
                                                 f"AllSite_{date}_0.7_{fold_name}_model.dat")
                    model_90_path = os.path.join('E:/USBR_Snow_Forecast/Fcst_Model/Results_Temp/all_basin_uaswe/without_SWE/',
                                                 f"AllSite_{date}_0.9_{fold_name}_model.dat")
                pred_model_10 = load(model_10_path)
                pred_model_30 = load(model_30_path)
                pred_model_50 = load(model_50_path)
                pred_model_70 = load(model_70_path)
                pred_model_90 = load(model_90_path)

                # import test data for forecast site
                test_df = pd.read_csv(f"{local_dir}/data/Train_Data_4Fold/Train_{site}_{date}_{fold_name}_test_df.csv")
                # Test data for GBR-SWE
                if SWE_ind == 'SWE':
                    X_test = test_df.drop(columns=['site_id', 'WY', 'volume', 'site_id_short'])
                    X_test = X_test[pred_model_10.feature_names_in_]
                # Test data for GBR-NoSWE
                elif SWE_ind == 'noSWE':
                    X_test = test_df[['PPT_acc',
                                      'Tmax_past3',
                                      'Tmin_past3',
                                      'Tmax_var',
                                      'Tmin_var',
                                      'NSF_past3', 'NSF_past2', 'NSF_past1',
                                      'drainage_area'
                                      ]]
                # get test data for total streamflow
                y_test = test_df['volume']
                df_test_result['site_id'] = test_df.reset_index(drop=True)['site_id']
                df_test_result['WY'] = test_df.reset_index(drop=True)['WY']
                df_test_result['value_10'] = pred_model_10.predict(X_test)
                df_test_result['value_30'] = pred_model_30.predict(X_test)
                df_test_result['value_50'] = pred_model_50.predict(X_test)
                df_test_result['value_70'] = pred_model_70.predict(X_test)
                df_test_result['value_90'] = pred_model_90.predict(X_test)
                # save test results to loca
                if save_test_results:
                    if SWE_ind == 'SWE':
                        df_test_result.to_csv(
                            'E:/USBR_Snow_Forecast/Fcst_Model/results_allbasin_bydate/with_SWE/%s_%s_%s.csv' % (site, date, fold_name),
                            index=False)
                    elif SWE_ind == 'noSWE':
                        df_test_result.to_csv(
                            'E:/USBR_Snow_Forecast/Fcst_Model/results_allbasin_bydate/without_SWE/%s_%s_%s.csv' % (site, date, fold_name),
                            index=False)




