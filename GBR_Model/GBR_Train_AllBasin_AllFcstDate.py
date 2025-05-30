# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 20:54:11 2025

@author: yue0004
"""

# This is the Python script to train the GBR model using all data together
# All basin and all forecast dates
# Last Modified on January 15, 2024
###############################################################################


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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from datetime import datetime
import pdb
import joblib
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
# import statsmodels.api as sm
from sklearn.linear_model import QuantileRegressor

#%% set global parameters
warnings.filterwarnings('ignore')
random_state = 42
save_model = True
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

#######################################################################
# Define 4-fold verification
#######################################################################
# Define the year ranges for each fold
folds = [
    ("f1", (1992, 2021, 1982, 1991)),
    ("f2", ((1982, 1991), (2002, 2021), 1992, 2001)),
    ("f3", ((1982, 2001), (2012, 2021), 2002, 2011)),
    ("f4", (1982, 2011, 2012, 2021))
]

# loop by fold
for fold_name, fold in folds:
    # initial list to save all df
    concatenated_Train = []

    # loop by forecast date
    for date in forecast_date:
        print(f'============= forecast date: {date} ===============')
    


        ###########################################################
        # Define train and test data
        ###########################################################

        # loop by forecast sites to combine all training/test data together
        for site in site_id_short_in_monthlyNSF:  # TODO 19 SITES FOR MONTHLY NSF site_id_short_in_monthlyNSF site_id
            # print('==== ', site)

            # Import training and testing data for each site
            train_df = pd.read_csv(f"{local_dir}/data/Train_Data_4Fold/Train_{site}_{date}_{fold_name}_train_df.csv")
            # append 
            concatenated_Train.append(train_df.copy())

    # Concatenate all DataFrames for different sites into one
    Train_DF_All = pd.concat(concatenated_Train, ignore_index=True)
    Train_DF_All.reset_index(drop=True, inplace=True)
    # Define training data for GBR-SWE
    X_train_SWE = Train_DF_All.drop(columns=['site_id', 'WY', 'volume', 'site_id_short'])
    
    # Define training data for GBR_NoSWE
    X_train_noSWE = Train_DF_All[['PPT_acc',
                              'Tmax_past3',
                              'Tmin_past3',
                              'Tmax_var',
                              'Tmin_var',
                              'NSF_past3', 'NSF_past2', 'NSF_past1',
                              'drainage_area'
                              ]]
    # Target/Predictant
    y_train = Train_DF_All['volume']
    
    
    # **************************************************************************** #
    # Gradient boost (for the current forecast date)
    # **************************************************************************** #
    # Define the hyperparameter grid to search
    param_grid = {
        'max_features': ['sqrt', None],
        'min_samples_leaf': [3, 5],
        'min_samples_split': [3, 5],
        'n_estimators': [50, 100, 200],  # Adjust the values as needed
        'learning_rate': [0.05, 0.01],  # Adjust the values as needed
        'max_depth': [3, 5]  # Adjust the values as needed
    }
    
    # Perform quantile regression for each quantile
    GB_regressor_SWE = GradientBoostingRegressor(loss='quantile', alpha=0.5, random_state=random_state)
    grid_search_SWE = GridSearchCV(estimator=GB_regressor_SWE, param_grid=param_grid,
                                    scoring='neg_mean_squared_error', cv=3)
    grid_search_SWE.fit(X_train_SWE, y_train)
    # Get the best hyperparameters
    best_params_SWE = grid_search_SWE.best_params_
    print("Best Hyperparameters for quantile ", 0.5, ": ", best_params_SWE)
    
    GB_regressor_noSWE = GradientBoostingRegressor(loss='quantile', alpha=0.5, random_state=random_state)
    grid_search_noSWE = GridSearchCV(estimator=GB_regressor_noSWE, param_grid=param_grid,
                                      scoring='neg_mean_squared_error', cv=3)
    grid_search_noSWE.fit(X_train_noSWE, y_train)
    # Get the best hyperparameters
    best_params_noSWE = grid_search_noSWE.best_params_
    print("Best Hyperparameters for quantile ", 0.5, ": ", best_params_noSWE)
    
    for quantile in quantiles:
        quantile_params = {
            'loss': 'quantile',
            'alpha': quantile,
            'random_state': random_state,
        }
        combined_params_SWE = {**best_params_SWE, **quantile_params}
        combined_params_noSWE = {**best_params_noSWE, **quantile_params}
        GB_regressor_SWE = GradientBoostingRegressor(**combined_params_SWE)
        GB_regressor_SWE.fit(X_train_SWE, y_train)
        GB_regressor_noSWE = GradientBoostingRegressor(**combined_params_noSWE)
        GB_regressor_noSWE.fit(X_train_noSWE, y_train)
        if save_model:
            model_filename_SWE = 'E:/USBR_Snow_Forecast/Fcst_Model/Results_Temp/all_basin_all_date/with_SWE/AllSite_AllDate_%s_%s_model.dat' % (
                 str(quantile), fold_name)
            joblib.dump(GB_regressor_SWE, model_filename_SWE)
            model_filename_noSWE = 'E:/USBR_Snow_Forecast/Fcst_Model/Results_Temp/all_basin_all_date/without_SWE/AllSite_AllDate_%s_%s_model.dat' % (
                 str(quantile), fold_name)
            joblib.dump(GB_regressor_noSWE, model_filename_noSWE)





