# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 14:21:59 2025

@author: yue0004
"""


# This is the Python script to train the GBR basin for all basins together by forecast issue dates
# Last Modified on January 14, 2025
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

# Write the prediction for the current issue date
pred_df_GBR = pd.DataFrame(columns=['site_id', 'WY', 'issue_date', 'volume_10', 'volume_50', 'volume_90'])
pred_df_XGB = pd.DataFrame(columns=['site_id', 'WY', 'issue_date', 'volume_10', 'volume_50', 'volume_90'])
pred_df_mean = pd.DataFrame(columns=['site_id', 'WY', 'issue_date', 'volume_10', 'volume_50', 'volume_90'])
pred_df_GBR_dict = {}
pred_df_XGB_dict = {}
pred_df_mean_dict = {}
Data_folder = os.path.join(src_dir, 'data/Updated')


#%% Prepare the training data by forecast date/site/fold
# loop by forecast date
for date in forecast_date:
    print(f'============= forecast date: {date} ===============')
    # concatenated_PPT = []
    # concatenated_SWE = []
    # concatenated_Tmax = []
    # concatenated_Tmin = []
    # concatenated_Tmax_var = []
    # concatenated_Tmin_var = []
    # concatenated_NSF = []
    # concatenated_NSF_test = []
    # concatenated_monthly_NSF = []
    # concatenated_drainage_area = []
    # loop by forecast sites
    for site in site_id_short_in_monthlyNSF:  # TODO 19 SITES FOR MONTHLY NSF site_id_short_in_monthlyNSF site_id
        print('==== ', site)
        # Predictors
        # Read in the predictors
        SWE_folder_path = os.path.join(Data_folder, 'SWE_UA')
        PPT_folder_path = os.path.join(Data_folder, 'PPT')
        Tmax_folder_path = os.path.join(Data_folder, 'T_max')
        Tmin_folder_path = os.path.join(Data_folder, 'T_min')

        SWE_df = Utils.read_predictors(SWE_folder_path + '/%s.csv' % site, site, metadata_path, predictor_is_SWE=True)
        PPT_df = Utils.read_predictors(PPT_folder_path + '/%s.csv' % site, site, metadata_path)
        Tmax_df = Utils.read_predictors(Tmax_folder_path + '/%s_Tmax.csv' % site, site, metadata_path)
        Tmin_df = Utils.read_predictors(Tmin_folder_path + '/%s_Tmin.csv' % site, site, metadata_path)
        NSF_traintest_df = Utils.read_predictors(monthly_NSF_tr_path, site,
                                                 metadata_path,
                                                 predictor_is_NSF=True)
        # process the predictors
        PPT_df, PPT_acc_df = Utils.compute_acc(PPT_df, date, None)
        SWE_new_df = Utils.get_current_value(SWE_df, date)
        SWE_mean_df = Utils.compute_monthly_mean(SWE_df, date)
        Tmax_mean_df = Utils.compute_monthly_mean(Tmax_df, date)
        Tmin_mean_df = Utils.compute_monthly_mean(Tmin_df, date)
        Tmax_var_df = Utils.compute_var(Tmax_df, date)
        Tmin_var_df = Utils.compute_var(Tmin_df, date)

        # Slice both predictors and predictant to desired period (defined earlier)
        NSF_df = Utils.slice_df(df_by_sites[site], start_year, end_year)
        PPT_df_tr = Utils.slice_df(PPT_acc_df, start_year, end_year).rename(columns={'Acc': 'PPT_acc'})

        SWE_df_tr = Utils.slice_df(SWE_new_df, start_year, end_year).rename(columns={'Value': 'SWE_current'})
        Tmax_df_tr = Utils.slice_df(Tmax_mean_df, start_year, end_year).rename(columns={'Mean_past3': 'Tmax_past3',
                                                                                        'Mean_past2': 'Tmax_past2',
                                                                                        'Mean_past1': 'Tmax_past1'})
        Tmin_df_tr = Utils.slice_df(Tmin_mean_df, start_year, end_year).rename(columns={'Mean_past3': 'Tmin_past3',
                                                                                        'Mean_past2': 'Tmin_past2',
                                                                                        'Mean_past1': 'Tmin_past1'})
        Tmax_var_df_tr = Utils.slice_df(Tmax_var_df, start_year, end_year).rename(columns={'Var': 'Tmax_var'})
        Tmin_var_df_tr = Utils.slice_df(Tmin_var_df, start_year, end_year).rename(columns={'Var': 'Tmin_var'})
        
        # # save the training data as csv
        # # NSF df
        # NSF_df.to_csv(f"{local_dir}/data/Train_Data/Train_{site}_NSF_df.csv", index=False)
        # # NSF train/test df
        # NSF_traintest_df.to_csv(f"{local_dir}/data/Train_Data/Train_{site}_{date}_NSF_traintest_df.csv", index=False)
        # PPT_df_tr.to_csv(f"{local_dir}/data/Train_Data/Train_{site}_{date}_PPT_df_tr.csv", index=False)
        # SWE_df_tr.to_csv(f"{local_dir}/data/Train_Data/Train_{site}_{date}_SWE_df_tr.csv", index=False)
        # Tmax_df_tr.to_csv(f"{local_dir}/data/Train_Data/Train_{site}_{date}_Tmax_df_tr.csv", index=False)
        # Tmin_df_tr.to_csv(f"{local_dir}/data/Train_Data/Train_{site}_{date}_Tmin_df_tr.csv", index=False)
        # Tmax_var_df_tr.to_csv(f"{local_dir}/data/Train_Data/Train_{site}_{date}_Tmax_var_df_tr.csv", index=False)
        # Tmin_var_df_tr.to_csv(f"{local_dir}/data/Train_Data/Train_{site}_{date}_Tmin_var_df_tr.csv", index=False)
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
        # Creating and storing all the folds
        train_test_splits = []
        for fold_name, fold in folds:
            train_year, test_year, NSF_train_df, NSF_test_df = Utils.get_fold_data_by_year(NSF_traintest_df, fold)
            train_test_splits.append((NSF_train_df, NSF_test_df))

            NSF_train_df = Utils.slice_df(NSF_train_df, start_year, end_year)
            NSF_test_df = Utils.slice_df(NSF_test_df, start_year, end_year)

            ##
            tab = np.where(
                NSF_train_df['month'] == 12 if int(date[0:2]) - 1 == 0 else NSF_train_df['month'] == int(date[0:2]) - 1)
            tab1 = [element for original_element in tab for element in
                    [original_element, original_element - 1, original_element - 2]]
            tab2 = np.array(tab1).flatten()
            tab2 = np.sort(tab2)
            NSF_train_values_tab = np.array(NSF_train_df['Value'])[tab2]
            if not pd.Series(NSF_train_values_tab).empty:
                non_nan_indices = np.where(~np.isnan(NSF_train_values_tab))[0]
                # Perform linear interpolation for NaN values
                arr_interp = np.interp(np.arange(len(NSF_train_values_tab)), non_nan_indices,
                                        NSF_train_values_tab[non_nan_indices])
                NSF_train_values_tab = arr_interp.reshape(30, 3)
            else:
                NSF_train_values_tab = np.full((30, 3), np.nan)

            ###
            tab = np.where(
                NSF_test_df['month'] == 12 if int(date[0:2]) - 1 == 0 else NSF_test_df['month'] == int(date[0:2]) - 1)
            tab1 = [element for original_element in tab for element in
                    [original_element, original_element - 1, original_element - 2]]
            tab2 = np.array(tab1).flatten()
            tab2 = np.sort(tab2)
            NSF_test_values_tab = np.array(NSF_test_df['Value'])[tab2]
            if not pd.Series(NSF_test_values_tab).empty:
                # Indices of non-NaN values
                non_nan_indices = np.where(~np.isnan(NSF_test_values_tab))[0]
                # Perform linear interpolation for NaN values
                arr_interp = np.interp(np.arange(len(NSF_test_values_tab)), non_nan_indices,
                                        NSF_test_values_tab[non_nan_indices])
                NSF_test_values_tab = arr_interp.reshape(10, 3)
            else:
                NSF_test_values_tab = np.full((10, 3), np.nan)

            NSF_df_tr = pd.DataFrame(NSF_train_values_tab, columns=['NSF_past3', 'NSF_past2', 'NSF_past1'])
            NSF_df_tr.insert(0, 'site_id', Utils.get_site_full_id(site, metadata_path))
            NSF_df_tr.insert(1, 'WY', train_year)
            NSF_df_tr['WY'] = pd.to_datetime(NSF_df_tr['WY'].astype(str) + '-' + date)

            NSF_df_test = pd.DataFrame(NSF_test_values_tab, columns=['NSF_past3', 'NSF_past2', 'NSF_past1'])
            NSF_df_test.insert(0, 'site_id', Utils.get_site_full_id(site, metadata_path))
            NSF_df_test.insert(1, 'WY', test_year)
            NSF_df_test['WY'] = pd.to_datetime(NSF_df_test['WY'].astype(str) + '-' + date)
            monthly_NSF_df_tr = pd.concat([NSF_df_tr, NSF_df_test], ignore_index=True).sort_values(by='WY')
            if site in site_id_short_not_in_monthlyNSF:
                USGS_path = os.path.join(src_dir, 'data\\%s.csv' % site)
                USGS_df = Utils.read_predictors(USGS_path, site, metadata_path)
                USGS_df['Date'] = pd.to_datetime(USGS_df['Date'], format='%m/%d/%Y')
                USGS_df['month_day'] = USGS_df['Date'].dt.strftime('%m-%d')
                # USGS_df = USGS_df[USGS_df['month_day'] == date].copy()
                USGS_mean_df = Utils.compute_monthly_mean(USGS_df, date)
                USGS_df_tr = Utils.slice_df(USGS_mean_df, start_year, end_year)
                monthly_NSF_df_tr['NSF_past3'] = USGS_df_tr['Mean_past3'].reset_index(drop=True)
                monthly_NSF_df_tr['NSF_past2'] = USGS_df_tr['Mean_past2'].reset_index(drop=True)
                monthly_NSF_df_tr['NSF_past1'] = USGS_df_tr['Mean_past1'].reset_index(drop=True)
            # get the drainage area of this site
            drainage_area = metadata[metadata['site_id'] == Utils.get_site_full_id(site, metadata_path)]['drainage_area']
            drainage_area_df_tr = PPT_df_tr.copy().rename(columns={'PPT_acc': 'drainage_area'})
            drainage_area_df_tr['drainage_area'] = drainage_area.values[0]
            # **************************************************************************** #
            # Train test data define
            # **************************************************************************** #
            PPT_df_tr.reset_index(drop=True, inplace=True)
            SWE_df_tr.reset_index(drop=True, inplace=True)
            Tmax_df_tr.reset_index(drop=True, inplace=True)
            Tmin_df_tr.reset_index(drop=True, inplace=True)
            Tmax_var_df_tr.reset_index(drop=True, inplace=True)
            Tmin_var_df_tr.reset_index(drop=True, inplace=True)
            monthly_NSF_df_tr.reset_index(drop=True, inplace=True)
            drainage_area_df_tr.reset_index(drop=True, inplace=True)

            # Concatenate the DataFrames along the columns axis
            input_df = pd.concat([
                PPT_df_tr,
                SWE_df_tr.drop(columns=['site_id', 'WY']),
                Tmax_df_tr.drop(columns=['site_id', 'WY']),
                Tmin_df_tr.drop(columns=['site_id', 'WY']),
                Tmax_var_df_tr.drop(columns=['site_id', 'WY']),
                Tmin_var_df_tr.drop(columns=['site_id', 'WY']),
                monthly_NSF_df_tr.drop(columns=['site_id', 'WY']),
                drainage_area_df_tr.drop(columns=['site_id', 'WY'])
            ], axis=1)
            merged_df = pd.merge(input_df, NSF_df, on=['WY', 'site_id'])

            # Split the data into training and testing sets
            # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0, random_state=random_state)
            train_df = merged_df[merged_df['WY'].isin(train_year)]
            test_df = merged_df[merged_df['WY'].isin(test_year)]
            
            # save the training test data to local
            train_df.to_csv(f"{local_dir}/data/Train_Data_4Fold/Train_{site}_{date}_{fold_name}_train_df.csv", index=False)
            test_df.to_csv(f"{local_dir}/data/Train_Data_4Fold/Train_{site}_{date}_{fold_name}_test_df.csv", index=False)
            
            
            


#%% Training the model
# loop by forecast date
for date in forecast_date:
    print(f'============= forecast date: {date} ===============')
    
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
        ###########################################################
        # Define train and test data
        ###########################################################
        # initial list to save all df
        concatenated_Train = []

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
                model_filename_SWE = 'E:/USBR_Snow_Forecast/Fcst_Model/Results_Temp/all_basin_uaswe/with_SWE/AllSite_%s_%s_%s_model.dat' % (
                    date, str(quantile), fold_name)
                joblib.dump(GB_regressor_SWE, model_filename_SWE)
                model_filename_noSWE = 'E:/USBR_Snow_Forecast/Fcst_Model/Results_Temp/all_basin_uaswe/without_SWE/AllSite_%s_%s_%s_model.dat' % (
                    date, str(quantile), fold_name)
                joblib.dump(GB_regressor_noSWE, model_filename_noSWE)


