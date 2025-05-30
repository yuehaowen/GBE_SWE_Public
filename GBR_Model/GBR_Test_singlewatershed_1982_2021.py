import glob
import os
import sys

import pandas as pd
import warnings
import Utils
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from datetime import datetime
import pdb
import joblib
from joblib import load

save_test_results = True
src_dir = r"C:\Users\yihan\OneDrive - University of Oklahoma\OU\Research\2023-2024 Forecasting"
# Note: this code is for forecasting on the most recent previous issue date.
# For example, if running this code on Jan 11, it will make forecast for issue date Jan 8.
# **************************************************************************** #
# Read in naturalized streamflow (target variable)
# **************************************************************************** #
# Train set
metadata_path = 'data/metadata.csv'
metadata = pd.read_csv(metadata_path)
# Train set
target_tr_path = 'data/train_1982_2022.csv'
target_df = pd.read_csv(target_tr_path)
target_df = target_df.rename(columns={'year': 'WY'})
target_df['site_id_short'] = target_df['site_id'].apply(
    lambda x: x[0].upper() + x.split('_')[1][0].upper() + x.split('_')[-1][0].upper() if isinstance(x, str) and len(
        x) > 0 else None)

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

site_id_short = target_df['site_id_short'].unique()  # site_id_short is an array
site_id_short = site_id_short[site_id_short != 'DLI']
monthly_NSF_tr_path = 'data/train_monthly_naturalized_flow_1982_2022.csv'
# monthly_NSF_test_path = 'data/test_monthly_naturalized_flow.csv'
site_id_in_monthlyNSF = pd.read_csv(monthly_NSF_tr_path)['site_id'].unique()
site_id_short_in_monthlyNSF = [Utils.get_site_short_id(x, metadata_path) for x in site_id_in_monthlyNSF]
site_id_short_not_in_monthlyNSF = list(set(target_df['site_id_short'].unique()) - set(site_id_short_in_monthlyNSF))
site_id_not_in_monthlyNSF = list(set(target_df['site_id'].unique()) - set(site_id_in_monthlyNSF))

site_id_short_in_monthlyNSF.remove('SRA')

forecast_date = []
days = ['01', '08', '15', '22']
# days = ['01']
months = ['01', '02', '03', '04', '05', '06', '07']
for month in months:
    for day in days:
        forecast_date.append(month + '-' + day)
# **************************************************************************** #
# Define training and testing period
# **************************************************************************** #
start_year = 1982
end_year = 2021
year_list = list(range(start_year, end_year + 1))
quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]

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
for date in forecast_date:
    print(f'============= forecast date: {date} ===============')
    for site in site_id_short_in_monthlyNSF:  # TODO 22 sites
        print('==== ', site)
        # Creating and storing all the folds

        SWE_folder_path = os.path.join(Data_folder, 'snotel')
        PPT_folder_path = os.path.join(Data_folder, 'PPT')
        Tmax_folder_path = os.path.join(Data_folder, 'T_max')
        Tmin_folder_path = os.path.join(Data_folder, 'T_min')

        used_snotel, SWE_df = Utils.read_snotel(SWE_folder_path, site)
        print(len(used_snotel), "snotel stations are used.")
        PPT_df = Utils.read_predictors(PPT_folder_path + '/%s.csv' % site, site, metadata_path)
        Tmax_df = Utils.read_predictors(Tmax_folder_path + '/%s_Tmax.csv' % site, site, metadata_path)
        Tmin_df = Utils.read_predictors(Tmin_folder_path + '/%s_Tmin.csv' % site, site, metadata_path)
        NSF_traintest_df = Utils.read_predictors('data/train_monthly_naturalized_flow_1982_2022.csv', site,
                                                 metadata_path,
                                                 predictor_is_NSF=True)

        # process the predictors
        PPT_df, PPT_acc_df = Utils.compute_acc(PPT_df, date, None)
        SWE_new_df = Utils.get_snotel_current_value(SWE_df, date)
        Tmax_mean_df = Utils.compute_monthly_mean(Tmax_df, date)
        Tmin_mean_df = Utils.compute_monthly_mean(Tmin_df, date)
        Tmax_var_df = Utils.compute_var(Tmax_df, date)
        Tmin_var_df = Utils.compute_var(Tmin_df, date)

        # Slice both predictors and predictant to desired period (defined earlier)
        NSF_df = Utils.slice_df(df_by_sites[site], start_year, end_year)
        PPT_df_tr = Utils.slice_df(PPT_acc_df, start_year, end_year).rename(columns={'Acc': 'PPT_acc'})

        SWE_df_tr = Utils.slice_df(SWE_new_df, start_year, end_year)
        Tmax_df_tr = Utils.slice_df(Tmax_mean_df, start_year, end_year).rename(columns={'Mean_past3': 'Tmax_past3',
                                                                                        'Mean_past2': 'Tmax_past2',
                                                                                        'Mean_past1': 'Tmax_past1'})
        Tmin_df_tr = Utils.slice_df(Tmin_mean_df, start_year, end_year).rename(columns={'Mean_past3': 'Tmin_past3',
                                                                                        'Mean_past2': 'Tmin_past2',
                                                                                        'Mean_past1': 'Tmin_past1'})
        Tmax_var_df_tr = Utils.slice_df(Tmax_var_df, start_year, end_year).rename(columns={'Var': 'Tmax_var'})
        Tmin_var_df_tr = Utils.slice_df(Tmin_var_df, start_year, end_year).rename(columns={'Var': 'Tmin_var'})

        train_test_splits = []
        for fold_name, fold in folds:
            train_year, test_year, NSF_train_df, NSF_test_df = Utils.get_fold_data_by_year(NSF_traintest_df, fold)
            train_test_splits.append((NSF_train_df, NSF_test_df))

            NSF_train_df = Utils.slice_df(NSF_train_df, start_year, end_year)
            NSF_test_df = Utils.slice_df(NSF_test_df, start_year, end_year)

            for SWE_ind in SWE_flags:
                if SWE_ind == 'SWE':
                    model_10_path = os.path.join('trained_models_single_basin_1982_2021/with_SWE/',
                                                 site + "_" + date + "_" + str(0.1) + "_" + fold_name + "_model.dat")
                    model_30_path = os.path.join('trained_models_single_basin_1982_2021/with_SWE/',
                                                 site + "_" + date + "_" + str(0.3) + "_" + fold_name + "_model.dat")
                    model_50_path = os.path.join('trained_models_single_basin_1982_2021/with_SWE/',
                                                 site + "_" + date + "_" + str(0.5) + "_" + fold_name + "_model.dat")
                    model_70_path = os.path.join('trained_models_single_basin_1982_2021/with_SWE/',
                                                 site + "_" + date + "_" + str(0.7) + "_" + fold_name + "_model.dat")
                    model_90_path = os.path.join('trained_models_single_basin_1982_2021/with_SWE/',
                                                 site + "_" + date + "_" + str(0.9) + "_" + fold_name + "_model.dat")
                elif SWE_ind == 'noSWE':
                    model_10_path = os.path.join('trained_models_single_basin_1982_2021/without_SWE',
                                                 site + "_" + date + "_" + str(0.1) + "_" + fold_name + "_model.dat")
                    model_30_path = os.path.join('trained_models_single_basin_1982_2021/without_SWE',
                                                 site + "_" + date + "_" + str(0.3) + "_" + fold_name + "_model.dat")
                    model_50_path = os.path.join('trained_models_single_basin_1982_2021/without_SWE',
                                                 site + "_" + date + "_" + str(0.5) + "_" + fold_name + "_model.dat")
                    model_70_path = os.path.join('trained_models_single_basin_1982_2021/without_SWE',
                                                 site + "_" + date + "_" + str(0.7) + "_" + fold_name + "_model.dat")
                    model_90_path = os.path.join('trained_models_single_basin_1982_2021/without_SWE',
                                                 site + "_" + date + "_" + str(0.9) + "_" + fold_name + "_model.dat")
                pred_model_10 = load(model_10_path)
                pred_model_30 = load(model_30_path)
                pred_model_50 = load(model_50_path)
                pred_model_70 = load(model_70_path)
                pred_model_90 = load(model_90_path)

                ##
                tab = np.where(
                    NSF_train_df['month'] == 12 if int(date[0:2]) - 1 == 0 else NSF_train_df['month'] == int(
                        date[0:2]) - 1)
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
                    NSF_test_df['month'] == 12 if int(date[0:2]) - 1 == 0 else NSF_test_df['month'] == int(
                        date[0:2]) - 1)
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
                    USGS_path = os.path.join(src_dir, 'data/%s.csv' % site)
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
                drainage_area = metadata[metadata['site_id'] == Utils.get_site_full_id(site, metadata_path)][
                    'drainage_area']
                drainage_area_df_tr = PPT_df_tr.copy().rename(columns={'PPT_acc': 'drainage_area'})
                drainage_area_df_tr['drainage_area'] = drainage_area.values[0]

                # **************************************************************************** #
                # Test data define
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

                test_df = merged_df[merged_df['WY'].isin(test_year)]
                if SWE_ind == 'SWE':
                    X_test = test_df.drop(columns=['site_id', 'WY', 'volume', 'site_id_short'])
                    X_test = X_test[pred_model_10.feature_names_in_]
                elif SWE_ind == 'noSWE':
                    X_test = test_df[['PPT_acc',
                                      'Tmax_past3',
                                      'Tmin_past3',
                                      'Tmax_var',
                                      'Tmin_var',
                                      'NSF_past3', 'NSF_past2', 'NSF_past1',
                                      'drainage_area'
                                      ]]

                y_test = test_df['volume']
                df_test_result['site_id'] = test_df.reset_index(drop=True)['site_id']
                df_test_result['WY'] = test_df.reset_index(drop=True)['WY']
                df_test_result['value_10'] = pred_model_10.predict(X_test)
                df_test_result['value_30'] = pred_model_30.predict(X_test)
                df_test_result['value_50'] = pred_model_50.predict(X_test)
                df_test_result['value_70'] = pred_model_70.predict(X_test)
                df_test_result['value_90'] = pred_model_90.predict(X_test)

                if save_test_results:
                    if SWE_ind == 'SWE':
                        df_test_result.to_csv(
                            'results_single_basin_1982_2021/with_SWE/%s_%s_%s.csv' % (site, date, fold_name),
                            index=False)
                    elif SWE_ind == 'noSWE':
                        df_test_result.to_csv(
                            'results_single_basin_1982_2021/without_SWE/%s_%s_%s.csv' % (site, date, fold_name),
                            index=False)
