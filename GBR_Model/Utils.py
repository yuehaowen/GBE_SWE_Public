import pdb

import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
import os

warnings.filterwarnings('ignore')


# Contains all the functions used in the scripts

def pinball_loss(y_true, y_pred, quantile):
    errors = y_true - y_pred
    return np.maximum(quantile * errors, (quantile - 1) * errors)


def mean_quantile_loss(y_true, quantile_predictions, quantiles):
    losses = []

    for quantile in quantiles:
        predictions = quantile_predictions[quantile]['value']
        loss = np.mean(pinball_loss(y_true, predictions, quantile))
        losses.append(loss)

    mean_loss = np.mean(losses)
    return mean_loss


def mean_quantile_loss_eval(y_true, quantile_predictions, quantiles):
    losses = []

    for quantile in quantiles:
        quantile_100 = int(quantile * 100)
        predictions = quantile_predictions[f'value_{quantile_100}']
        loss = np.mean(pinball_loss(y_true, predictions, quantile))
        losses.append(loss)

    mean_loss = np.mean(losses)
    return mean_loss


def read_predictors(file_path, site_id_short, metadata_path, predictor_is_SWE=False, predictor_is_NSF=False):
    '''
    :param file_path: path to the predictor csv time series
    :param site_id_short:
    :return: A dataframe ['site_id'(full name), Date, Value]
    '''
    # Read the CSV file
    df = pd.read_csv(file_path)
    # Rename columns
    if predictor_is_SWE:
        df = df.drop(columns=['Average Accumulated Water Year PPT (in)'])
        df.columns = ['Date', 'Value']
        df['Value'] = df['Value'] * 25.4  # convert inch to mm
    elif predictor_is_NSF:
        site_id_full = get_site_full_id(site_id_short, metadata_path)
        df.columns = ['site_id', 'WY', 'year', 'month', 'Value']
        # Check if site_id_full is in the DataFrame
        if site_id_full in df['site_id'].values:
            # Filter the DataFrame for the specific site_id_full
            df = df[df['site_id'] == site_id_full]
            df = df.drop(columns=['site_id'])
        else:
            water_years = list(range(1982, 2024))
            # water_years = pd.date_range(start='1981-01-01', end='2023-12-31', freq='D')
            # If site_id_full is not found, create a new DataFrame with the specified site_id
            new_data = {'site_id': [site_id_full] * len(water_years), 'WY': water_years, 'year': np.nan,
                        'month': np.nan,
                        'Value': np.nan}
            df = pd.DataFrame(new_data)
            return df
    else:
        df.columns = ['Date', 'Value']
    df.insert(0, 'site_id', get_site_full_id(site_id_short, metadata_path))

    return df


def read_snotel(snotel_folder_path, site_id_short):
    
    # define src dir
    src_dir = r"E:\OneDrive\OneDrive - University of Oklahoma\2023-2024 Forecasting"
    # unit: inch
    # identify snotel stations for current watershed
    snotel_stations_df = pd.read_csv(os.path.join(src_dir, 'data/Updated/snotel/sites_to_snotel_stations.csv'))
    site_id_full = get_site_full_id(site_id_short, os.path.join(src_dir, 'data', 'metadata.csv'))
    station_triplet = snotel_stations_df.loc[snotel_stations_df['site_id'] == site_id_full, 'stationTriplet'].values
    station_filename = [triplet.replace(":", "_") + ".csv" for triplet in station_triplet]

    # identify snotel stations that are avaliable from WY 1982 to WY 2023
    root_dir = snotel_folder_path
    subdirectories = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if
                      os.path.isdir(os.path.join(root_dir, d))]
    common_files = set(os.listdir(subdirectories[0]))
    # Iterate over the rest of the folders and find the common files
    for subdir in subdirectories[1:]:
        files_in_subdir = set(os.listdir(subdir))
        common_files.intersection_update(files_in_subdir)
    common_stations = set(station_filename).intersection(common_files)

    # Store the SWE from each snotel station at a specific watershed
    final_df = pd.DataFrame()
    # Iterate over each station
    used_stations = common_stations.copy()
    for station in common_stations:
        station_dfs = []
        # Iterate over each subdirectory
        for sub_dir in subdirectories:
            file_path = os.path.join(sub_dir, station)
            if 'WTEQ_DAILY' in pd.read_csv(file_path).columns:
                # Rename columns and add 'site_id' column
                SWE = pd.read_csv(file_path, usecols=['date', 'WTEQ_DAILY'])
                SWE.rename(columns={'date': 'Date', 'WTEQ_DAILY': station[:-4]}, inplace=True)
                SWE['site_id'] = site_id_short
                # Append the DataFrame to the list
                station_dfs.append(SWE)

            else:
                print('SWE does not exist. Snotel station skipped.')
                used_stations.remove(station)
                break

        # Concatenate DataFrames for the current station
        if station_dfs:
            station_concatenated_df = pd.concat(station_dfs, ignore_index=True)

            # Merge with the final DataFrame using the 'Date' column
            if final_df.empty:
                final_df = station_concatenated_df
            else:
                final_df = pd.merge(final_df, station_concatenated_df, on=["Date", "site_id"], how='outer')

    return used_stations, final_df


def get_site_full_id(site_id_short: str, metadata_path: str):
    df = pd.read_csv(metadata_path)
    site_id_dict = {}
    for i in range(len(df['site_id'])):
        short_id = df['site_id_short'][i]
        full_id = df['site_id'][i]
        site_id_dict[short_id] = full_id
    site_full_id = site_id_dict.get(site_id_short, None)
    return site_full_id


def get_site_short_id(site_id_full: str, metadata_path: str):
    df = pd.read_csv(metadata_path)
    site_id_dict = {}
    for i in range(len(df['site_id'])):
        short_id = df['site_id_short'][i]
        full_id = df['site_id'][i]
        site_id_dict[full_id] = short_id
    site_short_id = site_id_dict.get(site_id_full, None)  # Use site_id_full as the key
    return site_short_id


def get_site_usgs_id(site_id_short: str, metadata_path: str):
    df = pd.read_csv(metadata_path)
    site_id_dict = {}
    for i in range(len(df['site_id_short'])):
        short_id = df['site_id_short'][i]
        usgs_id = df['usgs_id'][i]
        if len(str(usgs_id)) == 7:
            usgs_id = "0" + str(usgs_id)
        site_id_dict[short_id] = str(usgs_id)
    site_usgs_id = site_id_dict.get(site_id_short, None)  # Use site_id_full as the key
    return site_usgs_id


def slice_df(df: pd.DataFrame,
             start_year: int,
             end_year: int):
    condition = (df['WY'] >= start_year) & (df['WY'] <= end_year)
    return df[condition]


def compute_acc(df: pd.DataFrame,
                forcast_date: str,
                saving_path: str,
                mode: str = 'acc',
                swe: bool = False
                ):
    '''
    :param df: Predictors time series data frame with
    1st col: "site_id"
    2nd col: "Date" ['%m-%d']. Do not specify the year.
    3rd col: "Value"
    :param forcast_date: str ['mm-dd'], e.g. "01-01","01-15"...
    :param saving_path: path to save the accumulated value df to csv. If None, not save.
    :param mode: str ['acc','mean']. Default: 'acc'
    mode = 'mean' to compute past mean values (designed for Tmax and Tmin)
    :return: accumualted value until the forecast date in the same WY
    '''

    df['Date'] = pd.to_datetime(df['Date'])
    # Define water year
    df['WY'] = df['Date'].dt.year
    mask = (df['Date'].dt.month >= 10) & (df['Date'].dt.month <= 12)
    df.loc[mask, 'WY'] = df.loc[mask, 'WY'] + 1
    WYs = None
    if swe:
        WYs = df.WY.unique()
    if not swe:
        WYs = df.WY.unique()[1:]
    if mode == 'mean':
        # Accumulate the past values in the same WY
        df['Mean'] = df.groupby('WY')['Value'].cumsum()
        mean_df = pd.DataFrame(columns=['site_id', 'WY', 'Mean'])
        for WY in WYs:  # remove WY 1981 since the record is incomplete
            date = pd.to_datetime(str(WY) + '-' + forcast_date, format='%Y-%m-%d')
            date_diff = (date - pd.to_datetime(str(WY) + '-' + '10-01', format='%Y-%m-%d')).days
            site_id = df[df['Date'] == date]['site_id'].values[0]
            mean = df[df['Date'] == date]['Mean'].values[0] / date_diff
            mean_df = pd.concat([mean_df, pd.DataFrame({'site_id': [site_id], 'WY': [WY], 'Mean': [mean]})])
        if saving_path:
            mean_df.to_csv(saving_path, index=False)
        return df, mean_df
    # Accumulate the past values in the same WY
    df['Acc'] = df.groupby('WY')['Value'].cumsum()
    acc_df = pd.DataFrame(columns=['site_id', 'WY', 'Acc'])
    for WY in WYs:  # remove WY 1981 since the record is incomplete
        date = pd.to_datetime(str(WY) + '-' + forcast_date, format='%Y-%m-%d')
        site_id = df[df['Date'] == date]['site_id'].values[0]
        acc = df[df['Date'] == date]['Acc'].values[0]
        acc_df = pd.concat([acc_df, pd.DataFrame({'site_id': [site_id], 'WY': [WY], 'Acc': [acc]})])
    if saving_path:
        acc_df.to_csv(saving_path, index=False)
    return df, acc_df


def compute_var(df: pd.DataFrame,
                forcast_date: str
                ):
    df['Date'] = pd.to_datetime(df['Date'])
    # Define water year
    df['WY'] = df['Date'].dt.year
    mask = (df['Date'].dt.month >= 10) & (df['Date'].dt.month <= 12)
    df.loc[mask, 'WY'] = df.loc[mask, 'WY'] + 1
    grouped_df = df.groupby('WY')
    var_df = pd.DataFrame(columns=['site_id', 'WY', 'Var'])
    for WY, group in grouped_df:
        date = pd.to_datetime(str(WY) + '-' + forcast_date, format='%Y-%m-%d')
        date_90before = date - timedelta(days=91)  # look back starting from the issue date - 1
        group['Date'] = pd.to_datetime(group['Date'], format='%Y-%m-%d')
        group_before_forecast = group[(group['Date'] < date) & (group['Date'] >= date_90before)]
        site_id = df[df['Date'] == date]['site_id'].values[0]
        var = group_before_forecast['Value'].var()
        var_df = pd.concat([var_df, pd.DataFrame({'site_id': [site_id], 'WY': [WY], 'Var': [var]})])
    return var_df


def calculate_mean(df, start_date, end_date):
    mask = (df['Date'] >= start_date) & (df['Date'] < end_date)
    subset = df.loc[mask]
    return subset['Value'].mean()


def calculate_acc(df, start_date, end_date):
    mask = (df['Date'] >= start_date) & (df['Date'] < end_date)
    subset = df.loc[mask]
    return subset['Value'].sum()


def compute_monthly_mean(df: pd.DataFrame,
                         forecast_date: str
                         ):
    df['Date'] = pd.to_datetime(df['Date'])
    # Define water year
    df['WY'] = df['Date'].dt.year
    mask = (df['Date'].dt.month >= 10) & (df['Date'].dt.month <= 12)
    df.loc[mask, 'WY'] = df.loc[mask, 'WY'] + 1

    mean_df = pd.DataFrame(columns=['site_id', 'WY', 'Mean_past3', 'Mean_past2', 'Mean_past1'])
    WYs = list(range(1982, 2023 + 1))
    for WY in WYs:
        df_WY = df[df['WY'] == WY]
        # Calculate the dates 90, 60, and 30 days before the forecast_date
        date = pd.to_datetime(str(WY) + '-' + forecast_date, format='%Y-%m-%d')
        date_90before = date - timedelta(days=91)  # look back starting from the issue date - 1
        date_60before = date - timedelta(days=60)
        date_30before = date - timedelta(days=30)

        # Calculate means for the specified date ranges
        mean_90to60 = calculate_mean(df_WY, date_90before, date_60before)
        mean_60to30 = calculate_mean(df_WY, date_60before, date_30before)
        mean_30tocurrent = calculate_mean(df_WY, date_30before, date)
        mean_90tocurrent = calculate_mean(df_WY, date_90before, date)

        # Update the DataFrame with the calculated means
        site_id = df_WY[df_WY['Date'] == date]['site_id'].values[0]
        mean_df = pd.concat([mean_df, pd.DataFrame({'site_id': [site_id], 'WY': [WY],
                                                    'Mean_past3': [mean_90tocurrent],
                                                    'Mean_past2': [mean_60to30],
                                                    'Mean_past1': [mean_30tocurrent]})])
    return mean_df


def compute_acc_90days(df: pd.DataFrame,
                       forecast_date: str,
                       saving_path: str,
                       mode: str = 'acc',
                       swe: bool = False
                       ):
    '''
    :param df: Predictors time series data frame with
    1st col: "site_id"
    2nd col: "Date" ['%m-%d']. Do not specify the year.
    3rd col: "Value"
    :param forecast_date: str ['mm-dd'], e.g. "01-01","01-15"...
    :param saving_path: path to save the accumulated value df to csv. If None, not save.
    :param mode: str ['acc','mean']. Default: 'acc'
    mode = 'mean' to compute past mean values (designed for Tmax and Tmin)
    :return: accumualted value until the forecast date in the same WY
    '''

    df['Date'] = pd.to_datetime(df['Date'])
    # Define water year
    df['WY'] = df['Date'].dt.year
    mask = (df['Date'].dt.month >= 10) & (df['Date'].dt.month <= 12)
    df.loc[mask, 'WY'] = df.loc[mask, 'WY'] + 1
    WYs = None
    if swe:
        WYs = df.WY.unique()
    if not swe:
        WYs = df.WY.unique()[1:]
    # Assuming your input date is in the format 'mm-dd'
    # input_format = '%m-%d'
    # forecast_date = datetime.strptime(forecast_date, input_format)
    # Iterate over the DataFrame and calculate means for each date range
    if mode == 'acc':
        new_df = pd.DataFrame(columns=['site_id', 'WY', 'Acc'])
    elif mode == 'mean':
        new_df = pd.DataFrame(columns=['site_id', 'WY', 'Mean'])
    for WY in WYs:
        df_WY = df[df['WY'] == WY]
        # Calculate the dates 90, 60, and 30 days before the forecast_date
        date = pd.to_datetime(str(WY) + '-' + forecast_date, format='%Y-%m-%d')
        date_90before = date - timedelta(days=90)

        # Calculate means for the specified date ranges
        mean_90tocurrent = calculate_mean(df_WY, date_90before, date)
        acc_90tocurrent = calculate_acc(df_WY, date_90before, date)

        # Update the DataFrame with the calculated means
        site_id = df_WY[df_WY['Date'] == date]['site_id'].values[0]
        if mode == 'acc':
            new_df = pd.concat([new_df, pd.DataFrame({'site_id': [site_id], 'WY': [WY], 'Acc': [acc_90tocurrent]})])
            # acc_df.to_csv(saving_path, index=False)
        if mode == 'mean':
            new_df = pd.concat([new_df, pd.DataFrame({'site_id': [site_id], 'WY': [WY], 'Mean': [mean_90tocurrent]})])
            # mean_df.to_csv(saving_path, index=False)
    return df, new_df


def calculate_interval_coverage(true_values, lower_quantile_preds, upper_quantile_preds):
    """
    Calculate the interval coverage.

    Parameters:
    - true_values: Array of true values.
    - lower_quantile_preds: Array of predicted values for the lower quantile (e.g., 0.10).
    - upper_quantile_preds: Array of predicted values for the upper quantile (e.g., 0.90).

    Returns:
    - Interval coverage: Proportion of true values that fall within the predicted interval.
    """
    # Check if true values fall within the predicted interval
    within_interval = np.logical_and(true_values >= lower_quantile_preds, true_values <= upper_quantile_preds)

    # Calculate the proportion of true values within the interval
    coverage = np.mean(within_interval)

    return coverage


def update_smoke_submission(df, pred_df_dict, date, site_id, WY, quantile):
    issue_date = str(WY) + '-' + date
    pred_df = pred_df_dict[date][quantile].loc[
        (pred_df_dict[date][quantile]['site_id'] == site_id) & (pred_df_dict[date][quantile]['WY'] == issue_date)
        ]
    df.loc[
        (df['site_id'] == site_id) & (df['issue_date'] == issue_date),
        f'volume_{int(quantile * 100)}'
    ] = pred_df['value'].values
    return df


def get_current_value(df, forecast_date):
    df['Date'] = pd.to_datetime(df['Date'])
    # Define water year
    df['WY'] = df['Date'].dt.year
    mask = (df['Date'].dt.month >= 10) & (df['Date'].dt.month <= 12)
    df.loc[mask, 'WY'] = df.loc[mask, 'WY'] + 1
    WYs = df.WY.unique()
    new_df = pd.DataFrame(columns=['site_id', 'WY', 'Value'])
    for WY in WYs:
        df_WY = df[df['WY'] == WY]
        # Calculate the dates 90, 60, and 30 days before the forecast_date
        date = pd.to_datetime(str(WY) + '-' + forecast_date, format='%Y-%m-%d')
        # Update the DataFrame with the calculated means
        site_id = df_WY[df_WY['Date'] == date]['site_id'].values[0]
        new_df = pd.concat([new_df, pd.DataFrame(
            {'site_id': [site_id], 'WY': [WY], 'Value': [df_WY[df_WY['Date'] == date]['Value'].values[0]]})])
    return new_df


def get_snotel_current_value(df, forecast_date):
    df['Date'] = pd.to_datetime(df['Date'])
    # Define water year
    df['WY'] = df['Date'].dt.year
    mask = (df['Date'].dt.month >= 10) & (df['Date'].dt.month <= 12)
    df.loc[mask, 'WY'] = df.loc[mask, 'WY'] + 1
    WYs = df.WY.unique()
    new_df_list = []
    for WY in WYs:
        df_WY = df[df['WY'] == WY]
        date = pd.to_datetime(str(WY) + '-' + forecast_date, format='%Y-%m-%d')
        if not df_WY[df_WY['Date'] == date].empty:
            site_id = df_WY[df_WY['Date'] == date]['site_id'].values[0]
            row_data = {'site_id': [site_id], 'WY': [WY]}
            for column in df_WY.columns:
                if column not in ['Date', 'site_id', 'WY']:  # Exclude 'Date', 'site_id', and 'WY' columns
                    row_data[column] = [df_WY[df_WY['Date'] == date][column].values[0]]
            new_df_list.append(pd.DataFrame(row_data))
        else:
            site_id = df_WY['site_id'].values[0]
            # Create a row with NaN values for all columns except 'Date', 'site_id', and 'WY'
            nan_row_data = {'site_id': site_id, 'WY': WY}
            for column in df_WY.columns:
                if column not in ['Date', 'site_id', 'WY']:
                    nan_row_data[column] = np.nan
            new_df_list.append(pd.DataFrame(nan_row_data, index=[0]))  # Append a DataFrame with NaN row

    new_df = pd.concat(new_df_list, ignore_index=True).dropna(axis=1, how='any')
    return new_df


# Define a custom scoring function to calculate the total loss
def total_loss(y_true, y_pred):
    quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
    total_loss = 0
    for quantile in quantiles:
        error = y_true - y_pred[:, quantiles.index(quantile)]
        total_loss += np.sum(np.maximum(quantile * error, (quantile - 1) * error))
    return total_loss


def compute_NS(y_true, y_pred):
    # y_true, y_pred: columns in dataframe
    climo = y_true.mean()
    NS = 1 - sum((y_pred - y_true) ** 2) / sum((climo - y_true) ** 2)
    return NS

def compute_RMSE(y_true, y_pred):
    # y_true, y_pred: columns in dataframe
    RMSE = np.sqrt(((y_pred - y_true) ** 2).mean())
    return RMSE


def compute_NRMSE(y_true, y_pred):
    # y_true, y_pred: columns in dataframe
    NRMSE = np.sqrt(((y_pred - y_true) ** 2).mean()) / y_true.mean() * 100
    return NRMSE
    


def calculate_cdf_values(forecast_df, quantiles, observations):
    """
    Calculate CDF values for forecasts and observations.

    Parameters:
    - forecast_df: DataFrame containing forecast quantiles (e.g., forecast_withswe or forecast_woswe)
    - quantiles: List of quantile levels (e.g., [10, 30, 50, 70, 90])
    - observations: Series or array of observed values

    Returns:
    - cdf_values_fcast: CDF values for the forecasts
    - cdf_values_obs: CDF values for the observations
    """
    from scipy.stats import norm
    # Calculate the CDF points corresponding to the quantiles
    cdf_points = np.array(quantiles)
    # Linearly interpolate between the quantile values to estimate the CDF
    quantile_values = forecast_df[['value_{}'.format(int(q * 100)) for q in quantiles]]
    sorted_quantile_values = quantile_values.apply(lambda row: sorted(row), axis=1, result_type='expand')
    sorted_quantile_values_df = pd.DataFrame(sorted_quantile_values.values, columns=quantile_values.columns,
                                             index=quantile_values.index)
    interp_func = np.interp(norm.cdf(np.linspace(0, 1, 100)), cdf_points, sorted_quantile_values_df.mean())

    # Normalize the CDF values to ensure they sum up to 1
    cdf_values_fcast = interp_func / np.sum(interp_func)

    # Find which CDF value the observation corresponds to
    index = np.searchsorted(interp_func, observations.mean())
    # Determine the corresponding CDF
    corresponding_cdf = index / len(interp_func)

    # Calculate the CDF for observations
    # 5 bins are 0-10, 10-30, 30-50, 50-70, 70-90, 90-100
    bins = [0, 10, 30, 50, 70, 90, 100]
    cdf_values_obs = np.ones(6)
    for i in range(len(bins) - 1):
        if corresponding_cdf * 100 < bins[i + 1]:
            break
        else:
            cdf_values_obs[i] = 0

    return cdf_values_obs


def calculate_cdf_climo(observations_train, observations_test):
    """
    Calculate climatological cumulative distribution function (CDF) based on observed values.

    Parameters:
    - observations_train: Series or array of observed values

    Returns:
    - cdf_values_obs: CDF values for the observed values
    """

    # Sort the observations
    from scipy.stats import norm
    observations_sorted = sorted(observations_train)

    # Calculate the number of observations and their corresponding CDF values
    n = len(observations_sorted)
    cdf_values = np.arange(1, n + 1) / n

    # Interpolate the CDF values
    interp_func = np.interp(norm.cdf(np.linspace(0, 1, 100)), cdf_values, observations_sorted)

    # Find which CDF value the observation corresponds to
    index = np.searchsorted(interp_func, observations_test.mean())

    # Determine the corresponding CDF
    corresponding_cdf = index / len(interp_func)

    # Calculate the CDF for observations
    # 6 bins are 0-10, 10-30, 30-50, 50-70, 70-90, 90-100
    bins = [0, 10, 30, 50, 70, 90, 100]
    cdf_values_obs = np.ones(6)
    for i in range(len(bins) - 1):
        if corresponding_cdf * 100 < bins[i + 1]:
            break
        else:
            cdf_values_obs[i] = 0

    return cdf_values_obs


def get_fold_data_by_year(NSF_traintest_df, fold):
    if isinstance(fold[0], tuple):
        train_ranges = fold[:2]
        NSF_train_df = pd.concat(
            [NSF_traintest_df[(NSF_traintest_df['WY'] >= train_start) & (NSF_traintest_df['WY'] <= train_end)] for
             train_start, train_end in train_ranges])
        train_years = np.concatenate([np.arange(train_start, train_end + 1) for train_start, train_end in train_ranges])
    else:
        train_start, train_end = fold[:2]
        NSF_train_df = NSF_traintest_df[
            (NSF_traintest_df['WY'] >= train_start) & (NSF_traintest_df['WY'] <= train_end)]
        train_years = np.arange(train_start, train_end + 1)

    test_start, test_end = fold[2:]
    NSF_test_df = NSF_traintest_df[(NSF_traintest_df['WY'] >= test_start) & (NSF_traintest_df['WY'] <= test_end)]
    test_years = np.arange(test_start, test_end + 1)

    return train_years, test_years, NSF_train_df, NSF_test_df


