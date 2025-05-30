import pdb

import pandas as pd
import warnings
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

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14
save_test_resutls = True
src_dir = r"C:\Users\yihan\OneDrive - University of Oklahoma\OU\Research\2023-2024 Forecasting"
# Note: this code is for forecasting on the most recent previous issue date.
# For example, if running this code on Jan 11, it will make forecast for issue date Jan 8.

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

df_test_result_withswe = pd.DataFrame(
    columns=['site_id', 'WY', 'value_10', 'value_30', 'value_50', 'value_70', 'value_90'])
mean_QL_df_withswe = pd.DataFrame(columns=['site_id', 'issue_date', 'GBR'])
IC_df_withswe = pd.DataFrame(columns=['site_id', 'issue_date', 'GBR'])
mean_df_withswe = pd.DataFrame(
    columns=['site_id', 'issue_date', 'value_10', 'value_30', 'value_50', 'value_70', 'value_90'])
mean_QL_df_withswe['issue_date'] = forecast_date
IC_df_withswe['issue_date'] = forecast_date
mean_df_withswe['issue_date'] = forecast_date

target_all_df = pd.DataFrame(
    columns=['site_id', 'issue_date', 'volume'])
target_all_df['issue_date'] = forecast_date

df_test_result_woswe = pd.DataFrame(
    columns=['site_id', 'WY', 'value_10', 'value_30', 'value_50', 'value_70', 'value_90'])
mean_QL_df_woswe = pd.DataFrame(columns=['site_id', 'issue_date', 'GBR'])
IC_df_woswe = pd.DataFrame(columns=['site_id', 'issue_date', 'GBR'])
mean_df_woswe = pd.DataFrame(
    columns=['site_id', 'issue_date', 'value_10', 'value_30', 'value_50', 'value_70', 'value_90'])
mean_QL_df_woswe['issue_date'] = forecast_date
IC_df_woswe['issue_date'] = forecast_date
mean_df_woswe['issue_date'] = forecast_date

NS_df_withswe = pd.DataFrame(columns=['site_id', 'issue_date', 'NS'])
NS_df_woswe = pd.DataFrame(columns=['site_id', 'issue_date', 'NS'])

target_all_df_list = []
target_allyears_df_list = []
mean_QL_df_withswe_list = []
IC_df_withswe_list = []
mean_QL_df_woswe_list = []
IC_df_woswe_list = []
mean_df_withswe_list = []
mean_df_woswe_list = []
result_df_withswe_list = []
result_df_woswe_list = []
target_train_df_list = []
site_id_short = site_id_short[site_id_short != 'DLI']



# Remove DLI as it ends in June.
# Remove SRR and LRI as they are part in Canada.
site_id_short_in_monthlyNSF = [site for site in site_id_short_in_monthlyNSF if site not in ['DLI', 'SRR', 'LRI']]
train_year = list(range(1982, 2022))
test_year = list(range(1982, 2022))

for site in site_id_short_in_monthlyNSF:
    mean_QL_df_withswe['site_id'] = site
    IC_df_withswe['site_id'] = site
    mean_QL_df_woswe['site_id'] = site
    IC_df_woswe['site_id'] = site
    target_train_df = df_by_sites[site][df_by_sites[site]['WY'].isin(train_year)].reset_index(drop=True)
    target_train_df_list.append(target_train_df)
    for date in forecast_date:
        result_df_withswe = pd.read_csv('results_single_basin_1982_2021/4fold_combined_with_SWE/%s_%s.csv' % (site, date))
        result_df_withswe['issue_date'] = date
        result_df_woswe = pd.read_csv('results_single_basin_1982_2021/4fold_combined_without_SWE/%s_%s.csv' % (site, date))
        result_df_woswe['issue_date'] = date

        NSF_df = Utils.slice_df(df_by_sites[site], 1982, 2021)
        target_df = NSF_df[NSF_df['WY'].isin(test_year)].reset_index(drop=True)
        target_df_temp = target_df.copy()
        target_df_temp['issue_date'] = date
        target_all_df.loc[target_all_df['issue_date'] == date, 'volume'] = target_df['volume'].mean()
        target_all_df.loc[target_all_df['issue_date'] == date, 'site_id'] = target_df['site_id_short'][0]

        result_df_withswe_list.append(result_df_withswe)
        result_df_woswe_list.append(result_df_woswe)
        target_allyears_df_list.append(target_df_temp)

        #################### mean QL and IC ##############################
        # mean QL loss with swe (mean of 5 quantiles and 10 test years)
        mean_QL = Utils.mean_quantile_loss_eval(target_df['volume'],
                                                result_df_withswe[
                                                    ['value_10', 'value_30', 'value_50', 'value_70', 'value_90']],
                                                quantiles)

        IC = Utils.calculate_interval_coverage(target_df['volume'], result_df_withswe['value_10'],
                                               result_df_withswe['value_90'])

        mean_QL_df_withswe.loc[mean_QL_df_withswe['issue_date'] == date, 'GBR'] = mean_QL
        IC_df_withswe.loc[IC_df_withswe['issue_date'] == date, 'GBR'] = IC

        # mean QL loss without swe (mean of 5 quantiles and 10 test years)
        mean_QL = Utils.mean_quantile_loss_eval(target_df['volume'],
                                                result_df_woswe[
                                                    ['value_10', 'value_30', 'value_50', 'value_70', 'value_90']],
                                                quantiles)

        IC = Utils.calculate_interval_coverage(target_df['volume'], result_df_withswe['value_10'],
                                               result_df_woswe['value_90'])

        mean_QL_df_woswe.loc[mean_QL_df_woswe['issue_date'] == date, 'GBR'] = mean_QL
        IC_df_woswe.loc[IC_df_woswe['issue_date'] == date, 'GBR'] = IC

        #################### NS ##############################
        NS_withswe = Utils.compute_NS(target_df.volume, result_df_withswe['value_50'])
        NS_woswe = Utils.compute_NS(target_df.volume, result_df_woswe['value_50'])
        NS_df_withswe = NS_df_withswe._append({'site_id': site, 'issue_date': date, 'NS': NS_withswe},
                                              ignore_index=True)
        NS_df_woswe = NS_df_woswe._append({'site_id': site, 'issue_date': date, 'NS': NS_woswe}, ignore_index=True)

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

    target_all_df_list.append(target_all_df.copy())

    mean_QL_df_withswe_list.append(mean_QL_df_withswe.copy())
    IC_df_withswe_list.append(IC_df_withswe.copy())

    mean_QL_df_woswe_list.append(mean_QL_df_woswe.copy())
    IC_df_woswe_list.append(IC_df_woswe.copy())

    mean_df_withswe_list.append(mean_df_withswe.copy())
    mean_df_woswe_list.append(mean_df_woswe.copy())

target_df_all = pd.concat(target_all_df_list, axis=0, ignore_index=True)

pred_df_all_withswe = pd.concat(result_df_withswe_list, axis=0, ignore_index=True)
pred_df_all_woswe = pd.concat(result_df_woswe_list, axis=0, ignore_index=True)
target_df_allyears = pd.concat(target_allyears_df_list, axis=0, ignore_index=True)
target_train_df_allyears = pd.concat(target_train_df_list, axis=0, ignore_index=True)

mean_QL_df_all_withswe = pd.concat(mean_QL_df_withswe_list, axis=0, ignore_index=True)
IC_df_all_withswe = pd.concat(IC_df_withswe_list, axis=0, ignore_index=True)
mean_QL_df_all_withswe.to_csv('results_single_basin_1982_2021/with_SWE/mean_QL_all.csv', index=False)
IC_df_all_withswe.to_csv('results_single_basin_1982_2021/with_SWE/IC_df_all.csv', index=False)

mean_QL_df_all_woswe = pd.concat(mean_QL_df_woswe_list, axis=0, ignore_index=True)
IC_df_all_woswe = pd.concat(IC_df_woswe_list, axis=0, ignore_index=True)
mean_QL_df_all_woswe.to_csv('results_single_basin_1982_2021/without_SWE/mean_QL_all.csv', index=False)
IC_df_all_woswe.to_csv('results_single_basin_1982_2021/without_SWE/IC_df_all.csv', index=False)

mean_all_withswe = pd.concat(mean_df_withswe_list, axis=0, ignore_index=True)
mean_all_woswe = pd.concat(mean_df_woswe_list, axis=0, ignore_index=True)

#######################################################################
#### PLOT ####

sns.set(style="whitegrid", font='Arial', rc={'font.size': 16})
sns.set(rc={'axes.edgecolor': 'black'})
plt.rcParams['axes.facecolor'] = 'white'
sns.set(style="whitegrid", font='Arial')
size = 14
sns.set(rc={'axes.edgecolor': 'black'})
plt.rcParams['axes.facecolor'] = 'white'

mean_per_issue_date_target = target_df_all.groupby(['issue_date'])['volume'].mean().reset_index()
mean_QL_per_issue_date_withswe = mean_QL_df_all_withswe.groupby(['issue_date'])['GBR'].mean().reset_index()
mean_QL_per_issue_date_woswe = mean_QL_df_all_woswe.groupby(['issue_date'])['GBR'].mean().reset_index()
mean_per_issue_date_withswe = mean_all_withswe.groupby(['issue_date'])[['value_10', 'value_30',
                                                                        'value_50', 'value_70',
                                                                        'value_90']].mean().reset_index()
mean_per_issue_date_woswe = mean_all_woswe.groupby(['issue_date'])[['value_10', 'value_30',
                                                                    'value_50', 'value_70',
                                                                    'value_90']].mean().reset_index()

mean_NS_df_withswe = NS_df_withswe.groupby('issue_date')['NS'].mean().reset_index()
mean_NS_df_woswe = NS_df_woswe.groupby('issue_date')['NS'].mean().reset_index()
with pd.ExcelWriter('results_single_basin_1982_2021/mean_NSS_per_issue_date.xlsx') as writer:
    mean_NS_df_withswe.to_excel(writer, sheet_name='with_SWE', index=False)
    mean_NS_df_woswe.to_excel(writer, sheet_name='without_SWE', index=False)
pdb.set_trace()
#####################################################################
####################### Fig 1. Mean QL series #######################
#####################################################################
plt.figure(figsize=(15, 8))
sns.lineplot(x='issue_date', y='GBR', data=mean_QL_per_issue_date_withswe, marker='o', color='blue', linewidth=1.5,
             label='With SWE')
sns.lineplot(x='issue_date', y='GBR', data=mean_QL_per_issue_date_woswe, marker='o', color='red', linewidth=1.5,
             label='Without SWE')

# Add labels and title
plt.xlabel('Issue Date', fontsize=15)
plt.ylabel('Quantile Loss (KAF)', fontsize=15)
# plt.ylim(50, 100)
plt.legend(fontsize=size)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right', fontsize=13)
plt.yticks(fontsize=13)
# Set x-axis limits based on data range
plt.xlim(mean_QL_per_issue_date_withswe['issue_date'].min(), mean_QL_per_issue_date_withswe['issue_date'].max())
plt.grid(color='black', linestyle='--', linewidth=0.5)
#plt.savefig('figures/single_basin_1982_2021/quantile_loss_series.png', dpi=600)
pdb.set_trace()
#####################################################################
####################### Fig 2. Mean quantile series #################
#####################################################################
plt.figure(figsize=(15, 8))
width = 2
mean_per_issue_date_withswe['issue_date'] = pd.to_datetime(mean_per_issue_date_withswe['issue_date'], format='%m-%d')
mean_per_issue_date_woswe['issue_date'] = pd.to_datetime(mean_per_issue_date_woswe['issue_date'], format='%m-%d')
mean_per_issue_date_target['issue_date'] = pd.to_datetime(mean_per_issue_date_target['issue_date'], format='%m-%d')
# with SWE
sns.lineplot(x='issue_date', y='value_50', data=mean_per_issue_date_withswe,
             label='With SWE: 50th Quantile', color='blue', linewidth=width)
# sns.lineplot(x='issue_date', y='value_10', data=mean_per_issue_date_withswe, color='cornflowerblue', linestyle='-',
#             linewidth=width)
# sns.lineplot(x='issue_date', y='value_90', data=mean_per_issue_date_withswe, color='cornflowerblue', linestyle='-',
#             linewidth=width)
# sns.lineplot(x='issue_date', y='value_30', data=mean_per_issue_date_withswe, color='blue', linestyle='-',
#             linewidth=width)
# sns.lineplot(x='issue_date', y='value_70', data=mean_per_issue_date_withswe, color='blue', linestyle='-',
#             linewidth=width)
plt.fill_between(x=mean_per_issue_date_withswe['issue_date'], y1=pd.to_numeric(mean_per_issue_date_withswe['value_10']),
                 y2=pd.to_numeric(mean_per_issue_date_withswe['value_90']), color='cornflowerblue', alpha=0.2,
                 label='With SWE: Between 10th and 90th Quantiles')
plt.fill_between(x=mean_per_issue_date_withswe['issue_date'], y1=pd.to_numeric(mean_per_issue_date_withswe['value_30']),
                 y2=pd.to_numeric(mean_per_issue_date_withswe['value_70']), color='blue', alpha=0.25,
                 label='With SWE: Between 30th and 70th Quantiles')
# without SWE
sns.lineplot(x='issue_date', y='value_50', data=mean_per_issue_date_woswe,
             label='Without SWE: 50th Quantile', color='maroon', linestyle='--', linewidth=width)
# sns.lineplot(x='issue_date', y='value_10', data=mean_per_issue_date_woswe, color='lightcoral', linestyle='--',
#             linewidth=width)
# sns.lineplot(x='issue_date', y='value_90', data=mean_per_issue_date_woswe, color='lightcoral', linestyle='--',
#             linewidth=width)
# sns.lineplot(x='issue_date', y='value_30', data=mean_per_issue_date_woswe, color='darkred', linestyle='--',
#             linewidth=width)
# sns.lineplot(x='issue_date', y='value_70', data=mean_per_issue_date_woswe, color='darkred', linestyle='--',
#             linewidth=width)
plt.fill_between(x=mean_per_issue_date_woswe['issue_date'], y1=pd.to_numeric(mean_per_issue_date_woswe['value_10']),
                 y2=pd.to_numeric(mean_per_issue_date_woswe['value_90']), color='lightcoral', alpha=0.2,
                 label='Without SWE: Between 10th and 90th Quantiles')
plt.fill_between(x=mean_per_issue_date_woswe['issue_date'], y1=pd.to_numeric(mean_per_issue_date_woswe['value_30']),
                 y2=pd.to_numeric(mean_per_issue_date_woswe['value_70']), color='darkred', alpha=0.25,
                 label='Without SWE: Between 30th and 70th Quantiles')

# Plot the lines
sns.lineplot(x='issue_date', y='volume', data=mean_per_issue_date_target,
             label='Observation', color='black', linewidth=width)

# Add labels and title
plt.xlabel('Issue Date', fontsize=15)
plt.ylabel('Cumulative NSF (KAF)', fontsize=size)

# Rotate x-axis labels for better readability
plt.xticks(mean_per_issue_date_withswe['issue_date'].unique(),
           mean_per_issue_date_withswe['issue_date'].dt.strftime('%m-%d').unique(),
           rotation=45, ha='right', fontsize=13)
plt.yticks(fontsize=13)

plt.xlim(mean_per_issue_date_withswe['issue_date'].min(), mean_per_issue_date_withswe['issue_date'].max())
# Show the legend
legend = plt.legend(fontsize=12)
legend.get_frame().set_alpha(0.7)  # Set the alpha for the legend background
plt.grid(color='black', linestyle='--', linewidth=0.5)
#plt.savefig('figures/single_basin_1982_2021/uncertainties_series.png', dpi=600)
pdb.set_trace()
#####################################################################
############## Fig 2.a Quantile series for each site# ###############
#####################################################################

width = 2
# with SWE
for site in site_id_short_in_monthlyNSF:
    df_site_withswe = mean_all_withswe[mean_all_withswe["site_id"] == site]
    df_site_withswe['issue_date'] = pd.to_datetime(df_site_withswe['issue_date'], format='%m-%d')
    df_site_woswe = mean_all_woswe[mean_all_woswe["site_id"] == site]
    df_site_woswe['issue_date'] = pd.to_datetime(df_site_woswe['issue_date'], format='%m-%d')
    df_site_target = target_df_all[target_df_all["site_id"] == site]
    df_site_target['issue_date'] = pd.to_datetime(df_site_target['issue_date'], format='%m-%d')

    plt.figure(figsize=(12, 6))
    sns.lineplot(x='issue_date', y='value_50', data=df_site_withswe,
                 label='With SWE: 50th Quantile', color='navy', linewidth=width)
    sns.lineplot(x='issue_date', y='value_10', data=df_site_withswe, color='cornflowerblue', linestyle='-',
                 linewidth=width)
    sns.lineplot(x='issue_date', y='value_90', data=df_site_withswe, color='cornflowerblue', linestyle='-',
                 linewidth=width)
    sns.lineplot(x='issue_date', y='value_30', data=df_site_withswe, color='blue', linestyle='-', linewidth=width)
    sns.lineplot(x='issue_date', y='value_70', data=df_site_withswe, color='blue', linestyle='-', linewidth=width)
    plt.fill_between(x=df_site_withswe['issue_date'], y1=pd.to_numeric(df_site_withswe['value_10']),
                     y2=pd.to_numeric(df_site_withswe['value_90']), color='cornflowerblue', alpha=0.2,
                     label='With SWE: Between 10th and 90th Quantiles')
    plt.fill_between(x=df_site_withswe['issue_date'], y1=pd.to_numeric(df_site_withswe['value_30']),
                     y2=pd.to_numeric(df_site_withswe['value_70']), color='blue', alpha=0.25,
                     label='With SWE: Between 30th and 70th Quantiles')
    # without SWE
    sns.lineplot(x='issue_date', y='value_50', data=df_site_woswe,
                 label='Without SWE: 50th Quantile', color='maroon', linestyle='--', linewidth=width)
    sns.lineplot(x='issue_date', y='value_10', data=df_site_woswe, color='lightcoral', linestyle='--', linewidth=width)
    sns.lineplot(x='issue_date', y='value_90', data=df_site_woswe, color='lightcoral', linestyle='--', linewidth=width)
    sns.lineplot(x='issue_date', y='value_30', data=df_site_woswe, color='darkred', linestyle='--', linewidth=width)
    sns.lineplot(x='issue_date', y='value_70', data=df_site_woswe, color='darkred', linestyle='--', linewidth=width)
    plt.fill_between(x=df_site_woswe['issue_date'], y1=pd.to_numeric(df_site_woswe['value_10']),
                     y2=pd.to_numeric(df_site_woswe['value_90']), color='lightcoral', alpha=0.2,
                     label='Without SWE: Between 10th and 90th Quantiles')
    plt.fill_between(x=df_site_woswe['issue_date'], y1=pd.to_numeric(df_site_woswe['value_30']),
                     y2=pd.to_numeric(df_site_woswe['value_70']), color='darkred', alpha=0.25,
                     label='Without SWE: Between 30th and 70th Quantiles')

    # Plot the lines
    sns.lineplot(x='issue_date', y='volume', data=df_site_target,
                 label='Observation', color='black', linewidth=width)

    # Add labels and title
    plt.xlabel('Issue Date', fontsize=15)
    plt.ylabel('Cumulative NSF (KAF)', fontsize=size)

    # Rotate x-axis labels for better readability
    plt.xticks(df_site_withswe['issue_date'].unique(),
               df_site_withswe['issue_date'].dt.strftime('%m-%d').unique(),
               rotation=45, ha='right', fontsize=13)
    plt.yticks(fontsize=13)

    plt.xlim(df_site_withswe['issue_date'].min(), df_site_withswe['issue_date'].max())
    # Show the legend
    legend = plt.legend(fontsize=12)
    legend.get_frame().set_alpha(0.7)  # Set the alpha for the legend background
    plt.grid(color='black', linestyle='--', linewidth=0.5)
    plt.title(site)
    plt.savefig('figures/single_basin_1982_2021/uncertainties_series_%s.png' % site, dpi=600)
pdb.set_trace()
#####################################################################
#################### Fig 3. Mean QL box plot series #################
#####################################################################
# Set the color palette for the plot
palette = {
    'With SWE': (0, 0, 1, 0.3),  # Light blue for With SWE
    'Without SWE': (0.5450980392156862, 0.0, 0.0, 0.3),  # Dark red for Without SWE
}

sns.set(style="whitegrid", font='Arial')
sns.set(rc={'axes.edgecolor': 'black'})
plt.rcParams['axes.facecolor'] = 'white'

size = 14  # Adjust the size according to your preference

# Set up the figure
plt.figure(figsize=(15, 8))

# Boxplot with SWE
sns.boxplot(
    data=mean_QL_df_all_withswe, x="issue_date", y="GBR",
    flierprops={"marker": "x", "markerfacecolor": palette['With SWE'], "markeredgecolor": palette['With SWE']},
    boxprops={"facecolor": palette['With SWE'], "linewidth": 2},
    whiskerprops={"color": palette['With SWE'], "linewidth": 2},
    medianprops={"color": palette['With SWE'], "linewidth": 2}
)
# Lineplot with SWE
sns.lineplot(
    x='issue_date', y='GBR', data=mean_QL_per_issue_date_withswe,
    marker='o', color='blue', linewidth=2, label='With SWE'
)

# Boxplot without SWE
sns.boxplot(
    data=mean_QL_df_all_woswe, x="issue_date", y="GBR",
    flierprops={"marker": "x", "markerfacecolor": palette['Without SWE'], "markeredgecolor": palette['Without SWE']},
    boxprops={"facecolor": palette['Without SWE'], "linewidth": 2},
    whiskerprops={"color": palette['Without SWE'], "linewidth": 2},
    medianprops={"color": "darkred", "linewidth": 2}
)

# Lineplot without SWE
sns.lineplot(
    x='issue_date', y='GBR', data=mean_QL_per_issue_date_woswe,
    marker='o', color='red', linewidth=2, linestyle='--', label='Without SWE'
)

# Set labels and ticks
plt.xlabel('Issue Date', fontsize=15)
plt.ylabel('Quantile loss (KAF)', fontsize=15)
plt.xticks(rotation=45, ha='right', fontsize=13)
plt.yticks(fontsize=13)
plt.ylim(0, 200)
plt.grid(color='black', linestyle='--', linewidth=0.5)

# Create legend entries manually using patches with alpha
blue_patch = mpatches.Patch(color=palette['With SWE'], label='With SWE', edgecolor='black')
red_patch = mpatches.Patch(color=palette['Without SWE'], label='Without SWE', edgecolor='black')

# Create legend entries for lines
blue_line = mlines.Line2D([], [], color='blue', marker='o', linewidth=2, label='With SWE')
red_line = mlines.Line2D([], [], color='red', marker='o', linewidth=2, linestyle='--', label='Without SWE')

# Adjust the order of legend entries
legend_handles = [blue_patch, red_patch, blue_line, red_line]

# Create legend
legend = plt.legend(handles=legend_handles, loc='upper right', fontsize=size)

# Set alpha for legend
for patch in legend.get_patches():
    patch.set_alpha(0.5)
plt.savefig('figures/single_basin_1982_2021/boxplot_combined_QL.png', dpi=600)
pdb.set_trace()

#####################################################################
################## Fig 5. NS box plot series ######################
#####################################################################

# Set the color palette for the plot
palette = {
    'With SWE': (0, 0, 1, 0.3),  # Light blue for With SWE
    'Without SWE': (0.5450980392156862, 0.0, 0.0, 0.3),  # Dark red for Without SWE
}

sns.set(style="whitegrid", font='Arial')
sns.set(rc={'axes.edgecolor': 'black'})
plt.rcParams['axes.facecolor'] = 'white'

size = 14  # Adjust the size according to your preference

# Set up the figure
plt.figure(figsize=(15, 8))

# Boxplot with SWE
sns.boxplot(
    data=NS_df_withswe, x="issue_date", y="NS",
    flierprops={"marker": "x", "markerfacecolor": palette['With SWE'], "markeredgecolor": palette['With SWE']},
    boxprops={"facecolor": palette['With SWE'], "linewidth": 2},
    whiskerprops={"color": palette['With SWE'], "linewidth": 2},
    medianprops={"color": palette['With SWE'], "linewidth": 2}
)
# Lineplot with SWE
sns.lineplot(
    x='issue_date', y='NS', data=NS_df_withswe.groupby(['issue_date'])['NS'].mean().reset_index(),
    marker='o', color='blue', linewidth=2, label='With SWE'
)

# Boxplot without SWE
sns.boxplot(
    data=NS_df_woswe, x="issue_date", y="NS",
    flierprops={"marker": "x", "markerfacecolor": palette['Without SWE'], "markeredgecolor": palette['Without SWE']},
    boxprops={"facecolor": palette['Without SWE'], "linewidth": 2},
    whiskerprops={"color": palette['Without SWE'], "linewidth": 2},
    medianprops={"color": "darkred", "linewidth": 2}
)

# Lineplot without SWE
sns.lineplot(
    x='issue_date', y='NS', data=NS_df_woswe.groupby(['issue_date'])['NS'].mean().reset_index(),
    marker='o', color='red', linewidth=2, linestyle='--', label='Without SWE'
)

# Set labels and ticks
plt.xlabel('Issue Date', fontsize=15)
plt.ylabel('NSS', fontsize=15)
plt.xticks(rotation=45, ha='right', fontsize=13)
plt.yticks(fontsize=13)
plt.ylim(-0.5, 1)
plt.grid(color='black', linestyle='--', linewidth=0.5)

# Create legend entries manually using patches with alpha
blue_patch = mpatches.Patch(color=palette['With SWE'], label='With SWE', edgecolor='black')
red_patch = mpatches.Patch(color=palette['Without SWE'], label='Without SWE', edgecolor='black')

# Create legend entries for lines
blue_line = mlines.Line2D([], [], color='blue', marker='o', linewidth=2, label='With SWE')
red_line = mlines.Line2D([], [], color='red', marker='o', linewidth=2, linestyle='--', label='Without SWE')

# Adjust the order of legend entries
legend_handles = [blue_patch, red_patch, blue_line, red_line]

# Create legend
legend = plt.legend(handles=legend_handles, loc='lower right', fontsize=size)

# Set alpha for legend
for patch in legend.get_patches():
    patch.set_alpha(0.5)
plt.savefig('figures/single_basin_1982_2021/boxplot_combined_NSE.png', dpi=600)
pdb.set_trace()
#####################################################################
################## Fig 4. scatter plot of each site's ###############
#############       prediction vs observation         ###############
#####################################################################
issue_dates = ['01-01', '04-01', '07-01']

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 18), sharex=False, sharey=False)
plt.rcParams.update({'font.size': 25,
                     'axes.labelsize': 25,
                     'axes.titlesize': 25,
                     'xtick.labelsize': 20,
                     'ytick.labelsize': 20})

for idx, issue_date in enumerate(issue_dates):
    pred_swe = mean_all_withswe[mean_all_withswe['issue_date'] == issue_date]
    pred_woswe = mean_all_woswe[mean_all_woswe['issue_date'] == issue_date]
    obs = target_df_all[target_df_all['issue_date'] == issue_date]

    # Merge the two dataframes on 'site_id' to have all the required data in one dataframe
    merged_df = pd.merge(pred_swe, pred_woswe, on='site_id', suffixes=('_pred_swe', '_pred_woswe'))
    merged_df = pd.merge(merged_df, obs, on='site_id', suffixes=('_merged', '_obs'))

    min_lim = [0, 200, 1000]
    max_lim = [200, 1000, 4000]
    marker_size = 80

    for idx_2, (min_, max_) in enumerate(zip(min_lim, max_lim)):
        row = idx // len(issue_dates) + idx_2
        col = (idx + idx_2 * len(issue_dates)) % len(issue_dates)
        # filtered_df = merged_df[(merged_df['volume'] >= min_) & (merged_df['volume'] <= max_) &
        #                        (merged_df['value_50_pred_swe'] >= min_) & (merged_df['value_50_pred_swe'] <= max_) &
        #                        (merged_df['value_50_pred_woswe'] >= min_) & (
        #                                    merged_df['value_50_pred_woswe'] <= max_)].reset_index(drop=True)
        filtered_df = merged_df[(merged_df['volume'] >= min_) & (merged_df['volume'] <= max_)].reset_index(drop=True)

        sns.scatterplot(x='volume', y='value_50_pred_swe', data=filtered_df, color='blue', marker='o', s=marker_size,
                        ax=axes[row, col])
        sns.scatterplot(x='volume', y='value_50_pred_woswe', data=filtered_df, color='red', marker='o',
                        s=marker_size, ax=axes[row, col])

        for i in range(len(filtered_df)):
            axes[row, col].text(filtered_df.loc[i, 'volume'], filtered_df.loc[i, 'value_50_pred_swe'],
                                filtered_df['site_id'][i],
                                color='blue', fontname='Arial')
            axes[row, col].text(filtered_df.loc[i, 'volume'], filtered_df.loc[i, 'value_50_pred_woswe'],
                                filtered_df['site_id'][i],
                                color='red', fontname='Arial')
        if row == 1 and col == 0:
            axes[row, col].set_ylabel('Forecast (KAF)',fontsize=25)
            axes[row, col].set_xlabel('')
        elif row == 2 and col == 1:
            axes[row, col].set_xlabel('Observation (KAF)',fontsize=25)
            axes[row, col].set_ylabel('')
        else:
            axes[row, col].set_xlabel('')
            axes[row, col].set_ylabel('')

        if row == 0 and col == 0:
            axes[row, col].set_title('Issue Date %s ' % issue_dates[0], fontsize=25)
        elif row == 0 and col == 1:
            axes[row, col].set_title('Issue Date %s ' % issue_dates[1], fontsize=25)
        elif row == 0 and col == 2:
            axes[row, col].set_title('Issue Date %s ' % issue_dates[2], fontsize=25)

        axes[row, col].set_xlim(min_, max_)
        axes[row, col].set_ylim(min_, max_)
        axes[row, col].plot([min_, max_], [min_, max_], linestyle='--', color='grey')
        axes[row, col].grid(color='black', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('figures/single_basin/scatter_plots_shared_axes.png', dpi=600)
pdb.set_trace()
#####################################################################
################# Fig 5. Evaluation statistics NS map ###############
#############       prediction vs observation         ###############
#####################################################################
# Define the geographic extent and other plot settings
lon_min, lon_max = -125, -103
lat_min, lat_max = 30, 50
vmin, vmax = -1, 1
plt.rcParams.update({'font.size': 20,
                     'axes.labelsize': 20,
                     'axes.titlesize': 20,
                     'xtick.labelsize': 16,
                     'ytick.labelsize': 16})
low_flow_sites = ['DRI', 'RRI', 'CRF', 'WRO', 'VRV', 'TPI', 'PRP'] # 'SRA' removed in 1982_2021 training
mid_flow_sites = ['ARD', 'PRI', 'ORD', 'GRD', 'FRI', 'YRM', 'SRS', 'BRI']
high_flow_sites = ['BRB', 'MRT', 'HHI', 'SRH']

flow_categories = {
    'low': low_flow_sites,
    'mid': mid_flow_sites,
    'high': high_flow_sites
}

watershed_list = high_flow_sites

for category, watershed_list in flow_categories.items():
    for issue_date in forecast_date:
        print(f"===== issue date: {issue_date} for {category} flow ====")
        print("===== issue date: ", issue_date, " ====")
        NS_df_withswe_sid = pd.DataFrame(columns=['site_id', 'latitude', 'longitude', 'NS'])  # single issue date
        NS_df_withswe_sid['site_id'] = site_id_short_in_monthlyNSF

        NS_df_woswe_sid = pd.DataFrame(columns=['site_id', 'latitude', 'longitude', 'NS'])  # single issue date
        NS_df_woswe_sid['site_id'] = site_id_short_in_monthlyNSF

        for site_id in site_id_short_in_monthlyNSF:
            NS_df_withswe_sid.loc[NS_df_withswe_sid['site_id'] == site_id, 'latitude'] = \
                metadata.loc[metadata['site_id_short'] == site_id, 'latitude'].values[0]
            NS_df_withswe_sid.loc[NS_df_withswe_sid['site_id'] == site_id, 'longitude'] = \
                metadata.loc[metadata['site_id_short'] == site_id, 'longitude'].values[0]
            NS_df_withswe_sid.loc[NS_df_withswe_sid['site_id'] == site_id, 'NS'] = \
                NS_df_withswe.loc[(NS_df_withswe['issue_date'] == issue_date) &
                                  (NS_df_withswe['site_id'] == site_id), 'NS'].values[0]

            NS_df_woswe_sid.loc[NS_df_woswe_sid['site_id'] == site_id, 'latitude'] = \
                metadata.loc[metadata['site_id_short'] == site_id, 'latitude'].values[0]
            NS_df_woswe_sid.loc[NS_df_woswe_sid['site_id'] == site_id, 'longitude'] = \
                metadata.loc[metadata['site_id_short'] == site_id, 'longitude'].values[0]
            NS_df_woswe_sid.loc[NS_df_woswe_sid['site_id'] == site_id, 'NS'] = \
                NS_df_woswe.loc[(NS_df_woswe['issue_date'] == issue_date) &
                                (NS_df_woswe['site_id'] == site_id), 'NS'].values[0]

        # Filter DataFrames to include only the sites in the current flow category
        NS_df_withswe_sid = NS_df_withswe_sid[NS_df_withswe_sid['site_id'].isin(watershed_list)].reset_index(drop=True)
        NS_df_woswe_sid = NS_df_woswe_sid[NS_df_woswe_sid['site_id'].isin(watershed_list)].reset_index(drop=True)

        print(NS_df_withswe_sid)

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        # Set extent
        ax.set_extent([lon_min, lon_max, lat_min, lat_max])

        # Plot CONUS
        ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor='black')
        # Scatter plot of NS values
        sc = ax.scatter(NS_df_withswe_sid["longitude"], NS_df_withswe_sid["latitude"], c=NS_df_withswe_sid['NS'],
                        cmap='seismic', s=100, transform=ccrs.PlateCarree(), zorder=2, vmin=vmin, vmax=vmax)

        # Add colorbar
        cbar = plt.colorbar(sc, ax=ax, extend='both')
        cbar.set_label('NS Value')

        # Adding labels
        for i, txt in enumerate(NS_df_withswe_sid['site_id']):
            if txt in watershed_list:
                ax.text(NS_df_withswe_sid["longitude"][i], NS_df_withswe_sid["latitude"][i], txt,
                        transform=ccrs.PlateCarree(),
                        ha='center', va='bottom')

        plt.title('Issue Date %s' % issue_date)
        plt.savefig(f'figures/single_basin_1982_2021/NS_map/{category}flow/with_SWE/NS_map_withSWE_%s.png' % issue_date, dpi=600)
        plt.close()

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        # Set extent
        ax.set_extent([lon_min, lon_max, lat_min, lat_max])

        # Plot CONUS
        ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor='black')
        # Scatter plot of NS values
        sc = ax.scatter(NS_df_woswe_sid["longitude"], NS_df_woswe_sid["latitude"], c=NS_df_woswe_sid['NS'],
                        cmap='seismic', s=100, transform=ccrs.PlateCarree(), zorder=2, vmin=vmin, vmax=vmax)

        # Add colorbar
        cbar = plt.colorbar(sc, ax=ax, extend='both')
        cbar.set_label('NS Value')

        # Adding labels
        for i, txt in enumerate(NS_df_woswe_sid['site_id']):
            if txt in watershed_list:
                ax.text(NS_df_woswe_sid["longitude"][i], NS_df_woswe_sid["latitude"][i], txt,
                        transform=ccrs.PlateCarree(),
                        ha='center', va='bottom')

        plt.title('Issue Date %s' % issue_date)
        plt.savefig(f'figures/single_basin_1982_2021/NS_map/{category}flow/without_SWE/NS_map_withoutSWE_%s.png' % issue_date, dpi=600)
        print(NS_df_woswe_sid)
        plt.close()

        # Difference plot
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        # Set extent
        ax.set_extent([lon_min, lon_max, lat_min, lat_max])

        # Plot CONUS
        ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor='black')

        # Scatter plot of difference in NS values with edges for specific points
        sc = ax.scatter(NS_df_woswe_sid["longitude"], NS_df_woswe_sid["latitude"],
                        c=NS_df_withswe_sid['NS'] - NS_df_woswe_sid['NS'],
                        cmap='seismic', s=150, transform=ccrs.PlateCarree(), zorder=2, vmin=vmin, vmax=vmax)

        # Scatter plot of NS values with edges for specific points
        sc_edges = ax.scatter(
            NS_df_woswe_sid.loc[NS_df_woswe_sid['site_id'].isin(watershed_list), "longitude"],
            NS_df_woswe_sid.loc[NS_df_woswe_sid['site_id'].isin(watershed_list), "latitude"],
            c=NS_df_withswe_sid.loc[NS_df_withswe_sid['site_id'].isin(watershed_list), 'NS'] -
              NS_df_woswe_sid.loc[NS_df_woswe_sid['site_id'].isin(watershed_list), 'NS'],
            cmap='seismic', s=150, transform=ccrs.PlateCarree(), zorder=2,
            vmin=vmin, vmax=vmax, edgecolor='black', linewidths=2)

        # Add colorbar
        # cbar = plt.colorbar(sc, ax=ax)
        # cbar.set_label('NS Value')

        # Adding labels for specific points
        #for i, txt in enumerate(NS_df_woswe_sid['site_id']):
        #    if txt in watershed_list:
        #        ax.text(NS_df_woswe_sid["longitude"][i], NS_df_woswe_sid["latitude"][i], txt,
        #                transform=ccrs.PlateCarree(), ha='center', va='bottom')

        plt.title('Issue Date %s' % issue_date)
        plt.savefig(f'figures/single_basin_1982_2021/NS_map/{category}flow/NS_map_diff_%s.png' % issue_date, dpi=600)
        plt.close()
pdb.set_trace()
#####################################################################
################# Fig 6. Evaluation statistics QL map ###############
#############       prediction vs observation         ###############
#####################################################################
lon_min, lon_max = -125, -103
lat_min, lat_max = 30, 50
vmin, vmax = 0, 50
plt.rcParams.update({'font.size': 20,
                     'axes.labelsize': 20,
                     'axes.titlesize': 20,
                     'xtick.labelsize': 16,
                     'ytick.labelsize': 16})

for issue_date in forecast_date:
    print("===== issue date: ", issue_date, " ====")
    QL_df_withswe_sid = pd.DataFrame(columns=['site_id', 'latitude', 'longitude', 'QL'])  # single issue date
    QL_df_withswe_sid['site_id'] = site_id_short_in_monthlyNSF

    QL_df_woswe_sid = pd.DataFrame(columns=['site_id', 'latitude', 'longitude', 'QL'])  # single issue date
    QL_df_woswe_sid['site_id'] = site_id_short_in_monthlyNSF

    for site_id in site_id_short_in_monthlyNSF:
        QL_df_withswe_sid.loc[QL_df_withswe_sid['site_id'] == site_id, 'latitude'] = \
            metadata.loc[metadata['site_id_short'] == site_id, 'latitude'].values[0]
        QL_df_withswe_sid.loc[QL_df_withswe_sid['site_id'] == site_id, 'longitude'] = \
            metadata.loc[metadata['site_id_short'] == site_id, 'longitude'].values[0]
        QL_df_withswe_sid.loc[QL_df_withswe_sid['site_id'] == site_id, 'QL'] = \
            mean_QL_df_all_withswe.loc[(mean_QL_df_all_withswe['issue_date'] == issue_date) &
                                       (mean_QL_df_all_withswe['site_id'] == site_id), 'GBR'].values[0]

        QL_df_woswe_sid.loc[QL_df_woswe_sid['site_id'] == site_id, 'latitude'] = \
            metadata.loc[metadata['site_id_short'] == site_id, 'latitude'].values[0]
        QL_df_woswe_sid.loc[QL_df_woswe_sid['site_id'] == site_id, 'longitude'] = \
            metadata.loc[metadata['site_id_short'] == site_id, 'longitude'].values[0]
        QL_df_woswe_sid.loc[QL_df_woswe_sid['site_id'] == site_id, 'QL'] = \
            mean_QL_df_all_woswe.loc[(mean_QL_df_all_woswe['issue_date'] == issue_date) &
                                     (mean_QL_df_all_woswe['site_id'] == site_id), 'GBR'].values[0]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Set extent
    ax.set_extent([lon_min, lon_max, lat_min, lat_max])

    # Plot CONUS
    ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor='black')
    # Scatter plot of NS values
    sc = ax.scatter(QL_df_withswe_sid["longitude"], QL_df_withswe_sid["latitude"], c=QL_df_withswe_sid['QL'],
                    cmap='turbo', s=100, transform=ccrs.PlateCarree(), zorder=2, vmin=vmin, vmax=vmax)

    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax, extend='both')
    cbar.set_label('QL Value')

    # Adding labels
    for i, txt in enumerate(QL_df_withswe_sid['site_id']):
        if txt in ['FRI', 'YRM', 'SRS', 'BRI']:
            ax.text(QL_df_withswe_sid["longitude"][i], QL_df_withswe_sid["latitude"][i], txt,
                    transform=ccrs.PlateCarree(),
                    ha='center', va='bottom')

    plt.title('Issue Date %s' % issue_date)
    plt.savefig('figures/single_basin/QL_map/with_SWE/QL_map_withSWE_%s.png' % issue_date, dpi=600)
    plt.close()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Set extent
    ax.set_extent([lon_min, lon_max, lat_min, lat_max])

    # Plot CONUS
    ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor='black')
    # Scatter plot of NS values
    sc = ax.scatter(QL_df_woswe_sid["longitude"], QL_df_woswe_sid["latitude"], c=QL_df_woswe_sid['QL'],
                    cmap='turbo', s=100, transform=ccrs.PlateCarree(), zorder=2, vmin=vmin, vmax=vmax)

    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax, extend='both')
    cbar.set_label('QL Value')

    # Adding labels
    for i, txt in enumerate(QL_df_woswe_sid['site_id']):
        if txt in ['FRI', 'YRM', 'SRS', 'BRI']:
            ax.text(QL_df_woswe_sid["longitude"][i], QL_df_woswe_sid["latitude"][i], txt,
                    transform=ccrs.PlateCarree(),
                    ha='center', va='bottom')

    plt.title('Issue Date %s' % issue_date)
    plt.savefig('figures/single_basin/QL_map/without_SWE/QL_map_withoutSWE_%s.png' % issue_date, dpi=600)
    plt.close()

    # Difference plot
    vmin, vmax = -20, 20
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Set extent
    ax.set_extent([lon_min, lon_max, lat_min, lat_max])

    # Plot CONUS
    ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor='black')

    # Scatter plot of difference in NS values with edges for specific points
    sc = ax.scatter(QL_df_woswe_sid["longitude"], QL_df_woswe_sid["latitude"],
                    c=QL_df_withswe_sid['QL'] - QL_df_woswe_sid['QL'],
                    cmap='seismic', s=150, transform=ccrs.PlateCarree(), zorder=2, vmin=vmin, vmax=vmax)

    # Scatter plot of NS values with edges for specific points
    # watershed_list = ['FRI', 'YRM', 'SRS', 'BRI']
    watershed_list = ['ARD', 'PRI', 'ORD', 'GRD', 'DRI', 'RRI', 'CRF', 'WRO',
                      'SRA', 'PRP', 'VRV', 'TPI']
    sc_edges = ax.scatter(
        QL_df_woswe_sid.loc[QL_df_woswe_sid['site_id'].isin(watershed_list), "longitude"],
        QL_df_woswe_sid.loc[QL_df_woswe_sid['site_id'].isin(watershed_list), "latitude"],
        c=QL_df_withswe_sid.loc[QL_df_withswe_sid['site_id'].isin(watershed_list), 'QL'] -
          QL_df_woswe_sid.loc[QL_df_woswe_sid['site_id'].isin(watershed_list), 'QL'],
        cmap='seismic', s=150, transform=ccrs.PlateCarree(), zorder=2,
        vmin=vmin, vmax=vmax, edgecolor='black', linewidths=2)

    # Add colorbar
    # cbar = plt.colorbar(sc, ax=ax)
    # cbar.set_label('NS Value')

    # Adding labels for specific points
    for i, txt in enumerate(QL_df_woswe_sid['site_id']):
       if txt in ['FRI', 'YRM', 'SRS', 'BRI']:
            ax.text(QL_df_woswe_sid["longitude"][i], QL_df_woswe_sid["latitude"][i], txt,
                    transform=ccrs.PlateCarree(), ha='center', va='bottom')

    plt.title('Issue Date %s' % issue_date)
    plt.savefig('figures/single_basin/QL_map/QL_map_diff_%s.png' % issue_date, dpi=600)
    plt.close()

    # Difference plot
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Set extent
    ax.set_extent([lon_min, lon_max, lat_min, lat_max])

    # Plot CONUS
    ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor='black')

    # Scatter plot of difference in NS values
    sc = ax.scatter(NS_df_woswe_sid["longitude"], NS_df_woswe_sid["latitude"],
                    c=NS_df_withswe_sid['NS'] - NS_df_woswe_sid['NS'],
                    cmap='seismic', s=100, edgecolor='black', linewidths=2,
                    transform=ccrs.PlateCarree(), zorder=2, vmin=vmin, vmax=vmax)

    # Add colorbar
    #cbar = plt.colorbar(sc, ax=ax, extend='both')
    #cbar.set_label('Difference in NS Value')

    # Adding labels
    #for i, txt in enumerate(NS_df_woswe_sid['site_id']):
    #    ax.text(NS_df_woswe_sid["longitude"].iloc[i], NS_df_woswe_sid["latitude"].iloc[i], txt,
    #            transform=ccrs.PlateCarree(), ha='center', va='bottom')

    plt.title(f'Issue Date {issue_date} - {category.capitalize()} Flow (Difference)')
    #plt.savefig(f'figures/single_basin_1982_2021/NS_map/{category}flow/NS_map_diff_{issue_date}.png', dpi=600)
    plt.close()
pdb.set_trace()
#####################################################################
################# Fig 7. Empirical CDF curve            #############
#############       prediction vs observation         ###############
# target_df_allyears, pred_df_all_withswe,  pred_df_all_woswe
# Sample data
issue_dates = ['01-01', '04-01']

# Define line styles
line_styles = ['-', '--', ':']

# Define colors for terciles
tercile_colors = ['blue']

# Create a new figure
plt.figure(figsize=(10, 10))
#tercile_colors = ['red']
site_id_list = target_df_allyears['site_id'].unique()
for idx, issue_date in enumerate(issue_dates):
    observations_sum = np.zeros(10)
    tercile_forecasts_sum = np.zeros((2, 5))
    site_num = len(site_id_list)
    for site_id in site_id_list:
        observations = target_df_allyears[
            (target_df_allyears["issue_date"] == issue_date) & (target_df_allyears["site_id"] == site_id)]["volume"]
        forecasts_withswe = pred_df_all_withswe[
            (pred_df_all_withswe["issue_date"] == issue_date) & (pred_df_all_withswe["site_id"] == site_id)]["value_50"]
        forecasts_woswe = \
            pred_df_all_woswe[
                (pred_df_all_woswe["issue_date"] == issue_date) & (pred_df_all_woswe["site_id"] == site_id)][
                "value_50"]

        if len(observations) != len(forecasts_withswe):
            print("Length mismatch for site_id: ", site_id)
            site_num -= 1
            continue

        # Step 1: Sort the observations and forecasts
        observations_sorted = sorted(observations)
        forecasts_sorted = [f for _, f in sorted(zip(observations, forecasts_woswe))]

        # Step 2: Divide the observations into terciles
        n = len(observations_sorted)
        tercile_size = 5
        terciles = [observations_sorted[i:i + tercile_size] for i in range(0, n, tercile_size)]

        # Assign labels based on terciles
        labels = []
        for obs in observations:
            if obs <= terciles[0][-1]:
                labels.append('Low')
            elif obs <= terciles[1][-1]:
                labels.append('Medium')
            else:
                labels.append('High')

        # Add the labels to the DataFrame
        observations['observation_type'] = labels

        observations_sum += np.array(observations_sorted)

        # Step 3: Calculate the empirical CDF for observations and forecasts within each tercile
        for i, tercile_observations in enumerate(terciles, start=1):
            tercile_forecasts = [forecasts_sorted[observations_sorted.index(obs)] for obs in tercile_observations]
            print(i)
            print(tercile_forecasts)
            # Calculate CDF for forecasts
            forecast_n = len(tercile_forecasts)
            forecast_cdf = np.arange(1, forecast_n + 1) / forecast_n

            tercile_forecasts_sum[i - 1:] += np.array(tercile_forecasts)

    # Plot the CDFs for each tercile with different line styles and the same color for each issue date
    for i, tercile_observations in enumerate(terciles, start=1):
        line_style = line_styles[idx % len(line_styles)]
        plt.plot(tercile_forecasts_sum[i - 1] / site_num, forecast_cdf, label=f"Issue Date: {issue_date} - Tercile {i}",
                 linestyle=line_style, color=tercile_colors[0], linewidth=2,
                 marker='o', markersize=5)

    # Plot the CDF for observations with a marker
    #obs_cdf = np.arange(1, n + 1) / n
    #obs_color = [tercile_colors[i % len(tercile_colors)] for i in range(len(observations_sorted))]
    #plt.step(observations_sum / site_num, obs_cdf, where='post', linestyle='-', color='black',
    #         linewidth=2, markersize=6, marker='s')

# Add labels and legend
plt.xlabel('April-July Volume (KAF)', fontsize=14)
plt.ylabel('Cumulative Probability', fontsize=14)
# plt.legend(loc='lower right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# Show plot
plt.tight_layout()
plt.savefig(
    r'C:\Users\yihan\OneDrive\OU\Class\24spring\METR-5743 Prediction Evaluation\final_paper\Figures\empirical_cdf.png',
    dpi=600)
plt.show()
pdb.set_trace()
#####################################################################
################# Fig 8. RPSS            #############
# Define terciles
issue_dates = ['01-01', '04-01']
# Define line styles
line_styles = ['-', '--', ':']
# Define colors for terciles
tercile_colors = ['b', 'r']

# Create a new figure
# plt.figure(figsize=(6, 5))
site_id_list = target_df_allyears['site_id'].unique()
RPSS_withswe_list = []
RPSS_woswe_list = []

for idx, issue_date in enumerate(forecast_date):
    site_num = len(site_id_list)
    rps_withswe = 0
    rps_woswe = 0
    rps_climo = 0
    for site_id in site_id_list:
        observations = target_df_allyears[
            (target_df_allyears["issue_date"] == issue_date) & (target_df_allyears["site_id"] == site_id)]["volume"]
        observations_train = target_train_df_allyears[
            target_train_df_allyears["site_id"] == site_id]["volume"]

        forecasts_withswe = pred_df_all_withswe[
            (pred_df_all_withswe["issue_date"] == issue_date) & (pred_df_all_withswe["site_id"] == site_id)]
        forecasts_woswe = pred_df_all_woswe[
            (pred_df_all_woswe["issue_date"] == issue_date) & (pred_df_all_woswe["site_id"] == site_id)]

        cdf_values_obs_withswe = Utils.calculate_cdf_values(forecasts_withswe, quantiles, observations)
        cdf_values_obs_woswe = Utils.calculate_cdf_values(forecasts_woswe, quantiles, observations)
        rps_withswe += np.sum((np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.0]) - cdf_values_obs_withswe) ** 2)
        rps_woswe += np.sum((np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.0]) - cdf_values_obs_woswe) ** 2)

        # Compute the RPS_climo
        cdf_values_obs = Utils.calculate_cdf_climo(observations_train, observations)
        rps_climo += np.sum((np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.0]) - cdf_values_obs) ** 2)

    rps_withswe_mean = rps_withswe / site_num
    rps_woswe_mean = rps_woswe / site_num
    rps_climo_mean = rps_climo / site_num
    RPSS_withswe_list.append((rps_withswe_mean - rps_climo_mean) / (0 - rps_climo_mean) * 100)
    RPSS_woswe_list.append((rps_woswe_mean - rps_climo_mean) / (0 - rps_climo_mean) * 100)

df_RPSS = pd.DataFrame({
    'Issue_Date': forecast_date,
    'RPSS_withswe': RPSS_withswe_list,
    'RPSS_woswe': RPSS_woswe_list
})

plt.figure(figsize=(10, 6))
# Plot RPSS values for forecasts with SWE
plt.plot(df_RPSS['Issue_Date'], df_RPSS['RPSS_withswe'], marker='o', linestyle='-', label='RPSS with SWE', color='blue')
# Plot RPSS values for forecasts without SWE
plt.plot(df_RPSS['Issue_Date'], df_RPSS['RPSS_woswe'], marker='o', linestyle='-', label='RPSS without SWE', color='red')
# Add a zero line
plt.axhline(y=0, color='k', linestyle='--')
# Set labels and title
plt.xlabel('Issue Date')
plt.ylabel('RPSS (%)')
# Add legend
plt.legend()
# Customize the x-axis labels
plt.xticks(rotation=45)
# Show grid
plt.grid(True)
# Show plot
plt.tight_layout()
plt.savefig(
    r'C:\Users\yihan\OneDrive\OU\Class\24spring\METR-5743 Prediction Evaluation\final_paper\Figures\RPSS_timeseries.png',
    dpi=600)
plt.show()
pdb.set_trace()
#####################################################################
################# Fig 9. Reliability diagram            #############
#####################################################################
issue_dates = ['01-01']
# Define line styles
line_styles = ['-', '--', ':']
# Define colors for terciles
tercile_colors = ['b', 'r']
plt.rcParams.update({'font.size': 14,
                     'axes.labelsize': 14,
                     'axes.titlesize': 14,
                     'xtick.labelsize': 14,
                     'ytick.labelsize': 14})
# Create a new figure
# plt.figure(figsize=(6, 5))
site_id_list = target_df_allyears['site_id'].unique()
corresponding_cdf_withswe_list = []
corresponding_cdf_woswe_list = []

for idx, issue_date in enumerate(issue_dates):
    site_num = len(site_id_list)
    rps_withswe = 0
    rps_woswe = 0
    rps_climo = 0
    for site_id in site_id_list:
        observations = target_df_allyears[
            (target_df_allyears["issue_date"] == issue_date) & (target_df_allyears["site_id"] == site_id)]["volume"]
        observations_train = target_train_df_allyears[
            target_train_df_allyears["site_id"] == site_id]["volume"]

        forecasts_withswe = pred_df_all_withswe[
            (pred_df_all_withswe["issue_date"] == issue_date) & (pred_df_all_withswe["site_id"] == site_id)]
        forecasts_woswe = pred_df_all_woswe[
            (pred_df_all_woswe["issue_date"] == issue_date) & (pred_df_all_woswe["site_id"] == site_id)]

        observations_sorted = sorted(observations_train)
        # Calculate the number of observations and their corresponding CDF values
        n = len(observations_sorted)
        cdf_values = np.arange(1, n + 1) / n
        # Interpolate the CDF values
        interp_func = np.interp(norm.cdf(np.linspace(0, 1, 100)), cdf_values, observations_sorted)
        # Find which CDF value the observation corresponds to
        index_withswe = np.searchsorted(interp_func,
                                        forecasts_withswe[
                                            ['value_10', 'value_30', 'value_50', 'value_70', 'value_90']].mean()
                                        )
        index_woswe = np.searchsorted(interp_func,
                                      forecasts_woswe[
                                          ['value_10', 'value_30', 'value_50', 'value_70', 'value_90']].mean()
                                      )
        index_climo = np.searchsorted(interp_func,
                                      observations_train.mean()
                                      )

        # Determine the corresponding CDF
        corresponding_cdf_withswe = index_withswe / len(interp_func)
        corresponding_cdf_woswe = index_woswe / len(interp_func)
        corresponding_cdf_climo = index_climo / len(interp_func)

        corresponding_cdf_withswe_list.append(corresponding_cdf_withswe)
        corresponding_cdf_woswe_list.append(corresponding_cdf_woswe)

df_rel_withswe = pd.DataFrame({
    'rel_withswe_10': np.array(corresponding_cdf_withswe_list)[:, 0],
    'rel_withswe_30': np.array(corresponding_cdf_withswe_list)[:, 1],
    'rel_withswe_50': np.array(corresponding_cdf_withswe_list)[:, 2],
    'rel_withswe_70': np.array(corresponding_cdf_withswe_list)[:, 3],
    'rel_withswe_90': np.array(corresponding_cdf_withswe_list)[:, 4]
})

df_rel_woswe = pd.DataFrame({
    'rel_woswe_10': np.array(corresponding_cdf_woswe_list)[:, 0],
    'rel_woswe_30': np.array(corresponding_cdf_woswe_list)[:, 1],
    'rel_woswe_50': np.array(corresponding_cdf_woswe_list)[:, 2],
    'rel_woswe_70': np.array(corresponding_cdf_woswe_list)[:, 3],
    'rel_woswe_90': np.array(corresponding_cdf_woswe_list)[:, 4]
})

x_values = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
y_values_withswe = df_rel_withswe.mean()
y_values_woswe = df_rel_woswe.mean()

# Create the plot with markers and connected lines
plt.figure(figsize=(5, 5))
plt.plot(x_values, y_values_withswe, color='blue', marker='o', linestyle='-', label='With SWE')
plt.plot(x_values, y_values_woswe, color='red', marker='o', linestyle='-', label='Without SWE')

# Plot horizontal climo line
plt.axhline(y=corresponding_cdf_climo, color='green', linestyle='--', label='Climatology')

# Plot vertical climo line
plt.axvline(x=corresponding_cdf_climo, color='green', linestyle='--')


# Plot diagonal line with slope 1
plt.plot([0, 1], [0, 1], color='black', linestyle='--')

# Set labels and title
plt.xlabel('Forecast Probability')
plt.ylabel('Relative Probability of Observations')
plt.title(f'Issue Date {issue_dates[0]}')
# Set x-axis and y-axis limits
plt.xlim([0, 1])
plt.ylim([0, 1])

# Rotate x-axis labels for better visibility
plt.xticks(np.arange(0, 1.1, 0.1))
# Rotate x-axis labels for better visibility
plt.xticks(rotation=45)
# Add legend
plt.legend()
# Show plot
plt.grid(True)
plt.tight_layout()
plt.savefig(
    r'C:\Users\yihan\OneDrive\OU\Class\24spring\METR-5743 Prediction Evaluation\final_paper\Figures\reliability_diagram_%s.png' %
    issue_dates[0],
    dpi=600)
plt.show()
