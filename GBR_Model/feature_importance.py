import os
import pandas as pd
import Utils
import numpy as np
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns
import pdb

save_test_resutls = True
src_dir = r"C:\Users\yihan\OneDrive - University of Oklahoma\OU\Research\2023-2024 Forecasting"
model_dict = {}

metadata_path = 'data/metadata.csv'
metadata = pd.read_csv(metadata_path)

target_tr_path = 'data/train_1982_2022.csv'
target_df = pd.read_csv(target_tr_path)
target_df = target_df.rename(columns={'year': 'WY'})
target_df['site_id_short'] = target_df['site_id'].apply(
    lambda x: x[0].upper() + x.split('_')[1][0].upper() + x.split('_')[-1][0].upper() if isinstance(x, str) and len(
        x) > 0 else None)

df_by_sites = {}
grouped_dataframes = target_df.groupby('site_id_short')
for site_id_short, group_df in grouped_dataframes:
    df_by_sites[site_id_short] = group_df
    df_by_sites[site_id_short].dropna(inplace=True)

site_id_short = target_df['site_id_short'].unique()
site_id_short = site_id_short[site_id_short != 'DLI']

monthly_NSF_tr_path = 'data/train_monthly_naturalized_flow_1982_2022.csv'
site_id_in_monthlyNSF = pd.read_csv(monthly_NSF_tr_path)['site_id'].unique()
site_id_short_in_monthlyNSF = [Utils.get_site_short_id(x, metadata_path) for x in site_id_in_monthlyNSF]
site_id_short_not_in_monthlyNSF = list(set(target_df['site_id_short'].unique()) - set(site_id_short_in_monthlyNSF))
site_id_not_in_monthlyNSF = list(set(target_df['site_id'].unique()) - set(site_id_in_monthlyNSF))

site_id_short_in_monthlyNSF.remove('SRA')

forecast_date = []
days = ['01', '08', '15', '22']
months = ['01', '02', '03', '04', '05', '06', '07']
for month in months:
    for day in days:
        forecast_date.append(month + '-' + day)

start_year = 1982
end_year = 2021
year_list = list(range(start_year, end_year + 1))
quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]

Data_folder = 'data/Updated'
concatenated_feature_importances = []
heatmap_list = []
swe_cond = 'without_SWE' # todo: manually change to "with_SWE".
folds = [
    ("f1", (1992, 2021, 1982, 1991)),
    ("f2", ((1982, 1991), (2002, 2021), 1992, 2001)),
    ("f3", ((1982, 2001), (2012, 2021), 2002, 2011)),
    ("f4", (1982, 2011, 2012, 2021))
]

# Define the feature names
feature_names = ['PPT', 'Tmax', 'Tmin', 'var(Tmax)', 'var(Tmin)', 'ANS3', 'ANS2', 'ANS1', 'Drainage Area']

# Define the custom features
custom_features = {
    'T': ['Tmax', 'Tmin', 'var(Tmax)', 'var(Tmin)'],
    'ANS': ['ANS3', 'ANS2', 'ANS1']
}

for date in forecast_date:
    print(f'============= forecast date: {date} ===============')

    for site in site_id_short_in_monthlyNSF:
        print('==== ', site)
        fold_feature_importances = []

        for fold_name, fold in folds:
            model_10_path = os.path.join(f'trained_models_single_basin_1982_2021/{swe_cond}',
                                         site + "_" + date + "_" + str(0.1) + "_" + fold_name + "_model.dat")
            model_30_path = os.path.join(f'trained_models_single_basin_1982_2021/{swe_cond}',
                                         site + "_" + date + "_" + str(0.3) + "_" + fold_name + "_model.dat")
            model_50_path = os.path.join(f'trained_models_single_basin_1982_2021/{swe_cond}',
                                         site + "_" + date + "_" + str(0.5) + "_" + fold_name + "_model.dat")
            model_70_path = os.path.join(f'trained_models_single_basin_1982_2021/{swe_cond}',
                                         site + "_" + date + "_" + str(0.7) + "_" + fold_name + "_model.dat")
            model_90_path = os.path.join(f'trained_models_single_basin_1982_2021/{swe_cond}',
                                         site + "_" + date + "_" + str(0.9) + "_" + fold_name + "_model.dat")

            pred_model_10 = load(model_10_path)
            pred_model_30 = load(model_30_path)
            pred_model_50 = load(model_50_path)
            pred_model_70 = load(model_70_path)
            pred_model_90 = load(model_90_path)

            model_dict[date] = [pred_model_10, pred_model_30, pred_model_50, pred_model_70, pred_model_90]

            if swe_cond == 'without_SWE':
                feature_importances_fold = pd.DataFrame(columns=['PPT', 'T', 'ANS', 'Drainage Area'])
                for ind in range(len(model_dict[date])):
                    model = model_dict[date][ind]
                    feature_imp = model.feature_importances_
                    #print(feature_imp.shape)
                    combined_importances = {'PPT': 0, 'T': 0, 'ANS': 0, 'Drainage Area': 0}
                    for i, imp in enumerate(feature_imp):
                        if feature_names[i] in custom_features['T']:
                            combined_importances['T'] += imp
                        elif feature_names[i] in custom_features['ANS']:
                            combined_importances['ANS'] += imp
                        else:
                            combined_importances[feature_names[i]] = imp

                    combined_importances['site_id'] = site
                    combined_importances['issue_date'] = date
                    combined_importances['quantile'] = f'{int(quantiles[ind] * 100)}%'

                    feature_importances_fold = pd.concat([feature_importances_fold, pd.DataFrame([combined_importances])], ignore_index=True)

                fold_feature_importances.append(feature_importances_fold)
            elif swe_cond == 'with_SWE':
                feature_importances_fold = pd.DataFrame(columns=['PPT', 'SWE', 'T', 'ANS', 'Drainage Area'])
                for ind in range(len(model_dict[date])):
                    model = model_dict[date][ind]
                    feature_imp = model.feature_importances_
                    print(feature_imp.shape)
                    num_features = feature_imp.shape[0]
                    num_swe = num_features - 1 - 8  # Total features minus 1 PPT and 8 other features

                    combined_importances = {'PPT': 0, 'SWE': 0, 'T': 0, 'ANS': 0, 'Drainage Area': 0}

                    for i, imp in enumerate(feature_imp):
                        if i == 0:
                            combined_importances['PPT'] = imp
                        elif i == num_features - 1:
                            combined_importances['Drainage Area'] = imp
                        elif i in range(num_features - 4, num_features - 1):
                            combined_importances['ANS'] += imp
                        elif i in range(num_features - 8, num_features - 4):
                            combined_importances['T'] += imp
                        elif 0 < i < num_features - 8:
                            combined_importances['SWE'] += imp

                    combined_importances['site_id'] = site
                    combined_importances['issue_date'] = date
                    combined_importances['quantile'] = f'{int(quantiles[ind] * 100)}%'

                    feature_importances_fold = pd.concat([feature_importances_fold, pd.DataFrame([combined_importances])], ignore_index=True)

                fold_feature_importances.append(feature_importances_fold)


        # Compute the mean feature importances over the folds
        mean_feature_importances = pd.concat(fold_feature_importances).groupby(['site_id', 'issue_date', 'quantile']).mean().reset_index()

        concatenated_feature_importances.append(mean_feature_importances.copy())

# Combine all the feature importances for all sites and dates
feature_importances_all = pd.concat(concatenated_feature_importances, ignore_index=True)
# Convert relevant columns to numeric
if swe_cond == 'without_SWE':
    cols = ['PPT', 'T', 'ANS']
elif swe_cond == 'with_SWE':
     cols = ['PPT', 'SWE', 'T', 'ANS']
for col in cols:
    feature_importances_all[col] = pd.to_numeric(feature_importances_all[col], errors='coerce')
# Melt the dataframe to long format
melted = feature_importances_all.melt(id_vars=['site_id', 'issue_date', 'quantile'],
                                      value_vars=cols, var_name='feature', value_name='importance')

# Group by issue_date, quantile, and feature, then calculate the mean of the importance
average_importances = melted.groupby(['issue_date', 'quantile', 'feature'])['importance'].mean().reset_index()

# Define the number of rows and columns for subplots
num_rows = 7
num_cols = 4

# Create subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 12), sharex=True, sharey=True)

# Flatten the axes array to make it easier to iterate over
axes = axes.flatten()

# Iterate over each issue_date and subplot
for i, (issue_date, df) in enumerate(average_importances.groupby('issue_date')):
    ax = axes[i]
    df_pivot = df.pivot(index='feature', columns='quantile', values='importance')
    sns.heatmap(df_pivot, annot=True, ax=ax, cmap='Reds', vmin=0, vmax=1, linewidths=1, linecolor='black')

    # Customize subplot appearance
    ax.set_title(f"{issue_date}")

    # Set y-ticks at the midpoint between consecutive columns
    ax.set_yticks(np.arange(len(df_pivot.index)) + 0.5)
    ax.set_yticklabels(df_pivot.index, rotation=0, va='center')

    # Set x-ticks at the midpoint between consecutive rows
    ax.set_xticks(np.arange(len(df_pivot.columns)) + 0.5)
    ax.set_xticklabels(df_pivot.columns, rotation=90, ha='center')

    # Hide the colorbar for each subplot
    ax.collections[0].colorbar.remove()
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
    ax.set_xlabel('')
    ax.set_ylabel('')
# Remove empty subplots if any
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
'''
plt.tight_layout()
plt.savefig(f'figures/single_basin_1982_2021/heatmap_{swe_cond}', dpi=600)
'''