# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:31:17 2025

@author: yue0004
"""
# This is the Python script to plot the spatial map of feature importance of SWE in a compact layout
# Last Modified on January 28, 2025
###################################################################################
#%% import packages
# import pdb
import pandas as pd
# import warnings
# import a customized module
import sys
sys.path.append('E:/USBR_Snow_Forecast/Fcst_Model')
import Utils
# import seaborn as sns
# import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import matplotlib.lines as mlines
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# from scipy.stats import norm
# from scipy.stats import gumbel_r
# from mpl_toolkits.axes_grid1 import make_axes_locatable

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
# days = ['01', '08', '15', '22']
days = ['01']
months = ['01', '02', '03', '04', '05', '06', '07']

for month in months:
    for day in days:
        forecast_date.append(month + '-' + day)

#%% Define the geographic extent and other plot settings
# Define the geographic extent and other plot settings
lon_min, lon_max = -125, -103
lat_min, lat_max = 30, 50
# vmin, vmax = -25, 25
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

#%% import FIresults
# FIwith SWE
FI_df_withswe_all = pd.read_csv('E:/USBR_Snow_Forecast/Results/FeatureImportance_AllData_withSWE.csv')
# extract results for 50% quantile
FI_df_withswe = FI_df_withswe_all[(FI_df_withswe_all['quantile']=='50%') & (FI_df_withswe_all['feature']=='SWE')]
# Define marker styles for each category
marker_styles = {
    'low': 'o',  # Circle for low flow
    'mid': '^',  # Triangle for mid flow
    'high': 's'  # Square for high flow
}

# Loop through forecast issue dates
for issue_date in forecast_date:
    print(f"===== Processing issue date: {issue_date} =====")

    # Create a single figure for all categories
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Set map extent and add state boundaries
    ax.set_extent([lon_min, lon_max, lat_min, lat_max])
    ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor='black')

    # Set colorbar range
    vmin, vmax = 0, 1

    # Plot each flow category on the same map
    scatter_plots = []  # To store scatter plots for legend
    for category, sites in flow_categories.items():
        # Prepare data for this category
        FI_df_withswe_sid = pd.DataFrame(columns=['site_id', 'latitude', 'longitude', 'importance'])
        FI_df_withswe_sid['site_id'] = site_id_short_in_monthlyNSF

        for site_id in site_id_short_in_monthlyNSF:
            # Extract latitude and longitude from metadata
            FI_df_withswe_sid.loc[FI_df_withswe_sid['site_id'] == site_id, 'latitude'] = \
                metadata.loc[metadata['site_id_short'] == site_id, 'latitude'].values[0]
            FI_df_withswe_sid.loc[FI_df_withswe_sid['site_id'] == site_id, 'longitude'] = \
                metadata.loc[metadata['site_id_short'] == site_id, 'longitude'].values[0]

            # Extract FI importance for the site
            FI_df_withswe_sid.loc[FI_df_withswe_sid['site_id'] == site_id, 'importance'] = \
                FI_df_withswe.loc[
                    (FI_df_withswe['issue_date'] == issue_date) &
                    (FI_df_withswe['site_id'] == site_id), 'importance'
                ].values[0]

        # Filter data for current flow category
        category_data = FI_df_withswe_sid[FI_df_withswe_sid['site_id'].isin(sites)].reset_index(drop=True)

        # Scatter plot for this category
        sc = ax.scatter(
            category_data["longitude"], category_data["latitude"],
            c=category_data["importance"],
            cmap='Reds', s=250, transform=ccrs.PlateCarree(),
            zorder=2, vmin=vmin, vmax=vmax,
            marker=marker_styles[category],  # Marker style for the category
            edgecolor='black', linewidths=1.5, label=f'{category.capitalize()} Flow'
        )
        scatter_plots.append(sc)

    # # Add colorbar
    # cbar = plt.colorbar(sc, ax=ax, extend='neither')
    # cbar.set_label('FI Importance')

    # # Add a legend for the categories
    # ax.legend(
    #     handles=scatter_plots,
    #     labels=[f'{cat.capitalize()} Flow' for cat in flow_categories.keys()],
    #     loc='lower right',
    #     title='Flow Categories'
    # )

    # Set title and save the plot
    plt.title(f'NSE Feature Importance - Issue Date {issue_date}')
    plt.savefig(f'E:/USBR_Snow_Forecast/Figure/Spatial/FI/FI_map_SWE_withSWE_allcategories_{issue_date}.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'E:/USBR_Snow_Forecast/Figure/Spatial/FI/FI_map_SWE_withSWE_allcategories_{issue_date}.eps', format='eps', dpi=600, bbox_inches='tight')
    plt.close()