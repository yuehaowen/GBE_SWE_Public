# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:46:06 2025

@author: yue0004
"""
# This is the python script to plot the spatial map of Quantile Loss differences (different styles)
# Plot sites with different flow categories together
# Last Modified on January 28, 2025
#%% import packages
import pdb
import pandas as pd
import warnings
# import a customized module
import sys
sys.path.append('E:/USBR_Snow_Forecast/Fcst_Model')
import Utils
# import seaborn as sns
# import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import cartopy.crs as ccrs
import cartopy.feature as cfeature

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

#%% import QL results
# QL with SWE
QL_df_withswe = pd.read_csv('E:/USBR_Snow_Forecast/Results/mean_QL_withSWE_all.csv')
# QL without SWE
QL_df_woswe = pd.read_csv('E:/USBR_Snow_Forecast/Results/mean_QL_withoutSWE_all.csv')

#%% Plot the figures for GBR-SWE
# Create a dictionary to map flow categories to marker shapes
marker_styles = {
    'low': 'o',    # circle
    'mid': '^',    # triangle
    'high': 's'    # square
}
# set value limits
vmin, vmax = 0, 320
# For each forecast issue date
for issue_date in forecast_date:
    print(f"===== Processing issue date: {issue_date} ====")
    
    # Create figure for with-SWE plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Set extent and add CONUS states
    ax.set_extent([lon_min, lon_max, lat_min, lat_max])
    ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor='black')
    
    # Create empty lists for legend
    scatter_plots = []
    legend_labels = []
    
    # Plot each flow category
    for category, sites in flow_categories.items():
        # Prepare data for this category
        QL_df_withswe_sid = pd.DataFrame(columns=['site_id', 'latitude', 'longitude', 'QL'])
        QL_df_withswe_sid['site_id'] = site_id_short_in_monthlyNSF
        
        # Fill in the data (same as your original code)
        for site_id in site_id_short_in_monthlyNSF:
            QL_df_withswe_sid.loc[QL_df_withswe_sid['site_id'] == site_id, 'latitude'] = \
                metadata.loc[metadata['site_id_short'] == site_id, 'latitude'].values[0]
            QL_df_withswe_sid.loc[QL_df_withswe_sid['site_id'] == site_id, 'longitude'] = \
                metadata.loc[metadata['site_id_short'] == site_id, 'longitude'].values[0]
            QL_df_withswe_sid.loc[QL_df_withswe_sid['site_id'] == site_id, 'QL'] = \
                QL_df_withswe.loc[(QL_df_withswe['issue_date'] == issue_date) &
                                  (QL_df_withswe['site_id'] == site_id), 'QL'].values[0]
        
        # Filter for current category
        category_data = QL_df_withswe_sid[QL_df_withswe_sid['site_id'].isin(sites)].reset_index(drop=True)
        
        # Plot this category with its specific marker
        sc = ax.scatter(category_data["longitude"], 
                       category_data["latitude"],
                       c=category_data['QL'],
                       marker=marker_styles[category],
                       s=150,
                       cmap='seismic',
                       transform=ccrs.PlateCarree(),
                       vmin=vmin,
                       vmax=vmax,
                       label=f'{category.capitalize()} Flow')
        
        # Add site labels
        for i, txt in enumerate(category_data['site_id']):
            ax.text(category_data["longitude"][i],
                   category_data["latitude"][i],
                   txt,
                   transform=ccrs.PlateCarree(),
                   ha='center',
                   va='bottom')
        
        scatter_plots.append(sc)
        legend_labels.append(f'{category.capitalize()} Flow')
    
    # Add colorbar and legend
    cbar = plt.colorbar(scatter_plots[0], ax=ax, extend='both')
    cbar.set_label('QL Value')
    
    # Add legend
    legend_elements = [plt.scatter([], [], marker=marker_styles[cat],
                                 c='gray', s=150,
                                 label=f'{cat.capitalize()} Flow')
                      for cat in flow_categories.keys()]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.title(f'QL Values (with SWE) - Issue Date {issue_date}')
    
    # Save figures
    plt.savefig(f'E:/USBR_Snow_Forecast/Figure/Spatial/QL/QL_map_withSWE_combined_{issue_date}.png',
                dpi=600, bbox_inches='tight')
    plt.savefig(f'E:/USBR_Snow_Forecast/Figure/Spatial/QL/QL_map_withSWE_combined_{issue_date}.eps',
                format='eps', dpi=600, bbox_inches='tight')
    plt.close()
    
#%% Plot the spatial difference figure for GBR-NoSWE
# set value limits
vmin, vmax = 0, 320
# For each forecast issue date
for issue_date in forecast_date:
    print(f"===== Processing issue date: {issue_date} ====")
    
    # Create figure for with-SWE plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Set extent and add CONUS states
    ax.set_extent([lon_min, lon_max, lat_min, lat_max])
    ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor='black')
    
    # Create empty lists for legend
    scatter_plots = []
    legend_labels = []
    
    # Plot each flow category
    for category, sites in flow_categories.items():
        # Prepare data for this category
        QL_df_woswe_sid = pd.DataFrame(columns=['site_id', 'latitude', 'longitude', 'QL'])
        QL_df_woswe_sid['site_id'] = site_id_short_in_monthlyNSF
        
        # Fill in the data (same as your original code)
        for site_id in site_id_short_in_monthlyNSF:
            # extract latitude of the site
            QL_df_woswe_sid.loc[QL_df_woswe_sid['site_id'] == site_id, 'latitude'] = \
                metadata.loc[metadata['site_id_short'] == site_id, 'latitude'].values[0]
            # extract longitude of the site
            QL_df_woswe_sid.loc[QL_df_woswe_sid['site_id'] == site_id, 'longitude'] = \
                metadata.loc[metadata['site_id_short'] == site_id, 'longitude'].values[0]
            # extract QL of the site
            QL_df_woswe_sid.loc[QL_df_woswe_sid['site_id'] == site_id, 'QL'] = \
                QL_df_woswe.loc[(QL_df_woswe['issue_date'] == issue_date) &
                                  (QL_df_woswe['site_id'] == site_id), 'QL'].values[0]
        
        # Filter for current category
        category_data = QL_df_woswe_sid[QL_df_woswe_sid['site_id'].isin(sites)].reset_index(drop=True)
        
        # Plot this category with its specific marker
        sc = ax.scatter(category_data["longitude"], 
                       category_data["latitude"],
                       c=category_data['QL'],
                       marker=marker_styles[category],
                       s=150,
                       cmap='seismic',
                       transform=ccrs.PlateCarree(),
                       vmin=vmin,
                       vmax=vmax,
                       label=f'{category.capitalize()} Flow')
        
        # Add site labels
        for i, txt in enumerate(category_data['site_id']):
            ax.text(category_data["longitude"][i],
                   category_data["latitude"][i],
                   txt,
                   transform=ccrs.PlateCarree(),
                   ha='center',
                   va='bottom')
        
        scatter_plots.append(sc)
        legend_labels.append(f'{category.capitalize()} Flow')
    
    # Add colorbar and legend
    cbar = plt.colorbar(scatter_plots[0], ax=ax, extend='both')
    cbar.set_label('QL Value')
    
    # Add legend
    legend_elements = [plt.scatter([], [], marker=marker_styles[cat],
                                 c='gray', s=150,
                                 label=f'{cat.capitalize()} Flow')
                      for cat in flow_categories.keys()]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.title(f'QL Values (without SWE) - Issue Date {issue_date}')
    
    # Save figures
    plt.savefig(f'E:/USBR_Snow_Forecast/Figure/Spatial/QL/QL_map_withoutSWE_combined_{issue_date}.png',
                dpi=600, bbox_inches='tight')
    plt.savefig(f'E:/USBR_Snow_Forecast/Figure/Spatial/QL/QL_map_withoutSWE_combined_{issue_date}.eps',
                format='eps', dpi=600, bbox_inches='tight')
    plt.close()

#%% Plot the spatial difference figure for delta QL between GBR-SWE and GBR-NoSWE
# calculate the difference in QL
# QL_df_diff = QL_df_withswe.copy()
# QL_df_diff['QL'] = QL_df_withswe['QL'] - QL_df_woswe['QL']
vmin, vmax = -50, 50
# For each forecast issue date
for issue_date in forecast_date:
    print(f"===== Processing issue date: {issue_date} ====")
    
    # Create figure for with-SWE plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Set extent and add CONUS states
    ax.set_extent([lon_min, lon_max, lat_min, lat_max])
    ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor='black')
    
    # Create empty lists for legend
    scatter_plots = []
    legend_labels = []
    
    # Plot each flow category
    for category, sites in flow_categories.items():
        # Prepare data for this category
        QL_df_diff_sid = pd.DataFrame(columns=['site_id', 'latitude', 'longitude', 'QL'])
        QL_df_diff_sid['site_id'] = site_id_short_in_monthlyNSF
        
        # Fill in the data (same as your original code)
        for site_id in site_id_short_in_monthlyNSF:
            # Extract latitude of the site
            QL_df_diff_sid.loc[QL_df_diff_sid['site_id'] == site_id, 'latitude'] = \
                metadata.loc[metadata['site_id_short'] == site_id, 'latitude'].values[0]
            
            # Extract longitude of the site
            QL_df_diff_sid.loc[QL_df_diff_sid['site_id'] == site_id, 'longitude'] = \
                metadata.loc[metadata['site_id_short'] == site_id, 'longitude'].values[0]
            
            # Calculate QL difference for this site and issue date
            QL_with_swe = QL_df_withswe.loc[
                (QL_df_withswe['issue_date'] == issue_date) &
                (QL_df_withswe['site_id'] == site_id), 'QL'].values[0]
            
            QL_wo_swe = QL_df_woswe.loc[
                (QL_df_woswe['issue_date'] == issue_date) &
                (QL_df_woswe['site_id'] == site_id), 'QL'].values[0]
            
            # Store the difference
            QL_df_diff_sid.loc[QL_df_diff_sid['site_id'] == site_id, 'QL'] = \
                QL_with_swe - QL_wo_swe
        
        # Filter for current category
        category_data = QL_df_diff_sid[QL_df_diff_sid['site_id'].isin(sites)].reset_index(drop=True)
        
        # Plot this category with its specific marker
        sc = ax.scatter(category_data["longitude"], 
                       category_data["latitude"],
                       c=category_data['QL'],
                       marker=marker_styles[category],
                       s=250,
                       cmap='seismic',
                       edgecolor='black',  # Add black edges to the points
                       linewidth=1,  # Line width for the edges
                       transform=ccrs.PlateCarree(),
                       vmin=vmin,
                       vmax=vmax,
                       label=f'{category.capitalize()} Flow')
        
    #     # Add site labels
    #     for i, txt in enumerate(category_data['site_id']):
    #         ax.text(category_data["longitude"][i],
    #                category_data["latitude"][i],
    #                txt,
    #                transform=ccrs.PlateCarree(),
    #                ha='center',
    #                va='bottom')
        
    #     scatter_plots.append(sc)
    #     legend_labels.append(f'{category.capitalize()} Flow')
    
    # # Add colorbar and legend
    # cbar = plt.colorbar(scatter_plots[0], ax=ax, extend='both')
    # cbar.set_label('QL Diff Value')
    
    # # Add legend
    # legend_elements = [plt.scatter([], [], marker=marker_styles[cat],
    #                              c='gray', s=150,
    #                              label=f'{cat.capitalize()} Flow')
    #                   for cat in flow_categories.keys()]
    # ax.legend(handles=legend_elements, loc='lower right')
    
    plt.title(f'QL Difference - Issue Date {issue_date}')
    
    # Save figures
    plt.savefig(f'E:/USBR_Snow_Forecast/Figure/Spatial/QL/QL_map_diff_combined_{issue_date}.png',
                dpi=600, bbox_inches='tight')
    plt.savefig(f'E:/USBR_Snow_Forecast/Figure/Spatial/QL/QL_map_diff_combined_{issue_date}.eps',
                format='eps', dpi=600, bbox_inches='tight')
    plt.close()



