# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 14:14:43 2025

@author: yue0004
"""

# This is the Python script to plot the boxplot for paper revision
# Boxplot with non-overlapping boxes and climatology is included as a baseline
# Last Modified on January 24, 2025
###############################################################################

#%% import packages
# import pdb
import pandas as pd
# import warnings
# import a customized module
import sys
sys.path.append('E:/USBR_Snow_Forecast/Fcst_Model')
import Utils
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# from scipy.stats import norm
# from scipy.stats import gumbel_r
# from mpl_toolkits.axes_grid1 import make_axes_locatable


#%% set global parameters
src_dir = "E:/USBR_Snow_Forecast/Fcst_Model"

#%% import evaluation metrics for GBR-SWE, GBR-NoSWE and Climo
# Extract evaluation metrics from GBR-SWE
# QL with SWE
mean_QL_df_all_withswe = pd.read_csv(src_dir + '/results_single_basin_1982_2021/mean_QL_withSWE_all.csv')
# QL, all quantiles, with SWE
mean_AllQL_df_all_withswe = pd.read_csv(src_dir + '/results_single_basin_1982_2021/mean_QL_AllQ_withSWE_all.csv')
# NSE with SWE
mean_NSE_df_all_withswe = pd.read_csv(src_dir + '/results_single_basin_1982_2021/mean_NSE_withSWE_all.csv')
# # RMSE with SWE
# mean_RMSE_df_all_withswe = pd.read_csv('E:/USBR_Snow_Forecast/Results/mean_RMSE_withSWE_all.csv')
# NRMSE with SWE
mean_NRMSE_df_all_withswe = pd.read_csv(src_dir + '/results_single_basin_1982_2021/mean_NRMSE_withSWE_all.csv')
# interval coverage
mean_IC_df_all_withswe = pd.read_csv(src_dir + '/results_single_basin_1982_2021/mean_IC_withSWE_all.csv')
# covert value to percent
mean_IC_df_all_withswe['IC'] = mean_IC_df_all_withswe['IC'] * 100
# Extract evaluation metrics from GBR-NoSWE
# QL without SWE
mean_QL_df_all_woswe = pd.read_csv(src_dir + '/results_single_basin_1982_2021/mean_QL_withoutSWE_all.csv')
# QL, all quantiles, without SWE
mean_AllQL_df_all_woswe = pd.read_csv(src_dir + '/results_single_basin_1982_2021/mean_QL_AllQ_withoutSWE_all.csv')
# NSE without SWE
mean_NSE_df_all_woswe = pd.read_csv(src_dir + '/results_single_basin_1982_2021/mean_NSE_withoutSWE_all.csv')
# # RMSE without SWE
# mean_RMSE_df_all_woswe = pd.read_csv('E:/USBR_Snow_Forecast/Results/mean_RMSE_withoutSWE_all.csv')
# NRMSE without SWE
mean_NRMSE_df_all_woswe = pd.read_csv(src_dir + '/results_single_basin_1982_2021/mean_NRMSE_withoutSWE_all.csv')
# interval coverage
mean_IC_df_all_woswe = pd.read_csv(src_dir + '/results_single_basin_1982_2021/mean_IC_withoutSWE_all.csv')
# covert value to percent
mean_IC_df_all_woswe['IC'] = mean_IC_df_all_woswe['IC'] * 100
# Extract evaluation metrics from climatology
# QL without SWE
mean_QL_df_all_climo = pd.read_csv(src_dir + '/results_single_basin_1982_2021/mean_QL_climo_all.csv')
# QL, all quantiles, without SWE
mean_AllQL_df_all_climo = pd.read_csv(src_dir + '/results_single_basin_1982_2021/mean_QL_AllQ_climo_all.csv')
# NSE without SWE
mean_NSE_df_all_climo = pd.read_csv(src_dir + '/results_single_basin_1982_2021/mean_NSE_climo_all.csv')
# NRMSE without SWE
mean_NRMSE_df_all_climo = pd.read_csv(src_dir + '/results_single_basin_1982_2021/mean_NRMSE_climo_all.csv')
# interval coverage
mean_IC_df_all_climo = pd.read_csv(src_dir + '/results_single_basin_1982_2021/mean_IC_climo_all.csv')
# covert value to percent
mean_IC_df_all_climo['IC'] = mean_IC_df_all_climo['IC'] * 100

#%% set figure settings
sns.set(style="whitegrid", font='Arial', rc={'font.size': 16})
sns.set(rc={'axes.edgecolor': 'black'})
plt.rcParams['axes.facecolor'] = 'white'
sns.set(style="whitegrid", font='Arial')
size = 14
sns.set(rc={'axes.edgecolor': 'black'})
plt.rcParams['axes.facecolor'] = 'white'

#%% organize the data for plot
# calculate the volume for observation
# mean_per_issue_date_target = target_df_all.groupby(['issue_date'])['volume'].mean().reset_index()
# calculate the mean of QL by forecast site for each issue date
mean_QL_per_issue_date_withswe = mean_QL_df_all_withswe.groupby(['issue_date'])['QL'].mean().reset_index()
mean_QL_per_issue_date_woswe = mean_QL_df_all_woswe.groupby(['issue_date'])['QL'].mean().reset_index()
mean_QL_per_issue_date_climo = mean_QL_df_all_climo.groupby(['issue_date'])['QL'].mean().reset_index()
# mean_per_issue_date_withswe = mean_all_withswe.groupby(['issue_date'])[['value_10', 'value_30',
#                                                                         'value_50', 'value_70',
#                                                                         'value_90']].mean().reset_index()
# mean_per_issue_date_woswe = mean_all_woswe.groupby(['issue_date'])[['value_10', 'value_30',
#                                                                     'value_50', 'value_70',
#                                                                     'value_90']].mean().reset_index()

# calculate the mean of NSS by forecast site for each issue date
mean_NSE_per_issue_date_withswe = mean_NSE_df_all_withswe.groupby('issue_date')['NSE'].mean().reset_index()
mean_NSE_per_issue_date_woswe = mean_NSE_df_all_woswe.groupby('issue_date')['NSE'].mean().reset_index()
mean_NSE_per_issue_date_climo = mean_NSE_df_all_climo.groupby('issue_date')['NSE'].mean().reset_index()

# calculate the mean of RMSE by forecast site for each issue date
mean_NRMSE_per_issue_date_withswe = mean_NRMSE_df_all_withswe.groupby('issue_date')['NRMSE'].mean().reset_index()
mean_NRMSE_per_issue_date_woswe = mean_NRMSE_df_all_woswe.groupby('issue_date')['NRMSE'].mean().reset_index()
mean_NRMSE_per_issue_date_climo = mean_NRMSE_df_all_climo.groupby('issue_date')['NRMSE'].mean().reset_index()

# calculate the mean of interval coverage by forecast site for each issue date
mean_IC_per_issue_date_withswe = mean_IC_df_all_withswe.groupby('issue_date')['IC'].mean().reset_index()
mean_IC_per_issue_date_woswe = mean_IC_df_all_woswe.groupby('issue_date')['IC'].mean().reset_index()
mean_IC_per_issue_date_climo = mean_IC_df_all_climo.groupby('issue_date')['IC'].mean().reset_index()


#%% plot the boxplot for mean QL
#####################################################################
# concate all datasets together
# Example: Add categorical "Source" column to the datasets
mean_QL_df_all_withswe['Source'] = 'With SWE'
mean_QL_df_all_woswe['Source'] = 'Without SWE'
mean_QL_df_all_climo['Source'] = 'Climo'
# Combine the datasets into one DataFrame for side-by-side plotting
QL_df_forplot = pd.concat([mean_QL_df_all_withswe, mean_QL_df_all_woswe, mean_QL_df_all_climo])

# # Set the color palette for the plot
# palette = {
#     'With SWE': '#619CFF',  # Light blue for With SWE
#     'Without SWE': '#F8766D',  # Light red for Without SWE
#     'Climo': '#00BA38' # Light green for Climo
# }

palette = {
    'With SWE': '#619CFF',  # Light blue
    'Without SWE': '#F8766D',  # Red
    'Climo': '#F0E442'  # Yellow
}

# Set font and axis
sns.set(style="whitegrid", rc={"axes.edgecolor": "black", "axes.facecolor": "white"})
plt.rcParams['axes.facecolor'] = 'white'

# size = 14  # Adjust the size according to your preference

# Set up the figure
plt.figure(figsize=(16, 8), tight_layout=True)

# Side-by-side boxplot
sns.boxplot(
    data=QL_df_forplot, x="issue_date", y="QL", hue="Source",  # Add 'hue' for side-by-side
    palette=palette,
    # set the property of fliers/outliers
    flierprops={"marker": "x", "markersize": 8, 'markeredgewidth': 2},
    boxprops={"linewidth": 1.5, "alpha": 1.0},
    whiskerprops={"linewidth": 1.5, "alpha": 1.0},
    medianprops={"linewidth": 1.5, "alpha": 1.0}
)


# Add lineplots for With SWE and Without SWE
sns.lineplot(
    x='issue_date', y='QL', data=mean_QL_per_issue_date_withswe,
    marker='o', color='#619CFF', linewidth=2, label='With SWE')

# plt.plot(
#     mean_QL_per_issue_date_withswe['issue_date'],
#     mean_QL_per_issue_date_withswe['QL'],
#     marker='o', color='#619CFF', linewidth=2, label='With SWE'
# )

sns.lineplot(
    x='issue_date', y='QL', data=mean_QL_per_issue_date_woswe,
    marker='o', color='#F8766D', linewidth=2, label='Without SWE')

sns.lineplot(
    x='issue_date', y='QL', data=mean_QL_per_issue_date_climo,
    marker='o',color='#DAD158', linewidth=2, label='Climo')

# Set labels and ticks
plt.xlabel('Issue Date', fontsize=28)
plt.ylabel('QL (KAF)', fontsize=28, fontstyle='italic')
plt.xticks(rotation=45, ha='right', fontsize=24)
plt.yticks(fontsize=24)
plt.ylim(0, 260)
plt.grid(color='darkgray', linestyle='--', linewidth=0.5)

# Add tick marks
plt.tick_params(bottom=True, left=True, axis='both', which='major', length=8, width=2, color='black')
# Remove the legend
plt.legend([], [], frameon=False)
# # Create legend entries manually using patches with alpha
# withswe_patch = mpatches.Patch(color=palette['With SWE'], label='With SWE', edgecolor='black')
# woswe_patch = mpatches.Patch(color=palette['Without SWE'], label='Without SWE', edgecolor='black')
# climo_patch = mpatches.Patch(color='#DAD158', label='Climatology', edgecolor='black')
# # Create legend entries for lines
# withswe_line = mlines.Line2D([], [], color='#619CFF', marker='o', linewidth=2, label='With SWE')
# woswe_line = mlines.Line2D([], [], color='#F8766D', marker='o', linewidth=2, label='Without SWE')
# climo_line = mlines.Line2D([], [], color='#DAD158', marker='o', linewidth=2, label='Climo')
# # Adjust the order of legend entries
# legend_handles = [withswe_patch, woswe_patch, climo_patch, withswe_line, woswe_line, climo_line]

# # Create legend
# legend = plt.legend(handles=legend_handles, loc='upper right', fontsize=24)

# # Set alpha for legend
# for patch in legend.get_patches():
#     patch.set_alpha(0.5)

plt.savefig('E:/USBR_Snow_Forecast/Figure/boxplot_combined_QL_v2_nolegend.png', dpi=600)
plt.savefig('E:/USBR_Snow_Forecast/Figure/boxplot_combined_QL_v2_nolegend.pdf', format='pdf', dpi=600)
plt.savefig('E:/USBR_Snow_Forecast/Figure/boxplot_combined_QL_v2_nolegend.eps',format='eps', dpi=600)
# Show the plot
plt.show()



#%% plot the boxplot for mean NSE 
#####################################################################
# Add categorical "Source" column to the datasets
mean_NSE_df_all_withswe['Source'] = 'With SWE'
mean_NSE_df_all_woswe['Source'] = 'Without SWE'
mean_NSE_df_all_climo['Source'] = 'Climo'
# Combine the datasets into one DataFrame for side-by-side plotting
NSE_df_forplot = pd.concat([mean_NSE_df_all_withswe, mean_NSE_df_all_woswe, mean_NSE_df_all_climo])
# # Set the color palette for the plot
palette = {
    'With SWE': '#619CFF',  # Light blue
    'Without SWE': '#F8766D',  # Red
    'Climo': '#F0E442'  # Yellow
}

# set font and axis
sns.set(style="whitegrid", font='Arial')
sns.set(rc={'axes.edgecolor': 'black'})
plt.rcParams['axes.facecolor'] = 'white'

# Set up the figure
plt.figure(figsize=(16, 8),tight_layout=True)

# Side-by-side boxplot
sns.boxplot(
    data=NSE_df_forplot, x="issue_date", y="NSE", hue="Source",  # Add 'hue' for side-by-side
    palette=palette,
    # set the property of fliers/outliers
    flierprops={"marker": "x", "markersize": 8, 'markeredgewidth': 2},
    boxprops={"linewidth": 1.5, "alpha": 1.0},
    whiskerprops={"linewidth": 1.5, "alpha": 1.0},
    medianprops={"linewidth": 1.5, "alpha": 1.0}
)


# Add lineplots for With SWE and Without SWE
sns.lineplot(
    x='issue_date', y='NSE', data=mean_NSE_per_issue_date_withswe,
    marker='o', color='#619CFF', linewidth=2, label='With SWE')

sns.lineplot(
    x='issue_date', y='NSE', data=mean_NSE_per_issue_date_woswe,
    marker='o', color='#F8766D', linewidth=2, label='Without SWE')

sns.lineplot(
    x='issue_date', y='NSE', data=mean_NSE_per_issue_date_climo,
    marker='o',color='#DAD158', linewidth=2, label='Climo')

# Set labels and ticks
plt.xlabel('Issue Date', fontsize=28)
plt.ylabel('NSE', fontsize=28, fontstyle='italic')
plt.xticks(rotation=45, ha='right', fontsize=24)
plt.yticks(fontsize=24)
plt.ylim(-0.35, 1)
plt.grid(color='darkgray', linestyle='--', linewidth=0.5)
# add tick marks
plt.tick_params(bottom=True,left=True,axis='both', which='major', length=8, width=2, color='black')
# Remove the legend
plt.legend([], [], frameon=False)
# # Create legend entries manually using patches with alpha
# withswe_patch = mpatches.Patch(color=palette['With SWE'], label='With SWE', edgecolor='black')
# woswe_patch = mpatches.Patch(color=palette['Without SWE'], label='Without SWE', edgecolor='black')
# climo_patch = mpatches.Patch(color='#DAD158', label='Climatology', edgecolor='black')
# # Create legend entries for lines
# withswe_line = mlines.Line2D([], [], color='#619CFF', marker='o', linewidth=2, label='With SWE')
# woswe_line = mlines.Line2D([], [], color='#F8766D', marker='o', linewidth=2, label='Without SWE')
# climo_line = mlines.Line2D([], [], color='#DAD158', marker='o', linewidth=2, label='Climo')
# # Adjust the order of legend entries
# legend_handles = [withswe_patch, woswe_patch, climo_patch, withswe_line, woswe_line, climo_line]

# # Create legend
# legend = plt.legend(handles=legend_handles, loc='upper right', fontsize=24)

plt.savefig('E:/USBR_Snow_Forecast/Figure/boxplot_combined_NSE_v2_nolegend.png', dpi=600)
plt.savefig('E:/USBR_Snow_Forecast/Figure/boxplot_combined_NSE_v2_nolegend.pdf', format='pdf', dpi=600)
plt.savefig('E:/USBR_Snow_Forecast/Figure/boxplot_combined_NSE_v2_nolegend.eps',format='eps', dpi=600)
# Show the plot
plt.show()


#%% plot the boxplot for mean NRMSE
#####################################################################
# Add categorical "Source" column to the datasets
mean_NRMSE_df_all_withswe['Source'] = 'With SWE'
mean_NRMSE_df_all_woswe['Source'] = 'Without SWE'
mean_NRMSE_df_all_climo['Source'] = 'Climo'
# Combine the datasets into one DataFrame for side-by-side plotting
NRMSE_df_forplot = pd.concat([mean_NRMSE_df_all_withswe, mean_NRMSE_df_all_woswe, mean_NRMSE_df_all_climo])
# # Set the color palette for the plot
palette = {
    'With SWE': '#619CFF',  # Light blue in R ggplot
    'Without SWE': '#F8766D',  # Red in R ggplot
    'Climo': '#F0E442'  # Yellow from Baker's paper
}

# set font and axis
sns.set(style="whitegrid", font='Arial')
sns.set(rc={'axes.edgecolor': 'black'})
plt.rcParams['axes.facecolor'] = 'white'

# Set up the figure
plt.figure(figsize=(16, 8),tight_layout=True)

# Side-by-side boxplot
sns.boxplot(
    data=NRMSE_df_forplot, x="issue_date", y="NRMSE", hue="Source",  # Add 'hue' for side-by-side
    palette=palette,
    # set the property of fliers/outliers
    flierprops={"marker": "x", "markersize": 8, 'markeredgewidth': 2},
    boxprops={"linewidth": 1.5, "alpha": 1.0},
    whiskerprops={"linewidth": 1.5, "alpha": 1.0},
    medianprops={"linewidth": 1.5, "alpha": 1.0}
)

# Add lineplots for With SWE 
sns.lineplot(
    x='issue_date', y='NRMSE', data=mean_NRMSE_per_issue_date_withswe,
    marker='o', color='#619CFF', linewidth=2, label='With SWE')
# Add lineplots for Without SWE 
sns.lineplot(
    x='issue_date', y='NRMSE', data=mean_NRMSE_per_issue_date_woswe,
    marker='o', color='#F8766D', linewidth=2, label='Without SWE')
# Add lineplots for Climatology
sns.lineplot(
    x='issue_date', y='NRMSE', data=mean_NRMSE_per_issue_date_climo,
    marker='o',color='#DAD158', linewidth=2, label='Climo')
# Set labels and ticks
plt.xlabel('Issue Date', fontsize=28)
plt.ylabel('NRMSE (%)', fontsize=28, fontstyle='italic')
plt.xticks(rotation=45, ha='right', fontsize=24)
plt.yticks(fontsize=24)
plt.ylim(0, 100)
plt.grid(color='darkgray', linestyle='--', linewidth=0.5)
# add tick marks
plt.tick_params(bottom=True,left=True,axis='both', which='major', length=8, width=2, color='black')
# Remove the legend
plt.legend([], [], frameon=False)
# # Create legend entries manually using patches with alpha
# withswe_patch = mpatches.Patch(color=palette['With SWE'], label='With SWE', edgecolor='black')
# woswe_patch = mpatches.Patch(color=palette['Without SWE'], label='Without SWE', edgecolor='black')
# climo_patch = mpatches.Patch(color='#DAD158', label='Climatology', edgecolor='black')
# # Create legend entries for lines
# withswe_line = mlines.Line2D([], [], color='#619CFF', marker='o', linewidth=2, label='With SWE')
# woswe_line = mlines.Line2D([], [], color='#F8766D', marker='o', linewidth=2, label='Without SWE')
# climo_line = mlines.Line2D([], [], color='#DAD158', marker='o', linewidth=2, label='Climo')
# # Adjust the order of legend entries
# legend_handles = [withswe_patch, woswe_patch, climo_patch, withswe_line, woswe_line, climo_line]

# # Create legend
# legend = plt.legend(handles=legend_handles, loc='upper right', fontsize=24)

plt.savefig('E:/USBR_Snow_Forecast/Figure/boxplot_combined_NRMSE_v2_nolegend.png', dpi=600)
plt.savefig('E:/USBR_Snow_Forecast/Figure/boxplot_combined_NRMSE_v2_nolegend.pdf', format='pdf', dpi=600)
plt.savefig('E:/USBR_Snow_Forecast/Figure/boxplot_combined_NRMSE_v2_nolegend.eps',format='eps', dpi=600)
# Show the plot
plt.show()

#%% plot the boxplot for mean interval coverage
#####################################################################
# Example: Add categorical "Source" column to the datasets
mean_IC_df_all_withswe['Source'] = 'With SWE'
mean_IC_df_all_woswe['Source'] = 'Without SWE'
mean_IC_df_all_climo['Source'] = 'Climo'
# Combine the datasets into one DataFrame for side-by-side plotting
IC_df_forplot = pd.concat([mean_IC_df_all_withswe, mean_IC_df_all_woswe, mean_IC_df_all_climo])
# # Set the color palette for the plot
palette = {
    'With SWE': '#619CFF',  # Light blue in R ggplot
    'Without SWE': '#F8766D',  # Red in R ggplot
    'Climo': '#F0E442'  # Yellow from Baker's paper
}

# set font and axis
sns.set(style="whitegrid", font='Arial')
sns.set(rc={'axes.edgecolor': 'black'})
plt.rcParams['axes.facecolor'] = 'white'

# Set up the figure
plt.figure(figsize=(16, 8),tight_layout=True)

# Side-by-side boxplot
sns.boxplot(
    data=IC_df_forplot, x="issue_date", y="IC", hue="Source",  # Add 'hue' for side-by-side
    palette=palette,
    # set the property of fliers/outliers
    flierprops={"marker": "x", "markersize": 8, 'markeredgewidth': 2},
    boxprops={"linewidth": 1.5, "alpha": 1.0},
    whiskerprops={"linewidth": 1.5, "alpha": 1.0},
    medianprops={"linewidth": 1.5, "alpha": 1.0}
)


# Add lineplots for With SWE 
sns.lineplot(
    x='issue_date', y='IC', data=mean_IC_per_issue_date_withswe,
    marker='o', color='#619CFF', linewidth=2, label='With SWE')
# Add lineplots for Without SWE 
sns.lineplot(
    x='issue_date', y='IC', data=mean_IC_per_issue_date_woswe,
    marker='o', color='#F8766D', linewidth=2, label='Without SWE')
# Add lineplots for Climatology
sns.lineplot(
    x='issue_date', y='IC', data=mean_IC_per_issue_date_climo,
    marker='o',color='#DAD158', linewidth=2, label='Climo')
# Set labels and ticks
plt.xlabel('Issue Date', fontsize=28)
plt.ylabel('IC (%)', fontsize=28, fontstyle='italic')
plt.xticks(rotation=45, ha='right', fontsize=24)
plt.yticks(fontsize=24)
plt.ylim(20, 100)
plt.grid(color='darkgray', linestyle='--', linewidth=0.5)
# add tick marks
plt.tick_params(bottom=True,left=True,axis='both', which='major', length=8, width=2, color='black')
# Remove the legend
plt.legend([], [], frameon=False)
# # Create legend entries manually using patches with alpha
# withswe_patch = mpatches.Patch(color=palette['With SWE'], label='With SWE', edgecolor='black')
# woswe_patch = mpatches.Patch(color=palette['Without SWE'], label='Without SWE', edgecolor='black')
# climo_patch = mpatches.Patch(color='#DAD158', label='Climatology', edgecolor='black')
# # Create legend entries for lines
# withswe_line = mlines.Line2D([], [], color='#619CFF', marker='o', linewidth=2, label='With SWE')
# woswe_line = mlines.Line2D([], [], color='#F8766D', marker='o', linewidth=2, label='Without SWE')
# climo_line = mlines.Line2D([], [], color='#DAD158', marker='o', linewidth=2, label='Climo')
# # Adjust the order of legend entries
# legend_handles = [withswe_patch, woswe_patch, climo_patch, withswe_line, woswe_line, climo_line]

# # Create legend
# legend = plt.legend(handles=legend_handles, loc='upper right', fontsize=24)

plt.savefig('E:/USBR_Snow_Forecast/Figure/boxplot_combined_IC_v2_nolegend.png', dpi=600)
plt.savefig('E:/USBR_Snow_Forecast/Figure/boxplot_combined_IC_v2_nolegend.pdf', format='pdf', dpi=600)
plt.savefig('E:/USBR_Snow_Forecast/Figure/boxplot_combined_IC_v2_nolegend.eps',format='eps', dpi=600)
# Show the plot
plt.show()


