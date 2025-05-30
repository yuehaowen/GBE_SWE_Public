import pandas as pd
import glob
import os
import Utils


def combine_fold_results(site_id, issue_date, swe_ind):
    # Define the directory based on the SWE indicator
    directory = f'results_single_basin_1982_2021/{swe_ind}'

    # Define the pattern to match the files for the specific site and issue date
    pattern = os.path.join(directory, f'{site_id}_{issue_date}_f*.csv')

    # Find all files matching the pattern
    file_list = glob.glob(pattern)

    if not file_list:
        print(f'No files found for site {site_id} and issue date {issue_date} in {directory}')
        return

    # Initialize an empty list to hold the dataframes
    df_list = []

    # Load each file and append to the list
    for file in file_list:
        df = pd.read_csv(file)
        df_list.append(df)

    # Concatenate all dataframes in the list
    combined_df = pd.concat(df_list, ignore_index=True)

    # Define the output directory and create it if it doesn't exist
    output_directory = f'results_single_basin_1982_2021/4fold_combined_{swe_ind}'
    os.makedirs(output_directory, exist_ok=True)

    # Save the combined dataframe to a new CSV file
    combined_filename = os.path.join(output_directory, f'{site_id}_{issue_date}.csv')
    combined_df.to_csv(combined_filename, index=False)
    print(f'Combined file saved as {combined_filename}')


monthly_NSF_tr_path = 'data/train_monthly_naturalized_flow_1982_2022.csv'
metadata_path = 'data/metadata.csv'
metadata = pd.read_csv(metadata_path)
# monthly_NSF_test_path = 'data/test_monthly_naturalized_flow.csv'
site_id_in_monthlyNSF = pd.read_csv(monthly_NSF_tr_path)['site_id'].unique()
site_id_short_in_monthlyNSF = [Utils.get_site_short_id(x, metadata_path) for x in site_id_in_monthlyNSF]
site_id_short_in_monthlyNSF.remove('SRA')

forecast_date = []
days = ['01', '08', '15', '22']
# days = ['01']
months = ['01', '02', '03', '04', '05', '06', '07']
for month in months:
    for day in days:
        forecast_date.append(month + '-' + day)

for site_id_short in site_id_short_in_monthlyNSF:
    for date in forecast_date:
        combine_fold_results(site_id_short, date, 'with_SWE')
        combine_fold_results(site_id_short, date, 'without_SWE')
