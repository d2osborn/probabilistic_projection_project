"""
Creates a function that fetches Statcast data to use for analysis
"""
import os
import pandas as pd
import time
import pybaseball.statcast as pyb

def get_statcast_data(start_date, end_date, filename="statcast_data.parquet"):
    """
    Fetches Statcast data if the file doesn't exist; otherwise, loads it from disk.
    """
    if os.path.exists(filename):
        print(f"  -> Local file {filename} found. Loading from disk...")
        return pd.read_parquet(filename)
    else:
        print(f"  -> Downloading data from {start_date} to {end_date}...")
        df = pyb(start_dt=start_date, end_dt=end_date)
        df.to_parquet(filename, index=False)
        return df

def safe_load(year, retries=5):
    """
    Safely loads the statcast years in order to prevent any API crashes from ruining the loading process
    """
    start_date = f'{year}-03-01'
    end_date = f'{year}-11-10'
    filename = f'files/statcast_{year}.parquet'
    
    for i in range(retries):
        try:
            print(f"Loading {year}, attempt {i+1}/{retries}")
            return get_statcast_data(start_date, end_date, filename)
            
        except Exception as e:
            print(f"  -> Year {year} failed (attempt {i+1}): {e}")
            time.sleep(5)
            
    print(f"Skipping {year} after repeated failures.")
    return None

os.makedirs('files', exist_ok=True)

df_list = []
failed_years = []

for yr in range(2018, 2026): 
    yearly_df = safe_load(yr)
    if yearly_df is not None:
        df_list.append(yearly_df)
        print("  -> Success!\n")
    else:
        failed_years.append(yr)
        print("\n")
    time.sleep(2) 

if df_list:
    df = pd.concat(df_list, ignore_index=True)
    print(f"Successfully combined data! Total rows: {len(df)}")
    print(df.head())
else:
    print("No data was successfully loaded.")

if failed_years:
    print(f"failures: {failed_years}") 

