#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
#:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:#
#                                                                                                             #
# Machine Learning for Finance                                                ########   ########             #
#                                                                            ##         ##                    #
#                                                                            ##   ####  ##                    #
#                                                                            ##     ##  ##                    #
# Date of creation: 15/04/2025                                                ########   ########             #
#                                                                                                             #
#:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:#
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#

#:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:#
# Libraries

import os
import pandas as pd
import glob

#:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:#
# Methods for data processing 

def clean_csvs(raw_data_path: str, processed_data_path: str, mode: str):
    '''
    Concatenate and format CSV files for different modes:
    - 'loads': processes actual and forecast load data
    - 'prices': processes day-ahead price data
    - 'wind_solar': processes wind/solar forecasts (treating 'n/e' as zero)
    '''
    valid_modes = ('loads', 'prices', 'wind_solar')
    assert mode in valid_modes, f"mode must be one of {valid_modes}"

    # pick the right timestamp format
    if mode == 'prices':
        date_format = '%d/%m/%Y %H:%M:%S'
    elif mode == 'loads':
        date_format = '%d/%m/%Y %H:%M'
    else:
        date_format = '%d.%m.%Y %H:%M'

    # group your CSVs by grid name
    csv_files = glob.glob(os.path.join(raw_data_path, '*.csv'))
    grids     = {}
    for file in csv_files:
        grid_name = os.path.basename(file).split('_')[0]
        grids.setdefault(grid_name, []).append(file)

    for grid, files in grids.items():
        data_frames = []
        for file in sorted(files):
            df = pd.read_csv(file)
            # This should only concern wind_solar but you never know
            df = df.replace('n/e', 0)
            df = df.infer_objects(copy = False)

            # Parse + clean timestamp
            df['timestamp'] = (
                df['MTU (CET/CEST)']
                .str.split(' - ')
                .str[0]
                .str.replace('(CET)',  '', regex = False)
                .str.replace('(CEST)', '', regex = False)
                .str.strip()
                .pipe(pd.to_datetime, dayfirst = True, format = date_format, errors='raise')
            )
            df.drop(columns=['MTU (CET/CEST)'], inplace = True)

            # Pick & rename columns
            if mode == 'prices':
                df = df[df['Sequence'] != 'Sequence 2']
                df.rename(columns={'Day-ahead Price (EUR/MWh)': 'Day-ahead Price'}, inplace=True)
                df = df[['timestamp', 'Day-ahead Price']]

            elif mode == 'loads':
                df = df[['timestamp', 'Actual Total Load (MW)','Day-ahead Total Load Forecast (MW)']]
                df.rename(columns = {'Actual Total Load (MW)': 'Actual Load',
                                    'Day-ahead Total Load Forecast (MW)': 'Forecasted Load'}, 
                                    inplace=True)

            else:
                df.rename(columns={f'Generation - Solar [MW] Day Ahead/ BZN|{grid}': 'Solar',
                                    f'Generation - Wind Offshore [MW] Day Ahead/ BZN|{grid}': 'Wind Offshore',
                                    f'Generation - Wind Onshore [MW] Day Ahead/ BZN|{grid}': 'Wind Onshore'}, 
                                    inplace=True)
                df = df[['timestamp', 'Solar', 'Wind Offshore', 'Wind Onshore']]


            # Coerce every non-timestamp column to numeric
            for col in df.columns:
                if col != 'timestamp':
                    df[col] = pd.to_numeric(df[col], errors = 'coerce')

            # One-liner beacuse I'm cool
            df_hourly = (
                df
                .set_index('timestamp')                               # put timestamp on the index
                .resample('h').mean(numeric_only=True)                # hourly grid
                .interpolate(method='linear', limit_direction='both') # interpolate
                .fillna(0)                                            # force-fill any big gaps with 0 (you made me do this, Finland)
                .reset_index()                                        # back to a normal DF
            )

            # Should be clean now
            data_frames.append(df_hourly)

        combined = pd.concat(data_frames, ignore_index = True)
        out_file = os.path.join(processed_data_path, f"{grid}.csv")
        combined.to_csv(out_file, index=False)
        print(f"Saved processed '{mode}' for {grid}: {out_file}")
        
    return

#——————————————————————————————————————————————————————————————————————————————————————————————————————————————#

def list_missing_hours(df):
    '''List missing hours in a dataframe.'''
    # Convert the timestamp column to datetime objects using the input format.
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")
    # Create a new column representing each timestamp rounded down to the hour.
    df["hour"]      = df["timestamp"].dt.floor("h")
    # Create a complete range of hours between the minimum and maximum hour in the dataset.
    all_hours       = pd.date_range(start=df["hour"].min(), end=df["hour"].max(), freq="h")
    # Find the unique hours that are actually present in the dataframe.
    hours_present   = pd.to_datetime(df["hour"].unique())
    # Identify which hours are in the complete range but missing from our data.
    missing_hours   = [hour for hour in all_hours if hour not in hours_present]
    return missing_hours

#——————————————————————————————————————————————————————————————————————————————————————————————————————————————#

def check_directory(data_path: str):
    '''
    List missing hours for all files and ensure no non-numerical values
    in non-timestamp columns within 'data_path' directory.
    '''
    csv_files = glob.glob(os.path.join(data_path, '*.csv'))
    for file in csv_files:
        df = pd.read_csv(file)
        # Identify non-numerical entries in non-timestamp columns
        cols_to_check = [c for c in df.columns if c != 'timestamp']
        # Attempt numeric conversion and flag NaNs originating from invalid formats
        non_numeric_mask = df[cols_to_check].map(lambda x: pd.to_numeric(x, errors='coerce')).isna()
        rows_with_non_numeric = non_numeric_mask.any(axis=1)
        if rows_with_non_numeric.any():
            bad_rows = df.index[rows_with_non_numeric].tolist()
            print(f"Non-numerical values found in {file} at rows: {bad_rows}")
        # Check for missing hours using provided utility
        missing_hours = list_missing_hours(df)
        if missing_hours:
            print(f"Missing hours for {file}: {missing_hours}")
        else:
            print(f"File {file} is complete")
    return

#——————————————————————————————————————————————————————————————————————————————————————————————————————————————#

def join_files(data_dir, output_filepath, start_time = "2019-01-01", end_time = "2024-12-31"):
    '''
    Join csvs into a big one.
    '''    
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    # Initialize merged DataFrame
    merged_df = None

    # Process each file
    for file in csv_files:
        bzn = os.path.splitext(os.path.basename(file))[0]  # Extract BZN from filename
        df  = pd.read_csv(file, parse_dates=["timestamp"])
        df  = df.rename(columns={"Day-ahead Price": bzn})
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on="timestamp", how="outer")

    # Sort by timestamp
    merged_df   = merged_df.sort_values("timestamp")
    filtered_df = merged_df[(merged_df["timestamp"] >= start_time) & (merged_df["timestamp"] <= end_time)]

    filtered_df.to_csv(output_filepath, index = False)

#——————————————————————————————————————————————————————————————————————————————————————————————————————————————#

def processed_csvs_to_parquet(
    input_dir    : str,
    gas_filepath : str,          #  NEW  <- path to a csv with columns ["date","gas_price"]
    output_file  : str
):
    """
    Reads every `{BZN}_merged.csv` in `input_dir`, adds a 'zone' column,
    merges in a global 'gas_price' series (daily granularity, broadcast to hours),
    reorders columns so 'Day-ahead Price' is first, then writes ONE parquet.
    """

    # ── 1. read / expand the dail y gas series  ────────────────────────────
    gdf = pd.read_csv(gas_filepath, parse_dates=["timestamp"])            # ['date','gas_price']
    gdf =  (
        gdf.set_index("timestamp")
            .resample("1h")                                    # make it hourly -> don't worry I am not using it as hourly attribute
            .ffill()                                           # forward-fill each day's value
            .reset_index()
    )
    # ── 2. collect zone csvs  ─────────────────────────────────────────────
    dfs = []
    for fp in glob.glob(os.path.join(input_dir, "*_merged.csv")):
        zone = os.path.basename(fp).replace("_merged.csv", "")
        df = pd.read_csv(fp, header=0)
        if df.columns[0] != "timestamp":
            df.columns.values[0] = "timestamp"
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        df["zone"] = zone

        # merge the global gas price
        df = df.merge(gdf, on="timestamp", how="left", validate="many_to_one")

        price_col = "Day-ahead Price"
        other     = [c for c in df.columns
                     if c not in ("timestamp", "zone", price_col)]
        # 'gas_price' is already in `other`
        df = df[["timestamp", "zone", price_col] + other]

        dfs.append(df)

    if not dfs:
        raise RuntimeError(f"No *_merged.csv found in {input_dir}")

    # ── 3. concat & save  ────────────────────────────────────────────────
    full = (
        pd.concat(dfs, ignore_index=True)
          .sort_values(["timestamp", "zone"])
          .reset_index(drop=True)
    )
    full.to_parquet(output_file, index=False, compression="zstd")
    print(f"Parquet written → {output_file}")



