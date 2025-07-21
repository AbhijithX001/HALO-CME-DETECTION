import os
import time
from pathlib import Path
import pandas as pd
import numpy as np
import cdflib
import netCDF4
import warnings

 
 
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "raw_data"
OUTPUT_DIR = BASE_DIR / "processed_data"
LOG_DIR = BASE_DIR / "logs"

 
OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

 
 

 
 
SMOOTHING_WINDOW = '300s'

 
 
 
FLUX_STD_MULTIPLIER = 2.5   

 
 
MIN_EVENT_DURATION_MINUTES = 5

 
CHUNK_SIZE = 10000


def find_variable(info, common_names):
    """
    Dynamically finds the most likely variable from a list of common names.
    Works for both CDF info objects and netCDF variable lists.
    """
     
    info_vars_lower = [str(v).lower() for v in info]
    for name in common_names:
        if name.lower() in info_vars_lower:
             
            for original_name in info:
                if str(original_name).lower() == name.lower():
                    return original_name
    return None

def load_cdf_data(path):
    """
    Loads data from a single CDF file into a pandas DataFrame.
    This version correctly handles 1D, 2D, and 3D data by processing in chunks
    to avoid memory errors and ensure all output is 1D.
    """
    try:
        with cdflib.CDF(str(path)) as cdf:
            cdf_info = cdf.cdf_info()
            all_vars = cdf_info.zVariables + cdf_info.rVariables
            
             
            time_var_names = ['epoch_for_cdf', 'epoch_for_cdf_mod', 'obs_time', 'Epoch', 'time']
            flux_var_names = ['proton_density', 'THA-1_spec', 'THA-2_spec', 'integrated_flux_mod', 'Flux']

            time_var_name = find_variable(all_vars, time_var_names)
            flux_var_name = find_variable(all_vars, flux_var_names)
            
            if not time_var_name:
                raise ValueError("Could not find a recognizable time variable.")
            if not flux_var_name:
                 raise ValueError("Could not find a recognizable flux/density/spec variable.")

            var_info = cdf.varinq(flux_var_name)
             
            num_records = var_info.Last_Rec + 1

            all_epoch_data = []
            all_flux_data = []

             
            for start_rec in range(0, num_records, CHUNK_SIZE):
                end_rec = min(start_rec + CHUNK_SIZE, num_records)
                if start_rec >= end_rec: continue
                
                epoch_chunk = cdf.varget(time_var_name, startrec=start_rec, endrec=end_rec - 1)
                flux_chunk = cdf.varget(flux_var_name, startrec=start_rec, endrec=end_rec - 1)

                if flux_chunk is None or epoch_chunk is None: continue

                 
                if flux_chunk.ndim > 1:
                    sum_axes = tuple(range(1, flux_chunk.ndim))
                    flux_1d = np.sum(flux_chunk, axis=sum_axes)
                else:
                    flux_1d = flux_chunk
                
                if epoch_chunk.ndim > 1:
                    slice_indices = [slice(None)] * (epoch_chunk.ndim - 1) + [0]
                    epoch_1d = epoch_chunk[tuple(slice_indices)]
                else:
                    epoch_1d = epoch_chunk

                if len(epoch_1d) != len(flux_1d):
                     warnings.warn(f"Skipping chunk in {path.name} due to mismatched data lengths after processing.")
                     continue

                all_epoch_data.append(epoch_1d)
                all_flux_data.append(flux_1d)
            
            if not all_flux_data:
                return None

            final_epoch = np.concatenate(all_epoch_data)
            final_flux = np.concatenate(all_flux_data)

            times = pd.to_datetime(final_epoch)
            df = pd.DataFrame(index=times)
            df['Flux'] = final_flux
            
            df.replace(-1.000000e+31, np.nan, inplace=True)
            return df

    except Exception as e:
        warnings.warn(f"Could not process file {path.name}. Reason: {e}")
        return None


def load_nc_data(path):
    """
    Loads magnetic field data from a NetCDF file. It calculates the total
    magnetic field strength from its components (Bx, By, Bz).
    """
    try:
        with netCDF4.Dataset(path) as nc:
            nc_vars = nc.variables.keys()

            time_var_name = find_variable(nc_vars, ['time'])
            
             
            bx_var_name = find_variable(nc_vars, ['Bx_gse'])
            by_var_name = find_variable(nc_vars, ['By_gse'])
            bz_var_name = find_variable(nc_vars, ['Bz_gse'])

            if not all([time_var_name, bx_var_name, by_var_name, bz_var_name]):
                raise ValueError("Could not find all required variables (time, Bx_gse, By_gse, Bz_gse).")

            time_var = nc.variables[time_var_name]
             
            if 'unix' in time_var.units.lower():
                 times_pd = pd.to_datetime(time_var[:], unit='s')
            else:
                 times = netCDF4.num2date(time_var[:], units=time_var.units)
                 times_pd = pd.to_datetime(times)

             
            bx = nc.variables[bx_var_name][:]
            by = nc.variables[by_var_name][:]
            bz = nc.variables[bz_var_name][:]

             
            flux_data = np.sqrt(bx**2 + by**2 + bz**2)

            df = pd.DataFrame(index=times_pd)
            df['Flux'] = flux_data
            
             
            if hasattr(nc.variables[bx_var_name], '_FillValue'):
                fill_val = nc.variables[bx_var_name]._FillValue
                df.replace(fill_val, np.nan, inplace=True)

            return df

    except Exception as e:
        warnings.warn(f"Could not process file {path.name}. Reason: {e}")
        return None


def load_all_data(directory):
    """
    Walks through a directory and loads all .cdf and .nc files, showing progress.
    """
    all_dfs = []
    success_count = 0
    failure_count = 0
    
    print(f"Searching for .cdf and .nc files in: {directory}")
    files_to_process = list(Path(directory).rglob('*.cdf')) + list(Path(directory).rglob('*.nc'))
    total_files = len(files_to_process)
    
    if not files_to_process:
        print("No .cdf or .nc files found in the directory.")
        return pd.DataFrame()

    print(f"Found {total_files} files to process.")

    for i, path in enumerate(files_to_process):
         
        print(f"--> Processing file {i+1}/{total_files}: {path.name}")
        df = None
         
        if path.suffix.lower() == '.nc':
            df = load_nc_data(path)
        else:
            df = load_cdf_data(path)
        
        if df is not None and not df.empty:
            all_dfs.append(df)
            success_count += 1
        else:
            failure_count += 1

    print("\n--- Data Loading Summary ---")
    print(f"Total files attempted: {total_files}")
    print(f"✅ Successfully loaded: {success_count}")
    print(f"❌ Failed to load: {failure_count}")
    print("--------------------------\n")

    if not all_dfs:
        return pd.DataFrame()

    try:
        print("Combining all loaded data...")
        combined_df = pd.concat(all_dfs)
        combined_df.sort_index(inplace=True)
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        return combined_df
    except Exception as e:
        print(f"--- Error during final concatenation ---")
        print(f"Error: {e}")
        print("This may happen if some files have fundamentally different data structures that could not be reconciled.")
        print("The script will now exit.")
        return pd.DataFrame()


def detect_cme_events(df):
    """
    Detects CME events in the combined DataFrame based on flux anomalies.
    """
    if df.empty or 'Flux' not in df.columns:
        print("DataFrame is empty or 'Flux' column is missing. Skipping event detection.")
        return df, []

     
    df_filled = df.copy()
    
     
    df_filled['Flux'] = df_filled['Flux'].ffill()
    df_filled['Flux'] = df_filled['Flux'].bfill()

    df_smooth = df_filled.rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
    df_smooth.rename(columns={'Flux': 'Flux_smooth'}, inplace=True)
    
    flux_mean = df_smooth["Flux_smooth"].mean()
    flux_std = df_smooth["Flux_smooth"].std()

    if pd.isna(flux_std) or flux_std == 0:
        print("Standard deviation of flux is zero or NaN. Cannot determine event threshold.")
        return df, []
        
    flux_threshold = flux_mean + FLUX_STD_MULTIPLIER * flux_std
    print(f"Flux event threshold calculated: {flux_threshold:.2f}")

    df_processed = df.join(df_smooth)
    df_processed['is_event'] = df_processed['Flux_smooth'] > flux_threshold

    detected_events = []
    in_event = False
    event_start_time = None
    
    print("Detecting events...")
     
    for timestamp, row in df_processed.iterrows():
        if row['is_event'] and not in_event:
            in_event = True
            event_start_time = timestamp
        elif not row['is_event'] and in_event:
            in_event = False
            event_end_time = timestamp
            duration = (event_end_time - event_start_time).total_seconds() / 60
            if duration >= MIN_EVENT_DURATION_MINUTES:
                detected_events.append({
                    'start_time': event_start_time,
                    'end_time': event_end_time,
                    'duration_minutes': duration,
                    'peak_flux': df_processed.loc[event_start_time:event_end_time, 'Flux'].max()
                })

    return df_processed, detected_events


def main():
    """
    Main function to run the data processing and event detection pipeline.
    """
    start_time = time.time()
    
    df = load_all_data(DATA_DIR)
    
    if df.empty:
        print("❌ Error: No valid data could be loaded. Please check the warnings above for details on failed files.")
        print("Halting execution.")
        return

    load_time = time.time()
    print(f"✅ Combined {len(df)} data points in {load_time - start_time:.2f} seconds.")

    df_processed, detected_events = detect_cme_events(df)
    
    process_time = time.time()
    print(f"✅ Processing finished in {process_time - load_time:.2f} seconds.")

    if detected_events:
        print(f"\n--- 🚨 {len(detected_events)} Potential CME Events Detected ---")
        events_df = pd.DataFrame(detected_events)
        print(events_df)
        events_csv_path = OUTPUT_DIR / "cme_events.csv"
        events_df.to_csv(events_csv_path, index=False)
        print(f"\nEvent list saved to {events_csv_path}")
    else:
        print("\n--- No significant CME events detected. ---")

    processed_data_path = OUTPUT_DIR / "full_processed_data.csv"
    df_processed.to_csv(processed_data_path)
    print(f"Full processed data saved to {processed_data_path}")


if __name__ == "__main__":
    main()