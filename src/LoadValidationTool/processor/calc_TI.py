import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import seaborn as sns
from pathlib import Path
from typing import Union, List
import re

from data_readers.read_LiDAR_data import load_lidar_data, load_lidar_data_10min, lidar_finder, load_and_concat_lidar
from data_readers.utils import color_text, NA_cols
from data_readers.read_MetMast_data import read_met, met_finder 
from data_readers.read_mat_data import read_matfile, mat_finder
from processor.bin_wind import bin_wind
from processor.bin_wdir import bin_wdir


def calc_ti(
    lidar_files=None,
    met_files=None,
    Matlab_mat_files=None,
    hub_height=120.0
):

    
    # wind_cols_avg = [c for c in lidar_avg.columns if "Horizontal Wind Speed" in c]
    if met_files is not None:

        metmast_data_all = pd.DataFrame()
        for i in range(len(met_files)):
            print('reading metmast file no : ',[i+1],met_files[i])
            metmast_data = read_met(met_files[i])
            metmast_data.index = metmast_data.index.round('s') # round to seconds

            metmast_data_all = pd.concat([metmast_data_all, metmast_data], ignore_index=False)
            # print(metmast_data.head())
        
        # mast TI
        metmast_data_all['ti'] = metmast_data_all['wind_speed_std'] / metmast_data_all['wind_speed']
        metmast_data_all.loc[metmast_data_all['wind_speed'] <= 0, 'ti'] = np.nan
        metmast_data_all = metmast_data_all.dropna(how="any")

        #==== Include hub wind speed for binning ---
        # Choose reference height for binning (e.g., hub height ~80m from mast)
        # hub_height = 120

        if hub_height not in metmast_data_all['height'].unique():
            hub_height = min(metmast_data_all['height'].unique(), key=lambda x: abs(x - hub_height))
        print(f"Using closest mast height {hub_height}m as reference for wind speed binning.")

        # Extract reference wind speed (from mast at ref_height)
        mast_ref = metmast_data_all[metmast_data_all['height'] == hub_height][['wind_speed']].rename(columns={'wind_speed': 'hub_ws'})

        # Merge ref_ws into mast_align (broadcast to all heights for each time)
        metmast_data_all = metmast_data_all.merge(mast_ref, left_index=True, right_index=True, how='left') # this one include hub wind speed at each height

        # clean up metmast_data_all dataframe and keep only values that Ti us less than 1
        metmast_data_all = metmast_data_all[metmast_data_all['ti'] < 1]
        
        #==== Now Compute binned TI by wind speed & wind directions ---
        
        metmast_data_all, bins_counts = bin_wind(metmast_data_all) # call the function to bin wind speed

        metmast_data_all, bins_counts_wdir = bin_wdir(metmast_data_all) # call the function to bin wind speed



        #----------------------- Met Mast TI -------------------------------
        # median of met mast TI and it is grouped by height:
        ti_profile_mast = metmast_data_all.groupby("height", as_index=False)["ti"].median()

       #----- calculate median TI per wind speed bin and height ---------------
        ti_profile_mast_binned = metmast_data_all.groupby(['height', 'ws_bin'], as_index=False, observed=True)['ti'].median()

         ### TO Do: calculate median TI per wind speed bin and wind direction bin:
         ## I need to spend more time on this one
        ti_profile_mast_binned_wdir = metmast_data_all.groupby(['height', 'ws_bin', 'wdir_bin'], as_index=False, observed=True)['ti'].median()



        return ti_profile_mast, metmast_data_all, ti_profile_mast_binned, bins_counts 


    #----------------------- LiDAR TI -------------------------------
    if lidar_files is not None:
        lidar_avg, lidar_std = load_and_concat_lidar(lidar_files)


        # Pick wind speed columns
        wind_cols_avg = [c for c in lidar_avg.columns if "Horizontal Wind Speed" in c]
        lidar_h_avg = lidar_avg[wind_cols_avg]

        wind_cols_std = [c for c in lidar_std.columns if "Horizontal Wind Speed" in c]
        lidar_h_std = lidar_std[wind_cols_std].apply(pd.to_numeric, errors="coerce")

        # Pick wind direction columns
        wdir_cols_avg = [c for c in lidar_avg.columns if "Wind Direction (deg) at" in c]
        lidar_h_avg_wdir = lidar_avg[wdir_cols_avg]



        # Keep only mean columns (avoid min/max)
        mean_cols = [c for c in lidar_h_avg.columns if "Horizontal Wind Speed (m/s) at" in c]
        lidar_h_avg_mean = lidar_h_avg[mean_cols].apply(pd.to_numeric, errors="coerce")


        mean_cols_wdir = [c for c in lidar_h_avg_wdir.columns if "Wind Direction (deg) at" in c]
        lidar_h_avg_wdir_mean = lidar_h_avg_wdir[mean_cols_wdir].apply(pd.to_numeric, errors="coerce")

        # --- Minimal mapping so std matches mean names (works for both naming styles) ---
        # If std columns are like "Std. Dev. of Horizontal Wind Speed (m/s) at ...", then 
        # map them to the corresponding mean column names before division.
        std_for_mean = {}
        for col in mean_cols:
            std_name_explicit = col.replace("Horizontal Wind Speed (m/s)", "Std. Dev. of Horizontal Wind Speed (m/s)")
            if std_name_explicit in lidar_h_std.columns:
                std_for_mean[col] = lidar_h_std[std_name_explicit]
            elif col in lidar_h_std.columns:
                # in resampled-1s case, std columns may share the same names as mean
                std_for_mean[col] = lidar_h_std[col]
            else:
                # if neither exists, skip this height
                continue

        # Build TI dataframe only for columns we found std for
        if not std_for_mean:
            raise ValueError("No matching std columns found for lidar mean wind speed columns.")

        lidar_ti = pd.DataFrame(index=lidar_h_avg_mean.index)
        lidar_wsp = pd.DataFrame(index=lidar_h_avg_mean.index)
        lidar_wdir = pd.DataFrame(index=lidar_h_avg_wdir_mean.index)

        for col, std_series in std_for_mean.items():
            vel = lidar_h_avg_mean[col]
            lidar_ti[col] = (std_series / vel)

            lidar_wsp[col] = vel # keep this wind speed so I can merge it later
            col_wdir = col.replace("Horizontal Wind Speed (m/s)", "Wind Direction (deg)")
            lidar_wdir[col_wdir] = lidar_h_avg_wdir_mean[col_wdir]


        # remove unrealistic/invalid values 
        lidar_ti = lidar_ti.mask((lidar_h_avg_mean.reindex(columns=lidar_ti.columns) <= 0) |
                                (lidar_h_avg_mean.reindex(columns=lidar_ti.columns) >= 999))

        lidar_wsp = lidar_wsp.mask((lidar_h_avg_mean.reindex(columns=lidar_wsp.columns) <= 0) |
                                (lidar_h_avg_mean.reindex(columns=lidar_wsp.columns) >= 999))
        
        lidar_wdir = lidar_wdir.mask((lidar_h_avg_wdir_mean.reindex(columns=lidar_wdir.columns) <= 0) |
                                (lidar_h_avg_wdir_mean.reindex(columns=lidar_wdir.columns) >= 999))
        # rename to TI columns
        lidar_ti.columns = [col.replace("Horizontal Wind Speed (m/s)", "TI") for col in lidar_ti.columns]


        # ensure index name for melt
        if lidar_ti.index.name != 'Time':
            lidar_ti.index.name = 'Time'
            lidar_wsp.index.name = 'Time'
            lidar_wdir.index.name = 'Time'

        # tidy lidar
        lidar_ti = lidar_ti.reset_index()  
        lidar_ti_tidy = lidar_ti.melt(id_vars="Time", var_name="col", value_name="ti") # melt and keep index as Time, And add a column for height,  and a column for TI
        lidar_ti_tidy["height"] = lidar_ti_tidy["col"].str.extract(r'at (\d+)m').astype(float) # now extract height from column and add it as a new column
        ti_profile_lidar = lidar_ti_tidy.groupby("height", as_index=False)["ti"].median() #


        lidar_wsp = lidar_wsp.reset_index()
        lidar_wsp_tidy = lidar_wsp.melt(id_vars="Time", var_name="col_wsp", value_name="wind_speed")
        lidar_wsp_tidy["height"] = lidar_wsp_tidy["col_wsp"].str.extract(r'at (\d+)m').astype(float)

        lidar_ti_tidy = pd.merge(lidar_ti_tidy, lidar_wsp_tidy, on=["Time", "height"], how="left") # merge lidar_ti_tidy and lidar_wsp_tidy



        lidar_wdir = lidar_wdir.reset_index()
        lidar_wdir_tidy = lidar_wdir.melt(id_vars="Time", var_name="col_dir", value_name="wind_direction")
        lidar_wdir_tidy["height"] = lidar_wdir_tidy["col_dir"].str.extract(r'at (\d+)m').astype(float)

        lidar_ti_tidy.drop(columns=['col', 'col_wsp'], inplace=True) # drop the columns we don't need anymore
        lidar_wdir_tidy.drop(columns=['col_dir'], inplace=True) # drop the columns we don't need anymore

        lidar_ti_tidy = pd.merge(lidar_ti_tidy, lidar_wdir_tidy, on=["Time", "height"], how="left") # merge lidar_ti_tidy and lidar_wdir_tidy
        # --- NEW: Compute binned TI by wind speed ---


        #==== Include hub wind speed for binning ---
        # Choose reference height for binning (e.g., hub height ~80m from mast)
        # hub_height = 120

        if hub_height not in lidar_ti_tidy['height'].unique():
            hub_height = min(lidar_ti_tidy['height'].unique(), key=lambda x: abs(x - hub_height))
        print(f"Using closest LiDAR height {hub_height}m as reference for wind speed binning.")


    
        # Extract reference wind speed (from data at ref_height)
        lidar_ti_tidy_backup = lidar_ti_tidy.copy() 

        
        LiDAR_ref= lidar_ti_tidy[lidar_ti_tidy['height'] == hub_height][['Time','wind_speed']].rename(columns={'wind_speed': 'hub_ws'})


        # Merge ref_ws into LiDAR (broadcast to all heights for each time), this one include hub wind speed at each height
        # lidar_ti_tidy = lidar_ti_tidy.merge(LiDAR_ref, left_index=True, right_index=True, how='left') #  it did not work using index
        lidar_ti_tidy = lidar_ti_tidy.merge(LiDAR_ref, on='Time', how='left') # Let's try with on='Time'

        # clean up lidar_ti_tidy so if lidar_ti_tidy['ti']>1 then remove the row
        lidar_ti_tidy = lidar_ti_tidy[lidar_ti_tidy['ti'] <= 1]
        #==== Now Compute binned TI by wind speed & wind directions ---
        
        lidar_ti_tidy, bins_counts = bin_wind(lidar_ti_tidy) # call the function to bin wind speed

        lidar_ti_tidy, bins_counts_wdir = bin_wdir(lidar_ti_tidy) # call the function to bin wind speed


      

        #----------------------- LiDAR Mast TI -------------------------------
        # median of LiDAR TI and it is grouped by height:
        ti_profile_Lidar = lidar_ti_tidy.groupby("height", as_index=False)["ti"].median() # median of LiDAR TI

       #----- calculate median TI per wind speed bin and height ---------------
        ti_profile_lidar_binned = lidar_ti_tidy.groupby(['height', 'ws_bin'], as_index=False, observed=True)['ti'].median() # median of LiDAR TI grouped by height and wind speed bin

         ### TO Do: calculate median TI per wind speed bin and wind direction bin:
         ## I need to spend more time on this one
        ti_profile_lidar_binned_wdir = lidar_ti_tidy.groupby(['height', 'ws_bin', 'wdir_bin'], as_index=False, observed=True)['ti'].median() # median of LiDAR TI grouped by height and wind speed bin

        return lidar_ti_tidy, ti_profile_Lidar, ti_profile_lidar_binned, ti_profile_lidar_binned_wdir, bins_counts

    if Matlab_mat_files is not None:


        # --- Ensure we have a list of paths ---
        if isinstance(Matlab_mat_files, (str, Path)):
            mat_paths = [str(Matlab_mat_files)]
        else:
            mat_paths = [str(p) for p in Matlab_mat_files]

        mat_ti_tidy_all = []

        for f in mat_paths:
            print(f"Reading MAT file: {f}")
            df = read_matfile(f)     # high-frequency data, Time index

            # --- Pick LiDAR-like wind speed / direction channels ---
            ws_cols = [c for c in df.columns if c.startswith("L_WS_")]
            wd_cols = [c for c in df.columns if c.startswith("L_WD_")]

            if not ws_cols:
                raise ValueError(f"No L_WS_* columns found in MAT file {f}")

            # Per-10-minute statistics (file is one 10-min block)
            ws_mean = df[ws_cols].mean()
            ws_std  = df[ws_cols].std()
            ti      = ws_std / ws_mean

            if wd_cols:
                wd_mean = df[wd_cols].mean()
            else:
                # If wind direction is missing, replace it with NaN
                wd_mean = pd.Series(np.nan, index=ws_cols)

            # Extract heights from column names: L_WS_1_44 -> 44
            heights = [float(col.split("_")[3]) for col in ws_cols]

            # Use mid-time of the 10-min block as representative timestamp
            # t_mid = df.index[0] + (df.index[-1] - df.index[0]) / 2

            tmp = pd.DataFrame({
                "Time":         [df.index[0]] * len(ws_cols),
                "height":       heights,
                "ti":           ti.values,
                "wind_speed":   ws_mean.values,
                "wind_direction": wd_mean.values,
            })

            mat_ti_tidy_all.append(tmp)

        # Concatenate all MAT intervals
        mat_ti_tidy = pd.concat(mat_ti_tidy_all, ignore_index=True)
        mat_ti_tidy["Time"] = pd.to_datetime(mat_ti_tidy["Time"])
        mat_ti_tidy = mat_ti_tidy.set_index("Time").sort_index().reset_index()

        #==== Include hub wind speed for binning ---
        if hub_height not in mat_ti_tidy["height"].unique():
            hub_height = min(mat_ti_tidy["height"].unique(), key=lambda x: abs(x - hub_height))
        print(f"Using closest MAT height {hub_height}m as reference for wind speed binning.")

        mat_ref = (
            mat_ti_tidy[mat_ti_tidy["height"] == hub_height]
            [["Time", "wind_speed"]]
            .rename(columns={"wind_speed": "hub_ws"})
        )

        # Broadcast hub_ws to all heights for each Time
        mat_ti_tidy = mat_ti_tidy.merge(mat_ref, on="Time", how="left")

        # Filter out unrealistic TI
        mat_ti_tidy = mat_ti_tidy[mat_ti_tidy["ti"] <= 1]

        #==== Bin by wind speed and wind direction ---
        mat_ti_tidy, bins_counts = bin_wind(mat_ti_tidy)
        mat_ti_tidy, bins_counts_wdir = bin_wdir(mat_ti_tidy)

        # Median TI per height
        ti_profile_mat = mat_ti_tidy.groupby("height", as_index=False)["ti"].median()

        # Median TI per wind-speed bin & height
        ti_profile_mat_binned = (
            mat_ti_tidy.groupby(["height", "ws_bin"], as_index=False, observed=True)["ti"]
            .median()
        )

        # Median TI per wind-speed bin, wind-direction bin & height
        ti_profile_mat_binned_wdir = (
            mat_ti_tidy
            .groupby(["height", "ws_bin", "wdir_bin"], as_index=False, observed=True)["ti"]
            .median()
        )

        return mat_ti_tidy, ti_profile_mat, ti_profile_mat_binned, ti_profile_mat_binned_wdir, bins_counts





