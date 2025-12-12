
import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import pandas as pd
import xarray as xr
from scipy.io import loadmat
import re

from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt

# import my own modules:
from data_readers.read_LiDAR_data import load_lidar_data, load_lidar_data_10min, lidar_finder
from data_readers.utils import color_text, NA_cols
from data_readers.read_MetMast_data import read_met, met_finder 
from data_readers.read_mat_data import read_matfile, mat_finder

from processor.calc_TI import calc_ti
from plot_result import plot_result

flag_MetMast = False
flag_LiDAR = True
flag_Matfile = False

if flag_MetMast is False:
    pth_met_base = None
else:
    pth_met_base = r'c:\Users\12650009\ArashData\Projects\47_Measurement_LoadValidation\Measurement_data\KNMI Data\cesar_tower' # metmast data high resolution


if flag_LiDAR is False:
    pth_lidar_base = None
else:
    pth_lidar_base = r'c:\Users\12650009\ArashData\Projects\47_Measurement_LoadValidation\Measurement_data\KNMI Data\ZP738_1s'

if flag_Matfile is False:
    pth_mat_base = None
else:
    pth_mat_base = r'C:\Users\12650009\ArashData\Projects\47_Measurement_LoadValidation\Measurement_data\Mat_files'

#------------- Read Metmast data: -------------------------
if pth_met_base is not None:

    start_date = pd.Timestamp("2020-05-01")           # inclusive
    end_date  = pd.Timestamp("2021-12-15") + pd.Timedelta(days=1)  # exclusive upper bound
    met_files = met_finder(pth_met_base, start_date, end_date) # find lidar files (all CSVs)



    if met_files is None:
        print("No met files found.")
    else:
        print(f"Found {len(met_files)} met files.")
    # ti_profile_mast, mast_df = calc_ti(met_files=met_files)
        ti_profile_mast, metmast_data_all, ti_profile_mast_binned, bins_counts = calc_ti(met_files=met_files, hub_height = 120.0)

        # lidar_ti, lidar_ti_tidy, mast_df, ti_profile_mast, ti_profile_lidar_binned, ti_profile_mast_binned, bin_counts = calc_ti(met_files=met_files)


        plot_result(ti_profile_mast, metmast_data_all)    

#-------------- Read LiDAR data: -------------------------
if pth_lidar_base is not None:

    start_date = pd.Timestamp("2020-05-01")           # inclusive
    # end_date  = pd.Timestamp("2020-05-14") + pd.Timedelta(days=1)  # exclusive upper bound
    end_date  = pd.Timestamp("2020-05-01") + pd.Timedelta(days=1)  # exclusive upper bound

    lidar_files = lidar_finder(pth_lidar_base,start_date, end_date) # find lidar files (all CSVs)

    # ti_profile_LiDAR, lidar_df  = calc_ti(lidar_files=lidar_files, hub_height = 120.0)
    if lidar_files is None:
        print("No lidar files found.")
    else:
        print(f"Found {len(lidar_files)} lidar files.")
        lidar_ti_tidy, ti_profile_Lidar, ti_profile_lidar_binned, ti_profile_lidar_binned_wdir, bins_counts = calc_ti(lidar_files=lidar_files, hub_height = 120.0)

        plot_result(ti_profile_Lidar, lidar_ti_tidy)    
        print('Lidar is done')




#-------------- Read Mat file data: -------------------------
if pth_mat_base is not None:

    # For now, mat_finder just finds all *.mat files in the folder (no date filter)
    start_date = pd.Timestamp("2025-07-07")           # inclusive
    end_date  = pd.Timestamp("2025-07-07") + pd.Timedelta(days=1)  # exclusive upper bound
    mat_files = mat_finder(pth_mat_base,start_date, end_date)

    # Compute TI from MAT files
    if mat_files is None:
        print("No mat files found.")
    else:
        print(f"Found {len(mat_files)} mat files.")
        mat_ti_tidy, ti_profile_mat, ti_profile_mat_binned, ti_profile_mat_binned_wdir, bins_counts = calc_ti(
            Matlab_mat_files=mat_files,
            hub_height=120.0,
        )

    # Reuse your existing plotting function:
        plot_result(ti_profile_mat, mat_ti_tidy)  

