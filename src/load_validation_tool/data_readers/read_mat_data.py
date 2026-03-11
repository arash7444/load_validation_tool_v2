from scipy.io import loadmat
import numpy as np
import pandas as pd
from utils import color_text, NA_cols
from pathlib import Path
import os
import re

def mat_finder(pth_mat_base: str, start_date=None, end_date=None) -> list:
    """
    Parameters
    ----------
    pth_mat_base : str
        Path to MAT files (directory or single file).

    start_date : pandas.Timestamp or datetime, optional
        Start date (inclusive).

    end_date : pandas.Timestamp or datetime, optional
        End date (exclusive, i.e. up to end_date - 1 day).

    Returns
    -------
    mat_files : list
        List of MAT file paths (as Path objects), optionally filtered by date.
    """

    # If date range is given, set up regex + info print
    if start_date is not None and end_date is not None:
        date_re = re.compile(r"(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)")
        print(
            f"Looking for MAT files between "
            f"{start_date.date()} and {(end_date - pd.Timedelta(days=1)).date()}."
        )
    else:
        date_re = None

    mat_files = []

    # --- If a single file is given, just return it ---
    if os.path.isfile(pth_mat_base):
        print("The path includes a file.")
        f = Path(pth_mat_base)

        if date_re is not None:
            m = date_re.search(f.name)
            if m:
                year, month, day = map(int, m.groups())
                file_date = pd.Timestamp(year=year, month=month, day=day)
                if start_date <= file_date < end_date:
                    mat_files.append(f)
            # if no match or out of range → nothing returned
        else:
            mat_files.append(f)

        return mat_files

    # --- Otherwise, search a directory recursively ---
    for f in Path(pth_mat_base).rglob("*.mat"):
        if date_re is not None:
            m = date_re.search(f.name)
            if not m:
                # filename without parsable date → skip
                continue

            year, month, day = map(int, m.groups())
            file_date = pd.Timestamp(year=year, month=month, day=day)

            if not (start_date <= file_date < end_date):
                continue  # outside date window, skip

        mat_files.append(f)

    # Sort for deterministic order
    mat_files = sorted(mat_files)

    print(f"Found {len(mat_files)} MAT files.")
    return mat_files


def extract_heights(df_ws):
    """
    extract heights from dataframe columns
    Parameters
    ----------
    df_ws : pandas.DataFrame
        Dataframe from read_matfile (Main office MAT files)

    Returns
    -------
    height : list
    
    """
    ws_fields = [f for f in df_ws.columns if f.startswith('L_WS_')]

    # for i in range(0,len(ws_fields)):
    #     height = ws_fields[i].split('_')[3]

    height = [f.split('_')[3] for f in ws_fields]
    print(height)
    return height

def read_matfile(matfile):
    """
    read mat file and return dataframe
    Parameters
    ----------
    matfile : str
        Path to MAT file.
    Returns
    -------
    df_ws : pandas.DataFrame
        Dataframe from read_matfile (Main office MAT files)
    """
    from scipy.io import loadmat, whosmat
    # matfile = r'h:\004_Loads\Data\H2A_RCA\H2A_2025-07-07_16-10-00.mat'
    # print(whosmat(matfile))  # -> list of (name, shape, dtype)



    # matfile = r'h:\004_Loads\Data\H2A_RCA\H2A_2025-07-07_16-10-00.mat'
    mat = loadmat(matfile, squeeze_me=True)

    #----------------- test the data---------------------
    # print(mat['DATA'].dtype) # check dtype
    # print(mat['DATA']['L_WS_1_44']) # test the data
    # print(mat['DATA']['L_WS_1_44'].shape) # check dimensions

    # ws = mat['DATA']['L_WS_1_44'].flatten()  # → 1D array
    # print(ws.shape)  # → (N,)

    # ws_fields = [f for f in mat['DATA'].dtype.names if f.startswith('L_WS_')]
    # ws_values = np.array([mat['DATA'][f].flatten()[0] for f in ws_fields])
    # print(ws_fields)
    # print(ws_values)
    #----------------- end of test the data---------------------


    all_fields = mat['DATA'].dtype.names # list of all fields

    df_ws = pd.DataFrame()
    for i in range(0,len(all_fields)):
        # df_ws[ws_fields[i]] = ws_values[i]
        df_ws[all_fields[i]] = mat['DATA'][all_fields[i]].item()
    

    print(df_ws.head())
    
    # let's find sampling frequency and dt 
    sample_length = df_ws.shape[0]
    sample_freq = sample_length / (60*10) # 10 min
    dt = 1 / sample_freq


    # let's find date and time (dont use it, it a bad way)
    # date_1 = os.path.splitext(os.path.basename(matfile))[0].split('_')[1]

    # hours_1 = os.path.splitext(os.path.basename(matfile))[0].split('_')[2]

    # df_ws['sample_freq'] = sample_freq
    # df_ws['dt'] = dt
    # df_ws['date'] = date_1
    # df_ws['hours'] = hours_1
    

    # df_ws['Time'] = pd.to_datetime(
    # df_ws['date'] + ' ' + df_ws['hours'],
    # format='%Y-%m-%d %H-%M-%S')

    # df_ws_3 = pd.DataFrame({field: mat['DATA'][field].item() for field in all_fields})

    #....
    df_ws = df_ws.copy()
    filename = Path(matfile).stem  # H2A_2025-07-07_16-10-00
    _, date_str, time_str = filename.split('_')

    time_str_clean = time_str.replace('-', ':')  # '16:10:00'

    start_time = pd.to_datetime(
        f"{date_str} {time_str_clean}",
        format="%Y-%m-%d %H:%M:%S",
    )

    df_ws["Time"] = start_time + pd.to_timedelta(df_ws.index * dt, unit="s")

    df_ws = df_ws.set_index("Time")
    # df_ws.index = df_ws.index.round('s') # round to seconds


    print(df_ws.head())

    return df_ws

    # height = extract_heights(df_ws)
    # print(height)


if __name__ == "__main__":


    # pth_mat_base = r'H:\004_Loads\Data\H2A_RCA'
    pth_mat_base = r'C:\Users\12650009\ArashData\Projects\47_Measurement_LoadValidation\Measurement_data\Mat_files'


    mat_files = mat_finder(pth_mat_base)


    #---------- read mat file: -------------------------
    mat_df_data_all = pd.DataFrame() # empty dataframe to store all lidar data
    for i in range(0,2):#range(0,len(mat_files)):
        df_ws = read_matfile(mat_files[i])
        mat_df_data_all = pd.concat([mat_df_data_all, df_ws], ignore_index=True)
        
    # height = extract_heights(df_ws)
    # print(height)
    print(mat_df_data_all.head())

    # lidar_data['rho'] = cal_air_density(lidar_avg)  # Add column
    print('single mat file dimensions: ',df_ws.shape)
    print('all mat files dimensions: ',mat_df_data_all.shape)
