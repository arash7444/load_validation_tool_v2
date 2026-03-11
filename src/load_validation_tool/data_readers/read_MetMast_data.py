import numpy as np
import pandas as pd
import xarray as xr
from scipy.io import loadmat

import re
import os
from pathlib import Path
from data_readers.utils import color_text, NA_cols
from typing import List, Tuple

def read_met(met_nc_file: str) -> pd.DataFrame:
    """
    read Metmast data and return a dataframe 
    
    Parameters
    ----------
        met_nc_file: str
        path to Metmast netcdf file

    Returns
    -------
        mast_df: pandas.DataFrame
    
    """

    met_ds = xr.open_dataset(met_nc_file) 

    # print(met_ds.data_vars)
    # print(met_ds.keys())
    # print(met_ds.sel(z=10).D.values)


    met_ds['z'].attrs # height
    met_ds['D'].attrs # wind direction and corrected
    met_ds['SD'].attrs # Sdv wind direction
    met_ds['Q'].attrs # humidity
    met_ds['TA'].attrs # air temperature
    met_ds['PF'].attrs # Max wind speed of the wind gust and corrected
    met_ds['F'].attrs # windspeed and corrected
    met_ds['MF'].attrs # Min windspeed not corrected
    met_ds['SF'].attrs # Sdv windspeedm and corrected


    mast_df = met_ds[["z","F", "SF","D", "SD", "TA","Q"]].to_dataframe().reset_index().set_index("time")
    mast_df = mast_df.rename(columns={"z":"height", "F":"wind_speed", "SF":"wind_speed_std", "D":"wind_direction","SD":"wind_direction_std", "TA":"air_temp","Q":"humidity"})
    print("Met mast data frame with N/A values:")
    print(mast_df)
    print('----------------------------------------------------------')
    print("Met mast data frame without N/A values:")
    mast_df = mast_df.dropna(how="any")
    print(mast_df)

    # mast_df_z = met_ds[["time","F", "D", "SF", "TA"]].to_dataframe().reset_index().set_index("z")

    return mast_df

# def read_matfile(matfile):
#     mat = loadmat(matfile, squeeze_me=True)

#     all_fields = mat['DATA'].dtype.names # list of all fields
#     df_ws = pd.DataFrame({field: mat['DATA'][field].item() for field in all_fields})
#     ws_fields = [f for f in df_ws.columns if f.startswit<h('L_WS_')]
#     height = [f.split('_')[3] for f in ws_fields]
#     df_ws['height'] = height

#     return df_ws





def met_finder(pth_met_base: str,start_date=None, end_date=None):
    """
    Parameters
    ----------
    pth_met_base : str
        Path to metmast folder (directory or single file).

    start_date : pandas.Timestamp or datetime, optional. the format is YYYY-MM-DD     

    end_date : pandas.Timestamp or datetime, optional. the format is YYYY-MM-DD 
        

    Returns
    -------
    met_nc : list
        List of met_nc file paths, optionally filtered by date.
    """

    #--- check if start and end date are provided
    if start_date is not None and end_date is not None:
        date_re = re.compile(r'(20\d{2})[-_]?([01]\d)[-_]?([31]\d)')
        print(f"Looking for Met-mast files between {start_date.date()} and {(end_date - pd.Timedelta(days=1)).date()}.")



    if os.path.isfile(pth_met_base):
        print("The path includes a file.")
        met_nc = [pth_met_base]
    else:
        met_nc = []
        for p in Path(pth_met_base).rglob("*.nc"):
            if start_date is None:
                met_nc.append(str(p))
                continue
            else:
                filename = Path(p).name

                
                underscore_index = filename.rfind('_') #  finds the index of the last underscore in the filename

                # WARNING: This is a hacky way to extract the date from the filename and it is hard coded for the current filename format
                date_str = filename[underscore_index+1:-3] #Extract the date from the filename 

                # Split the date string into year and month
                y = date_str[:4]
                mo = date_str[4:6]

                # year = int(y)
                # month = int(mo)
                
                # m = date_re.search(p.name)
                # if not m:
                #     continue
                # year, month = map(int, m.groups())
                file_day = pd.Timestamp(year=int(y), month=int(mo), day=1) # pd.Timestamp(year=y, month=mo, day=1) # i dont have day i can not use it

                if (file_day >= start_date) and (file_day < end_date):
                    met_nc.append(str(p))


        
        # # find all nc file
        # met_nc = []
        # for files in Path(pth_met_base).rglob("*.nc"):
        #     # print(files)
        #     met_nc.append(files)
        #     # pass
    return met_nc


if __name__ == "__main__":

    pth_met_base = r'c:\Users\12650009\ArashData\Projects\47_Measurement_LoadValidation\Measurement_data\KNMI Data\cesar_tower'


    met_files = met_finder(pth_met_base) # find lidar files (all CSVs)


    #------------- read met mast data: -------------------------
    df_met_all = pd.DataFrame()
    for i in range(0,2):#range(0,len(met_files)):
        met_ds = read_met(met_files[i])
        # print(met_ds.head())

        df_met_all = pd.concat([df_met_all, met_ds], ignore_index=True)


    # lidar_data['rho'] = cal_air_density(lidar_avg)  # Add column
    # print(met_ds.columns)
    print('single metmast file dimensions: ',met_ds.shape)
    print('all metmast files dimensions: ',df_met_all.shape)



