import numpy as np
import pandas as pd
import xarray as xr
from scipy.io import loadmat
import warnings

import re
import os
from pathlib import Path
from .utils import color_text, NA_cols
from typing import List, Tuple


def load_lidar_data_10min(
    lidar_file: str,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Load LiDAR data that is already averaged to 10-minute intervals.
    Therefore, we don't need to resample the data and standard deviation is not needed
    Parameters:
    -----------
    lidar_file : str
        Path to the CSV file containing the LiDAR data.
        CSV file is expected to have columns for Time and Date, wind speed, and wind direction

    Returns:
    --------
    lidar_avg : pd.DataFrame
        DataFrame containing the average wind speed and direction for each 10-minute interval.
    height_lidar : np.ndarray
        Array containing the heights of the LiDAR data.
    lidar_std : pd.DataFrame
        DataFrame containing the standard deviation of wind speed and direction for each 10-minute interval.
    """

    ##1- load lidar data
    lidar_data = pd.read_csv(lidar_file, skiprows=1)
    lidar_data.columns = [
        col.strip() for col in lidar_data.columns
    ]  # remove whitespace from beginning and end of column names

    ##2- parse date and time  and use it as index so that it would be easy to compare with the mast data
    lidar_data["Time"] = pd.to_datetime(
        lidar_data["Time and Date"], format="%d/%m/%Y %H:%M:%S"
    )  # parse date and time

    # Now I need to convert the time to seconds:
    lidar_data["Time_seconds"] = (
        lidar_data["Time"].dt.hour * 3600  # convert hours to seconds
        + lidar_data["Time"].dt.minute * 60  # convert minutes to seconds
        + lidar_data["Time"].dt.second
    )  # convert seconds to seconds

    # I use  lidar_data['Time'] as index so that it would be easy to resample and compare it with the mast data
    lidar_data = lidar_data.set_index(
        "Time"
    )  # Use the datetime as the index (for resampling).

    ###3- take only wind related data
    # Take only wind related data and covert them to numbers
    wind_cols = [
        cols
        for cols in lidar_data.columns
        if "Wind Speed" in cols or "Wind Direction" in cols
    ]

    # 4- Create a new DataFrame with selected columns
    lidar_numeric = lidar_data[
        wind_cols
    ]  # Create a new DataFrame with selected columns

    lidar_avg_org = lidar_numeric

    lidar_std = pd.DataFrame()
    ##5- seperate standard deviation
    std_col = [cols for cols in lidar_avg_org if "Std. Dev." in cols]
    lidar_std[std_col] = lidar_avg_org[std_col]
    lidar_avg = lidar_avg_org.drop(columns=std_col)

    # lidar_avg['Time_seconds'] = lidar_avg['Time_seconds'] - lidar_avg['Time_seconds'].min()

    ##6- pick heights from column names:
    height_lidar = []  # create empty list to store heights
    for col in lidar_avg.columns:
        if "Wind Speed" in col or "Wind Direction" in col:
            """ I want to extract the height from the column name I can use regex """
            height_flag = True
            if height_flag == True:
                if re.search(
                    "at", col
                ):  # check at in column name so that I can extract the height
                    h_match = re.search(
                        r"at \d+m", col
                    )  # check if height is in column name
                    if h_match:
                        height = float(
                            h_match.group().replace("at ", "").replace("m", "")
                        )  # extract height from column name
                        height_lidar.append(height)
                        # new_col = col.replace(f'at {height}m', f'{height}') # add height to column name
                        # lidar_avg = lidar_avg.rename(columns={col: new_col}) # rename column

    height_lidar = np.sort(np.unique(height_lidar))

    return lidar_avg, height_lidar, lidar_std


def load_lidar_data(lidar_file):
    """
    Load high frequency LiDAR data from a CSV file and resample it to 10-minute intervals.
    Parameters:
    -----------
    lidar_file : str
        Path to the CSV file containing the LiDAR data.
        CSV file is expected to have columns for Time and Date, wind speed, and wind direction

    Returns:
    --------
    lidar_avg : pd.DataFrame
        DataFrame containing the average wind speed and direction for each 10-minute interval.
    lidar_numeric : pd.DataFrame
        DataFrame containing the wind speed and direction data.
    height_lidar : np.ndarray
        Array containing the heights of the LiDAR data.
    lidar_std : pd.DataFrame
        DataFrame containing the standard deviation of wind speed and direction for each 10-minute interval.
    """
    lidar_data = pd.read_csv(lidar_file, skiprows=1)
    lidar_data.columns = [
        col.strip() for col in lidar_data.columns
    ]  # remove whitespace from beginning and end of column names
    lidar_data["Time"] = pd.to_datetime(
        lidar_data["Time and Date"], format="%d/%m/%Y %H:%M:%S"
    )  # parse date and time

    # Now I need to convert the time to seconds:
    lidar_data["Time_seconds"] = (
        lidar_data["Time"].dt.hour * 3600  # convert hours to seconds
        + lidar_data["Time"].dt.minute * 60  # convert minutes to seconds
        + lidar_data["Time"].dt.second
    )  # convert seconds to seconds

    # I use  lidar_data['Time'] as index so that it would be easy to resample and compare it with the mast data
    lidar_data = lidar_data.set_index(
        "Time"
    )  # Use the datetime as the index (for resampling).

    # Take only wind related data and covert them to numbers
    wind_cols = [
        cols
        for cols in lidar_data.columns
        if "Wind Speed" in cols or "Wind Direction" in cols
    ]

    lidar_numeric = lidar_data[
        wind_cols
    ]  # Create a new DataFrame with selected columns
    lidar_numeric["Time_seconds"] = (
        lidar_numeric.index.hour * 3600  # convert hours to seconds
        + lidar_numeric.index.minute * 60  # convert minutes to seconds
        + lidar_numeric.index.second
    )  # convert seconds to seconds

    lidar_avg = lidar_numeric.resample(
        "10min"
    ).mean()  # Resample to 10-minute intervals
    lidar_std = lidar_numeric.resample("10min").std()  # Resample to 10-minute intervals
    # if lidar_std.isnull().any().any(): # check if any of the values are missing
    #     raise ValueError("Lidar data is not high frequency because all std values are missing\n"
    #                      "use a high frequency lidar file or call 'load_lidar_data_10min' instead")
    if lidar_std.isnull().any().any():  # check if any of the values are missing
        warnings.warn(
            "Lidar data is not high frequency because all std values are missing\n"
            "use a high frequency lidar file or call 'load_lidar_data_10min' instead"
        )
        try:
            lidar_avg, height_lidar, lidar_std = load_lidar_data_10min(
                lidar_file
            )  # load lidar data if they are low resolution (10 min average)
            # return lidar_avg, lidar_numeric, height_lidar, lidar_std
        except:
            raise ValueError(
                "Lidar data is not high frequency because all std values are missing\n"
                "use a high frequency lidar file or call 'load_lidar_data_10min' instead"
            )

    # lidar_avg['Time_seconds'] = lidar_avg['Time_seconds'] - lidar_avg['Time_seconds'].min()

    height_lidar = []  # create empty list to store heights
    for col in lidar_avg.columns:
        if "Wind Speed" in col or "Wind Direction" in col:
            """ I want to extract the height from the column name I can use regex """
            height_flag = True
            if height_flag == True:
                if re.search(
                    "at", col
                ):  # check at in column name so that I can extract the height
                    h_match = re.search(
                        r"at \d+m", col
                    )  # check if height is in column name
                    if h_match:
                        height = float(
                            h_match.group().replace("at ", "").replace("m", "")
                        )  # extract height from column name
                        height_lidar.append(height)
                        # new_col = col.replace(f'at {height}m', f'{height}') # add height to column name
                        # lidar_avg = lidar_avg.rename(columns={col: new_col}) # rename column

    height_lidar = np.sort(np.unique(height_lidar))

    return lidar_avg, lidar_numeric, height_lidar, lidar_std


def lidar_finder(pth_lidar_base: str, start_date=None, end_date=None) -> List[str]:
    """
    find all csv files in a directory and subdirectories
    Paremeters:
    -----------
        pth_lidar_base: str
            path to lidar files
    start_date: pd.Timestamp or None
        start date to filter files
    end_date: pd.Timestamp or None
        end date to filter files

    Returns:
    --------
      lidar_csvs: List[str]
        list of lidar csv files found in the directory and subdirectories

    """
    print(
        f"Looking for lidar files between {start_date.date()} and {(end_date - pd.Timedelta(days=1)).date()}."
    )

    if os.path.isfile(pth_lidar_base):
        print("The path includes a file.")
        lidar_csvs = [pth_lidar_base]
    else:
        lidar_csvs = []
        for p in Path(pth_lidar_base).rglob("*.csv"):
            if start_date is None:
                lidar_csvs.append(str(p))
                continue
            else:
                filename = Path(p).name

                # underscore_index = filename.rfind('_') #  finds the index of the last underscore in the filename
                matches = re.finditer("_", filename)
                underscore_index = [match.start() for match in matches]

                # WARNING: This is a hacky way to extract the date from the filename and it is hard coded for the current filename format
                date_str = filename[
                    underscore_index[-2] + 1 : underscore_index[-1]
                ]  # Extract the date from the filename

                # Split the date string into year and month
                y = date_str[:4]
                mo = date_str[4:6]
                d = date_str[6:]

                file_day = pd.Timestamp(
                    year=int(y), month=int(mo), day=int(d)
                )  # pd.Timestamp(year=y, month=mo, day=1) # i dont have day i can not use it

                if (file_day >= start_date) and (file_day < end_date):
                    lidar_csvs.append(str(p))

    return lidar_csvs


def _as_path_list(lidar_files):
    """
    No matter how the user specifies LiDAR input, the rest code always receives a list of file paths.
    becasue a users might pass a single file path or a list  or tuple of file paths or a directory containing many CSV files
    This function normalizes all of those cases into one consistent format.

    Parameters
    ----------
    lidar_files : str or list or tuple
        Path to LiDAR CSV file(s) or directory containing LiDAR CSV files.
    Returns:
    -------
    list of str
        List of LiDAR CSV file paths.

    """
    if isinstance(lidar_files, (list, tuple)):  # list or tuple case
        return [str(p) for p in lidar_files]
    p = Path(lidar_files)  # convert to Path object
    if p.is_dir():  # directory case
        return [str(x) for x in sorted(p.glob("*.CSV"))] + [
            str(x) for x in sorted(p.glob("*.csv"))
        ]
    return [str(p)]  # single file path case


def load_and_concat_lidar(lidar_files):
    """
    Reads one or many lidar CSVs and returns two concatenated frames.
    Note that this function works with your two loaders:
      - load_lidar_data (1 s data -> returns avg, numeric, heights, std)
      - load_lidar_data_10min (10 min data -> returns avg, heights, std)

    Parameters:
    ----------
    lidar_files : str or list or tuple
        Path to LiDAR CSV file(s) or directory containing LiDAR CSV files.

    Returns:
    -------
        lidar_avg_all : pd.DataFrame
            Concatenated DataFrame of LiDAR average data from all files.
        lidar_std_all : pd.DataFrame
            Concatenated DataFrame of LiDAR standard deviation data from all files.
    """
    paths = _as_path_list(lidar_files)
    if not paths:
        raise FileNotFoundError("No lidar CSV files found in the given input.")

    avg_list, std_list = [], []
    for f in paths:
        try:
            avg, _, _, std = load_lidar_data(f)  # 1-second files (your function)
        except TypeError:
            avg, _, std = load_lidar_data_10min(
                f
            )  # native 10-min files (your function)
        avg_list.append(avg.sort_index())
        std_list.append(std.sort_index())

    lidar_avg_all = pd.concat(avg_list).sort_index()
    lidar_std_all = pd.concat(std_list).sort_index()

    # De-dup timestamps, align indices, keep structure unchanged elsewhere
    lidar_avg_all = lidar_avg_all[~lidar_avg_all.index.duplicated(keep="first")]
    lidar_std_all = lidar_std_all.loc[lidar_avg_all.index]
    lidar_avg_all.index = lidar_avg_all.index.round("s")
    lidar_std_all.index = lidar_std_all.index.round("s")

    return lidar_avg_all, lidar_std_all


# --- Main Execution ---
if __name__ == "__main__":
    lidar_file = r"h:\004_Loads\Projects\LoadValidation_1to1\Data\ZP738_1s\ZephIR_Cabauw_ZP738_raw_20200607_v1.CSV"

    lidar_avg, lidar_numeric, height_lidar, lidar_std = load_lidar_data(lidar_file)
    print(height_lidar)
