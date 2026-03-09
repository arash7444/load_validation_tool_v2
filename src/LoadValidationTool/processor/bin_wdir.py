import pandas as pd
import xarray as xr
import numpy as np

import re


def bin_wdir(df):
    #==== Now Compute binned TI by wind speed ---
    bins = np.arange(0, 360.0, 10)
    labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins) - 1)] # e.g., 0-10, 10-20, etc. so i can add them as columns easily
    # Assign bins to metmast_data based on hub wind speed
    df['wdir_bin'] = pd.cut(df['wind_direction'], bins=bins, labels=labels, include_lowest=True)

    bin_counts = df.groupby('wdir_bin', observed=True).size() / len(df['height'].unique())  # Divide by num heights since data is repeated per height
    bin_counts = bin_counts.astype(int)  # Since counts should be integers

    return df, bin_counts
    # count the number of observations per bin