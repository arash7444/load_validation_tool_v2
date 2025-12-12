import pandas as pd
import xarray as xr
import numpy as np

import re


def bin_wind(df):
    
    #==== Now Compute binned TI by wind speed ---
    bins = np.arange(0, 31.0, 1)
    labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins) - 1)] # e.g., 0-1, 1-2, 2-3, etc. so i can add them as columns easily
    # Assign bins to metmast_data based on hub wind speed
    df['ws_bin'] = pd.cut(df['hub_ws'], bins=bins, labels=labels, include_lowest=True)

    # count the number of observations per bin

    bin_counts = df.groupby('ws_bin', observed=True).size() / len(df['height'].unique())  # Divide by num heights since data is repeated per height
    bin_counts = bin_counts.astype(int)  # Since counts should be integers
    df['ws_bincount'] = df['ws_bin'].map(bin_counts) # assign each bin count to corresponding wind speed bin
    

    return df, bin_counts