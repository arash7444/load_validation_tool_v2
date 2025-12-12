import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Sequence, Union
from pathlib import Path

# Use your existing readers (same ones you use for TI)
from data_readers.read_MetMast_data import read_met          # mast: nc → DataFrame
from data_readers.read_LiDAR_data import (
    load_lidar_data, load_lidar_data_10min                   # lidar: csv → (avg, heights, std)
)
from data_readers.read_mat_data import read_matfile          # SCADA: mat → DataFrame

def _fit_alpha_with_uncertainty(heights_m, wsp):
    """
    Fit power-law shear exponent alpha from log(U) vs log(z)
    and estimate standard error of alpha from regression residuals.

    Returns
    -------
    alpha : float
    se_alpha : float   (np.nan if not enough points)
    n : int           number of points used
    """
    heights_m = np.asarray(heights_m, float)
    wsp = np.asarray(wsp, float)

    mask = (heights_m > 0.0) & np.isfinite(wsp) & (wsp > 0.0)
    x = np.log(heights_m[mask])
    y = np.log(wsp[mask])
    n = x.size

    if n < 2 or np.allclose(x, x.mean()):
        return np.nan, np.nan, n

    x_mean = x.mean()
    y_mean = y.mean()
    Sxx = np.sum((x - x_mean) ** 2)
    Sxy = np.sum((x - x_mean) * (y - y_mean))

    alpha = Sxy / Sxx
    b = y_mean - alpha * x_mean

    # residuals
    y_pred = alpha * x + b
    resid = y - y_pred
    if n > 2:
        sigma2 = np.sum(resid ** 2) / (n - 2)
        se_alpha = np.sqrt(sigma2 / Sxx)
    else:
        se_alpha = np.nan

    return float(alpha), float(se_alpha), int(n)


def _shear_from_profiles(wsp_profiles: pd.DataFrame,
                         window: int = 6,
                         name: str = "alpha"):
    """
    wsp_profiles: index = time, columns = heights (floats), values = wind speed.

    Returns
    -------
    alpha : pd.Series
    alpha_err : pd.Series   (standard error of alpha)
    alpha_roll_med : pd.Series
    alpha_roll_mean : pd.Series
    """
    heights = np.asarray(wsp_profiles.columns, float)

    alpha = pd.Series(index=wsp_profiles.index, dtype=float, name=name)
    alpha_err = pd.Series(index=wsp_profiles.index, dtype=float, name=name + "_err")

    for t, row in wsp_profiles.iterrows():
        a, se, _ = _fit_alpha_with_uncertainty(heights, row.values.astype(float))
        alpha.loc[t] = a
        alpha_err.loc[t] = se

    alpha = alpha.sort_index()
    alpha_err = alpha_err.sort_index()

    alpha_roll_med = alpha.rolling(window, center=True, min_periods=3).median()
    alpha_roll_mean = alpha.rolling(window, center=True, min_periods=3).mean()
    alpha_roll_med.name = name + "_roll_med"
    alpha_roll_mean.name = name + "_roll_mean"

    return alpha, alpha_err, alpha_roll_med, alpha_roll_mean



# def plot_shear_series(alpha, alpha_roll_med, alpha_roll_mean, label: str):
#     """
#     Simple 3-panel plot for ONE source (mast *or* lidar *or* mat).
#     """
#     fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

#     ax = axes[0]
#     ax.plot(alpha.index, alpha.values, "o-", label=f"shear slope for {label}")
#     ax.set_ylabel("Shear slope [-]")
#     ax.legend()
#     ax.grid(True)

#     ax = axes[1]
#     ax.plot(alpha_roll_med.index, alpha_roll_med.values, "o-",
#             label=f"shear slope for {label} - rolling median")
#     ax.set_ylabel("Shear slope [-]")
#     ax.legend()
#     ax.grid(True)

#     ax = axes[2]
#     ax.plot(alpha_roll_mean.index, alpha_roll_mean.values, "o-",
#             label=f"shear slope for {label} - rolling mean")
#     ax.set_ylabel("Shear slope [-]")
#     ax.set_xlabel("Time [hour]")
#     ax.set_title("Shear slope vs hour")
#     ax.legend()
#     ax.grid(True)

#     fig.tight_layout()
#     return fig



import matplotlib.pyplot as plt

def plot_shear_series(alpha: pd.Series,
                      alpha_err: pd.Series,
                      alpha_roll_med: pd.Series,
                      alpha_roll_mean: pd.Series,
                      label: str):
    """
    3-panel plot for ONE source (mast/lidar/SCADA) with error bars on raw alpha.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # 1) raw alpha + error bars
    ax = axes[0]
    ax.errorbar(
        alpha.index, alpha.values,
        yerr=alpha_err.values,
        fmt="o", capsize=2, label=f"shear slope for {label}"
    )
    ax.set_ylabel("Shear slope [-]")
    ax.legend()
    ax.grid(True)

    # 2) rolling median
    ax = axes[1]
    ax.plot(alpha_roll_med.index, alpha_roll_med.values, "o-",
            label=f"{label} - rolling median")
    ax.set_ylabel("Shear slope [-]")
    ax.legend()
    ax.grid(True)

    # 3) rolling mean
    ax = axes[2]
    ax.plot(alpha_roll_mean.index, alpha_roll_mean.values, "o-",
            label=f"{label} - rolling mean")
    ax.set_ylabel("Shear slope [-]")
    ax.set_xlabel("Time [hour]")
    ax.set_title("Shear slope vs hour")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    return fig


# ============================================================
# 1) Met-mast shear (NC files)
# ============================================================

def calc_shear_metmast_nc(met_files: Sequence[Union[str, Path]],
                          roll_window: int = 6):
    """
    Compute shear time series from met-mast NC files.

    Assumes read_met() returns a DataFrame with at least:
      index  : DatetimeIndex
      columns: 'height', 'wind_speed'

    Returns:
      alpha_mast, alpha_mast_roll_med, alpha_mast_roll_mean
    """
    met_files = [str(f) for f in met_files]

    dfs = []
    for f in met_files:
        df = read_met(f)             
        df.index = df.index.round("s")
        dfs.append(df)

    mast_df = pd.concat(dfs).sort_index()

    # Wide table: index=time, columns=height (m), values=wind_speed
    wsp_profiles = (
        mast_df
        .pivot_table(index=mast_df.index, columns="height", values="wind_speed")
        .sort_index()
    )
    wsp_profiles.columns = wsp_profiles.columns.astype(float)
    # alpha, alpha_roll_med, alpha_roll_mean = _shear_from_profiles(wsp_profiles, window=roll_window, name="alpha_mast")

    return _shear_from_profiles(wsp_profiles, window=roll_window, name="alpha_mast") # now returns: alpha_mast, alpha_mast_err, alpha_mast_roll_med, alpha_mast_roll_mean

    # return _shear_from_profiles(wsp_profiles, window=roll_window, name="alpha_mast")
    # return alpha, alpha_roll_med, alpha_roll_mean, wsp_profiles

# ============================================================
# 2) LiDAR shear (CSV files)
# ============================================================

def _build_lidar_profiles(lidar_avg: pd.DataFrame, height_lidar):
    """
    lidar_avg : 10-min averaged lidar data (from your loader).
    height_lidar : 1D array of heights (floats) returned by loader.

    We build a frame: index=time, columns=heights, values=wind_speed.
    """
    # Columns that contain wind speed
    speed_cols = [c for c in lidar_avg.columns if "Wind Speed" in c]

    mapping = {}
    for h in np.unique(np.asarray(height_lidar, float)):
        pattern = f"at {int(h)}m"
        cols_h = [c for c in speed_cols if pattern in c]
        if not cols_h:
            continue
        if len(cols_h) == 1:
            series = lidar_avg[cols_h[0]]
        else:
            # If there are multiple columns for the same height, average them
            series = lidar_avg[cols_h].mean(axis=1)
        mapping[h] = series

    wsp_profiles = pd.DataFrame(mapping)
    wsp_profiles = wsp_profiles.sort_index()
    wsp_profiles = wsp_profiles.reindex(sorted(wsp_profiles.columns), axis=1)
    wsp_profiles.columns = wsp_profiles.columns.astype(float)

    return wsp_profiles


def calc_shear_lidar_csv(lidar_files: Sequence[Union[str, "Path"]],
                         use_10min_loader: bool = False,
                         roll_window: int = 6):
    """
    Compute shear time series from LiDAR CSV files ONLY (no mast).

    - If use_10min_loader = False:
        uses load_lidar_data()  (1 s → 10 min inside)
    - If use_10min_loader = True:
        uses load_lidar_data_10min()  (files already 10-min averaged)

    Returns:
      alpha_lidar, alpha_lidar_roll_med, alpha_lidar_roll_mean
    """
    paths = [str(p) for p in lidar_files]

    avg_list = []
    heights_all = []

    for f in paths:
        if use_10min_loader:
            lidar_avg, height_lidar, lidar_std = load_lidar_data_10min(f)
        else:
            lidar_avg, lidar_numeric, height_lidar, lidar_std = load_lidar_data(f)
        avg_list.append(lidar_avg)
        heights_all.append(np.asarray(height_lidar, float))

    lidar_avg_all = pd.concat(avg_list).sort_index()
    height_lidar_all = np.unique(np.concatenate(heights_all))

    wsp_profiles = _build_lidar_profiles(lidar_avg_all, height_lidar_all)

    # return _shear_from_profiles(wsp_profiles, window=roll_window, name="alpha_lidar")
    alpha_lidar, alpha_lidar_err, aL_med, aL_mean = _shear_from_profiles(
    wsp_profiles, window=roll_window, name="alpha_lidar")

    return alpha_lidar, alpha_lidar_err, aL_med, aL_mean
# ============================================================
# 3- (MAT files)
# ============================================================
def calc_shear_mat_scada(mat_files, roll_window: int = 6):
    """
    Compute shear time series from SCADA MAT files.

    Assumes read_matfile() returns a DataFrame with:
      - DatetimeIndex (or 'Time' column)
      - columns 'L_WS_*' for wind speeds at different heights.

    We:
      * concatenate all MAT files
      * resample to 10-min means
      * compute one alpha per 10-min block
    """
    mat_files = [str(f) for f in mat_files]

    df_all = []

    for f in mat_files:
        df = read_matfile(f)

        # ensure DatetimeIndex
        if "Time" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index("Time")
        df = df.sort_index()

        df_all.append(df)

    if not df_all:
        raise ValueError("No MAT data frames provided.")

    df_all = pd.concat(df_all).sort_index()

    # pick SCADA wind-speed channels
    ws_cols = [c for c in df_all.columns if c.startswith("L_WS_")]
    if not ws_cols:
        raise ValueError("No L_WS_* columns found in MAT data.")

    # map columns → heights (e.g. L_WS_1_44 → 44.0)
    heights = [float(c.split("_")[3]) for c in ws_cols]
    col_to_h = dict(zip(ws_cols, heights))

    # 10-min mean at each channel
    ws_10 = df_all[ws_cols].resample("10T").mean()

    # rename columns to heights and merge possible duplicates
    ws_10 = ws_10.rename(columns=col_to_h)
    # if multiple channels share the same height, average them
    ws_profiles = ws_10.groupby(axis=1, level=0).mean()
    ws_profiles = ws_profiles.sort_index()
    ws_profiles.columns = ws_profiles.columns.astype(float)

    # now compute alpha per 10-min interval
    # alpha_mat, alpha_mat_roll_med, alpha_mat_roll_mean = _shear_from_profiles(
    #     ws_profiles, window=roll_window, name="alpha_mat"
    # )
    alpha_mat, alpha_mat_err, aMat_med, aMat_mean = _shear_from_profiles(
    ws_profiles, window=roll_window, name="alpha_mat"
)

    return alpha_mat, alpha_mat_err, aMat_med, aMat_mean
