import numpy as np
import pandas as pd

def cal_air_density(df):
    """
    Calculate air density from DataFrame with columns: 'Met Air Temp (C)', 'Met Pressure (mbar)', 'Met Humidity (%)'.
    Assumes 10-min resampled data.
    Returns: Series of rho (kg/m3) indexed by time.
    """
    T_C = df['Met Air Temp (C)']
    T_K = T_C + 273.15

    RH = df['Met Humidity (%)']
    if 'Met Pressure (mbar)' in df.columns:
        P_mbar = df['Met Pressure (mbar)']

    
        e_s = 610.78 * np.exp((17.27 * T_C) / (T_C + 237.3))  # Pa
        e = (RH / 100) * e_s  # Pa
        P_Pa = P_mbar * 100
        P_d = P_Pa - e
        M_d = 0.0289644  # kg/mol
        M_v = 0.01801528  # kg/mol
        R = 8.314462618  # J/(mol·K)
        rho = (P_d * M_d + e * M_v) / (R * T_K)

    else:
        P_mbar = 1013.25  # mbar, per ISO 2533 and IEC 61400-1
        R_s = 287.058  # J/(kg·K)
        rho = (P_mbar * 100) / (R_s * T_K)  # P in Pa

    return rho


if __name__ == "__main__":
        
    # Example usage after loading/resampling LiDAR to 10-min (lidar_avg from your func)
    lidar_avg['rho'] = cal_air_density(lidar_avg)  # Add column

    # Then bin by WS (e.g., ref WS at hub height)
    bins = np.arange(0, 25.5, 0.5)  # Per IEC 61400-12
    lidar_avg['ws_bin'] = pd.cut(lidar_avg['Horizontal Wind Speed (m/s) at 79m'], bins)  # Example hub height
    rho_binned = lidar_avg.groupby('ws_bin')['rho'].median()  # Or mean; use in HAWC2