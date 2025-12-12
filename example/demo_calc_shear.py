import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_readers.read_MetMast_data import met_finder
from data_readers.read_LiDAR_data import lidar_finder
from data_readers.read_mat_data import mat_finder

from processor.calc_shear_simple import calc_shear_metmast_nc, plot_shear_series, calc_shear_lidar_csv, calc_shear_mat_scada

pth_met_base = r'c:\Users\12650009\ArashData\Projects\47_Measurement_LoadValidation\Measurement_data\KNMI Data\cesar_tower' 

start = pd.Timestamp("2020-05-01")
end   = pd.Timestamp("2020-05-03") + pd.Timedelta(days=1)

met_files = met_finder(pth_met_base, start, end)

alpha_mast, alpha_mast_err, alpha_mast_roll_med, alpha_mast_roll_mean = calc_shear_metmast_nc(met_files, roll_window=6)

fig = plot_shear_series(alpha_mast,alpha_mast_err, alpha_mast_roll_med, alpha_mast_roll_mean, label="metmast")

fig.show()
print(alpha_mast.head())

# plt.figure()
# plt.plot(wsp_profiles_mast.iloc[0,:],wsp_profiles_mast.columns, "o-")
# plt.show()



pth_lidar_base = r'c:\Users\12650009\ArashData\Projects\47_Measurement_LoadValidation\Measurement_data\KNMI Data\ZP738_1s'

start = pd.Timestamp("2020-05-01")
end   = pd.Timestamp("2020-05-03") + pd.Timedelta(days=1)

lidar_files = lidar_finder(pth_lidar_base, start, end)

alpha_lidar, alpha_lidar_err, aL_med, aL_mean = calc_shear_lidar_csv(
    lidar_files,
    use_10min_loader=False,   # or True, depending on whether LiDAR files are already 10-min averaged
    roll_window=6,
)

fig = plot_shear_series(alpha_lidar, alpha_lidar_err, aL_med, aL_mean, label="lidar")

fig.show()
print(alpha_lidar.head())


pth_mat_base = r'H:\004_Loads\Data\H2A_RCA\data'

start = pd.Timestamp("2025-07-15")
end   = pd.Timestamp("2025-07-18")

mat_files = mat_finder(pth_mat_base, start, end)

alpha_mat, alpha_mat_err, aMat_med, aMat_mean = calc_shear_mat_scada(mat_files, roll_window=6)
fig = plot_shear_series(alpha_mat, alpha_mat_err, aMat_med, aMat_mean, label="SCADA")

fig.show()
print(alpha_mat.head())