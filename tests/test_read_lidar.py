# from read_LiDAR_data import load_lidar_data, load_lidar_data_10min

from load_validation_tool.data_readers import (
    load_lidar_data,
    load_lidar_data_10min,
    lidar_finder,
)
from load_validation_tool.data_readers import (
    detect_heights,
    color_text,
    NA_cols,
    _SPEED_RE,
    _DIR_RE,
)
import pandas as pd

pth_lidar_base = r"tests\lidar_data"

start = pd.Timestamp("2020-05-01")
end = pd.Timestamp("2020-05-03") + pd.Timedelta(days=1)

lidar_files = lidar_finder(pth_lidar_base, start, end)
if lidar_files is None:
    raise Exception("No lidar files found.")

lidar_avg, lidar_numeric, height_lidar, lidar_std = load_lidar_data(lidar_files[0])
if len(lidar_avg) == 0:
    raise Exception("Lidar Dataframe is empty.")
