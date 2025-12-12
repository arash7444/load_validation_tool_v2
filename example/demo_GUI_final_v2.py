import os
import sys
import datetime

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


from data_readers.read_LiDAR_data import load_lidar_data, load_lidar_data_10min, lidar_finder

from data_readers.utils import color_text, NA_cols
from data_readers.read_MetMast_data import read_met, met_finder
from data_readers.read_mat_data import read_matfile, mat_finder

from processor.calc_TI import calc_ti
from processor.calc_shear_simple import calc_shear_metmast_nc, calc_shear_lidar_csv, calc_shear_mat_scada

from plot_result import plot_result_plotly


# ==================================================================
#                         TI TAB CONTENT
# ==================================================================
def ti_tab():
    st.header("Turbulence Intensity (TI) Calculator")

    st.write(
        "This tab calculates Turbulence Intensity (TI) based on the following data sources:\n"
        "- `Met mast (nc files), LiDAR (csv files), Mat files (Matlab files)`\n"
        "- `The met mast and LiDAR data are coming from the same source (KNMI open source data)`\n"
        "- `The Mat files are coming from our own data`"
    )

    # ----------------- 1. Source type + hub height -----------------
    st.subheader("1. Select data source")

    source_type = st.selectbox(
        "Choose data source (TI)",
        ["Met mast (nc files)", "LiDAR (CSV files)", "Mat files"],
        key="ti_source_type",
    )

    hub_height = st.number_input("choosing desired height for TI calculation [m] (hub height)", value=120.0, key="ti_hub_height")

    # ----------------- 2. Path + time period -----------------------
    st.subheader("2. Select dataset path & time period")

    default_paths = {
        "Met mast (nc files)": r"c:\Users\12650009\ArashData\Projects\47_Measurement_LoadValidation\Measurement_data\KNMI Data\cesar_tower",
        "LiDAR (CSV files)": r"c:\Users\12650009\ArashData\Projects\47_Measurement_LoadValidation\Measurement_data\KNMI Data\ZP738_1s",
        "Mat files": r"C:\Users\12650009\ArashData\Projects\47_Measurement_LoadValidation\Measurement_data\Mat_files",
    }

    pth_base = st.text_input(
        "Base directory / file path (TI)",
        value=default_paths[source_type],
        key="ti_pth_base",
    )

    if "Mat files" in source_type:
        default_start = datetime.date(2025, 7, 7)
        default_end = datetime.date(2025, 7, 7)
    else:
        default_start = datetime.date(2020, 5, 1)
        default_end = datetime.date(2020, 5, 14)

    col1, col2 = st.columns(2)
    with col1:
        start_date_ui = st.date_input("Start date (TI)", value=default_start, key="ti_start")
    with col2:
        end_date_ui = st.date_input("End date (TI)", value=default_end, key="ti_end")

    start = pd.Timestamp(start_date_ui)
    end = pd.Timestamp(end_date_ui) + pd.Timedelta(days=1)

    # ----------------- 3. Find files -------------------------------
    st.markdown("---")
    st.subheader("3. Find files in selected period (TI)")

    if st.button("Find files in selected period (TI)"):
        try:
            if "Met mast (nc files)" in source_type:
                files = met_finder(pth_base, start, end)
            elif "LiDAR (CSV files)" in source_type:
                files = lidar_finder(pth_base, start, end)
            elif "Mat files" in source_type:
                files = mat_finder(pth_base, start, end)

            st.session_state["ti_files"] = files

            if not files:
                st.warning("No files found.")
            else:
                st.success(f"Found {len(files)} files:")
                # Show a short preview inline
                max_preview = 5
                st.write("Preview of first files:")
                for f in files[:max_preview]:
                    st.write(f"• `{f}`")
                if len(files) > max_preview:
                    st.write(f"... and {len(files) - max_preview} more.")

                # Optional: full list in an expander as a small table
                with st.expander("Show full file list (TI)", expanded=False):
                    df_files = pd.DataFrame({"files": files})
                    st.dataframe(df_files, height=200)

        except Exception as e:
            st.error("Error while searching for TI files:")
            st.exception(e)

    # ----------------- 4. Calculate TI -----------------------------
    st.markdown("---")
    if "ti_files" in st.session_state and st.session_state["ti_files"]:
        if st.button(" Calculate TI "):
            files = st.session_state["ti_files"]

            try:
                with st.spinner("Calculating TI..."):

                    if "Met mast" in source_type:
                        (
                            ti_profile_mast,
                            metmast_data_all,
                            ti_profile_mast_binned,
                            bins_counts,
                        ) = calc_ti(met_files=files, hub_height=hub_height)

                        ti_profile_to_plot = ti_profile_mast
                        data_to_plot = metmast_data_all
                        preview_df = metmast_data_all

                    elif "LiDAR" in source_type:
                        (
                            lidar_ti_tidy,
                            ti_profile_Lidar,
                            ti_profile_lidar_binned,
                            ti_profile_lidar_binned_wdir,
                            bins_counts,
                        ) = calc_ti(lidar_files=files, hub_height=hub_height)

                        ti_profile_to_plot = ti_profile_Lidar
                        data_to_plot = lidar_ti_tidy
                        preview_df = lidar_ti_tidy

                    elif "Mat files" in source_type:
                        (
                            mat_ti_tidy,
                            ti_profile_mat,
                            ti_profile_mat_binned,
                            ti_profile_mat_binned_wdir,
                            bins_counts,
                        ) = calc_ti(
                            Matlab_mat_files=files,
                            hub_height=hub_height,
                        )

                        ti_profile_to_plot = ti_profile_mat
                        data_to_plot = mat_ti_tidy
                        preview_df = mat_ti_tidy

                st.success("TI calculation completed.")

                # st.subheader("TI data preview")
                # try:
                #     st.dataframe(preview_df.head())
                # except Exception:
                #     st.write("Could not display TI dataframe. Type:", type(preview_df))

                st.subheader("TI data preview")
                try:
                    st.dataframe(preview_df, height=400)  # scrollable table
                except Exception:
                    st.write("Could not display TI dataframe. Type:", type(preview_df))





                st.subheader("TI plot(s)")
                try:
                    figs = plot_result_plotly(ti_profile_to_plot, data_to_plot)
                    st.plotly_chart(figs["profile"], use_container_width=True)
                    st.plotly_chart(figs["scatter"], use_container_width=True)
                    st.plotly_chart(figs["box"], use_container_width=True)
                except Exception as e:
                    st.error("Error while plotting TI result:")
                    st.exception(e)

            except Exception as e:
                st.error("Error during TI calculation:")
                st.exception(e)
    else:
        st.info("Click 'Find files' first (TI).")


# ==================================================================
#                         SHEAR TAB CONTENT
# ==================================================================
def shear_tab():
    st.header("Vertical Shear Calculator")

    st.write(
        "This tab calculates Vertical Shear value based on the following data sources:\n"
        "- `Met mast (nc files), LiDAR (csv files), Mat files (Matlab files)`\n"
        "- `The met mast and LiDAR data are coming from the same source (KNMI open source data)`\n"
        "- `The Mat files are coming from our own data`"
    )

    # ----------------- 1. Source type ------------------------------
    st.subheader("1. Select data source")

    source_type = st.selectbox(
        "Choose data source (shear)",
        ["Met mast (nc files)", "LiDAR (CSV files)", "Mat files"],
        key="shear_source_type",
    )

    # ----------------- 2. Path + time period -----------------------
    st.subheader("2. Select dataset path & time period")

    default_paths = {
        "Met mast (nc files)": r"c:\Users\12650009\ArashData\Projects\47_Measurement_LoadValidation\Measurement_data\KNMI Data\cesar_tower",
        "LiDAR (CSV files)": r"c:\Users\12650009\ArashData\Projects\47_Measurement_LoadValidation\Measurement_data\KNMI Data\ZP738_1s",
        "Mat files": r"H:\004_Loads\Data\H2A_RCA\data",
    }

    pth_base = st.text_input(
        "Base directory / file path (shear)",
        value=default_paths[source_type],
        key="shear_pth_base",
    )
    
    if "Mat file" in source_type:
        default_start = datetime.datetime(2025,7,7)
        default_end = datetime.datetime(2025,7,7)
    else:
        default_start = datetime.datetime(2020, 5, 1)
        default_end = datetime.datetime(2020, 5, 3)

    col1, col2 = st.columns(2)
    with col1:
        start_date_ui = st.date_input(
            "Start date (shear)", value=default_start, key="shear_start"
        )
    with col2:
        end_date_ui = st.date_input(
            "End date (shear)", value=default_end, key="shear_end"
        )

    start = pd.Timestamp(start_date_ui)
    end = pd.Timestamp(end_date_ui) + pd.Timedelta(days=1)

    # ----------------- 3. Find files -------------------------------
    st.markdown("---")
    st.subheader("3. Find files in selected period (shear)")

    if st.button("Find files in selected period (shear)"):
        try:
            if "Met mast" in source_type:
                files = met_finder(pth_base, start, end)
            elif "LiDAR" in source_type:
                files = lidar_finder(pth_base, start, end)
            elif "Mat files" in source_type:
                files = mat_finder(pth_base, start, end)

            st.session_state["shear_files"] = files

            if not files:
                st.warning("No files found.")
            else:
                st.success(f"Found {len(files)} files:")
                max_preview = 5
                st.write("Preview of first files:")
                for f in files[:max_preview]:
                    st.write(f"• `{f}`")
                if len(files) > max_preview:
                    st.write(f"... and {len(files) - max_preview} more.")

                with st.expander("Show full file list (shear)", expanded=False):
                    df_files = pd.DataFrame({"files": files})
                    st.dataframe(df_files, height=200)
                    
        except Exception as e:
            st.error("Error while searching for shear files:")
            st.exception(e)

    # ----------------- 4. Calculate shear --------------------------
    st.markdown("---")
    if "shear_files" in st.session_state and st.session_state["shear_files"]:
        if st.button("Calculate vertical shear"):
            files = st.session_state["shear_files"]

            try:
                with st.spinner("Calculating shear..."):

                    if "Met mast" in source_type:
                        alpha, alpha_err, alpha_med, alpha_mean = calc_shear_metmast_nc(
                            files, roll_window=6
                        )

                    elif "LiDAR" in source_type:
                        alpha, alpha_err, alpha_med, alpha_mean = calc_shear_lidar_csv(
                            files,
                            use_10min_loader=False,
                            roll_window=6,
                        )

                    elif "Mat files" in source_type:
                        alpha, alpha_err, alpha_med, alpha_mean = calc_shear_mat_scada(
                            files, roll_window=6
                        )

                st.success("Shear calculation completed.")

                st.subheader("Shear time series (alpha)")
                # st.dataframe(alpha.head())

                try:
                    st.dataframe(alpha, height=400)  # scrollable table
                except Exception:
                    st.write("Could not display TI dataframe. Type:", type(alpha))





                st.subheader("Shear plot")
                try:
                    fig = make_subplots(
                        rows=3,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.06,
                        subplot_titles=[
                            f"{source_type} shear slope with uncertainty",
                            f"{source_type} rolling median",
                            f"{source_type} rolling mean",
                        ],
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=alpha.index,
                            y=alpha.values,
                            mode="lines+markers",
                            error_y=dict(
                                type="data",
                                array=alpha_err.values,
                                visible=True,
                            ),
                            name=f"{source_type} shear slope",
                        ),
                        row=1,
                        col=1,
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=alpha_med.index,
                            y=alpha_med.values,
                            mode="lines+markers",
                            name=f"{source_type} rolling median",
                        ),
                        row=2,
                        col=1,
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=alpha_mean.index,
                            y=alpha_mean.values,
                            mode="lines+markers",
                            name=f"{source_type} rolling mean",
                        ),
                        row=3,
                        col=1,
                    )

                    fig.update_layout(
                        height=850,
                        hovermode="x unified",
                        showlegend=True,
                        legend_title_text="Series",
                        margin=dict(t=80, b=60),
                    )
                    fig.update_yaxes(title_text="Shear slope [-]", row=1, col=1)
                    fig.update_yaxes(title_text="Shear slope [-]", row=2, col=1)
                    fig.update_yaxes(title_text="Shear slope [-]", row=3, col=1)
                    fig.update_xaxes(title_text="Time [hour]", row=3, col=1)

                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error("Error while plotting shear result:")
                    st.exception(e)

            except Exception as e:
                st.error("Error during shear calculation:")
                st.exception(e)
    else:
        st.info("Click 'Find files' first (shear).")


# ==================================================================
#                               MAIN
# ==================================================================
def main():
    st.set_page_config(page_title="TI & Shear Calculator", layout="centered")

    st.title("TI & Shear Calculator (Met mast / LiDAR / MAT)")

    st.markdown(
        """
        <style>
            /* Make tab labels bigger & add padding */
            .stTabs [data-baseweb="tab-list"] button {
                font-size: 20px !important;
                font-weight: 700 !important;
                padding: 12px 25px !important;   /* top/bottom = 12px, left/right = 25px */
                margin-right: 8px !important;    /* spacing between tabs */
                border-radius: 6px 6px 0px 0px !important;
            }

            /* Active tab styling */
            .stTabs [aria-selected="true"] {
                background-color: #e6f0ff !important;  /* light blue background */
                color: #1a73e8 !important;             /* blue text */
                border-bottom: 3px solid #1a73e8 !important;
            }

            /* Inactive tab styling */
            .stTabs [aria-selected="false"] {
                background-color: #f6f6f6 !important;
                color: #555 !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


    ti_tab_obj, shear_tab_obj = st.tabs(["TI", "Shear"])

    with ti_tab_obj:
        ti_tab()

    with shear_tab_obj:
        shear_tab()


if __name__ == "__main__":
    main()
