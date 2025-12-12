# plot_result.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go


def plot_result(ti_profile_mast, mast_df):
    plt.figure(figsize=(6, 5))
    plt.plot(ti_profile_mast["height"], ti_profile_mast["ti"], marker="o")
    plt.xlabel("Height [m]"); plt.ylabel("TI [-]")
    plt.title("Mast Median TI Profile"); plt.grid(True)

    plt.figure(figsize=(6, 5))
    sns.scatterplot(x="height", y="ti", data=mast_df, s=20)
    plt.xlabel("Height [m]"); plt.ylabel("TI [-]")
    plt.title("Mast TI Scatter (All 10-min Windows)"); plt.grid(True)

    plt.figure(figsize=(7, 5))
    sns.boxplot(x="height", y="ti", data=mast_df)
    plt.xlabel("Height [m]"); plt.ylabel("TI [-]")
    plt.title("TI Distribution per Height (Mast)"); plt.grid(True)

    print("plot done")


def plot_result_plotly(ti_profile_df: pd.DataFrame, ti_tidy_df: pd.DataFrame):
    """

    Parameters
    ----------
    ti_profile_df : DataFrame
        Median TI per height (columns: 'height', 'ti')
    ti_tidy_df : DataFrame
        All 10-min TI points (columns include 'height', 'ti')

    Returns
    -------
    figs : dict
        {
          'profile': <Figure>,
          'scatter': <Figure>,
          'box': <Figure>,
        }
    """

    # 1) Median TI profile vs height
    fig_profile = go.Figure()
    fig_profile.add_trace(
        go.Scatter(
            x=ti_profile_df["height"],
            y=ti_profile_df["ti"],
            mode="lines+markers",
            name="Median TI",
        )
    )
    fig_profile.update_layout(
        title="Median TI Profile vs Height",
        xaxis_title="Height [m]",
        yaxis_title="TI [-]",
        hovermode="x unified",
    )

    # 2) Scatter of all 10-min TI values
    fig_scatter = px.scatter(
        ti_tidy_df,
        x="height",
        y="ti",
        title="TI Scatter (All 10-min Windows)",
        labels={"height": "Height [m]", "ti": "TI [-]"},
    )
    fig_scatter.update_traces(mode="markers")

    # 3) Boxplot of TI per height
    fig_box = px.box(
        ti_tidy_df,
        x="height",
        y="ti",
        title="TI Distribution per Height",
        labels={"height": "Height [m]", "ti": "TI [-]"},
    )

    return {
        "profile": fig_profile,
        "scatter": fig_scatter,
        "box": fig_box,
    }
