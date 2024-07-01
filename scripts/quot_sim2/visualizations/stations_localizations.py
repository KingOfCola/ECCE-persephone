# -*-coding:utf-8 -*-
"""
@File      :   stations_localizations.py
@Time      :   2024/06/28 13:07:28
@Author    :   Urvan Christen
@Version   :   1.0
@Contact   :   urvan.christen@gmail.com
@Desc      :   Scripts for visualizing the localizations of the stations from the
               QUOT_SIM2 study.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from utils import paths
from plots.mapplot import plot_map, set_lims
from plots.histograms import custom_histogram

if __name__ == "__main__":
    # Define the directories
    dirname = paths.data(r"Quot_SIM2")
    raw_dir = rf"{dirname}\Raw"
    preprocessed_dir = rf"{dirname}\Preprocessed"
    output_dir = paths.output(r"Quot_SIM2\Description")
    os.makedirs(output_dir, exist_ok=True)

    # Load the stations data
    stations = pd.read_parquet(rf"{preprocessed_dir}/stations.parquet")

    # Plot the stations on a map of Europe
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        stations["longitude"],
        stations["latitude"],
        c=stations["altitude"],
        cmap="RdYlGn_r",
        s=4,
        marker="s",
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Plot the map of Europe
    plot_map("europe", ax=ax, edgecolor="black", linewidth=0.5)
    set_lims(ax, 40, 55, -7, 13)
    plt.show()

    fig.savefig(rf"{output_dir}/stations_localizations.png")

    # Plot a histogram of the altitudes of the stations
    fig, ax = plt.subplots(figsize=(10, 6))
    custom_histogram(
        stations["altitude"],
        bins=[0, 100, 200, 500, 1000, 3500],
        labelpos="center",
        color="blue",
        alpha=0.7,
        ax=ax,
    )
    ax.set_xlabel("Altitude (m)")
    ax.set_ylabel("Number of stations")
    plt.show()

    fig.savefig(rf"{output_dir}/stations_altitude_histogram.png")
