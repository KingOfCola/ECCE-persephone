# -*-coding:utf-8 -*-
"""
@File    :   grabbing.py
@Time    :   2024/10/25 12:50:08
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Grabbing data from NCBN
"""

import pandas as pd
from urllib import request
import os
import matplotlib.pyplot as plt
import seaborn as sns

from plots.mapplot import plot_map, set_lims
from utils.paths import data_dir

if __name__ == "__main__":
    DATA_DIR = data_dir("MeteoSwiss_NCBN/Raw")
    os.makedirs(DATA_DIR, exist_ok=True)

    ncbn_list = pd.read_csv(
        os.path.join(DATA_DIR, "liste-download-nbcn-d.csv"), sep=";", encoding="ANSI"
    )
    ncbn_list.dropna(subset=["Station"], inplace=True)

    # Plot the stations on a map
    fig, ax = plt.subplots()
    plot_map("switzerland-country-zone", ax=ax, ec="black", lw=1.2, fc="none")
    plot_map("switzerland-canton-zone", ax=ax, ec="black", lw=0.5, fc="none", alpha=0.3)
    plot_map(
        "switzerland-district-zone", ax=ax, ec="black", lw=0.5, fc="none", alpha=0.1
    )
    sns.scatterplot(
        ncbn_list,
        x="Longitude",
        y="Latitude",
        hue="Climate region",
        linewidths=0.7,
        edgecolors="k",
    )
    set_lims(ax, "Switzerland")
    for idx, row in ncbn_list.iterrows():
        ax.annotate(
            row["station/location"],
            (row["Longitude"], row["Latitude"]),
            (0, 5),
            textcoords="offset points",
            fontsize=8,
            ha="center",
            va="bottom",
        )

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.show()

    # Download the data
    station_dir = os.path.join(DATA_DIR, "Stations")
    os.makedirs(station_dir, exist_ok=True)
    for idx, row in ncbn_list.iterrows():
        print(f"Downloading {row['Station']}...")
        url = row["URL Previous years (verified data)"]
        station = row["station/location"]

        station_file = os.path.join(station_dir, f"ncbn-daily_{station}.csv")
        request.urlretrieve(url, station_file)

    # Simplify station names
    ncbn_simple = ncbn_list[
        [
            "Station",
            "station/location",
            "Data since",
            "Station height m. a. sea level",
            "CoordinatesE",
            "CoordinatesN",
            "Latitude",
            "Longitude",
            "Climate region",
            "Canton",
        ]
    ].copy()
    ncbn_simple.rename(
        columns={
            "Station": "station_full",
            "station/location": "station",
            "Data since": "start",
            "Station height m. a. sea level": "elevation",
            "CoordinatesE": "easting",
            "CoordinatesN": "northing",
            "Latitude": "latitude",
            "Longitude": "longitude",
            "Climate region": "climate_region",
            "Canton": "canton",
        },
        inplace=True,
    )
    ncbn_simple.to_csv(os.path.join(DATA_DIR, "ncbn_stations.csv"), index=False)
