# -*-coding:utf-8 -*-
"""
@File    :   synop_raw.py
@Time    :   2024/07/09 10:20:47
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   This script explores the raw data from the SYNOP dataset.
"""

import pandas as pd
import numpy as np
import os
import re
from tqdm import tqdm

from matplotlib import pyplot as plt

from core.mathematics.correlations import autocorrelation
from plots.mapplot import plot_map, set_lims
from utils.paths import data_dir, output


if __name__ == "__main__":
    RAW_DIR = data_dir("Meteo-France_SYNOP/Raw")
    OUT_DIR = output("Meteo-France_SYNOP/Description")
    os.makedirs(OUT_DIR, exist_ok=True)

    DATA_PATTERN = "synop\.\d{6}\.csv\.gz"
    # List all the files in the directory
    filenames = [f for f in os.listdir(RAW_DIR) if re.match(DATA_PATTERN, f)]

    # Read the first file
    data_list = []

    for filename in tqdm(filenames, total=len(filenames)):
        filepath = os.path.join(RAW_DIR, filename)
        data_dir = pd.read_csv(filepath, sep=";", na_values="mq")
        data_list.append(data_dir)

    # Read the file
    data_dir = pd.concat(data_list).sort_values("date")

    LAT_LIMS = (40, 55)
    LON_LIMS = (-7, 13)

    stations = pd.read_csv(os.path.join(RAW_DIR, "postesSynop.csv"), sep=";")
    metropolitan = stations.loc[
        stations["Latitude"].between(*LAT_LIMS)
        & stations["Longitude"].between(*LON_LIMS)
    ]

    data_dir = data_dir.loc[data_dir["numer_sta"].isin(metropolitan["ID"])]

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_map("europe", ax=ax, ec="k", lw=0.5)
    set_lims(ax, *LAT_LIMS, *LON_LIMS)
    ax.scatter(metropolitan["Longitude"], metropolitan["Latitude"], c="red", s=10)
    for i, row in metropolitan.iterrows():
        ax.annotate(
            row["ID"],
            (row["Longitude"], row["Latitude"]),
            ha="center",
            fontsize=8,
            textcoords="offset points",
            xytext=(0, 5),
        )
    fig.savefig(os.path.join(OUT_DIR, "metropolitan_stations.png"))

    data_dir["datetime"] = pd.to_datetime(data_dir["date"], format="%Y%m%d%H%M%S")
    YEARS = int(
        (data_dir["datetime"].max() - data_dir["datetime"].min())
        / pd.Timedelta(days=1)
        // 365
    )

    # Trim the data so that only an integer number of years is there
    data_dir = data_dir.loc[
        data_dir["datetime"]
        < data_dir["datetime"].min() + YEARS * pd.Timedelta(days=365.25)
    ]

    MEASURES_PER_DAY = 8

    KELVIN_0C = 273.15

    data_sta = data_dir.loc[data_dir["numer_sta"] == 7005]
    data_sta_daily_max = data_sta.groupby(data_sta["datetime"].dt.date).max()

    temp_sta = data_sta_daily_max["t"].astype(float).ffill().values - KELVIN_0C
    datetime = data_sta_daily_max["datetime"].values

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(datetime, temp_sta)

    fft = np.fft.fft(temp_sta)
    fft[0] = 0
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(np.linspace(0, len(fft) / 9, len(fft)), np.abs(fft))
    ax.set_xlim(0, 700)

    a_season = fft[YEARS] / len(fft) * 2
    a_day = fft[len(fft) // MEASURES_PER_DAY] / len(fft) * 2

    temp_season = np.abs(a_season) * np.cos(
        2 * np.pi * YEARS * np.arange(len(temp_sta)) / len(temp_sta)
        + np.angle(a_season)
    )
    temp_day = np.abs(a_day) * np.cos(
        2
        * np.pi
        * len(temp_sta)
        // MEASURES_PER_DAY
        * np.arange(len(temp_sta))
        / len(temp_sta)
        + np.angle(a_day)
    )
    temp_deseason = temp_sta - temp_season

    fig, axes = plt.subplots(nrows=2, figsize=(12, 8), sharex=True)
    axes[0].plot(datetime, temp_sta)
    axes[0].plot(datetime, temp_season + np.mean(temp_sta))
    axes[0].plot(datetime, temp_day + temp_season + np.mean(temp_sta))
    axes[1].plot(datetime, temp_deseason)
    axes[1].plot(datetime, temp_deseason - temp_day)

    # Autocorrelation
    auto = autocorrelation(temp_deseason)

    fig, axes = plt.subplots(ncols=3, figsize=(12, 6), sharey=True)
    axes[0].plot(auto, "o")
    axes[1].plot(auto)
    axes[2].plot(auto)
    axes[0].set_xlim(0, 50)
    axes[1].set_xlim(50, 500)
    axes[2].set_xlim(500, len(temp_deseason))
    axes[1].set_xlabel("Lag")
    axes[0].set_ylabel("Autocorrelation")

    ## Heat waves
    data_dir["day"] = data_dir["datetime"].dt.date
    metropolitan_data = data_dir.loc[data_dir["numer_sta"].isin(metropolitan["ID"])]
    data_max = metropolitan_data.groupby(["day", "numer_sta"]).max()
    data_min = metropolitan_data.groupby(["day", "numer_sta"]).min()

    temp_mean = (
        data_max[["t"]].astype(float).ffill() + data_min[["t"]].astype(float).ffill()
    ) / 2 - KELVIN_0C
    temp_mean.reset_index(inplace=True)

    metropolitan_temp_mean = temp_mean.groupby("day").mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(metropolitan_temp_mean["day"], metropolitan_temp_mean["t"])

    HEATWAVE_DURATION_OUTER = 3
    HEATWAVE_INTENSITY_OUTER = 23.4
    HEATWAVE_DURATION_INNER = 1
    HEATWAVE_INTENSITY_INNER = 25.3

    heatwaves = []
    candidate_start_outer = None
    candidate_start_inner = None
    candidate_duration_inner = 0
    candidate_duration_outer = 0

    mean_temperatures = metropolitan_temp_mean["t"].values

    for i, t in enumerate(np.concat([mean_temperatures, [0]])):
        # Check if current timepoint exceeds the thresholds for a heatwave
        if t >= HEATWAVE_INTENSITY_OUTER:
            if candidate_start_outer is None:
                candidate_start_outer = i

            # finds the maximal duration inner window
            if t >= HEATWAVE_INTENSITY_INNER:
                if candidate_start_inner is None:
                    candidate_start_inner = i
            elif candidate_start_inner is not None:
                candidate_duration_inner = max(
                    i - candidate_start_inner, candidate_duration_inner
                )
                candidate_start_inner = None
        elif candidate_start_outer is not None:
            #  Check if the candidate heatwave is actually a heatwave
            candidate_duration_outer = i - candidate_start_outer
            if (
                candidate_duration_inner >= HEATWAVE_DURATION_INNER
                and candidate_duration_outer >= HEATWAVE_DURATION_OUTER
            ):
                heatwaves.append((candidate_start_outer, i - 1))
            candidate_start_outer = None
            candidate_start_inner = None
            candidate_duration_inner = 0
            candidate_duration_outer = 0

    for start, end in heatwaves:
        temps = metropolitan_temp_mean["t"].values[start : end + 1]
        start_day = metropolitan_temp_mean["day"].values[start]
        end_day = metropolitan_temp_mean["day"].values[end]
        temps_average = temps.mean()

        print(f"{start_day} -> {end_day} : Average temperature: {temps.mean():.2f}°C")

    doy = pd.to_datetime(metropolitan_temp_mean["day"]).dt.dayofyear
    years = pd.to_datetime(metropolitan_temp_mean["day"]).dt.year

    ## Plot of the heatwaves
    fig, ax = plt.subplots()
    for start, end in heatwaves:
        temps = metropolitan_temp_mean["t"].values[start : end + 1]
        start_day = doy.values[start]
        end_day = doy.values[end]
        year = years.values[start]

        ax.fill_between(
            np.arange(start_day, end_day + 1),
            year,
            year + (temps - HEATWAVE_INTENSITY_OUTER) / 5,
            fc="powderblue",
            ec="skyblue",
        )

    ax.set_xticks(
        [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365],
        minor=False,
        labels=[],
    )
    ax.set_xticks([15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345], minor=True)
    ax.set_xticklabels(
        [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ],
        minor=True,
    )

    ax.grid(True, axis="both", linestyle="--", color="gainsboro")
    ax.set_xlim(151, 243)
    fig.suptitle("Heatwaves in France")
