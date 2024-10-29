# -*-coding:utf-8 -*-
"""
@File    :   data_processing.py
@Time    :   2024/10/25 13:48:11
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Data processing functions
"""
import pandas as pd
import os
import re

import matplotlib.pyplot as plt

from core.optimization.interpolation import spline_interpolation
from utils.paths import data_dir

if __name__ == "__main__":
    DATA_DIR = data_dir("MeteoSwiss_NCBN/Raw")
    data_path = DATA_DIR / "Stations/ncbn-daily_LUG.csv"
    data = pd.read_csv(data_path, sep=";", encoding="ANSI", na_values=["-", "-"])
    metrics = data.columns[2:]

    METRICS = pd.read_csv(DATA_DIR / "ncbn_codes.csv", index_col="nbcn_code")

    data.rename(
        columns={"date": "date-string", "station/location": "station"}, inplace=True
    )
    data["date"] = pd.to_datetime(data["date-string"], format="%Y%m%d")

    print(data.head())
    fig, ax = plt.subplots(len(metrics), 1, figsize=(10, 2 * len(metrics)), sharex=True)
    for i, metric in enumerate(metrics):
        t = data["date"].values
        year = data["date"].dt.year.values
        doy = data["date"].dt.dayofyear.values
        yearf = year + doy / 366
        y = data[metric].values
        where = ~pd.isna(y)
        f = spline_interpolation(yearf[where], y[where], step=20)
        t_where = t[where]
        yearf_where = yearf[where]
        y_int = f(yearf[where])
        y_int_en = (y_int - y_int.mean()) * 5 + y_int.mean()

        ax_twin = ax[i]  # .twinx()

        ax[i].plot(yearf, y)
        ax[i].set_ylabel(METRICS.loc[metric, "name"])
        ax_twin.plot(yearf_where, y_int, color="red")
        ax_twin.plot(yearf_where, y_int_en, color="darkred", ls=":")
    # ax[-1].set_xlim(2000, 2001)
    plt.show()

    # Preparing data
    STATIONS_DIR = data_dir("MeteoSwiss_NCBN/Raw/Stations")
    OUTDIR = data_dir("MeteoSwiss_NCBN/Preprocessed")

    data_stations = {}
    for station_file in os.listdir(STATIONS_DIR):
        m = re.match(r"ncbn-daily_(\w+).csv", station_file)
        if not m:
            continue
        station = m.group(1)
        data = pd.read_csv(
            os.path.join(STATIONS_DIR, station_file),
            sep=";",
            encoding="ANSI",
            na_values=["-"],
        )
        data.rename(
            columns={"date": "date-string", "station/location": "station"}, inplace=True
        )
        data["date"] = pd.to_datetime(data["date-string"], format="%Y%m%d")
        data.set_index("date-string", inplace=True)
        data_stations[station] = data

    # Aggregating data by metric
    data_metrics = {}
    for metric in metrics:
        data_metric = pd.concat(
            [data_stations[station][metric] for station in data_stations],
            axis=1,
            keys=[station for station in data_stations],
        )
        data_metric.sort_index(axis=0, inplace=True)
        data_metric.reset_index(inplace=True)
        data_metric["date"] = pd.to_datetime(
            data_metric["date-string"], format="%Y%m%d"
        )
        data_metric["yearf"] = (
            data_metric["date"].dt.year + data_metric["date"].dt.dayofyear / 366
        )
        data_metric.drop(columns="date-string", inplace=True)
        data_metrics[metric] = data_metric

    # Saving data
    os.makedirs(OUTDIR, exist_ok=True)

    for metric, data_metric in data_metrics.items():
        label = METRICS.loc[metric, "label"]
        data_metric.to_parquet(os.path.join(OUTDIR, f"{label}.parquet"))
