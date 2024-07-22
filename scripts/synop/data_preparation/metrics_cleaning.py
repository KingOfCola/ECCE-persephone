# -*-coding:utf-8 -*-
"""
@File    :   metrics_cleaning.py
@Time    :   2024/07/09 15:20:48
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   This script cleans the metrics from the SYNOP dataset.
"""

import pandas as pd
import numpy as np
import os
import re
from tqdm import tqdm

from utils.paths import data_dir


def fill_missing_days(data: pd.DataFrame, value: any = np.nan) -> pd.DataFrame:
    """
    Fill the missing days in the dataset.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to fill.
    value : any
        The value to fill the missing days with.

    Returns
    -------
    pd.DataFrame
        The dataset with the missing days filled.
    """
    index = data.index.to_frame()
    year = index["year"].astype(str)
    doy = index["day_of_year"].astype(str)
    datetime = pd.to_datetime(year + "-" + doy, format="%Y-%j")

    # Extract the unique days
    days = pd.date_range(
        start=datetime.min().floor("D"),
        end=datetime.max().ceil("D"),
        freq="D",
    )

    new_years = days.year
    new_days = days.dayofyear

    # Create the new index
    new_index = pd.MultiIndex.from_arrays(
        [new_years, new_days], names=["year", "day_of_year"]
    )

    # Reindex the data
    return data.reindex(new_index, fill_value=value)


if __name__ == "__main__":
    RAW_DIR = data_dir("Meteo-France_SYNOP/Raw")
    PROCESSED_DIR = data_dir("Meteo-France_SYNOP/Preprocessed")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Pattern for the data files
    DATA_PATTERN = "synop\.\d{6}\.csv\.gz"

    # Metropolitan France limits
    LAT_LIMS = (40, 55)
    LON_LIMS = (-7, 13)

    # Conversion from Kelvin to Celsius
    KELVIN_TO_CELSIUS = 273.15

    # Aggregation methods
    AGGREGATIONS = {
        "AVG": "mean",
        "MAX": "max",
        "MIN": "min",
        "SUM": "sum",
    }

    METRICS = {
        "preliq": {
            "column": "rr3",
            "methods": ["SUM", "MAX"],
            "fillna": 0,
        },
        "t": {
            "column": "t",
            "methods": ["MIN", "MAX", "AVG"],
            "fillna": "ffill",
        },
    }

    # List all the files in the directory
    filenames = [f for f in os.listdir(RAW_DIR) if re.match(DATA_PATTERN, f)]

    # Read the data from the monthly files
    data_list = []

    for filename in tqdm(filenames, total=len(filenames)):
        filepath = os.path.join(RAW_DIR, filename)
        data = pd.read_csv(filepath, sep=";", na_values="mq")
        data_list.append(data)

    # Concatenate the data
    data = pd.concat(data_list)

    # Extracts the stations in metropolitan France
    stations = pd.read_csv(os.path.join(RAW_DIR, "postesSynop.csv"), sep=";")
    metropolitan = stations.loc[
        stations["Latitude"].between(*LAT_LIMS)
        & stations["Longitude"].between(*LON_LIMS)
    ]

    # Corrects the temperature to get it in °C
    data["t"] = data["t"] - KELVIN_TO_CELSIUS

    # Filter the data
    data = data.loc[data["numer_sta"].isin(metropolitan["ID"])]
    data.sort_values(["date", "numer_sta"], inplace=True)
    data["datetime"] = pd.to_datetime(data["date"], format="%Y%m%d%H%M%S")
    data["year"] = data["datetime"].dt.year
    data["day_of_year"] = data["datetime"].dt.dayofyear

    # ===========================================================================
    # Extraction of the daily aggregates for each metric of interest
    # ===========================================================================

    for metric, settings in METRICS.items():
        column = settings["column"]
        methods = settings["methods"]
        fillna = settings["fillna"]

        # Fill the missing values
        if fillna == "ffill":
            data[column] = data[column].ffill()
        else:
            data.fillna({column: fillna}, inplace=True)

        # Extract the data
        for method in methods:
            metric_data = (
                data.groupby(["year", "day_of_year", "numer_sta"])[column]
                .agg(AGGREGATIONS[method])
                .unstack(level=-1)
            )
            metric_data.columns.name = "station_id"
            metric_data = fill_missing_days(metric_data, value=np.nan)

            # Fill the missing values
            if fillna == "ffill":
                metric_data = metric_data.ffill().bfill()
            else:
                metric_data.fillna(fillna, inplace=True)

            # Store the data
            metric_data.to_parquet(
                os.path.join(PROCESSED_DIR, f"{metric}_{method}.parquet")
            )
