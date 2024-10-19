# -*-coding:utf-8 -*-
"""
@File    :   synop_loader.py
@Time    :   2024/10/18 11:11:32
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Synop data loader
"""


import pandas as pd
import os

from core.data.ts_data import TSData, HarmonicTSData
from core.data.labels import Label
from core.distributions.dist import HarmonicDistribution
from core.distributions.trend import TrendRemoval
from core.distributions.sged import HarmonicSGED
from core.distributions.pipe import Pipe
from utils.paths import data_dir


def year_doy_to_datetime(year, doy) -> pd.Timestamp:
    return pd.to_datetime(f"{year}-{doy}", format="%Y-%j")


def year_doy_to_datetime_series(series: pd.Series) -> pd.Series:
    return series.apply(lambda x: year_doy_to_datetime(x[0], x[1]))


def load_synop(data_path: str, stations_path: str) -> TSData:
    """
    Load a SYNOP dataset
    """
    stations = pd.read_csv(stations_path, sep=";").set_index("ID")
    station_labels = {idx: Label(idx, name) for idx, name in stations["Nom"].items()}
    data = pd.read_parquet(data_path)
    data.columns = [station_labels.get(col, col) for col in data.columns]
    data.index = year_doy_to_datetime_series(data.index.to_series())

    meta = {
        "data_path": data_path,
        "stations_path": stations_path,
        "metric": os.path.basename(data_path).split(".")[0],
        "stations": stations,
    }
    return TSData(data, meta=meta)


def fit_model_synop(data: TSData, model: HarmonicDistribution) -> TSData:
    """
    Fit a model to a SYNOP dataset
    """
    return HarmonicTSData(data, model=model, meta=data.meta)


def load_fit_synop(data_path: str, stations_path: str = None, **kwargs) -> TSData:
    """
    Load and fit a model to a SYNOP dataset
    """
    if "/" not in data_path:
        data_path = data_dir(rf"Meteo-France_SYNOP/Preprocessed/{data_path}.parquet")
    if stations_path is None:
        stations_path = data_dir(r"Meteo-France_SYNOP/Raw/postesSynop.csv")

    model = Pipe(
        TrendRemoval(step=kwargs.get("step", 5)),
        HarmonicSGED(n_harmonics=kwargs.get("n_harmonics", 2)),
    )
    data = load_synop(data_path, stations_path)
    return fit_model_synop(data, model)
