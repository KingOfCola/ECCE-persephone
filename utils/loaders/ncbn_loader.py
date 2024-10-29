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
import pickle

from core.data.ts_data import TSData, HarmonicTSData
from core.data.labels import Label
from core.distributions.base.dist import HarmonicDistribution
from core.distributions.mixtures.threshold_mixtures import HarmonicThresholdModelMixture
from core.distributions.standard.constant_degen import HarmonicDegenerateConstant
from core.distributions.standard.log_transform import LogTransform
from core.distributions.trend import TrendRemoval
from core.distributions.sged import HarmonicSGED
from core.distributions.base.pipe import Pipe
from utils.paths import data_dir

HARMONIC_SGED_MODEL = "harmonic_sged"
POSITIVE_LOG_SGED_MODEL = "positive_log_sged"
NONE_MODEL = "none"


def year_doy_to_datetime(year, doy) -> pd.Timestamp:
    return pd.to_datetime(f"{year}-{doy}", format="%Y-%j")


def year_doy_to_datetime_series(series: pd.Series) -> pd.Series:
    return series.apply(lambda x: year_doy_to_datetime(x[0], x[1]))


def load_ncbn(data_path: str, stations_path: str) -> TSData:
    """
    Load a NCBN dataset
    """
    stations = pd.read_csv(stations_path).set_index("station")
    station_labels = {
        idx: Label(idx, name) for idx, name in stations["station_full"].items()
    }
    data = pd.read_parquet(data_path)
    data.columns = [station_labels.get(col, col) for col in data.columns]
    data.set_index("date", inplace=True)
    data.drop(columns=["yearf"], inplace=True)

    meta = {
        "data_path": data_path,
        "stations_path": stations_path,
        "metric": os.path.basename(data_path).split(".")[0],
        "stations": stations,
    }
    return TSData(data, meta=meta)


def fit_model(data: TSData, model: HarmonicDistribution) -> TSData:
    """
    Fit a model to a SYNOP dataset
    """
    return HarmonicTSData(data, model=model, meta=data.meta)


def load_fit_ncbn(
    data_path: str,
    stations_path: str = None,
    model_type: str = None,
    **kwargs,
) -> TSData:
    """
    Load and fit a model to a SYNOP dataset
    """
    # If data_path is not a path, assume it is a metric
    data_path = str(data_path)
    if "/" not in data_path or "\\" not in data_path:
        metric = data_path
        data_path = data_dir(rf"MeteoSwiss_NCBN/Preprocessed/{data_path}.parquet")
    else:
        metric = os.path.basename(data_path).split(".")[0]
    if stations_path is None:
        stations_path = data_dir(r"MeteoSwiss_NCBN/Preprocessed/ncbn_stations.csv")
    cache_path = data_dir(
        f"MeteoSwiss_NCBN/Preprocessed/TSHarmonics/{metric}_model.pkl"
    )

    # Check if the model is the default model and load it from cache if it exists
    default_model = model_type is None
    if default_model and cache_path.exists() and not kwargs.get("force", False):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    # If no model is specified, use the default model
    elif default_model:
        model_type = (
            POSITIVE_LOG_SGED_MODEL
            if "preliq" in str(data_path)
            else HARMONIC_SGED_MODEL
        )

    # Load and fit model
    model = __make_model(model_type=model_type, **kwargs)
    data = load_ncbn(data_path, stations_path)
    data_fit = fit_model(data, model)

    # Cache the model if it is the default model
    if default_model:
        os.makedirs(cache_path.parent, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(data_fit, f)

    return data_fit


def __make_model(model_type: str, **kwargs) -> HarmonicDistribution:
    if model_type == HARMONIC_SGED_MODEL:
        return __make_harmonic_sged(**kwargs)
    if model_type == POSITIVE_LOG_SGED_MODEL:
        return __make_positive_log_sged(**kwargs)
    if model_type == NONE_MODEL:
        return None
    if model_type is None:
        return None
    raise ValueError(f"Invalid model type {model_type}")


def __make_harmonic_sged(
    n_harmonics: int = 2, step: float = 5.0, **kwargs
) -> HarmonicDistribution:
    return Pipe(
        TrendRemoval(step=kwargs.get("step", 5)),
        HarmonicSGED(n_harmonics=kwargs.get("n_harmonics", 2)),
    )


def __make_positive_log_sged(
    n_harmonics: int = 2,
    step: float = None,
    period: float = 1.0,
    threshold: float = 0.3,
    **kwargs,
) -> HarmonicDistribution:
    pipe_models = []
    pipe_models.append(LogTransform())
    if step is not None:
        pipe_models.append(TrendRemoval(step=step))
    pipe_models.append(HarmonicSGED(n_harmonics=n_harmonics, period=period))
    pipe = Pipe(*pipe_models)

    constant = HarmonicDegenerateConstant(period=period, value=0.0)
    return HarmonicThresholdModelMixture(
        period=period,
        thresholds=[threshold],
        models=[constant, pipe],
        n_harmonics=n_harmonics,
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    data = load_fit_ncbn("t-AVG", model_type=None)

    print(data.labels)
    station = data.labels[2].value

    fig, ax = plt.subplots()
    ax.plot(data.time, data[station], "o", ms=2, alpha=0.3)

    fig, axes = plt.subplots(ncols=2)
    for station in data.labels:
        y = data._raw_data[station]
        x = data._data[station]
        where = np.isfinite(x)
        x = x[where]
        y = y[where]
        p = np.arange(len(x)) / len(x)
        axes[0].plot(p, np.sort(x))
        axes[0].axline((0, 0), (1, 1), c="k", ls="--")
    axes[1].plot(x, y, "o")
