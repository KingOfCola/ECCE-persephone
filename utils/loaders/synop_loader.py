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


def load_fit_synop(
    data_path: str,
    stations_path: str = None,
    model_type: str = None,
    **kwargs,
) -> TSData:
    """
    Load and fit a model to a SYNOP dataset
    """
    if "/" not in data_path:
        data_path = data_dir(rf"Meteo-France_SYNOP/Preprocessed/{data_path}.parquet")
    if stations_path is None:
        stations_path = data_dir(r"Meteo-France_SYNOP/Raw/postesSynop.csv")

    if model_type is None:
        model_type = (
            POSITIVE_LOG_SGED_MODEL if "preliq" in data_path else HARMONIC_SGED_MODEL
        )

    model = __make_model(model_type=model_type, **kwargs)
    data = load_synop(data_path, stations_path)
    return fit_model_synop(data, model)


def __make_model(model_type: str, **kwargs) -> HarmonicDistribution:
    if model_type == HARMONIC_SGED_MODEL:
        return __make_harmonic_sged(**kwargs)
    if model_type == POSITIVE_LOG_SGED_MODEL:
        return __make_positive_log_sged(**kwargs)
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

    data = load_fit_synop("preliq_SUM", model_type="positive_log_sged")

    print(data.labels)
    station = data.labels[2].value

    fig, ax = plt.subplots()
    ax.plot(data.time, data[station])

    y = data._raw_data[station]
    x = data[station]
    p = np.arange(len(x)) / len(x)
    fig, axes = plt.subplots(ncols=2)
    axes[0].plot(p, np.sort(x))
    axes[0].axline((0, 0), (1, 1), c="k", ls="--")
    axes[1].plot(x, y, "o")
