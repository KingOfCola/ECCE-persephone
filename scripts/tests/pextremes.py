# -*-coding:utf-8 -*-
"""
@File    :   pextremes.py
@Time    :   2024/07/29 10:58:31
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Test of the pyextremes library on P-SUM dataset from the SYNOP database
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import pyextremes as pe
from scipy import stats
from tqdm import tqdm

from core.mathematics.correlations import autocorrelation
from core.optimization.harmonics import extract_harmonics, reconstruct_harmonics
from plots.annual import month_xaxis

from scripts.tests.exgpd import LV

from utils.paths import data_dir, output


if __name__ == "__main__":
    plt.rcParams.update({"text.usetex": True})  # Use LaTeX rendering

    # ================================================================================================
    # Data loading
    # ================================================================================================
    METRIC = "t_MAX"
    temperatures_stations = pd.read_parquet(
        data_dir(rf"Meteo-France_SYNOP/Preprocessed/{METRIC}.parquet")
    ).reset_index()

    temperatures_stations["date"] = pd.to_datetime(
        temperatures_stations.apply(
            lambda row: f"{row['year']:.0f}-{row['day_of_year']:.0f}", axis=1
        ),
        format="%Y-%j",
    )

    # ================================================================================================
    # Parameters
    # ================================================================================================
    DAYS_IN_YEAR = 365
    N_HARMONICS = 2

    # Finds the first and last full years in the dataset
    FULL_YEAR_MIN = temperatures_stations.loc[
        temperatures_stations["day_of_year"] == 1, "year"
    ].min()
    FULL_YEAR_MAX = temperatures_stations.loc[
        temperatures_stations["day_of_year"] == DAYS_IN_YEAR, "year"
    ].max()
    YEARS = FULL_YEAR_MAX - FULL_YEAR_MIN + 1

    N = YEARS * DAYS_IN_YEAR

    # Station to consider
    STATION = 7481

    # Output directory
    OUTPUT_DIR = output(f"Meteo-France_SYNOP/Point extremes/{METRIC}/{STATION}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Extracts the Series of the station to consider
    data_station = pd.Series(
        data=temperatures_stations[STATION].values,
        index=temperatures_stations["date"].values,
    )

    # ================================================================================================
    # Data visualization
    # ================================================================================================
    THRESHOLD = np.quantile(data_station, 0.995)

    # BM plot
    model = pe.EVA(data_station)
    model.get_extremes("BM")
    for distribution in ["genextreme", "gumbel_r"]:
        model.fit_model(distribution=distribution)
        print(model.distribution)
        print(f"Log-likelihood: {model.loglikelihood}")
        print(f"AIC: {model.AIC}")
        print()

    fig, ax = model.plot_extremes()
    plt.show()

    model = pe.EVA(data_station)
    model.get_extremes("POT", threshold=THRESHOLD)
    for distribution in ["genpareto", "expon"]:
        model.fit_model(distribution=distribution)
        print(model.distribution)
        print(f"Log-likelihood: {model.loglikelihood}")
        print(f"AIC: {model.AIC}")
        print()

    fig, ax = model.plot_extremes()
    plt.show()

    fig, ax = model.plot_diagnostic(alpha=0.95)
    plt.show()

    fig, ax = pe.plot_parameter_stability(data_station.iloc[1600:], r="24H", alpha=0.95)
    plt.show()

    # Zero percipitation periods
    no_precipitation = []
    start_no_precip = None
    end_no_precip = None
    MIN_PRECIPITATION = 0.2
    for t, prec in data_station.items():
        if prec <= MIN_PRECIPITATION:
            if start_no_precip is None:
                start_no_precip = t
            end_no_precip = t
        else:
            if start_no_precip is not None:
                no_precipitation.append(
                    {
                        "start": start_no_precip,
                        "end": end_no_precip,
                        "duration": int(
                            (end_no_precip - start_no_precip) / pd.Timedelta(days=1)
                        ),
                    }
                )
            start_no_precip = None
            end_no_precip = None

    no_precipitation = pd.DataFrame(no_precipitation)

    fig, ax = plt.subplots()
    geom = stats.fit(stats.geom, no_precipitation["duration"].values)
    geom.plot(ax=ax)
    ax.set_title("Geometric distribution of zero precipitation periods")
    ax.set_xlabel("Duration (days)")
    ax.set_yscale("log")
    plt.show()

    # ================================================================================================
    # GEV graphical fit
    n = len(data_station)
    q = (np.arange(n) + 0.5) / n
    fig, ax = plt.subplots()
    ax.plot(1 - q, np.sort(data_station.values), "o", markersize=2)
    # ax.set_yscale("log")
    ax.set_xscale("log")
    plt.show()

    # ================================================================================================
    # Autocorrelation analysis
    # ================================================================================================
    # Autocorrelation
    auto = autocorrelation(data_station.values)
    fig, ax = plt.subplots()
    ax.plot(auto)
    ax.set_title("Autocorrelation of the precipitation series")
    ax.set_xlim(0, 30)
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(auto)
    ax.set_title("Autocorrelation of the precipitation series")
    ax.set_xlim(0, 2000)
    plt.show()

    doy = data_station.index.dayofyear
    year = data_station.index.year
    x = data_station.values

    where = (
        (year >= FULL_YEAR_MIN)
        & (year <= FULL_YEAR_MAX)
        & (doy >= 1)
        & (doy <= DAYS_IN_YEAR)
    )
    x = x[where]
    doy = doy[where]
    year = year[where]

    t = np.linspace(0, 1, DAYS_IN_YEAR + 1, endpoint=True)

    # Probability of precipitation
    harmonics_bin = extract_harmonics(x >= 0.3, n_harmonics=3, period=DAYS_IN_YEAR)
    x_bin_reconstructed = reconstruct_harmonics(harmonics_bin, t=t)

    fig, ax = plt.subplots()
    ax.plot(t * DAYS_IN_YEAR, x_bin_reconstructed, "r")
    ax.set_ylabel("Precipitation probability")
    month_xaxis(ax)
    ax.set_xlim(0, DAYS_IN_YEAR)
    ax.set_ylim(0, 1)
    plt.show()

    # Precipitation intensity
    harmonics = extract_harmonics(x, n_harmonics=3, period=DAYS_IN_YEAR)
    x_reconstructed = reconstruct_harmonics(harmonics, t=t)

    fig, ax = plt.subplots()
    ax.plot(data_station.index.dayofyear, data_station.values, "o", markersize=2)
    ax.plot(t * DAYS_IN_YEAR, x_reconstructed, "r")
    ax.plot(t * DAYS_IN_YEAR, x_reconstructed / x_bin_reconstructed, "r--")
    ax.set_yscale("log")
    ax.set_ylabel("Precipitation (mm)")
    month_xaxis(ax)
    plt.show()

    # ================================================================================================
    # Log-variance analysis
    # ================================================================================================
    # LV
    Ntot = temperatures_stations.shape[0]
    stations = [col for col in temperatures_stations.columns if isinstance(col, int)]
    lvs = np.zeros((Ntot, len(stations)))

    for i, station in tqdm(enumerate(stations), total=len(stations)):
        x = temperatures_stations.loc[:, station].values
        x = x + np.random.uniform(low=0, high=0.1, size=Ntot)
        lv = LV(x, p=None)
        lvs[:, i] = lv

    q = np.linspace(0, 1, Ntot)
    quantiles_lv = np.quantile(lvs, [0.1, 0.25, 0.5, 0.75, 0.9], axis=1)

    c = "C0"
    ksi_med = np.nanmedian(quantiles_lv[2, : Ntot // 20])

    fig, ax = plt.subplots()
    ax.plot(q, quantiles_lv[2, :])
    ax.fill_between(q, quantiles_lv[0, :], quantiles_lv[1, :], fc=c, alpha=0.2)
    ax.fill_between(q, quantiles_lv[1, :], quantiles_lv[3, :], fc=c, alpha=0.5)
    ax.fill_between(q, quantiles_lv[3, :], quantiles_lv[4, :], fc=c, alpha=0.2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel("Quantile of POT")
    ax.set_ylabel("$\\xi$")
    ax.set_xscale("log")
    ax.axhline(ksi_med, c="r", ls="--")
    ax.grid(ls=":", alpha=0.5)
    plt.show()

    fig.savefig(os.path.join(OUTPUT_DIR, "LV_analysis.png"))
