# -*-coding:utf-8 -*-
"""
@File    :   precipitations.py
@Time    :   2024/07/31 14:49:23
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   This script explores the precipitations data from the SYNOP dataset.
"""

import pandas as pd
import numpy as np
import os
import re
from tqdm import tqdm
from scipy.signal import find_peaks
from itertools import pairwise

import seaborn as sns
from matplotlib import pyplot as plt

from core.mathematics.correlations import autocorrelation
from core.optimization.piecewise import (
    fit_piecewise_linear,
    fit_piecewise_linear_AIC,
    piecewise_linear_breakpoints,
)
from core.optimization.harmonics import reconstruct_harmonics
from core.distributions.sged import maximize_llhood_sged_harmonics, sged_cdf
from plots.mapplot import plot_map, set_lims
from plots.annual import month_xaxis, MONTHS_STARTS, MONTHS_LABELS_3, MONTHS_CENTER
from utils.paths import data_dir, output
from utils.strings import capitalize


if __name__ == "__main__":
    plt.rcParams.update(
        {
            "text.usetex": True,  # Use LaTeX rendering
            "text.latex.preamble": r"\usepackage{amsfonts}",
            "font.family": "serif",
            "font.size": "12",
        }
    )

    # ================================================================================================
    # Data loading
    # ================================================================================================
    METRIC = "preliq_SUM"
    precip_all = pd.read_parquet(
        data_dir(rf"Meteo-France_SYNOP/Preprocessed/{METRIC}.parquet")
    ).reset_index()
    precip_all["date"] = precip_all.apply(
        lambda x: pd.to_datetime(
            f"{x['year']:.0f}-{x['day_of_year']:.0f}", format="%Y-%j"
        ),
        axis=1,
    )

    RAW_DIR = data_dir("Meteo-France_SYNOP/Raw")
    OUT_DIR = output("Meteo-France_SYNOP/Description")
    os.makedirs(OUT_DIR, exist_ok=True)

    STATIONS = pd.read_csv(os.path.join(RAW_DIR, "postesSynop.csv"), sep=";")
    station_list = sorted(
        [station for station in STATIONS["ID"] if int(station) in precip_all.columns]
    )
    STATIONS = STATIONS.loc[STATIONS["ID"].isin(station_list)].sort_values("ID")

    # ================================================================================================
    # Parameters
    # ================================================================================================
    LAT_LIMS = (40, 55)
    LON_LIMS = (-7, 13)

    DAYS_IN_YEAR = 365
    Z_ALPHA = 1.96

    MIN_PRECIPITATION = 0.3
    WINDOW_SIZE = 3
    STATIONS_OF_INTEREST = [7072, 7110, 7690, 7149, 7535, 7190]
    STATION = STATIONS_OF_INTEREST[0]

    # ================================================================================================
    # Data exploration
    # ================================================================================================
    # Null precipitation probability
    # ------------------------------
    null_precip = (precip_all[station_list] <= MIN_PRECIPITATION).mean(axis=0)[
        station_list
    ]
    avg_precip = precip_all[station_list].mean(axis=0)[station_list]
    avg_effective_precip = avg_precip / (1 - null_precip)

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_map("europe", ax=ax, ec="k", lw=0.5, zorder=-1)
    set_lims(ax, *LAT_LIMS, *LON_LIMS)
    scat = ax.scatter(
        STATIONS["Longitude"],
        STATIONS["Latitude"],
        c=null_precip,
        s=30,
        cmap="RdYlBu_r",
        edgecolors="k",
        linewidths=0.5,
    )
    cbar = fig.colorbar(scat, ax=ax)
    cbar.set_label(
        f"Probability of precipitation $\\leq {MIN_PRECIPITATION}$mm", fontsize=12
    )
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    fig.savefig(os.path.join(OUT_DIR, "null_precipitation_probability.png"))

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_map("europe", ax=ax, ec="k", lw=0.5, zorder=-1)
    set_lims(ax, *LAT_LIMS, *LON_LIMS)
    scat = ax.scatter(
        STATIONS["Longitude"],
        STATIONS["Latitude"],
        c=avg_precip * DAYS_IN_YEAR,
        s=30,
        cmap="RdYlBu",
        edgecolors="k",
        linewidths=0.5,
    )
    cbar = fig.colorbar(scat, ax=ax)
    cbar.set_label(f"Yearly precipitation (mm)", fontsize=12)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    fig.savefig(os.path.join(OUT_DIR, "yearly_precipitation.png"))
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_map("europe", ax=ax, ec="k", lw=0.5, zorder=-1)
    set_lims(ax, *LAT_LIMS, *LON_LIMS)
    scat = ax.scatter(
        STATIONS["Longitude"],
        STATIONS["Latitude"],
        c=avg_effective_precip,
        s=30,
        cmap="RdYlBu",
        edgecolors="k",
        linewidths=0.5,
    )
    cbar = fig.colorbar(scat, ax=ax)
    cbar.set_label(f"Average precipitation on a rainy day (mm)", fontsize=12)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    fig.savefig(os.path.join(OUT_DIR, "average_precipitation.png"))
    plt.show()

    # Distribution functions
    precip_max = np.max(precip_all[station_list])
    fig, ax = plt.subplots(figsize=(6, 6))

    for station in STATIONS_OF_INTEREST:
        precip = precip_all[station]
        precip = precip[precip > 0]
        station_name = capitalize(
            STATIONS.loc[STATIONS["ID"] == station, "Nom"].values[0], sep="-"
        )
        sns.kdeplot(precip, ax=ax, label=f"{station_name} ({station})")
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_xlabel("Precipitation (mm)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Station {station}")
    ax.set_xlim(1, 50)
    ax.legend()
    ax.grid(ls=":", alpha=0.5)
    fig.savefig(os.path.join(OUT_DIR, f"precipitation_distribution_stations.png"))
    plt.show()

    def mean_excess(values, min_value: float = -np.inf, window_size: int = WINDOW_SIZE):
        peaks = find_peaks(values, height=min_value, distance=WINDOW_SIZE)[0]
        importance = np.argsort(values[peaks])
        data = values[peaks][importance]

        n = len(data)
        excess_expected = np.zeros(n)
        excess_std = np.zeros(n)

        for i, u in enumerate(data):
            if i == n - 1:
                excess_expected[i] = np.nan
                excess_std[i] = np.nan
            else:
                excess_expected[i] = np.mean(data[i + 1 :]) - u
                excess_std[i] = np.std(data[i + 1 :]) / np.sqrt(n - i - 1)

        return data, excess_expected, excess_std, peaks[importance]

    def truncate(
        values, *others, lims: tuple = (-np.inf, np.inf), skip: tuple = (0, 0)
    ):
        where = (values >= lims[0]) & (values < lims[1])
        n = np.sum(where)
        return (value[where][skip[0] : n - skip[1]] for value in (values, *others))

    def plot_ksi_breakpoints(
        thresholds, expected_excesses, ax, bp_lines=True, ksi_labels=True
    ):
        ax.plot(
            thresholds, expected_excesses, c="r", ls="--", label="Piecewise linear fit"
        )  # Add the slope

        if bp_lines:
            for i in range(len(thresholds) - 1):
                u_start = thresholds[i]
                u_end = thresholds[i + 1]
                exex_start = expected_excesses[i]
                exex_end = expected_excesses[i + 1]
                u_mean = (u_start + u_end) / 2
                slope = (exex_end - exex_start) / (u_end - u_start)
                ksi_1 = 1 - 1 / (1 + slope)

                ax.axvline(u_start, c="r", ls=":", lw=0.5)
                if ksi_labels:
                    ax.text(
                        u_mean,
                        0,
                        rf"$\xi={ksi_1:.2f}$",
                        fontsize=8,
                        c="r",
                        ha="center",
                        va="bottom",
                    )

    # ME-plot
    for station in STATIONS_OF_INTEREST:
        station_name = capitalize(
            STATIONS.loc[STATIONS["ID"] == station, "Nom"].values[0], sep="-"
        )

        data, excess_expected, excess_std, peaks = mean_excess(precip_all[station])
        dates = precip_all["date"].values[peaks]
        doy = precip_all["day_of_year"].values[peaks]

        print(f"\nStation: {station_name}")  # Print the station name
        for i in range(1, 6):
            print(f"{data[-i]:.1f} : {dates[-i]}")

        U_MIN = 5
        skip_last = 3

        # Do a polyfit on the expected excess to get the slope
        data_trunc, excess_expected_trunc, excess_std_trunc = truncate(
            data,
            excess_expected,
            excess_std,
            lims=(U_MIN, np.inf),
            skip=(0, skip_last),
        )

        popt = fit_piecewise_linear(data_trunc, excess_expected_trunc, 2)
        popt, summary = fit_piecewise_linear_AIC(
            data_trunc, excess_expected_trunc, sigma=excess_std_trunc
        )
        us, exexs = piecewise_linear_breakpoints(
            popt, data_trunc.min(), data_trunc.max()
        )

        # ksi_1 = 1 - 1 / (1 + slope)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(
            data, excess_expected, "o", markersize=2, label=r"$\mathbb{E}[X-u|X>u]$"
        )
        ax.fill_between(
            data,
            excess_expected - Z_ALPHA * excess_std,
            excess_expected + Z_ALPHA * excess_std,
            alpha=0.3,
            label=r"$\pm 1.96\hat{\sigma}$",
        )
        plot_ksi_breakpoints(us, exexs, ax)

        ax.axvline(u_end, c="r", ls=":", lw=0.5)
        ax.set_ylim(0, None)
        ax.set_xlim(0, data.max())
        ax.set_xlabel("Threshold $u$", fontsize=12)
        ax.set_ylabel("Expected excess", fontsize=12)
        ax.legend()
        fig.suptitle(f"Station {station_name} ({station})")
        plt.show()

        quantiles = [0.25, 0.5, 0.75, 1]
        month_quantiles = np.zeros((12, len(quantiles)))

        for i in range(12):
            where = (doy >= MONTHS_STARTS[i]) & (doy < MONTHS_STARTS[i + 1])
            month_quantiles[i, :] = np.quantile(data[where], quantiles)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(
            MONTHS_CENTER, month_quantiles[:, 1], c="k", label="Median", lw=2, zorder=2
        )
        ax.fill_between(
            MONTHS_CENTER,
            month_quantiles[:, 0],
            month_quantiles[:, 2],
            alpha=0.3,
            label="IQR",
            zorder=1,
        )
        ax.plot(
            MONTHS_CENTER,
            month_quantiles[:, 3],
            "+k",
            markersize=5,
            label="Max",
            zorder=2,
        )
        ax.set_yscale("log")
        ax.grid(axis="y", which="major", ls=":", alpha=0.5, c="gray", lw=1)
        ax.grid(axis="y", which="minor", ls=":", alpha=0.3)
        month_xaxis(ax)
        plt.show()

    max_threshold = 0
    max_expected_excess = 0

    ROWS, COLS = 3, 4
    fig, axes = plt.subplots(ROWS, COLS, figsize=(16, 9), sharex=True, sharey=True)

    for month_idx in range(12):
        month_start = MONTHS_STARTS[month_idx]
        month_end = MONTHS_STARTS[month_idx + 1]
        month_name = MONTHS_LABELS_3[month_idx]

        precip_month = precip_all.loc[
            (precip_all["day_of_year"] >= month_start)
            & (precip_all["day_of_year"] < month_end),
            station,
        ].values
        data_month, excess_month, excess_std_month, peaks_month = mean_excess(
            precip_month, min_value=MIN_PRECIPITATION
        )
        data_month_trunc, excess_month_trunc, excess_std_month_trunc = truncate(
            data_month,
            excess_month,
            excess_std_month,
            lims=(U_MIN, np.inf),
            skip=(0, 3),
        )

        popt_month, summary = fit_piecewise_linear(
            data_month_trunc, excess_month_trunc, sigma=excess_std_month_trunc
        )
        us_month, exexs_month = piecewise_linear_breakpoints(
            popt_month, data_month_trunc.min(), data_month_trunc.max()
        )

        ax = axes[month_idx // COLS, month_idx % COLS]
        ax.plot(
            data_month, excess_month, "o", markersize=2, label=r"$\mathbb{E}[X-u|X>u]$"
        )
        ax.fill_between(
            data_month,
            excess_month - Z_ALPHA * excess_std_month,
            excess_month + Z_ALPHA * excess_std_month,
            alpha=0.3,
            label=r"$\pm 1.96\hat{\sigma}$",
        )
        plot_ksi_breakpoints(us_month, exexs_month, ax)
        ax.set_title(month_name)
        max_threshold = max(max_threshold, data_month_trunc.max())
        max_expected_excess = max(max_expected_excess, excess_month_trunc.max())

    ax.set_xlim(0, max_threshold)
    ax.set_ylim(0, max_expected_excess * 1.1)

    # ===============================================================================================
    # SGED model log precipitations
    # ===============================================================================================
    n_harm = 2

    precip_station = precip_all.loc[:, station].values
    days_of_year = precip_all["day_of_year"].values
    data_station, excess_station, excess_std_station, peaks_station = mean_excess(
        precip_station, min_value=MIN_PRECIPITATION
    )
    log_precip_station = np.log(data_station)
    days_of_year_station = days_of_year[peaks_station]

    sged_fit = maximize_llhood_sged_harmonics(
        days_of_year_station / DAYS_IN_YEAR, x=log_precip_station, n_harmonics=n_harm
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.histplot(log_precip_station, ax=ax, kde=True, stat="density")
    plt.show()

    popt = sged_fit["x"]
    doy = np.arange(365)

    fig, axes = plt.subplots(4, figsize=(6, 6), sharex=True)
    for i, param in enumerate([r"\mu", r"\sigma", r"\lambda", "p"]):
        ax = axes[i]
        n_params_coeff = n_harm * 2 + 1
        params_harmonics = np.zeros(n_harm + 1, dtype="complex")
        params_harmonics[0] = popt[i * n_params_coeff]

        for j in range(n_harm):
            params_harmonics[j + 1] += popt[i * n_params_coeff + 2 * j + 1]
            params_harmonics[j + 1] += popt[i * n_params_coeff + 2 * j + 2] * 1j

        reconstructed_param = reconstruct_harmonics(
            params_harmonics, doy / DAYS_IN_YEAR
        )
        ax.plot(doy, reconstructed_param)
        ax.set_ylabel(f"${param}$")
        ax.grid(ls=":", alpha=0.5)
        month_xaxis(ax)
    axes[-1].set_xlabel("Day of year")
    axes[-1].set_xlim(0, 365)
    plt.show()
