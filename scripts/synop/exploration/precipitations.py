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

import seaborn as sns
from matplotlib import pyplot as plt

from core.mathematics.correlations import autocorrelation
from core.optimization.piecewise import (
    fit_piecewise_linear,
    fit_piecewise_linear_AIC,
    piecewise_linear_breakpoints,
)
from plots.mapplot import plot_map, set_lims
from utils.paths import data_dir, output
from utils.strings import capitalize


if __name__ == "__main__":
    plt.rcParams.update({"text.usetex": True})  # Use LaTeX rendering
    plt.rcParams.update({"font.family": "serif"})
    plt.rcParams.update({"font.size": "12"})

    # ================================================================================================
    # Data loading
    # ================================================================================================
    METRIC = "preliq_SUM"
    precip_all = pd.read_parquet(
        data_dir(rf"Meteo-France_SYNOP/Preprocessed/{METRIC}.parquet")
    ).reset_index()

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

    # ME-plot
    for station in STATIONS_OF_INTEREST:
        station_name = capitalize(
            STATIONS.loc[STATIONS["ID"] == station, "Nom"].values[0], sep="-"
        )

        data = np.sort(precip_all[station].values)

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

        U_MIN = 5
        # Do a polyfit on the expected excess to get the slope
        excess_expected_trunc = excess_expected[data >= U_MIN][:-1]
        data_trunc = data[data >= U_MIN][:-1]

        popt = fit_piecewise_linear(data_trunc, excess_expected_trunc, 2)
        popt, aics = fit_piecewise_linear_AIC(data_trunc, excess_expected_trunc)
        us, exexs = piecewise_linear_breakpoints(
            popt, data_trunc.min(), data_trunc.max()
        )

        # ksi_1 = 1 - 1 / (1 + slope)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(data, excess_expected, "o", markersize=2)
        ax.fill_between(
            data,
            excess_expected - Z_ALPHA * excess_std,
            excess_expected + Z_ALPHA * excess_std,
            alpha=0.3,
        )
        ax.plot(us, exexs, c="r", ls="--")  # Add the slope
        for i in range(len(us) - 1):
            u_start = us[i]
            u_end = us[i + 1]
            exex_start = exexs[i]
            exex_end = exexs[i + 1]
            u_mean = (u_start + u_end) / 2
            slope = (exex_end - exex_start) / (u_end - u_start)
            ksi_1 = 1 - 1 / (1 + slope)

            ax.axvline(u_start, c="r", ls=":", lw=0.5)
            ax.text(
                u_mean,
                0,
                rf"$\xi={ksi_1:.2f}$",
                fontsize=8,
                c="r",
                ha="center",
                va="bottom",
            )

        ax.axvline(u_end, c="r", ls=":", lw=0.5)
        ax.set_ylim(0, None)
        ax.set_xlim(0, data_trunc.max())
        ax.set_xlabel("Threshold $u$", fontsize=12)
        ax.set_ylabel("Expected excess", fontsize=12)
        ax.legend()
        fig.suptitle(f"Station {station_name} ({station})")
        plt.show()
