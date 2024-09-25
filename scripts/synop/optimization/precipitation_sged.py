# -*-coding:utf-8 -*-
"""
@File    :   precipitation_sged.py
@Time    :   2024/09/04 14:22:29
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   SGED fit of raw precipitation data
"""
# -*-coding:utf-8 -*-
"""
@File      :   time_fluctuation.py
@Time      :   2024/07/01 17:33:24
@Author    :   Urvan Christen
@Version   :   1.0
@Contact   :   urvan.christen@gmail.com
@Desc      :   Script for the analysis of the time fluctuation of a metric profile
    in the SYNOP dataset
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LogNorm
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.signal import find_peaks
from tqdm import tqdm

from core.distributions.sged import HarmonicSGED
from core.distributions.bernoulli import HarmonicBernoulli
from plots.annual import month_xaxis, MONTHS_CENTER, MONTHS_LABELS_3, MONTHS_STARTS

from utils.paths import data_dir, output


if __name__ == "__main__":
    plt.rcParams.update(
        {"text.usetex": True, "font.family": "serif"}
    )  # Use LaTeX rendering

    # ================================================================================================
    # Data loading
    # ================================================================================================
    METRIC = "preliq_SUM"
    precip_all = pd.read_parquet(
        data_dir(rf"Meteo-France_SYNOP/Preprocessed/{METRIC}.parquet")
    ).reset_index()
    stations = pd.read_csv(
        data_dir(r"Meteo-France_SYNOP/Raw/postesSynop.csv"), sep=";"
    ).set_index("ID")

    # ================================================================================================
    # Parameters
    # ================================================================================================
    DAYS_IN_YEAR = 365
    N_HARMONICS = 2

    # Finds the first and last full years in the dataset
    FULL_YEAR_MIN = precip_all.loc[precip_all["day_of_year"] == 1, "year"].min()
    FULL_YEAR_MAX = precip_all.loc[
        precip_all["day_of_year"] == DAYS_IN_YEAR, "year"
    ].max()
    YEARS = FULL_YEAR_MAX - FULL_YEAR_MIN + 1

    N = YEARS * DAYS_IN_YEAR
    MIN_PRECIPITATION = 1.0

    # Station to consider
    STATION = 7690
    STATION_NAME = stations.loc[STATION, "Nom"]

    # Output directory
    OUTPUT_DIR = output(f"Meteo-France_SYNOP/Precipitations/{METRIC}/{STATION}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ================================================================================================
    # Data processing
    # ================================================================================================
    # Seasonality and trend removal
    # ---------------------------------------------

    # Extraction of the temperature profile
    precip_all = precip_all.loc[
        (precip_all["year"].between(FULL_YEAR_MIN, FULL_YEAR_MAX))
        & (precip_all["day_of_year"] <= DAYS_IN_YEAR)
    ]

    precip = precip_all[STATION].values

    # Time vector (in years)
    years = precip_all["year"].values
    days = precip_all["day_of_year"].values
    time = years + days / DAYS_IN_YEAR

    # Remove no-precipitation days.
    # ---------------------------------------------
    where = precip > MIN_PRECIPITATION
    time_of_rain = time[where]
    years_of_rain = years[where]
    days_of_rain = days[where]
    proba_of_rain = np.mean(where)

    transforms = {
        "raw": precip,
        "log": np.log(precip),
    }
    for method, transformed_precip in transforms.items():
        precip_of_rain = transformed_precip[where]

        # Fitting the SGED model
        # ---------------------------------------------
        sged = HarmonicSGED(n_harmonics=N_HARMONICS)
        sged.fit(time_of_rain, precip_of_rain)

        prob_rain_model = HarmonicBernoulli(n_harmonics=N_HARMONICS)
        prob_rain_model.fit(time, where)

        # ================================================================================================
        # Results
        # ================================================================================================
        # Summary
        # ---------------------------------------------
        print(sged)

        # Plot the probability probability plot
        # ---------------------------------------------
        precip_cdf = sged.cdf(time_of_rain, precip_of_rain)
        precip_cdf_sorted = np.sort(precip_cdf)
        p_th = np.arange(1, len(precip_cdf) + 1) / (len(precip_cdf) + 1)

        # Plot the probability plot
        # ---------------------------------------------
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(p_th, precip_cdf_sorted, "o", markersize=2)
        ax.axline((0, 0), slope=1, ls="--", color="black")
        ax.set_xlabel(r"Theoretical probabilities")
        ax.set_ylabel(r"Empirical probabilities")
        ax.set_aspect(1, adjustable="datalim")
        ax.set_title(rf"Probability plot - Station {STATION_NAME} ({STATION})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.savefig(os.path.join(OUTPUT_DIR, f"{method}_SGED_probability_plot.png"))
        plt.show()

        # Plot the QQ plot
        # ---------------------------------------------
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(
            stats.norm.ppf(p_th), stats.norm.ppf(precip_cdf_sorted), "o", markersize=2
        )
        ax.axline((0, 0), slope=1, ls="--", color="black")
        ax.set_xlabel(r"Theoretical normal quantiles")
        ax.set_ylabel(r"Empirical normalized quantiles")
        ax.set_aspect(1, adjustable="datalim")
        ax.set_title(rf"QQ plot - Station {STATION_NAME} ({STATION})")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        fig.savefig(os.path.join(OUTPUT_DIR, f"{method}_SGED_qq_plot.png"))
        plt.show()

        # Return period plot
        # ---------------------------------------------
        return_periods = 1 / ((1 - precip_cdf) * proba_of_rain)

        # All time
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time_of_rain, return_periods / DAYS_IN_YEAR, "o", markersize=2)
        ax.set_xlabel(r"Time (years)")
        ax.set_ylabel(r"Return period (years)")
        ax.set_title(rf"Return period - Station {STATION_NAME} ({STATION})")
        ax.set_yscale("log")
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.grid(which="major", axis="both", alpha=0.5)
        ax.grid(which="minor", axis="both", alpha=0.2)
        fig.savefig(os.path.join(OUTPUT_DIR, f"{method}_SGED_return_period.png"))
        plt.show()

        # Monthly
        fig, axes = plt.subplots(2, figsize=(10, 6), height_ratios=[4, 1], sharex=True)
        ax = axes[0]
        ax.scatter(days_of_rain, return_periods / DAYS_IN_YEAR, s=precip[where])
        month_xaxis(ax)
        ax.set_ylabel(r"Return period (years)")
        ax.set_title(rf"Return period - Station {STATION_NAME} ({STATION})")
        ax.set_yscale("log")
        ax.grid(which="major", axis="y", alpha=0.5)
        ax.grid(which="minor", axis="y", alpha=0.2)

        axes[1].plot(
            np.arange(DAYS_IN_YEAR),
            prob_rain_model.pdf(t=np.arange(DAYS_IN_YEAR) / DAYS_IN_YEAR, x=1),
            color="black",
        )
        axes[1].set_xlim(0, DAYS_IN_YEAR)
        axes[1].set_ylim(0, None)
        axes[1].set_xlabel("Month")
        month_xaxis(axes[1])
        axes[1].grid(which="major", axis="y", alpha=0.5)
        axes[1].set_ylabel("Probability of rain")
        fig.savefig(
            os.path.join(OUTPUT_DIR, f"{method}_SGED_return_period_monthly.png")
        )
        plt.show()

        # ================================================================================================
        # Return period plot
        # ================================================================================================
        levels = np.logspace(0, 2.2, 100)
        doys = np.arange(1, DAYS_IN_YEAR, 5)
        return_periods = np.zeros(
            (
                len(levels),
                len(doys),
            )
        )
        for i, level in enumerate(tqdm(levels)):
            return_periods[i, :] = 1 / (
                1
                - sged.cdf(
                    doys / DAYS_IN_YEAR, level if method == "raw" else np.log(level)
                )
            )
            return_periods[i, :] /= prob_rain_model.pdf(t=doys / DAYS_IN_YEAR, x=1) * 30

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.contour(
            doys,
            levels,
            return_periods,
            levels=[1, 10, 100, 1000],
            cmap="Spectral",
            norm=LogNorm(),
            linewidths=2,
        )
        ax.set_xlabel("Day of year")
        ax.set_ylabel("Return level (mm)")
        ax.set_title(rf"Return period - Station {STATION_NAME} ({STATION})")
        ax.set_yscale("log")
        month_xaxis(ax)
        ax.grid(which="major", axis="both", alpha=0.5)

        cbar = fig.colorbar(
            im, ax=ax, orientation="vertical", label="Return period (months)"
        )

        fig.savefig(
            os.path.join(OUTPUT_DIR, f"{method}_SGED_return_period_contour.png")
        )
