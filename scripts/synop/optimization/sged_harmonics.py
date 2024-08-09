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
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.signal import find_peaks
from tqdm import tqdm
from time import time as timer

from core.distributions.sged import (
    sged,
    sged_cdf,
    maximize_llhood_sged,
    maximize_llhood_sged_harmonics,
)
from core.distributions.excess_likelihood import (
    pce,
    pcei,
    correlation_to_alpha,
    excess_likelihood_inertial,
)
from core.optimization.harmonics import (
    harmonics_parameter_valuation,
    extract_harmonics,
    reconstruct_harmonics,
)
from core.optimization.interpolation import spline_interpolation

from plots.annual import month_xaxis, MONTHS_CENTER, MONTHS_LABELS_3, MONTHS_STARTS

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
    OUTPUT_DIR = output(f"Meteo-France_SYNOP/SGED harmonics/{METRIC}/{STATION}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ================================================================================================
    # Data processing
    # ================================================================================================
    # Seasonality and trend removal
    # ---------------------------------------------

    # Extraction of the temperature profile
    temperatures_stations = temperatures_stations.loc[
        (temperatures_stations["year"].between(FULL_YEAR_MIN, FULL_YEAR_MAX))
        & (temperatures_stations["day_of_year"] <= DAYS_IN_YEAR)
    ]

    temperatures = temperatures_stations[STATION].values

    # Time vector (in years)
    years = temperatures_stations["year"].values
    days = temperatures_stations["day_of_year"].values
    time = years + days / DAYS_IN_YEAR

    # Harmonic extraction for seasonality removal
    harmonics = extract_harmonics(
        temperatures, n_harmonics=N_HARMONICS, period=DAYS_IN_YEAR
    )
    seasonality = reconstruct_harmonics(harmonics, t=np.arange(N) / DAYS_IN_YEAR)

    deseasoned_temperatures = temperatures - seasonality

    # Trend removal
    f = spline_interpolation(time, deseasoned_temperatures, step=5)

    trend = f(time)
    detrended_temperatures = deseasoned_temperatures - trend

    # Plot the temperature profile
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"Temperature profile of station {STATION}")

    ax[0].plot(time, temperatures)

    ax[0].grid(which="major", axis="both", linewidth=0.5)
    ax[0].grid(which="minor", axis="both", linestyle="dotted", linewidth=0.5)
    ax[0].set_ylabel("Mean temperature (absolute)")

    ax[1].plot(time, deseasoned_temperatures)
    ax[1].plot(time, trend, c="r")
    ax[1].xaxis.set_major_locator(MultipleLocator(5))
    ax[1].xaxis.set_minor_locator(MultipleLocator(1))
    ax[1].yaxis.set_major_locator(MultipleLocator(5))
    ax[1].yaxis.set_minor_locator(MultipleLocator(1))
    ax[1].set_ylabel("Mean temperature (residuals)")
    ax[1].grid(which="major", axis="both", linewidth=0.5)
    ax[1].grid(which="minor", axis="both", linestyle="dotted", linewidth=0.5)
    ax[1].set_xlim(FULL_YEAR_MIN, FULL_YEAR_MAX)
    plt.show()

    # QQ-plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    sm.qqplot(temperatures, line="s", ax=axes[0])
    sm.qqplot(deseasoned_temperatures, line="s", ax=axes[1])
    sm.qqplot(deseasoned_temperatures - f(time), line="s", ax=axes[2])
    axes[0].set_title("Raw mean temperatures")
    axes[1].set_title("Mean temperatures\nseasonality removed")
    axes[2].set_title("Mean temperatures,\nseasonality and long-term trend removed")
    plt.show()

    # ================================================================================================
    # SGED-fitting
    # ================================================================================================
    popt_ = maximize_llhood_sged(detrended_temperatures)
    popt = popt_["x"]
    t_min = np.min(detrended_temperatures)
    t_max = np.max(detrended_temperatures)
    temp = np.linspace(t_min, t_max, 100)
    binwidth = 0.5
    N = len(detrended_temperatures)

    pdf = sged(temp, *popt)
    ci_inf = pdf - 1.96 * np.sqrt(pdf * (1 - pdf) / (N * binwidth))
    ci_sup = pdf + 1.96 * np.sqrt(pdf * (1 - pdf) / (N * binwidth))

    fig, ax = plt.subplots()
    sns.histplot(detrended_temperatures, binwidth=binwidth, stat="density", kde=True)
    ax.plot(
        temp,
        pdf,
        c="r",
        label=f"SGED($\\mu={popt[0]:.1f}$, $\\sigma={popt[1]:.1f}$, $\\lambda={popt[2]:.3f}$, $p={popt[3]:.3f}$)",
    )
    ax.fill_between(temp, ci_inf, ci_sup, alpha=0.5, fc="r")
    ax.legend()
    ax.set_xlabel("Mean temperatures seasonality and long-term trend removed")
    ax.set_ylim(0, None)
    ax.grid(which="both", axis="both", linewidth=0.5)

    # ================================================================================================
    # SGED-fitting with harmonics
    # ================================================================================================
    # Fitting the SGED model with cyclic parameters
    # ---------------------------------------------
    # Fitting of the parameters
    popt_ = maximize_llhood_sged_harmonics(
        t=time, x=detrended_temperatures, n_harmonics=N_HARMONICS
    )
    popt = popt_["x"]
    N = len(detrended_temperatures)

    # Analysis of the fit in terms of cumulative distribution function
    local_popt = harmonics_parameter_valuation(popt, time, N_HARMONICS, 4)
    local_cdf = np.zeros(N)
    for i in tqdm(range(N), total=N, smoothing=0):
        local_cdf[i] = sged_cdf(detrended_temperatures[i], *local_popt[:, i])

    # Projection of the SGED-fitted temperatures on a normal distribution with equivalent quantiles
    normal_projection = stats.norm.ppf(local_cdf)

    # Analysis of the SGED cdf
    # ---------------------------------------------
    # Histogram of the theoretical quantiles of the temperature values with respect to the fitted SGED model
    fig, ax = plt.subplots()
    sns.histplot(local_cdf, stat="density", kde=False)
    ax.set_xlabel("cdf of the SGED-fitted temperatures")
    ax.set_ylabel("Density")
    fig.savefig(
        os.path.join(OUTPUT_DIR, "sged-harmonics-empirical-cdf-distribution.png")
    )

    # Visualization of the temporal dependence of the cdf
    fig, ax = plt.subplots()
    sns.histplot(
        x=np.arange(N) / DAYS_IN_YEAR,
        y=local_cdf,
        stat="density",
        vmin=0,
        vmax=2 / YEARS,
        cmap="RdYlGn",
        cbar=True,
        cbar_kws={"label": "Density (Average value in yellow)"},
    )
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("cdf of the SGED-fitted temperatures")
    fig.savefig(os.path.join(OUTPUT_DIR, "sged-harmonics-cdf-vs-time.png"))

    # QQ-plot
    # ---------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    sm.qqplot(temperatures, line="s", ax=axes[0])
    sm.qqplot(detrended_temperatures, line="s", ax=axes[1])
    sm.qqplot(normal_projection, line="s", ax=axes[2])

    axes[0].set_title("Raw daily mean temperatures")
    axes[1].set_title("Standardized and\ndetrended temperatures")
    axes[2].set_title("SGED-fitted temperatures")
    fig.savefig(os.path.join(OUTPUT_DIR, "qqplots-raw-to-sged-fitted.png"))

    # Parameters valuation visualization
    # ---------------------------------------------
    # Visualization of the parameters of the SGED model
    doy = np.linspace(0, 1, DAYS_IN_YEAR)
    popt_season = np.copy(popt)
    popt_season[0] += np.abs(harmonics)[0]
    popt_season[1 : len(harmonics) * 2 - 1 : 2] += np.real(harmonics)[1:]
    popt_season[2 : len(harmonics) * 2 : 2] += np.imag(harmonics)[1:]

    popt_doy = harmonics_parameter_valuation(
        popt_season, t=doy, n_harmonics=N_HARMONICS, n_params=4
    )

    fig, axes = plt.subplots(4, 1, figsize=(6, 10), sharex=True)
    fig.suptitle("Parameters of the SGED model")
    for i, (ax, parameter) in enumerate(
        zip(axes, ["$\mu$", "$\sigma$", "$\lambda$", "$p$"])
    ):
        ax.plot(np.arange(DAYS_IN_YEAR), popt_doy[i, :])
        ax.set_ylabel(parameter)
        month_xaxis(ax)

    ax.set_xlim(0, DAYS_IN_YEAR)
    fig.savefig(os.path.join(OUTPUT_DIR, "sged-parameters-wrt-doy.png"))
    plt.show()

    # SGED visualization month per month
    # ---------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 10))
    fig.suptitle("SGED-fitted temperatures per month")
    popt_months = harmonics_parameter_valuation(
        popt_season,
        t=np.array(MONTHS_CENTER) / DAYS_IN_YEAR,
        n_harmonics=N_HARMONICS,
        n_params=4,
    )

    season_colors = ["#70d6ff", "#90a955", "#ffd670", "#ff9770"]
    temp = np.linspace(-15, 45, 100)

    for month in range(12):
        month_center = MONTHS_CENTER[month]
        month_label = MONTHS_LABELS_3[month]

        popt_month = popt_months[:, month]
        pdf_month = sged(temp, *popt_month)
        pdf_month /= np.max(pdf_month)

        ax.fill_between(
            temp,
            month,
            month + pdf_month * 0.8,
            alpha=0.5,
            fc=season_colors[(month + 1) // 3 % 4],
        )

    ax.set_yticks(np.arange(12))
    ax.set_yticklabels(MONTHS_LABELS_3)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Month")
    ax.set_ylim(0, 12)
    ax.grid(which="both", axis="both", linewidth=0.5)
    fig.savefig(os.path.join(OUTPUT_DIR, "sged-distributions-wrt-month.png"))

    # Return period visualization
    # ---------------------------------------------
    # Visualization of the return period of the SGED-fitted temperatures
    return_period_days = 1 / (1 - local_cdf)
    return_period_years = return_period_days / DAYS_IN_YEAR

    n_extremes = 5
    most_extremes = np.argsort(return_period_years)[-n_extremes:]
    time_of_occurence = time[most_extremes]
    date_of_occurence = [
        pd.to_datetime(f"{years[i]:.0f}-{days[i]:.0f}", format="%Y-%j")
        for i in most_extremes
    ]
    temperatures_of_occurence = temperatures[most_extremes]

    # qq-plot of the pce independent
    q = np.linspace(0, 1, len(local_cdf), endpoint=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(q, np.sort(local_cdf), "o", markersize=2)
    ax.axline((0, 0), slope=1, c="r", ls="--")
    ax.set_xlabel("Quantile")
    ax.set_ylabel("Probability of consecutive excesses")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect(1)
    fig.savefig(os.path.join(OUTPUT_DIR, "pcei-n1-qqplot.png"))

    # Temporal evolution of the return period
    fig, axes = plt.subplots(2, sharex=True, figsize=(10, 8))
    axes[0].plot(time, return_period_years)
    axes[0].scatter(time_of_occurence, return_period_years[most_extremes], c="r")
    for i, txt in enumerate(date_of_occurence):
        axes[0].annotate(
            txt.strftime("%Y-%m-%d") + f"\n{temperatures_of_occurence[i]:.1f}°C",
            (time_of_occurence[i], return_period_years[most_extremes[i]]),
            xytext=(5, 5),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3"),
        )
    axes[0].set_ylim(0.1, None)
    axes[0].set_yscale("log")

    axes[1].plot(time, temperatures)
    axes[1].plot(time, seasonality + trend, c="r")

    axes[0].set_ylabel("Return period (years)")
    axes[1].set_ylabel("Temperature (°C)")
    axes[1].set_xlabel("Time (years)")

    axes[1].xaxis.set_major_locator(MultipleLocator(5))
    axes[1].xaxis.set_minor_locator(MultipleLocator(1))
    axes[0].grid(True, axis="both", which="major")
    axes[0].grid(True, axis="both", which="minor", alpha=0.3)
    axes[1].grid(True, axis="both", which="major")
    axes[1].grid(True, axis="x", which="minor", alpha=0.3)

    fig.savefig(os.path.join(OUTPUT_DIR, "return-period-vs-time.png"))
    plt.show()

    # Return period of consecutive days
    # ---------------------------------------------
    n_consecutives = [2, 8, 4]
    for n_consecutive in n_consecutives:
        local_cdf_ndim = np.zeros((N, n_consecutive))
        for i in range(n_consecutive):
            local_cdf_ndim[-i:, i] = 0.5
            local_cdf_ndim[: N - i, i] = local_cdf[i:]

        # Spearman correlation
        rho = np.corrcoef(local_cdf_ndim, rowvar=False)[0, 1]
        rho_spearman, _ = stats.spearmanr(local_cdf_ndim)
        alpha = correlation_to_alpha(rho)

        # excess_llhood_consecutive = excess_likelihood_inertial(
        #     local_cdf_ndim, alpha=alpha
        # )
        start = timer()
        excess_cdf_consecutive = pcei(
            local_cdf_ndim, alpha=alpha, factor_in=20, factor_out=1000
        )
        end = timer()
        print(f"pcei computation time: {end - start:.2f}s")

        return_period_consecutive_days = 1 / excess_cdf_consecutive
        return_period_consecutive_years = return_period_consecutive_days / DAYS_IN_YEAR

        consecutive_extremes = find_peaks(
            return_period_consecutive_years, distance=10, height=1.0
        )[0]
        consecutive_extremes_order = np.argsort(
            return_period_consecutive_years[consecutive_extremes]
        )
        most_consecutive_extremes = consecutive_extremes[
            consecutive_extremes_order[-n_extremes:]
        ]

        time_of_consecutive_occurence = time[most_consecutive_extremes]
        date_of_consecutive_occurence = [
            pd.to_datetime(f"{years[i]:.0f}-{days[i]:.0f}", format="%Y-%j")
            for i in most_consecutive_extremes
        ]
        temperatures_of_consecutive_occurence = temperatures[most_consecutive_extremes]

        # qq-plot of the pce inertial
        q = np.linspace(0, 1, len(excess_cdf_consecutive), endpoint=True)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(q, np.sort(excess_cdf_consecutive), "o", markersize=2)
        ax.axline((0, 0), slope=1, c="r", ls="--")
        ax.set_xlabel("Quantile")
        ax.set_ylabel("Probability of consecutive excesses")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect(1)
        fig.savefig(os.path.join(OUTPUT_DIR, f"pcei-n{n_consecutive}-qqplot.png"))
        plt.show()

        # qq-plot of the pce inertial
        q = (np.arange(len(excess_cdf_consecutive)) + 0.5) / len(excess_cdf_consecutive)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(
            stats.norm.ppf(q),
            np.clip(stats.norm.ppf(np.sort(excess_cdf_consecutive)), -5, 5),
            "o",
            markersize=2,
        )
        ax.axline((0, 0), slope=1, c="r", ls="--")
        ax.set_xlabel("Quantile of normal distribution")
        ax.set_ylabel("Normal equivalent of probability of consecutive excesses")
        ax.set_aspect(1)
        fig.savefig(
            os.path.join(OUTPUT_DIR, f"pcei-n{n_consecutive}-normal-qqplot.png")
        )
        plt.show()

        # Temporal evolution of the return period of consecutive days
        fig, axes = plt.subplots(2, sharex=True, figsize=(10, 8))
        axes[0].plot(time, return_period_consecutive_years)
        axes[0].scatter(
            time_of_consecutive_occurence,
            return_period_consecutive_years[most_consecutive_extremes],
            c="r",
        )
        for i, txt in enumerate(date_of_consecutive_occurence):
            axes[0].annotate(
                txt.strftime("%Y-%m-%d")
                + f"\n{temperatures_of_consecutive_occurence[i]:.1f}°C",
                (
                    time_of_consecutive_occurence[i],
                    return_period_consecutive_years[most_consecutive_extremes[i]],
                ),
                xytext=(5, 5),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3"),
            )
        axes[0].set_ylim(0.1, None)
        axes[0].set_yscale("log")

        axes[1].plot(time, temperatures)
        axes[1].plot(time, seasonality + trend, c="r")

        axes[0].set_ylabel("Return period (years)")
        axes[1].set_ylabel("Temperature (°C)")
        axes[1].set_xlabel("Time (years)")

        axes[1].xaxis.set_major_locator(MultipleLocator(5))
        axes[1].xaxis.set_minor_locator(MultipleLocator(1))
        axes[0].grid(True, axis="both", which="major")
        axes[0].grid(True, axis="both", which="minor", alpha=0.3)
        axes[1].grid(True, axis="both", which="major")
        axes[1].grid(True, axis="x", which="minor", alpha=0.3)
        fig.savefig(
            os.path.join(
                OUTPUT_DIR, f"return-period-consecutive-{n_consecutive}-vs-time.png"
            )
        )
        plt.show()

    # ===============================================================================================
    # Return period comparison
    # ================================================================================================
    # Return period of consecutive days vs single day
    u = np.zeros((1001, 2))
    u[:, 0] = np.linspace(1e-6, 1 - 1e-6, u.shape[0], endpoint=True)  # Single day
    pce_single = pcei(u[:, :1], alpha=alpha)
    pce_consecutive = pcei(u, alpha=alpha)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(return_period_years, return_period_consecutive_years, "o", markersize=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(1 / pce_single / DAYS_IN_YEAR, 1 / pce_consecutive / DAYS_IN_YEAR, "r--")
    # ax.axline((.01, .005), (1, .2), c="r", lw=1, ls="--")
    ax.set_xlabel("Return period of single days (years)")
    ax.set_ylabel(f"Return period of {n_consecutive}-consecutive days (years)")
    ax.grid(True, which="major", axis="both", linewidth=0.5)
    ax.grid(True, which="minor", axis="both", linewidth=0.5, alpha=0.3)
    fig.savefig(os.path.join(OUTPUT_DIR, "return-period-comparison.png"))
