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
from scipy import stats
from itertools import pairwise

import seaborn as sns
from matplotlib import pyplot as plt

from core.optimization.piecewise import (
    fit_piecewise_linear,
    fit_piecewise_linear_AIC,
    piecewise_linear_breakpoints,
)
from core.distributions.sged import HarmonicSGED
from plots.mapplot import plot_map, set_lims
from plots.annual import (
    month_xaxis,
    MONTHS_STARTS,
    MONTHS_LABELS_3,
    MONTHS_CENTER,
    MONTHS_COLORS,
    SEASONS_COLORS,
    SEASONS_CENTER,
    SEASONS_FULL,
    SEASONS_3,
    MONTHS_TO_SEASON,
    DOY_CMAP,
)
from utils.paths import data_dir, output
from utils.strings import capitalize


def mean_excess(values, min_value: float = -np.inf, window_size: int = 3):
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


def truncate(values, *others, lims: tuple = (-np.inf, np.inf), skip: tuple = (0, 0)):
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
    OUT_DIR = output("Meteo-France_SYNOP/Precipitations/log-SGED")
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
    STATION = STATIONS_OF_INTEREST[2]
    STATION_NAMES = {
        station: capitalize(
            STATIONS.loc[STATIONS["ID"] == station, "Nom"].values[0], sep="-"
        )
        for station in STATIONS_OF_INTEREST
    }

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
        sns.kdeplot(precip, ax=ax, label=f"{STATION_NAMES[station]} ({station})")
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
        out_dir_station = os.path.join(OUT_DIR, f"station_{station}")
        os.makedirs(out_dir_station, exist_ok=True)

        data, excess_expected, excess_std, peaks = mean_excess(
            precip_all[station].values
        )
        dates = precip_all["date"].values[peaks]
        doy = precip_all["day_of_year"].values[peaks]

        print(f"\nStation: {STATION_NAMES[station]}")  # Print the station name
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

        ax.set_ylim(0, None)
        ax.set_xlim(0, data.max())
        ax.set_xlabel("Threshold $u$", fontsize=12)
        ax.set_ylabel("Expected excess", fontsize=12)
        ax.legend()
        fig.suptitle(f"Station {STATION_NAMES[station]} ({station})")
        fig.savefig(os.path.join(out_dir_station, "mean-excess-plot.png"))
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
        fig.savefig(os.path.join(out_dir_station, "monthly_quantiles.png"))
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
    for station in STATIONS_OF_INTEREST:
        out_dir_station = os.path.join(OUT_DIR, f"station_{station}")
        os.makedirs(out_dir_station, exist_ok=True)

        n_harm = 2

        precip_station = precip_all.loc[:, station].values
        days_of_year = precip_all["day_of_year"].values
        dates = precip_all["date"].values
        years = precip_all["year"].values
        data_station, excess_station, excess_std_station, peaks_station = mean_excess(
            precip_station, min_value=MIN_PRECIPITATION
        )
        log_precip_station = np.log(data_station)
        days_of_year_station = days_of_year[peaks_station]
        years_station = years[peaks_station]
        dates_station = dates[peaks_station]

        sged_harmonic = HarmonicSGED(n_harmonics=n_harm, period=DAYS_IN_YEAR)
        sged_harmonic.fit(t=days_of_year_station, x=log_precip_station)

        fig, ax = plt.subplots(figsize=(6, 6))
        sns.histplot(log_precip_station, ax=ax, kde=True, stat="density")
        ax.set_xlabel("Log precipitation (mm)")
        ax.set_ylabel("Density")
        ax.set_title(f"Station {STATION_NAMES[station]} ({station})")
        ax.grid(ls=":", alpha=0.5)
        fig.savefig(os.path.join(out_dir_station, "log_precipitation_density.png"))
        plt.show()

        doy = np.arange(365)

        mu, sigma, lamb, p = sged_harmonic.param_valuation(doy)
        PARAMS_SETTINGS = [
            {"name": r"$\mu$", "lims": (0, 3), "param": mu},
            {"name": r"$\sigma$", "lims": (0, 3), "param": sigma},
            {"name": r"$\lambda$", "lims": (-1, 1), "param": lamb},
            {"name": r"$p$", "lims": (0, 6), "param": p},
        ]

        fig, axes = plt.subplots(4, figsize=(6, 6), sharex=True)
        for i in range(4):
            ax = axes[i]
            param_name = PARAMS_SETTINGS[i]["name"]
            param_value = PARAMS_SETTINGS[i]["param"]

            ax.plot(doy, param_value)
            ax.set_ylabel(f"${param_name}$")
            ax.grid(ls=":", alpha=0.5)
            ax.set_ylim(*PARAMS_SETTINGS[i]["lims"])

            month_xaxis(ax)
        axes[-1].set_xlabel("Day of year")
        axes[-1].set_xlim(0, 365)
        fig.suptitle(f"Station {STATION_NAMES[station]} ({station})")
        fig.savefig(os.path.join(out_dir_station, "harmonics_parameters.png"))
        plt.show()

        # ================================================================================================
        # CDF check
        # ================================================================================================
        sged_cdf_station = sged_harmonic.cdf(
            t=days_of_year_station,
            x=log_precip_station,
        )

        normal_quantiles = stats.norm.ppf(
            (np.arange(len(data_station)) + 0.5) / len(data_station)
        )
        sged_quantiles = stats.norm.ppf(sged_cdf_station)

        # QQ-plot
        # -------
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(normal_quantiles, np.sort(sged_quantiles), "o")
        ax.axline((0, 0), (1, 1), c="k", ls="--")
        ax.set_xlabel("Normal quantiles")
        ax.set_ylabel(r"$\Phi^{-1}(F_{SGED}(x))$")
        ax.set_title(f"Station {STATION_NAMES[station]} ({station})")

        fig.savefig(os.path.join(out_dir_station, "normal-qqplot.png"))
        plt.show()

        # Cumulative distribution function
        # --------------------------------
        fig, ax = plt.subplots(figsize=(6, 6))
        scatter = ax.scatter(
            data_station, sged_cdf_station, c=days_of_year_station, s=3, cmap=DOY_CMAP
        )
        cbar = fig.colorbar(scatter, ax=ax, ticks=MONTHS_CENTER)
        cbar.ax.set_yticklabels(MONTHS_LABELS_3)
        ax.set_xlabel("Precipitation (mm)")
        ax.set_ylabel("SGED CDF")
        ax.set_ylim(0, 1.1)
        ax.set_xlim(0, None)
        ax.set_title(f"Station {STATION_NAMES[station]} ({station})")
        fig.savefig(os.path.join(out_dir_station, "sged-cdf-by-month.png"))
        plt.show()

        # ================================================================================================
        # Return periods
        # ================================================================================================
        n_annotations = 5
        prob_of_rain = 1 - null_precip[station]
        return_period = 1 / (1 - sged_cdf_station) / (DAYS_IN_YEAR * prob_of_rain)
        order_return_period = np.argsort(return_period)

        fig, ax = plt.subplots(figsize=(8, 5))
        scat = ax.scatter(
            dates_station,
            return_period,
            c=days_of_year_station,
            s=(data_station / data_station[-1] * 10) ** 2,
            cmap=DOY_CMAP,
        )
        cbar = fig.colorbar(scat, ax=ax, ticks=MONTHS_CENTER)
        cbar.ax.set_yticklabels(MONTHS_LABELS_3)

        for i in order_return_period[-n_annotations:]:
            date = dates_station[i]
            rp = return_period[i]
            precip = data_station[i]

            ax.annotate(
                f"{np.datetime_as_string(date, unit='D')}: {precip:.1f}mm\nRP: {rp:.1f} years",
                (date, rp),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=8,
            )

        ax.set_xlabel("Precipitation (mm)")
        ax.set_ylabel("Return period (years)")
        ax.set_ylim(0, max(return_period) * 1.1)
        fig.suptitle(f"Station {STATION_NAMES[station]} ({station})")
        fig.savefig(os.path.join(out_dir_station, "return_periods.png"))
        plt.show()

        # ================================================================================================
        # Return periods by season
        # ================================================================================================

        y_max = 0

        fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
        for season, ax in enumerate(axes.flatten()):
            alpha = 0.05
            n_extremes_season = 10

            return_periods_years_season = np.logspace(
                -1, 3, 11
            )  # Return periods in number of seasons
            return_periods_days_season = return_periods_years_season * DAYS_IN_YEAR / 4
            ts_season = (
                return_periods_days_season * (len(data_station)) / len(precip_all)
            )
            us_low = stats.beta.ppf(alpha / 2, ts_season, 1)
            us_high = stats.beta.ppf(1 - alpha / 2, ts_season, 1)
            ts_low = 1 / (1 - us_low)
            ts_high = 1 / (1 - us_high)

            t_season = SEASONS_CENTER[season]

            log_precip_season = sged_harmonic.ppf(t=t_season, q=1 - 1 / ts_season)
            log_precip_low = sged_harmonic.ppf(t=t_season, q=1 - 1 / ts_low)
            log_precip_high = sged_harmonic.ppf(t=t_season, q=1 - 1 / ts_high)

            precip_season = np.exp(log_precip_season)
            precip_low = np.exp(log_precip_low)
            precip_high = np.exp(log_precip_high)

            if season == 0:
                where_season = (days_of_year_station < MONTHS_STARTS[2]) | (
                    days_of_year_station >= MONTHS_STARTS[11]
                )
            else:
                where_season = (
                    days_of_year_station >= MONTHS_STARTS[2 + (season - 1) * 3]
                ) & (days_of_year_station < MONTHS_STARTS[2 + season * 3])

            precip_season_rnd = np.random.permutation(data_station[where_season])
            n_extremes = int(np.floor(np.log2(len(precip_season_rnd))))

            erp_extremes_years = np.zeros(n_extremes)
            extremes_season = np.zeros(n_extremes)

            for k in range(n_extremes):
                extremes_season[k] = np.max(precip_season_rnd[2**k : 2 ** (k + 1)])
                erp_extremes_years[k] = (
                    (2**k) / len(precip_season_rnd) * len(precip_all) / DAYS_IN_YEAR
                )

            ax.fill_between(
                return_periods_years_season,
                precip_low,
                precip_high,
                alpha=0.3,
                fc="C0",
                label=f"{1 - alpha:.0%} prediction interval".replace("%", r"\%"),
            )
            ax.plot(
                return_periods_years_season,
                precip_season,
                c="C0",
                lw=2,
                label=f"Return level (middle of season)",
            )
            ax.plot(
                erp_extremes_years[-n_extremes_season:],
                extremes_season[-n_extremes_season:],
                "k+",
                label="Empirical return levels",
            )
            ax.set_xscale("log")
            ax.set_xticks([1, 10, 100, 1000])
            ax.set_title(SEASONS_3[season])
            ax.grid(ls=":", alpha=0.5)
            y_max = max(y_max, precip_high[-1])

        ax.set_ylim(0, y_max)
        axes[0, 0].legend(loc="upper left")

        fig.text(0.5, 0.04, "Return period (years)", ha="center")
        fig.text(0.04, 0.5, "Precipitation (mm)", va="center", rotation="vertical")
        fig.suptitle(f"Station {STATION_NAMES[station]} ({station})")
        fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
        fig.savefig(
            os.path.join(out_dir_station, f"return_periods_{SEASONS_3[season]}.png")
        )
        plt.show()

        # ================================================================================================
        # Return periods by month
        # ================================================================================================
        y_max = 0
        fig, axes = plt.subplots(4, 3, figsize=(10, 12), sharex=True, sharey=True)
        for month, ax in enumerate(axes.flatten(), start=-1):
            month = month % 12
            alpha = 0.05
            n_extremes_month = 10
            month_duration = MONTHS_STARTS[month + 1] - MONTHS_STARTS[month]

            return_periods_cycle_months = np.logspace(
                -1, 3, 11
            )  # Return periods in number of cycles (months)
            return_periods_days_month = return_periods_cycle_months * month_duration
            ts_month = return_periods_days_month * (len(data_station)) / len(precip_all)
            us_low = stats.beta.ppf(alpha / 2, ts_month, 1)
            us_high = stats.beta.ppf(1 - alpha / 2, ts_month, 1)
            ts_low = 1 / (1 - us_low)
            ts_high = 1 / (1 - us_high)

            t_month = MONTHS_CENTER[month]

            log_precip_month = sged_harmonic.ppf(t=t_month, q=1 - 1 / ts_month)
            log_precip_low = sged_harmonic.ppf(t=t_month, q=1 - 1 / ts_low)
            log_precip_high = sged_harmonic.ppf(t=t_month, q=1 - 1 / ts_high)

            precip_month = np.exp(log_precip_month)
            precip_low = np.exp(log_precip_low)
            precip_high = np.exp(log_precip_high)

            where_month = (days_of_year_station >= MONTHS_STARTS[month]) & (
                days_of_year_station < MONTHS_STARTS[month + 1]
            )

            precip_month_rnd = np.random.permutation(data_station[where_month])
            n_extremes = int(np.floor(np.log2(len(precip_month_rnd))))

            erp_extremes_years = np.zeros(n_extremes)
            extremes_month = np.zeros(n_extremes)

            for k in range(n_extremes):
                extremes_month[k] = np.max(precip_month_rnd[2**k : 2 ** (k + 1)])
                erp_extremes_years[k] = (
                    (2**k) / len(precip_month_rnd) * len(precip_all) / DAYS_IN_YEAR
                )

            ax.fill_between(
                return_periods_cycle_months,
                precip_low,
                precip_high,
                alpha=0.3,
                fc="C0",
                label=f"{1 - alpha:.0%} prediction interval".replace("%", r"\%"),
            )
            ax.plot(
                return_periods_cycle_months,
                precip_month,
                c="C0",
                label="Expected return level",
            )
            ax.plot(
                erp_extremes_years[-n_extremes_month:],
                extremes_month[-n_extremes_month:],
                "k+",
                label="Empirical quantile",
            )
            ax.set_title(MONTHS_LABELS_3[month])
            y_max = max(y_max, precip_high[-1])

        ax.set_xscale("log")
        ax.set_xticks([1, 10, 100, 1000])
        ax.set_ylim(0, y_max)

        for season in range(4):
            ax = axes[season, 0]
            ax.set_ylabel(f"{SEASONS_3[season]}")

        axes[0, 0].legend(loc="upper left")

        fig.suptitle(f"Station {STATION_NAMES[station]} ({station})")
        fig.text(0.5, 0.04, "Return period (years)", ha="center")
        fig.text(0.04, 0.5, "Precipitation (mm)", va="center", rotation="vertical")
        fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
        fig.savefig(os.path.join(out_dir_station, "return_periods_by_month.png"))

        plt.show()
