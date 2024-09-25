# -*-coding:utf-8 -*-
"""
@File    :   precipitations_gpd.py
@Time    :   2024/08/21 14:49:23
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   This script explores the precipitations data using POT methods from the SYNOP dataset.
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
from core.distributions.gpd import GPD
from core.distributions.sgpd import SGPD

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
    doy_to_season,
)
from utils.paths import data_dir, output
from utils.strings import capitalize


def mean_excess(values, thresholds: np.ndarray, window_size: int = 3):
    n_th = len(thresholds)
    excess_expected = np.zeros(n_th)
    excess_std = np.zeros(n_th)
    counts = np.zeros(n_th)

    for i, u in enumerate(thresholds):
        peaks = find_peaks(values, height=u, distance=window_size)[0]
        excesses = values[peaks]

        n = len(excesses)
        excess_expected[i] = np.mean(excesses) - u
        excess_std[i] = np.std(excesses) / np.sqrt(n)
        counts[i] = n

    return excess_expected, excess_std, counts


def gpd_fits(values, thresholds: np.ndarray, window_size: int = 3):
    n_th = len(thresholds)

    scales = np.zeros(n_th)
    locs = np.zeros(n_th)
    ksis = np.zeros(n_th)
    counts = np.zeros(n_th)

    for i, u in enumerate(thresholds):
        peaks = find_peaks(values, height=u, distance=window_size)[0]
        excesses = values[peaks] - u

        n = len(excesses)
        gpd = GPD()
        gpd.fit(excesses)

        scales[i] = gpd.sigma
        locs[i] = gpd.mu
        ksis[i] = gpd.ksi

        # params = stats.genpareto.fit(excesses, floc=0)

        # locs[i] = params[1]
        # scales[i] = params[2]
        # ksis[i] = params[0]
        counts[i] = n

    return locs, scales, ksis, counts


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

    # Seasonal split
    # --------------
    for season in range(-1, 4):
        precip_raw = precip_all[STATION].values
        precip_peaks = find_peaks(
            precip_raw, height=MIN_PRECIPITATION, distance=WINDOW_SIZE
        )[0]

        years = precip_all["year"].values[precip_peaks]
        doy = precip_all["day_of_year"].values[precip_peaks]
        precip = precip_raw[precip_peaks]
        log_precip = np.log(np.maximum(precip, 0.1))

        if season == -1:
            where_season = np.ones_like(doy, dtype=bool)
            season_name = "All year"
        else:
            where_season = doy_to_season(doy) == season
            season_name = SEASONS_FULL[season]

        years_season = years[where_season]
        doy_season = doy[where_season]
        precip_season = precip[where_season]
        log_precip_season = log_precip[where_season]
        days_in_year = DAYS_IN_YEAR * (len(precip_season) / len(precip_raw))

        # # ================================================================================================
        # # Mean excess plot
        # # ================================================================================================
        # thresh_min = 20
        # thresh_max = np.sort(precip_season)[-10]

        # thresholds = np.arange(thresh_min, thresh_max, 2)
        # excess_expected, excess_std, counts = mean_excess(
        #     precip_season, thresholds, window_size=WINDOW_SIZE
        # )
        # excess_std_ = np.where(excess_std > 0, excess_std, 5.0)
        # excess_pwl, summary = fit_piecewise_linear(
        #     thresholds, excess_expected, sigma=excess_std_, n_breakpoints=3
        # )
        # excess_bps = piecewise_linear_breakpoints(excess_pwl, thresh_min, thresh_max)

        # fig, axes = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        # ax = axes[0]
        # ax.plot(thresholds, excess_expected, c="k", label="Mean excess")
        # ax.plot(
        #     thresholds,
        #     excess_expected - 1.96 * excess_std,
        #     thresholds,
        #     excess_expected + 1.96 * excess_std,
        #     c="k",
        #     ls=":",
        # )
        # ax.set_xlabel("Threshold")
        # ax.set_ylabel("Mean excess")
        # ax.set_title(
        #     f"Mean excess plot for station {STATION_NAMES[STATION]} in {season_name}"
        # )
        # ax.set_ylim(0, None)
        # ax.plot(excess_bps[0], excess_bps[1], "r--")

        # ax_counts = axes[1]
        # ax_counts.plot(thresholds, counts, c="b", label="Number of excesses")
        # ax_counts.set_ylabel("Number of excesses")
        # ax_counts.set_ylim(1, None)
        # ax_counts.set_yscale("log")
        # plt.tight_layout()
        # plt.show()

        # # ================================================================================================
        # # GPD fits
        # # ================================================================================================
        # scales, locs, ksis, counts = gpd_fits(
        #     precip_season, thresholds, window_size=WINDOW_SIZE
        # )
        # modified_scales = scales - thresholds * ksis
        # uncensored_data = precip_season[
        #     (precip_season > thresholds[0]) & (precip_season < thresholds[-1])
        # ]

        # fig, axes = plt.subplots(
        #     3, figsize=(6, 6), sharex=True, gridspec_kw={"height_ratios": [2, 2, 1]}
        # )
        # for ax, param, param_std, name in zip(
        #     axes[:2],
        #     [modified_scales, ksis],
        #     [0, 0],
        #     ["Modified scales", "Shape"],
        # ):
        #     ax.plot(thresholds, param, c="k", label=name)
        #     ax.plot(thresholds, param - 1.96 * param_std, c="k", ls=":")
        #     ax.plot(thresholds, param + 1.96 * param_std, c="k", ls=":")
        #     ax.set_ylabel(name)

        # axes[-1].plot(thresholds, counts, c="b", label="Number of excesses")
        # axes[-1].scatter(uncensored_data, [1] * len(uncensored_data), marker="|", c="k")
        # axes[-1].set_ylabel("Number of excesses")
        # axes[-1].set_ylim(1, None)
        # axes[-1].set_yscale("log")
        # plt.tight_layout()
        # plt.show()

        # ================================================================================================
        # SGPD fits
        # ================================================================================================
        sgpd = SGPD()
        th = 0.3
        data = precip_season[precip_season > th]
        sgpd.fit(data)
        print(sgpd)
        print(SGPD._neg_llhood(sgpd.params, data))

        xx = np.linspace(data.min() - 10, data.max() + 10, 1000)
        fig, axes = plt.subplots(2, sharex=True)
        axes[0].plot(xx, sgpd.pdf(xx), c="k")
        axes[0].hist(data, bins=50, density=True, fc="C0")
        axes[0].set_yscale("log")
        axes[0].set_ylabel("Density")

        axes[1].plot(xx, sgpd.cdf(xx), c="k")
        axes[1].hist(data, bins=50, density=True, cumulative=True, fc="C0")
        axes[1].set_ylabel("Cumulative density")
        fig.suptitle(f"Station {STATION_NAMES[STATION]} in {season_name}")
        plt.show()

        alpha = 0.05
        n = len(data)
        k = np.arange(1, n + 1)
        p = k / (n + 1)
        pl = stats.beta.ppf(alpha / 2, k, n - k + 1)
        pu = stats.beta.ppf(1 - alpha / 2, k, n - k + 1)

        q = sgpd.ppf(p)
        ql = sgpd.ppf(pl)
        qu = sgpd.ppf(pu)

        data_sorted = np.sort(data)
        xmax = max(data_sorted[-1], q[-1]) * 1.2

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(q, data_sorted, "o", c="C0", markersize=2)
        ax.fill_between(q, ql, qu, color="C0", alpha=0.1)
        ax.set_xlabel("SGPD quantiles")
        ax.set_ylabel("Data")
        ax.axline([0, 0], [1, 1], c="k", ls="--")
        ax.set_xlim(0, data_sorted[-1] * 1.2)
        ax.set_ylim(0, data_sorted[-1] * 1.2)
        ax.set_aspect(aspect=1, anchor="E", adjustable="box")
        fig.suptitle(f"Station {STATION_NAMES[STATION]} in {season_name}")
        plt.show()

        # Return levels
        return_periods_years = np.logspace(-2, 3, 101, endpoint=True)
        return_periods_points = return_periods_years * days_in_year
        p = 1 - 1 / (return_periods_points + 1)
        pl = stats.beta.ppf(alpha / 2, return_periods_points, 1)
        pu = stats.beta.ppf(1 - alpha / 2, return_periods_points, 1)

        q = sgpd.ppf(p)
        ql = sgpd.ppf(pl)
        qu = sgpd.ppf(pu)

        fig, ax = plt.subplots()
        ax.plot(return_periods_years, q, c="C0")
        ax.fill_between(return_periods_years, ql, qu, color="C0", alpha=0.1)
        # ax.axline([0, 0], [1, 1], c="k", ls="--")
        ax.set_xlabel("Return period (years)")
        ax.set_ylabel("Return level (mm)")
        ax.set_xscale("log")
        ax.set_ylim(0, None)
        ax.set_xlim(return_periods_years[0], return_periods_years[-1])
        ax.yaxis.set_minor_locator(plt.MultipleLocator(20))
        ax.yaxis.set_major_locator(plt.MultipleLocator(100))
        ax.grid(True, alpha=0.3, lw=0.7)
        ax.grid(True, which="minor", alpha=0.2, lw=0.5)

        fig.suptitle(f"Station {STATION_NAMES[STATION]} in {season_name}")
        plt.show()

        # Empirical return periods
        return_periods_empirical = [0.2, 0.5, 1, 2, 5]
        return_levels_empirical = []
        return_levels_empirical_boxes = []
        for rp in return_periods_empirical:
            n_points_per_window = int(days_in_year * rp)
            return_levels_empirical.append(
                [
                    max(data[i : i + n_points_per_window])
                    for i in range(0, n - n_points_per_window, n_points_per_window)
                ]
            )
        return_levels_empirical_boxes = np.array(
            [
                np.quantile(rl, [0.5, alpha / 2, 1 - alpha / 2])
                for rl in return_levels_empirical
            ]
        )

        fig, ax = plt.subplots()
        ax.plot(return_periods_years, q, c="C0", label="Expected return levels")
        ax.fill_between(
            return_periods_years,
            ql,
            qu,
            color="C0",
            alpha=0.1,
            label=f"{1 - alpha:.0%} CI".replace("%", "\\%"),
        )
        # ax.boxplot(
        #     return_levels_empirical,
        #     positions=return_periods_empirical,
        #     widths=0.2 * np.array(return_periods_empirical),
        #     patch_artist=True,
        #     boxprops=dict(facecolor="none"),
        #     flierprops=dict(marker="o", markersize=2, color="k"),
        #     label="Empirical return levels",
        # )
        ax.errorbar(
            return_periods_empirical,
            return_levels_empirical_boxes[:, 0],
            yerr=[
                return_levels_empirical_boxes[:, 0]
                - return_levels_empirical_boxes[:, 1],
                return_levels_empirical_boxes[:, 2]
                - return_levels_empirical_boxes[:, 0],
            ],
            fmt="x",
            c="k",
            label="Empirical return levels",
        )
        ax.set_xlabel("Return period (years)")
        ax.set_ylabel("Return level (mm)")
        ax.set_xscale("log")
        ax.set_ylim(0, None)
        ax.set_xlim(return_periods_years[0], return_periods_years[-1])
        ax.yaxis.set_minor_locator(plt.MultipleLocator(20))
        ax.yaxis.set_major_locator(plt.MultipleLocator(100))
        ax.grid(True, alpha=0.3, lw=0.7)
        ax.grid(True, which="minor", alpha=0.2, lw=0.5)
        ax.legend(loc="upper left")

        fig.suptitle(f"Station {STATION_NAMES[STATION]} in {season_name}")
        plt.show()
