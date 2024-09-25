# -*-coding:utf-8 -*-
"""
@File    :   precipitations_cv.py
@Time    :   2024/08/17 14:49:23
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   This script explores the precipitations data from the SYNOP dataset.
                It fits a SGED model on the log-transformed precipitations and computes the return periods.
                It also performs a cross-validation to assess the model's performance on unknown data.
"""

import pandas as pd
import numpy as np
import os
from scipy import stats
from scipy.signal import find_peaks

from matplotlib import pyplot as plt
import matplotlib

from core.distributions.sged import HarmonicSGED
from core.mathematics.correlations import autocorrelation
from plots.annual import month_xaxis
from utils.paths import data_dir, output
from utils.strings import capitalize

JET_CMAP = matplotlib.colormaps.get_cmap("jet")


def peaks_over_threshold(values, min_value: float = -np.inf, window_size: int = 3):
    peaks = find_peaks(values, height=min_value, distance=window_size)[0]
    importance = np.argsort(values[peaks])
    data = values[peaks][importance]

    return data, peaks[importance]


def return_period_confidence_interval_k(k, n, alpha):
    uk = stats.beta.ppf(alpha, k, n - k + 1)
    return 1 / (1 - uk)


def return_period_confidence_interval(n, alphas=0.05):
    ks = np.arange(1, n + 1)
    return np.array(
        [
            [return_period_confidence_interval_k(k, n, alpha=alpha) for alpha in alphas]
            for k in ks
        ]
    )


def return_period_plot(cdf, x=None, ax=None, alpha=0.05, scale: float = 1.0, **kwargs):
    if ax is None:
        ax = plt.gca()

    if x is None:
        x = 1 / (1 - (np.arange(len(cdf)) + 1) / (len(cdf) + 1)) / scale
        xlabel = "Empirical return period"
    else:
        xlabel = "Value"

    return_period = 1 / (1 - cdf) / scale
    ax.plot(
        x,
        return_period,
        marker="o",
        c=kwargs.get("c", "C0"),
        ls="none",
        markersize=2,
        label="Return period",
    )

    alphas = [alpha / 2, 0.5, 1 - alpha / 2]
    rp_ci = return_period_confidence_interval(len(x), alphas=alphas) / scale
    ax.plot(x, rp_ci[:, 1], color=kwargs.get("c", "r"), ls="--", label="Median")
    ax.fill_between(
        x,
        rp_ci[:, 0],
        rp_ci[:, 2],
        color=kwargs.get("c", "C0"),
        alpha=0.1,
        label=f"{1-alpha:.0%} CI",
    )
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Return period based on CDF")

    return ax


if __name__ == "__main__":
    plt.rcParams.update(
        {
            "text.usetex": True,  # Use LaTeX rendering
            "text.latex.preamble": r"\usepackage{amsfonts}",
            "font.family": "serif",
            "font.size": "12",
            "savefig.dpi": 300,
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
    YEAR_MIN = precip_all["year"].min()
    YEAR_MAX = precip_all["year"].max()
    Z_ALPHA = 1.96

    MIN_PRECIPITATION = 0.3
    MAX_AUTO = 0.2

    WINDOW_SIZE = 3
    STATIONS_OF_INTEREST = [7110, 7690, 7149, 7535, 7190]
    STATION = STATIONS_OF_INTEREST[3]
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

    # ===============================================================================================
    # SGED model log precipitations
    # ===============================================================================================
    # station = STATION
    for station in STATIONS_OF_INTEREST:
        out_dir_station = os.path.join(OUT_DIR, f"station_{station}")
        os.makedirs(out_dir_station, exist_ok=True)

        n_harm = 2

        # Extracts the data of interest for the station
        precip_station = precip_all.loc[:, station].values
        days_of_year = precip_all["day_of_year"].values
        dates = precip_all["date"].values
        years = precip_all["year"].values

        auto = autocorrelation(precip_station > MIN_PRECIPITATION)
        min_lag = np.where(auto < MAX_AUTO)[0][0]
        print(f"Station {STATION_NAMES[station]} ({station}): Min lag: {min_lag}")

        # Finds the peaks over the threshold for the precipitation
        data_station, peaks_station = peaks_over_threshold(
            precip_station, min_value=MIN_PRECIPITATION, window_size=min_lag
        )
        log_precip_station = np.log(data_station)
        days_of_year_station = days_of_year[peaks_station]
        dates_station = dates[peaks_station]
        years_station = years[peaks_station]

        # Fits the SGED model
        sged_harmonic = HarmonicSGED(n_harmonics=n_harm, period=DAYS_IN_YEAR)
        sged_harmonic.fit(t=days_of_year_station, x=log_precip_station)

        # Computes the return period
        sged_cdf_station = sged_harmonic.cdf(
            t=days_of_year_station, x=log_precip_station
        )
        return_period = (
            1 / (1 - sged_cdf_station) / (DAYS_IN_YEAR * (1 - null_precip[station]))
        )

        # Most extreme events
        # -------------------
        n_extreme = 10
        order_return_period = np.argsort(return_period)[::-1]
        t_extreme = np.array(
            [dates_station[i] for i in order_return_period[:n_extreme]]
        )
        x_extreme = np.array([data_station[i] for i in order_return_period[:n_extreme]])
        years_extreme = np.array(
            [years_station[i] for i in order_return_period[:n_extreme]]
        )
        extreme_names = [
            f"{np.datetime_as_string(t, unit='D')}\n{x:.1f} mm"
            for t, x in zip(t_extreme, x_extreme)
        ]

        # Cross-validation
        # ----------------
        doy = np.arange(365)

        # Plots the parameters for the full dataset
        mu, sigma, lamb, p = sged_harmonic.param_valuation(doy)
        PARAMS_SETTINGS = [
            {"name": r"$\mu$", "lims": (0, 3), "param": mu},
            {"name": r"$\sigma$", "lims": (0, 3), "param": sigma},
            {"name": r"$\lambda$", "lims": (-1, 1), "param": lamb},
            {"name": r"$p$", "lims": (0, 8), "param": p},
        ]

        fig, axes_all = plt.subplots(
            4, 2, figsize=(8, 6), sharex=True, width_ratios=[4, 0.4]
        )
        # Plot the parameters for the full dataset
        axes = axes_all[:, 0]
        for i in range(4):
            ax = axes[i]
            param_name = PARAMS_SETTINGS[i]["name"]
            param_value = PARAMS_SETTINGS[i]["param"]

            ax.plot(doy, param_value, c="k", lw=2)
            ax.set_ylabel(f"${param_name}$")
            ax.grid(ls=":", alpha=0.5)
            ax.set_ylim(*PARAMS_SETTINGS[i]["lims"])

            month_xaxis(ax)
        axes[-1].set_xlabel("Day of year")
        axes[-1].set_xlim(0, 365)

        # Add the colorbar
        gs = axes_all[0, 1].get_gridspec()
        for ax in axes_all[:, 1]:
            ax.remove()
        colorbar_ax = fig.add_subplot(gs[:, 1])
        norm = matplotlib.colors.Normalize(vmin=YEAR_MIN, vmax=YEAR_MAX)
        cbar = matplotlib.colorbar.ColorbarBase(
            colorbar_ax, cmap=JET_CMAP, norm=norm, orientation="vertical"
        )
        colorbar_ax.set_ylabel("Year")

        n_cv = YEAR_MAX - YEAR_MIN + 1
        return_periods_cv = np.zeros((n_cv, len(data_station)))
        sged_cdf_cv = np.zeros((n_cv, len(data_station)))

        for i in range(n_cv):
            idx = np.where(years_station != YEAR_MIN + i)[0]
            print(f"Year: {YEAR_MIN + i}: {len(idx)}/{len(data_station)}")
            sged_harmonic_cv = HarmonicSGED(n_harmonics=n_harm, period=DAYS_IN_YEAR)
            sged_harmonic_cv.fit(t=days_of_year_station[idx], x=log_precip_station[idx])
            sged_cdf_station = sged_harmonic_cv.cdf(
                t=days_of_year_station, x=log_precip_station
            )
            sged_cdf_cv[i] = sged_cdf_station
            return_periods_cv[i] = (
                1 / (1 - sged_cdf_station) / (DAYS_IN_YEAR * (1 - null_precip[station]))
            )
            mu_b, sigma_b, lamb_b, p_b = sged_harmonic_cv.param_valuation(doy)

            for j, param_value in enumerate([mu_b, sigma_b, lamb_b, p_b]):
                axes[j].plot(
                    doy,
                    param_value,
                    c=JET_CMAP(i / n_cv),
                    lw=0.5,
                )

        fig.suptitle(f"Station {STATION_NAMES[station]} ({station})")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir_station, "cross_validation_parameters.png"))
        plt.show()

        # Return values boxplot
        # ---------------------
        fig, ax = plt.subplots(figsize=(8, 5))
        # Do the boxplot with the log return periods (Otherwise there are too many outliers)
        for i in range(n_extreme):
            ax.boxplot(
                np.log10(return_periods_cv[:, order_return_period[i]]),
                positions=[i],
                widths=0.6,
            )
        # Add the return periods observed when fitting on all data
        ax.plot(
            np.arange(n_extreme),
            np.log10(return_period[order_return_period[:n_extreme]]),
            "o",
            c="r",
            label="Training on all years",
        )
        # Add the return periods observed when fitting on all data except the one of the extreme
        ax.plot(
            np.arange(n_extreme),
            np.log10(
                return_periods_cv[
                    years_extreme - YEAR_MIN, order_return_period[:n_extreme]
                ]
            ),
            "o",
            c="b",
            label="Training on all years except the one of the extreme",
        )

        # Adds the ticks for a nice log scale
        vals = np.log10(return_periods_cv[:, order_return_period[:n_extreme]])
        y_min = 0
        y_max = int(np.max(vals[np.isfinite(vals)]) + 1)
        y_ticks_major = np.arange(y_min, y_max + 1)
        y_ticks_minor = np.array(
            [y + np.log10(i) for y in y_ticks_major[:-1] for i in range(1, 10)]
        )

        ax.set_xticks(np.arange(n_extreme), extreme_names, rotation=90)
        ax.set_yticks(y_ticks_major)
        ax.set_yticks(y_ticks_minor, minor=True)
        ax.set_yticklabels([f"$10^{{{i}}}$" for i in y_ticks_major])

        ax.set_ylabel("Return period (years)")
        ax.set_xlabel("Extreme event")
        ax.grid(ls=":", alpha=0.7, axis="both", which="major")
        ax.grid(ls=":", alpha=0.3, axis="y", which="minor")
        ax.legend()

        fig.suptitle(f"Station {STATION_NAMES[station]} ({station})")
        fig.savefig(
            os.path.join(out_dir_station, "cross_validation_return_periods.png")
        )
        plt.show()

        # QQ-plot
        # -------
        cdf = sged_cdf_cv[years_station - YEAR_MIN, np.arange(len(years_station))]
        q = np.arange(0.5, len(cdf) + 0.5) / (len(cdf) + 1)  # Quantiles
        q_norm = stats.norm.ppf(q)  # Normal quantiles
        q_cdf = stats.norm.ppf(
            np.sort(cdf)
        )  # Normal eqauivalent quantiles of the empirical CDF

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(q_norm, q_cdf, "o", markersize=3)
        ax.axline((0, 0), slope=1, c="r")
        ax.set_xlabel("Normal quantiles")
        ax.set_ylabel("Empirical quantiles")
        ax.grid(ls=":", alpha=0.7)
        ax.set_title(f"Station {STATION_NAMES[station]} ({station})")
        fig.savefig(os.path.join(out_dir_station, "cross_validation_qq_plot.png"))

        # Return period plot
        # ------------------
        sged_cdf_cv_agnostic = sged_cdf_cv[
            years_station - YEAR_MIN, np.arange(len(years_station))
        ]

        for cdfs, name in [
            (sged_cdf_cv_agnostic, "cross-validation"),
            (sged_cdf_station, "full-data"),
        ]:
            fig, ax = plt.subplots(figsize=(8, 6))
            return_period_plot(np.sort(cdfs), ax=ax, scale=DAYS_IN_YEAR)
            ax.set_xscale("log")
            ax.grid(ls=":", which="major", lw=0.7)
            ax.grid(ls=":", which="minor", lw=0.7, alpha=0.5)
            ax.set_title(f"Station {STATION_NAMES[station]} ({station})")
            fig.savefig(os.path.join(out_dir_station, f"return_period_{name}.png"))
            plt.show()
