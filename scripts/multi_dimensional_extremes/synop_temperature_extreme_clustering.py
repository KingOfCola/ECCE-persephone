# -*-coding:utf-8 -*-
"""
@File    :   temperature_extreme_clustering.py
@Time    :   2024/09/20 11:51:03
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Clustering of temperature extremes
"""


from matplotlib.ticker import MultipleLocator
import pandas as pd
import numpy as np
from numba import njit
import os
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.signal import find_peaks
from tqdm import tqdm

from core.distributions.sged import HarmonicSGED
from core.distributions.bernoulli import HarmonicBernoulli
from core.clustering.cluster_sizes import extremal_index
from core.optimization.interpolation import spline_interpolation
from plots.annual import month_xaxis, MONTHS_CENTER, MONTHS_LABELS_3, MONTHS_STARTS
from plots.scales import LogShiftScale

from cythonized import mbst
from scripts.tests.salvadori_demichele.gaussian_copulas_isolines import (
    find_rho,
    correlated_ecdf,
)


from utils.paths import data_dir, output
from utils.timer import Timer


@njit
def index_of_first_intersection(levels: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Compute the index of the first intersection of a curve with horizontal lines.

    Parameters
    ----------
    levels : np.ndarray
        The levels of the horizontal lines.
    x : np.ndarray
        The monotonic curve to intersect with the horizontal lines.

    Returns
    -------
    np.ndarray
        The index of the first intersection of the curve with the horizontal lines.
    """
    n = levels.size
    m = x.size
    increasing = x[0] < x[-1]
    indices = np.zeros(n, dtype=np.int64)

    levels_order = np.argsort(levels)
    if not increasing:
        levels_order = levels_order[::-1]
    levels_sorted = levels[levels_order]

    j = 0

    # x increasing case
    if increasing:
        for i in range(n):
            # Find the first index where x[j] >= levels[i]
            while j < m and x[j] < levels_sorted[i]:
                j += 1
            indices[levels_order[i]] = j

    # x decreasing case
    else:
        for i in range(n):
            # Find the first index where x[j] <= levels[i]
            while j < m and x[j] > levels_sorted[i]:
                j += 1
            indices[levels_order[i]] = j

    return indices


@njit
def probability_of_isoregion(
    levels: np.ndarray, copula_values: np.ndarray
) -> np.ndarray:
    """
    Compute the probability of being on an iso-line of a copula.

    Parameters
    ----------
    levels : np.ndarray
        The levels of the iso-lines delimiting the iso-regions of interest on the copula.
    copula_values : np.ndarray
        The copula values of the dataset.

    Returns
    -------
    np.ndarray
        The probability of being in the corresponding iso-regions.
    """
    copula_values = np.sort(copula_values)
    intersection_indexes = index_of_first_intersection(levels, copula_values)

    return intersection_indexes / copula_values.size


if __name__ == "__main__":
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        }
    )  # Use LaTeX rendering
    CMAP = matplotlib.colormaps.get_cmap("Spectral")

    # ================================================================================================
    # Data loading
    # ================================================================================================
    METRIC = "t_MAX"
    temperatures_all = pd.read_parquet(
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
    FULL_YEAR_MIN = temperatures_all.loc[
        temperatures_all["day_of_year"] == 1, "year"
    ].min()
    FULL_YEAR_MAX = temperatures_all.loc[
        temperatures_all["day_of_year"] == DAYS_IN_YEAR, "year"
    ].max()
    YEARS = FULL_YEAR_MAX - FULL_YEAR_MIN + 1

    N = YEARS * DAYS_IN_YEAR
    MIN_PRECIPITATION = 1.0

    # Station to consider
    STATION = 7460
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
    temperatures_all = temperatures_all.loc[
        (temperatures_all["year"].between(FULL_YEAR_MIN, FULL_YEAR_MAX))
        & (temperatures_all["day_of_year"] <= DAYS_IN_YEAR)
    ]

    temperatures = temperatures_all[STATION].values

    # Time vector (in years)
    years = temperatures_all["year"].values
    days = temperatures_all["day_of_year"].values
    time = years + days / DAYS_IN_YEAR

    # Trend removal
    f = spline_interpolation(time, temperatures, step=5)

    trend = f(time)
    detrended_temperatures = temperatures - trend

    # ================================================================================================
    # Extreme temperature clustering
    # ================================================================================================
    # SGED fit
    # ---------------------------------------------
    sged = HarmonicSGED(n_harmonics=N_HARMONICS)
    sged.fit(time, detrended_temperatures)
    temperatures_cdf = sged.cdf(time, detrended_temperatures)

    # Extreme temperature clustering
    # ---------------------------------------------
    thresholds = 1 - np.logspace(-3, 0, 101, endpoint=True)[::-1]
    extremal_indexes = extremal_index(
        temperatures_cdf, threshold=thresholds, delta=0, ci=True
    )
    extremal_indexes_r = extremal_index(
        1 - temperatures_cdf, threshold=thresholds, delta=0, ci=True
    )
    auto = np.corrcoef(temperatures_cdf[1:], temperatures_cdf[:-1])[0, 1]

    n_bins = 100
    dxy = 1 / n_bins
    hist, xbins, ybins = np.histogram2d(
        temperatures_cdf[:-1],
        temperatures_cdf[1:],
        bins=np.linspace(0, 1, n_bins + 1, endpoint=True),
        density=True,
    )
    x_centers = (xbins[:-1] + xbins[1:]) / 2
    y_centers = (ybins[:-1] + ybins[1:]) / 2

    # Histogram based
    # ---------------------------------------------
    ecdf2d = np.cumsum(np.cumsum(hist[::-1, ::-1] * dxy**2, axis=1), axis=0)[::-1, ::-1]
    # ecdf2d = np.cumsum(np.cumsum(hist * dxy**2, axis=1), axis=0)
    temperatures_cdf_bin = np.digitize(temperatures_cdf, xbins) - 1
    iso_cdf = ecdf2d[temperatures_cdf_bin[:-1], temperatures_cdf_bin[1:]]

    levels = np.arange(0.1, 1, 0.1)
    isoregions_probabilities = probability_of_isoregion(levels, iso_cdf)
    iso_diagonal = x_centers[
        index_of_first_intersection(
            levels, ecdf2d[np.arange(n_bins - 1), np.arange(n_bins - 1)]
        )
    ]

    # Exceedance probability
    # ---------------------------------------------
    ds = np.arange(1, 15)
    ecdf_ds = []
    mbst_ds = []

    for d in ds:
        multi_cdf = np.zeros((temperatures_cdf.shape[0] - d + 1, d))
        for i in range(d):
            multi_cdf[:, i] = temperatures_cdf[
                i : temperatures_cdf.shape[0] - d + i + 1
            ]

        with Timer("MBST creation: %duration"):
            mbst_d = mbst.MBST(multi_cdf, None)

        with Timer("MBST counting: %duration"):
            ecdf_d = mbst_d.count_points_below_multiple(multi_cdf) / multi_cdf.shape[0]

        mbst_ds.append(mbst_d)
        ecdf_ds.append(ecdf_d)

    # ================================================================================================
    # Plots
    # ================================================================================================
    # ECDF
    # ---------------------------------------------
    rho_fits = np.zeros(len(ds))
    fig, ax = plt.subplots()
    for d, ecdf_d in zip(ds, ecdf_ds):
        q = np.arange(1, ecdf_d.shape[0] + 1) / (ecdf_d.shape[0] + 1)
        ecdf_d_sorted = np.sort(ecdf_d)
        rho_fit = find_rho(q, ecdf_d_sorted, d=d)
        rho_fits[d - 1] = rho_fit

        # rho_fit = auto
        ecdf_theoretical = correlated_ecdf(q, rho_fit, d=d)

        ax.plot(
            q,
            ecdf_d_sorted,
            c=CMAP(d / len(ds)),
            label=rf"d={d} ($\rho={rho_fit:.2f}$)",
        )
        ax.plot(ecdf_theoretical, q, c=CMAP(d / len(ds)), ls=":")
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", zorder=-1, lw=2)
    ax.set_xlabel("Proportion of $d$-consecutive temperature values")
    ax.set_ylabel(
        "Probability of all coordinates to be below\nthose of the $d$-consecutive values"
    )
    ax.legend()
    ax.set_title("Probability of extremality")
    plt.show()

    # Return periods
    # ---------------------------------------------
    fig, axes = plt.subplots(len(ds), 1, figsize=(6, 16), sharex=True, sharey=True)
    for i, d in enumerate(ds):
        ax = axes[i]
        corrected_ecdf = correlated_ecdf(ecdf_ds[i], rho_fits[i], d=d)
        if d == 1:
            corrected_ecdf *= 1 - 1 / N

        return_periods = 1 / (1 - corrected_ecdf) / DAYS_IN_YEAR
        ax.plot(
            time[: len(return_periods)],
            return_periods,
            label=f"d={d}",
            ls="none",
            markersize=2,
            marker="o",
            c="k",
        )
        ax.set_yscale("log")
        ax.set_ylabel(f"$d={d}$")

        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.xaxis.set_major_locator(MultipleLocator(5))

        ax.grid(True, axis="both", which="major", ls=":")
        ax.grid(True, axis="both", which="minor", ls=":", alpha=0.3)

    # Zoom on 2003-2004
    # ---------------------------------------------
    sub_ds = [1, 2, 5, 10]
    year = 2003
    fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
    ax = axes[0]
    for j, d in enumerate(sub_ds):
        c = CMAP(j / (len(sub_ds) - 1))
        i = d - 1
        corrected_ecdf = correlated_ecdf(ecdf_ds[d - 1], rho_fits[d - 1], d=d)
        if d == 1:
            corrected_ecdf *= 1 - 1 / N

        return_periods = 1 / (1 - corrected_ecdf) / DAYS_IN_YEAR
        ax.plot(
            (time[: len(return_periods)] - year) * DAYS_IN_YEAR,
            return_periods,
            label=f"d={d}",
            ls="none",
            markersize=2,
            marker="o",
            c=c,
        )
        ax.set_yscale("log")

        ax.grid(True, axis="y", which="major", ls=":")
        ax.grid(True, axis="y", which="minor", ls=":", alpha=0.3)
        month_xaxis(ax)

    ax = ax.legend()

    ax = axes[-1]
    ax.plot((time - year) * DAYS_IN_YEAR, temperatures, label="Temperature", c="k")
    month_xaxis(ax)
    ax.set_xlim(0, DAYS_IN_YEAR)

    # Equivalent probability of 2003 heat wave in different years
    # ---------------------------------------------
    days_sim = [1, 2, 5, 10]
    temp_anomalies = np.arange(-1.0, 3.5, 0.5)
    where_period = (time >= 2003 + 210 / DAYS_IN_YEAR) & (
        time <= 2003 + 240 / DAYS_IN_YEAR
    )

    temperatures_heat_wave = detrended_temperatures[where_period]

    fig, ax = plt.subplots()
    for i, temp_anomaly in enumerate(temp_anomalies):
        corrected_temperatures = temperatures_heat_wave - temp_anomaly
        corrected_uni_ecdf = sged.cdf(time[where_period], corrected_temperatures)

        iso_prob_of_probs = np.zeros_like(days_sim, dtype=float)
        for j, d in enumerate(days_sim):
            multi_cdf = np.zeros((corrected_uni_ecdf.size - d + 1, d))
            for k in range(d):
                multi_cdf[:, k] = corrected_uni_ecdf[
                    k : corrected_uni_ecdf.size - d + k + 1
                ]
            iso_prob = mbst_ds[d - 1].count_points_below_multiple(multi_cdf) / (
                N - d + 1
            )
            iso_prob_of_prob = correlated_ecdf(iso_prob, rho_fits[d - 1], d=d)
            if d == 1:
                iso_prob_of_prob *= 1 - 1 / N

            iso_prob_of_probs[j] = np.max(iso_prob_of_prob)

        corrected_return_periods = 1 / (1 - iso_prob_of_probs) / DAYS_IN_YEAR
        days_index = np.arange(len(days_sim))

        ax.plot(
            days_index,
            corrected_return_periods,
            label=f"{temp_anomaly:+.1f}°C",
            ls="none",
            markersize=4,
            marker="o",
            mew=0.5,
            mec="k",
            mfc=CMAP(1 - i / (len(temp_anomalies) - 1)),
        )

    ax.set_xticks(days_index)
    ax.set_xticklabels([f"{d}" for d in days_sim])
    ax.set_yscale("log")
    ax.set_xlabel("Number of consecutive days")
    ax.set_ylabel("Return period (years)")
    ax.grid(True, axis="both", which="major", ls=":")
    ax.grid(True, axis="y", which="minor", ls=":", alpha=0.3)

    ax.legend(
        title="Temperature anomaly\nwrt. 2003", bbox_to_anchor=(1, 1), loc="upper left"
    )

    # Copula
    # ---------------------------------------------
    fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
    axes[0].imshow(
        ecdf2d, origin="lower", extent=[0, 1, 0, 1], cmap="plasma", zorder=-1
    )
    axes[0].contour(
        x_centers,
        y_centers,
        ecdf2d,
        levels=np.arange(0.1, 1.0, 0.1),
        colors="black",
        linestyles=np.where(np.arange(0.1, 1.0, 0.1) == 0.5, "-", ":"),
        zorder=1,
    )
    axes[0].set_xlabel("$F(u_{t-1})$")
    axes[0].set_ylabel("$F(u_t)$")
    axes[0].set_title(
        r"Copula of temperature extremes $\bar{C}(u_{t-1}, u_t)=\mathbb{P}[U_{t-1}\geq u_{t-1}, U_t \geq u_t]$"
    )
    for i, level in enumerate(levels):
        p = isoregions_probabilities[i]
        axes[0].text(0, level, f"$p_t={p:.2f}$", color="black", ha="left", va="bottom")
    axes[0].grid(True)

    axes[1].imshow(hist, origin="lower", extent=[0, 1, 0, 1], cmap="plasma")
    axes[1].set_xlabel("$F(u_{t-1})$")
    axes[1].set_ylabel("$F(u_t)$")
    axes[1].set_title("Density of temperature extremes")
    axes[1].grid(True)

    # Probability of being on an iso-line
    # ---------------------------------------------
    u = np.linspace(0, 1, 101, endpoint=True)
    rho = auto
    fig, ax = plt.subplots()
    sns.histplot(iso_cdf, kde=True, ax=ax, stat="density", cumulative=True)
    ax.plot([0, 1], [0, 1], linestyle="--", color="black")
    ax.plot(u, u * (1 - np.log(u)), linestyle=":", color="black")
    ax.plot(u, u * (1 - np.log(u)) ** rho, linestyle=":", color="black")
    ax.set_xlabel("Cumulative copula iso-line")
    ax.set_ylabel("Density")
    ax.set_title("Probability of being on an iso-line")

    # Extremal index
    # ---------------------------------------------
    fig, ax = plt.subplots()
    ax.plot(thresholds, extremal_indexes)
    ax.plot(thresholds, extremal_indexes_r)
    ax.fill_between(
        thresholds, extremal_indexes.lower, extremal_indexes.upper, alpha=0.3
    )
    ax.fill_between(
        thresholds, extremal_indexes_r.lower, extremal_indexes_r.upper, alpha=0.3
    )
    ax.plot(thresholds, thresholds, linestyle="--", color="black")
    ax.plot(thresholds, 1 - (1 - thresholds) ** (auto**2), linestyle=":", color="black")
    ax.set_xscale("logShift")
    ax.set_xlabel("$u$")
    ax.set_xlim(0, None)
    ax.set_ylim(0, 1.1)
    ax.grid(True)
    ax.grid(True, which="minor", linestyle=":")
    ax.set_ylabel(r"Extremal index $\theta$")
    ax.set_title(f"Extremal index of temperature extremes at station {STATION_NAME}")
    plt.tight_layout()

    # ================================================================================================
    # Polar-max-stable plot
    # ================================================================================================
    # Transformation of the margins to unit Frechet
    # ---------------------------------------------
    Y = -1 / np.log(temperatures_cdf)
    Y2 = np.hstack([Y[:-1, None], Y[1:, None]])
    theta = np.arctan2(Y2[:, 0], Y2[:, 1])
    r = np.linalg.norm(Y2, axis=1, ord=1)
    quantiles = np.concatenate([np.arange(0, 0.91, 0.1), [0.95, 0.99]])

    # Polar plot
    # ---------------------------------------------
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.plot(theta, r, linestyle="None", marker="o", markersize=2, alpha=0.3)

    fig, ax = plt.subplots()
    for q in quantiles:
        # sns.histplot(theta[r > np.quantile(r, q)], ax=ax, label=f"{q:.1f}", ec=CMAP(q), element="step", stat="density", fc=(0, 0, 0, 0))
        sns.kdeplot(
            theta[r > np.quantile(r, q)],
            ax=ax,
            color=CMAP(q),
            bw_adjust=min(2 * np.sqrt(1 - q), 1),
            cut=0,
        )
    ax.set_xlim(0, np.pi / 2)
    ax.set_xticks(np.linspace(0, np.pi / 2, 3, endpoint=True))
    ax.set_xticklabels([r"$0$", r"$\pi/4$", r"$\pi/2$"])
    ax.set_xlabel(r"$\theta$")
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    smap = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    fig.colorbar(smap, ax=ax, label="Radius quantile")
    plt.show()

    # ================================================================================================
    # Spectral decomposition d=3
    # ================================================================================================
    # Spectral decomposition
    # ---------------------------------------------
    d = 3
    Y = -1 / np.log(temperatures_cdf)
    Yd = np.zeros((Y.size - d, d))
    for i in range(d):
        Yd[:, i] = Y[i : len(Y) - d + i]

    # Polar coordinates
    r = np.sum(Yd, axis=1)
    r_norm = np.exp(-1 / r)
    omega = Yd / r[:, None]

    # Simplex coordinates
    e1 = np.array([1, 0])
    e2 = np.array([1 / 2, np.sqrt(3) / 2])
    e3 = np.array([0, 0])
    omega_proj = omega[:, 0][:, None] * e1[None, :] + omega[:, 1][:, None] * e2[None, :]

    # Projection on the simplex
    # ---------------------------------------------
    rmin = np.quantile(r_norm, 0.8)
    where = r_norm > rmin
    fig, ax = plt.subplots()
    ax.scatter(
        omega_proj[where, 0],
        omega_proj[where, 1],
        c=r_norm[where],
        linestyle="None",
        marker="o",
        s=10,
        alpha=0.3,
        cmap="Spectral",
        vmin=rmin,
        vmax=1,
    )
