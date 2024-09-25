# -*-coding:utf-8 -*-
"""
@File    :   temperature_extreme_clustering.py
@Time    :   2024/09/20 11:51:03
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Clustering of temperature extremes
"""


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
from plots.annual import month_xaxis, MONTHS_CENTER, MONTHS_LABELS_3, MONTHS_STARTS
from plots.scales import LogShiftScale


from utils.paths import data_dir, output


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
    temperatures_all = temperatures_all.loc[
        (temperatures_all["year"].between(FULL_YEAR_MIN, FULL_YEAR_MAX))
        & (temperatures_all["day_of_year"] <= DAYS_IN_YEAR)
    ]

    temperatures = temperatures_all[STATION].values

    # Time vector (in years)
    years = temperatures_all["year"].values
    days = temperatures_all["day_of_year"].values
    time = years + days / DAYS_IN_YEAR

    # ================================================================================================
    # Extreme temperature clustering
    # ================================================================================================
    # SGED fit
    # ---------------------------------------------
    sged = HarmonicSGED(n_harmonics=N_HARMONICS)
    sged.fit(time, temperatures)
    temperatures_cdf = sged.cdf(time, temperatures)

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
