# -*-coding:utf-8 -*-
"""
@File    :   temperature_extreme_clustering.py
@Time    :   2024/10/08 11:51:03
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Clustering of temperature extremes
"""


from matplotlib.ticker import LogLocator, MultipleLocator
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
from core.distributions.mecdf import (
    MultivariateMarkovianECDF,
    MultivariateInterpolatedECDF,
)
from core.clustering.cluster_sizes import extremal_index
from core.optimization.interpolation import spline_interpolation
from core.optimization.mecdf import cdf_of_mcdf, find_effective_dof
from plots.annual import month_xaxis, MONTHS_CENTER, MONTHS_LABELS_3, MONTHS_STARTS
from plots.scales import LogShiftScale
from utils.loaders.synop_loader import load_fit_synop


from utils.arrays import sliding_windows
from utils.paths import data_dir, output
from utils.timer import Timer


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
    ts_data = load_fit_synop(METRIC)

    # ================================================================================================
    # Parameters
    # ================================================================================================
    DAYS_IN_YEAR = 365

    # Station to consider
    STATION = ts_data.labels[23]

    # Output directory
    OUTPUT_DIR = output(
        f"Meteo-France_SYNOP/Clustered_extremes/{METRIC}/{STATION.value}"
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ================================================================================================
    # Data processing
    # ================================================================================================
    # Seasonality and trend removal
    # ---------------------------------------------
    # Extraction of the temperature profile
    temperatures_sdf = 1 - ts_data.data[STATION].values
    temperatures = ts_data.raw_data[STATION].values
    N = len(temperatures_sdf)

    # ================================================================================================
    # Extreme temperature clustering
    # ================================================================================================
    # Exceedance probability
    # ---------------------------------------------
    ds = np.arange(1, 15)
    ecdf_ds = []
    mmecdf_ds = []

    for d in ds:
        with Timer("MBST creation: %duration"):
            mmecdf = MultivariateInterpolatedECDF(d=d, hmin=1e-3, hmax=5e-2)
            # mmecdf = MultivariateMarkovianECDF(d=d)
            mmecdf.fit(temperatures_sdf)

        with Timer("MBST counting: %duration"):
            ecdf_d = mmecdf.cdf(temperatures_sdf)

        mmecdf_ds.append(mmecdf)
        ecdf_ds.append(ecdf_d)

    # ================================================================================================
    # Plots
    # ================================================================================================
    # ECDF
    # ---------------------------------------------
    dofs = np.zeros(len(ds))
    fig, ax = plt.subplots()
    for d, ecdf_d, mmecdf in zip(ds, ecdf_ds, mmecdf_ds):
        q = np.arange(1, ecdf_d.shape[0] + 1) / (ecdf_d.shape[0] + 1)
        ecdf_d_sorted = np.sort(ecdf_d)
        effective_dof = find_effective_dof(q, ecdf_d_sorted)
        dofs[d - 1] = effective_dof

        # rho_fit = auto
        ecdf_theoretical = cdf_of_mcdf(q, dof=effective_dof)

        ax.plot(
            q,
            ecdf_d_sorted,
            c=CMAP(d / len(ds)),
            label=rf"d={d} ($\text{{dof}}={effective_dof:.2f}$)",
        )
        ax.plot(ecdf_theoretical, q, c=CMAP(d / len(ds)), ls=":")
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", zorder=-1, lw=2)
    ax.set_xlabel("Proportion of $d$-consecutive temperature values")
    ax.set_ylabel(
        "Probability of all coordinates to be below\nthose of the $d$-consecutive values"
    )
    ax.legend()
    ax.set_title("Probability of extremality")
    fig.savefig(os.path.join(OUTPUT_DIR, "ecdf.png"), dpi=300)
    plt.show()

    # Same plot, logit scale
    dofs = np.zeros(len(ds))
    fig, ax = plt.subplots(figsize=(8, 6))
    for d, ecdf_d, mmecdf in zip(ds, ecdf_ds, mmecdf_ds):
        q = np.arange(1, ecdf_d.shape[0] + 1) / (ecdf_d.shape[0] + 1)
        ecdf_d_sorted = np.sort(ecdf_d)
        effective_dof = find_effective_dof(q, ecdf_d_sorted)
        dofs[d - 1] = effective_dof

        # rho_fit = auto
        ecdf_theoretical = cdf_of_mcdf(q, dof=effective_dof)

        ax.plot(
            ecdf_d_sorted,
            q,
            c=CMAP(d / len(ds)),
            label=rf"d={d} ($\text{{dof}}={effective_dof:.2f}$)",
        )
        ax.plot(q, ecdf_theoretical, c=CMAP(d / len(ds)), ls=":")
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", zorder=-1, lw=2)
    ax.set_xlabel(r"$\pi$")
    ax.set_ylabel(r"$\mathbf{P}\left(\Pi \leq \pi\right)$")
    ax.set_xscale("logit")
    ax.set_yscale("logit")
    ax.grid(True, which="major", ls=":")
    ax.grid(True, which="minor", ls=":", alpha=0.3)
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    ax.set_title("Probability of extremality")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "ecdf.png"), dpi=300)
    plt.show()

    # Return periods
    # ---------------------------------------------
    fig, axes = plt.subplots(
        int(np.ceil(len(ds) / 2)), 2, figsize=(8, 12), sharex=True, sharey=True
    )
    for i, d in enumerate(ds):
        ax = axes.flat[i]
        corrected_ecdf = cdf_of_mcdf(ecdf_ds[i], dofs[i])
        if d == 1:
            corrected_ecdf *= 1 - 1 / N

        return_periods = 1 / corrected_ecdf / DAYS_IN_YEAR
        ax.plot(
            ts_data.yearf[: len(return_periods)],
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
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=15))

    fig.suptitle("Return periods in years of temperature extremes")
    fig.savefig(os.path.join(OUTPUT_DIR, "return_periods_consecutive.png"), dpi=300)

    # Zoom on 2003-2004
    # ---------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
    ax = axes[0]
    year = 2003

    ax.plot(
        (ts_data.yearf[: len(temperatures_sdf)] - year) * DAYS_IN_YEAR,
        temperatures_sdf,
        label=f"d={d}",
        ls="none",
        markersize=2,
        marker="o",
        c="k",
    )
    # ax.set_yscale("log")

    ax.grid(True, axis="y", which="major", ls=":")
    ax.grid(True, axis="y", which="minor", ls=":", alpha=0.3)
    ax.set_ylabel("Return period (years)")
    month_xaxis(ax)

    ax = ax.legend()

    ax = axes[-1]
    ax.plot(
        (ts_data.yearf - year) * DAYS_IN_YEAR, temperatures, label="Temperature", c="k"
    )
    ax.set_ylabel("Temperature (°C)")
    month_xaxis(ax)
    ax.set_xlim(0, DAYS_IN_YEAR)

    sub_ds = [1, 2, 5, 10]
    year = 2003
    fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
    ax = axes[0]
    for j, d in enumerate(sub_ds):
        c = CMAP(j / (len(sub_ds) - 1))
        i = d - 1
        corrected_ecdf = cdf_of_mcdf(ecdf_ds[d - 1], dofs[d - 1])
        if d == 1:
            corrected_ecdf *= 1 - 1 / N

        return_periods = 1 / corrected_ecdf / DAYS_IN_YEAR
        ax.plot(
            (ts_data.yearf[: len(return_periods)] - year) * DAYS_IN_YEAR,
            ecdf_ds[d - 1],
            label=f"d={d}",
            ls="none",
            markersize=2,
            marker="o",
            c=c,
        )
        ax.set_yscale("log")

        ax.grid(True, axis="y", which="major", ls=":")
        ax.grid(True, axis="y", which="minor", ls=":", alpha=0.3)
        ax.set_ylabel("Return period (years)")
        month_xaxis(ax)

    ax = ax.legend()

    ax = axes[-1]
    ax.plot(
        (ts_data.yearf - year) * DAYS_IN_YEAR, temperatures, label="Temperature", c="k"
    )
    ax.set_ylabel("Temperature (°C)")
    month_xaxis(ax)
    ax.set_xlim(180, 240)
    fig.suptitle("Return periods in years of temperature extremes")
    fig.savefig(os.path.join(OUTPUT_DIR, "return_periods_2003.png"), dpi=300)

    # Equivalent probability of 2003 heat wave in different years
    # ---------------------------------------------
    days_sim = [1, 2, 5, 10]
    temp_anomalies = np.arange(-1.0, 3.5, 0.5)
    where_period = (ts_data.yearf >= 2003 + 210 / DAYS_IN_YEAR) & (
        ts_data.yearf <= 2003 + 240 / DAYS_IN_YEAR
    )

    temperatures_heat_wave = temperatures[where_period]

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, temp_anomaly in enumerate(temp_anomalies):
        corrected_temperatures = temperatures_heat_wave - temp_anomaly
        corrected_uni_sdf = 1 - ts_data.models[STATION].cdf(
            ts_data.yearf[where_period], corrected_temperatures
        )

        iso_prob_of_probs = np.zeros_like(days_sim, dtype=float)
        for j, d in enumerate(days_sim):
            iso_prob = mmecdf_ds[d - 1].cdf(corrected_uni_sdf)
            iso_prob_of_prob = cdf_of_mcdf(iso_prob, dofs[d - 1])
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
    fig.suptitle("Return periods of 2003 heat wave in temperature anomalies")
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR, "return_periods_2003_different_years.png"), dpi=300
    )
