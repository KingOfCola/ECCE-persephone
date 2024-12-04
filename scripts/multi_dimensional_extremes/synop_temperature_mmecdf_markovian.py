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
import os
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.signal import find_peaks
from tqdm import tqdm
from pickle import dump, load

from core.distributions.sged import HarmonicSGED
from core.distributions.bernoulli import HarmonicBernoulli
from core.distributions.mecdf import (
    MultivariateMarkovianECDF,
    MultivariateInterpolatedECDF,
    MarkovMCDF,
    phi_kde,
    phi_int_kde,
)
from core.distributions.kde.beta_kde import BetaKDE, BetaKDEInterpolated
from core.clustering.cluster_sizes import extremal_index
from core.mathematics.functions import expit
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
    CMAP = matplotlib.colormaps.get_cmap("jet")

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
        f"Meteo-France_SYNOP/Clustered_extremes_Markov/{METRIC}/{STATION.value}"
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
    ds = np.arange(1, 11)
    ecdf_ds = []
    mmecdf_ds = []
    ecdf_raw_ds = []
    mmecdf_raw_ds = []
    temperatures_sdf_w2 = sliding_windows(temperatures_sdf, w=2)
    temperatures_sdf_w2 = temperatures_sdf_w2[
        np.isfinite(temperatures_sdf_w2).all(axis=1)
    ]
    start_idx_ds = []

    with Timer("KDE creation:"):
        beta_kde = BetaKDEInterpolated(temperatures_sdf_w2, ddof=1, condidx=[0])

    for d in ds:
        print(f"Processing d={d}")
        with Timer("MBST creation: %duration"):
            mmecdf_raw = MultivariateMarkovianECDF(d=d)
            mmecdf_raw.fit(temperatures_sdf)
        with Timer("MBST counting: %duration"):
            ecdf_raw_d = mmecdf_raw.cdf(temperatures_sdf)

        mmecdf_raw_ds.append(mmecdf_raw)
        ecdf_raw_ds.append(ecdf_raw_d)

        with Timer("MBST creation: %duration"):
            mmecdf = MarkovMCDF(
                phi=phi_kde,
                phi_int=phi_int_kde,
                order=d,
                t_bins=51,
                x_bins=51,
                phi_kwargs={"kde": beta_kde},
            )
        if d == 1:
            start_idx_ds.append(np.where(np.isfinite(temperatures_sdf))[0])
            ecdf_d = temperatures_sdf
        else:
            with Timer("MBST counting: %duration"):
                temperatures_sdf_w = sliding_windows(temperatures_sdf, w=d)
                start_idx = np.where(np.all(np.isfinite(temperatures_sdf_w), axis=1))[0]
                start_idx_ds.append(start_idx)
                temperatures_sdf_w = temperatures_sdf_w[start_idx]
                ecdf_d = mmecdf.cdf(temperatures_sdf_w)

        mmecdf_ds.append(mmecdf)
        ecdf_ds.append(ecdf_d)

    # ================================================================================================
    # Save data
    # ================================================================================================
    # Save data
    pkl_path = os.path.join(OUTPUT_DIR, "data.pkl")
    with open(pkl_path, "wb") as f:
        data_dict = {
            "ds": ds,
            "ecdf_ds": ecdf_ds,
            # "mmecdf_ds": mmecdf_ds,
            "ecdf_raw_ds": ecdf_raw_ds,
            # "mmecdf_raw_ds": mmecdf_raw_ds,
            # "beta_kde": beta_kde,
        }
        dump(
            data_dict,
            f,
        )

    # Load data
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            data_dict = load(f)
        ds = data_dict["ds"]
        ecdf_ds = data_dict["ecdf_ds"]
        # mmecdf_ds = data_dict["mmecdf_ds"]
        ecdf_raw_ds = data_dict["ecdf_raw_ds"]
        # mmecdf_raw_ds = data_dict["mmecdf_raw_ds"]
        # beta_kde = data_dict["beta_kde"]

    # ================================================================================================
    # Plots
    # ================================================================================================
    # KDE
    # ---------------------------------------------
    q = np.linspace(0, 1, 101)
    xx, yy = np.meshgrid(q, q)
    beta_kde_values = beta_kde.pdf(np.array([xx.flatten(), yy.flatten()]).T).reshape(
        xx.shape
    )
    beta_kde_cond_values = beta_kde.conditional_pdf(
        np.array([xx.flatten(), yy.flatten()]).T
    ).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(
        beta_kde_values.T,
        extent=(0, 1, 0, 1),
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmax=5,
    )
    fig, ax = plt.subplots()
    ax.imshow(
        beta_kde_cond_values.T,
        extent=(0, 1, 0, 1),
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmax=5,
    )

    fig, ax = plt.subplots()
    for i in range(0, len(q), 20):
        ax.plot(q, beta_kde_cond_values[i], c=CMAP(q[i]), lw=0.5)
    plt.show()

    # ECDF
    # ---------------------------------------------
    LIM = 1e-6
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
    dofs_raw = np.zeros(len(ds))
    fig, ax = plt.subplots(figsize=(8, 6))
    for d, ecdf_d, mmecdf, ecdf_raw_d, mmecdf_raw in zip(
        ds, ecdf_ds, mmecdf_ds, ecdf_raw_ds, mmecdf_raw_ds
    ):
        q = np.arange(1, ecdf_d.shape[0] + 1) / (ecdf_d.shape[0] + 1)
        ecdf_d_sorted = np.sort(ecdf_d)
        ecdf_raw_d_sorted = np.sort(ecdf_raw_d)
        effective_dof = find_effective_dof(q, ecdf_d_sorted)
        dofs[d - 1] = effective_dof
        dofs_raw[d - 1] = find_effective_dof(q, ecdf_raw_d_sorted)

        # rho_fit = auto
        ecdf_theoretical = cdf_of_mcdf(q, dof=effective_dof)

        ax.plot(
            ecdf_d_sorted,
            q,
            c=CMAP(d / len(ds)),
            label=rf"d={d} ($\text{{dof}}={effective_dof:.2f}$)",
        )
        ax.plot(
            ecdf_raw_d_sorted,
            q,
            c=CMAP(d / len(ds)),
            ls=":",
            # label=rf"d={d} ($\text{{dof}}={effective_dof:.2f}$)",
        )
        # ax.plot(q, ecdf_theoretical, c=CMAP(d / len(ds)), ls=":")

    q = expit(np.linspace(-30, 20, 100))
    for d in ds:
        ax.plot(
            q,
            cdf_of_mcdf(q, dof=d),
            c="k",
            ls="--",
            lw=0.7,
            # label=rf"$\text{{dof}}={d}$",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", zorder=-1, lw=2)
    ax.set_xlabel(r"$\pi$")
    ax.set_ylabel(r"$\mathbf{P}\left(\Pi \leq \pi\right)$")
    ax.set_xscale("logit")
    ax.set_yscale("logit")
    ax.grid(True, which="major", ls=":")
    ax.grid(True, which="minor", ls=":", alpha=0.3)
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    ax.set_title("Probability of extremality")
    ax.set_xlim(1e-10, 1 - 1e-4)
    ax.set_ylim(1e-4, 1 - 1e-4)
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
    ax.set_ylabel("Multivariate survival function")
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
    mcdf_days = []
    iso_prob_of_probs = np.zeros((len(days_sim), len(temp_anomalies)), dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, d in enumerate(days_sim):
        corrected_uni_sdfs = []
        n_windows = len(temperatures_heat_wave) - d + 1
        for j, temp_anomaly in enumerate(temp_anomalies):
            corrected_temperatures = temperatures_heat_wave - temp_anomaly
            corrected_uni_sdf = 1 - ts_data.models[STATION].cdf(
                ts_data.yearf[where_period], corrected_temperatures
            )
            corrected_uni_sdfs.extend(sliding_windows(corrected_uni_sdf, w=d))

        corrected_uni_sdfs = np.array(corrected_uni_sdfs)
        iso_prob = mmecdf_ds[d - 1].cdf(corrected_uni_sdfs)
        mcdf_days.append(iso_prob)
        iso_prob_of_prob = cdf_of_mcdf(iso_prob, dofs[d - 1])
        if d == 1:
            iso_prob_of_prob *= 1 - 1 / N

        for j, temp_anomaly in enumerate(temp_anomalies):
            iso_prob_of_probs[i, j] = np.min(
                iso_prob_of_prob[j * n_windows : (j + 1) * n_windows]
            )

    days_index = np.arange(len(days_sim))
    for j, temp_anomaly in enumerate(temp_anomalies):
        corrected_return_periods = 1 / iso_prob_of_probs[:, j] / DAYS_IN_YEAR

        ax.plot(
            days_index,
            corrected_return_periods,
            label=f"{temp_anomaly:+.1f}°C",
            ls="none",
            markersize=4,
            marker="o" if temp_anomaly != 0 else "s",
            mew=0.5,
            mec="k",
            mfc=CMAP(1 - j / (len(temp_anomalies) - 1)),
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
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=15))
    fig.suptitle("Return periods of 2003 heat wave in temperature anomalies")
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR, "return_periods_2003_different_years.png"), dpi=300
    )

    # ================================================================================================
    # Extreme temperature clustering - Examples
    # ================================================================================================
    n_extremes = 10
    w = 5

    ecdf_d = ecdf_ds[w - 1]
    corrected_ecdf = cdf_of_mcdf(ecdf_ds[w - 1], dofs[w - 1])
    # ecdf_d = ts_data[STATION.value]

    peaks = find_peaks(-np.log(corrected_ecdf), height=4, distance=w)[0]
    peaks = peaks[np.argsort(np.log(corrected_ecdf[peaks]))]

    yearf = ts_data.yearf.values
    t_ci = np.zeros((3, len(yearf)))
    with Timer("PPF: %duration"):
        t_ci[0, :] = ts_data.models[STATION].ppf(yearf, 0.025)
    with Timer("PPF: %duration"):
        t_ci[1, :] = ts_data.models[STATION].ppf(yearf, 0.5)
    with Timer("PPF: %duration"):
        t_ci[2, :] = ts_data.models[STATION].ppf(yearf, 0.975)

    fig, axes = plt.subplots(n_extremes // 2, 2, figsize=(12, 12), sharex=True)
    for i, (peak, ax) in enumerate(zip(peaks[:n_extremes], axes.flatten())):
        year = ts_data.year.values[peak]
        yearf = DAYS_IN_YEAR * (ts_data.yearf.values - year)
        date = ts_data.time[peak]
        date_str = f"{date.year:04d}-{date.month:02d}-{date.day:02d}"
        rp = 1 / corrected_ecdf[peak] / DAYS_IN_YEAR
        rp_str = f"${rp:.2g}$".replace("e", "\\times 10^").replace("+0", "")
        ax.plot(yearf, temperatures, c="k", label="Temperature")
        ax.plot(yearf, t_ci[1, :], c="r", label="Normal")
        ax.fill_between(
            yearf, t_ci[0, :], t_ci[2, :], fc="r", alpha=0.4, label="95\\% CI"
        )
        ax.axvspan(yearf[peak], yearf[peak + w], fc="b", alpha=0.5)
        ax.set_ylabel(
            f"Temperature (°C)\nYear {date_str}\nRP: {1 / corrected_ecdf[peak] / DAYS_IN_YEAR:.2f} years"
        )
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.grid(True, axis="y", which="major", ls=":")
        month_xaxis(ax)
        if i == 0:
            ax.legend(ncols=3, loc="lower left", bbox_to_anchor=(0, 1.05))
    ax.set_xlabel("Month")
    ax.set_xlim(0, DAYS_IN_YEAR)
    fig.suptitle("Extreme temperature examples")
    fig.tight_layout()
    plt.show()

    n_extremes = 5
    for start in [0, 50]:
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, peak in enumerate(peaks[start : start + n_extremes]):
            c = CMAP(i / (n_extremes - 1))
            days = np.arange(w)
            year = ts_data.year.values[peak]
            yearf = DAYS_IN_YEAR * (ts_data.yearf.values - year)
            date = ts_data.time[peak]
            date_str = f"{date.year:04d}-{date.month:02d}-{date.day:02d}"
            rp = 1 / corrected_ecdf[peak] / DAYS_IN_YEAR
            rp_str = f"${rp:.2g}$".replace("e", "\\times 10^").replace("+0", "")
            ax.plot(
                days,
                temperatures_sdf[peak : peak + w],
                "o-",
                c=c,
                label=f"{date_str} - RP: {rp_str} years",
            )

        ax.set_yscale("log")
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.grid(True, which="major", ls=":")

        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0))
        ax.set_xlabel("Days")
        ax.set_xlim(-0.5, w - 0.5)
        fig.suptitle("Extreme temperature examples")
        fig.tight_layout()
        plt.show()
