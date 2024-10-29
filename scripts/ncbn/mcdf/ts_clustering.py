# -*-coding:utf-8 -*-
"""
@File    :   ts_clustering.py
@Time    :   2024/10/29 12:45:07
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Time series clustering
"""


import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from core.distributions.mecdf import MultivariateMarkovianECDF
from utils.loaders.ncbn_loader import load_fit_ncbn
from utils.paths import data_dir, output
from utils.timer import Timer
from utils.arrays import sliding_windows


if __name__ == "__main__":
    # Load stations
    DATA_DIR = data_dir("MeteoSwiss_NCBN/Preprocessed")
    stations_list = pd.read_csv(DATA_DIR / "ncbn_stations.csv")
    OUT_DIR = output("NCBN/Clustered_extremes/MCDF")
    os.makedirs(OUT_DIR, exist_ok=True)
    METRICS = [
        "rad-AVG",
        "snow-SUM",
        "cloud-AVG",
        "presta-AVG",
        "preliq-SUM",
        "sun-SUM",
        # "t-AVG",
        # "t-MIN",
        # "t-MAX",
        "hum-AVG",
    ]
    metric = "preliq-SUM"

    out_dir = OUT_DIR / metric
    os.makedirs(out_dir, exist_ok=True)

    print("#" * 80)
    print("Processing: ", metric)

    # Load data
    with Timer(f"Loading data '{metric}': %duration"):
        data = load_fit_ncbn(metric, model_type=None)
    valid_labels = ~data.data.isna().all(axis=0)
    valid_labels = valid_labels[valid_labels].index.values
    data = data.subset(valid_labels)

    station = valid_labels[0]
    w = 5

    # Prepare data
    t = data.time.values
    x = 1 - data.data[station].values
    x_raw = data.raw_data[station].values

    xx = sliding_windows(x, w)
    xx_raw = sliding_windows(x_raw, w)
    where = np.where(np.isfinite(xx).all(axis=1))[0]

    t_valid = t[where]
    x_valid = x[where]
    xx_valid = xx[where]
    xx_raw_valid = xx_raw[where]

    # Fit MECDF model
    with Timer("Fitting MECDF: %duration"):
        mecdf = MultivariateMarkovianECDF()
        mecdf.fit(xx_valid)

    p_valid = mecdf.cdf(xx_valid)
    x_valid_avg = xx_valid.mean(axis=1)
    xx_raw_valid_avg = xx_raw_valid.mean(axis=1)
    p_order = np.argsort(p_valid)

    # Plot 1D ECDF
    x_sorted = np.sort(x_valid)
    q = (np.arange(len(x_sorted)) + 1) / (len(x_sorted) + 1)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x_sorted, q, "o", ms=2)
    ax.set_xlabel("percentile")
    ax.set_ylabel("ECDF")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axline((0, 0), (1, 1), c="k", ls="--")

    # Plot MECDF with time
    fig, axes = plt.subplots(3, sharex=True, figsize=(8, 8), height_ratios=[1, 1, 2])
    axes[0].plot(t, x, "o", ms=2)
    axes[0].set_ylabel(metric)

    axes[1].plot(t, x_raw, "o", ms=2)
    axes[1].set_ylabel(metric)

    axes[2].plot(t_valid, p_valid, "o", ms=2)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("MECDF")
    axes[2].set_xlabel("Time")
    plt.show()

    # Show some extreme events
    fig, axes = plt.subplots(2, figsize=(8, 8))
    for idx in p_order[:5]:
        t_ex = str(t_valid[idx]).split("T")[0]
        axes[0].plot(
            xx_raw_valid[idx], "o-", ms=4, label=f"{t_ex} - {p_valid[idx]:.2e}"
        )
        axes[1].plot(xx_valid[idx], "o-", ms=4, label=f"{t_ex} - {p_valid[idx]:.2e}")
    axes[0].legend()
    axes[0].set_ylabel(metric)
    axes[1].set_ylabel("ECDF")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Time")
    plt.show()

    # Comparison between MECDF and average rainfall
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(xx_raw_valid_avg, p_valid, "o", ms=2, label=metric)
    ax.plot(xx_raw_valid.max(axis=1), p_valid, "o", ms=2, label=metric)
    ax.set_yscale("log")
    ax.set_xlabel("Average rainfall")
    ax.set_ylabel("MECDF")
    plt.show()

    # 2D MECDF
    x2 = sliding_windows(x, 2)
    where_2 = np.where(np.isfinite(x2).all(axis=1))[0]
    x2_valid = x2[where_2]

    with Timer("Fitting 2D MECDF: %duration"):
        mecdf_2 = MultivariateMarkovianECDF()
        mecdf_2.fit(x2_valid)

    q = np.linspace(0, 0.35, 100)
    q1, q2 = np.meshgrid(q, q)

    p2 = mecdf_2.cdf(np.stack([q1.ravel(), q2.ravel()], axis=1)).reshape(q1.shape)
    p2_ind = q1 * q2

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "3d"})
    # 3d projection
    ax.plot_surface(q1, q2, p2, cmap="Grays")
    ax.contour(q1, q2, p2, cmap="viridis")
    ax.contour(q1, q2, p2_ind, cmap="viridis", linestyles=":")
    ax.set_xlabel("q1")
