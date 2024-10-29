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
from sklearn.linear_model import LinearRegression

from core.distributions.mecdf import MultivariateMarkovianECDF
from utils.loaders.ncbn_loader import load_fit_ncbn
from utils.paths import data_dir, output
from utils.timer import Timer
from utils.arrays import sliding_windows

LR = LinearRegression(fit_intercept=False)


def extrapolate_mecdf(cdf, x, threshold=1e-1):
    c = cdf(x)[0]
    if c > threshold:
        return c
    else:
        return extrapolate_mecdf_linear(cdf, x)


def extrapolate_mecdf_linear(cdf, x):
    alpha_x = np.min(x)
    alphas = np.geomspace(0.5, 2, 11) * alpha_x
    points = [1 - (1 - x) * (1 - alpha) / (1 - alpha_x) for alpha in alphas]
    c = np.array([cdf(p)[0] for p in points])
    where = c > 0
    lr = np.polyfit(np.log(alphas[where]), np.log(c[where]), 1)

    # evaluation in alpha = 1.
    return np.exp(np.polyval(lr, np.log(alpha_x)))


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
    metric = "t-AVG"

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

    # 2D MECDF
    x2 = sliding_windows(x, 2)
    where_2 = np.where(np.isfinite(x2).all(axis=1))[0]
    x2_valid = x2[where_2]

    with Timer("Fitting 2D MECDF: %duration"):
        mecdf_2 = MultivariateMarkovianECDF()
        mecdf_2.fit(x2_valid)

    p2_data = mecdf_2.cdf(x2_valid)
    with Timer("Extrapolating 2D MECDF: %duration"):
        p2_data_extra = np.array(
            [extrapolate_mecdf(mecdf_2.cdf, x, threshold=1e-1) for x in x2_valid]
        )

    # Plot 2D MECDF
    fig, ax = plt.subplots()
    ax.plot(p2_data, p2_data_extra, "o", ms=2, alpha=0.3)
    ax.set_xlim(0, 1e-1)
    ax.set_ylim(0, 1e-1)
    ax.set_xlabel("p2")
    ax.set_ylabel("p2_extrapolated")
    ax.axline((0, 0), (1, 1), c="k", ls="--")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(p2_data, p2_data_extra, "o", ms=2, alpha=0.3)
    ax.set_xlabel("p2")
    ax.set_ylabel("p2_extrapolated")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.show()

    q = np.linspace(1e-4, 0.35, 100)
    q1, q2 = np.meshgrid(q, q)
    q1_flat = q1.ravel()
    q2_flat = q2.ravel()

    p2 = mecdf_2.cdf(np.stack([q1_flat, q2_flat], axis=1)).reshape(q1.shape)
    p2_extra = np.array(
        [
            extrapolate_mecdf(mecdf_2.cdf, np.array([q1_flat[i], q2_flat[i]]))
            for i in range(len(q1_flat))
        ]
    ).reshape(q1.shape)
    p2_ind = q1 * q2

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "3d"})
    # 3d projection
    ax.plot_surface(q1, q2, p2, cmap="Grays")
    ax.contour(q1, q2, p2, cmap="viridis")
    # ax.contour(q1, q2, p2_ind, cmap="viridis", linestyles=":")
    ax.set_xlabel("q1")
    plt.show()

    fig, ax = plt.subplots()
    levels = np.arange(0, 0.3, 0.02)
    ax.contour(q1, q2, p2, levels=levels, cmap="viridis")
    ax.contour(q1, q2, p2_extra, levels=levels, cmap="viridis", linestyles=":")
    ax.grid(alpha=0.7, lw=0.7)
