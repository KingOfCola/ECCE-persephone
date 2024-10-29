# -*-coding:utf-8 -*-
"""
@File    :   bin_Tree_bug.py
@Time    :   2024/10/29 17:23:47
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Binary tree bug diagnostic
"""

import os
import pandas as pd
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

    x = x2_valid[8591]

    alpha_x = np.min(x)
    alphas = np.geomspace(0.5, 2, 11) * alpha_x
    points = np.array([1 - (1 - x) * (1 - alpha) / (1 - alpha_x) for alpha in alphas])
    print("BinTree: ", list(mecdf_2.tree.count_points_below_multiple(points)))
    print("BinTree: ", [mecdf_2.tree.count_points(point) for point in points])
    print(
        "Naive:   ",
        [(x2_valid < point.reshape(1, -1)).all(axis=1).sum() for point in points],
    )
