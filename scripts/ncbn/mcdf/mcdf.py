# -*-coding:utf-8 -*-
"""
@File    :   mcdf.py
@Time    :   2024/10/28 15:58:00
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   MCDF protocols
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import MDS
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from plots.mapplot import plot_map, set_lims
from protocol.cdf_of_mcdf.cdf_of_mcdf import CDFofMCDFProtocol
from utils.loaders.ncbn_loader import load_fit_ncbn
from utils.paths import data_dir, output
from utils.timer import Timer


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

    for metric in METRICS:
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

        print(data.labels)
        if len(data.labels) < 2:
            print("Not enough data")
            continue
        station = data.labels[0].value

        process_generator = data.data.values
        process_args = None
        process_kwargs = None

        for w in [2, 3, 4, 5, 7, 10]:
            protocol = CDFofMCDFProtocol(
                process_generator=process_generator,
                process_args=process_args,
                process_kwargs=process_kwargs,
                n_sim=100,
                w=w,
            )

            protocol.run_simulation()

            fig, ax = plt.subplots(figsize=(8, 8))
            protocol.plot_results(ax, individual=True)
            fig.tight_layout()
            fig.savefig(
                out_dir / f"cdf_of_mcdf_w{w}.png",
                dpi=300,
            )
            plt.show()
