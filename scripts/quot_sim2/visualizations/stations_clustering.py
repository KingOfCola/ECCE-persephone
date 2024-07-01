# -*-coding:utf-8 -*-
"""
@File      :   stations_clustering.py
@Time      :   2024/07/01 14:51:47
@Author    :   Urvan Christen
@Version   :   1.0
@Contact   :   urvan.christen@gmail.com
@Desc      :   Scripts for clustering the stations from the QUOT_SIM2 study. based on their
                temperature profiles.
"""
import numpy as np
import pandas as pd
from time import time
import statsmodels.api as sm
import tqdm

from matplotlib import pyplot as plt

from utils.paths import data, output

if __name__ == "__main__":
    DIRNAME = data("Quot_SIM2/Raw")
    filename = r"QUOT_SIM2_2000-2009.csv.gz"

    start = time()
    data = pd.read_csv(rf"{DIRNAME}\{filename}", sep=";")
    end = time()

    print(f"Elapsed time: {end-start:.2f}s")
    print(data.columns)
    print(data.shape)
    print(data["DATE"].nunique())

    data["LAMBX"].nunique()
    data["LAMBY"].nunique()
    data["DATEDAY"] = pd.to_datetime(data["DATE"], format="%Y%m%d")
    data["YEAR"] = data["DATEDAY"].dt.year
    data["DAY_OF_YEAR"] = data["DATEDAY"].dt.dayofyear

    temperatures_df = data.pivot_table(
        index=("LAMBX", "LAMBY"),
        columns=("YEAR", "DAY_OF_YEAR"),
        values="T_Q",
        aggfunc="sum",
        fill_value=0,
    )
    temperatures = temperatures_df.values
    temperatures_centered = (
        temperatures - temperatures.mean(axis=1, keepdims=True)
    ) / temperatures.std(axis=1, keepdims=True)
    temperatures_centered -= temperatures_centered.mean(axis=0, keepdims=True)

    corr = np.corrcoef(temperatures_centered, rowvar=True)

    from sklearn.cluster import KMeans

    for n_cluster in range(2, 9):
        model = KMeans(n_clusters=n_cluster)
        model.fit(temperatures_centered)

        lamberts = temperatures_df.index.to_numpy()
        lamberts = np.array([list(l) for l in lamberts])

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(
            lamberts[:, 0],
            lamberts[:, 1],
            c=model.labels_,
            cmap="rainbow",
            s=5,
            marker="s",
        )
        ax.set_xlabel(r"$\lambda_x (hm)$")
        ax.set_ylabel(r"$\lambda_y (hm)$")
        ax.set_aspect(1)
        ax.set_title(f"Number of clusters: {n_cluster}")
        plt.show()

    from sklearn.mixture import GaussianMixture

    for n_cluster in range(2, 9):
        model = GaussianMixture(n_components=n_cluster)
        model.fit(temperatures_centered)
        y = model.predict(temperatures_centered)

        lamberts = temperatures_df.index.to_numpy()
        lamberts = np.array([list(l) for l in lamberts])

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(lamberts[:, 0], lamberts[:, 1], c=y, cmap="rainbow", s=5, marker="s")
        ax.set_xlabel(r"$\lambda_x (hm)$")
        ax.set_ylabel(r"$\lambda_y (hm)$")
        ax.set_aspect(1)
        ax.set_title(f"Number of clusters: {n_cluster}")
        plt.show()

    from scipy.cluster.hierarchy import cut_tree, linkage, dendrogram

    n_clusters = list(range(2, 9))

    method = "complete"
    Z = linkage(temperatures_centered, method=method)
    cuttree = cut_tree(Z, n_clusters=n_clusters)

    for i, n_cluster in enumerate(n_clusters):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(
            lamberts[:, 0],
            lamberts[:, 1],
            c=cuttree[:, i],
            cmap="rainbow",
            s=5,
            marker="s",
        )
        ax.set_title(f"Number of clusters: {n_cluster}")
        plt.show()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        lamberts[:, 0],
        lamberts[:, 1],
        c=cuttree[:, i],
        cmap="rainbow",
        s=5,
        marker="s",
    )
    ax.set_title(f"Number of clusters: {n_cluster} (method={method})")
    fig.savefig(output("Quot_SIM2/Clustering/temperature-clustering_complete-8.png"))
    plt.show()

    # Plot the dendrogram
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(Z, ax=ax, truncate_mode="level", p=5)
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        temperatures.columns.get_level_values(0),
        temperatures.columns.get_level_values(1),
    )
    plt.show()
