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
import os

from matplotlib import pyplot as plt

from plots.mapplot import plot_map, set_lims
from utils.paths import data_dir, output

if __name__ == "__main__":
    FULL_YEAR_MIN = 1959
    FULL_YEAR_MAX = 2023
    YEARS = FULL_YEAR_MAX - FULL_YEAR_MIN + 1

    DAYS_IN_YEAR = 365
    N = YEARS * DAYS_IN_YEAR

    SEED = 42

    OUTPUT_DIR = output("Meteo-France_QUOT-SIM/Stations clustering")

    temperatures_stations = pd.read_parquet(
        data_dir(r"Meteo-France_QUOT-SIM/Preprocessed/1958_2024-05_T_Q.parquet")
    )
    stations = pd.read_parquet(
        data_dir(r"Meteo-France_QUOT-SIM/Preprocessed/stations.parquet")
    )

    temperatures_stations.reset_index(inplace=True)
    temperatures_stations = temperatures_stations.loc[
        (temperatures_stations["year"].between(FULL_YEAR_MIN, FULL_YEAR_MAX))
        & (temperatures_stations["day_of_year"] <= DAYS_IN_YEAR)
    ]

    temperatures = temperatures_stations.drop(columns=["year", "day_of_year"]).values.T
    temperatures_centered = (
        temperatures - temperatures.mean(axis=1, keepdims=True)
    ) / temperatures.std(axis=1, keepdims=True)
    temperatures_centered -= temperatures_centered.mean(axis=0, keepdims=True)

    corr = np.corrcoef(temperatures_centered, rowvar=True)

    # ---------------------------------------------
    # Clustering methods
    # ---------------------------------------------
    ## KMeans
    from sklearn.cluster import KMeans

    coordinates = stations[["longitude", "latitude"]].values

    kmeans_dir = os.path.join(OUTPUT_DIR, "KMeans")
    os.makedirs(kmeans_dir, exist_ok=True)

    for n_cluster in range(2, 9):
        model = KMeans(n_clusters=n_cluster, random_state=SEED)
        model.fit(temperatures_centered)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=model.labels_,
            cmap="rainbow",
            s=5,
            marker="s",
        )
        plot_map("europe", ax=ax, ec="k")
        set_lims(ax, 40, 55, -7, 13)
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.set_title(f"Number of clusters: {n_cluster}")
        fig.savefig(os.path.join(kmeans_dir, f"temperature-clustering_{n_cluster}.png"))
        plt.show()

    ## Gaussian Mixture
    from sklearn.mixture import GaussianMixture

    gaussian_dir = os.path.join(OUTPUT_DIR, "GaussianMixture")
    os.makedirs(gaussian_dir, exist_ok=True)

    for n_cluster in range(2, 2):
        model = GaussianMixture(n_components=n_cluster, random_state=SEED)
        model.fit(temperatures_centered)
        y = model.predict(temperatures_centered)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(
            coordinates[:, 0], coordinates[:, 1], c=y, cmap="rainbow", s=5, marker="s"
        )
        plot_map("europe", ax=ax, ec="k")
        set_lims(ax, 40, 55, -7, 13)
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.set_aspect(1)
        ax.set_title(f"Number of clusters: {n_cluster}")
        fig.savefig(
            os.path.join(gaussian_dir, f"temperature-clustering_{n_cluster}.png")
        )
        plt.show()

    ## Hierarchical clustering
    from scipy.cluster.hierarchy import cut_tree, linkage, dendrogram

    hierarchical_dir = os.path.join(OUTPUT_DIR, "Hierarchical")
    os.makedirs(hierarchical_dir, exist_ok=True)

    n_clusters = list(range(2, 9))

    method = "complete"
    Z = linkage(temperatures_centered, method=method)
    cuttree = cut_tree(Z, n_clusters=n_clusters)

    for i, n_cluster in enumerate(n_clusters):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=cuttree[:, i],
            cmap="rainbow",
            s=5,
            marker="s",
        )
        plot_map("europe", ax=ax, ec="k")
        set_lims(ax, 40, 55, -7, 13)
        ax.set_title(f"Number of clusters: {n_cluster}")
        fig.savefig(
            output(
                "Meteo-France_QUOT-SIM/Quot_SIM2/Clustering/temperature-clustering_complete-8.png"
            )
        )
        plt.show()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=cuttree[:, i],
        cmap="rainbow",
        s=5,
        marker="s",
    )
    plot_map("europe", ax=ax, ec="k")
    set_lims(ax, 40, 55, -7, 13)
    ax.set_title(f"Number of clusters: {n_cluster} (method={method})")
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

    # ---------------------------------------------
    # Dimensionality reduction
    # ---------------------------------------------
    ## PCA
    from sklearn.decomposition import PCA
    import seaborn as sns

    pca = PCA(n_components=3)
    X = pca.fit_transform(temperatures_centered)

    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    sns.histplot(pd.DataFrame(X), x=0, y=1, ax=axes[0, 0])
    sns.histplot(pd.DataFrame(X), x=0, y=2, ax=axes[0, 1])
    sns.histplot(pd.DataFrame(X), x=1, y=2, ax=axes[1, 0])
    sns.histplot(pd.DataFrame(X), x=0, ax=axes[1, 1])
    plt.show()
