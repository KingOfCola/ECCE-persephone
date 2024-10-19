# -*-coding:utf-8 -*-
"""
@File    :   geographical_clustering.py
@Time    :   2024/10/18 17:15:13
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Geographical clustering of temperature stations
"""

from matplotlib import pyplot as plt
from plots.mapplot import plot_map, set_lims
from utils.loaders.synop_loader import load_fit_synop, load_synop
from utils.paths import data_dir, output


if __name__ == "__main__":
    from sklearn import manifold
    import numpy as np

    synop = load_synop(
        data_dir(r"Meteo-France_SYNOP/Preprocessed/t_MAX.parquet"),
        data_dir(r"Meteo-France_SYNOP/Raw/postesSynop.csv"),
    )
    synop_model = load_fit_synop("t_MAX")

    labels = [lab for lab in synop_model.labels if synop_model.models[lab]._isfit()]

    corr = np.corrcoef(synop_model.data[labels].values, rowvar=False)
    cities = [f"{lab.name} - {lab.value}" for lab in labels]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), cities, rotation=90)
    ax.set_yticks(range(len(labels)), cities)
    plt.show()

    mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=6)
    results = mds.fit(1 - corr)

    coords = results.embedding_
    lille_label = 7015
    brest_label = 7110
    lille_idx = labels.index(lille_label)
    brest_idx = labels.index(brest_label)
    lille_coor = coords[lille_idx, :]
    brest_coor = coords[brest_idx, :]
    lille_angle = np.arctan2(lille_coor[1], lille_coor[0])
    brest_angle = np.arctan2(brest_coor[1], brest_coor[0])
    lille_to_brest = (brest_angle - lille_angle) % (2 * np.pi)

    complex_coors = coords[:, 0] + 1j * coords[:, 1]
    complex_coors = np.exp(1j * (-lille_angle + np.pi / 2)) * complex_coors
    if lille_to_brest > np.pi:
        complex_coors = -complex_coors.conj()

    corrected_coors = np.array([np.real(complex_coors), np.imag(complex_coors)]).T

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(corrected_coors[:, 0], corrected_coors[:, 1], marker="o")
    for label, x, y in zip(cities, corrected_coors[:, 0], corrected_coors[:, 1]):
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
            # arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        )

    plt.show()

    stations = synop_model.meta["stations"]
    geo_coors = stations.loc[
        [lab.value for lab in labels], ["Latitude", "Longitude"]
    ].values
    geo_coors_quartiles = np.quantile(geo_coors, [0.25, 0.5, 0.75], axis=0)
    geo_coors_median = np.mean(geo_coors, axis=0)
    geo_coors_iqr = geo_coors_quartiles[2, :] - geo_coors_quartiles[0, :]

    corrected_coors_swapped = corrected_coors.copy()
    corrected_coors_swapped[:, 0] = corrected_coors[:, 1]
    corrected_coors_swapped[:, 1] = corrected_coors[:, 0]
    corrected_coors_quartiles = np.quantile(
        corrected_coors_swapped, [0.25, 0.5, 0.75], axis=0
    )
    corrected_coors_median = corrected_coors_quartiles[1, :]
    for i in range(2):
        where = (corrected_coors_quartiles[0, i] <= corrected_coors_swapped[:, i]) & (
            corrected_coors_swapped[:, i] <= corrected_coors_quartiles[2, i]
        )
        corrected_coors_median[i] = np.mean(corrected_coors_swapped[where, i])
    corrected_coors_iqr = (
        corrected_coors_quartiles[2, :] - corrected_coors_quartiles[0, :]
    )
    manifold_coors = (
        corrected_coors_swapped - corrected_coors_median
    ) / corrected_coors_iqr * geo_coors_iqr + geo_coors_median

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(
        geo_coors[:, 1], geo_coors[:, 0], c="darkred", marker="o", label="Geographical"
    )
    ax.scatter(
        manifold_coors[:, 1],
        manifold_coors[:, 0],
        c="red",
        marker="s",
        label="Temperature similarity based",
    )
    for label, x, y, xm, ym in zip(
        cities,
        geo_coors[:, 1],
        geo_coors[:, 0],
        manifold_coors[:, 1],
        manifold_coors[:, 0],
    ):
        ax.annotate(
            "",
            xy=(x, y),
            xytext=(xm, ym),
            textcoords="data",
            ha="center",
            va="center",
            arrowprops=dict(arrowstyle="<-", connectionstyle="arc3,rad=0", ec="gray"),
        )
    plot_map(ax=ax, map_name="europe-10m", ec="k")
    set_lims(ax, 39, 56, -8, 14)
    ax.legend(loc="upper right", framealpha=1)
    fig.savefig(
        output(r"Meteo-France_SYNOP\Geography\temperature_similarity.png"), dpi=300
    )
