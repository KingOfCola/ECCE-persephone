import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import MDS
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from plots.mapplot import plot_map, set_lims
from utils.loaders.ncbn_loader import load_fit_ncbn
from utils.paths import data_dir, output
from utils.timer import Timer


def nancorr(x, y):
    where = np.isfinite(x) & np.isfinite(y)
    return np.corrcoef(x[where], y[where])[0, 1]


if __name__ == "__main__":
    # Load stations
    DATA_DIR = data_dir("MeteoSwiss_NCBN/Preprocessed")
    stations_list = pd.read_csv(DATA_DIR / "ncbn_stations.csv")
    OUTDIR = output("NCBN/Geography")
    os.makedirs(OUTDIR, exist_ok=True)
    METRICS = [
        "rad-AVG",
        "snow-SUM",
        "cloud-AVG",
        "presta-AVG",
        "preliq-SUM",
        "sun-SUM",
        "t-AVG",
        "t-MIN",
        "t-MAX",
        "hum-AVG",
    ]

    for metric in METRICS:
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

        fig, ax = plt.subplots()
        ax.plot(data.time, data[station], "o", ms=2, alpha=0.3)

        fig, axes = plt.subplots(ncols=2)
        for station in data.labels:
            y = data.raw_data[station]
            x = data._data[station]
            where = np.isfinite(x)
            x = x[where]
            y = y[where]
            p = np.arange(len(x)) / len(x)
            axes[0].plot(p, np.sort(x))
            axes[0].axline((0, 0), (1, 1), c="k", ls="--")
        axes[1].plot(x, y, "o")

        c = np.array(
            [
                [nancorr(data._data[s0], data._data[s1]) for s1 in data.labels]
                for s0 in data.labels
            ]
        )
        c = 0.5 * (c + c.T)
        plt.matshow(c)

        if np.isnan(c).any():
            print("NaNs in the correlation matrix")
            plt.show()
            continue

        mds = MDS(n_components=2, dissimilarity="precomputed")
        mds.fit(1 - np.abs(c))
        coords = mds.embedding_
        fig, ax = plt.subplots()
        ax.scatter(coords[:, 0], coords[:, 1])
        for i, label in enumerate(data.labels):
            ax.text(coords[i, 0], coords[i, 1], label.value)

        # Dendrogram
        distance_m = 1 - np.abs(c)
        # set diagonal to 0
        np.fill_diagonal(distance_m, 0)

        LNKAGE_METHODS = [
            "complete",
            "ward",
        ]

        for method in LNKAGE_METHODS:
            fig, axes = plt.subplots(ncols=2, figsize=(12, 5), width_ratios=[1, 2])
            Z = linkage(squareform(distance_m), method=method)
            dendro = dendrogram(
                Z,
                labels=[l.value for l in data.labels],
                ax=axes[0],
                color_threshold=0.5 * max(Z[:, 2]),
                leaf_rotation=90,
            )
            axes[0].yaxis.set_visible(False)

            clusters = list(zip(dendro["leaves"], dendro["leaves_color_list"]))

            # Plot the stations on a map
            ax = axes[1]
            plot_map("switzerland-country-zone", ax=ax, ec="black", lw=1.2, fc="none")
            plot_map(
                "switzerland-canton-zone",
                ax=ax,
                ec="black",
                lw=0.5,
                fc="none",
                alpha=0.3,
            )
            plot_map(
                "switzerland-district-zone",
                ax=ax,
                ec="black",
                lw=0.5,
                fc="none",
                alpha=0.1,
            )
            set_lims(ax, "Switzerland")
            for i, color in clusters:
                row = stations_list.iloc[i]
                ax.scatter(row["longitude"], row["latitude"], color=color)
                ax.annotate(
                    row["station"],
                    (row["longitude"], row["latitude"]),
                    (0, 5),
                    textcoords="offset points",
                    fontsize=8,
                    ha="center",
                    va="bottom",
                )
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

            # ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            fig.suptitle(f"{metric} - Dendrogram with {method} linkage")
            plt.show()
            fig.savefig(OUTDIR / f"{metric}_{method}_map-dendro.png", dpi=300)
