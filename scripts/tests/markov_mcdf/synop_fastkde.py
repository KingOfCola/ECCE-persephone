import fastkde.fastKDE
from matplotlib import pyplot as plt
import numpy as np
from copy import copy
from scipy.stats import norm
from tqdm import tqdm
import fastkde

from core.distributions.mecdf import MultivariateMarkovianECDF
from scripts.tests.markov_mcdf.markov_mcdf import (
    MarkovMCDF_alt,
    MarkovMECDF,
    phi_markov,
    phi_int_markov,
)
from utils.loaders.ncbn_loader import load_fit_ncbn
from utils.arrays import sliding_windows
from utils.timer import Timer
from utils.paths import output
from core.mathematics.functions import logit, expit
from core.optimization.mecdf import cdf_of_mcdf, find_effective_dof

if __name__ == "__main__":
    OUT_DIR = output("Museum")
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    W = 5
    metric = "t-MAX"
    CMAP = plt.get_cmap("Spectral")

    with Timer("Loading data"):
        ncbn_data = load_fit_ncbn(metric)
        STATION = ncbn_data.labels[0]

        samples = sliding_windows(ncbn_data[STATION.value], W)
        samples = samples[np.isfinite(samples).all(axis=1)]
        N = samples.shape[0]

    fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
    with Timer("Fitting Fast-kde"):
        fast_kde_pdf = fastkde.pdf(samples[:, 1], samples[:, 0], var_names=["X1", "X2"])
        fast_kde_pdf.plot(ax=axes[0])

    with Timer("Fitting Fast-kde"):
        fast_kde_cond = fastkde.conditional(
            samples[:, 1], samples[:, 0], var_names=["X1", "X2"]
        )
        fast_kde_cond.plot(ax=axes[1])

    with Timer("Fitting Fast-kde"):
        fast_kde_kde = fastkde.fastKDE.fastKDE(samples[:, :2].T)

    fig, ax = plt.subplots(figsize=(12, 6))
    x1s = np.linspace(0, 1, 11)
    x2s = np.linspace(0, 1, 1001)
    for x in tqdm(x1s):
        points = np.zeros((len(x2s), 2))
        points[:, 0] = x
        points[:, 1] = x2s

        ax.plot(
            x2s,
            fastkde.pdf_at_points(
                samples[:, 1],
                samples[:, 0],
                list_of_points=points,
                do_approximate_ecf=True,
            ),
            c=CMAP(x),
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    fig.savefig(OUT_DIR / "fastkde-original.png", dpi=300)
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 6))
    x1s = np.linspace(0, 1, 11)
    x2s = np.linspace(0, 1, 1001)
    for x in tqdm(x1s):
        points = np.zeros((len(x2s), 2))
        points[:, 0] = x
        points[:, 1] = x2s

        ax.fill_between(
            x2s,
            np.zeros_like(x2s),
            fastkde.pdf_at_points(
                samples[:, 1],
                samples[:, 0],
                list_of_points=points,
                do_approximate_ecf=True,
            ),
            fc=CMAP(x),
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    ax.set_facecolor("midnightblue")
    ax.set_xticks(())
    ax.set_yticks(())
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(OUT_DIR / "fastkde.png", dpi=300)
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 6))
    x1s = np.linspace(0, 1, 11)
    x2s = np.linspace(0, 1, 1001)
    for x in tqdm(x1s):
        points = np.zeros((len(x2s), 2))
        points[:, 0] = x
        points[:, 1] = x2s

        ax.fill_between(
            x2s,
            np.zeros_like(x2s),
            fastkde.pdf_at_points(
                samples[:, 1],
                samples[:, 0],
                list_of_points=points,
                do_approximate_ecf=True,
            ),
            fc=CMAP(x),
        )

    lims = ax.get_ylim()
    np.random.seed(0)
    ax.scatter(
        np.random.uniform(0, 1, 1000),
        np.random.uniform(0, lims[1], 1000),
        s=1,
        c="w",
        alpha=0.3,
        zorder=-1,
    )
    ax.scatter(
        np.random.uniform(0, 1, 100),
        np.random.uniform(0, lims[1], 100),
        s=4,
        c="w",
        alpha=0.7,
        zorder=-1,
    )
    ax.scatter(
        np.random.uniform(0, 1, 10),
        np.random.uniform(0, lims[1], 10),
        s=20,
        c="w",
        alpha=1.0,
        zorder=-1,
    )
    ax.scatter(0.2, 0.8 * lims[1], s=300, c="w", alpha=1.0, zorder=-1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    ax.set_facecolor("midnightblue")
    ax.set_xticks(())
    ax.set_yticks(())
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(OUT_DIR / "fastkde-fancy.png", dpi=300)
    plt.show()
