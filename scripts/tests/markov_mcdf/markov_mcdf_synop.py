from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm
from tqdm import tqdm
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
from copy import copy
from core.optimization.mecdf import cdf_of_mcdf, find_effective_dof


if __name__ == "__main__":
    OUT_DIR = output("Material/MCDF/Markov-MCDF")
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    W = 5
    metric = "t-MAX"

    with Timer("Loading data"):
        ncbn_data = load_fit_ncbn(metric)
        STATION = ncbn_data.labels[0]

        samples = sliding_windows(ncbn_data[STATION.value], W)
        samples = samples[np.isfinite(samples).all(axis=1)]
        N = samples.shape[0]

    with Timer("Fitting Markov-MCDF"):
        mecdf_2d = MultivariateMarkovianECDF()
        mecdf_2d.fit(samples[:, :2])

    with Timer("Fitting Markov-MECDF"):
        markov_mecdf = MarkovMECDF(mecdf_2d.cdf, bins=31, xmin=1e-3, delta=5e-2)

    x_bins = np.linspace(0, 1, 51)
    y_bins = np.linspace(0, 1, 51)
    x_c = x_bins[:-1] + np.diff(x_bins) / 2
    y_c = y_bins[:-1] + np.diff(y_bins) / 2
    xx, yy = np.meshgrid(x_c, y_c)

    phi_xy = np.array(
        [markov_mecdf.phi(x_i, y_i) for x_i, y_i in tqdm(zip(xx.ravel(), yy.ravel()))]
    )
    phi_xy = phi_xy.reshape(xx.shape)
    fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
    ax[0].imshow(phi_xy, extent=(0, 1, 0, 1), origin="lower")
    ax[0].set_title("Phi empirical")
    ax[1].imshow(
        np.histogram2d(samples[:, 0], samples[:, 1], bins=(x_bins, y_bins))[0],
        extent=(0, 1, 0, 1),
        origin="lower",
    )
    ax[1].set_title("Phi histogram")
    ax[2].scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=4)
    ax[2].set_title("Samples")
    plt.show()

    N = 1000
    samples = samples[:N, :]
    markov = MarkovMCDF_alt(
        phi_markov,
        phi_int_markov,
        W,
        7,
        7,
        phi_kwargs={"markov_mecdf": markov_mecdf},
    )
    markov_ = markov
    markov_.t_bins = 101
    markov_.x_bins = 101

    q = (np.arange(N) + 1) / (N + 1)
    cdfs = markov_.cdf(samples)
    cdfs_emp = np.array(
        [(samples <= x_i[None, :]).all(axis=1).mean() for x_i in samples]
    )
    cdfs_emp_mean = np.sort(cdfs_emp)
    cdfs_emp_std = np.sqrt(cdfs_emp_mean * (1 - cdfs_emp_mean) / N)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.plot(np.sort(cdfs), q, c="C0", label="Markov-MCDF")
    ax.plot(cdfs_emp_mean, q, c="C1", label="Empirical MCDF")
    ax.fill_betweenx(
        q,
        np.maximum(1e-8, cdfs_emp_mean - 1.96 * cdfs_emp_std),
        np.minimum(1 - 1e-3, cdfs_emp_mean + 1.96 * cdfs_emp_std),
        alpha=0.3,
        fc="C1",
    )
    for i in range(1, W):
        qq = expit(np.linspace(-20, 20, 100))
        ax.plot(
            qq,
            cdf_of_mcdf(qq, dof=i),
            c="k",
            ls="--",
        )
    ax.set_xscale("logit")
    ax.set_yscale("logit")
    ax.grid(which="major", axis="both", alpha=0.7)
    ax.grid(which="minor", axis="both", alpha=0.3)
    ax.set_title("CDFs")
    ax.set_xlabel("CDF")
    ax.set_ylabel("quantile")
    ax.legend()
    plt.show()
    fig.savefig(OUT_DIR / "markov_mcdf_synop.png", dpi=300)
