from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm
from scripts.tests.markov_mcdf.markov_mcdf import (
    MarkovMCDF_alt,
    phi_gaussian,
    phi_int_gaussian,
)
from copy import copy


def gaussian_ar1(n, w, rho):
    x = np.random.normal(0, 1, size=(n, w))
    for i in range(1, w):
        x[:, i] = rho * x[:, i - 1] + x[:, i] * np.sqrt(1 - rho**2)
    return norm.cdf(x)


if __name__ == "__main__":

    W = 4
    N = 10_000
    RHO = 0.7

    samples = gaussian_ar1(N, W, RHO)
    x_bins = np.linspace(0, 1, 21)
    y_bins = np.linspace(0, 1, 21)
    x_c = x_bins[:-1] + np.diff(x_bins) / 2
    y_c = y_bins[:-1] + np.diff(y_bins) / 2
    xx, yy = np.meshgrid(x_c, y_c)
    markov = MarkovMCDF_alt(
        phi_gaussian, phi_int_gaussian, W, 21, 21, phi_kwargs={"rho": RHO}
    )
    # markov.fit()
    markov_ = markov
    markov_.t_bins = 101
    markov_.x_bins = 101

    phi_xy = np.array(
        [phi_gaussian(x_i, y_i, rho=RHO) for x_i, y_i in zip(xx.ravel(), yy.ravel())]
    )
    phi_xy = phi_xy.reshape(xx.shape)

    q = (np.arange(N) + 1) / (N + 1)
    cdfs = markov_.cdf(samples)
    cdfs_emp = np.array(
        [(samples <= x_i[None, :]).all(axis=1).mean() for x_i in samples]
    )
    cdfs_emp_mean = np.sort(cdfs_emp)
    cdfs_emp_std = np.sqrt(cdfs_emp_mean * (1 - cdfs_emp_mean) / N)

    # cdfs_cond = np.array([markov_.cdf_cond(t, x_i[1]) for x_i in samples])
    t = 0.7

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.plot(q, np.sort(cdfs), c="C0", label="Markov-MCDF")
    ax.plot(q, cdfs_emp_mean, c="C1", label="Empirical MCDF")
    ax.fill_between(
        q,
        np.maximum(1e-8, cdfs_emp_mean - 1.96 * cdfs_emp_std),
        np.minimum(1 - 1e-3, cdfs_emp_mean + 1.96 * cdfs_emp_std),
        alpha=0.3,
        fc="C1",
    )
    ax.set_xscale("logit")
    ax.set_yscale("logit")
    ax.grid(which="major", axis="both", alpha=0.7)
    ax.grid(which="minor", axis="both", alpha=0.3)
    ax.set_title("CDFs")
    ax.set_xlabel("q")
    ax.set_ylabel("CDF")
    ax.legend()
    plt.show()

    markov_2 = markov
    while markov_2.order > 2:
        markov_2 = markov_2.child

    cdf_true = np.cumsum(
        phi_xy * np.diff(x_bins)[None, :] * np.diff(y_bins)[:, None], axis=1
    ).cumsum(axis=0)
    cdf_markov = np.array(
        [[markov_.cdf(np.array([x_i, y_i])) for x_i in x_c] for y_i in y_c]
    )
    fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
    ax[0].imshow(cdf_true, extent=(0, 1, 0, 1), origin="lower")
    ax[0].set_title("True CDF")
    ax[1].imshow(cdf_markov, extent=(0, 1, 0, 1), origin="lower")
    ax[1].set_title("Markov CDF")
    ax[2].scatter(cdf_true.ravel(), cdf_markov.ravel(), alpha=0.5, s=4)
    ax[2].set_title("True vs Markov CDF")
    ax[2].axline((0, 0), (1, 1), color="black", ls="--")
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    plt.show()

    cdf_true = x_c
    cdf_markov = np.array([markov_.cdf(np.array([x_i])) for x_i in x_c])
    fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
    ax[0].plot(x_c, cdf_true)
    ax[0].set_title("True CDF")
    ax[1].plot(x_c, cdf_markov)
    ax[1].set_title("Markov CDF")
    ax[2].scatter(cdf_true.ravel(), cdf_markov.ravel(), alpha=0.5, s=4)
    ax[2].set_title("True vs Markov CDF")
    ax[2].axline((0, 0), (1, 1), color="black", ls="--")
    plt.show()
