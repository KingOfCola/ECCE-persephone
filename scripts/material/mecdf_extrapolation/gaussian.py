# -*-coding:utf-8 -*-
"""
@File    :   gaussian.py
@Time    :   2024/10/29 17:37:56
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Gaussian distribution
"""

from utils.timer import Timer
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
from tqdm import tqdm

from core.distributions.mecdf import MultivariateMarkovianECDF


def extrapolate_mecdf(cdf, x, threshold=1e-1):
    c = cdf(x)[0]
    if c > threshold:
        return c
    else:
        return extrapolate_mecdf_linear(cdf, x)


def extrapolate_mecdf_linear(cdf, x):
    alpha_x = np.min(x)
    alphas = np.geomspace(0.5 * alpha_x, max(2 * alpha_x, 2e-3), 11)
    points = [1 - (1 - x) * (1 - alpha) / (1 - alpha_x) for alpha in alphas]
    c = np.array([cdf(p)[0] for p in points])
    where = c > 0
    lr = np.polyfit(np.log(alphas[where]), np.log(c[where]), 1)

    # evaluation in alpha = 1.
    return np.exp(np.polyval(lr, np.log(alpha_x)))


def rmsle(x, y):
    return np.sqrt(np.mean((np.log(x) - np.log(y)) ** 2))


def error_comparison(rho, N, w, threshold=1e-1):
    mu = np.zeros(w)
    sigma = np.array([[rho ** abs(i - j) for i in range(w)] for j in range(w)])
    multi_n = stats.multivariate_normal(mu, sigma)
    x = multi_n.rvs(N)
    u = stats.norm.cdf(x)
    mcdf_true = multi_n.cdf(x)

    mecdf = MultivariateMarkovianECDF()
    mecdf.fit(u)
    p_data = mecdf.cdf(u)
    p_data_extra = np.array(
        [extrapolate_mecdf(mecdf.cdf, x, threshold=threshold) for x in u]
    )

    where = (mcdf_true < threshold) & (p_data > 0)
    error_mecdf = rmsle(p_data[where], mcdf_true[where])
    error_mecdf_extra = rmsle(p_data_extra[where], mcdf_true[where])
    return error_mecdf, error_mecdf_extra


if __name__ == "__main__":
    # 2D MECDF
    rho = 0.8
    N = 10_000
    w = 3
    mu = np.zeros(w)
    sigma = np.array([[rho ** abs(i - j) for i in range(w)] for j in range(w)])
    multi_n = stats.multivariate_normal(mu, sigma)
    x2 = multi_n.rvs(N)
    u2 = stats.norm.cdf(x2)
    mcdf_true = multi_n.cdf(x2)

    with Timer("Fitting 2D MECDF: %duration"):
        mecdf_2 = MultivariateMarkovianECDF()
        mecdf_2.fit(u2)

    p2_data = mecdf_2.cdf(u2)
    with Timer("Extrapolating 2D MECDF: %duration"):
        p2_data_extra = np.array(
            [extrapolate_mecdf(mecdf_2.cdf, x, threshold=1e-1) for x in u2]
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

    # Plot 2D MECDF
    fig, ax = plt.subplots()
    ax.plot(mcdf_true, p2_data, "o", ms=2, alpha=0.3, label="Empirical CDF")
    ax.plot(mcdf_true, p2_data_extra, "o", ms=2, alpha=0.3, label="Extrapolated CDF")
    ax.set_xlabel("True CDF")
    ax.set_ylabel("Empirical CDF")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axline((0, 0), (1, 1), c="k", ls="--")
    plt.show()

    where = (mcdf_true < 1e-1) & (p2_data > 0)
    error_mecdf = rmsle(p2_data[where], mcdf_true[where])
    error_mecdf_extra = rmsle(p2_data_extra[where], mcdf_true[where])
    print(f"Error MECDF: {error_mecdf:.3f}")
    print(f"Error MECDF Extra: {error_mecdf_extra:.4f}")

    # Plot 2D MECDF
    fig, ax = plt.subplots()
    ax.plot(np.sort(mcdf_true), np.sort(p2_data), label="Empirical CDF")
    ax.plot(np.sort(mcdf_true), np.sort(p2_data_extra), label="Extrapolated CDF")
    ax.set_xlabel("True CDF")
    ax.set_ylabel("Empirical CDF")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axline((0, 0), (1, 1), c="k", ls="--")
    plt.show()

    errors = [list(error_comparison(rho, N, w)) for _ in tqdm(range(10))]
    errors = np.array(errors)

    print(f"Error MECDF: {errors[:, 0].mean():.3f} +/- {errors[:, 0].std():.3f}")
    print(f"Error MECDF Extra: {errors[:, 1].mean():.3f} +/- {errors[:, 1].std():.3f}")
