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


def extrapolate_mecdf_single(cdf, x, threshold=1e-1):
    alpha_x = np.min(x)
    c = cdf(x)[0]
    if alpha_x > threshold:
        return c
    else:
        return extrapolate_mecdf_power(cdf, x, threshold=threshold)


def extrapolate_mecdf_dicho(cdf, x, threshold=1e-1, min_count=100):
    alpha_x = np.min(x)
    c = cdf(x)[0]
    if alpha_x > threshold:
        return c
    else:
        return extrapolate_dicho(cdf, x, min_count=min_count)


def extrapolate_mecdf_power(cdf, x, threshold=1e-1):
    alpha_x = np.min(x)
    h1 = threshold
    h2 = h1 / 2
    point1 = 1 - (1 - x) * (1 - h1) / (1 - alpha_x)
    point2 = 1 - (1 - x) * (1 - h2) / (1 - alpha_x)
    c1 = cdf(point1)[0]
    c2 = cdf(point2)[0]
    theta = np.log(c2 / c1) / np.log(h2 / h1)
    scale = c1 / h1**theta

    return alpha_x**theta * scale


def extrapolate_mecdf_power_quad(cdf, x, threshold=1e-1):
    alpha_x = np.min(x)
    alphas = np.array([threshold / 10, threshold / 3, threshold])
    points = [1 - (1 - x) * (1 - alpha) / (1 - alpha_x) for alpha in alphas]
    cdfs = np.array([cdf(p)[0] for p in points])
    lr = np.polyfit(np.log(alphas), np.log(cdfs), 2)

    return np.exp(np.polyval(lr, np.log(alpha_x)))


def extrapolate_dicho(cdf, x, min_count=100):
    alpha_x = np.min(x)
    alpha = 1.0
    c = 1.0
    c_emp_dicho = []
    alphas_dicho = []
    while c > 0:
        alphas_dicho.append(alpha)
        c_emp_dicho.append(c)
        alpha /= 1.5
        c = cdf(np.array([1 - (1 - x) * (1 - alpha) / (1 - alpha_x)]))[0]

    c0 = c_emp_dicho[-1]

    while c_emp_dicho[-1] <= c0 * min_count:
        alphas_dicho.pop()
        c_emp_dicho.pop()

    i = np.digitize(alpha_x, alphas_dicho)
    end = min(i + 3, len(alphas_dicho))
    start = max(end - 5, 0)

    lr = np.polyfit(np.log(alphas_dicho)[start:end], np.log(c_emp_dicho)[start:end], 1)
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

    where = (mcdf_true < threshold) & (p_data > 0)
    error_mecdf = rmsle(p_data[where], mcdf_true[where])

    errors = [error_mecdf]
    for method in [
        extrapolate_mecdf,
        extrapolate_mecdf_single,
        extrapolate_mecdf_dicho,
    ]:
        p_data_extra = np.array([method(mecdf.cdf, x, threshold=threshold) for x in u])
        error = rmsle(p_data_extra[where], mcdf_true[where])
        errors.append(error)

    return errors


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
            [
                extrapolate_mecdf(mecdf_2.cdf, x, threshold=1e-1)
                for x in tqdm(u2, total=N)
            ]
        )
    with Timer("Extrapolating 2D MECDF Power mode: %duration"):
        p2_data_single = np.array(
            [
                extrapolate_mecdf_single(mecdf_2.cdf, x, threshold=5e-2)
                for x in tqdm(u2, total=N)
            ]
        )
    with Timer("Extrapolating 2D MECDF Dicho mode: %duration"):
        p2_data_dicho = np.array(
            [
                extrapolate_mecdf_dicho(mecdf_2.cdf, x, threshold=1e-1)
                for x in tqdm(u2, total=N)
            ]
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
    ax.plot(mcdf_true, p2_data_single, "o", ms=2, alpha=0.3, label="Extrapolated CDF")
    ax.plot(mcdf_true, p2_data_dicho, "o", ms=2, alpha=0.3, label="Extrapolated CDF")
    ax.set_xlabel("True CDF")
    ax.set_ylabel("Empirical CDF")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axline((0, 0), (1, 1), c="k", ls="--")
    plt.show()

    # Plot error wrt threshold
    thresholds = np.geomspace(1e-4, 1e-1, 100)
    errors_single = np.zeros_like(thresholds)
    for i, threshold in tqdm(enumerate(thresholds), total=len(thresholds), smoothing=0):
        mcdf_single = np.array(
            [extrapolate_mecdf_single(mecdf_2.cdf, x, threshold=threshold) for x in u2]
        )
        where = (mcdf_true < threshold) & (mcdf_single > 0)
        errors_single[i] = rmsle(mcdf_true[where], mcdf_single[where])

    fig, ax = plt.subplots()
    ax.plot(thresholds, errors_single)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("RMSLE")
    ax.set_xscale("log")
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
    ax.plot(np.sort(mcdf_true), np.sort(p2_data_dicho), label="Dicho CDF")
    ax.set_xlabel("True CDF")
    ax.set_ylabel("Empirical CDF")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axline((0, 0), (1, 1), c="k", ls="--")
    plt.show()

    errors = [list(error_comparison(rho, 10000, w)) for _ in tqdm(range(4))]
    errors = np.array(errors)

    for i, method in enumerate(["Raw", "Extra", "Single", "Dicho"]):
        print(f"Error {method}: {errors[:, i].mean():.3f} +/- {errors[:, i].std():.3f}")

    # CDF of the CDF
    p = (np.arange(N) + 1) / (N + 1)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(np.sort(mcdf_true), p, label="True CDF")
    ax.plot(np.sort(p2_data), p, label="Empirical CDF")
    ax.plot(np.sort(p2_data_extra), p, label="Extrapolated CDF")
    ax.plot(np.sort(p2_data_single), p, label="Extrapolated CDF (single approach)")
    ax.plot(np.sort(p2_data_dicho), p, label="Extrapolated CDF (dichotomic approach)")
    ax.axline((0.2, 0.2), (0.5, 0.5), c="k", ls="--")
    ax.set_xscale("logit")
    ax.set_yscale("logit")
    ax.grid(alpha=0.5)
    ax.grid(which="minor", alpha=0.5, ls=":")
    ax.legend()

    # Plot a MECDF line
    # 2D MECDF
    rho = 0.8
    N = 50_000
    w = 3
    mu = np.zeros(w)
    with Timer("Generating samples:"):
        sigma = np.array([[rho ** abs(i - j) for i in range(w)] for j in range(w)])
        multi_n = stats.multivariate_normal(mu, sigma)
        x2 = multi_n.rvs(N)
        u2 = stats.norm.cdf(x2)

    with Timer("Computing true CDF:"):
        mcdf_true = multi_n.cdf(x2)

    with Timer("Fitting 2D MECDF:"):
        mecdf_2 = MultivariateMarkovianECDF()
        mecdf_2.fit(u2)

    p2_data = mecdf_2.cdf(u2)
    with Timer("Extrapolating 2D MECDF: %duration"):
        p2_data_extra = np.array(
            [extrapolate_mecdf(mecdf_2.cdf, x, threshold=1e-1) for x in u2]
        )
    with Timer("Extrapolating 2D MECDF Power mode: %duration"):
        p2_data_single = np.array(
            [extrapolate_mecdf_single(mecdf_2.cdf, x, threshold=5e-2) for x in u2]
        )
    with Timer("Extrapolating 2D MECDF Dicho mode: %duration"):
        p2_data_dicho = np.array(
            [extrapolate_mecdf_dicho(mecdf_2.cdf, x, threshold=5e-2) for x in u2]
        )

    idx = np.argmax(np.abs(np.log(mcdf_true) - np.log(p2_data_single)))
    x0 = u2[idx]
    x0 = np.array([0.1, 0.3, 0.1])
    print(f"Max error at x0: {x0}")

    alpha_x = np.min(x0)
    alphas = np.geomspace(min(1e-4, alpha_x), 1, 100)
    points = np.array([1 - (1 - x0) * (1 - alpha) / (1 - alpha_x) for alpha in alphas])
    c_true = multi_n.cdf(stats.norm.ppf(points))
    c_emp = mecdf_2.cdf(points)
    c_extra = np.array(
        [extrapolate_mecdf(mecdf_2.cdf, x, threshold=1e-1) for x in points]
    )
    c_extra_single = np.array(
        [extrapolate_mecdf_single(mecdf_2.cdf, x, threshold=1e-1) for x in points]
    )

    # Dichotomic Approach
    alpha = 1.0
    c = 1.0
    c_emp_dicho = []
    alphas_dicho = []
    while c > 0:
        alphas_dicho.append(alpha)
        c_emp_dicho.append(c)
        alpha /= 2
        c = mecdf_2.cdf(np.array([1 - (1 - x0) * (1 - alpha) / (1 - alpha_x)]))[0]

    while c_emp_dicho[-1] == c_emp_dicho[-2]:
        alphas_dicho.pop()
        c_emp_dicho.pop()

    i = np.digitize(alpha_x, alphas_dicho)
    start = max(i - 2, 0)
    end = min(i + 3, len(alphas_dicho))

    lr = np.polyfit(np.log(alphas_dicho)[start:end], np.log(c_emp_dicho)[start:end], 1)
    c_fit = np.exp(np.polyval(lr, np.log(alphas)))

    fig, ax = plt.subplots()
    ax.plot(alphas, c_true, label="True CDF")
    ax.plot(alphas, c_emp, label="Empirical CDF")
    ax.plot(alphas, c_extra, label="Extrapolated CDF")
    ax.plot(alphas, c_extra_single, label="Extrapolated CDF (single approach)")
    ax.plot(alphas, c_fit, "k--", label="Dichotomic fit CDF")
    ax.plot(
        alphas_dicho, c_emp_dicho, "ko", label="Empirical CDF (dichotomic approach)"
    )
    ax.set_xlabel("alpha")
    ax.set_ylabel("CDF")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.5)
    ax.grid(which="minor", alpha=0.5, ls=":")
    ax.legend()
