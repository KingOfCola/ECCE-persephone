# -*-coding:utf-8 -*-
"""
@File    :   clayton.py
@Time    :   2024/11/04 18:43:09
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Clayton copula
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from core.data.confidence_intervals import ConfidenceInterval
from core.distributions.copulas.clayton_copula import ClaytonCopula
from core.distributions.mecdf import MultivariateMarkovianECDF
from scripts.material.mecdf_extrapolation.gaussian import (
    extrapolate_mecdf_dicho,
    extrapolate_mecdf_single,
    extrapolate_mecdf,
    rmsle,
)
from utils.timer import Timer


def rmsrpe(y_true, y_pred):
    return np.sqrt(np.mean((1 / y_true - 1 / y_pred) ** 2))


def error_comparison(theta, N, w, threshold=1e-1, min_count=10):
    clayton = ClaytonCopula(theta)
    u = clayton.rvs(N, w)
    mcdf_true = clayton.cdf(u)

    mecdf = MultivariateMarkovianECDF()
    mecdf.fit(u)
    p_data = mecdf.cdf(u)

    where = p_data > 0
    error_mecdf = rmsle(p_data[where], mcdf_true[where])

    errors = [error_mecdf]
    for method in [
        lambda x: extrapolate_mecdf(mecdf.cdf, x, threshold=threshold),
        lambda x: extrapolate_mecdf_single(mecdf.cdf, x, threshold=threshold),
        lambda x: extrapolate_mecdf_dicho(mecdf.cdf, x, threshold, min_count=min_count),
    ]:
        p_data_extra = np.array([method(x) for x in u])
        error = rmsle(p_data_extra[where], mcdf_true[where])
        errors.append(error)

    return errors


if __name__ == "__main__":
    theta = 1.0
    alpha = 0.8
    z = np.linspace(0, 1, 101)
    u, v = np.meshgrid(z, z)
    ur = u.ravel()
    vr = v.ravel()
    uv = np.array([ur, vr, alpha * np.ones_like(ur), alpha * np.ones_like(ur)]).T
    uv = uv[:, :3]

    clayton = ClaytonCopula(theta)
    cdf = clayton.cdf(uv).reshape(u.shape)
    pdf = clayton.pdf(uv).reshape(u.shape)
    rvs = clayton.rvs(10_000_000, d=3)
    rvs = rvs[np.abs(rvs[:, 2] - alpha) < 5e-2]
    rvs_hist = np.histogram2d(
        rvs[:, 0], rvs[:, 1], bins=len(z), range=[[0, 1], [0, 1]], density=True
    )[0]

    fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
    for ax, name, stat, lim in zip(
        axes, ["CDF", "PDF", "Empirical hist."], [cdf, pdf, rvs_hist], [alpha, 5, 5]
    ):
        levels = np.linspace(0, lim, 101)
        cs = ax.contourf(u, v, stat, levels=levels, cmap="viridis")
        fig.colorbar(cs, ax=ax)
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_title(name)

    plt.show()

    # cdf on a line
    alphas = np.linspace(0, 1, 1001)
    x0 = np.ones(5) * 0.0
    x0[0] = 0
    ones = np.ones_like(x0)
    x = x0[None, :] * (1 - alphas[:, None]) + ones[None, :] * alphas[:, None]
    cdf = clayton.cdf(x)

    fig, ax = plt.subplots()
    ax.plot(alphas, cdf)
    ax.set_xlabel("alpha")
    ax.set_ylabel("CDF")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.5)
    ax.grid(which="minor", alpha=0.5, ls=":")
    plt.show()

    # Comparing errors
    errors = [list(error_comparison(2.0, 10000, 3)) for _ in tqdm(range(10))]
    errors = np.array(errors)

    for i, method in enumerate(["Raw", "Extra", "Single", "Dicho"]):
        print(f"Error {method}: {errors[:, i].mean():.3f} +/- {errors[:, i].std():.3f}")

    # Plot a MECDF line
    # 2D MECDF
    theta = 2.0
    N = 100_000
    w = 3
    clayton = ClaytonCopula(theta)

    with Timer("Generating samples:"):
        q2 = np.random.uniform(size=(N, w))
        u2 = clayton.ppf(q2)
        # u2 = clayton.rvs(N, d=w)

    with Timer("Computing true CDF:"):
        mcdf_true = clayton.cdf(u2)

    with Timer("Fitting 2D MECDF:"):
        mecdf_2 = MultivariateMarkovianECDF()
        mecdf_2.fit(u2[:, :3])

    with Timer("Computing empirical MECDF:"):
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
            [
                extrapolate_mecdf_dicho(mecdf_2.cdf, x, threshold=1e-1, min_count=10)
                for x in u2
            ]
        )

    fig, ax = plt.subplots()
    ax.plot(mcdf_true, p2_data, ".", label="Empirical")
    ax.plot(mcdf_true, p2_data_extra, ".", label="Extrapolated")
    ax.plot(mcdf_true, p2_data_single, ".", label="Extrapolated single")
    ax.plot(mcdf_true, p2_data_dicho, ".", label="Extrapolated dicho")
    ax.axline((0.5, 0.5), slope=1, color="black", ls="--")
    ax.set_xlabel("True CDF")
    ax.set_ylabel("Empirical CDF")
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.5)
    ax.grid(which="minor", alpha=0.5, ls=":")
    plt.show()

    abs_errors = np.abs(np.log(mcdf_true) - np.log(p2_data_dicho))
    idx = np.argmax(abs_errors)
    x0 = u2[idx]
    # x0 = np.array([0.1, 0.1, 0.1])
    print(
        f"Max error at x0: {x0} : {abs_errors[idx]:.2f} ({p2_data_dicho[idx]:.2g} instead of {mcdf_true[idx]:.2g})"
    )

    alpha_x = np.min(x0)
    alphas = np.geomspace(min(1e-4, alpha_x), 1, 100)
    points = np.array([1 - (1 - x0) * (1 - alpha) / (1 - alpha_x) for alpha in alphas])
    c_true = clayton.cdf(points)
    c_emp = mecdf_2.cdf(points)
    c_extra = np.array(
        [extrapolate_mecdf(mecdf_2.cdf, x, threshold=1e-1) for x in points]
    )
    c_extra_single = np.array(
        [extrapolate_mecdf_single(mecdf_2.cdf, x, threshold=1e-1) for x in points]
    )
    c_extra_dicho = np.array(
        [
            extrapolate_mecdf_dicho(mecdf_2.cdf, x, threshold=1e-1, min_count=10)
            for x in points
        ]
    )
    c_emp_ci = ConfidenceInterval(c_emp.shape)
    c_emp_ci.lower = c_emp - 1.96 * np.sqrt(c_emp * (1 - c_emp) / N)
    c_emp_ci.upper = c_emp + 1.96 * np.sqrt(c_emp * (1 - c_emp) / N)
    c_emp_ci.values = c_emp

    fig, ax = plt.subplots()
    ax.plot(alphas, c_true, label="True", c="k")
    ax.plot(alphas, c_emp, label="Empirical")
    ax.fill_between(alphas, c_emp_ci.lower, c_emp_ci.upper, alpha=0.5, fc="C0")
    ax.plot(alphas, c_extra, label="Extrapolated")
    ax.plot(alphas, c_extra_single, label="Extrapolated single")
    ax.plot(alphas, c_extra_dicho, label="Extrapolated dicho")
    ax.axvline(alpha_x, color="black", ls="--")
    ax.set_xlabel("alpha")
    ax.set_ylabel("CDF")
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.5)
    ax.grid(which="minor", alpha=0.5, ls=":")
    plt.show()
