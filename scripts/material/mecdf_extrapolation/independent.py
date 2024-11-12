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
from sklearn.linear_model import LinearRegression
from core.distributions.mecdf import MultivariateMarkovianECDF
from core.distributions.copulas.clayton_copula import ClaytonCopula
from core.distributions.copulas.independent_copula import IndependentCopula


def cord_points(x0: np.ndarray, hs: np.ndarray) -> np.ndarray:
    """
    Compute the copula points from the marginal points
    """
    h_x = np.min(x0)
    points = 1 - (1 - x0[:, None]) * (1 - hs[None, :]) / (1 - h_x)
    return points


def transform_coordinates(hs: np.ndarray, order: int = 3) -> np.ndarray:
    """
    Transform the coordinates of the copula points
    """
    X = np.array([np.log(hs), np.ones_like(hs), hs]).T
    return X[:, :order]


def fit_lower_tail(h: np.ndarray, c: np.ndarray, order=3) -> LinearRegression:
    """
    Fit the lower tail of the CDF using a polynomial of order `order`
    """
    X = transform_coordinates(h, order=order)
    y = np.log(c)
    reg = LinearRegression(fit_intercept=False).fit(X, y)
    return reg


def predict_lower_tail(
    cdf: callable, x0: np.ndarray, hmin: float, hmax: float, order: int = 3
) -> float:
    """
    Predicts the lower tail of the CDF using a polynomial of order `order`
    """
    hs = np.geomspace(hmin, hmax, 11)

    # Get the points on the cord passing through x0
    points = cord_points(x0, hs)
    h_x = np.min(x0)

    # Compute the CDF
    c = cdf(points)

    # Fit the lower tail
    reg = fit_lower_tail(hs, c, order=order)
    X_x = transform_coordinates(np.array([h_x]), order=order)
    return reg.predict(X_x)[0]


if __name__ == "__main__":
    x0 = np.array([0.1, 0.2, 0.3])
    print(f"x0: {x0}")

    alpha_x = np.min(x0)
    x_proj = 1 - (1 - x0) / (1 - alpha_x)
    plim = 1e-6
    order = 3

    alphas = np.geomspace(min(plim, alpha_x), 1, 101)
    where_eff = (1e-4 <= alphas) & (alphas <= 1e-2)
    X = transform_coordinates(np.array(alphas), order=order)
    points = np.array([1 - (1 - x0) * (1 - alpha) / (1 - alpha_x) for alpha in alphas])

    # Independent copula
    copula_ind = IndependentCopula()
    c_true_ind = copula_ind.cdf(points)

    K = np.sum(x_proj <= 1e-4)
    x_proj_non_null = x_proj[x_proj > 1e-4]
    log_c = np.sum(np.log(x_proj_non_null))
    beta_1 = np.sum(1 / x_proj_non_null - 1)

    lh = np.log(alphas)
    c_1 = K * lh
    c_2 = K * lh + log_c
    c_3 = K * lh + log_c + beta_1 * alphas

    # Linear fitting
    # Fit the lower tail
    reg = fit_lower_tail(alphas[where_eff], c_true_ind[where_eff], order=order)
    c_x = reg.predict(X)

    fig, ax = plt.subplots()
    ax.plot(alphas, c_true_ind, label="True", c="r")
    ax.plot(alphas, np.exp(c_1), label=r"$K\cdot\log\left(h\right)$", c="k")
    ax.plot(
        alphas,
        np.exp(c_2),
        label=r"$K\cdot\log\left(h\right)+\sum_{i=1}^{d-K}\log\left(u_i\right)$",
        c="g",
    )
    ax.plot(
        alphas,
        np.exp(c_3),
        label=r"$K\cdot\log\left(h\right)+\sum_{i=1}^{d-K}\log\left(u_i\right)+h\cdot \sum_{i=1}^{d-K}\left(\frac{1}{u_i}-1\right)$",
        c="b",
    )
    ax.plot(alphas, np.exp(c_x), label=f"Fit (order {order})", c="orange")
    ax.axvline(alpha_x, color="black", ls="--")
    ax.set_xlabel("$h$")
    ax.set_ylabel("CDF")
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.5)
    ax.grid(which="minor", alpha=0.5, ls=":")
    plt.show()

    # Clayton copula
    theta = 0.5
    copula_clayton = ClaytonCopula(theta)
    c_true_clayton = copula_clayton.cdf(points)

    K = np.sum(x_proj <= 1e-4)
    x_proj_non_null = x_proj[x_proj > 1e-4]
    const_clay = np.sum(x_proj_non_null ** (-theta) - 1) - K + 1

    lh = np.log(alphas)
    c_1_clay = lh
    c_2_clay = lh - np.log(K) / theta
    c_3_clay = lh - np.log(K) / theta - const_clay / theta * alphas**theta / K

    # Linear fitting
    # Fit the lower tail
    reg_clay = fit_lower_tail(alphas[where_eff], c_true_clayton[where_eff], order=order)
    c_x_clay = reg_clay.predict(X)

    fig, ax = plt.subplots()
    ax.plot(alphas, c_true_clayton, label="True", c="r")
    ax.plot(alphas, np.exp(c_1_clay), label=r"$K\cdot\log\left(h\right)$", c="k")
    ax.plot(
        alphas,
        np.exp(c_2_clay),
        label=r"$K\cdot\log\left(h\right)-\frac{\log(K)}{\theta}$",
        c="g",
    )
    ax.plot(
        alphas,
        np.exp(c_3_clay),
        label=r"$K\cdot\log\left(h\right)-\frac{\log(K)}{\theta}-\frac{h^\theta}{K\theta}\left(\sum_{i=1}^{d-K}\left(u_i^\theta-1\right)+K-1\right)$",
        c="b",
    )
    ax.plot(alphas, np.exp(c_x_clay), label=f"Fit (order {order})", c="orange")
    ax.axvline(alpha_x, color="black", ls="--")
    ax.set_xlabel("$h$")
    ax.set_ylabel("CDF")
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.5)
    ax.grid(which="minor", alpha=0.5, ls=":")
    ax.set_ylim(plim, 1)
    plt.show()
