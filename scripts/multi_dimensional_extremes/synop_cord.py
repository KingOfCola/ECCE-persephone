# -*-coding:utf-8 -*-
"""
@File    :   clayton_extrapolation.py
@Time    :   2024/11/07 14:47:55
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Clayton copula extrapolation
"""
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
from core.optimization.mecdf import cdf_of_mcdf, find_effective_dof
from core.distributions.copulas.clayton_copula import ClaytonCopula
from core.distributions.copulas.independent_copula import IndependentCopula
from utils.arrays import sliding_windows
from utils.loaders.synop_loader import load_fit_synop
from utils.timer import Timer
from plots.annual import month_xaxis


def cord_points(x0: np.ndarray, hs: np.ndarray) -> np.ndarray:
    """
    Compute the copula points from the marginal points
    """
    h_x = np.min(x0)
    points = 1 - (1 - x0[None, :]) * (1 - hs[:, None]) / (1 - h_x)
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
    y = np.full_like(c, -np.inf)
    y[c > 0] = np.log(c[c > 0])

    where = np.isfinite(y)
    X = X[where]
    y = y[where]

    reg = LinearRegression(fit_intercept=False).fit(X, y)
    return reg


def fit_lower_tail_hetero(h: np.ndarray, c: np.ndarray, order=3) -> LinearRegression:
    """
    Fit the lower tail of the CDF using a polynomial of order `order`
    """
    X = transform_coordinates(h, order=order)
    y = np.full_like(c, -np.inf)
    y[c > 0] = np.log(c[c > 0])

    where = np.isfinite(y)
    c = c[where]
    X = X[where]
    y = y[where]
    gamma = (1 - c) / c
    X_n = X / gamma[:, None]
    y_n = y / gamma

    reg = LinearRegression(fit_intercept=False).fit(X_n, y_n)
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
    return np.exp(reg.predict(X_x)[0])


def predict_lower_tail_hetero(
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
    reg = fit_lower_tail_hetero(hs, c, order=order)
    X_x = transform_coordinates(np.array([h_x]), order=order)
    return np.exp(reg.predict(X_x)[0])


def extrapolate_mecdf_fit(cdf, x, hmin=1e-4, hmax=1e-2, order: int = 3):
    """
    Extrapolate the MECDF using a linear regression
    """
    if np.min(x) > hmax:
        return cdf(x)
    return predict_lower_tail(cdf, x, hmin, hmax, order=order)


def extrapolate_mecdf_fit_hetero(cdf, x, hmin=1e-4, hmax=1e-2, order: int = 3):
    """
    Extrapolate the MECDF using a linear regression
    """
    if np.min(x) > hmax:
        return cdf(x)
    return predict_lower_tail_hetero(cdf, x, hmin, hmax, order=order)


if __name__ == "__main__":
    # Plot a MECDF line
    # 2D MECDF
    METRIC = "t_MAX"
    ts_data = load_fit_synop(METRIC)
    STATION = ts_data.labels[23]
    w = 3
    DAYS_IN_YEAR = 366

    with Timer("Generating samples:"):
        u2 = sliding_windows(1 - ts_data.data[STATION].values, w=w)
        N = len(u2)

    with Timer("Computing true CDF:"):
        mcdf_true = np.arange(1, N + 1) / (N + 1)

    with Timer("Fitting 2D MECDF:"):
        mecdf_2 = MultivariateMarkovianECDF()
        mecdf_2.fit(u2)

    with Timer("Computing empirical MECDF:"):
        p2_data = mecdf_2.cdf(u2)
    with Timer("Extrapolating 2D MECDF Dicho mode:"):
        p2_data_fit = np.array(
            [
                extrapolate_mecdf_fit(mecdf_2.cdf, x, hmin=5e-2, hmax=2e-1, order=3)
                for x in u2
            ]
        )
    with Timer("Extrapolating 2D MECDF Hetero mode:"):
        p2_data_fit_hetero = np.array(
            [
                extrapolate_mecdf_fit_hetero(
                    mecdf_2.cdf, x, hmin=5e-2, hmax=2e-1, order=3
                )
                for x in u2
            ]
        )

    edof_emp = find_effective_dof(mcdf_true, np.sort(p2_data))
    edof_fit = find_effective_dof(mcdf_true, np.sort(p2_data_fit))
    edof_fit_hetero = find_effective_dof(mcdf_true, np.sort(p2_data_fit_hetero))

    cdf_emp = cdf_of_mcdf(p2_data, edof_emp)
    cdf_fit = cdf_of_mcdf(p2_data_fit, edof_fit)
    cdf_fit_hetero = cdf_of_mcdf(p2_data_fit_hetero, edof_fit_hetero)
    idx_min = np.argmin(cdf_fit_hetero)
    print(ts_data.time[idx_min], 1 / DAYS_IN_YEAR / cdf_fit_hetero[idx_min])

    q = np.arange(1, N + 1) / (N + 1)
    fig, ax = plt.subplots()
    ax.plot(q, np.sort(cdf_emp), ".", label="Empirical")
    ax.plot(q, np.sort(cdf_fit), ".", label="Extrapolated Fit")
    ax.plot(q, np.sort(cdf_fit_hetero), ".", label="Extrapolated Fit Hetero")
    ax.axline((0.5, 0.5), slope=1, color="black", ls="--")
    ax.set_xlabel("Quantile")
    ax.set_ylabel("Empirical CDF")
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.5)
    ax.grid(which="minor", alpha=0.5, ls=":")
    plt.show()

    YEAR = 2003
    fig, ax = plt.subplots()
    ax.plot(
        (ts_data.yearf[: len(p2_data)] - YEAR) * DAYS_IN_YEAR,
        1 / (DAYS_IN_YEAR * cdf_emp),
        ".",
        label="Empirical",
    )
    ax.plot(
        (ts_data.yearf[: len(p2_data)] - YEAR) * DAYS_IN_YEAR,
        1 / (DAYS_IN_YEAR * cdf_fit),
        ".",
        label="Extrapolated Fit",
    )
    ax.plot(
        (ts_data.yearf[: len(p2_data)] - YEAR) * DAYS_IN_YEAR,
        1 / (DAYS_IN_YEAR * cdf_fit_hetero),
        ".",
        label="Extrapolated Fit Hetero",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Return Period")
    ax.legend()
    month_xaxis(ax)
    ax.set_xlim(0, DAYS_IN_YEAR)
    ax.set_yscale("log")

    CMAP = plt.get_cmap("Spectral")

    possible_ks = np.where(p2_data < 1e-3)[0]
    ks = np.random.choice(possible_ks, 2, replace=False)
    fig, ax = plt.subplots()
    for i, k in enumerate(ks):
        color = CMAP(i / (len(ks) - 1))
        x0 = u2[k]
        # x0 = np.ones(w) *1e-3
        hs = np.geomspace(1e-6, 1, 101)
        points = cord_points(x0, hs)
        c_emp = mecdf_2.cdf(points)
        c_emp_fit = np.array(
            [
                extrapolate_mecdf_fit(mecdf_2.cdf, x, hmin=1e-3, hmax=1e-1, order=3)
                for x in points
            ]
        )
        c_emp_fit_hetero = np.array(
            [
                extrapolate_mecdf_fit_hetero(
                    mecdf_2.cdf, x, hmin=1e-3, hmax=1e-1, order=2
                )
                for x in points
            ]
        )
        h_x = np.min(x0)
        c_emp_x = extrapolate_mecdf_fit(mecdf_2.cdf, x0, hmin=1e-3, hmax=1e-1, order=3)
        c_emp_x2 = extrapolate_mecdf_fit(mecdf_2.cdf, x0, hmin=1e-3, hmax=1e-1, order=3)
        c_emp_x2_hetero = extrapolate_mecdf_fit_hetero(
            mecdf_2.cdf, x0, hmin=1e-3, hmax=1e-1, order=2
        )

        ax.plot(hs, c_emp, c=color, marker="o", ls="none", ms=2, label=f"$k$={k}")
        ax.plot(hs, c_emp_fit, c=color, ls="--")
        ax.plot(hs, c_emp_fit_hetero, c=color, ls=":")
        ax.scatter(
            [h_x, h_x, h_x],
            [c_emp_x, c_emp_x2, c_emp_x2_hetero],
            c=[color],
            marker="*",
            s=100,
        )
    # ax.plot(mcdf_true, p2_data_fit, ".", label="Extrapolated Fit")
    ax.axline((0.5, 0.5), slope=1, color="black", ls="--")
    ax.set_xlabel("h")
    ax.set_ylabel("Empirical CDF")
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.5)
    ax.grid(which="minor", alpha=0.5, ls=":")
    plt.show()
