# -*-coding:utf-8 -*-
"""
@File    :   clayton_extrapolation.py
@Time    :   2024/11/07 14:47:55
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Clayton copula extrapolation
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.linear_model import LinearRegression
from core.distributions.mecdf import MultivariateMarkovianECDF
from core.distributions.copulas.clayton_copula import ClaytonCopula
from core.distributions.copulas.independent_copula import IndependentCopula
from utils.timer import Timer


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
    theta = -0.3
    N = 100_000
    w = 3
    clayton = ClaytonCopula(theta)

    with Timer("Generating samples:"):
        u2 = clayton.rvs(N, d=w)

    with Timer("Computing true CDF:"):
        mcdf_true = clayton.cdf(u2)

    with Timer("Fitting 2D MECDF:"):
        mecdf_2 = MultivariateMarkovianECDF()
        mecdf_2.fit(u2[:, :3])

    with Timer("Computing empirical MECDF:"):
        p2_data = mecdf_2.cdf(u2)
    with Timer("Extrapolating 2D MECDF Dicho mode:"):
        p2_data_fit = np.array(
            [
                extrapolate_mecdf_fit(mecdf_2.cdf, x, hmin=1e-3, hmax=1e-1, order=3)
                for x in u2
            ]
        )
    with Timer("Extrapolating 2D MECDF Hetero mode:"):
        p2_data_hetero_fit = np.array(
            [
                extrapolate_mecdf_fit_hetero(
                    mecdf_2.cdf, x, hmin=1e-3, hmax=1e-1, order=3
                )
                for x in u2
            ]
        )

    with Timer("Computing convex hull:"):
        hull = ConvexHull(u2[:100000, :])
        hull_points = u2[hull.vertices]

    with Timer("Extracting relevant points:"):
        p2_data_hull = mecdf_2.cdf(hull_points)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(*u2.T, s=1)
    for simplex in hull.simplices:
        simplex = np.append(simplex, simplex[0])
        ax.plot(u2[simplex, 0], u2[simplex, 1], u2[simplex, 2], "k-")

    fig, ax = plt.subplots(2, 2)
    for i in range(2):
        for j in range(i, 2):
            ax[i, j].scatter(u2[:, i], u2[:, j + 1], s=1)
            ax[i, j].set_xlabel(f"X{i}")
            ax[i, j].set_ylabel(f"X{j+1}")

    fig, ax = plt.subplots()
    ax.plot(mcdf_true, p2_data, ".", label="Empirical")
    ax.plot(mcdf_true, p2_data_fit, ".", label="Extrapolated Fit")
    ax.plot(mcdf_true, p2_data_hetero_fit, ".", label="Heteroschedastic Fit")
    ax.axline((0.5, 0.5), slope=1, color="black", ls="--")
    ax.set_xlabel("True CDF")
    ax.set_ylabel("Empirical CDF")
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.5)
    ax.grid(which="minor", alpha=0.5, ls=":")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(mcdf_true, p2_data / mcdf_true, ".", label="Empirical", ms=2)
    ax.plot(mcdf_true, p2_data_fit / mcdf_true, ".", label="Extrapolated Fit", ms=2)
    ax.plot(
        mcdf_true,
        p2_data_hetero_fit / mcdf_true,
        ".",
        label="Heteroschedastic Fit",
        ms=2,
    )
    ax.axhline(0, color="black", ls="--")
    ax.set_xlabel("True CDF")
    ax.set_ylabel("Empirical CDF")
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.5)
    ax.grid(which="minor", alpha=0.5, ls=":")
    plt.show()

    error_fit_emp = np.sqrt(np.mean((np.log(p2_data) - np.log(mcdf_true)) ** 2))
    error_fit_extra = np.sqrt(np.mean((np.log(p2_data_fit) - np.log(mcdf_true)) ** 2))
    error_fit_hetero = np.sqrt(
        np.mean((np.log(p2_data_hetero_fit) - np.log(mcdf_true)) ** 2)
    )

    print(f"Error empirical: {error_fit_emp:.2e}")
    print(f"Error extrapolated: {error_fit_extra:.2e}")
    print(f"Error heteroschedastic: {error_fit_hetero:.2e}")
