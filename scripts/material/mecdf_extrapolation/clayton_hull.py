# -*-coding:utf-8 -*-
"""
@File    :   clayton_hull.py
@Time    :   2024/11/11 14:08:57
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Clayton copula hull
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from core.distributions.mecdf import MultivariateMarkovianECDF
from core.distributions.copulas.clayton_copula import ClaytonCopula
from core.distributions.copulas.independent_copula import IndependentCopula
from utils.timer import Timer
from utils.paths import output


if __name__ == "__main__":
    OUT_DIR = output("Material/MCDF/Clayton-hull")
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    # Plot a MECDF line
    # 2D MECDF
    theta = -0.3
    N = 10_000
    w = 2
    clayton = ClaytonCopula(theta)

    with Timer("Generating samples:"):
        # u2 = clayton.rvs(N, d=w)
        u2 = IndependentCopula().rvs(N, d=w)

    hulls = []
    hull_points_lowers = []
    hull_simplices = []

    u2_remaining = u2.copy()

    m = 100
    for i in tqdm(range(m), total=m, desc="Computing hulls"):
        hull = ConvexHull(u2_remaining)
        hull_points = u2_remaining[hull.vertices]

        count_lower = np.array(
            [
                np.sum(np.all(hull_points <= hull_point[None, :], axis=1))
                for hull_point in hull_points
            ]
        )
        indices_lower = np.where(count_lower <= 1)[0]
        hull_vertices_lower = hull.vertices[count_lower <= 1]
        hull_points_lower = u2_remaining[hull_vertices_lower]

        simplices = np.array(
            [
                u2_remaining[simplex]
                for simplex in hull.simplices
                if np.all(np.isin(simplex, hull_vertices_lower))
            ]
        )

        hulls.append(hull)
        hull_points_lowers.append(hull_points_lower)
        hull_simplices.append(simplices)

        u2_remaining = np.ascontiguousarray(
            np.delete(u2_remaining, hull_vertices_lower, axis=0)
        )
    print(f"Remaining points: {len(u2_remaining)}")

    if w == 2:
        fig, ax = plt.subplots()
        ax.scatter(*u2.T, s=1, alpha=0.3, c="k")
        hull_points_lower = hull_points_lowers[0]
        ax.scatter(*hull_points_lower.T, s=10, c="r")
        fig.savefig(OUT_DIR / "clayton-hull.png")

    CMAP = plt.get_cmap("Spectral")
    if w == 3:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # ax.scatter(*u2.T, s=1)
        for i, simplices in enumerate(hull_simplices):
            color = CMAP(i / (len(hull_simplices) - 1))
            for simplex in simplices:
                simplex = np.append(simplex, [simplex[0]], axis=0)
                ax.plot(simplex[:, 0], simplex[:, 1], simplex[:, 2], c=color)

    if w == 2:
        fig, ax = plt.subplots()
        # ax.scatter(*u2.T, s=1)
        for i, simplices in enumerate(hull_simplices):
            color = CMAP(i / (len(hull_simplices) - 1))
            for simplex in simplices:
                simplex = np.append(simplex, [simplex[0]], axis=0)
                ax.plot(simplex[:, 0], simplex[:, 1], c=color)

    fig, ax = plt.subplots()
    ax.plot([len(hull_points_lower) for hull_points_lower in hull_points_lowers])
    ax.set_xlabel("Hull layer")
    ax.set_ylabel("Number of points")
    plt.show()

    fig, ax = plt.subplots(w - 1, w - 1, squeeze=False)
    for i in range(w - 1):
        for j in range(i, w - 1):
            ax[j, i].scatter(u2[:, i], u2[:, j + 1], s=1)
            ax[j, i].set_xlabel(f"X{i}")
            ax[j, i].set_ylabel(f"X{j+1}")

    plt.show()
