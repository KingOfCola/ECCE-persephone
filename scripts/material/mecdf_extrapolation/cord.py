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
    theta = 0.3
    N = 10_000
    w = 2
    clayton = ClaytonCopula(theta)

    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    alphas = np.geomspace(1e-4, 1, 100)

    x0 = 0.3
    points = np.stack([1 - (1 - alphas) * (1 - x0), alphas], axis=1)

    xx, yy = np.meshgrid(x, y)
    cdf = clayton.cdf(np.stack([xx.ravel(), yy.ravel()], axis=1)).reshape(xx.shape)
    cdf_cord = clayton.cdf(points)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(cdf, extent=(0, 1, 0, 1), origin="lower")
    axes[0].plot([x0, 1], [0, 1], color="red")
    axes[0].set_title("Clayton copula")
    axes[1].plot(alphas, cdf_cord)
    axes[1].set_title("Clayton copula CDF on the cord")
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("CDF")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlim(1e-4, 1)
    axes[1].set_ylim(1e-4, 1)
    axes[1].grid(which="major", alpha=0.7)
    axes[1].grid(which="minor", alpha=0.3)
    plt.show()

    fig.savefig(OUT_DIR / "clayton_cord.png")
