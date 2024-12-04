# -*-coding:utf-8 -*-
"""
@File    :   folding_kde.py
@Time    :   2024/11/20 14:51:55
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Tests for folding KDE
"""

import numpy as np
from scipy.stats import beta, norm, gaussian_kde
from core.distributions.copulas.clayton_copula import ClaytonCopula
from core.distributions.kde.beta_kde import BetaKDE
from core.mathematics.functions import expit, logit
from scripts.tests.markov_mcdf.markov_mcdf import quad_vals
from utils.timer import Timer


class WeightedKDE:
    def __init__(self, samples: np.ndarray):
        self.samples = samples
        self.N = samples.shape[0]
        self.d = samples.shape[1]
        self.order = np.zeros_like(samples, dtype=int)
        self.kde = gaussian_kde(samples.T)
        self.norm = norm(loc=0, scale=np.sqrt(np.mean(np.diag(self.kde.covariance))))

    def pdf(self, x):
        x = np.array(x)
        if x.ndim == 1:
            return self.pdf(x[None, :])[0]
        kde = self.kde(x)
        weight = np.prod(self.norm.cdf(1 - x) - self.norm.cdf(-x), axis=0)
        return kde / weight


class ReflectedKDE:
    def __init__(self, samples: np.ndarray):
        self.samples = samples
        self.reflected_samples = self.reflect(samples)
        self.N = samples.shape[0]
        self.d = samples.shape[1]
        self.order = np.zeros_like(samples, dtype=int)
        self.kde = gaussian_kde(self.reflected_samples.T)

    @staticmethod
    def reflect(x):
        x = np.array(x)
        for axis in range(x.shape[1]):
            x = ReflectedKDE.reflect_axis(x, axis)
        return x

    @staticmethod
    def reflect_axis(x, axis):
        x = np.array(x)
        x_left = np.copy(x)
        x_left[:, axis] = -x_left[:, axis]
        x_right = np.copy(x)
        x_right[:, axis] = 2 - x_right[:, axis]
        return np.vstack([x_left, x, x_right])

    def pdf(self, x):
        x = np.array(x)
        if x.ndim == 1:
            return self.pdf(x[None, :])[0]
        return self.kde(x) * 3**self.d


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tqdm import tqdm
    from itertools import product

    N = 10000
    d = 2

    # RV Generation
    clayton_copula = ClaytonCopula(0.2, d)
    u = clayton_copula.rvs(N)

    # KDE Fitting
    with Timer("Weighted KDE Fit"):
        weighted_kde = WeightedKDE(u)
    with Timer("Reflected KDE Fit"):
        reflected_kde = ReflectedKDE(u)

    # PDF Evaluation on the grid
    x = y = np.linspace(0, 1, 100)
    xx, yy = np.meshgrid(x, y)
    z = np.stack([xx.flatten(), yy.flatten()], axis=-1)

    with Timer("Weighted KDE PDF"):
        weighted_pdf = weighted_kde.pdf(z.T).reshape(xx.shape)
    with Timer("Reflected KDE PDF"):
        reflected_pdf = reflected_kde.pdf(z.T).reshape(xx.shape)
    with Timer("True PDF"):
        pdf_true = clayton_copula.pdf(z).reshape(xx.shape)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.flatten()
    axes[0].imshow(weighted_pdf, extent=(0, 1, 0, 1), origin="lower", vmin=0, vmax=5)
    axes[0].set_title("Weighted KDE")
    axes[1].imshow(pdf_true, extent=(0, 1, 0, 1), origin="lower", vmin=0, vmax=5)
    axes[1].set_title("True KDE")
    axes[2].imshow(reflected_pdf, extent=(0, 1, 0, 1), origin="lower", vmin=0, vmax=5)
    axes[2].set_title("Reflected KDE")
    axes[3].scatter(*u.T, alpha=0.3, s=4)
    axes[3].set_xlim(0, 1)
    axes[3].set_ylim(0, 1)
    axes[3].set_title("Samples")

    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), squeeze=False)
    for ax, name, pdf in zip(
        axes.flatten(), ["Weighted KDE", "Reflected KDE"], [weighted_pdf, reflected_pdf]
    ):
        ax.scatter(
            pdf_true.flatten(),
            pdf.flatten(),
            s=1,
            # c=((1 - xx.flatten()) ** 2 + (1 - yy.flatten()) ** 2),
        )
        ax.set_xlabel("True PDF")
        ax.set_ylabel(name)
        ax.axline((0.5, 0.5), (1, 1), c="k", ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="major", alpha=0.5)
        ax.grid(True, which="minor", alpha=0.2)

    plt.show()
