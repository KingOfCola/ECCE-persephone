# -*-coding:utf-8 -*-
"""
@File    :   beta_kde.py
@Time    :   2024/11/14 17:56:19
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Beta KDE
"""

import numpy as np
from scipy.stats import beta, norm
from core.distributions.copulas.clayton_copula import ClaytonCopula
from core.distributions.kde.beta_kde import BetaKDE
from core.mathematics.functions import expit, logit
from scripts.tests.markov_mcdf.markov_mcdf import quad_vals
from utils.timer import Timer


def logit_prime(x):
    return 1 / (x * (1 - x))


def logexpit_prime(x):
    return logit_prime(expit(x))


class BetaKDE_:
    def __init__(self, samples, ddof=1.0):
        self.samples = samples
        self.N = samples.shape[0]
        self.d = samples.shape[1]
        self.N_apparent = np.sqrt(self.N) * ddof
        self.ddof = ddof
        self.order = np.zeros_like(samples, dtype=int)
        self.samples_sorted = np.zeros_like(samples)
        self._compute_order()

    def _compute_order(self):
        for i in range(self.samples.shape[1]):
            self.order[:, i] = np.argsort(self.samples[:, i])
            self.samples_sorted[:, i] = self.samples[self.order[:, i], i]

    def pdf(self, x):
        return self.pdf_at_points(x)

    def pdf_at_points(self, x):
        x = np.asarray(x)
        idx = np.zeros_like(x, dtype=int)
        for i in range(self.d):
            idx[i] = np.searchsorted(self.samples_sorted[:, i], x[i])

        tol = 5 / np.sqrt(self.N_apparent)

        mask = (np.abs(self.samples - x[None, :]) < tol).all(axis=1)
        # mask[:] = True

        return np.sum(self.beta_prod(x, self.samples[mask])) / self.N

    def beta_prod(self, x_new, y_data):
        alpha_ = self.N_apparent * y_data + 1
        beta_ = self.N_apparent * (1 - y_data) + 1

        return np.prod(beta.pdf(x_new, alpha_, beta_), axis=1)


class LogitNormalKDE:
    def __init__(self, samples, bandwidth=1.0):
        self.samples = samples
        self.N = samples.shape[0]
        self.d = samples.shape[1]
        self.bandwidth = bandwidth
        self.samples_logit = logit(samples)

    def pdf(self, x):
        return self.pdf_at_points(x)

    def pdf_at_points(self, x):
        x = np.asarray(x)
        x_logit = logit(x)

        tol = 5 * self.bandwidth

        mask = (np.abs(self.samples_logit - x_logit[None, :]) < tol).all(axis=1)
        # mask[:] = True

        return np.sum(self.normal_prod(x_logit, self.samples_logit[mask])) / self.N

    def normal_prod(self, x_new, y_data):
        return np.prod(
            norm.pdf((x_new - y_data) / self.bandwidth)
            * logexpit_prime(x_new)
            / self.bandwidth,
            axis=1,
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tqdm import tqdm
    from itertools import product

    N = 10000
    d = 2

    clayton_copula = ClaytonCopula(0.5, d)
    u = clayton_copula.rvs(N)

    beta_kde = BetaKDE(u, ddof=1)
    beta_kde_npy = BetaKDE_(u, ddof=1)
    logit_normal_kde = LogitNormalKDE(u, bandwidth=20 / np.sqrt(N))

    x = y = np.linspace(1e-3, 1 - 1e-3, 100)
    xx, yy = np.meshgrid(x, y)
    beta_pdf = np.zeros_like(xx)
    logit_pdf = np.zeros_like(xx)
    for i, j in tqdm(
        product(range(len(x)), range(len(y))), total=len(x) * len(y), smoothing=0
    ):
        beta_pdf[i, j] = beta_kde.pdf([xx[i, j], yy[i, j]])
        logit_pdf[i, j] = logit_normal_kde.pdf([xx[i, j], yy[i, j]])

    z = np.stack([xx.flatten(), yy.flatten()], axis=-1)[::15]
    with Timer("Beta PDF"):
        beta_pdf_ = beta_kde.pdf(z)
    with Timer("Beta PDF numpy"):
        beta_pdf_npy_ = np.zeros(len(z))
        for i in range(len(z)):
            beta_pdf_npy_[i] = beta_kde_npy.pdf(z[i])

    pdf_true = clayton_copula.pdf(
        np.stack([xx.flatten(), yy.flatten()], axis=-1)
    ).reshape(xx.shape)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.flatten()
    axes[0].imshow(logit_pdf, extent=(0, 1, 0, 1), origin="lower", vmin=0, vmax=5)
    axes[0].set_title("Logit KDE")
    axes[1].imshow(beta_pdf, extent=(0, 1, 0, 1), origin="lower", vmin=0, vmax=5)
    axes[1].set_title("Beta KDE")
    axes[2].imshow(pdf_true, extent=(0, 1, 0, 1), origin="lower", vmin=0, vmax=5)
    axes[2].set_title("True PDF")
    axes[3].scatter(*u.T, alpha=0.3, s=4)
    axes[3].set_title("Samples")

    plt.show()

    fig, axes = plt.subplots(ncols=2, figsize=(8, 4))
    for ax, name, pdf in zip(axes, ["Beta KDE", "Logit KDE"], [beta_pdf, logit_pdf]):
        ax.scatter(
            pdf_true.flatten(),
            pdf.flatten(),
            s=1,
            # c=((1 - xx.flatten()) ** 2 + (1 - yy.flatten()) ** 2),
        )
        ax.set_xlabel("True PDF")
        ax.set_ylabel(name)
        ax.axline((0, 0), (1, 1), c="k", ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
    plt.show()

    fig, axes = plt.subplots(ncols=2, figsize=(8, 4))
    for ax, name, pdf in zip(axes, ["Beta KDE", "Logit KDE"], [beta_pdf, logit_pdf]):
        where = pdf_true.flatten() < 10
        sns.histplot(
            pdf.flatten()[where] - pdf_true.flatten()[where],
            ax=ax,
            bins=np.linspace(-2, 2, 101),
            stat="density",
        )

    CMAP = plt.get_cmap("Spectral")
    x1s = np.linspace(1e-3, 1 - 1e-3, 11)
    x2s = np.linspace(1e-4, 1 - 1e-4, 101)
    cpdf = np.zeros((len(x1s), len(x2s)))
    ccdf = np.zeros((len(x1s), len(x2s)))

    fig, ax = plt.subplots()
    for i, x1 in enumerate(x1s):
        cpdf[i, :] = np.array(
            [
                beta_kde.conditional_pdf(
                    [x1, x2], cond_idx=np.array([0], dtype="int32")
                )
                for x2 in x2s
            ]
        )
        ax.plot(x2s, cpdf[i, :], c=CMAP(x1))

    ax.set_xlabel("x2")
    ax.set_ylabel("Conditional pdf")
    plt.show()

    fig, ax = plt.subplots()
    for i, x1 in tqdm(enumerate(x1s), total=len(x1s), smoothing=0):
        ccdf[i, :] = np.array(
            [
                beta_kde.conditional_cdf(
                    [x1, x2], cond_idx=np.array([0], dtype="int32")
                )
                for x2 in x2s
            ]
        )
        ax.plot(x2s, ccdf[i, :], c=CMAP(x1))

    ax.set_xlabel("x2")
    ax.set_ylabel("Conditional cdf")
    plt.show()

    for i, x1 in enumerate(x1s):
        print(f"Integral: {np.trapz(cpdf[i, :], x2s)}")
