# -*-coding:utf-8 -*-
"""
@File    :   diagnostic_time_condition.py
@Time    :   2024/12/06 10:33:04
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Diagnostic plots for harmonic distributions for checking time decorrelation.
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import os

from core.distributions.sged import HarmonicSGED
from utils.paths import output


class TemporalKDE:
    def __init__(self, t, y):
        self.t = t
        self.y = y
        self._make_kde()

    def pdf(self, t, y):
        X = TemporalKDE._wrap_data(t, y)
        return self.kde(X) * 3

    def conditional_pdf(self, t, y):
        X = TemporalKDE._wrap_data(t, y)
        return self.kde(X) / self.kde_t(t) * 3

    def _make_kde(self):
        X = TemporalKDE._reflect_data(self.t, self.y)
        self.kde = gaussian_kde(X)
        self.kde_t = self.kde.marginal(0)

    @staticmethod
    def _wrap_data(t: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.array([t, y])

    @staticmethod
    def _reflect_data(t: np.ndarray, y: np.ndarray) -> np.ndarray:
        y_reflected = np.concatenate([-y, y, 2 - y])
        t_reflected = np.concatenate([t, t, t])

        return TemporalKDE._wrap_data(t_reflected, y_reflected)


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 12, "text.usetex": True, "font.family": "serif"})
    CMAP = plt.get_cmap("turbo")
    OUT_DIR = output("Material/Harmonics/Diagnostics")
    os.makedirs(OUT_DIR, exist_ok=True)

    model = HarmonicSGED(period=1, n_harmonics=2)
    model._mu = np.array([15.0, -3, 1, 0.0, 0.0])
    model._sigma = np.array([5.0, 0.3, 0.2, 0.4, 0.3])
    model._lamb = np.array([0.6, 0.5, 0.2, 0.3, 0.4])
    model._p = np.array([2.0, 0.3, 0.3, 0.2, 0.3])

    model_3 = HarmonicSGED(period=1, n_harmonics=3)
    model_3._mu = np.array([15.0, -3, 1, 0.0, 0.0, 1.0, 0.0])
    model_3._sigma = np.array([5.0, 1.3, 0.2, 0.4, 0.3, 1.0, 0.0])
    model_3._lamb = np.array([0.6, 0.5, 0.2, 0.3, 0.4, 0.0, 0.3])
    model_3._p = np.array([2.0, 0.3, 0.3, 0.2, 0.3, 0.0, 0.0])

    np.random.seed(42)
    xmin, xmax = 0, 30
    t = np.linspace(0, 1, 1001)
    x = np.linspace(xmin, xmax, 1001)

    # Pdf
    tt, xx = np.meshgrid(t, x)
    pdf = model.pdf(tt.flatten(), xx.flatten()).reshape(tt.shape)

    # Random samples
    n_samples = 10_000
    t_samples = (np.arange(n_samples) + 0.5) / n_samples
    data = model.rvs(t_samples)
    data_3 = model_3.rvs(t_samples)

    # Evaluate CDF at samples with the model with 2 harmonics
    data_cdf = model.cdf(t_samples, data)
    data_3_cdf = model.cdf(t_samples, data_3)

    # KDE
    temporal_kde = TemporalKDE(t_samples, data_cdf)
    temporal_kde_3 = TemporalKDE(t_samples, data_3_cdf)

    x = np.linspace(0, 1, 101)
    y = np.linspace(0, 1, 101)
    xx, yy = np.meshgrid(x, y)
    positions = np.vstack([xx.ravel(), yy.ravel()])
    conditional_pdf = np.reshape(
        temporal_kde.conditional_pdf(xx.ravel(), yy.ravel()), xx.shape
    )
    conditional_pdf_3 = np.reshape(
        temporal_kde_3.conditional_pdf(xx.ravel(), yy.ravel()), xx.shape
    )

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 6), width_ratios=[1, 1, 0.05])
    for ax, pdf, y in zip(
        axes, [conditional_pdf, conditional_pdf_3], [data_cdf, data_3_cdf]
    ):
        im = ax.imshow(
            pdf,
            origin="lower",
            extent=[0, 1, 0, 1],
            aspect="auto",
            cmap="RdYlGn",
            vmin=0,
            vmax=2,
        )
        ax.scatter(t_samples, y, c="black", s=1, edgecolors="none")
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$x$")
        ax.set_title(r"$p(x|t)$")
    fig.colorbar(im, cax=axes[-1], label="Density")
    fig.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "diagnostic_time_condition.png"))

from fastkde import fastKDE

fastKDE.conditional
