# -*-coding:utf-8 -*-
"""
@File    :   harmonic_diagnostic_plot.py
@Time    :   2024/12/06 14:11:10
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Diagnostic plots for harmonic distributions for checking time decorrelation.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, beta
import os


from core.distributions.base.dist import HarmonicDistribution


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


class HarmonicDiagnosticPlot:
    def __init__(
        self,
        model: HarmonicDistribution,
        t: np.ndarray,
        x: np.ndarray,
        time_unit: str = None,
        period: float = 1,
    ):
        self.model = model
        self.period = period

        self.t = t
        self.x = x

        self.t_wrapped = t % period
        self.x_cdf = model.cdf(t, x)

        self.kde = TemporalKDE(self.t_wrapped, self.x_cdf)
        self.time_unit = "" if time_unit is None else f" ({time_unit})"

    def return_level_lines(self, ax: plt.Axes = None, **kwargs):
        fs = kwargs.get("fs", 1)
        nmin = kwargs.get("nmin", 10)
        sublevels = kwargs.get("sublevels", [1, 2, 5])

        count = len(self.x)
        rps = return_periods_of_interest(count, fs, nmin, sublevels)
        cmap = plt.get_cmap(kwargs.get("cmap", "turbo"))

        ax.scatter(self.t_wrapped, self.x_cdf, c="black", s=4, edgecolors="none")
        for i, rp in enumerate(rps):
            c = cmap(i / (len(rps) - 1))
            label = f"{rp}{self.time_unit}"
            ax.axhline(rp, c=c, ls="-", lw=0.7, label=label)

    def qqplot(self, ax: plt.Axes = None, bins: int = 100):
        if ax is None:
            ax = plt.gca()

        where = np.isfinite(self.x)
        x = np.sort(self.x[where])
        t = self.t[where]
        n = len(x)
        # t = np.linspace(0, 1, bins, endpoint=False)
        p = np.arange(1, n + 1) / (n + 1)

        tt = np.concatenate([np.random.permutation(t) for _ in range(bins)])

        qq = self.model.ppf(tt, np.tile(p, bins))
        qq = np.sort(qq)
        pp = np.arange(1, len(qq) + 1) / (len(qq) + 1)
        q = np.interp(p, pp, qq)

        xmin = min(np.min(x), np.min(q))
        xmax = max(np.max(x), np.max(q))

        ax.scatter(x, q, s=4)
        ax.set_xlabel("Empirical quantiles")
        ax.set_ylabel("Model quantiles")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(xmin, xmax)
        ax.axline((xmin, xmin), (xmax, xmax), c="k", ls="--", lw=0.7)
        ax.grid(which="major", alpha=0.5, lw=0.7)
        ax.grid(which="minor", alpha=0.2, lw=0.7)
        return ax

    def ppplot(self, ax: plt.Axes = None, **kwargs):
        if ax is None:
            ax = plt.gca()

        where = np.isfinite(self.x)
        x = self.x[where]
        t = self.t[where]

        n = len(x)
        p_emp = np.arange(1, n + 1) / (n + 1)

        p_model = self.model.cdf(t, x)
        p_model = np.sort(p_model)

        ax.scatter(p_emp, p_model, s=4)
        ax.set_xlabel("Empirical probabilities")
        ax.set_ylabel("Model probabilities")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axline((0, 0), (1, 1), c="k", ls="--", lw=0.7)
        ax.grid(which="major", alpha=0.5, lw=0.7)
        ax.grid(which="minor", alpha=0.2, lw=0.7)
        return ax

    def return_period_plot(self, ax: plt.Axes = None, **kwargs):
        if ax is None:
            ax = plt.gca()

        duration_step = np.median(np.diff(self.t))

        where = np.isfinite(self.x)
        x = self.x[where]
        t = self.t[where]

        n = len(x)
        p_emp = np.arange(1, n + 1) / (n + 1)

        p_model = self.model.cdf(t, x)
        p_model = np.sort(p_model)

        rp_emp = duration_step / (1 - p_emp)
        rp_model = duration_step / (1 - p_model)
        rp_min = min(np.min(rp_emp), np.min(rp_model))
        rp_max = max(np.max(rp_emp), np.max(rp_model)) * 1.1

        ax.scatter(rp_emp, rp_model, s=4)
        ax.set_xlabel(f"Empirical return period {self.time_unit}")
        ax.set_ylabel(f"Model return period {self.time_unit}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(rp_min, rp_max)
        ax.set_ylim(rp_min, rp_max)
        ax.axline((rp_min, rp_min), (rp_max, rp_max), c="k", ls="--", lw=0.7)
        ax.grid(which="major", alpha=0.5, lw=0.7)
        ax.grid(which="minor", alpha=0.2, lw=0.7)
        return ax

    def return_level_plot(
        self,
        ax: plt.Axes = None,
        bins: int = 100,
        **kwargs,
    ):
        if ax is None:
            ax = plt.gca()

        duration_step = np.median(np.diff(t))

        alpha = kwargs.get("alpha", 0.05)

        where = np.isfinite(self.x)
        x = np.sort(self.x[where])
        t = self.t[where]
        n = len(x)
        # t = np.linspace(0, 1, bins, endpoint=False)
        ks = np.arange(1, n + 1)
        p_emp = ks / (n + 1)
        p_emp_low = beta.ppf(alpha / 2, ks, n + 1 - ks)
        p_emp_up = beta.ppf(1 - alpha / 2, ks, n + 1 - ks)
        q_emp = x

        tt = np.concatenate([np.random.permutation(t) for _ in range(bins)])

        q_model = self.model.ppf(tt, np.tile(p_emp, bins))
        q_model = np.sort(q_model)
        p_model = np.arange(1, len(q_model) + 1) / (len(q_model) + 1)

        q_model_low = np.interp(p_emp_low, p_model, q_model)
        q_model_up = np.interp(p_emp_up, p_model, q_model)

        rp_emp = duration_step / (1 - p_emp)
        rp_model = duration_step / (1 - p_model)

        xmin = np.min(x)
        xmax = np.max(x)
        rp_min = np.min(rp_emp)
        rp_max = np.max(rp_emp) * 1.1

        ax.scatter(rp_emp, q_emp, s=4)
        ax.plot(rp_model, q_model, c="k", lw=0.7)
        ax.fill_between(rp_emp, q_model_low, q_model_up, color="k", alpha=0.2)
        ax.set_xlabel(f"Return period {self.time_unit}")
        ax.set_ylabel("Return level")
        ax.set_xlim(rp_min, rp_max)
        ax.set_ylim(xmin, xmax)
        ax.set_xscale("log")
        ax.grid(which="major", alpha=0.5, lw=0.7)
        ax.grid(which="minor", alpha=0.2, lw=0.7)
        return ax

    def conditional_pdf_plot(self, ax: plt.Axes = None, **kwargs):
        if ax is None:
            ax = plt.gca()

        t = np.linspace(0, self.period, 101)
        y = np.linspace(0, 1, 101)
        tt, yy = np.meshgrid(t, y)

        conditional_pdf = np.reshape(
            self.kde.conditional_pdf(tt.ravel(), yy.ravel()), tt.shape
        )

        im = ax.imshow(
            conditional_pdf,
            origin="lower",
            extent=[0, 1, 0, 1],
            aspect="auto",
            cmap="RdYlGn",
            vmin=0,
            vmax=2,
        )
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$x$")
        plt.colorbar(im, ax=ax)
        return ax

    def cdf_scatter_plot(self, ax: plt.Axes = None, **kwargs):
        if ax is None:
            ax = plt.gca()

        ax.scatter(self.t, self.x_cdf, c="black", s=1, edgecolors="none")
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$x$")
        return ax

    def diagnostic_plots(self, figsize=(12, 8), **kwargs):
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        self.return_level_lines(axes[0, 0], **kwargs)
        self.qqplot(axes[0, 1], **kwargs)
        self.ppplot(axes[0, 2], **kwargs)
        self.return_period_plot(axes[1, 0], **kwargs)
        self.return_level_plot(axes[1, 1], **kwargs)
        if kwargs.get("conditional_pdf", False):
            self.conditional_pdf_plot(axes[1, 2], **kwargs)

        if kwargs.get("cdf_scatter", True):
            self.cdf_scatter_plot(axes[1, 2], **kwargs)

        fig.tight_layout()

        return fig, axes


def return_periods_of_interest(
    count: int, fs: float = 1.0, nmin: int = 10, sublevels: list = [1, 2, 5]
):
    exp = 10 ** np.floor(np.log10(nmin / fs))
    i = 0
    n = exp

    rps = [n]

    while n * fs < count:
        n = exp * sublevels[i]
        rps.append(n)

        i += 1
        if i >= len(sublevels):
            i = 0
            exp *= 10

    return np.array(rps)
