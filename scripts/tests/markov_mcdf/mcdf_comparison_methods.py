# -*-coding:utf-8 -*-
"""
@File    :   mcdf_comparison_methods.py
@Time    :   2024/11/26 09:18:32
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Comparison methods for MECDF
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.timer import chrono

STYLES = [
    {"marker": "o", "linestyle": "-", "color": "k", "mfc": "w"},
    {"marker": "s", "linestyle": "--", "color": "k", "mfc": "w"},
    {"marker": "^", "linestyle": "-.", "color": "k", "mfc": "w"},
    {"marker": "D", "linestyle": ":", "color": "k", "mfc": "w"},
    {"marker": "x", "linestyle": "-", "color": "k", "mfc": "w"},
    {"marker": "v", "linestyle": "--", "color": "k", "mfc": "w"},
    {"marker": ">", "linestyle": "-.", "color": "k", "mfc": "w"},
    {"marker": "<", "linestyle": ":", "color": "k", "mfc": "w"},
]


class MECDFProtocol:
    def __init__(self, sample_generator: callable):
        self.sample_generator = sample_generator

    def mecdf(
        self,
        samples: np.ndarray,
        method: callable,
        margs: list = [],
        mkwargs: dict = {},
    ):
        return method(samples)

    def chrono_method(
        self,
        n_samples: int,
        d: int,
        n_trials: int,
        method: callable,
        margs: list = [],
        mkwargs: dict = {},
    ):
        computation_times = np.zeros(n_trials)
        for i in range(n_trials):
            samples = self.sample_generator(n_samples, d)
            mean, _ = chrono(method, 1, samples, *margs, **mkwargs)
            computation_times[i] = mean
        return np.mean(computation_times), (
            np.std(computation_times) if n_trials > 1 else 0
        )

    def compare_methods_single_sample_size(
        self,
        methods: list,
        n_samples: int,
        d: int,
        n_trials: int,
        skip_methods: np.ndarray,
    ):
        computation_times_mean = np.zeros(len(methods))
        computation_times_std = np.zeros(len(methods))
        computation_times_mean[skip_methods] = np.nan
        computation_times_std[skip_methods] = np.nan

        for i, method in enumerate(methods):
            # Skip methods that are marked as NaN. This means the last execution time exceeded the timeout
            if np.isnan(computation_times_mean[i]):
                continue

            if callable(method):
                method, margs, mkwargs = method, [], {}
            elif isinstance(method, tuple):
                if len(method) == 2:
                    method, margs = method
                    mkwargs = {}
                elif len(method) == 3:
                    method, margs, mkwargs = method
                else:
                    raise ValueError("Invalid method tuple")

            computation_times_mean[i], computation_times_std[i] = self.chrono_method(
                n_samples, d, n_trials, method, margs, mkwargs
            )

        return computation_times_mean, computation_times_std

    def compare_methods(
        self,
        methods: list,
        nmin: int = 100,
        nmax: int = 1e6,
        steps_per_decade: int = 3,
        d: int = 2,
        n_trials: int = 1,
        timeout: int = 10,
    ):
        n_samples = (
            10 ** np.arange(np.log10(nmin), np.log10(nmax + 1), 1 / steps_per_decade)
        ).astype(int)

        computation_times_mean = np.zeros((len(n_samples), len(methods)))
        computation_times_std = np.zeros((len(n_samples), len(methods)))
        skip_methods = np.zeros(len(methods), dtype=bool)

        for i, n_sample in tqdm(enumerate(n_samples), total=len(n_samples)):
            computation_times_mean[i], computation_times_std[i] = (
                self.compare_methods_single_sample_size(
                    methods, n_sample, d, n_trials, skip_methods=skip_methods
                )
            )
            skip_methods = np.isnan(computation_times_mean[i]) | (
                computation_times_mean[i] >= timeout
            )

        self.n_samples = n_samples
        self.computation_times_mean = computation_times_mean
        self.computation_times_std = computation_times_std

        return computation_times_mean, computation_times_std

    def plot_computation_times(
        self, ax: plt.Axes = None, method_names: list = None, ci=True
    ):
        if ax is None:
            fig, ax = plt.subplots()

        if method_names is None:
            method_names = [
                f"Method {i}" for i in range(self.computation_times_mean.shape[1])
            ]

        for i in range(self.computation_times_mean.shape[1]):
            if ci:
                ax.errorbar(
                    self.n_samples,
                    self.computation_times_mean[:, i],
                    yerr=self.computation_times_std[:, i],
                    label=method_names[i],
                    **STYLES[i],
                )
            else:
                ax.plot(
                    self.n_samples,
                    self.computation_times_mean[:, i],
                    label=method_names[i],
                    **STYLES[i],
                )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of samples")
        ax.set_ylabel("Computation time (s)")
        ax.legend()

        return ax


if __name__ == "__main__":
    from time import sleep
    from core.distributions.copulas.clayton_copula import ClaytonCopula
    from core.distributions.copulas.gumbel_copula import GumbelCopula

    from core.distributions.mecdf import (
        MultivariateMarkovianECDF,
        NaiveMultivariateECDF,
        MergeSortMultivariateECDF,
    )

    def naive_mecdf(samples):
        mecdf = NaiveMultivariateECDF()
        mecdf.fit(samples)
        return mecdf.cdf(samples)

    def bintree_mecdf(samples):
        mecdf = MultivariateMarkovianECDF()
        mecdf.fit(samples)
        return mecdf.cdf(samples)

    def merge_mecdf(samples):
        mecdf = MergeSortMultivariateECDF()
        mecdf.fit(samples)
        return mecdf.cdf(samples)

    theta = 1.0

    methods = {
        "Naive": naive_mecdf,
        "Bintree": bintree_mecdf,
        "Merge": merge_mecdf,
    }

    np.random.seed(0)
    samples = ClaytonCopula(theta).rvs(10000, 2)
    results = {}
    for name, method in methods.items():
        print(f"Running {name}")
        results[name] = method(samples)

    for name_1, result_1 in results.items():
        for name_2, result_2 in results.items():
            if name_1 < name_2:
                print(f"{name_1} vs {name_2}: {np.allclose(result_1, result_2)}")

    n_methods = len(methods)
    fig, axes = plt.subplots(
        n_methods, n_methods, figsize=(4 * n_methods, 4 * n_methods)
    )
    for i, (name_1, result_1) in enumerate(results.items()):
        for j, (name_2, result_2) in enumerate(results.items()):
            ax = axes[i, j]
            if i == j:
                ax.text(0.5, 0.5, name_1, ha="center", va="center", fontsize=20)
                ax.axis("off")
            else:
                ax.scatter(result_2, result_1)
                ax.text(
                    0.1,
                    0.9,
                    f"Abs error: {np.max(np.abs(result_1 - result_2)):.2g}",
                    ha="center",
                    va="center",
                    fontsize=12,
                )

    protocol = MECDFProtocol(lambda n, d: ClaytonCopula(theta).rvs(n, d))

    computation_times_mean, computation_times_std = protocol.compare_methods(
        methods.values(),
        nmin=100,
        nmax=1e5,
        steps_per_decade=3,
        d=2,
        n_trials=3,
        timeout=5.0,
    )

    fig, ax = plt.subplots()
    protocol.plot_computation_times(ax, list(methods.keys()), ci=True)
    plt.show()
