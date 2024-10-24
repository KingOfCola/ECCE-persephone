# -*-coding:utf-8 -*-
"""
@File    :   multi_bernoulli.py
@Time    :   2024/10/24 11:11:48
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Multi-Level Bernoulli distribution helper functions
"""

import numpy as np
from scipy.optimize import minimize

from core.distributions.base.dist import HarmonicDistribution, DiscreteDistributionError

from core.mathematics.functions import selu, sigmoid, narctan
from core.mathematics.harmonics import harmonics_valuation


class HarmonicMultiBernoulli(HarmonicDistribution):
    def __init__(
        self,
        kernel: str = "sigmoid",
        mus: float | None = None,
        n_levels: int | None = None,
        n_harmonics: int | None = None,
        period: float = 1.0,
    ):
        """Bernoulli harmonic distribution.

        Parameters
        ----------
        kernel : str, optional
            Kernel function used in the distribution. Should be either 'arctan' or 'sigmoid', by default 'sigmoid'.
        mu : float array, optional
            Location parameter, by default None. It should be a float array of shape (2*n_harm + 1,).
            Can't be specified if n_harmonics is provided.
        n_harmonics : int, optional
            Number of harmonics to consider, by default None. Can't be specified if mu is provided.
        period : float, optional
            Period of the harmonics, by default 1.0.
        """
        super().__init__(period)
        self._kernel = None
        self._kernel_func = None

        self.kernel = kernel
        self.fit_summary = None

        if mus is not None:
            if n_harmonics is not None or n_levels is not None:
                raise ValueError(
                    "The number of harmonics and levels can't be specified if mus is provided."
                )
            self.mus = mus
            self.n_harmonics = (mus.shape[1] - 1) // 2
            self.n_levels = mus.shape[0]
        else:
            if n_harmonics is None or n_levels is None:
                raise ValueError(
                    "The number of harmonics and levels should be specified if mus is not provided."
                )
            self.n_harmonics = n_harmonics
            self.n_levels = n_levels
            self.mus = None

    @property
    def kernel(self):
        return self._kernel

    @kernel.setter
    def kernel(self, value):
        match value:
            case "arctan":
                self._kernel_func = narctan
            case "sigmoid":
                self._kernel_func = sigmoid
            case _:
                raise ValueError("The kernel should be either 'arctan' or 'sigmoid'.")

        self._kernel = value

    @property
    def mus(self):
        return self._mus

    @mus.setter
    def mus(self, value):
        if value is None:
            self._mus = None
            return

        if value.ndim != 2:
            raise ValueError("The mus should be a 2D array.")
        HarmonicMultiBernoulli._check_harmonics_shape(value[0, :])
        self._mus = np.array(value)
        self.n_harmonics = (value.shape[1] - 1) // 2
        self.n_levels = value.shape[0]

    def cdf(self, t: float, x: float) -> float:
        """Cumulative distribution function.

        Parameters
        ----------
        t : float-like
            Timepoint at which the CDF is evaluated.
        x : float-like
            The value at which the CDF is evaluated.

        Returns
        -------
        float-like
            The value of the CDF at x.
        """
        x = np.clip(np.int_(np.floor(np.array(x))), -1, self.n_levels)
        p = self.param_valuation(t)
        cp = np.cumsum(p, axis=0)
        cp = np.concatenate((np.zeros((1, len(t))), cp), axis=0)
        return cp[x + 1, np.arange(len(t))]

    def pdf(self, t: float, x: float) -> float:
        """Probability density function.

        Parameters
        ----------
        t : float-like
            Timepoint at which the PDF is evaluated.
        x : float-like
            The value at which the PDF is evaluated.

        Returns
        -------
        float-like
            The value of the PDF at x.
        """
        x = np.array(x)
        self._check_values_validity(x)
        p = self.param_valuation(t)
        return p[x, np.arange(len(t))]

    def ppf(self, t: float, q: float) -> float:
        """Percent point function.

        Parameters
        ----------
        t : float-like
            Timepoint at which the ppf is evaluated.
        q : float-like
            The quantile at which the ppf is evaluated. Should be between 0 and 1.

        Returns
        -------
        float-like
            The value of the ppf at q.
        """
        raise DiscreteDistributionError(
            "The method ppf is not available for discrete distributions."
        )

    def rvs(self, t: float) -> float:
        """Random variates.

        Parameters
        ----------
        t : float-like
            Timepoint at which the random variate is generated.

        Returns
        -------
        float-like
            The random variate generated.
        """
        p = self.param_valuation(t)
        cp = np.cumsum(p, axis=0)
        return (np.random.rand(1, *t.shape) < cp).sum(axis=0)

    def fit(self, t: float, x: float):
        """Fit the distribution to the data.

        Parameters
        ----------
        t : float-like
            Timepoints at which the data is measured.
        data : float-like
            Data to fit the distribution to.
        """
        x = np.array(x, dtype=int)
        self._check_values_validity(x)
        summary = HarmonicMultiBernoulli._fit_harmonics(
            t / self.period, x, self.n_harmonics, self.n_levels, self._kernel_func
        )
        self.mus = summary.mus
        self.fit_summary = summary

    def param_valuation(self, t: float) -> list:
        """Compute the actual value of the parameters for each timepoint.

        Parameters
        ----------
        t : float-like
            Timepoint at which the parameters should be evaluated.

        Returns
        -------
        mu : array-like
            Actual values of the parameter p of the Bernoulli distribution for each timepoint.
        """
        return HarmonicMultiBernoulli._param_valuation(
            t, self.mus, self.period, self._kernel_func
        )

    @staticmethod
    def _param_valuation(
        t: np.ndarray, mus: np.ndarray, period: float, kernel_func: callable
    ) -> np.ndarray:
        """Compute the actual value of the parameters for each timepoint.

        Parameters
        ----------
        t : array-like
            Timepoints at which the parameters should be evaluated.
        mus : array-like
            Harmonics of the distribution.
        period : float
            Period of the harmonics.
        kernel_func : callable
            Kernel function of the distribution.

        Returns
        -------
        array-like
            Actual values of the parameters for each timepoint.
        """
        n = t.shape[0]
        mus_derivative = np.array(
            [
                harmonics_valuation(mus[i, :], t=t, period=period)
                for i in range(mus.shape[0])
            ]
        )
        mus_derivative[1:, :] = selu(mus_derivative[1:, :])

        mus_infinite = np.cumsum(mus_derivative, axis=0)
        cp = kernel_func(mus_infinite)
        cp = np.concatenate((np.zeros((1, n)), cp, np.ones((1, n))), axis=0)
        return np.diff(cp, axis=0)

    @staticmethod
    def _fit_harmonics(
        t: float, x: float, n_levels: int, n_harmonics: int, kernel_func: callable
    ) -> any:
        """Fit the harmonics to the data.

        Parameters
        ----------
        t : float-like
            Timepoints at which the data is measured.
        x : float-like
            Data to fit the distribution to.
        n_harmonics : int
            Number of harmonics to consider.
        kernel_func : callable
            Kernel function of the distribution

        Returns
        -------
        array-like
            The optimal fit of the harmonics.
        """
        mus0 = np.zeros((n_levels, 2 * n_harmonics + 1))
        mus0_flat = mus0.flatten()
        mus_opt = minimize(
            lambda mu: -HarmonicMultiBernoulli.log_likelihhod(
                t, x, mu.reshape((n_levels, 2 * n_harmonics + 1)), kernel_func
            ),
            mus0_flat,
        )
        mus_opt.mus = mus_opt.x.reshape((n_levels, 2 * n_harmonics + 1))
        return mus_opt

    @staticmethod
    def log_likelihhod(
        t: float, x: float, mus: np.ndarray, kernel_func: callable
    ) -> float:
        """Log-likelihood of the distribution.

        Parameters
        ----------
        t : float-like
            Timepoints at which the data is measured.
        x : float-like
            Data to fit the distribution to.
        mu : list
            Harmonics of the distribution.
        kernel_func : callable
            Kernel function of the distribution

        Returns
        -------
        float-like
            The log-likelihood of the distribution.
        """
        n = t.shape[0]

        p = HarmonicMultiBernoulli._param_valuation(t, mus, 1.0, kernel_func)
        return np.sum(np.log(p[x, np.arange(n)]))

    def _isfit(self) -> bool:
        return super()._isfit() and self.mus is not None

    def _check_values_validity(self, x: float):
        """Check if the values are valid for the distribution.

        Valid values are either 0 or 1.

        Parameters
        ----------
        x : float-like
            The value to check.

        Raises
        ------
        ValueError
            If the value is not valid.
        """
        values_valid = np.arange(self.n_levels + 1)
        if not np.all(np.isin(x, values_valid)):
            raise ValueError(
                f"The value should be integer and between 0 and the number of levels {self.n_levels}."
            )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    mus = np.array([[0.2, 0.1, 0.1, 0.1, 0.1], [0.8, 0.1, 0.1, 0.1, 0.1]])

    bernoulli = HarmonicMultiBernoulli(mus=mus, kernel="sigmoid")
    t = np.linspace(0, 5, 1000)
    t1 = np.linspace(0, 1, 1001, endpoint=True)

    # Tests the PDF
    fig, ax = plt.subplots()
    ax.plot(t, bernoulli.pdf(t, 0), label="PDF at x=0")
    ax.plot(t, bernoulli.pdf(t, 1), label="PDF at x=1")
    ax.plot(t, bernoulli.pdf(t, 2), label="PDF at x=2")
    ax.set_xlabel("Time")
    ax.set_ylabel("Density")
    ax.legend()
    plt.show()

    # Tests the random variates generation
    fig, axes = plt.subplots(2, height_ratios=[2, 1], sharex=True)
    axes[0].plot(
        t % 1, bernoulli.rvs(t) + 0.05 * np.random.randn(t.shape[0]), "o", markersize=2
    )
    axes[0].set_ylabel("Random variate")
    for i in range(bernoulli.n_levels + 1):
        axes[1].fill_between(
            t1, bernoulli.cdf(t1, i - 1), bernoulli.cdf(t1, i), label=f"CDF {i}"
        )
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("CDF")
    axes[1].set_ylim(0, 1.1)
    plt.show()

    # Tests the fit the distribution to the data
    p = np.array(
        [
            0.25 * (np.sin(2 * np.pi * t + 1)) + 0.3,
            0.2 * (np.cos(2 * np.pi * t + 0.8)) + 0.7,
        ]
    )
    x = np.sum(np.random.rand(1, *t.shape) > p, axis=0)
    bernoulli_to_fit = HarmonicMultiBernoulli(
        kernel="sigmoid", n_harmonics=2, n_levels=2
    )
    bernoulli_to_fit.fit(t, x)

    fig, axes = plt.subplots(2, height_ratios=[2, 1], sharex=True)
    axes[0].plot(t % 1, x + 0.05 * np.random.randn(t.shape[0]), "o", markersize=2)
    axes[0].set_ylabel("Random variate")
    for i in range(bernoulli_to_fit.n_levels + 1):
        axes[1].fill_between(
            t,
            bernoulli_to_fit.cdf(t, i - 1),
            bernoulli_to_fit.cdf(t, i),
            label=f"CDF {i}",
            alpha=0.5,
        )
        axes[1].plot(
            t,
            p[i] if i != bernoulli_to_fit.n_levels else np.ones_like(t),
            label=f"True CDF {i}",
            linestyle="--",
        )
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("CDF")
    axes[1].set_ylim(0, 1.1)
    axes[1].set_xlim(0, 1)
    plt.show()
