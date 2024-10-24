# -*-coding:utf-8 -*-
"""
@File    :   bernoulli.py
@Time    :   2024/08/16 11:11:48
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Bernoulli distribution helper functions
"""

import numpy as np
from scipy.optimize import minimize

from core.distributions.base.dist import HarmonicDistribution, DiscreteDistributionError

from core.mathematics.functions import sigmoid
from core.mathematics.harmonics import harmonics_valuation


class HarmonicBernoulli(HarmonicDistribution):
    def __init__(
        self,
        kernel: str = "sigmoid",
        mu: float | None = None,
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

        if mu is not None:
            if n_harmonics is not None:
                raise ValueError(
                    "The number of harmonics can't be specified if mu is provided."
                )
            self.mu = mu
        else:
            self.n_harmonics = n_harmonics
            self.mu = None

    @property
    def kernel(self):
        return self._kernel

    @kernel.setter
    def kernel(self, value):
        match value:
            case "arctan":
                self._kernel_func = HarmonicBernoulli.narctan
            case "sigmoid":
                self._kernel_func = sigmoid
            case _:
                raise ValueError("The kernel should be either 'arctan' or 'sigmoid'.")

        self._kernel = value

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, value):
        if value is None:
            self._mu = None
            return

        HarmonicBernoulli._check_harmonics_shape(value)
        self._mu = np.array(value)
        self.n_harmonics = (len(value) - 1) // 2

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
        raise DiscreteDistributionError(
            "The method cdf is not available for discrete distributions."
        )

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
        HarmonicBernoulli._check_values_validity(x)
        log_p = harmonics_valuation(self.mu, t=t, period=self.period)
        p = self._kernel_func(log_p)
        return p * (x == 1) + (1 - p) * (x == 0)

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
        log_p = harmonics_valuation(self.mu, t=t, period=self.period)
        p = self._kernel_func(log_p)
        return np.random.rand(*t.shape) < p

    def fit(self, t: float, x: float):
        """Fit the distribution to the data.

        Parameters
        ----------
        t : float-like
            Timepoints at which the data is measured.
        data : float-like
            Data to fit the distribution to.
        """
        HarmonicBernoulli._check_values_validity(x)
        summary = HarmonicBernoulli._fit_harmonics(
            t / self.period, x, self.n_harmonics, self._kernel_func
        )
        self.mu = summary.x
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
        return harmonics_valuation(self.mu, t=t, period=self.period)

    @staticmethod
    def _fit_harmonics(
        t: float, x: float, n_harmonics: int, kernel_func: callable
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
        mu0 = np.zeros(2 * n_harmonics + 1)
        return minimize(
            lambda mu: -HarmonicBernoulli.log_likelihhod(t, x, mu, kernel_func),
            mu0,
        )

    @staticmethod
    def log_likelihhod(t: float, x: float, mu: list, kernel_func: callable) -> float:
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
        log_p = harmonics_valuation(mu, t=t, period=1.0)
        p = kernel_func(log_p)
        return np.sum(np.log(p * (x == 1) + (1 - p) * (x == 0)))

    def _isfit(self) -> bool:
        return super()._isfit() and self.mu is not None

    @classmethod
    def _check_values_validity(cls, x: float):
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
        if ((x != 0) & (x != 1)).any():
            raise ValueError("The value should be either 0 or 1.")

    @staticmethod
    def narctan(x: float) -> float:
        """Normalized arctan kernel function.

        Parameters
        ----------
        x : float-like
            The value at which the kernel is evaluated.

        Returns
        -------
        float-like
            The value of the kernel at x.
        """
        return np.arctan(x) / np.pi + 0.5


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    bernoulli = HarmonicBernoulli(mu=[0.2, 2.0, 0.8, 0.6, 0.8], kernel="sigmoid")
    t = np.linspace(0, 5, 1000)
    t1 = np.linspace(0, 1, 1001, endpoint=True)

    # Tests the PDF
    fig, ax = plt.subplots()
    ax.plot(t, bernoulli.pdf(t, 0), label="PDF at x=0")
    ax.plot(t, bernoulli.pdf(t, 1), label="PDF at x=1")
    ax.set_xlabel("Time")
    ax.set_ylabel("Density")
    ax.legend()
    plt.show()

    # Tests the random variates generation
    fig, ax = plt.subplots()
    ax.plot(
        t % 1, bernoulli.rvs(t) + 0.05 * np.random.randn(t.shape[0]), "o", markersize=2
    )
    ax.plot(t1, bernoulli.pdf(t1, 1), label="PDF")
    ax.set_xlabel("Time")
    ax.set_ylabel("Random variate")
    plt.show()

    # Tests the fit the distribution to the data
    p = 0.3 * (np.sin(2 * np.pi * t) + 1) + 0.3
    x = p > np.random.rand(*t.shape)
    bernoulli_to_fit = HarmonicBernoulli(kernel="sigmoid", n_harmonics=2)
    bernoulli_to_fit.fit(t, x)

    fig, ax = plt.subplots()
    ax.plot(
        t % 1, x + 0.05 * np.random.randn(t.shape[0]), "o", label="Data", markersize=2
    )
    ax.plot(t1, bernoulli_to_fit.pdf(t1, 1), label="Fitted PDF")
    ax.plot(t[t < 1], p[t < 1], label="True PDF")
