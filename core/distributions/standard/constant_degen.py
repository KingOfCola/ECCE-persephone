# -*-coding:utf-8 -*-
"""
@File    :   constant.py
@Time    :   2024/10/24 14:35:59
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Degenerate constant distribution
"""

import numpy as np
from core.distributions.base.dist import (
    Distribution,
    HarmonicDistribution,
    DistributionNotFitError,
)


class HarmonicDegenerateConstant(HarmonicDistribution):
    def __init__(self, period: float = 1.0, value: float = None):
        super().__init__(period)
        self._value = value

    def cdf(self, t: float, x: float) -> float:
        """Cumulative distribution function.

        Parameters
        ----------
        t : float-like
            The time at which the CDF is evaluated.
        x : float-like
            The value at which the CDF is evaluated.

        Returns
        -------
        float-like
            The value of the CDF at x. Always 1, for the degenerate constant distribution.
        """
        if not self._isfit():
            raise DistributionNotFitError("Degenerate constant distribution not fitted")
        return np.ones_like(x)

    def pdf(self, t: float, x: float) -> float:
        """Probability density function.

        Parameters
        ----------
        t : float-like
            The time at which the PDF is evaluated.
        x : float-like
            The value at which the PDF is evaluated.

        Returns
        -------
        float-like
            The value of the PDF at x. Always 1, for the degenerate constant distribution.
        """
        if not self._isfit():
            raise DistributionNotFitError("Degenerate constant distribution not fitted")
        return np.ones_like(x)

    def ppf(self, t: float, q: float) -> float:
        """Percent point function.

        Parameters
        ----------
        t : float-like
            The time at which the ppf is evaluated.
        q : float-like
            The quantile at which the ppf is evaluated. Should be between 0 and 1.

        Returns
        -------
        float-like
            The value of the ppf at q.
        """
        if not self._isfit():
            raise DistributionNotFitError("Degenerate constant distribution not fitted")
        return np.full_like(q, self._value)

    def rvs(self, t: list) -> list:
        """Generate random variates.

        Parameters
        ----------
        t : array-like
            The times at which the random variates are generated.

        Returns
        -------
        array-like
            The random variates.
        """
        if not self._isfit():
            raise DistributionNotFitError("Degenerate constant distribution not fitted")
        return np.full_like(t, self._value)

    def fit(self, t: list, x: list):
        """Fit the distribution to the data.

        Parameters
        ----------
        t : array-like
            The timepoints of the data. Should be the same length as data.
        x : array-like
            The data to which the distribution is fitted.
        """
        self.value = np.nanmean(x)

    def param_valuation(self, t: float) -> list:
        """Compute the actual value of the parameters for each timepoint.

        Parameters
        ----------
        t : float-like
            Timepoint at which the parameters should be evaluated.

        Returns
        -------
        array-like
            Actual values of the parameters for each timepoint.
        """
        return np.full_like(t, self._value)

    def _isfit(self) -> bool:
        """Check if the distribution is fitted.

        Returns
        -------
        bool
            True if the distribution is fitted, False otherwise.
        """
        return self._value is not None
