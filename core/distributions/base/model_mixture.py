# -*-coding:utf-8 -*-
"""
@File    :   model_mixture.py
@Time    :   2024/10/24 13:58:06
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Mixture by model choice
"""
import numpy as np
from core.distributions.base.dist import (
    HarmonicDistribution,
    TSTransform,
    DistributionNotFitError,
)


class HarmonicModelMixture(HarmonicDistribution):
    def __init__(self, period: float):
        super().__init__(period=period)
        self._models: list[HarmonicDistribution] = []
        self._weights: HarmonicDistribution = None

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
            The value of the CDF at x.
        """
        if not self._isfit():
            raise DistributionNotFitError("Model mixture not fitted")
        ps = np.zeros(t.shape)
        for k, model in enumerate(self._models):
            ps += model.cdf(t, x) * self._weights.pdf(t, k)
        return ps

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
            The value of the PDF at x.
        """
        if not self._isfit():
            raise DistributionNotFitError("Model mixture not fitted")
        ps = np.zeros((len(self._models),) + t.shape)
        for k, model in enumerate(self._models):
            ps += model.pdf(t, x) * self._weights.pdf(t, k)
        return ps

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
        raise NotImplementedError("The method ppf is not implemented.")

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
            raise DistributionNotFitError("Model mixture not fitted")
        ks = self._weights.rvs(t)
        return [self._models[k].rvs(t_) for k, t_ in zip(ks, t)]

    def fit(self, t: list, x: list):
        """Fit the distribution to the data.

        Parameters
        ----------
        t : array-like
            The timepoints of the data. Should be the same length as data.
        x : array-like
            The data to which the distribution is fitted.
        """
        raise NotImplementedError("The method fit is not implemented.")

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
        if not self._isfit():
            raise DistributionNotFitError("Model mixture not fitted")
        return None

    def _isfit(self) -> bool:
        """Check if the distribution is fitted.

        Returns
        -------
        bool
            True if the distribution is fitted, False otherwise.
        """
        return (
            all([model._isfit() for model in self._models])
            and self._weights is not None
        )
