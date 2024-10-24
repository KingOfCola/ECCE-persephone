# -*-coding:utf-8 -*-
"""
@File    :   model_mixture.py
@Time    :   2024/10/24 13:58:06
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Mixture by model choice. The choice is performed according to thresholds
"""
import numpy as np
from core.distributions.base.dist import (
    HarmonicDistribution,
    TSTransform,
    DistributionNotFitError,
)

from core.distributions.standard.multi_bernoulli import HarmonicMultiBernoulli


class HarmonicThresholdModelMixture(HarmonicDistribution):
    def __init__(
        self,
        period: float,
        thresholds: np.ndarray,
        models: list[HarmonicDistribution],
        n_harnonics: int = 2,
    ):
        super().__init__(period=period)
        self._models: list[HarmonicDistribution] = models
        self._thresholds: np.ndarray = thresholds
        self._levels = len(self._thresholds) + 1
        self._weights: HarmonicDistribution = HarmonicMultiBernoulli(
            period=period, n_harnonics=n_harnonics, n_levels=self._levels
        )
        self._n_harnonics = n_harnonics

        if len(self._models) != len(self._thresholds) + 1:
            raise ValueError(
                f"Invalid number of thresholds {len(self._thresholds)} for {len(self._models)} models"
            )

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
        ks = self._weights.ppf(t, q)
        ks_cdf_low = self._weights.cdf(t, ks)
        ks_cdf_up = self._weights.cdf(t, ks + 1)
        return self._models[ks].ppf(t, (q - ks_cdf_low) / (ks_cdf_up - ks_cdf_low))

    def fit(self, t: list, x: list):
        """Fit the distribution to the data.

        Parameters
        ----------
        t : array-like
            The timepoints of the data. Should be the same length as data.
        x : array-like
            The data to which the distribution is fitted.
        """
        ks = np.digitize(x, self._thresholds)
        self._weights.fit(t, ks)

        for k in range(self._levels):
            mask = ks == k
            self._models[k].fit(t[mask], x[mask])

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
