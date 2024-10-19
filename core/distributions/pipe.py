# -*-coding:utf-8 -*-
"""
@File    :   pipe.py
@Time    :   2024/10/18 14:35:38
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Pipe class
"""

from core.distributions.dist import (
    HarmonicDistribution,
    TSTransform,
    DistributionNotFitError,
)


class Pipe(HarmonicDistribution):
    def __init__(self, *transforms: TSTransform | HarmonicDistribution):
        for i, t in enumerate(transforms[:-1]):
            if not isinstance(t, TSTransform):
                raise TypeError(
                    f"Invalid transform ({i}) {t} should be a TSTransform but got {type(t)}"
                )

        if not isinstance(transforms[-1], HarmonicDistribution):
            raise TypeError(
                f"Invalid transform ({len(transforms)-1}) {transforms[-1]} should be a HarmonicDistribution but got {type(transforms[-1])}"
            )
        super().__init__(transforms[-1].period)
        self._transforms: list[TSTransform] = transforms[:-1]
        self._model: HarmonicDistribution = transforms[-1]

    def transform(self, t: float, x: float) -> float:
        if not self._isfit():
            raise DistributionNotFitError("Pipe not fitted")
        for transform in self._transforms:
            x = transform.transform(t, x)
        return x

    def inverse_transform(self, t: float, y: float) -> float:
        if not self._isfit():
            raise DistributionNotFitError("Pipe not fitted")
        for transform in reversed(self._transforms):
            y = transform.inverse_transform(t, y)
        return y

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
            raise DistributionNotFitError("Pipe not fitted")
        x = self.transform(t, x)
        return self._model.cdf(t, x)

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
            raise DistributionNotFitError("Pipe not fitted")
        x = self.transform(t, x)
        return self._model.pdf(t, x)

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
            raise DistributionNotFitError("Pipe not fitted")
        y = self._model.ppf(t, q)
        return self.inverse_transform(t, y)

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
            raise DistributionNotFitError("Pipe not fitted")
        y = self._model.rvs(t)
        return self.inverse_transform(t, y)

    def fit(self, t: list, x: list):
        """Fit the distribution to the data.

        Parameters
        ----------
        t : array-like
            The timepoints of the data. Should be the same length as data.
        x : array-like
            The data to which the distribution is fitted.
        """
        for transform in self._transforms:
            transform.fit(t, x)
            x = transform.transform(t, x)
        self._model.fit(t, x)

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
            raise DistributionNotFitError("Pipe not fitted")
        return self._model.param_valuation(t)

    def _isfit(self) -> bool:
        """Check if the distribution is fitted.

        Returns
        -------
        bool
            True if the distribution is fitted, False otherwise.
        """
        return self._model._isfit() and all([t._isfit() for t in self._transforms])
