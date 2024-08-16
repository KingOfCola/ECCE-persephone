# -*-coding:utf-8 -*-
"""
@File    :   rv.py
@Time    :   2024/08/16 11:14:10
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Abstract class for random variables
"""


class Distribution:
    def __init__(self):
        pass

    def cdf(self, x: float) -> float:
        """Cumulative distribution function.

        Parameters
        ----------
        x : float-like
            The value at which the CDF is evaluated.

        Returns
        -------
        float-like
            The value of the CDF at x.
        """
        raise NotImplementedError("The method cdf is not implemented.")

    def pdf(self, x: float) -> float:
        """Probability density function.

        Parameters
        ----------
        x : float-like
            The value at which the PDF is evaluated.

        Returns
        -------
        float-like
            The value of the PDF at x.
        """
        raise NotImplementedError("The method pdf is not implemented.")

    def ppf(self, q: float) -> float:
        """Percent point function.

        Parameters
        ----------
        q : float-like
            The quantile at which the ppf is evaluated. Should be between 0 and 1.

        Returns
        -------
        float-like
            The value of the ppf at q.
        """
        raise NotImplementedError("The method ppf is not implemented.")

    def rvs(self, n: int) -> list:
        """Generate random variates.

        Parameters
        ----------
        n : int
            The number of random variates to generate.

        Returns
        -------
        array-like
            The random variates.
        """
        raise NotImplementedError("The method rvs is not implemented.")

    def fit(self, data: list):
        """Fit the distribution to the data.

        Parameters
        ----------
        data : array-like
            The data to which the distribution is fitted.
        """
        raise NotImplementedError("The method fit is not implemented.")

    def _isfit(self) -> bool:
        """Check if the distribution is fitted.

        Returns
        -------
        bool
            True if the distribution is fitted, False otherwise.
        """
        raise NotImplementedError("The method isfit is not implemented.")


class HarmonicDistribution:
    def __init__(self, period: float):
        self.period = period

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
        raise NotImplementedError("The method cdf is not implemented.")

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
        raise NotImplementedError("The method pdf is not implemented.")

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
        raise NotImplementedError("The method rvs is not implemented.")

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

    def _isfit(self) -> bool:
        """Check if the distribution is fitted.

        Returns
        -------
        bool
            True if the distribution is fitted, False otherwise.
        """
        raise NotImplementedError("The method isfit is not implemented.")

    @staticmethod
    def _check_harmonics_shape(array: list, name: str = "array") -> bool:
        """Check if the shape of the harmonics array is correct.

        Parameters
        ----------
        array : array-like
            The array to check.

        Returns
        -------
        bool
            True if the shape is correct, False otherwise.
        """
        if (len(array) - 1) % 2 != 0:
            raise ValueError(
                f"The number of elements in {name} should be odd to correspond to harmonics parametrization."
            )

    @classmethod
    def _check_values_validity(cls, x: float):
        """Check if the values are valid for the distribution.

        Parameters
        ----------
        x : float-like
            The value to check.
        """
        return


class DiscreteDistributionError(Exception):
    """Exception raised for errors due to non continuous distributions.

    Attributes
    ----------
    message : str
        Explanation of the error.
    """

    def __init__(self, message="The distribution is discrete."):
        self.message = message
        super().__init__(self.message)
