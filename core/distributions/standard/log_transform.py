# -*-coding:utf-8 -*-
"""
@File    :   log_transform.py
@Time    :   2024/10/24 14:42:14
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Logarithmic transform
"""

import numpy as np

from core.distributions.base.dist import TSTransform


class LogTransform(TSTransform):
    def __init__(self):
        pass

    def transform(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Transform the data.

        Parameters
        ----------
        t : array-like
            The timepoints of the data
        x : array-like
            The data to transform.

        Returns
        -------
        array-like
            The transformed data.
        """
        tr = np.full_like(x, -np.inf)
        tr[x > 0] = np.log(x[x > 0])
        return tr

    def inverse_transform(self, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Inverse transform the data.

        Parameters
        ----------
        t : array-like
            The timepoints of the data
        y : array-like
            The data to inverse transform.

        Returns
        -------
        array-like
            The inverse-transformed data.
        """
        return np.exp(y)

    def fit(self, t: np.ndarray, x: np.ndarray):
        """Fit the transformation to the data.

        Parameters
        ----------
        t : array-like
            The timepoints of the data
        x : array-like
            The data to which the transformation is fitted.
        """
        return

    def _isfit(self) -> bool:
        """Check if the transformation is fitted.

        Returns
        -------
        bool
            True if the transformation is fitted, False otherwise.
        """
        return True
