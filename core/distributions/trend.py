# -*-coding:utf-8 -*-
"""
@File    :   trend.py
@Time    :   2024/10/18 13:46:28
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Trend removal transformations
"""

import numpy as np

from core.distributions.dist import TSTransform, DistributionNotFitError
from core.optimization.interpolation import spline_interpolation


class TrendRemoval(TSTransform):
    def __init__(self, step: int = 5):
        super().__init__()
        self.step = step
        self._f = None

    def fit(self, t: np.ndarray, x: np.ndarray):
        # Trend removal
        self._f = spline_interpolation(t, x, step=self.step)

    def transform(self, t: np.ndarray, x: np.ndarray):
        if not self._isfit():
            raise DistributionNotFitError("Trend removal not fitted")
        return x - self._f(t)

    def inverse_transform(self, t: np.ndarray, y: np.ndarray):
        if not self._isfit():
            raise DistributionNotFitError("Trend removal not fitted")
        return y + self._f(t)

    def trend(self, t: np.ndarray):
        if not self._isfit():
            raise DistributionNotFitError("Trend removal not fitted")
        return self._f(t)

    def _isfit(self):
        return self._f is not None
