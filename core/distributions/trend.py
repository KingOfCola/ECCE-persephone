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

from core.distributions.base.dist import TSTransform, DistributionNotFitError
from core.optimization.interpolation import spline_interpolation


class TrendRemoval(TSTransform):
    def __init__(self, step: int = 5):
        super().__init__()
        self.step = step
        self._f = None
        self._tlims = None

    def fit(self, t: np.ndarray, x: np.ndarray):
        # Trend removal
        where = np.isfinite(x)
        t = t[where]
        x = x[where]
        self._tlims = (t.min(), t.max())
        self._f = spline_interpolation(t, x, step=self.step)

    def transform(self, t: np.ndarray, x: np.ndarray):
        if not self._isfit():
            raise DistributionNotFitError("Trend removal not fitted")
        t = np.clip(t, *self._tlims)
        return x - self._f(t)

    def inverse_transform(self, t: np.ndarray, y: np.ndarray):
        if not self._isfit():
            raise DistributionNotFitError("Trend removal not fitted")
        t = np.clip(t, *self._tlims)
        return y + self._f(t)

    def trend(self, t: np.ndarray):
        if not self._isfit():
            raise DistributionNotFitError("Trend removal not fitted")
        t = np.clip(t, *self._tlims)
        return self._f(t)

    def _isfit(self):
        return self._f is not None
