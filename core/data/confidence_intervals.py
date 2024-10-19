# -*-coding:utf-8 -*-
"""
@File    :   confidence_intervals.py
@Time    :   2024/09/20 18:16:42
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Data class for confidence intervals
"""

import numpy as np


class ConfidenceInterval:
    def __init__(self, shape: tuple):
        self._shape = (shape,) if isinstance(shape, int) else shape
        self._values = np.zeros(shape)
        self._lower = np.zeros(shape)
        self._upper = np.zeros(shape)

    @property
    def shape(self):
        return self._shape

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, values):
        if values.shape != self._shape:
            raise ValueError(
                f"Invalid shape. Expected {self._shape}, got {values.shape}"
            )
        self._values = values

    @property
    def lower(self):
        return self._lower

    @lower.setter
    def lower(self, lower):
        if lower.shape != self._shape:
            raise ValueError(
                f"Invalid shape. Expected {self._shape}, got {lower.shape}"
            )
        self._lower = lower

    @property
    def upper(self):
        return self._upper

    @upper.setter
    def upper(self, upper):
        if upper.shape != self._shape:
            raise ValueError(
                f"Invalid shape. Expected {self._shape}, got {upper.shape}"
            )
        self._upper = upper

    def __getitem__(self, key):
        return self._values[key]

    def __setitem__(self, key, value):
        self._values[key] = value

    def __repr__(self):
        return f"ConfidenceIntervals({self.shape})"

    def __str__(self):
        return f"ConfidenceIntervals({self.shape})"
