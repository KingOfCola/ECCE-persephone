# -*-coding:utf-8 -*-
"""
@File    :   folding_kde.py
@Time    :   2024/11/20 14:51:55
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Tests for folding KDE
"""

import numpy as np
from scipy.stats import beta, norm, gaussian_kde


class WeightedKDE:
    def __init__(self, samples: np.ndarray):
        self.samples = samples
        self.N = samples.shape[0]
        self.d = samples.shape[1]
        self.order = np.zeros_like(samples, dtype=int)
        self.kde = gaussian_kde(samples.T)
        self.norm = norm(loc=0, scale=np.sqrt(np.mean(np.diag(self.kde.covariance))))

    def pdf(self, x):
        x = np.array(x)
        if x.ndim == 1:
            return self.pdf(x[None, :])[0]
        kde = self.kde(x.T)
        weight = np.prod(self.norm.cdf(1 - x) - self.norm.cdf(-x), axis=1)
        return kde / weight

    def conditional_pdf(self, x, cond_idx: np.ndarray):
        x = np.array(x)
        if x.ndim == 1:
            return self.conditional_pdf(x[None, :], cond_idx)[0]
        pdf = self.pdf(x)

        sub_kde_gen = WeightedKDE(self.samples[:, cond_idx])
        sub_pdf = sub_kde_gen.pdf(x[:, cond_idx])

        return pdf / sub_pdf

    def cdf(self, x):
        x = np.array(x)
        if x.ndim == 1:
            return self.cdf(x[None, :])[0]
        kde = self.kde.integrate_box(np.zeros_like(x), x)
        weight = np.prod(self.norm.cdf(1 - x) - self.norm.cdf(-x), axis=0)
        return np.cumsum(kde / weight)


class ReflectedKDE:
    def __init__(self, samples: np.ndarray):
        self.samples = samples
        self.reflected_samples = self.reflect(samples)
        self.N = samples.shape[0]
        self.d = samples.shape[1]
        self.order = np.zeros_like(samples, dtype=int)
        self.kde = gaussian_kde(self.reflected_samples.T)

    @staticmethod
    def reflect(x):
        x = np.array(x)
        for axis in range(x.shape[1]):
            x = ReflectedKDE.reflect_axis(x, axis)
        return x

    @staticmethod
    def reflect_axis(x, axis):
        x = np.array(x)
        x_left = np.copy(x)
        x_left[:, axis] = -x_left[:, axis]
        x_right = np.copy(x)
        x_right[:, axis] = 2 - x_right[:, axis]
        return np.vstack([x_left, x, x_right])

    def pdf(self, x):
        x = np.array(x)
        if x.ndim == 1:
            return self.pdf(x[None, :])[0]
        return self.kde(x) * 3**self.d
