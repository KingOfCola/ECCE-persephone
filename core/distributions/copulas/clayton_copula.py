# -*-coding:utf-8 -*-
"""
@File    :   clayton_copula.py
@Time    :   2024/11/05 16:16:21
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Clayton copula class
"""

import numpy as np
from core.distributions.copulas.archimedean_copulas import ArchimedeanCopula


class ClaytonCopula(ArchimedeanCopula):
    def __init__(self, theta, d: int = 2):
        super().__init__(theta, d)
        if theta < -1 or theta == 0:
            raise ValueError("The parameter theta must be in (-1, inf) - {0}")

    def psi(self, t):
        return (t ** (-self.theta) - 1) / self.theta

    def psi_inv(self, u):
        c = (1 + self.theta * u) ** (-1 / self.theta)
        if self.theta > 0:
            return c
        return np.where(u > -1 / self.theta, 0, c)

    def psi_prime(self, t):
        return -(t ** (-1 - self.theta))

    def psi_inv_nprime(self, u, d):
        p = np.prod(1 + self.theta * np.arange(1, d))

        c = (-1) ** d * p * (1 + self.theta * u) ** (-1 / self.theta - d)
        if self.theta > 0:
            return c
        return np.where(u > -1 / self.theta, 0, c)

    def psi_inv_nprime_inv(self, u, d):
        p = np.prod(1 + self.theta * np.arange(1, d))

        c = (1 / self.theta) * ((u * (-1) ** d / p) ** (-1 / (1 / self.theta + d)) - 1)
        if self.theta > 0:
            return c
        return c
