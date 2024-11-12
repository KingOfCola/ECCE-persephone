# -*-coding:utf-8 -*-
"""
@File    :   archimedean_copulas.py
@Time    :   2024/11/05 16:16:10
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Base class for Archimedean copulas
"""


import numpy as np
from core.distributions.base.dist import Distribution


class ArchimedeanCopula(Distribution):
    def __init__(self, theta, d: int = 2):
        super().__init__()
        self.theta = theta
        self.d = d

    def psi(self, u):
        """Archimedean generator function."""
        raise NotImplementedError

    def psi_inv(self, t):
        """Inverse of the Archimedean generator function."""
        raise NotImplementedError

    def psi_prime(self, u):
        """Derivative of the Archimedean generator function."""
        raise NotImplementedError

    def psi_inv_nprime(self, t, d: int):
        """nth derivative of the inverse of the Archimedean generator function."""
        raise NotImplementedError

    def psi_inv_nprime_inv(self, t, d: int):
        """Inverse of the nth derivative of the inverse of the Archimedean generator function."""
        raise NotImplementedError

    def rvs(self, n, d: int = None):
        v = np.random.uniform(size=(n, self.d if d is None else d))
        return self.ppf(v)

    def ppf(self, q: np.ndarray):
        if q.ndim == 1:
            return self.ppf(q[None, :])[0, :]

        if q.ndim > 2:
            raise ValueError("q must be 1D or 2D")

        d = q.shape[1]

        # In the case d is one, this is equivalent to a uniform distribution
        if d == 1:
            return q

        # Generate RVs for the copula with n-1 elements
        u = self.ppf(q[:, :-1])

        # Generate the last element quantile conditioned on the previous elements
        v = q[:, -1]

        # Calculate the value of the last element based on the previous elements and the quantile
        c = np.sum(self.psi(u), axis=1)
        c_ninv = self.psi_inv_nprime(c, d - 1)
        ud = self.psi_inv(self.psi_inv_nprime_inv(v * c_ninv, d - 1) - c)

        # Return the generated random variates
        return np.hstack([u, ud[:, None]])

    def cdf(self, u: np.ndarray):
        if u.ndim == 1:
            return self.psi_inv(np.sum(self.psi(u), axis=0))
        elif u.ndim == 2:
            return self.psi_inv(np.sum(self.psi(u), axis=1))
        else:
            raise ValueError("u must be 1D or 2D")

    def pdf(self, u: np.ndarray):
        if u.ndim > 2:
            raise ValueError("u must be 1D or 2D")

        is_2d = u.ndim == 2
        if not is_2d:
            u = u.reshape(1, -1)
        d = u.shape[1]

        u_psi_prime = self.psi_prime(u)
        u_psi = self.psi(u)

        p = np.prod(u_psi_prime, axis=1) * self.psi_inv_nprime(np.sum(u_psi, axis=1), d)
        if not is_2d:
            return p[0]
        else:
            return p

    def logpdf(self, u):
        return np.log(self.pdf(u))
