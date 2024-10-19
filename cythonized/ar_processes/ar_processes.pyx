"""
This module contains the Cython implementation of Auto-Regressive processes.
"""

cimport cython
import numpy as np
from libc.math cimport sqrt



cdef wrap_windows(double *windows, double *x, int w, int n):
    """
    Wrap the windows around the time series x.
    """
    cdef int i, j, pos
    pos = 0

    for i in range(n):
        for j in range(w):
            windows[pos] = x[i + j]
            pos += 1

# =========================================================================== #
# AR(1) process functions
# =========================================================================== #
# Gaussian AR(1) process
# --------------------------------------------------------------------------- #
cdef apply_gaussian_ar_process(double* x, double rho, int n):
    """
    Apply the AR(1) process to the time series x.
    """
    cdef int i
    cdef double alpha = sqrt(1 - rho**2)

    for i in range(1, n):
        x[i] = rho * x[i - 1] + alpha * x[i]


def gaussian_ar_process(int n, double rho):
    """
    Generate a time series of length n from an AR(1) process with parameters alpha, beta, and sigma.
    """
    cdef double[::1] x
    
    x = np.random.normal(loc=0.0, scale=1.0, size=n)
    apply_gaussian_ar_process(&x[0], rho, n)

    return np.array(x)

def gaussian_ar_process_by_windows(int n, double rho, int w):
    """
    Generate a time series of length n from an AR(1) process with parameters alpha, beta, and sigma.
    """
    cdef double[::1] x
    cdef double[:, ::1] windows

    x = gaussian_ar_process(n + w - 1, rho)
    windows = np.zeros((n, w))

    wrap_windows(&windows[0][0], &x[0], w, n)

    return np.array(windows)

# =========================================================================== #
# GARCH process functions
# --------------------------------------------------------------------------- #
cdef apply_garch_process(double *x, double *h, double *alpha, double *beta, int n, int q, int p):
    """
    Generate a GARCH(1, 1) process
    """
    cdef int k, i

    for k in range(1, n):
        hh = alpha[0]
        for i in range(1, min(k, q)):
            hh += alpha[i] * x[k - i]**2
        for i in range(min(k - 1, p)):
            hh += beta[i] * h[k - i - 1]

        h[k] = hh
        x[k] *= sqrt(hh)
    
def garch_process(int n, double[::1] alpha, double[::1] beta):
    """
    Generate a time series of length n from a GARCH(1, 1) process with parameters alpha, beta, and sigma.
    """
    cdef double[::1] x, h
    cdef int i

    x = np.random.normal(loc=0.0, scale=1.0, size=n)
    h = np.zeros(n)
    h[0] = 1.0

    apply_garch_process(&x[0], &h[0], &alpha[0], &beta[0], n, len(alpha), len(beta))

    return np.array(x)

def garch_process_by_windows(int n, double[::1] alpha, double[::1] beta, int w):
    """
    Generate a time series of length n from a GARCH(1, 1) process with parameters alpha, beta, and sigma.
    """
    cdef double[::1] x
    cdef double[:, ::1] windows

    x = garch_process(n + w - 1, alpha, beta)
    windows = np.zeros((n, w))

    wrap_windows(&windows[0][0], &x[0], w, n)

    return np.array(windows)
