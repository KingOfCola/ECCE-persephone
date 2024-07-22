# -*-coding:utf-8 -*-
"""
@File    :   correlations.py
@Time    :   2024/07/09 15:22:46
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   This script contains the functions for computing the correlations.
"""

import numpy as np


def autocorrelation(x: np.ndarray) -> np.ndarray:
    """
    Compute the autocorrelation of a signal.

    Uses the FFT of the zero-padded signal to compute the autocorrelation.

    Code from https://stackoverflow.com/a/51168178/25980698

    Parameters
    ----------
    x : np.ndarray
        The signal to compute the autocorrelation of.

    Returns
    -------
    np.ndarray
        The autocorrelation of the signal.
    """
    n = len(x)

    # Centering the signal
    x_centered = x - np.mean(x)
    var = np.var(x)

    # Pads the signal with zeros to the next power of 2 above 2n-1
    ext_size = 2 * n - 1
    fsize = 2 ** np.ceil(np.log2(ext_size)).astype("int")

    # Computes the convolution in the frequency domain
    fft_x = np.fft.fft(x_centered, fsize)
    fft_conv = fft_x.conjugate() * fft_x
    convolved = np.fft.ifft(fft_conv).real
    corr = convolved / var / n

    return corr[:n]
