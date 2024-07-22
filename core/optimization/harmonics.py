# -*-coding:utf-8 -*-
"""
@File    :   harmonics.py
@Time    :   2024/07/10 16:03:36
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   This script contains the harmonics parameter valuation function.
"""


import numpy as np


def harmonics_parameter_valuation(
    params: np.ndarray,
    t: np.ndarray,
    n_harmonics: int,
    n_params: int,
    period: float = 1.0,
) -> np.ndarray:
    """
    Computes the actual value of the parameters for each timepoint

    Parameters
    ----------
    params : array of floats
        Encoding of the harmonics of the parameters. The array should have a shape of
        `(n_params * (2 * n_harmonics + 1))`, where `n_params` is the number of parameters
        to consider and `n_harmonics` is the number of harmonics to consider.
        The parameters are cyclicly dependent on time, with `n_harmonics` harmonics
        considered.
        `params[i * (2 * n_harmonics + 1):(i + 1) * (2 * n_harmonics + 1)]` contains the harmonics encoding
        of the `i`-th parameter. The first element of the encoding is the constant term, and the
        following elements are the coefficients of the cosine and sine terms of the harmonics.
    t : array of floats
        Timepoints at which the parameters should be evaluated
    n_harmonics : int
        Number of harmonics to consider
    n_params : int
        Number of parameters to consider
    period : float, optional
        Period of the harmonics. The default is 1.0.

    Returns
    -------
    params_val : array of floats of shape `(n_params, len(t))`
        Actual values of the parameters for each timepoint. The array has a shape of
        `(n_params, len(t))`, where `n_params` is the number of parameters to consider and
        `len(t)` is the number of timepoints at which the parameters should be evaluated.
        `params_val[i, j]` contains the value of the `i`-th parameter at the `j`-th timepoint.
    """
    # Reshapes the parameters in a more convenient 2D structure
    params_t = np.reshape(params, (n_params, 2 * n_harmonics + 1))

    # Initializes the actual values of the parameters for each timepoint
    params_val = np.zeros((n_params, len(t)))

    # Constant term
    params_val[...] = params_t[:, :1]

    # Higher order harmonics
    for k in range(1, n_harmonics + 1):
        params_val += params_t[:, 2 * k - 1].reshape(-1, 1) * np.cos(
            2 * np.pi * k * t.reshape(1, -1) / period
        )
        params_val += params_t[:, 2 * k].reshape(-1, 1) * np.sin(
            2 * np.pi * k * t.reshape(1, -1) / period
        )

    return params_val


def extract_harmonics(
    x: np.ndarray,
    n_harmonics: int,
    n_periods: int = None,
    period: int = None,
    exact: bool = True,
) -> np.ndarray:
    """
    Extracts the harmonics of a signal

    Parameters
    ----------
    x : array of floats
        The signal from which the harmonics should be extracted
    n_harmonics : int
        Number of harmonics to extract
    n_periods : int, optional
        Number of periods in the data. The default is None.
    period : int, optional
        Period of the data. The default is None.
    exact : bool, optional
        If True, the length of the data should be a multiple of `n_periods` or `period`. The default is True.

    Returns
    -------
    harmonics : array of complex
        The harmonics of the signal
    """
    n = len(x)

    # Checks the input arguments
    if n_periods is None and period is None:
        raise ValueError("Either `n_periods` or `period` should be provided")

    if n_periods is not None and period is not None:
        raise ValueError("Only one of `n_periods` or `period` should be provided")

    # Extracts the number of periods and validates the input
    if n_periods is not None:
        if n % n_periods != 0 and exact:
            raise ValueError(
                "The length of the data should be a multiple of `n_periods`"
            )

    if period is not None:
        if n % period != 0 and exact:
            raise ValueError("The length of the data should be a multiple of `period`")
        n_periods = n // period

    # Computes the Fourier transform of the data
    fft = np.fft.fft(x)

    # Extracts the harmonics of interest
    harmonics = np.zeros(n_harmonics + 1, dtype=complex)

    # Normalizes the harmonics
    harmonics = fft[np.arange(n_harmonics) * n_periods]
    harmonics[0] /= 2
    harmonics *= 2 / n

    return harmonics


def reconstruct_harmonics(harmonics: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Reconstructs the signal from the harmonics

    Parameters
    ----------
    harmonics : array of complex
        The harmonics of the signal
    t : array of floats
        Timepoints at which the signal should be reconstructed

    Returns
    -------
    signal : array of floats
        Reconstructed signal
    """
    # Initializes the signal
    signal = np.zeros(len(t))

    # Constant term
    signal += np.abs(harmonics[0])

    # Higher order harmonics
    for k in range(1, len(harmonics)):
        signal += np.abs(harmonics[k]) * np.cos(
            2 * np.pi * k * t + np.angle(harmonics[k])
        )

    return signal
