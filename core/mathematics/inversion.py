import numpy as np


def pwl_approximation(f, xmin, xmax, n_pwl):
    """Piecewise linear approximation of a function.

    Parameters
    ----------
    f : callable
        The function to approximate.
    xmin : float
        The minimum value of the domain.
    xmax : float
        The maximum value of the domain.
    n_pwl : int
        The number of breakpoints.

    Returns
    -------
    np.ndarray
        The breakpoints.
    np.ndarray
        The values of the function at the breakpoints.
    """
    x = np.linspace(xmin, xmax, n_pwl + 1, endpoint=True)
    y = f(x)
    return x, y


def pwl_inverse(y, f, xmin, xmax, n_pwl):
    """Inverse of a piecewise linear function.

    Parameters
    ----------
    y : np.ndarray
        The values at which the inverse is evaluated.
    f : callable
        The function to invert.
    xmin : float
        The minimum value of the domain.
    xmax : float
        The maximum value of the domain.
    n_pwl : int
        The number of breakpoints.

    Returns
    -------
    np.ndarray
        The inverse values.
    """
    x_bps, y_bps = pwl_approximation(f, xmin, xmax, n_pwl)
    return np.interp(y, y_bps, x_bps)
