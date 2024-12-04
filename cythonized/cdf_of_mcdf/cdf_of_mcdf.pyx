from libc.math cimport log, tgamma, exp
import numpy as np

cpdef pi(double c, double d):
    """
    Approximates the reciprocal of the probability of isofrequency regions in the independent case.

    Parameters
    ----------
    c : float
        The level of the isofrequency line delimiting the isofrequency region.
    d : int
        The degree of freedom.

    Returns
    -------
    float
        The reciprocal of the probability of the isofrequency region for the given degree of freedom.
    """

    cdef double f, lc, s, fmax

    if c == 0.0:
        return 1.0
    lc = -log(c)
    f = lc**d / tgamma(d + 1)
    s = 0.0
    fmax = f

    while f > 1e-4 * fmax:
        s += f
        d += 1
        f *= lc / d
        fmax = max(fmax, f)
    return c * s


cpdef pi_comp_low(double c, double d):
    """
    Approximates the reciprocal of the probability of isofrequency regions in the independent case.

    Parameters
    ----------
    c : float
        The level of the isofrequency line delimiting the isofrequency region.
    d : int
        The degree of freedom.

    Returns
    -------
    float
        The reciprocal of the probability of the isofrequency region for the given degree of freedom.
    """

    cdef double lc

    if c == 0.0:
        return 0.0

    lc = np.log(c)
    return exp(lc + (d - 1) * log(-lc) - log(tgamma(d)) -d / lc + (d**2/2 - d) / lc**2)

def cdf_of_mcdf(q: np.ndarray, dof: float) -> np.ndarray:
    """
    Compute the cumulative distribution function of the maximum cumulative distribution function.

    Parameters
    ----------
    q : np.ndarray
        The quantiles.
    dof : float
        The degree of freedom.  

    Returns
    -------
    np.ndarray
        The cumulative distribution function of the maximum cumulative distribution function.
    """
    cdef int i
    cdef double[::1] res = np.zeros(q.shape[0])

    for i in range(q.shape[0]):
        res[i] = 1.0 - pi(q[i], dof) 
        if res[i] < 1e-3:
            res[i] = pi_comp_low(q[i], dof)
    return np.array(res)