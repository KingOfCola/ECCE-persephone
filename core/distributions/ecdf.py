import numpy as np
from numba import njit, prange
from scipy import stats

from core.data.confidence_intervals import ConfidenceInterval


@njit(parallel=True)
def ecdf_multivariate(x: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """
    Compute the empirical cumulative distribution function of a multivariate dataset.

    Parameters
    ----------
    x : np.ndarray of shape `(n, p)`
        The points on which to compute the empirical cumulative distribution function.
    samples : np.ndarray of shape `(m, p)`
        Samples from the theoretical distribution of `x`. It should be much larger than `x`.

    Returns
    -------
    np.ndarray of shape `(n,)`
        The empirical cumulative distribution function evaluated on points of `x`.
    """
    n_x, p = x.shape
    n_samples, _ = samples.shape

    # Storage for empirical cumulative distribution function
    fn_emp = np.zeros(n_x, dtype="float64")

    # Storage for boolean reflecting whether eqch sample is below the point x[i]
    all_below = np.zeros(n_samples, dtype="bool")

    for i in prange(n_x):
        # Counts the number of samples below the point x[i]
        for j in prange(n_samples):
            all_below[j] = True
            for k in range(p):
                if samples[j, k] > x[i, k]:
                    all_below[j] = False
                    break
        fn_emp[i] = np.mean(all_below)

    return fn_emp


def ecdf_ci_dvoretzky(
    ecdf: np.ndarray, alpha: float = 0.05, decimation_factor: float = 1
) -> ConfidenceInterval:
    """
    Compute the confidence interval of the empirical cumulative distribution function.
    Using Dvoretzky-Kiefer-Wolfowitz inequality

    Parameters
    ----------
    ecdf : np.ndarray
        The empirical cumulative distribution function.
    alpha : float
        The confidence level.
    decimation_factor : float
        The decimation factor to use in the computation of the confidence interval.

    Returns
    -------
    ConfidenceInterval
        The confidence interval of the empirical cumulative distribution function.
    """
    n = len(ecdf)
    n_eff = n / decimation_factor
    epsilon = np.sqrt(np.log(2 / alpha) / (2 * n_eff))
    ci = ConfidenceInterval(n)
    ci.lower = np.maximum(ecdf - epsilon, 0)
    ci.upper = np.minimum(ecdf + epsilon, 1)
    ci.values = ecdf
    return ci


def ecdf_ci_binomial(
    ecdf: np.ndarray, alpha: float = 0.05, decimation_factor: float = 1
) -> ConfidenceInterval:
    """
    Compute the confidence interval of the empirical cumulative distribution function.
    Using Dvoretzky-Kiefer-Wolfowitz inequality

    Parameters
    ----------
    ecdf : np.ndarray
        The empirical cumulative distribution function.
    alpha : float
        The confidence level.
    decimation_factor : float
        The decimation factor to use in the computation of the confidence interval.

    Returns
    -------
    ConfidenceInterval
        The confidence interval of the empirical cumulative distribution function.
    """
    n = len(ecdf)
    n_eff = n / decimation_factor
    alpha_all = alpha / n_eff
    var = np.sqrt(ecdf * (1 - ecdf) / n_eff)
    z_alpha = stats.norm.ppf(1 - alpha_all / 2)
    ci = ConfidenceInterval(n)
    ci.lower = np.maximum(ecdf - z_alpha * var, 0)
    ci.upper = np.minimum(ecdf + z_alpha * var, 1)
    ci.values = ecdf
    return ci
