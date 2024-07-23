import numpy as np
from numba import njit, prange


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
