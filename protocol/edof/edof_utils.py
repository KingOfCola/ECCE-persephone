# -*-coding:utf-8 -*-
"""
@File    :   edof_utils.py
@Time    :   2024/10/09 14:03:31
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Utilities for the computation of effective degrees of freedom
"""


import numpy as np
from core.distributions.mecdf import MultivariateMarkovianECDF
from core.optimization.mecdf import find_effective_dof, cdf_of_mcdf


def ecdf_of_mcdf(x: np.ndarray, w: int) -> np.ndarray:
    mmecdf = MultivariateMarkovianECDF(d=w)
    mmecdf.fit(x)
    y = mmecdf.cdf(x)
    return np.where(np.isnan(y), 0, y)


def bv_cdf_of_mcdf(q, pi_emp_sorted, dof_emp) -> float:
    p_th = cdf_of_mcdf(pi_emp_sorted, dof_emp)
    bias = np.mean(p_th - q)
    var = np.mean((p_th - q) ** 2)
    return bias, np.sqrt(var)


def __generate_safe(process_generator, process_args, process_kwargs):
    last_exc = None
    for _ in range(10):
        try:
            X = process_generator(*process_args, **process_kwargs)
            break
        except Exception as exc:
            last_exc = exc
    else:
        raise ValueError(
            f"Could not generate process with parameters args={process_args}, kwargs={process_kwargs}"
        ) from last_exc
    return X


def compute_edof(
    process_generator: callable, process_args: list, process_kwargs: dict, auc=True
) -> float:
    # return 1
    X = __generate_safe(process_generator, process_args, process_kwargs)
    n = X.shape[0]
    w = X.shape[1]

    q = np.arange(1, n + 1) / (n + 1)
    pi_emp = ecdf_of_mcdf(X, w=w)
    pi_emp_sorted = np.sort(pi_emp)
    dof_emp = find_effective_dof(q, pi_emp_sorted)
    if not auc:
        return dof_emp

    bias, var = bv_cdf_of_mcdf(q, pi_emp_sorted, dof_emp)
    return dof_emp, bias, var
