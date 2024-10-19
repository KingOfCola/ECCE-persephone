# -*-coding:utf-8 -*-
"""
@File    :   _utils.py
@Time    :   2024/10/18 17:47:52
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Utility functions for data manipulation
"""

import numpy as np


def process_data(params):
    """
    Process the data

    Parameters
    ----------
    params : tuple
        The parameters of the process. Should contain:
        - yearf : array of shape (n,)
            The floating point year
        - data : Series
            The data to process
        - model : object
            The model to fit
        - label : str
            The label of the data

    Returns
    -------
    label : str
        The label of the data
    model : object
        The model fitted
    cdf : array of shape (n,)
        The CDF of the data
    """

    yearf, data, model, label = params
    try:
        model.fit(yearf.values, data.values)
    except Exception as exc:
        pass

    if model._isfit():
        cdf = model.cdf(yearf, data.values)
    else:
        cdf = np.full_like(data.values, np.nan)

    return label, model, cdf
