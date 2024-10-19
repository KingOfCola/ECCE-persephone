# -*-coding:utf-8 -*-
"""
@File    :   iter.py
@Time    :   2024/10/09 13:21:53
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Iteration utilities
"""


def product_list(*args):
    """
    Returns the Cartesian product of the given lists

    Parameters
    ----------
    args : list
        The lists to compute the Cartesian product

    Yields
    -------
    list
        The Cartesian product of the given lists
    """
    if not args:
        return []
    if len(args) == 1:
        for a in args[0]:
            yield [a]

    for p in product_list(*args[:-1]):
        for a in args[-1]:
            yield list(p) + [a]


def product_dict(**kwargs):
    """
    Returns the Cartesian product of the given dictionaries

    Parameters
    ----------
    kwargs : dict
        The dictionaries to compute the Cartesian product

    Yields
    -------
    dict
        The Cartesian product of the given dictionaries
    """
    keys = sorted(list(kwargs.keys()))
    values = [kwargs[k] for k in keys]

    for p in _product_dict_lists(keys, values):
        yield p


def _product_dict_lists(keys: list, values: list):
    """
    Returns the Cartesian product of the given dictionaries

    Parameters
    ----------
    keys : list
        The keys of the dictionaries to compute the Cartesian product
    values : list
        The values of the dictionaries to compute the Cartesian product

    Yields
    -------
    dict
        The Cartesian product of the given dictionaries
    """
    if not keys:
        return {}
    if len(keys) == 1:
        for v in values[0]:
            yield {keys[0]: v}

    for p in _product_dict_lists(keys[:-1], values[:-1]):
        for v in values[-1]:
            yield {**p, keys[-1]: v}
