# -*-coding:utf-8 -*-
"""
@File      :   paths.py
@Time      :   2024/07/01 09:54:36
@Author    :   Urvan Christen
@Version   :   1.0
@Contact   :   urvan.christen@gmail.com
@Desc      :   Utility functions for handling paths.
"""


import os

__ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def asset(path: str) -> str:
    """
    Return the path to an asset file in the assets folder.

    Parameters
    ----------
    path : str
        The path to the asset file.

    Returns
    -------
    str
        The full path to the asset file.
    """
    return os.path.join(__ROOT_PATH, "assets", path)
