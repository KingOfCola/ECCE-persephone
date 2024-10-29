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
from pathlib import Path
import json
import warnings

__ROOT_PATH = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
__PATHS_JSON = "paths.json"

__PATHS = None
__PATH_STRUCTURE = {
    "data": "Data folder",
    "output": "Output folder",
}


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
    return __ROOT_PATH / "assets" / path


def load_path_json() -> dict:
    """
    Load the paths from the paths.json file.

    Returns
    -------
    dict
        The paths as a dictionary.
    """
    global __PATHS

    if __PATHS is not None:
        return __PATHS

    # Load the paths from the JSON file
    path_json = asset(__PATHS_JSON)
    if not os.path.exists(path_json):
        create_path_json(paths=__PATH_STRUCTURE)
        raise ValueError(
            f"The paths file '{path_json}' does not exist. It has been created with default paths. Please fill it with the correct paths."
        )

    with open(path_json, "r") as f:
        __PATHS = json.load(f)

    if not validate_paths(paths=__PATHS):
        raise ValueError(
            f"The paths stored in '{path_json}' are not valid. Please check them."
        )

    return __PATHS


def create_path_json(paths):
    """
    Create the paths.json file with the given paths.

    Parameters
    ----------
    paths : dict
        The paths to write to the JSON file.
    """
    # Get the path to the JSON file
    path_json = asset(__PATHS_JSON)

    # Write the paths to the JSON file
    with open(path_json, "w") as f:
        json.dump(paths, f, indent=4)

    if not validate_paths(paths=paths):
        warnings.warn(
            f"The paths stored in '{path_json}' are not valid. Please check them."
        )


def validate_paths(paths: dict) -> bool:
    """
    Validate the paths dictionary.

    Parameters
    ----------
    paths : dict
        The paths to validate.

    Returns
    -------
    bool
        True if the paths are valid, False otherwise.
    """
    for key, value in __PATH_STRUCTURE.items():
        if key not in paths:
            return False
        if not os.path.exists(paths[key]):
            return False
    return True


def get_path_factory(root: str):
    paths = load_path_json()
    root_path = Path(paths[root])

    def get_path(path: str) -> str:
        """
        Get an absolute path from a path relative to a global root.

        Parameters
        ----------
        path : str
            The local path of the file relative to the root.

        Returns
        -------
        str
            The absolute path.
        """
        return root_path / path

    return get_path


data_dir = get_path_factory("data")
output = get_path_factory("output")
