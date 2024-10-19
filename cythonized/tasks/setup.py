# -*-coding:utf-8 -*-
"""
@File    :   setup.py
@Time    :   2024/10/01 08:59:42
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Setup script for Cython files
    To compile the Cython files, run the following command:
    cd cythonized
    python tasks/setup.py build_ext --inplace
"""

from setuptools import setup
from Cython.Build import cythonize
import os


ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

setup(
    name="Hello world app",
    ext_modules=cythonize(os.path.join(ROOT_PATH, "**/*.pyx"), annotate=True),
)
