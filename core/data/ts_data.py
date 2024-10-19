# -*-coding:utf-8 -*-
"""
@File    :   ts_data.py
@Time    :   2024/10/18 11:12:24
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Time series data structure
"""

from copy import deepcopy
from multiprocessing import Pool
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

from core.data._utils import process_data
from core.distributions.dist import HarmonicDistribution
from utils.constants import DAYS_IN_YEAR


class TSData:
    def __init__(self, data: pd.DataFrame, meta: dict = None):
        self._data = data
        self._labels = data.columns
        self._time = data.index
        self._meta = meta if meta is not None else {}
        self.__prepare_time()

    @property
    def data(self):
        return self._data[self.labels]

    @property
    def time(self):
        return self._time

    @property
    def day(self):
        return self._data["day"]

    @property
    def month(self):
        return self._data["month"]

    @property
    def year(self):
        return self._data["year"]

    @property
    def doy(self):
        return self._data["doy"]

    @property
    def yearf(self):
        return self._data["year"] + self._data["doy"] / DAYS_IN_YEAR

    @property
    def meta(self):
        return self._meta

    @property
    def labels(self):
        return self._labels

    @property
    def n(self):
        return self._data.shape[0]

    @property
    def p(self):
        return self._data.shape[1]

    @property
    def first_full_year(self):
        return self.year[self.day == 1].min()

    @property
    def last_full_year(self):
        return self.year[self.day == DAYS_IN_YEAR].max()

    def years_slice(self, start: int, end: int):
        """
        Returns the data between the years start and end.

        Parameters
        ----------
        start : int
            The start year, inclusive.
        end : int
            The end year, inclusive.
        """
        return self[(self.year >= start) & (self.year <= end)]

    def __getitem__(self, key):
        """
        Gets the column at key if key is a string or an integer, or
        a TSData object representing the data with index filtered by the key.

        Parameters
        ----------
        key : str, int
            The key to get the data.

        Returns
        -------
        np.ndarray or TSData
            The data at key.
        """
        if isinstance(key, (str, int)):
            return self._data[key].values
        else:
            df = self.to_dataframe()
            return TSData(df[key].copy(), meta=self.meta)

    def __prepare_time(self):
        self._data.loc[:, "day"] = self._data.index.day
        self._data.loc[:, "month"] = self._data.index.month
        self._data.loc[:, "year"] = self._data.index.year
        self._data.loc[:, "doy"] = self._data.index.dayofyear

    def __str__(self):
        return f"TSData(n={self.n}, p={self.p})"

    def __repr__(self):
        return self.__str__()

    def to_dataframe(self):
        return pd.DataFrame(
            self._data[self.labels], columns=self.labels, index=self.time
        )


class HarmonicTSData(TSData):
    def __init__(
        self,
        data: pd.DataFrame,
        model: HarmonicDistribution,
        meta: dict = None,
    ):
        super().__init__(
            data.to_dataframe() if isinstance(data, TSData) else data, meta
        )
        self._model = model
        self._models = {}
        self._raw_data = None

        self.meta.update(
            {
                "model_base": self._model,
            }
        )

        self.fit_harmonics()

    @property
    def models(self):
        return self._models

    @property
    def raw_data(self):
        return self._raw_data

    def fit_harmonics_monothread(self):
        cdfs = self.data.copy()
        unfit_labels = {}
        for label in (
            bar := tqdm(
                self.labels,
                total=self.p,
                desc="Fitting models",
                leave=False,
                smoothing=0,
            )
        ):
            bar.set_postfix_str(label.name)
            harmonic_model = deepcopy(self._model)
            try:
                harmonic_model.fit(self.yearf.values, self.data[label].values)
            except Exception as exc:
                unfit_labels[label] = exc
            self._models[label] = harmonic_model
            if harmonic_model._isfit():
                cdfs[label] = harmonic_model.cdf(self.yearf, self.data[label].values)
            else:
                cdfs[label] = np.full_like(self.data[label].values, np.nan)

        if unfit_labels:
            warnings.warn(
                f"Could not fit the following labels: {list(unfit_labels.keys())}"
            )

        self._raw_data = self._data
        self._data = cdfs

    def fit_harmonics(self):
        cdfs = self.data.copy()
        unfit_labels = []
        params = [
            (
                deepcopy(self.yearf),
                deepcopy(self.data[label]),
                deepcopy(self._model),
                label,
            )
            for label in self.labels
        ]
        with Pool() as pool:
            results = list(
                tqdm(
                    pool.imap(process_data, params),
                    total=self.p,
                    desc="Fitting models",
                    leave=False,
                    smoothing=0,
                )
            )
        for label, model, cdf in results:
            self._models[label] = model
            cdfs[label] = cdf
            if not model._isfit():
                unfit_labels.append(label)

        if unfit_labels:
            warnings.warn(f"Could not fit the following labels: {unfit_labels}")

        self._raw_data = self._data
        self._data = cdfs
