
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import statsmodels.api as sm
from scipy import special, stats
from scipy.optimize import minimize
from scipy.integrate import quad
from tqdm import tqdm

from utils.paths import data, output
from plots.mapplot import plot_map, set_lims

if __name__ == "__main__":
    FULL_YEAR_MIN = 1959
    FULL_YEAR_MAX = 2023
    YEARS = FULL_YEAR_MAX - FULL_YEAR_MIN + 1

    DAYS_IN_YEAR = 365
    N = YEARS * DAYS_IN_YEAR
    STATION = "S8234"

    station_locations = pd.read_parquet(data(r"Preprocessed/stations.parquet"))
    station_loc = station_locations.loc[station_locations["station_id"] == STATION]

    temperatures_stations = pd.read_parquet(
        data(r"Preprocessed/1958_2024-05_T_Q.parquet")
    )

    temperatures_stations.reset_index(inplace=True)
    temperatures_stations = temperatures_stations.loc[
        (temperatures_stations["year"].between(FULL_YEAR_MIN, FULL_YEAR_MAX))
        & (temperatures_stations["day_of_year"] <= DAYS_IN_YEAR)
    ]

    years = temperatures_stations["year"].values
    days = temperatures_stations["day_of_year"].values
    time = years + days / DAYS_IN_YEAR


    temperatures = temperatures_stations[STATION].values

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time, temperatures)
    ax.set_xlabel("Time")
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(1/12))
    ax.set_ylabel("Mean temperature")
    ax.set_title(f"Temperature profile of station {STATION}")

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_map("europe", ax=ax)
    set_lims(ax, 40, 55, -7, 13)
    ax.scatter(station_loc["longitude"], station_loc["latitude"], c="red", s=100)
    plt.show()