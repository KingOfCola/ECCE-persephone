import numpy as np
import pandas as pd
from time import time
import statsmodels.api as sm

from matplotlib import pyplot as plt

if __name__ == "__main__":
    DIRNAME = r"C:\Users\Urvan\Documents\ECCE\Data\Quot_SIM2"
    filename = r"QUOT_SIM2_2000-2009.csv.gz"

    ORIGIN_YEAR = 1968
    ORIGIN_DATE = pd.to_datetime(f"{ORIGIN_YEAR}0101", format="%Y%m%d")
    YEAR = pd.to_timedelta("1 day") * 365.25

    start = time()
    data = pd.read_csv(rf"{DIRNAME}\{filename}", sep=";")
    end = time()

    print(f"Elapsed time: {end-start:.2f}s")
    print(data.columns)
    print(data.shape)
    print(data["DATE"].nunique())

    data["LAMBX"].nunique()
    data["LAMBY"].nunique()
    data["DATEDAY"] = pd.to_datetime(data["DATE"], format="%Y%m%d")
    data["YEAR"] = (data["DATEDAY"] - ORIGIN_DATE) / YEAR + ORIGIN_YEAR

    lambdas = data[["LAMBX", "LAMBY"]].drop_duplicates()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(lambdas["LAMBX"], lambdas["LAMBY"])
    plt.show()

    np.random.seed(0)

    N = len(lambdas)
    i = np.random.randint(0, N)
    lambx = lambdas["LAMBX"].values[i]
    lamby = lambdas["LAMBY"].values[i]

    print(f"Station {i}: (LAMBX = {lambx}, LAMBY = {lamby})")

    station_data = data.loc[(data["LAMBX"] == lambx) & (data["LAMBY"] == lamby)]
    variables = {
        "T_Q": {"c": "orange"},
        "TINF_H_Q": {"c": "blue"},
        "TSUP_H_Q": {"c": "red"},
    }

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(lambdas["LAMBX"], lambdas["LAMBY"], s=10)
    ax.scatter(lambx, lamby, c="r")
    ax.set_xlabel(r"$\lambda_x (hm)$")
    ax.set_ylabel(r"$\lambda_y (hm)$")
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    for key, settings in variables.items():
        ax.plot(station_data["YEAR"], station_data[key], c=settings["c"], label=key)
    ax.legend()
    plt.plot()

    ## QQ-plot
    fig, axes = plt.subplots(ncols=3, figsize=(10, 3))
    for key, ax in zip(variables, axes):
        sm.qqplot(station_data[key], ax=ax, line="s")
        ax.set_xlabel("Theoretical quantile")
        ax.set_ylabel(key)

    ## Date heatmap
    date = 20030701
    date_data = data.loc[data["DATE"] == date]
    heatmap = date_data.pivot_table(
        values="T_Q", index="LAMBY", columns="LAMBX", aggfunc="sum", fill_value=np.nan
    )

    fig, ax = plt.subplots()
    ax.imshow(heatmap, origin="lower", cmap="RdBu_r")
    plt.show()
