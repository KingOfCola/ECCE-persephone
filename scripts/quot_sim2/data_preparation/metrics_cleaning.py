import pandas as pd
from time import time
from tqdm import tqdm

from utils.paths import data

filenames = [
    # "QUOT_SIM2_1958-1959.csv.gz",
    # "QUOT_SIM2_1960-1969.csv.gz",
    # "QUOT_SIM2_1970-1979.csv.gz",
    # "QUOT_SIM2_1980-1989.csv.gz",
    # "QUOT_SIM2_1990-1999.csv.gz",
    # "QUOT_SIM2_2000-2009.csv.gz",
    # "QUOT_SIM2_2010-2019.csv.gz",
    "QUOT_SIM2_2020-2024-05.csv.gz",
]
metrics = [
    "T_Q",
    "TINF_H_Q",
    "TSUP_H_Q",
    "PRELIQ_Q",
    "PRENEI_Q",
    "HU_Q",
]

if __name__ == "__main__":
    raw_dir = data("Quot_SIM2/Raw")
    preprocessed_dir = data("Quot_SIM2/Preprocessed")

    stations = pd.read_parquet(rf"{preprocessed_dir}/stations.parquet")

    metrics_data = {}

    start = time()

    # Iterate over the files
    for filename in tqdm(filenames, desc="files", total=len(filenames)):
        # Iterate over the metrics
        # Read the data
        data = pd.read_csv(rf"{raw_dir}\{filename}", sep=";")
        data.rename(columns={"LAMBX": "lambert_x", "LAMBY": "lambert_y"}, inplace=True)

        # Merge the stations data
        data = data.merge(stations, on=["lambert_x", "lambert_y"])

        # Convert the date to year and doy
        data["dateday"] = pd.to_datetime(data["DATE"], format="%Y%m%d")
        data["year"] = data["dateday"].dt.year
        data["day_of_year"] = data["dateday"].dt.dayofyear

        for metric in metrics:
            # Pivot the data
            if metric not in metrics_data:
                metrics_data[metric] = []

            metrics_data[metric].append(
                data.pivot_table(
                    index=["year", "date_of_year"],
                    columns="station_id",
                    values=metric,
                    aggfunc="sum",
                    fill_value=0,
                ).copy()
            )

        del data

    # Store the data in the preprocess directory
    for metric in metrics:
        metrics_data[metric] = pd.concat(metrics_data[metric])
        metrics_data[metric].to_parquet(
            rf"{preprocessed_dir}/1958_2024-05_{metric}.parquet"
        )

        end = time()
        print(f"Stored {metric} data in {preprocessed_dir} ({end-start:.2f}s)")