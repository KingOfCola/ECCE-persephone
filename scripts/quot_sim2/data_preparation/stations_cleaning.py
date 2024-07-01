# -*-coding:utf-8 -*-
"""
@File      :   stations_cleaning.py
@Time      :   2024/06/28 12:39:54
@Author    :   Urvan Christen
@Version   :   1.0
@Contact   :   urvan.christen@gmail.com
@Desc      :   Scripts for assigning a unique identifier for the stations from the
Meteo-France QUOT-SIM2 data set and create a mapping to their coordinates (latitude,
longitude, altitude) and lambert coordinates (lambert x and lambert y).
"""

import pandas as pd
import numpy as np
from pykml import parser


def read_kml(filename: str) -> any:
    """
    Read a KML file and return the parsed object.

    Parameters
    ----------
    filename : str
        The path to the KML file.

    Returns
    -------
    kml : pykml.KML
        The parsed KML object.
    """
    with open(filename, "r") as f:
        kml = parser.parse(f)
    return kml


def extract_coordinates(kml: any) -> list[dict]:
    """
    Extract the coordinates from a KML object.

    Parameters
    ----------
    kml : pykml.KML
        The parsed KML object.

    Returns
    -------
    coordinates : list[dict]
        The list of coordinates with the following keys:
        - name: the name of the station
        - longitude: the longitude of the station
        - latitude: the latitude of the station
        - altitude: the altitude of the station
    """
    coordinates = []

    # Extract for each station its id and its coordinates
    for placemark in kml.getroot().Document.Placemark:
        name = placemark.name.text
        coords = placemark.Point.coordinates.text.strip().split(",")
        coordinates.append(
            {
                "station_id": f"S{name.strip():0>4}",
                "longitude": float(coords[0]),
                "latitude": float(coords[1]),
                "altitude": float(coords[2]) if len(coords) == 3 else 0,
            }
        )
    return coordinates


if __name__ == "__main__":
    # Working directory
    dirname = r"C:\Users\Urvan\Documents\ECCE\Data\Quot_SIM2"
    raw_dir = rf"{dirname}\Raw"
    preprocessed_dir = rf"{dirname}\Preprocessed"

    # Read the KML file
    kml = read_kml(rf"{raw_dir}\SIM2.kml")
    coordinates = extract_coordinates(kml)
    coordinates_df = pd.DataFrame(coordinates)

    # Read the lambert coordinates
    lambert_coors = pd.read_csv(
        rf"{raw_dir}\coordonnees_grille_safran_lambert-2-etendu.csv",
        sep=";",
        decimal=",",
    )
    lambert_coors.rename(
        columns={
            "LAMBX (hm)": "lambert_x",
            "LAMBY (hm)": "lambert_y",
            "LAT_DG": "latitude",
            "LON_DG": "longitude",
        },
        inplace=True,
    )

    # Merge coordinate systems (Both are floating points, but precision seems to be the same)
    stations = coordinates_df.merge(
        lambert_coors, validate="1:1", on=["latitude", "longitude"]
    )

    # Save data as csv (for readability) and parquet (for efficiency)
    stations.to_csv(rf"{preprocessed_dir}\stations.csv", index=False)
    stations.to_parquet(rf"{preprocessed_dir}\stations.parquet", index=False)
