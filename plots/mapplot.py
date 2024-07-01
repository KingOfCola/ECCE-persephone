import os
import geopandas as gpd
import numpy as np

from utils.paths import asset

__MAPS_URL = {
    "europe-01m": asset("maps/ref-countries-2020-01m/CNTR_BN_01M_2020_4326.geojson"),
    "europe-03m": asset("maps/ref-countries-2020-03m/CNTR_BN_03M_2020_4326.geojson"),
    "europe-10m": asset("maps/ref-countries-2020-10m/CNTR_BN_10M_2020_4326.geojson"),
    "europe-20m": asset("maps/ref-countries-2020-20m/CNTR_BN_20M_2020_4326.geojson"),
    "europe-60m": asset("maps/ref-countries-2020-60m/CNTR_BN_60M_2020_4326.geojson"),
    "europe": asset("maps/ref-countries-2020-10m/CNTR_BN_10M_2020_4326.geojson"),
}
__MAPS = {}


def get_map(map_name):
    """
    Get a map from the global maps dictionary. If the map is not already loaded, load it.

    Parameters
    ----------
    map_name : str
        The name or URL of the map to load.

    Returns
    -------
    geopandas.GeoDataFrame
        The map as a GeoDataFrame.
    """
    # If the map has already been loaded, return it
    if map_name in __MAPS:
        return __MAPS[map_name]

    # If the map is a URL from the alias dictionary, load it
    elif map_name in __MAPS_URL:
        map_path = os.path.join(__MAPS_URL[map_name])
        __MAPS[map_name] = gpd.read_file(map_path)
        return __MAPS[map_name]

    # If the map is a URL, load it
    else:
        try:
            __MAPS[map_name] = gpd.read_file(map_name)
        except Exception as e:
            raise ValueError(f"Map {map_name} not found") from e
        return __MAPS[map_name]


def plot_map(map_name, ax=None, **kwargs):
    """
    Plot a map on a given axis.

    Parameters
    ----------
    map_name : str or geopandas.GeoDataFrame
        The name or URL of the map to plot, or the map itself.
    ax : matplotlib.axes.Axes, optional
        The axis on which to plot the map. If None, a new axis is created.
    **kwargs
        Additional keyword arguments to pass to the plot function.

    Returns
    -------
    matplotlib.axes.Axes
        The axis on which the map is plotted.
    """
    # If the map_name is a string, get the corresponding map
    if isinstance(map_name, str):
        gdf = get_map(map_name)
    else:
        gdf = map_name

    if ax is None:
        ax = gdf.plot(**kwargs)
    else:
        gdf.plot(ax=ax, **kwargs)
    return ax


def set_lims(ax, lat_min, lat_max, lon_min, lon_max):
    """
    Set the limits of a plot axis to a given latitude and longitude range. Rescale the aspect ratio to match the Lambert
    projection.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to set the limits of.
    lat_min : float
        The minimum latitude.
    lat_max : float
        The maximum latitude.
    lon_min : float
        The minimum longitude.
    lon_max : float
        The maximum longitude.

    Returns
    -------
    None
    """
    # Set the limits of the axis
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    # Set the aspect ratio of the plot close to the one of the Lambert projection
    ax.set_aspect(1 / np.cos(np.pi / 180 * (lat_min + lat_max) / 2))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    europe = get_map("europe-10m")
    fig, ax = plt.subplots(figsize=(7, 7))
    plot_map(europe, ax=ax, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    set_lims(ax, 40, 55, -7, 13)
    plt.show()
