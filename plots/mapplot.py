import os
import geopandas as gpd
import numpy as np

from utils.paths import asset

__MAPS_URL = {
    "europe-01m": {
        "path": asset("maps/ref-countries-2020-01m/CNTR_BN_01M_2020_4326.geojson"),
        "acknowledgement": "@EuroGeographics",
    },
    "europe-03m": {
        "path": asset("maps/ref-countries-2020-03m/CNTR_BN_03M_2020_4326.geojson"),
        "acknowledgement": "@EuroGeographics",
    },
    "europe-10m": {
        "path": asset("maps/ref-countries-2020-10m/CNTR_BN_10M_2020_4326.geojson"),
        "acknowledgement": "@EuroGeographics",
    },
    "europe-20m": {
        "path": asset("maps/ref-countries-2020-20m/CNTR_BN_20M_2020_4326.geojson"),
        "acknowledgement": "@EuroGeographics",
    },
    "europe-60m": {
        "path": asset("maps/ref-countries-2020-60m/CNTR_BN_60M_2020_4326.geojson"),
        "acknowledgement": "@EuroGeographics",
    },
    "europe": {
        "path": asset("maps/ref-countries-2020-10m/CNTR_BN_10M_2020_4326.geojson"),
        "acknowledgement": "@EuroGeographics",
    },
    "switzerland-district-zone": {
        "path": asset(
            "maps/switzerland/swissBOUNDARIES3D_1_3_TLM_BEZIRKSGEBIET.geojson"
        ),
        "acknowledgement": "Swiss Federal Office of Topography",
    },
    "switzerland-territory-zone": {
        "path": asset(
            "maps/switzerland/swissBOUNDARIES3D_1_3_TLM_HOHEITSGEBIET.geojson"
        ),
        "acknowledgement": "Swiss Federal Office of Topography",
    },
    "switzerland-territory-bounds": {
        "path": asset(
            "maps/switzerland/swissBOUNDARIES3D_1_3_TLM_HOHEITSGRENZE.geojson"
        ),
        "acknowledgement": "Swiss Federal Office of Topography",
    },
    "switzerland-canton-zone": {
        "path": asset(
            "maps/switzerland/swissBOUNDARIES3D_1_3_TLM_KANTONSGEBIET.geojson"
        ),
        "acknowledgement": "Swiss Federal Office of Topography",
    },
    "switzerland-country-zone": {
        "path": asset(
            "maps/switzerland/swissBOUNDARIES3D_1_3_TLM_LANDESGEBIET.geojson"
        ),
        "acknowledgement": "Swiss Federal Office of Topography",
    },
    "switzerland": {
        "path": asset(
            "maps/switzerland/swissBOUNDARIES3D_1_3_TLM_KANTONSGEBIET.geojson"
        ),
        "acknowledgement": "Swiss Federal Office of Topography",
    },
}
__MAPS = {}
MAP_LIST = list(__MAPS_URL.keys())
__BOUNDARIES = {
    "France": {"lat_min": 41.3, "lat_max": 51.1, "lon_min": -5.2, "lon_max": 9.6},
    "Switzerland": {"lat_min": 45.8, "lat_max": 47.8, "lon_min": 5.9, "lon_max": 10.6},
    "Italy": {"lat_min": 35.5, "lat_max": 47.1, "lon_min": 6.6, "lon_max": 18.8},
    "Germany": {"lat_min": 47.3, "lat_max": 55.2, "lon_min": 5.5, "lon_max": 15.1},
    "Austria": {"lat_min": 46.3, "lat_max": 49.2, "lon_min": 9.5, "lon_max": 17.2},
    "Europe": {"lat_min": 35.5, "lat_max": 71.2, "lon_min": -11.2, "lon_max": 33.2},
}


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
    # Get acknowledgement if available
    acknowledgement = __MAPS_URL[map_name].get("acknowledgement", "")

    # If the map has already been loaded, return it
    if map_name in __MAPS:
        return __MAPS[map_name], acknowledgement

    # If the map is a URL from the alias dictionary, load it
    elif map_name in __MAPS_URL:
        map_path = os.path.join(__MAPS_URL[map_name]["path"])
        __MAPS[map_name] = gpd.read_file(map_path)
        return __MAPS[map_name], acknowledgement

    # If the map is a URL, load it
    else:
        try:
            __MAPS[map_name] = gpd.read_file(map_name)
        except Exception as e:
            raise ValueError(f"Map {map_name} not found") from e
        return __MAPS[map_name], acknowledgement


def plot_map(map_name, ax=None, acknowledgement: str = "", **kwargs):
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
        gdf, acknowledgement = get_map(map_name)
    else:
        gdf = map_name

    if ax is None:
        ax = plt.gca()

    gdf.plot(ax=ax, **kwargs)
    ax.annotate(
        acknowledgement,
        xy=(0.99, 0.01),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=8,
        bbox=dict(facecolor="w", edgecolor="k", alpha=0.5),
    )
    return ax


def set_lims(
    ax,
    lat_min: float | str,
    lat_max: float = None,
    lon_min: float = None,
    lon_max: float = None,
):
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

    if isinstance(lat_min, str):
        country = lat_min
        lat_min = __BOUNDARIES[country]["lat_min"]
        lat_max = __BOUNDARIES[country]["lat_max"]
        lon_min = __BOUNDARIES[country]["lon_min"]
        lon_max = __BOUNDARIES[country]["lon_max"]

    # Set the limits of the axis
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    # Set the aspect ratio of the plot close to the one of the Lambert projection
    ax.set_aspect(1 / np.cos(np.pi / 180 * (lat_min + lat_max) / 2))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    plot_map("europe", ax=ax, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    set_lims(ax, "France")
    plt.show()
