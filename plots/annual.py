import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import numpy as np

MONTHS_DURATIONS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MONTHS_STARTS = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
MONTHS_CENTER = [15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]
MONTHS_LABELS_F = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
MONTHS_LABELS_3 = [month[:3] for month in MONTHS_LABELS_F]
MONTHS_LABELS_1 = [month[0] for month in MONTHS_LABELS_F]

MONTHS_LABELS = {
    "full": MONTHS_LABELS_F,
    "three": MONTHS_LABELS_3,
    "one": MONTHS_LABELS_1,
}

SEASONS_FULL = ["Winter", "Spring", "Summer", "Autumn"]  # DJL, MAM, JJA, SON
SEASONS_3 = ["DJF", "MAM", "JJA", "SON"]
SEASONS_CENTER = [15, 105, 196, 288]
MONTHS_TO_SEASON = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0]

SEASONS_COLORS = [
    "#70d6ff",
    "#90a955",
    "#ffd670",
    "#ff9770",
]  # Winter, Spring, Summer, Autumn
MONTHS_COLORS = [
    "#49cbfe",  # January
    "#23c0ff",  # February
    "#90a955",  # March
    "#7a8f48",  # April
    "#64763b",  # May
    "#ffd670",  # June
    "#fecb49",  # July
    "#ffc023",  # August
    "#ff9770",  # September
    "#fe7b49",  # October
    "#ff5f23",  # November
    "#70d5ff",  # December
]

MONTHS_CMAP = ListedColormap(MONTHS_COLORS)
DOY_CMAP = ListedColormap(
    [c for c, days in zip(MONTHS_COLORS, MONTHS_DURATIONS) for _ in range(days)]
)


@np.vectorize
def doy_to_season(doy: int) -> int:
    """Converts a day of the year to a season.

    Parameters
    ----------
    doy : int
        The day of the year to convert.

    Returns
    -------
    season : int
        The season corresponding to the day of the year.
    """
    # Seasons are defined as DJF, MAM, JJA, SON
    # They all last 3 months, and the first season covers 2 months at the beginning of the year and 1 month at the end
    for season in range(4):
        if doy < MONTHS_STARTS[season * 3 + 2]:
            return season

    # Last month of the year is in winter
    return 0


@np.vectorize
def doy_to_month(doy: int) -> int:
    """Converts a day of the year to a month.

    Parameters
    ----------
    doy : int
        The day of the year to convert.

    Returns
    -------
    month : int
        The month corresponding to the day of the year.
    """
    for month, start in enumerate(MONTHS_STARTS):
        if doy < start:
            return month - 1

    # Should not happen
    return -1


def month_xaxis(ax: plt.Axes, grid: bool | str = "season", labels: str = "three"):
    """Set the x-axis of a plot to display months.

    Parameters
    ----------
    ax : plt.Axes
        The axes to set the x-axis of.
    grid : bool|str
        Whether to display a grid on the plot. If 'season', display a grid
        for each season.
    labels : str
        The number of letters to display for each month. Can be 'full',
        'three', or 'one'.

    Returns
    -------
    None
    """
    ax.set_xticks(
        MONTHS_STARTS,
        minor=False,
        labels=[],
    )
    ax.set_xticks(MONTHS_CENTER, minor=True)
    ax.set_xticklabels(
        MONTHS_LABELS[labels],
        minor=True,
    )

    # Adds a grid
    if grid:
        ax.grid(True, axis="x", linestyle="dotted", color="gray", alpha=0.3)

    # Highlights the seasons
    if grid == "season":
        ax.axvline(59, color="gray", linestyle="dotted", alpha=0.5, lw=1)
        ax.axvline(151, color="gray", linestyle="dotted", alpha=0.5, lw=1)
        ax.axvline(243, color="gray", linestyle="dotted", alpha=0.5, lw=1)
        ax.axvline(334, color="gray", linestyle="dotted", alpha=0.5, lw=1)


def month_yaxis(ax: plt.Axes, grid: bool = True, labels: str = "three"):
    """Set the y-axis of a plot to display months.

    Parameters
    ----------
    ax : plt.Axes
        The axes to set the y-axis of.
    grid : bool
        Whether to display a grid on the plot.
    labels : str
        The number of letters to display for each month. Can be 'full',
        'three', or 'one'.

    Returns
    -------
    None
    """
    ax.set_yticks(
        MONTHS_STARTS,
        minor=False,
        labels=[],
    )
    ax.set_yticks(MONTHS_CENTER, minor=True)
    ax.set_yticklabels(
        MONTHS_LABELS[labels],
        minor=True,
    )

    # Adds a grid
    if grid:
        ax.grid(True, axis="y", linestyle="dotted", color="gray", alpha=0.3)

    # Highlights the seasons
    if grid == "season":
        ax.axhline(59, color="gray", linestyle="dotted", alpha=0.5, lw=1)
        ax.axhline(151, color="gray", linestyle="dotted", alpha=0.5, lw=1)
        ax.axhline(243, color="gray", linestyle="dotted", alpha=0.5, lw=1)
        ax.axhline(334, color="gray", linestyle="dotted", alpha=0.5, lw=1)
