import matplotlib.pyplot as plt

MONTHS_STARTS = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
MONTHS_CENTER = [15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345]
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
        ax.grid(True, axis="both", linestyle="--", color="gainsboro")

    # Highlights the seasons
    if grid == "season":
        ax.axvline(59, color="gray", linestyle="--")
        ax.axvline(151, color="gray", linestyle="--")
        ax.axvline(243, color="gray", linestyle="--")
        ax.axvline(334, color="gray", linestyle="--")


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
        ax.grid(True, axis="both", linestyle="--", color="gainsboro")

    # Highlights the seasons
    if grid == "season":
        ax.axhline(59, color="gray", linestyle="--")
        ax.axhline(151, color="gray", linestyle="--")
        ax.axhline(243, color="gray", linestyle="--")
        ax.axhline(334, color="gray", linestyle="--")
