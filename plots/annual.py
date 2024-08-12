import matplotlib.pyplot as plt

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
