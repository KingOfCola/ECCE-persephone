import colorsys
import matplotlib.pyplot as plt
from itertools import product


def tuple_to_hex(t):
    return f"#{int(t[0] * 255):02x}{int(t[1] * 255):02x}{int(t[2] * 255):02x}"


def hex_to_tuple(h):
    return (int(h[1:3], 16) / 255, int(h[3:5], 16) / 255, int(h[5:7], 16) / 255)


def plot_colormap(colors, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    if isinstance(colors[0], str):
        colors = [list(hex_to_tuple(c)) for c in colors]

    ax.imshow([colors], aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])


def analogous_luminosity(colors, luminosity_shifts, rgb=True):
    # Convert from RGB to HSL if needed
    if rgb:
        colors = [colorsys.rgb_to_hls(*c) for c in colors]
    colors = [
        (h, l + shift, s) for (h, l, s), shift in product(colors, luminosity_shifts)
    ]

    # Convert back to RGB if needed
    if rgb:
        colors = [colorsys.hls_to_rgb(*c) for c in colors]
    return colors


if __name__ == "__main__":
    MONTHS_COLORS = ["#70d6ff", "#90a955", "#ffd670", "#ff9770"]
    MONTHS_COLORS_RGB = [hex_to_tuple(c) for c in MONTHS_COLORS]
    MONTHS_COLORS_HLS = [colorsys.rgb_to_hls(*c) for c in MONTHS_COLORS_RGB]

    MONTHS_COLORS_RGB_EXTENDED = analogous_luminosity(
        MONTHS_COLORS_RGB, [0.0, -0.075, -0.15]
    )
    MONTHS_COLORS_EXTENDED = [tuple_to_hex(c) for c in MONTHS_COLORS_RGB_EXTENDED]

    fig, axs = plt.subplots(2, 1, figsize=(6, 4))
    plot_colormap(MONTHS_COLORS, axs[0])
    plot_colormap(MONTHS_COLORS_EXTENDED, axs[1])
