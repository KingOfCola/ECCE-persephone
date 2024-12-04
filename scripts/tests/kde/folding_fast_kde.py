# -*-coding:utf-8 -*-
"""
@File    :   folding_fast_kde.py
@Time    :   2024/11/27 11:33:41
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Tests for the Folded version of Fast KDE
"""

from fastkde import fastKDE
import numpy as np
import matplotlib.pyplot as plt


class FoldedFastKDE:
    def __init__(self, samples, bbox: np.ndarray = None):
        """
        Initiates a Folded Fast KDE instance

        Parameters
        ----------
        samples : array of floats of shape `(n, d)`
            Samples of dimension `d` to compute the kde on.
        bbox : array of floats of shape `(d, 2)`, optional.
            Bounding rectangle where the distribution lies in.
            This is the [0, 1]^d rectangle or the smallest rectangle
            encompassing all the points by default.
        """
        self.samples = samples
        self.bbox = self.__find_bbox(bbox, samples)
        self.bbox_center = np.mean(self.bbox, axis=1)
        self.folded_samples = self.fold_samples(samples)

    def __find_bbox(self, bbox, samples):
        """
        Finds the bounding box of the samples

        Parameters
        ----------
        samples : array of floats of shape `(n, d)`

            bbox : array of floats of shape `(d, 2)`, optional.

        """
        smallest_box = np.array([samples.min(axis=0), samples.max(axis=0)]).T
        # If the bbox is provided, ensures it has the correct shape and encompasses all elements
        expected_shape = (samples.shape[1], 2)
        if bbox:
            try:
                bbox = np.array(bbox)
            except Exception as exc:
                raise ValueError(
                    "`bbox` should be an array-like of shape (d, 2)."
                ) from exc

            if bbox.shape != expected_shape:
                raise ValueError(
                    "`bbox` shape is incompatible with samples dimensionality. "
                    f"Expected an array of shape {expected_shape}"
                )

            if (bbox[:, 0] > smallest_box[:, 0]).any() or (
                bbox[:, 1] < smallest_box[:, 1]
            ).any():
                raise ValueError("`bbox` should contain all the data points.")

            return bbox

        # If the bbox is not provided, returns the smallest box containing all the points
        # or [0, 1]^d if the smallest box is inside the [0, 1]^d rectangle
        if smallest_box.min() > 0 and smallest_box.max() < 1:
            smallest_box = np.array([[0, 1] for _ in range(samples.shape[1])])

        return smallest_box

    def fold_samples(self, samples):
        """
        Folds the samples around the center of the bounding box

        Parameters
        ----------
        samples : array of floats of shape `(n, d)`
            Samples to fold.

        Returns
        -------
        array of floats of shape `(n, d)`
            Folded samples.
        """
        # return samples
        d = samples.shape[1]
        for j in range(d):
            folded_samples = samples.copy()
            where_left = samples[:, j] < self.bbox_center[j]
            folded_samples[where_left, j] = (
                2 * self.bbox[j, 0] - folded_samples[where_left, j]
            )
            folded_samples[~where_left, j] = (
                2 * self.bbox[j, 1] - folded_samples[~where_left, j]
            )

            samples = np.concatenate((samples, folded_samples), axis=0)

        return samples

    def pdf(self):
        """
        Computes the probability density function at the given points

        Returns
        -------
        array of floats of shape `(n,)`
            Probability density function values at the given points.
        """
        folded_samples = [
            self.folded_samples[:, j] for j in range(self.samples.shape[1])
        ]
        scale = len(self.samples) / len(self.folded_samples)
        pdf, values = fastKDE.pdf(*folded_samples, numPoints=257)
        return pdf / scale, values


class WeightedCorrKDE:
    def __init__(Self, samples, bbox: np.ndarray = None):
        self.samples = samples
        self.bbox = self.__find_bbox(bbox, samples)
        self.weights = self.__compute_weights(samples)

    def __compute_weights(self, samples):
        return np.ones(samples.shape[0])

    def __find_bbox(self, bbox, samples):
        smallest_box = np.array([samples.min(axis=0), samples.max(axis=0)]).T
        # If the bbox is provided, ensures it has the correct shape and encompasses all elements
        expected_shape = (samples.shape[1], 2)
        if bbox:
            try:
                bbox = np.array(bbox)
            except Exception as exc:
                raise ValueError(
                    "`bbox` should be an array-like of shape (d, 2)."
                ) from exc

            if bbox.shape != expected_shape:
                raise ValueError(
                    "`bbox` shape is incompatible with samples dimensionality. "
                    f"Expected an array of shape {expected_shape}"
                )

            if (bbox[:, 0] > smallest_box[:, 0]).any() or (
                bbox[:, 1] < smallest_box[:, 1]
            ).any():
                raise ValueError("`bbox` should contain all the data points.")

            return bbox

        # If the bbox is not provided, returns the smallest box containing all the points
        # or [0, 1]^d if the smallest box is inside the [0, 1]^d rectangle
        if smallest_box.min() > 0 and smallest_box.max() < 1:
            smallest_box = np.array([[0, 1] for _ in range(samples.shape[1])])

        return smallest_box


if __name__ == "__main__":
    from core.distributions.copulas.clayton_copula import ClaytonCopula
    from core.distributions.kde.beta_kde import BetaKDE
    from core.distributions.kde.bounded_kde import WeightedKDE

    # Generate samples from a Clayton Copula
    N = 10000
    d = 2
    copula = ClaytonCopula(3)
    samples = copula.rvs(N, d)

    q = np.linspace(0, 1, 51)
    x, y = np.meshgrid(q, q)
    z = np.stack([x.flatten(), y.flatten()], axis=1)

    # Compute the KDE
    fast_kde = FoldedFastKDE(samples)
    fast_pdf, fast_values = fast_kde.pdf()

    # Compute the KDE using the BetaKDE
    beta_kde = BetaKDE(samples)
    beta_pdf = beta_kde.pdf(z).reshape(x.shape)

    # Compute the KDE using the WeightedKDE
    weighted_kde = WeightedKDE(samples)
    weighted_pdf = weighted_kde.pdf(z).reshape(x.shape)

    # True PDF
    true_pdf = copula.pdf(z).reshape(x.shape)

    # Plot the results
    VMIN = 0
    VMAX = 5
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    axes = axes.flatten()
    axes[0].imshow(
        fast_pdf,
        extent=[
            fast_values[0].min(),
            fast_values[0].max(),
            fast_values[1].min(),
            fast_values[1].max(),
        ],
        origin="lower",
        cmap="viridis",
        vmin=VMIN,
        vmax=VMAX,
        interpolation="bilinear",
    )
    axes[0].set_title("Folded Fast KDE")
    axes[1].imshow(
        beta_pdf,
        extent=[0, 1, 0, 1],
        origin="lower",
        cmap="viridis",
        vmin=VMIN,
        vmax=VMAX,
        interpolation="bilinear",
    )
    axes[1].set_title("Beta KDE")
    axes[2].imshow(
        weighted_pdf,
        extent=[0, 1, 0, 1],
        origin="lower",
        cmap="viridis",
        vmin=VMIN,
        vmax=VMAX,
        interpolation="bilinear",
    )
    axes[2].set_title("Weighted KDE")
    axes[3].imshow(
        true_pdf,
        extent=[0, 1, 0, 1],
        origin="lower",
        cmap="viridis",
        vmin=VMIN,
        vmax=VMAX,
        interpolation="bilinear",
    )
    axes[4].scatter(*samples.T, s=1, alpha=0.5)
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    for axes in axes[5:]:
        axes.axis("off")
