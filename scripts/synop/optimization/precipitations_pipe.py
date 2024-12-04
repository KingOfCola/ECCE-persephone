import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

from core.distributions.base.dist import HarmonicDistribution
from utils.loaders.ncbn_loader import load_fit_ncbn
from plots.annual import month_xaxis


class HarmonicQQPlot:
    def __init__(self, model: HarmonicDistribution, time_unit: str = None):
        self.model = model
        self.time_unit = "" if time_unit is None else f" ({time_unit})"

    def qqplot(
        self, t: np.ndarray, data: np.ndarray, ax: plt.Axes = None, bins: int = 100
    ):
        if ax is None:
            ax = plt.gca()

        where = np.isfinite(data)
        x = np.sort(data[where])
        n = len(x)
        # t = np.linspace(0, 1, bins, endpoint=False)
        p = np.arange(1, n + 1) / (n + 1)

        tt = np.concatenate([np.random.permutation(t) for _ in range(bins)])

        qq = self.model.ppf(tt, np.tile(p, bins))
        qq = np.sort(qq)
        pp = np.arange(1, len(qq) + 1) / (len(qq) + 1)
        q = np.interp(p, pp, qq)

        xmin = min(np.min(x), np.min(q))
        xmax = max(np.max(x), np.max(q))

        ax.scatter(x, q, s=4)
        ax.set_xlabel("Empirical quantiles")
        ax.set_ylabel("Model quantiles")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(xmin, xmax)
        ax.axline((xmin, xmin), (xmax, xmax), c="k", ls="--", lw=0.7)
        ax.grid(which="major", alpha=0.5, lw=0.7)
        ax.grid(which="minor", alpha=0.2, lw=0.7)
        return ax

    def ppplot(self, t: np.ndarray, data: np.ndarray, ax: plt.Axes = None, **kwargs):
        if ax is None:
            ax = plt.gca()

        where = np.isfinite(data)
        x = data[where]
        t = t[where]

        n = len(x)
        p_emp = np.arange(1, n + 1) / (n + 1)

        p_model = self.model.cdf(t, x)
        p_model = np.sort(p_model)

        ax.scatter(p_emp, p_model, s=4)
        ax.set_xlabel("Empirical probabilities")
        ax.set_ylabel("Model probabilities")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axline((0, 0), (1, 1), c="k", ls="--", lw=0.7)
        ax.grid(which="major", alpha=0.5, lw=0.7)
        ax.grid(which="minor", alpha=0.2, lw=0.7)
        return ax

    def return_period_plot(
        self, t: np.ndarray, data: np.ndarray, ax: plt.Axes = None, **kwargs
    ):
        if ax is None:
            ax = plt.gca()

        duration_step = np.median(np.diff(t))

        where = np.isfinite(data)
        x = data[where]
        t = t[where]

        n = len(x)
        p_emp = np.arange(1, n + 1) / (n + 1)

        p_model = self.model.cdf(t, x)
        p_model = np.sort(p_model)

        rp_emp = duration_step / (1 - p_emp)
        rp_model = duration_step / (1 - p_model)
        rp_min = min(np.min(rp_emp), np.min(rp_model))
        rp_max = max(np.max(rp_emp), np.max(rp_model)) * 1.1

        ax.scatter(rp_emp, rp_model, s=4)
        ax.set_xlabel(f"Empirical return period {self.time_unit}")
        ax.set_ylabel(f"Model return period {self.time_unit}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(rp_min, rp_max)
        ax.set_ylim(rp_min, rp_max)
        ax.axline((rp_min, rp_min), (rp_max, rp_max), c="k", ls="--", lw=0.7)
        ax.grid(which="major", alpha=0.5, lw=0.7)
        ax.grid(which="minor", alpha=0.2, lw=0.7)
        return ax

    def return_level_plot(
        self,
        t: np.ndarray,
        data: np.ndarray,
        ax: plt.Axes = None,
        bins: int = 100,
        **kwargs,
    ):
        if ax is None:
            ax = plt.gca()

        duration_step = np.median(np.diff(t))

        alpha = kwargs.get("alpha", 0.05)

        where = np.isfinite(data)
        x = np.sort(data[where])
        n = len(x)
        # t = np.linspace(0, 1, bins, endpoint=False)
        ks = np.arange(1, n + 1)
        p_emp = ks / (n + 1)
        p_emp_low = beta.ppf(alpha / 2, ks, n + 1 - ks)
        p_emp_up = beta.ppf(1 - alpha / 2, ks, n + 1 - ks)
        q_emp = x

        tt = np.concatenate([np.random.permutation(t) for _ in range(bins)])

        q_model = self.model.ppf(tt, np.tile(p_emp, bins))
        q_model = np.sort(q_model)
        p_model = np.arange(1, len(q_model) + 1) / (len(q_model) + 1)

        q_model_low = np.interp(p_emp_low, p_model, q_model)
        q_model_up = np.interp(p_emp_up, p_model, q_model)

        rp_emp = duration_step / (1 - p_emp)
        rp_model = duration_step / (1 - p_model)

        xmin = np.min(x)
        xmax = np.max(x)
        rp_min = np.min(rp_emp)
        rp_max = np.max(rp_emp) * 1.1

        ax.scatter(rp_emp, q_emp, s=4)
        ax.plot(rp_model, q_model, c="k", lw=0.7)
        ax.fill_between(rp_emp, q_model_low, q_model_up, color="k", alpha=0.2)
        ax.set_xlabel(f"Return period {self.time_unit}")
        ax.set_ylabel("Return level")
        ax.set_xlim(rp_min, rp_max)
        ax.set_ylim(xmin, xmax)
        ax.set_xscale("log")
        ax.grid(which="major", alpha=0.5, lw=0.7)
        ax.grid(which="minor", alpha=0.2, lw=0.7)
        return ax


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    CMAP = plt.get_cmap("turbo")

    data = load_fit_ncbn("t-MAX", model_type=None)
    data._raw_data["PAY"].values

    print(data.labels)
    station = data.labels[1].value
    station = "PAY"
    DAYS_IN_YEAR = 365.25

    where = np.isfinite(data._raw_data[station].values)
    t = data.time.values[where]
    yearf = data.yearf.values[where]
    x = data._data[station].values[where]
    y = data._raw_data[station].values[where]
    model = data.models[station]
    qqplot = HarmonicQQPlot(model, time_unit="years")

    tt = np.linspace(0, 1)
    sged = model._models[1]._model if hasattr(model, "_models") else model
    m, s, l, p = sged.param_valuation(tt)
    fig, axes = plt.subplots(4)
    axes[0].plot(tt, m)
    axes[1].plot(tt, s)
    axes[2].plot(tt, l)
    axes[3].plot(tt, p)
    plt.show()

    return_periods = [1 / 12, 1 / 4, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

    fig, ax = plt.subplots()
    for i, rp in enumerate(return_periods):
        level = 1 - 1 / (DAYS_IN_YEAR * rp)
        label = f"{rp:.0f} yrs" if rp >= 1 else f"{rp * 12:.0f} months"
        ax.plot(
            tt,
            model.ppf(tt, np.full_like(tt, level)),
            label=label,
            c=CMAP(i / (len(return_periods) - 1)),
        )
    ax.scatter(data.yearf[where] % 1, y, s=1)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(bbox_to_anchor=(1.0, 1), loc="upper left")
    fig.tight_layout()
    plt.show()

    # QQ-Plots
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8, 12))
    axes[0, 0].plot(t, x, "o", ms=2, alpha=0.3)
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Probability")

    axes[0, 1].hist(x, bins=np.linspace(0, 1, 101), density=True)
    axes[0, 1].set_xlabel("Probability")

    qqplot.ppplot(yearf, y, ax=axes[1, 0])
    qqplot.qqplot(yearf, y, ax=axes[1, 1], bins=12)
    qqplot.return_period_plot(yearf, y, ax=axes[2, 0])
    qqplot.return_level_plot(yearf, y, ax=axes[2, 1])

    # Distribution parameters comparison
    stations = data.meta["stations"]
    unique_climate_regions = np.unique(stations["climate_region"])
    climate_region_colors = {
        climate_region: CMAP(i / len(unique_climate_regions))
        for i, climate_region in enumerate(unique_climate_regions)
    }
    station_colors = {
        station: climate_region_colors[climate_region]
        for station, climate_region in zip(stations.index, stations["climate_region"])
    }

    fig, axes = plt.subplots(4, sharex=True, figsize=(8, 12))
    for station in data.labels:
        model = data.models[station.value]
        tt = np.linspace(0, 1)
        sged = model._models[1]._model if hasattr(model, "_models") else model
        m, s, l, p = sged.param_valuation(tt)
        c = station_colors[station.value]
        axes[0].plot(tt * DAYS_IN_YEAR, m, c=c)
        axes[1].plot(tt * DAYS_IN_YEAR, s, c=c)
        axes[2].plot(tt * DAYS_IN_YEAR, l, c=c)
        axes[3].plot(tt * DAYS_IN_YEAR, p, c=c)
    for clim_reg, c in climate_region_colors.items():
        axes[0].plot([], [], c=c, label=clim_reg)
    axes[0].legend(bbox_to_anchor=(1.0, 1), loc="upper left")
    for ax in axes:
        month_xaxis(ax)

    axes[0].set_ylabel(r"$\mu$")
    axes[1].set_ylabel(r"$\sigma$")
    axes[2].set_ylabel(r"$\lambda$")
    axes[3].set_ylabel("$p$")
    axes[1].set_ylim(0, None)
    axes[2].set_ylim(-1, 1)
    axes[3].set_ylim(1, None)
    axes[-1].set_xlim(0, DAYS_IN_YEAR)
    plt.show()

    # Distribution comparison
    fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
    for station in data.labels:
        y = data._raw_data[station]
        x = data._data[station]
        where = np.isfinite(x) & np.isfinite(y)
        x = x[where]
        y = y[where]
        p = np.arange(1, len(x) + 1) / (len(x) + 1)
        axes[0].plot(p, np.sort(x))
        axes[0].axline((0.5, 0.5), (0.7, 0.7), c="k", ls="--")
        axes[0].set_ylabel(r"$F(x_i;\theta(t_i))$")
        axes[0].set_xlabel(r"Uniform quantiles")
        axes[0].grid(which="both", alpha=0.5, lw=0.7)
        axes[0].set_xlim(0, 1)
        axes[0].set_ylim(0, 1)

        axes[1].plot(p, np.sort(x))
        axes[1].axline((0.5, 0.5), (0.7, 0.7), c="k", ls="--")
        axes[1].set_xscale("logit")
        axes[1].set_yscale("logit")
        axes[1].set_ylabel(r"$F(x_i;\theta(t_i))$")
        axes[1].set_xlabel(r"Uniform quantiles")
        axes[1].grid(which="major", alpha=0.5, lw=0.7)
        axes[1].grid(which="minor", alpha=0.2, lw=0.7)
        axes[1].set_xlim(1e-5, 1 - 1e-5)
        axes[1].set_ylim(1e-5, 1 - 1e-7)

    c = data.doy.values[where]
    axes[2].scatter(x, y, c=c, cmap="hsv", s=4)
    # axes[2].set_xscale("logit")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Probability")
    axes[2].set_ylabel("Value")
    fig.tight_layout()
