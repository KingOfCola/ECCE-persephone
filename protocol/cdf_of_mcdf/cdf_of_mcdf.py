# -*-coding:utf-8 -*-
"""
@File    :   cdf_of_mcdf.py
@Time    :   2024/10/16 15:23:56
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Protocol for the computation of the CDF of the MCDF
"""

from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from core.data.confidence_intervals import ConfidenceInterval
from core.distributions.ecdf import ecdf_ci_binomial
from core.mathematics.functions import sigmoid
from core.optimization.mecdf import cdf_of_mcdf

from protocol.cdf_of_mcdf.utils import compute_pi_emp, predefined_generator


class CDFofMCDFProtocol:
    def __init__(
        self,
        process_generator,
        process_args=None,
        process_kwargs=None,
        w: int = None,
        n_sim=1,
    ):
        self.process_generator = process_generator
        self.process_args = process_args if process_args is not None else []
        self.process_kwargs = process_kwargs if process_kwargs is not None else {}

        self.w = w

        self.pi_emps = None
        self.dof_emps = None

        self.pi_emp = None
        self.dof_emp = None

        self.n_sim = (
            n_sim
            if not isinstance(process_generator, np.ndarray)
            else process_generator.shape[1]
        )

    def run_simulation(self):
        # Finds the number of dimensions and the number of samples
        if isinstance(self.process_generator, np.ndarray):
            w = self.w
        else:
            x = self.process_generator(*self.process_args, **self.process_kwargs)
            w = x.shape[1]

        self.w = w

        # Initialize the arrays to store the results
        self.pi_emps = [[] for _ in range(self.n_sim)]
        self.dof_emps = np.full(self.n_sim, np.nan)

        # Generate the parameters for the simulations
        if isinstance(self.process_generator, np.ndarray):
            params = [
                (predefined_generator, (self.process_generator, i, w), {})
                for i in range(self.process_generator.shape[1])
            ]
        else:
            params = [
                (self.process_generator, self.process_args, self.process_kwargs)
            ] * self.n_sim

        # Run the simulations
        with Pool() as pool:
            results = tqdm(
                pool.imap(
                    compute_pi_emp,
                    params,
                ),
                total=len(params),
                desc="Simulations",
                smoothing=0,
            )
            for i, x in enumerate(results):
                self.pi_emps[i] = x[0]
                self.dof_emps[i] = x[1]

        n = len(self.pi_emps[0])
        if all([len(pi_emp) == n for pi_emp in self.pi_emps]):
            self.pi_emps = np.array(self.pi_emps)

        # Compute the statistics of the simulations
        self.dof_emp = np.nanmean(self.dof_emps)

        if len(self.pi_emps) == 1:
            self.pi_emp = ecdf_ci_binomial(self.pi_emps[0])
        elif isinstance(self.pi_emps, np.ndarray):
            pemp_ci = np.nanpercentile(self.pi_emps, [2.5, 50, 97.5], axis=0)
            self.pi_emp = ConfidenceInterval(n)
            self.pi_emp.lower = pemp_ci[0]
            self.pi_emp.values = pemp_ci[1]
            self.pi_emp.upper = pemp_ci[2]
        else:
            self.pi_emp = None

    def plot_results(
        self, ax: plt.Axes = None, lim=1e-4, cmap="viridis", alpha=1, individual=False
    ):
        ax = plt.gca() if ax is None else ax
        llim = -np.log(lim)

        if self.pi_emp is not None and not individual:
            n = len(self.pi_emp)
            q = np.arange(1, n + 1) / (n + 1)
            ax.plot(self.pi_emp, q, c="blue", label="Empirical CDF of the MCDF")
            ax.fill_betweenx(
                q,
                np.clip(self.pi_emp.lower, lim, 1 - lim),
                np.clip(self.pi_emp.upper, lim, 1 - lim),
                color="blue",
                alpha=0.1,
            )
        else:
            p = len(self.pi_emps)
            if isinstance(cmap, str):
                cmap = plt.get_cmap(cmap)

            for i in range(p):
                n = len(self.pi_emps[i])
                q = np.arange(1, n + 1) / (n + 1)
                ax.plot(
                    self.pi_emps[i],
                    q,
                    c=cmap(i / p),
                    alpha=alpha,
                    lw=0.7,
                    label="Empirical CDF of the MCDF" if i == 0 else None,
                )

        q = sigmoid(np.linspace(-llim, llim, 1000))
        for i in range(1, self.w + 1):
            y = cdf_of_mcdf(q, i)
            y0 = cdf_of_mcdf(np.array([lim]), i)[0]
            ax.plot(
                q,
                y,
                c="k",
                ls="--",
                lw=0.7,
                label="Independent dimensions" if i == 1 else None,
            )
            ax.annotate(
                f"$\delta={i}$",
                (lim, y0),
                # (0, -5),
                # textcoords="offset points",
                ha="left",
                va="top",
            )
        ax.plot(
            q,
            cdf_of_mcdf(q, self.dof_emp),
            lw=0.7,
            c="red",
            label=rf"Effective DOF $\delta={self.dof_emp:.2f}$",
        )
        ax.set_xscale("logit")
        ax.set_yscale("logit")
        ax.grid(True, axis="both", which="major", ls=":", lw=0.7, alpha=1)
        ax.grid(True, axis="both", which="minor", ls=":", lw=0.7, alpha=0.5)
        ax.set_xlim(lim, 1 - lim)
        ax.set_ylim(lim * 0.5, 1 - lim)
        ax.set(xlabel="$t$", ylabel=r"$\mathbb{P}(F_{\mathbf{X}}(\mathbf{X}) \leq t)$")
        ax.legend(loc="lower right")

    def plot_results_individual(
        self, ax: plt.Axes = None, lim=1e-4, cmap="viridis", alpha=1
    ):
        ax = plt.gca() if ax is None else ax
        n = self.pi_emps.shape[0]
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)

        for i in range(n):
            ax.plot(
                self.pi_emps[i],
                self.q,
                c=cmap(i / n),
                alpha=alpha,
                lw=0.7,
                label="Empirical CDF of the MCDF" if i == 0 else None,
            )
        for i in range(1, self.w + 1):
            y = cdf_of_mcdf(self.q, i)
            y0 = cdf_of_mcdf(np.array([lim]), i)[0]
            ax.plot(
                self.q,
                y,
                c="k",
                ls="--",
                lw=0.7,
                label="Independent dimensions" if i == 1 else None,
            )
            ax.annotate(
                f"$\delta={i}$",
                (lim, y0),
                # (0, -5),
                # textcoords="offset points",
                ha="left",
                va="top",
            )
        ax.plot(
            self.q,
            cdf_of_mcdf(self.q, self.dof_emp),
            lw=1.5,
            c="red",
            label=rf"Effective DOF $\delta={self.dof_emp:.2f}$",
        )
        ax.set_xscale("logit")
        ax.set_yscale("logit")
        ax.grid(True, axis="both", which="major", ls=":", lw=0.7, alpha=1)
        ax.grid(True, axis="both", which="minor", ls=":", lw=0.7, alpha=0.5)
        ax.set_xlim(lim, 1 - lim)
        ax.set_ylim(lim * 0.5, 1 - lim)
        ax.set(xlabel="$t$", ylabel=r"$\mathbb{P}(F_{\mathbf{X}}(\mathbf{X}) \leq t)$")
        ax.legend(loc="lower right")


if __name__ == "__main__":
    import seaborn as sns
    import matplotlib.pyplot as plt
    from utils.paths import output
    import os

    from core.random.ar_processes import (
        garch_process_rho,
        independent_process,
        gaussian_ar_process,
        warren_process,
        gaver_lewis_process,
        decimated_gaussian,
        decimated_gaussian_interp,
    )
    from utils.loaders.synop_loader import load_fit_synop

    METHOD = "SYNOP_preliq-MAX"
    OUT_DIR = output(f"Simulations/EDOF/{METHOD}")
    os.makedirs(OUT_DIR, exist_ok=True)

    meteorological_metrics = ["t_MAX", "t_MIN", "t_AVG", "preliq_SUM", "preliq_MAX"]
    SYNOP_DATA = {
        metric: load_fit_synop(metric).data.values
        for metric in meteorological_metrics[4:5]
    }
    SYNOP_GENERATORS = {
        ("SYNOP_" + metric.replace("_", "-")): {
            "process": 1 - data,
            "args": None,
            "kwargs": None,
        }
        for metric, data in SYNOP_DATA.items()
    }

    n = 10_000
    w = 10
    rho = 0.9
    tau = np.pi
    alpha = 1
    beta = 2

    default_args = (n,)
    process_generators = {
        "Gaussian_AR1": {
            "process": gaussian_ar_process,
            "args": default_args,
            "kwargs": {"rho": rho, "w": w},
        },
        "GARCH": {
            "process": garch_process_rho,
            "args": default_args,
            "kwargs": {"rho": rho, "w": w},
        },
        "Warren": {
            "process": warren_process,
            "args": default_args,
            "kwargs": {"rho": rho, "w": w, "alpha": alpha, "beta": beta},
        },
        "Gaver_Lewis": {
            "process": gaver_lewis_process,
            "args": default_args,
            "kwargs": {"rho": rho, "w": w, "alpha": alpha, "beta": beta},
        },
        "Decimated_Gaussian": {
            "process": decimated_gaussian,
            "args": default_args,
            "kwargs": {"tau": tau, "w": w},
        },
        "Decimated_Gaussian_Interpolation": {
            "process": decimated_gaussian_interp,
            "args": default_args,
            "kwargs": {"tau": tau, "w": w},
        },
        "Independent": {
            "process": independent_process,
            "args": default_args,
            "kwargs": {"w": w},
        },
    }
    process_generators.update(SYNOP_GENERATORS)

    process_settings = process_generators[METHOD]
    process_generator = process_settings["process"]
    process_args = process_settings["args"]
    process_kwargs = process_settings["kwargs"]

    for w in [2, 3, 4, 5, 7, 10]:
        protocol = CDFofMCDFProtocol(
            process_generator=process_generator,
            process_args=process_args,
            process_kwargs=process_kwargs,
            n_sim=100,
            w=w,
        )

        protocol.run_simulation()

        fig, ax = plt.subplots(figsize=(8, 8))
        protocol.plot_results_individual(ax)
        plt.show()
        fig.tight_layout()
        fig.savefig(
            os.path.join(OUT_DIR, f"cdf_of_mcdf_vs_effective_independent_w{w}.png"),
            dpi=300,
        )
