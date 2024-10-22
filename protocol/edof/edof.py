# -*-coding:utf-8 -*-
"""
@File    :   edof.py
@Time    :   2024/10/09 13:05:51
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Protocol for the computation of effective degrees of freedom for different parameters
"""

from multiprocessing import Pool
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from core.random.ar_processes import (
    decimated_gaussian,
    decimated_gaussian_interp,
    garch_process,
    gaussian_ar_process,
    independent_process,
    warren_process,
    gaver_lewis_process,
)
from protocol.edof.edof_utils import compute_edof
from utils.iter import product_list, product_dict
from utils import mpp


class EDOFProtocol:
    def __init__(self, process_generator, process_args, process_kwargs, n_sim=1):
        self.process_generator = process_generator
        self.process_args = process_args
        self.process_kwargs = process_kwargs
        self.n_sim = n_sim

        self.__edof = np.full((self.n_params() * n_sim), np.nan)
        self.__bias = np.full((self.n_params() * n_sim), np.nan)
        self.__var = np.full((self.n_params() * n_sim), np.nan)

    def params_generator(self):
        for arg in product_list(*self.process_args):
            for kwarg in product_dict(**self.process_kwargs):
                yield self.process_generator, arg, kwarg

    def params_nsim_generator(self, n_sim):
        for x in self.params_generator():
            for _ in range(n_sim):
                yield x

    def run_simulations(self):
        params = list(self.params_nsim_generator(self.n_sim))
        with Pool() as pool:
            edof = tqdm(
                pool.istarmap(
                    compute_edof,
                    params,
                ),
                total=len(params),
                desc="Simulations",
                smoothing=0,
            )
            edof = [x for x in edof]
            self.__edof = np.array([x[0] for x in edof])
            self.__bias = np.array([x[1] for x in edof])
            self.__var = np.array([x[2] for x in edof])

    def n_params(self):
        """
        Returns the number of parameters of the grid computing the effective degrees of freedom

        Returns
        -------
        int
            The number of parameters of the grid computing the effective degrees of freedom
        """
        n = 1
        for arg in self.process_args:
            n *= 1 if isinstance(arg, (int, float)) else len(arg)
        for k, v in self.process_kwargs.items():
            n *= 1 if isinstance(v, (int, float)) else len(v)

        return n

    def edof_array(self):
        return self.__edof.reshape((self.n_params(), self.n_sim))

    def edof_df(self):
        edof_array = self.edof_array()
        data = [
            {
                "trial": j,
                **{f"arg_{k}": arg[k] for k in range(len(arg))},
                **kwarg,
                "edof": edof_array[i, j],
                "bias": self.__bias[i * self.n_sim + j],
                "var": self.__var[i * self.n_sim + j],
            }
            for i, (_, arg, kwarg) in enumerate(self.params_generator())
            for j in range(self.n_sim)
        ]
        return pd.DataFrame(data)

    def plot_edof(self, x: str, hue: str, ax: plt.Axes = None, cmap: str = "Spectral"):
        ax = ax or plt.gca()
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)

        sns.lineplot(
            data=self.edof_df(),
            x=x,
            y="edof",
            hue=hue,
            palette=cmap,
            ax=ax,
        )

        ax.set_title("Effective DoF")
        # ax.plot(rhos, 1 + (d-1) * (1 - rhos) ** 2 / (1 - rhos**2), c="k", label="Tentative")
        ax.set_ylabel("DoF $\\delta$")
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_minor_locator(MultipleLocator(1))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, None)
        ax.grid(True, axis="both", which="major", ls=":", lw=0.7, alpha=1)
        ax.grid(True, axis="both", which="minor", ls=":", lw=0.7, alpha=0.5)
        ax.legend()

        return ax

    def plot_edof_normalized(
        self, x: str, hue: str, ax: plt.Axes = None, cmap: str = "Spectral"
    ):

        ax = ax or plt.gca()
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)

        df = self.edof_df()
        df["edof_normalized"] = (df["edof"] - 1) / (df["w"] - 1)

        sns.lineplot(
            data=df,
            x=x,
            y="edof_normalized",
            hue=hue,
            palette=cmap,
            ax=ax,
        )
        ax.set_title("Effective DoF")
        ax.set_ylim(0, 1)
        # ax.plot(rhos, 1 + (d-1) * (1 - rhos) ** 2 / (1 - rhos**2), c="k", label="Tentative")
        ax.set_ylabel(r"Normalized DoF $\left(\frac{\delta - 1}{w - 1}\right)$")
        ax.legend()
        ax.grid(True, axis="both", which="major", ls=":", lw=0.7, alpha=1)
        return ax


def garch_process_rho(n, rho, w):
    return garch_process(n, np.array([0.1, rho]), np.array([0.99 - rho]), w)


if __name__ == "__main__":
    from utils.paths import output
    import os

    n = 30_000
    n_sim = 10
    w = np.arange(2, 11)
    rho = np.concatenate([1 - np.geomspace(0.01, 1.0, 30, endpoint=False), [0.0]])
    tau = np.linspace(1, 11, 21)
    alpha = [1]
    beta = [2]

    PARAMETERS_LEGENDS = {
        "rho": "Correlation $\\rho$",
        "tau": "Decimation factor $\\tau$",
        "trial": "Trial",
        "w": "Dimension $w$",
        "alpha": "Alpha $\\alpha$",
        "beta": "Beta $\\beta$",
        "edof": "Effective DoF $\\delta$",
    }

    default_args = ([n],)
    process_generators = {
        "Gaussian_AR1": {
            "process": gaussian_ar_process,
            "args": default_args,
            "kwargs": {"rho": rho, "w": w},
            "parameter": "rho",
        },
        "GARCH": {
            "process": garch_process_rho,
            "args": default_args,
            "kwargs": {"rho": rho, "w": w},
            "parameter": "rho",
        },
        "Warren": {
            "process": warren_process,
            "args": default_args,
            "kwargs": {"rho": rho, "w": w, "alpha": alpha, "beta": beta},
            "parameter": "rho",
        },
        "Gaver_Lewis": {
            "process": gaver_lewis_process,
            "args": default_args,
            "kwargs": {"rho": rho, "w": w, "alpha": alpha, "beta": beta},
            "parameter": "rho",
        },
        "Decimated_Gaussian": {
            "process": decimated_gaussian,
            "args": default_args,
            "kwargs": {"tau": tau, "w": w},
            "parameter": "tau",
        },
        "Decimated_Gaussian_Interpolation": {
            "process": decimated_gaussian_interp,
            "args": default_args,
            "kwargs": {"tau": tau, "w": w},
            "parameter": "tau",
        },
        "Independent": {
            "process": independent_process,
            "args": default_args,
            "kwargs": {"w": w},
            "parameter": "trial",
        },
    }

    for METHOD in list(process_generators.keys())[::-1]:
        print(f"Running {METHOD}")
        OUT_DIR = output(f"Simulations/EDOF/{METHOD}")
        os.makedirs(OUT_DIR, exist_ok=True)

        process_settings = process_generators[METHOD]
        process_generator = process_settings["process"]
        process_args = process_settings["args"]
        process_kwargs = process_settings["kwargs"]
        process_parameter = process_settings["parameter"]

        protocol = EDOFProtocol(
            process_generator=process_generator,
            process_args=process_args,
            process_kwargs=process_kwargs,
            n_sim=n_sim,
        )

        protocol.run_simulations()
        d = protocol.edof_df()
        p_min = d[process_parameter].min()
        p_max = d[process_parameter].max()

        fig, ax = plt.subplots()
        ax = protocol.plot_edof(process_parameter, "w", ax=ax, cmap="Spectral")
        fig.tight_layout()
        ax.set_xlim(p_min, p_max)
        ax.set_xlabel(PARAMETERS_LEGENDS.get(process_parameter, process_parameter))
        fig.savefig(os.path.join(OUT_DIR, "effective_dof.png"), dpi=300)
        plt.show()

        fig, ax = plt.subplots()
        ax = protocol.plot_edof_normalized(
            process_parameter, "w", ax=ax, cmap="Spectral"
        )
        ax.set_xlim(p_min, p_max)
        ax.set_xlabel(PARAMETERS_LEGENDS.get(process_parameter, process_parameter))
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "effective_dof_normalized.png"), dpi=300)
        plt.show()

        d.to_csv(os.path.join(OUT_DIR, "edof.csv"), index=False)
