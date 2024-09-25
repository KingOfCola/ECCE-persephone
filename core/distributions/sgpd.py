import numpy as np
from scipy import stats
from scipy.optimize import minimize

from core.distributions.dist import Distribution
from core.distributions.fit_cdf import fit_cdf, _get_ppf_function, distance_quantile
from core.mathematics.functions import sigmoid, gpd_cdf, gpd_pdf


def g(x, eta):
    if eta == 0.0:
        return np.log()
    return np.log(1 + (np.log(1 + np.exp(-x))) ** (-1 / eta))


class SGPD(Distribution):
    _X0 = 0.0

    def __init__(
        self,
        ksi: float = 0.5,
        eta: float = 0.0,
        mu: float = 0.0,
        sigma: float = 1.0,
    ):
        super().__init__()
        self.ksi = ksi
        self.eta = eta
        self.sigma = sigma
        self.mu = mu
        self._update_lims()

    @property
    def params(self):
        return self.ksi, self.eta, self.mu, self.sigma

    def __str__(self):
        return f"SGPD(ksi={self.ksi}, eta={self.eta}, mu={self.mu}, sigma={self.sigma})"

    def cdf2(self, x: np.ndarray) -> np.ndarray:
        x0 = ((1 - 2**self.ksi) / self.ksi if self.ksi != 0.0 else -np.log(2)) + 1
        xn = (x - x0 - self.mu) / self.sigma

        gxn = np.piecewise(
            xn,
            [xn < 0.0, xn >= 0.0],
            [lambda x: 1 - gpd_cdf(-x, self.eta), lambda x: x + 1],
        )
        return gpd_cdf(gxn, self.ksi)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return SGPD._cdf(x, self.ksi, self.eta, self.mu, self.sigma)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return SGPD._pdf(x, self.ksi, self.eta, self.mu, self.sigma)

    @property
    def support(self):
        if self.eta >= 0:
            xmin = -np.inf
        else:
            xmin = self.mu + self.sigma / self.eta

        if self.ksi >= 0:
            xmax = np.inf
        else:
            xmax = self.mu + self.sigma / self.ksi

        return xmin, xmax

    def _update_lims2(self):
        x0 = ((1 - 2**self.ksi) / self.ksi if self.ksi != 0.0 else -np.log(2)) + 1
        pi = 1e-9
        if self.eta != 0.0:
            _xmin = -self.sigma * (pi ** (-self.eta) - 1) / self.eta
        else:
            _xmin = self.sigma * np.log(pi)

        if self.ksi != 0.0:
            _xmax = self.sigma * (pi ** (-self.ksi) - 1) / self.ksi
        else:
            _xmax = -self.sigma * np.log(pi)

        self._xmin = x0 + self.mu + _xmin
        self._xmax = x0 + self.mu + _xmax

    def _update_lims(self):
        pi = 1e-9
        if self.eta != 0.0:
            _xmin = -self.sigma * (pi ** (-self.eta) - 1) / self.eta
        else:
            _xmin = self.sigma * np.log(pi)

        if self.ksi != 0.0:
            _xmax = self.sigma * (pi ** (-self.ksi) - 1) / self.ksi
        else:
            _xmax = -self.sigma * np.log(pi)

        self._xmin = self.mu + _xmin
        self._xmax = self.mu + _xmax

    def fit(self, x: np.ndarray):
        mu = np.median(x)
        sigma = np.std(x)
        vksi = np.var(x[x > mu])
        veta = np.var(-x[x < mu])
        ksi = 1 - (sigma**2 / vksi) ** (1 / 3)
        eta = 1 - (sigma**2 / veta) ** (1 / 3)

        x0 = [ksi, eta, mu, sigma]
        for _ in range(20):
            if not np.isfinite(SGPD._neg_llhood_bounded(x0, x)):
                x0[0] = max(x0[0] / 1.2, x0[0])
                x0[1] = max(x0[1] / 1.2, x0[1])

        if not np.isfinite(SGPD._neg_llhood_bounded(x0, x)):
            x0 = [max(ksi, 0.0), max(eta, 0.0), mu, sigma]

        bounds = [(None, None), (None, None), (None, None), (0.0, None)]

        res = minimize(
            self._neg_llhood_bounded,
            x0=x0,
            args=(x,),
            # bounds=bounds,
        )
        self.fit_summary = res

        self.ksi, self.eta, self.mu, self.sigma = res.x
        self._update_lims()

    def fit_by_cdf(self, x: np.ndarray, quantile_method: str = "uniform"):
        # --------------------------------------------
        # Find reasonable estimates for the parameters
        mu = np.median(x)
        sigma = np.std(x)
        vksi = np.var(x[x > mu])
        veta = np.var(-x[x < mu])
        ksi = 1 - (sigma**2 / vksi) ** (1 / 3)
        eta = 1 - (sigma**2 / veta) ** (1 / 3)

        # Adjust the shape parameters to ensure that the likelihood is non-null
        params_0 = [ksi, eta, mu, sigma]
        for _ in range(20):
            if not np.isfinite(SGPD._neg_llhood_bounded(params_0, x)):
                params_0[0] = max(params_0[0] / 1.2, params_0[0])
                params_0[1] = max(params_0[1] / 1.2, params_0[1])

        if not np.isfinite(SGPD._neg_llhood_bounded(params_0, x)):
            params_0 = [max(ksi, 0.0), max(eta, 0.0), mu, sigma]

        # --------------------------------------------
        # Fit the distribution
        params = fit_cdf(
            SGPD._cdf,
            x,
            params_0,
            ppf=quantile_method,
            constraints=SGPD.constraint_method_factory(x),
            method="SLSQP",
        )

        # --------------------------------------------
        # Update the parameters
        self.ksi, self.eta, self.mu, self.sigma = params
        self._update_lims()

    @staticmethod
    def _pdf2(x, ksi, eta, mu, sigma) -> np.ndarray:
        x0 = ((1 - 2**ksi) / ksi if ksi != 0.0 else -np.log(2)) + 1
        xn = (x - x0 - mu) / sigma
        return (
            np.piecewise(
                xn,
                [xn < 0.0, xn >= 0.0],
                [
                    lambda x: gpd_pdf(-x, eta) * gpd_pdf(1 - gpd_cdf(-x, eta), ksi),
                    lambda x: gpd_pdf(x + 1, ksi),
                ],
            )
            / sigma
        )

    @staticmethod
    def _cdf(
        x: np.ndarray, ksi: float, eta: float, mu: float, sigma: float
    ) -> np.ndarray:
        xn = (x - mu) / sigma

        return np.piecewise(
            xn,
            [xn < 0.0, xn >= 0.0],
            [
                lambda x: 0.5 - gpd_cdf(-x, eta) / 2,
                lambda x: 0.5 + gpd_cdf(x, ksi) / 2,
            ],
        )

    @staticmethod
    def _pdf(x, ksi, eta, mu, sigma) -> np.ndarray:
        xn = (x - mu) / sigma

        return (
            np.piecewise(
                xn,
                [xn < 0.0, xn >= 0.0],
                [
                    lambda x: gpd_pdf(-x, eta) / 2,
                    lambda x: gpd_pdf(x, ksi) / 2,
                ],
            )
            / sigma
        )

    @staticmethod
    def _neg_llhood(params, x):
        ksi, eta, mu, sigma = params
        lhood = SGPD._pdf(x, ksi, eta, mu, sigma)
        llhood = np.zeros_like(lhood)
        where = lhood > 0
        llhood[where] = np.log(lhood[where])
        llhood[~where] = -np.inf

        return -np.sum(llhood)

    @staticmethod
    def _neg_llhood_bounded(params, x):
        ksi, eta, mu, sigma = params
        lhood = SGPD._pdf_bounded(x, ksi, eta, mu, sigma)
        llhood = np.zeros_like(lhood)
        where = lhood > 0
        llhood[where] = np.log(lhood[where])
        llhood[~where] = -np.inf

        return -np.sum(llhood)

    @staticmethod
    def _pdf_bounded(x, ksi, eta, mu, sigma):
        xn = (x - mu) / sigma

        return (
            np.piecewise(
                xn,
                [xn < 0.0, xn >= 0.0],
                [
                    lambda t: SGPD._gpd_pdf_bounded(-t, eta) / 2,
                    lambda t: SGPD._gpd_pdf_bounded(t, ksi) / 2,
                ],
            )
            / sigma
        )

    @staticmethod
    def _gpd_pdf_bounded(x, ksi):
        if ksi < -1:
            return np.piecewise(
                x,
                [x < 0, x <= (-1 / ksi), x >= (-1 / ksi)],
                [lambda t: 0.0, lambda t: (ksi**2 + ksi) * t**2 + t, lambda t: 0.0],
            )
        return gpd_pdf(x, ksi)

    @staticmethod
    def _quantile_distance(x, params, ppf):
        ppf_func = _get_ppf_function(ppf)
        return distance_quantile(SGPD._cdf, ppf=ppf_func, x=x, params=params)

    @staticmethod
    def constraint_method_factory(x):
        xmin = np.min(x)
        xmax = np.max(x)
        return [
            {
                "type": "ineq",
                "fun": SGPD.constraint_eta_factory(xmin),
                "jac": SGPD.constraint_eta_jac_factory(xmin),
            },
            {
                "type": "ineq",
                "fun": SGPD.constraint_ksi_factory(xmax),
                "jac": SGPD.constraint_ksi_jac_factory(xmax),
            },
            {
                "type": "ineq",
                "fun": SGPD.constraint_sigma,
                "jac": SGPD.constraint_sigma_jac,
            },
        ]

    @staticmethod
    def constraint_eta_factory(xmin):
        def constraint_eta(params):
            ksi, eta, mu, sigma = params
            return eta + sigma / (mu - xmin)

        return constraint_eta

    @staticmethod
    def constraint_ksi_factory(xmax):
        def constraint_ksi(params):
            ksi, eta, mu, sigma = params
            return ksi + sigma / (xmax - mu)

        return constraint_ksi

    @staticmethod
    def constraint_sigma(params):
        return params[3]

    @staticmethod
    def constraint_eta_jac_factory(xmin):
        def constraint_eta_jac(params):
            ksi, eta, mu, sigma = params
            return np.array([0, 1, -sigma / (mu - xmin) ** 2, 1 / (mu - xmin)])

        return constraint_eta_jac

    @staticmethod
    def constraint_ksi_jac_factory(xmax):
        def constraint_ksi_jac(params):
            ksi, eta, mu, sigma = params
            return np.array([1, 0, -sigma / (xmax - mu) ** 2, 1 / (xmax - mu)])

        return constraint_ksi_jac

    @staticmethod
    def constraint_sigma_jac(params):
        return np.array([0, 0, 0, 1])


class TTGPD(Distribution):
    def __init__(
        self,
        ksi: float = 0.5,
        eta: float = 0.0,
        mu: float = 0.0,
        sigma: float = 1.0,
    ):
        super().__init__()
        self.ksi = ksi
        self.eta = eta
        self.sigma = sigma
        self.mu = mu
        self._update_lims()

    @property
    def params(self):
        return self.ksi, self.eta, self.mu, self.sigma

    def __str__(self):
        return (
            f"TTGPD(ksi={self.ksi}, eta={self.eta}, mu={self.mu}, sigma={self.sigma})"
        )

    def cdf(self, x: np.ndarray) -> np.ndarray:
        xn = (x - self.mu) / self.sigma

        return np.piecewise(
            xn,
            [xn < 0.0, xn >= 0.0],
            [
                lambda x: 0.5 - gpd_cdf(-x, self.eta) / 2,
                lambda x: 0.5 + gpd_cdf(x, self.ksi) / 2,
            ],
        )

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return TTGPD._pdf(x, self.ksi, self.eta, self.mu, self.sigma)

    def _update_lims(self):
        pi = 1e-9
        if self.eta != 0.0:
            _xmin = -self.sigma * (pi ** (-self.eta) - 1) / self.eta
        else:
            _xmin = self.sigma * np.log(pi)

        if self.ksi != 0.0:
            _xmax = self.sigma * (pi ** (-self.ksi) - 1) / self.ksi
        else:
            _xmax = -self.sigma * np.log(pi)

        self._xmin = self.mu + _xmin
        self._xmax = self.mu + _xmax

    def fit(self, x: np.ndarray):
        mu = np.median(x)
        sigma = np.std(x)
        vksi = np.var(x[x > mu])
        veta = np.var(-x[x < mu])
        ksi = 1 - (sigma**2 / vksi) ** (1 / 3)
        eta = 1 - (sigma**2 / veta) ** (1 / 3)

        x0 = [ksi, eta, mu, sigma]
        for _ in range(20):
            if not np.isfinite(TTGPD._KL_divergence(x0, x)):
                x0[0] = max(x0[0] / 1.2, x0[0])
                x0[1] = max(x0[1] / 1.2, x0[1])

        if not np.isfinite(TTGPD._KL_divergence(x0, x)):
            x0 = [max(ksi, 0.0), max(eta, 0.0), mu, sigma]

        res = minimize(
            TTGPD._KL_divergence,
            x0=x0,
            args=(x,),
        )
        self.fit_summary = res

        self.ksi, self.eta, self.mu, self.sigma = res.x
        self._update_lims()

    @staticmethod
    def _pdf(x, ksi, eta, mu, sigma) -> np.ndarray:
        xn = (x - mu) / sigma

        return (
            np.piecewise(
                xn,
                [xn < 0.0, xn >= 0.0],
                [
                    lambda x: gpd_pdf(-x, eta) / 2,
                    lambda x: gpd_pdf(x, ksi) / 2,
                ],
            )
            / sigma
        )

    @staticmethod
    def _KL_divergence(params, x):
        ksi, eta, mu, sigma = params
        lhood = SGPD._pdf(x, ksi, eta, mu, sigma)
        llhood = np.zeros_like(lhood)
        where = lhood > 0
        llhood[where] = np.log(lhood[where]) * lhood[where]
        llhood[~where] = 0.0

        return -np.sum(llhood)
