"""Analytical Black-Scholes Greeks (first- and second-order sensitivities).

All Greeks are derived by differentiating the closed-form Black-Scholes price
(see :mod:`pyfinlib_practice.models.black_scholes`). With ``phi`` the normal PDF,
``N`` the normal CDF, and ``d1, d2`` as defined there (RESULT):

    delta_call =  e^{-qT} N(d1)            delta_put = -e^{-qT} N(-d1)
    gamma      =  e^{-qT} phi(d1) / (S sigma sqrt(T))     [same for call/put]
    vega       =  S e^{-qT} phi(d1) sqrt(T)               [same for call/put]
    rho_call   =  K T e^{-rT} N(d2)        rho_put  = -K T e^{-rT} N(-d2)

Theta is the negative time derivative (value decay per unit calendar time):
    theta_call = -S e^{-qT} phi(d1) sigma / (2 sqrt(T))
                 - r K e^{-rT} N(d2) + q S e^{-qT} N(d1)
    theta_put  = -S e^{-qT} phi(d1) sigma / (2 sqrt(T))
                 + r K e^{-rT} N(-d2) - q S e^{-qT} N(-d1)

CONVENTION: ``vega`` and ``rho`` are per unit (i.e. per 1.00 = 100 vol points /
100% rate). ``theta`` is per year. Divide by 100 / 365 for the usual desk units.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.special import ndtr  # type: ignore[import-untyped]

from pyfinlib_practice.models.black_scholes import d1_d2, norm_pdf

OptionType = Literal["call", "put"]


def _norm_cdf(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.asarray(ndtr(x), dtype=np.float64)


def delta(
    spot: npt.ArrayLike,
    strike: npt.ArrayLike,
    r: npt.ArrayLike,
    sigma: npt.ArrayLike,
    t: npt.ArrayLike,
    *,
    option_type: OptionType = "call",
    q: npt.ArrayLike = 0.0,
) -> npt.NDArray[np.float64]:
    """Sensitivity of price to spot, ``dV/dS``."""
    ta = np.asarray(t, dtype=np.float64)
    qa = np.asarray(q, dtype=np.float64)
    d1, _ = d1_d2(spot, strike, r, sigma, ta, q=qa)
    disc_q = np.exp(-qa * ta)
    if option_type == "call":
        return disc_q * _norm_cdf(d1)
    return -disc_q * _norm_cdf(-d1)


def gamma(
    spot: npt.ArrayLike,
    strike: npt.ArrayLike,
    r: npt.ArrayLike,
    sigma: npt.ArrayLike,
    t: npt.ArrayLike,
    *,
    q: npt.ArrayLike = 0.0,
) -> npt.NDArray[np.float64]:
    """Second derivative of price w.r.t. spot, ``d2V/dS2`` (call = put)."""
    s = np.asarray(spot, dtype=np.float64)
    sig = np.asarray(sigma, dtype=np.float64)
    ta = np.asarray(t, dtype=np.float64)
    qa = np.asarray(q, dtype=np.float64)
    d1, _ = d1_d2(s, strike, r, sig, ta, q=qa)
    return np.exp(-qa * ta) * norm_pdf(d1) / (s * sig * np.sqrt(ta))


def vega(
    spot: npt.ArrayLike,
    strike: npt.ArrayLike,
    r: npt.ArrayLike,
    sigma: npt.ArrayLike,
    t: npt.ArrayLike,
    *,
    q: npt.ArrayLike = 0.0,
) -> npt.NDArray[np.float64]:
    """Sensitivity of price to volatility, ``dV/dsigma`` (call = put)."""
    s = np.asarray(spot, dtype=np.float64)
    ta = np.asarray(t, dtype=np.float64)
    qa = np.asarray(q, dtype=np.float64)
    d1, _ = d1_d2(s, strike, r, sigma, ta, q=qa)
    return s * np.exp(-qa * ta) * norm_pdf(d1) * np.sqrt(ta)


def theta(
    spot: npt.ArrayLike,
    strike: npt.ArrayLike,
    r: npt.ArrayLike,
    sigma: npt.ArrayLike,
    t: npt.ArrayLike,
    *,
    option_type: OptionType = "call",
    q: npt.ArrayLike = 0.0,
) -> npt.NDArray[np.float64]:
    """Sensitivity of price to calendar time, ``dV/dt`` (per year)."""
    s = np.asarray(spot, dtype=np.float64)
    k = np.asarray(strike, dtype=np.float64)
    ra = np.asarray(r, dtype=np.float64)
    sig = np.asarray(sigma, dtype=np.float64)
    ta = np.asarray(t, dtype=np.float64)
    qa = np.asarray(q, dtype=np.float64)
    d1, d2 = d1_d2(s, k, ra, sig, ta, q=qa)

    decay = -s * np.exp(-qa * ta) * norm_pdf(d1) * sig / (2.0 * np.sqrt(ta))
    if option_type == "call":
        return (
            decay
            - ra * k * np.exp(-ra * ta) * _norm_cdf(d2)
            + qa * s * np.exp(-qa * ta) * _norm_cdf(d1)
        )
    return (
        decay
        + ra * k * np.exp(-ra * ta) * _norm_cdf(-d2)
        - qa * s * np.exp(-qa * ta) * _norm_cdf(-d1)
    )


def rho(
    spot: npt.ArrayLike,
    strike: npt.ArrayLike,
    r: npt.ArrayLike,
    sigma: npt.ArrayLike,
    t: npt.ArrayLike,
    *,
    option_type: OptionType = "call",
    q: npt.ArrayLike = 0.0,
) -> npt.NDArray[np.float64]:
    """Sensitivity of price to the risk-free rate, ``dV/dr``."""
    k = np.asarray(strike, dtype=np.float64)
    ra = np.asarray(r, dtype=np.float64)
    ta = np.asarray(t, dtype=np.float64)
    _, d2 = d1_d2(spot, k, ra, sigma, ta, q=q)
    if option_type == "call":
        return k * ta * np.exp(-ra * ta) * _norm_cdf(d2)
    return -k * ta * np.exp(-ra * ta) * _norm_cdf(-d2)
