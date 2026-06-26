"""The Black-Scholes-Merton closed-form option pricing model.

Under the BSM assumptions the time-0 value of a European option with spot ``S``,
strike ``K``, rate ``r``, continuous dividend yield ``q``, volatility ``sigma``
and maturity ``T`` is (RESULT, via Feynman-Kac / the risk-neutral expectation):

    call = S e^{-qT} N(d1) - K e^{-rT} N(d2),
    put  = K e^{-rT} N(-d2) - S e^{-qT} N(-d1),

with
    d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt(T)),
    d2 = d1 - sigma sqrt(T),

and ``N`` the standard normal CDF.

All functions are vectorised over array-like inputs and handle the degenerate
limit ``sigma sqrt(T) -> 0`` by returning the discounted forward intrinsic value,
so no ``inf``/``nan`` leaks for zero volatility or zero maturity.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.special import ndtr  # type: ignore[import-untyped]

OptionType = Literal["call", "put"]

_SQRT_2PI = float(np.sqrt(2.0 * np.pi))


def _norm_cdf(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Standard normal CDF (scipy ``ndtr`` is the standard, fast choice)."""
    return np.asarray(ndtr(x), dtype=np.float64)


def norm_pdf(x: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Standard normal PDF ``phi(x) = exp(-x^2/2) / sqrt(2 pi)``."""
    xa = np.asarray(x, dtype=np.float64)
    return np.exp(-0.5 * xa**2) / _SQRT_2PI


def d1_d2(
    spot: npt.ArrayLike,
    strike: npt.ArrayLike,
    r: npt.ArrayLike,
    sigma: npt.ArrayLike,
    t: npt.ArrayLike,
    *,
    q: npt.ArrayLike = 0.0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the Black-Scholes ``d1`` and ``d2`` terms (broadcast over inputs).

    Where ``sigma sqrt(T) == 0`` the returned values are not meaningful; callers
    must guard that branch separately (the price functions here do).
    """
    s = np.asarray(spot, dtype=np.float64)
    k = np.asarray(strike, dtype=np.float64)
    ra = np.asarray(r, dtype=np.float64)
    sig = np.asarray(sigma, dtype=np.float64)
    ta = np.asarray(t, dtype=np.float64)
    qa = np.asarray(q, dtype=np.float64)

    vol_sqrt_t = sig * np.sqrt(ta)
    # Avoid division warnings; the degenerate branch is overwritten by callers.
    denom = np.where(vol_sqrt_t > 0.0, vol_sqrt_t, 1.0)
    d1 = (np.log(s / k) + (ra - qa + 0.5 * sig**2) * ta) / denom
    d2 = d1 - vol_sqrt_t
    return d1, d2


def black_scholes_price(
    spot: npt.ArrayLike,
    strike: npt.ArrayLike,
    r: npt.ArrayLike,
    sigma: npt.ArrayLike,
    t: npt.ArrayLike,
    *,
    option_type: OptionType = "call",
    q: npt.ArrayLike = 0.0,
) -> npt.NDArray[np.float64]:
    """Black-Scholes price of a European call or put (vectorised)."""
    s = np.asarray(spot, dtype=np.float64)
    k = np.asarray(strike, dtype=np.float64)
    ra = np.asarray(r, dtype=np.float64)
    sig = np.asarray(sigma, dtype=np.float64)
    ta = np.asarray(t, dtype=np.float64)
    qa = np.asarray(q, dtype=np.float64)

    disc_s = s * np.exp(-qa * ta)
    disc_k = k * np.exp(-ra * ta)
    d1, d2 = d1_d2(s, k, ra, sig, ta, q=qa)
    vol_sqrt_t = sig * np.sqrt(ta)
    non_degenerate = vol_sqrt_t > 0.0

    if option_type == "call":
        general = disc_s * _norm_cdf(d1) - disc_k * _norm_cdf(d2)
        degenerate = np.maximum(disc_s - disc_k, 0.0)
    else:
        general = disc_k * _norm_cdf(-d2) - disc_s * _norm_cdf(-d1)
        degenerate = np.maximum(disc_k - disc_s, 0.0)

    return np.where(non_degenerate, general, degenerate)


def put_call_parity_residual(
    call: npt.ArrayLike,
    put: npt.ArrayLike,
    spot: npt.ArrayLike,
    strike: npt.ArrayLike,
    r: npt.ArrayLike,
    t: npt.ArrayLike,
    *,
    q: npt.ArrayLike = 0.0,
) -> npt.NDArray[np.float64]:
    """Put-call parity residual ``C - P - (S e^{-qT} - K e^{-rT})``.

    Exactly zero for arbitrage-consistent prices (RESULT); useful as a test.
    """
    c = np.asarray(call, dtype=np.float64)
    p = np.asarray(put, dtype=np.float64)
    s = np.asarray(spot, dtype=np.float64)
    k = np.asarray(strike, dtype=np.float64)
    ra = np.asarray(r, dtype=np.float64)
    ta = np.asarray(t, dtype=np.float64)
    qa = np.asarray(q, dtype=np.float64)
    return c - p - (s * np.exp(-qa * ta) - k * np.exp(-ra * ta))


def digital_price(
    spot: npt.ArrayLike,
    strike: npt.ArrayLike,
    r: npt.ArrayLike,
    sigma: npt.ArrayLike,
    t: npt.ArrayLike,
    *,
    option_type: OptionType = "call",
    q: npt.ArrayLike = 0.0,
) -> npt.NDArray[np.float64]:
    """Cash-or-nothing digital option paying one unit of cash if in the money.

    call = e^{-rT} N(d2),  put = e^{-rT} N(-d2)  (RESULT).
    """
    ra = np.asarray(r, dtype=np.float64)
    ta = np.asarray(t, dtype=np.float64)
    _, d2 = d1_d2(spot, strike, ra, sigma, ta, q=q)
    disc = np.exp(-ra * ta)
    if option_type == "call":
        return disc * _norm_cdf(d2)
    return disc * _norm_cdf(-d2)


def butterfly_price(
    spot: npt.ArrayLike,
    strike: npt.ArrayLike,
    r: npt.ArrayLike,
    sigma: npt.ArrayLike,
    t: npt.ArrayLike,
    spread: float,
    *,
    q: npt.ArrayLike = 0.0,
) -> npt.NDArray[np.float64]:
    """Long call butterfly: ``C(K - h) - 2 C(K) + C(K + h)`` with ``h = spread``.

    A discrete second difference in strike; up to ``O(h^2)`` it is proportional
    to the risk-neutral density at ``K`` (the Breeden-Litzenberger link). Always
    non-negative for arbitrage-free prices. ``spread`` must be positive.
    """
    if spread <= 0.0:
        raise ValueError("spread must be positive")
    k = np.asarray(strike, dtype=np.float64)
    lower = black_scholes_price(spot, k - spread, r, sigma, t, option_type="call", q=q)
    mid = black_scholes_price(spot, k, r, sigma, t, option_type="call", q=q)
    upper = black_scholes_price(spot, k + spread, r, sigma, t, option_type="call", q=q)
    return lower - 2.0 * mid + upper
