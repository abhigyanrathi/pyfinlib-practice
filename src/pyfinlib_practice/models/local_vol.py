"""Local volatility building blocks: implied volatility and Breeden-Litzenberger.

This module covers the mentor-confirmed Chapter 4 scope:

* **Section 4.1 - Black-Scholes implied volatility.** Given a market option
  price, recover the volatility ``sigma`` such that the Black-Scholes price
  equals the market price. Because the Black-Scholes price is strictly
  increasing in ``sigma`` on ``(0, infinity)`` for a price strictly inside the
  no-arbitrage bounds, the root is unique (RESULT). Three root-finders are
  provided: Newton-Raphson, the secant method, and a bracketed Newton/bisection
  hybrid with guaranteed convergence.

* **Section 4.2 - Option prices and densities.** Breeden-Litzenberger: the
  risk-neutral density is the discounted second strike-derivative of the call
  price, ``q(K) = e^{rT} d^2 C / dK^2`` (RESULT). Given that density, any
  European payoff can be valued by quadrature.

CONVENGENCE NOTE (important): the iterative solvers test convergence on the
*volatility step* ``|sigma_{n+1} - sigma_n| < xtol``, **not** on the price
residual. A price-residual stopping rule is unreliable for deep in/out-of-the
money options, where vega is tiny: a large change in ``sigma`` moves the price
by almost nothing, so a small price residual can be reached while ``sigma`` is
still far from the true implied volatility.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import numpy.typing as npt

from pyfinlib_practice.models.black_scholes import OptionType, black_scholes_price
from pyfinlib_practice.pricing.greeks import vega

Method = Literal["newton", "secant", "bracketed"]

_MIN_VEGA = 1e-12
_SIGMA_FLOOR = 1e-8


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _no_arbitrage_bounds(
    spot: float, strike: float, r: float, t: float, option_type: OptionType, q: float
) -> tuple[float, float]:
    """Lower/upper no-arbitrage price bounds for a European option (RESULT)."""
    disc_s = spot * np.exp(-q * t)
    disc_k = strike * np.exp(-r * t)
    if option_type == "call":
        return max(disc_s - disc_k, 0.0), disc_s
    return max(disc_k - disc_s, 0.0), disc_k


def _validate_target_price(
    price: float, spot: float, strike: float, r: float, t: float,
    option_type: OptionType, q: float,
) -> None:
    lower, upper = _no_arbitrage_bounds(spot, strike, r, t, option_type, q)
    # Small tolerance so a price exactly on a bound is admissible.
    if not (lower - 1e-12 <= price <= upper + 1e-12):
        raise ValueError(
            f"price {price:.6g} is outside the no-arbitrage bounds "
            f"[{lower:.6g}, {upper:.6g}]; no implied volatility exists"
        )


def _make_price_error(
    target: float, spot: float, strike: float, r: float, t: float,
    option_type: OptionType, q: float,
) -> Callable[[float], float]:
    """Return f(sigma) = BlackScholes(sigma) - target, as a scalar function."""

    def price_error(sigma: float) -> float:
        model = float(
            black_scholes_price(spot, strike, r, sigma, t, option_type=option_type, q=q)
        )
        return model - target

    return price_error


def _brenner_subrahmanyam_guess(price: float, spot: float, t: float) -> float:
    """At-the-money implied-vol approximation, used as a starting point.

    ``sigma ~ sqrt(2 pi / T) * price / S`` (RESULT, Brenner-Subrahmanyam 1988).
    """
    guess = float(np.sqrt(2.0 * np.pi / t) * price / spot)
    return min(max(guess, 1e-3), 5.0)


# --------------------------------------------------------------------------- #
# Implied volatility solvers
# --------------------------------------------------------------------------- #
def implied_volatility_newton(
    price: float, spot: float, strike: float, r: float, t: float,
    *, option_type: OptionType = "call", q: float = 0.0,
    initial_guess: float | None = None, xtol: float = 1e-8, max_iter: int = 100,
) -> float:
    """Implied volatility by Newton-Raphson, using vega as the derivative.

    Iteration: ``sigma <- sigma - (BS(sigma) - price) / vega(sigma)``.
    Raises ``RuntimeError`` if vega collapses or convergence is not reached;
    for deep ITM/OTM quotes prefer :func:`implied_volatility_bracketed`.
    """
    _validate_target_price(price, spot, strike, r, t, option_type, q)
    f = _make_price_error(price, spot, strike, r, t, option_type, q)
    sigma = initial_guess if initial_guess is not None else _brenner_subrahmanyam_guess(
        price, spot, t
    )

    for _ in range(max_iter):
        v = float(vega(spot, strike, r, sigma, t, q=q))
        if v < _MIN_VEGA:
            raise RuntimeError("vega too small for Newton; use the bracketed solver")
        sigma_next = sigma - f(sigma) / v
        if sigma_next <= 0.0:
            sigma_next = sigma / 2.0  # damp back into the positive region
        if abs(sigma_next - sigma) < xtol:
            return sigma_next
        sigma = sigma_next
    raise RuntimeError("Newton-Raphson failed to converge")


def implied_volatility_secant(
    price: float, spot: float, strike: float, r: float, t: float,
    *, option_type: OptionType = "call", q: float = 0.0,
    initial_guess: float | None = None, xtol: float = 1e-8, max_iter: int = 100,
) -> float:
    """Implied volatility by the secant method (no analytic derivative needed)."""
    _validate_target_price(price, spot, strike, r, t, option_type, q)
    f = _make_price_error(price, spot, strike, r, t, option_type, q)
    s0 = initial_guess if initial_guess is not None else _brenner_subrahmanyam_guess(
        price, spot, t
    )
    s1 = s0 * 1.1 + 1e-3
    f0 = f(s0)

    for _ in range(max_iter):
        f1 = f(s1)
        denom = f1 - f0
        if abs(denom) < 1e-14:
            raise RuntimeError("secant step ill-conditioned (flat price difference)")
        s2 = s1 - f1 * (s1 - s0) / denom
        if s2 <= 0.0:
            s2 = s1 / 2.0
        if abs(s2 - s1) < xtol:
            return s2
        s0, f0 = s1, f1
        s1 = s2
    raise RuntimeError("secant method failed to converge")


def implied_volatility_bracketed(
    price: float, spot: float, strike: float, r: float, t: float,
    *, option_type: OptionType = "call", q: float = 0.0,
    xtol: float = 1e-8, max_iter: int = 100,
) -> float:
    """Robust implied volatility: Newton step inside a maintained bracket, else bisect.

    A bracket ``[lo, hi]`` with ``f(lo) <= 0 <= f(hi)`` is established and never
    lost. Each step tries a Newton update; if it lands outside the bracket it is
    rejected in favour of a bisection step. This converges for any admissible
    price, including the deep wings where pure Newton fails.
    """
    _validate_target_price(price, spot, strike, r, t, option_type, q)
    f = _make_price_error(price, spot, strike, r, t, option_type, q)

    lo, hi = _SIGMA_FLOOR, 5.0
    f_hi = f(hi)
    while f_hi < 0.0 and hi < 1e3:  # expand upper bound if price implies very high vol
        hi *= 2.0
        f_hi = f(hi)
    f_lo = f(lo)
    if f_lo > 0.0 or f_hi < 0.0:
        raise ValueError("could not bracket the implied volatility for this price")

    sigma = 0.5 * (lo + hi)
    for _ in range(max_iter):
        fs = f(sigma)
        if fs > 0.0:
            hi = sigma
        else:
            lo = sigma

        v = float(vega(spot, strike, r, sigma, t, q=q))
        newton = sigma - fs / v if v > _MIN_VEGA else float("inf")
        sigma_next = newton if lo < newton < hi else 0.5 * (lo + hi)

        if abs(sigma_next - sigma) < xtol or (hi - lo) < xtol:
            return sigma_next
        sigma = sigma_next
    raise RuntimeError("bracketed solver failed to converge")


def implied_volatility(
    price: float, spot: float, strike: float, r: float, t: float,
    *, option_type: OptionType = "call", q: float = 0.0,
    method: Method = "bracketed", xtol: float = 1e-8, max_iter: int = 100,
) -> float:
    """Dispatch to one of the implied-volatility solvers (default: bracketed)."""
    if method == "newton":
        return implied_volatility_newton(
            price, spot, strike, r, t, option_type=option_type, q=q,
            xtol=xtol, max_iter=max_iter,
        )
    if method == "secant":
        return implied_volatility_secant(
            price, spot, strike, r, t, option_type=option_type, q=q,
            xtol=xtol, max_iter=max_iter,
        )
    return implied_volatility_bracketed(
        price, spot, strike, r, t, option_type=option_type, q=q,
        xtol=xtol, max_iter=max_iter,
    )


# --------------------------------------------------------------------------- #
# Breeden-Litzenberger (Section 4.2)
# --------------------------------------------------------------------------- #
def risk_neutral_density(
    strikes: npt.ArrayLike,
    call_prices: npt.ArrayLike,
    r: float,
    t: float,
) -> npt.NDArray[np.float64]:
    """Risk-neutral density from call prices via Breeden-Litzenberger.

    ``q(K) = e^{rT} d^2 C / dK^2`` (RESULT), with the second derivative taken
    numerically. ``strikes`` must be sorted ascending; non-uniform spacing is
    handled by ``numpy.gradient``.
    """
    k = np.asarray(strikes, dtype=np.float64)
    c = np.asarray(call_prices, dtype=np.float64)
    if k.ndim != 1 or k.shape != c.shape:
        raise ValueError("strikes and call_prices must be 1-D arrays of equal length")
    if k.size < 3:
        raise ValueError("need at least three strikes for a second derivative")

    first = np.gradient(c, k)
    second = np.gradient(first, k)
    return np.asarray(np.exp(r * t) * second, dtype=np.float64)


def price_from_density(
    payoff_values: npt.ArrayLike,
    asset_grid: npt.ArrayLike,
    density: npt.ArrayLike,
    r: float,
    t: float,
) -> float:
    """Value a European payoff against a risk-neutral density by quadrature.

    ``V = e^{-rT} integral payoff(S) q(S) dS`` (RESULT), evaluated with the
    trapezoidal rule over ``asset_grid``.
    """
    payoff = np.asarray(payoff_values, dtype=np.float64)
    grid = np.asarray(asset_grid, dtype=np.float64)
    dens = np.asarray(density, dtype=np.float64)
    if not (payoff.shape == grid.shape == dens.shape):
        raise ValueError("payoff_values, asset_grid and density must share a shape")
    integrand = payoff * dens
    return float(np.exp(-r * t) * np.trapezoid(integrand, grid))
