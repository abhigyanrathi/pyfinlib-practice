"""Option payoff functions and simple profit-and-loss (P&L) helpers.

All functions are vectorised: ``spot`` (and where relevant ``strike``) may be a
scalar or any array-like, and the result is a float64 array broadcast to the
common shape.

Conventions (CONVENTION):
* A *digital* (a.k.a. binary, cash-or-nothing) option pays one unit of cash if
  it finishes in the money and zero otherwise.
* P&L is expressed per unit of the underlying contract, measured at expiry, and
  ignores the time value of the premium (no discounting of the paid premium).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

Position = Literal["long", "short"]


def call_payoff(spot: npt.ArrayLike, strike: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """European call payoff ``max(S - K, 0)`` (RESULT)."""
    s = np.asarray(spot, dtype=np.float64)
    k = np.asarray(strike, dtype=np.float64)
    return np.maximum(s - k, 0.0)


def put_payoff(spot: npt.ArrayLike, strike: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """European put payoff ``max(K - S, 0)`` (RESULT)."""
    s = np.asarray(spot, dtype=np.float64)
    k = np.asarray(strike, dtype=np.float64)
    return np.maximum(k - s, 0.0)


def digital_call_payoff(
    spot: npt.ArrayLike, strike: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Cash-or-nothing call: pays 1 if ``S > K`` else 0."""
    s = np.asarray(spot, dtype=np.float64)
    k = np.asarray(strike, dtype=np.float64)
    return (s > k).astype(np.float64)


def digital_put_payoff(
    spot: npt.ArrayLike, strike: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Cash-or-nothing put: pays 1 if ``S < K`` else 0."""
    s = np.asarray(spot, dtype=np.float64)
    k = np.asarray(strike, dtype=np.float64)
    return (s < k).astype(np.float64)


def option_pnl(
    spot: npt.ArrayLike,
    strike: float,
    premium: float,
    *,
    option_type: Literal["call", "put"],
    position: Position,
) -> npt.NDArray[np.float64]:
    """Profit-and-loss at expiry for a single vanilla option.

    For a long position the holder pays ``premium`` upfront, so
    ``P&L = payoff - premium``. A short position is the negative of the long
    position's P&L. ``premium`` must be non-negative.
    """
    if premium < 0.0:
        raise ValueError("premium must be non-negative")
    payoff = call_payoff(spot, strike) if option_type == "call" else put_payoff(spot, strike)
    long_pnl = payoff - premium
    return long_pnl if position == "long" else -long_pnl


def breakeven(
    strike: float, premium: float, *, option_type: Literal["call", "put"]
) -> float:
    """Underlying price at which a long vanilla option breaks even (RESULT).

    Call: ``K + premium``. Put: ``K - premium``.
    """
    if premium < 0.0:
        raise ValueError("premium must be non-negative")
    return strike + premium if option_type == "call" else strike - premium
