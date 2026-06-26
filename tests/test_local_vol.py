"""Tests for implied volatility solvers and Breeden-Litzenberger.

Strategy: price an option with a known volatility using the Black-Scholes
formula, then confirm each solver recovers that volatility (a round trip). The
Breeden-Litzenberger density recovered from a constant-volatility call surface
must integrate to one, have mean equal to the forward, and reprice a call to
match the closed form (three-way agreement).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyfinlib_practice.models.black_scholes import black_scholes_price
from pyfinlib_practice.models.local_vol import (
    Method,
    implied_volatility,
    implied_volatility_bracketed,
    implied_volatility_newton,
    implied_volatility_secant,
    price_from_density,
    risk_neutral_density,
)
from pyfinlib_practice.pricing.payoffs import call_payoff

S, K, R, T = 100.0, 100.0, 0.05, 1.0


def test_newton_round_trip() -> None:
    sigma_true = 0.25
    price = float(black_scholes_price(S, K, R, sigma_true, T, option_type="call"))
    recovered = implied_volatility_newton(price, S, K, R, T, option_type="call")
    assert recovered == pytest.approx(sigma_true, abs=1e-7)


def test_secant_round_trip() -> None:
    sigma_true = 0.25
    price = float(black_scholes_price(S, K, R, sigma_true, T, option_type="call"))
    recovered = implied_volatility_secant(price, S, K, R, T, option_type="call")
    assert recovered == pytest.approx(sigma_true, abs=1e-7)


def test_bracketed_round_trip() -> None:
    sigma_true = 0.25
    price = float(black_scholes_price(S, K, R, sigma_true, T, option_type="call"))
    recovered = implied_volatility_bracketed(price, S, K, R, T, option_type="call")
    assert recovered == pytest.approx(sigma_true, abs=1e-7)


def test_all_methods_agree_via_dispatcher() -> None:
    sigma_true = 0.4
    price = float(black_scholes_price(S, K, R, sigma_true, T, option_type="put"))
    methods: tuple[Method, ...] = ("newton", "secant", "bracketed")
    vals = [
        implied_volatility(price, S, K, R, T, option_type="put", method=m)
        for m in methods
    ]
    for v in vals:
        assert v == pytest.approx(sigma_true, abs=1e-6)


def test_bracketed_handles_deep_otm() -> None:
    # Deep OTM: small price, small vega -- the bracketed solver must still work.
    sigma_true = 0.30
    price = float(black_scholes_price(100.0, 200.0, R, sigma_true, 0.5, option_type="call"))
    recovered = implied_volatility_bracketed(
        price, 100.0, 200.0, R, 0.5, option_type="call"
    )
    assert recovered == pytest.approx(sigma_true, abs=1e-6)


def test_bracketed_handles_deep_itm() -> None:
    sigma_true = 0.30
    price = float(black_scholes_price(200.0, 100.0, R, sigma_true, 0.5, option_type="call"))
    recovered = implied_volatility_bracketed(
        price, 200.0, 100.0, R, 0.5, option_type="call"
    )
    assert recovered == pytest.approx(sigma_true, abs=1e-6)


def test_price_below_intrinsic_raises() -> None:
    # A call cannot be worth less than its discounted intrinsic value.
    with pytest.raises(ValueError):
        implied_volatility(0.01, 150.0, 100.0, R, T, option_type="call")


def test_price_above_upper_bound_raises() -> None:
    # A call cannot be worth more than the (dividend-discounted) spot.
    with pytest.raises(ValueError):
        implied_volatility(150.0, 100.0, 100.0, R, T, option_type="call")


# --------------------------------------------------------------------------- #
# Breeden-Litzenberger
# --------------------------------------------------------------------------- #
def _call_surface() -> tuple[np.ndarray, np.ndarray]:
    strikes = np.linspace(1.0, 400.0, 4000)
    prices = black_scholes_price(S, strikes, R, 0.20, T, option_type="call")
    return strikes, prices


def test_density_integrates_to_one() -> None:
    strikes, prices = _call_surface()
    density = risk_neutral_density(strikes, prices, R, T)
    mass = float(np.trapezoid(density, strikes))
    assert mass == pytest.approx(1.0, abs=2e-3)


def test_density_mean_is_forward() -> None:
    strikes, prices = _call_surface()
    density = risk_neutral_density(strikes, prices, R, T)
    mean = float(np.trapezoid(strikes * density, strikes))
    forward = S * np.exp(R * T)  # q = 0
    assert mean == pytest.approx(forward, rel=2e-3)


def test_price_from_density_matches_black_scholes() -> None:
    strikes, prices = _call_surface()
    density = risk_neutral_density(strikes, prices, R, T)
    payoff = call_payoff(strikes, 110.0)
    mc_like = price_from_density(payoff, strikes, density, R, T)
    closed_form = float(black_scholes_price(S, 110.0, R, 0.20, T, option_type="call"))
    assert mc_like == pytest.approx(closed_form, abs=2e-2)


def test_density_rejects_short_grid() -> None:
    with pytest.raises(ValueError):
        risk_neutral_density([100.0, 101.0], [1.0, 0.5], R, T)
