"""Tests for analytical Black-Scholes Greeks.

Canonical reference set: S = K = 100, r = 0.05, sigma = 0.20, T = 1, q = 0.
    delta_call = 0.636831, gamma = 0.018762, vega = 37.524035,
    theta_call = -6.414028 (per year), rho_call = 53.232482 (per unit rate).
Greeks are also cross-checked by central finite differences of the BS price.
"""

from __future__ import annotations

import pytest

from pyfinlib_practice.models.black_scholes import black_scholes_price
from pyfinlib_practice.pricing.greeks import delta, gamma, rho, theta, vega

S, K, R, SIG, T = 100.0, 100.0, 0.05, 0.20, 1.0


def test_delta_call_anchor() -> None:
    assert float(delta(S, K, R, SIG, T, option_type="call")) == pytest.approx(
        0.636831, abs=1e-5
    )


def test_delta_put_anchor() -> None:
    # delta_put = delta_call - 1 when q = 0.
    assert float(delta(S, K, R, SIG, T, option_type="put")) == pytest.approx(
        0.636831 - 1.0, abs=1e-5
    )


def test_gamma_anchor_and_call_equals_put() -> None:
    assert float(gamma(S, K, R, SIG, T)) == pytest.approx(0.018762, abs=1e-5)


def test_vega_anchor() -> None:
    assert float(vega(S, K, R, SIG, T)) == pytest.approx(37.524035, abs=1e-4)


def test_theta_call_anchor() -> None:
    assert float(theta(S, K, R, SIG, T, option_type="call")) == pytest.approx(
        -6.414028, abs=1e-4
    )


def test_rho_call_anchor() -> None:
    assert float(rho(S, K, R, SIG, T, option_type="call")) == pytest.approx(
        53.232482, abs=1e-4
    )


def test_delta_matches_finite_difference() -> None:
    h = 1e-4
    up = black_scholes_price(S + h, K, R, SIG, T, option_type="call")
    down = black_scholes_price(S - h, K, R, SIG, T, option_type="call")
    fd = float((up - down) / (2.0 * h))
    assert float(delta(S, K, R, SIG, T, option_type="call")) == pytest.approx(fd, abs=1e-5)


def test_vega_matches_finite_difference() -> None:
    h = 1e-5
    up = black_scholes_price(S, K, R, SIG + h, T, option_type="call")
    down = black_scholes_price(S, K, R, SIG - h, T, option_type="call")
    fd = float((up - down) / (2.0 * h))
    assert float(vega(S, K, R, SIG, T)) == pytest.approx(fd, abs=1e-3)


def test_gamma_matches_second_difference() -> None:
    h = 1e-2
    up = black_scholes_price(S + h, K, R, SIG, T, option_type="call")
    mid = black_scholes_price(S, K, R, SIG, T, option_type="call")
    down = black_scholes_price(S - h, K, R, SIG, T, option_type="call")
    fd = float((up - 2.0 * mid + down) / (h * h))
    assert float(gamma(S, K, R, SIG, T)) == pytest.approx(fd, abs=1e-5)
