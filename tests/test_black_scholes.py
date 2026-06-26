"""Tests for Black-Scholes closed-form prices.

Canonical reference set: S = K = 100, r = 0.05, sigma = 0.20, T = 1, q = 0.
    call = 10.450584, put = 5.573526, digital call = 0.532325.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyfinlib_practice.models.black_scholes import (
    black_scholes_price,
    butterfly_price,
    digital_price,
    put_call_parity_residual,
)

S, K, R, SIG, T = 100.0, 100.0, 0.05, 0.20, 1.0


def test_call_anchor() -> None:
    price = black_scholes_price(S, K, R, SIG, T, option_type="call")
    assert float(price) == pytest.approx(10.450584, abs=1e-5)


def test_put_anchor() -> None:
    price = black_scholes_price(S, K, R, SIG, T, option_type="put")
    assert float(price) == pytest.approx(5.573526, abs=1e-5)


def test_put_call_parity_holds() -> None:
    call = black_scholes_price(S, K, R, SIG, T, option_type="call")
    put = black_scholes_price(S, K, R, SIG, T, option_type="put")
    residual = put_call_parity_residual(call, put, S, K, R, T)
    assert float(residual) == pytest.approx(0.0, abs=1e-12)


def test_parity_holds_with_dividends() -> None:
    q = 0.03
    call = black_scholes_price(S, K, R, SIG, T, option_type="call", q=q)
    put = black_scholes_price(S, K, R, SIG, T, option_type="put", q=q)
    residual = put_call_parity_residual(call, put, S, K, R, T, q=q)
    assert float(residual) == pytest.approx(0.0, abs=1e-12)


def test_digital_call_anchor() -> None:
    price = digital_price(S, K, R, SIG, T, option_type="call")
    assert float(price) == pytest.approx(0.532325, abs=1e-5)


def test_digital_call_put_sum_is_discount_factor() -> None:
    # A digital call + digital put = e^{-rT} (one of them must pay).
    dc = digital_price(S, K, R, SIG, T, option_type="call")
    dp = digital_price(S, K, R, SIG, T, option_type="put")
    assert float(dc + dp) == pytest.approx(np.exp(-R * T), abs=1e-12)


def test_butterfly_is_positive() -> None:
    fly = butterfly_price(S, K, R, SIG, T, spread=1.0)
    assert float(fly) > 0.0


def test_butterfly_rejects_bad_spread() -> None:
    with pytest.raises(ValueError):
        butterfly_price(S, K, R, SIG, T, spread=0.0)


def test_zero_vol_returns_discounted_intrinsic() -> None:
    # sigma -> 0: call = max(S e^{-qT} - K e^{-rT}, 0).
    call = black_scholes_price(120.0, 100.0, R, 0.0, T, option_type="call")
    expected = max(120.0 - 100.0 * np.exp(-R * T), 0.0)
    assert float(call) == pytest.approx(expected, abs=1e-12)


def test_zero_maturity_returns_intrinsic() -> None:
    call = black_scholes_price(120.0, 100.0, R, SIG, 0.0, option_type="call")
    assert float(call) == pytest.approx(20.0, abs=1e-12)


def test_price_is_vectorised_over_strikes() -> None:
    strikes = np.array([80.0, 100.0, 120.0])
    prices = black_scholes_price(S, strikes, R, SIG, T, option_type="call")
    assert prices.shape == (3,)
    # Call price is decreasing in strike.
    assert np.all(np.diff(prices) < 0.0)
