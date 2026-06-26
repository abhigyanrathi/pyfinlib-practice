"""Tests for the Monte Carlo European pricer.

The convergence anchors are the standard textbook Black-Scholes values for
S0 = K = 100, r = 0.05, sigma = 0.20, T = 1, q = 0:
    call = 10.4506, put = 5.5735.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyfinlib_practice.numerical.monte_carlo import monte_carlo_european_price

CALL_ANCHOR = 10.4506
PUT_ANCHOR = 5.5735


def test_mc_call_converges_to_bs() -> None:
    res = monte_carlo_european_price(
        100.0, 100.0, 0.05, 0.20, 1.0, 400_000,
        option_type="call", rng=np.random.default_rng(0),
    )
    # Estimate must sit within four standard errors of the analytic price.
    assert abs(res.price - CALL_ANCHOR) < 4.0 * res.std_error


def test_mc_put_converges_to_bs() -> None:
    res = monte_carlo_european_price(
        100.0, 100.0, 0.05, 0.20, 1.0, 400_000,
        option_type="put", rng=np.random.default_rng(1),
    )
    assert abs(res.price - PUT_ANCHOR) < 4.0 * res.std_error


def test_mc_standard_error_shrinks_with_more_paths() -> None:
    small = monte_carlo_european_price(
        100.0, 100.0, 0.05, 0.20, 1.0, 10_000, rng=np.random.default_rng(2)
    )
    large = monte_carlo_european_price(
        100.0, 100.0, 0.05, 0.20, 1.0, 160_000, rng=np.random.default_rng(2)
    )
    # 16x paths -> ~4x smaller standard error.
    assert large.std_error < 0.5 * small.std_error


def test_mc_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        monte_carlo_european_price(-1.0, 100.0, 0.05, 0.2, 1.0, 1000)
    with pytest.raises(ValueError):
        monte_carlo_european_price(100.0, 100.0, 0.05, 0.2, 1.0, 0)
