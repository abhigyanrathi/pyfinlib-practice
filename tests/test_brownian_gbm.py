"""Tests for Brownian motion and GBM simulators."""

from __future__ import annotations

import numpy as np
import pytest

from pyfinlib_practice.core.brownian import simulate_brownian_paths
from pyfinlib_practice.core.gbm import simulate_gbm_paths


def test_brownian_shapes_and_initial_value() -> None:
    times, paths = simulate_brownian_paths(50, 100, 1.0, rng=np.random.default_rng(0))
    assert times.shape == (101,)
    assert paths.shape == (50, 101)
    assert np.allclose(paths[:, 0], 0.0)


def test_brownian_terminal_moments() -> None:
    # Standard BM at t=T has mean 0 and variance T.
    rng = np.random.default_rng(1)
    _, paths = simulate_brownian_paths(200_000, 1, 2.0, rng=rng)
    terminal = paths[:, -1]
    assert terminal.mean() == pytest.approx(0.0, abs=0.02)
    assert terminal.var() == pytest.approx(2.0, rel=0.02)


def test_brownian_drift_and_volatility() -> None:
    rng = np.random.default_rng(2)
    _, paths = simulate_brownian_paths(
        200_000, 1, 3.0, drift=0.5, volatility=2.0, rng=rng
    )
    terminal = paths[:, -1]
    # X(T) = drift*T + vol*W(T) ~ Normal(drift*T, vol^2 * T)
    assert terminal.mean() == pytest.approx(1.5, abs=0.03)
    assert terminal.var() == pytest.approx(4.0 * 3.0, rel=0.02)


def test_brownian_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        simulate_brownian_paths(0, 10, 1.0)
    with pytest.raises(ValueError):
        simulate_brownian_paths(10, 10, -1.0)
    with pytest.raises(ValueError):
        simulate_brownian_paths(10, 10, 1.0, volatility=-0.5)


def test_gbm_shapes_and_initial_value() -> None:
    times, paths = simulate_gbm_paths(
        100.0, 0.05, 0.2, 1.0, 50, 100, rng=np.random.default_rng(0)
    )
    assert times.shape == (101,)
    assert paths.shape == (50, 101)
    assert np.allclose(paths[:, 0], 100.0)


def test_gbm_terminal_mean_matches_theory() -> None:
    # E[S(T)] = S0 * exp(mu * T) for GBM (RESULT).
    s0, mu, sigma, t = 100.0, 0.07, 0.25, 1.0
    rng = np.random.default_rng(3)
    _, paths = simulate_gbm_paths(s0, mu, sigma, t, 400_000, 1, rng=rng)
    expected = s0 * np.exp(mu * t)
    assert paths[:, -1].mean() == pytest.approx(expected, rel=0.01)


def test_gbm_paths_stay_positive_under_exact_scheme() -> None:
    _, paths = simulate_gbm_paths(
        100.0, 0.0, 0.8, 1.0, 1000, 250, scheme="exact", rng=np.random.default_rng(4)
    )
    assert np.all(paths > 0.0)


def test_gbm_exact_is_reproducible_with_seed() -> None:
    a = simulate_gbm_paths(100.0, 0.05, 0.2, 1.0, 10, 50, rng=np.random.default_rng(7))[1]
    b = simulate_gbm_paths(100.0, 0.05, 0.2, 1.0, 10, 50, rng=np.random.default_rng(7))[1]
    assert np.array_equal(a, b)


def test_gbm_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        simulate_gbm_paths(-1.0, 0.05, 0.2, 1.0, 10, 10)
    with pytest.raises(ValueError):
        simulate_gbm_paths(100.0, 0.05, -0.2, 1.0, 10, 10)
