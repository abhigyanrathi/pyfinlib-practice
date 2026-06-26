"""Brownian motion (Wiener process) simulation.

A standard Wiener process ``W(t)`` satisfies (RESULT):
    * ``W(0) = 0``,
    * independent increments,
    * ``W(t) - W(s) ~ Normal(0, t - s)`` for ``0 <= s < t``,
    * continuous sample paths.

This module simulates the more general *arithmetic* Brownian motion
    ``X(t) = mu * t + sigma * W(t)``
on an equally spaced grid. Setting ``mu = 0`` and ``sigma = 1`` recovers the
standard Wiener process.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def simulate_brownian_paths(
    n_paths: int,
    n_steps: int,
    t: float,
    *,
    drift: float = 0.0,
    volatility: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Simulate paths of ``X(t) = drift * t + volatility * W(t)``.

    Parameters
    ----------
    n_paths:
        Number of independent paths to simulate.
    n_steps:
        Number of time steps; the grid has ``n_steps + 1`` points including 0.
    t:
        Time horizon (in years). Must be positive.
    drift, volatility:
        Coefficients ``mu`` and ``sigma``. ``volatility`` must be non-negative.
    rng:
        A ``numpy`` random generator. If ``None``, a fresh default generator is
        created (non-deterministic). Pass ``np.random.default_rng(seed)`` for
        reproducibility.

    Returns
    -------
    times:
        Shape ``(n_steps + 1,)`` grid of time points from 0 to ``t``.
    paths:
        Shape ``(n_paths, n_steps + 1)`` array; ``paths[:, 0]`` is identically 0.

    Notes
    -----
    The increment over a step of size ``dt = t / n_steps`` is exactly
    ``Normal(drift * dt, (volatility ** 2) * dt)`` (RESULT) — there is no
    discretisation error for arithmetic Brownian motion because the exact
    transition law is Gaussian and is sampled directly.
    """
    if n_paths <= 0 or n_steps <= 0:
        raise ValueError("n_paths and n_steps must be positive integers")
    if t <= 0.0:
        raise ValueError("t must be positive")
    if volatility < 0.0:
        raise ValueError("volatility must be non-negative")
    if rng is None:
        rng = np.random.default_rng()

    dt = t / n_steps
    times = np.linspace(0.0, t, n_steps + 1)

    increments = rng.normal(
        loc=drift * dt,
        scale=volatility * np.sqrt(dt),
        size=(n_paths, n_steps),
    )

    paths = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
    paths[:, 1:] = np.cumsum(increments, axis=1)
    return times, paths
