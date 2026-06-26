"""Geometric Brownian motion (GBM) simulation.

The GBM stochastic differential equation
    ``dS(t) = mu * S(t) dt + sigma * S(t) dW(t)``
has the closed-form solution (RESULT, obtained by applying Ito's lemma to
``log S``):
    ``S(t) = S(0) * exp((mu - sigma**2 / 2) * t + sigma * W(t))``.

Two simulation schemes are provided:

* ``"exact"`` — samples the log-increment from its exact Gaussian law. There is
  **no time-discretisation error**: the marginal of ``S(t_k)`` is exactly
  log-normal regardless of step size.
* ``"euler"`` — the Euler-Maruyama discretisation
  ``S_{k+1} = S_k * (1 + mu * dt + sigma * sqrt(dt) * Z)``. Included for
  comparison; it carries ``O(dt)`` weak / ``O(sqrt(dt))`` strong error and can
  even go negative for large steps (MODELING CHOICE: kept deliberately simple).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

Scheme = Literal["exact", "euler"]


def simulate_gbm_paths(
    s0: float,
    mu: float,
    sigma: float,
    t: float,
    n_paths: int,
    n_steps: int,
    *,
    scheme: Scheme = "exact",
    rng: np.random.Generator | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Simulate GBM paths starting at ``s0``.

    Parameters
    ----------
    s0:
        Initial asset price, must be positive.
    mu, sigma:
        Drift and volatility. ``sigma`` must be non-negative.
    t:
        Horizon in years, must be positive.
    n_paths, n_steps:
        Number of paths and time steps. The grid has ``n_steps + 1`` points.
    scheme:
        ``"exact"`` (default) or ``"euler"``.
    rng:
        Random generator; ``None`` creates a fresh non-deterministic one.

    Returns
    -------
    times:
        Shape ``(n_steps + 1,)``.
    paths:
        Shape ``(n_paths, n_steps + 1)``; ``paths[:, 0] == s0``.
    """
    if s0 <= 0.0:
        raise ValueError("s0 must be positive")
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")
    if t <= 0.0:
        raise ValueError("t must be positive")
    if n_paths <= 0 or n_steps <= 0:
        raise ValueError("n_paths and n_steps must be positive integers")
    if rng is None:
        rng = np.random.default_rng()

    dt = t / n_steps
    times = np.linspace(0.0, t, n_steps + 1)
    z = rng.standard_normal(size=(n_paths, n_steps))
    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    paths[:, 0] = s0

    if scheme == "exact":
        log_increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        paths[:, 1:] = s0 * np.exp(np.cumsum(log_increments, axis=1))
    else:  # "euler"
        factors = 1.0 + mu * dt + sigma * np.sqrt(dt) * z
        paths[:, 1:] = s0 * np.cumprod(factors, axis=1)

    return times, paths
