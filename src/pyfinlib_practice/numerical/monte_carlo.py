"""Monte Carlo pricing of European options.

Under the risk-neutral measure Q the discounted asset price is a martingale and
the terminal price is log-normal (RESULT):
    ``S(T) = S(0) * exp((r - q - sigma**2 / 2) * T + sigma * sqrt(T) * Z)``,
    ``Z ~ Normal(0, 1)``.

The price of a European claim with payoff ``g`` is the discounted expectation
    ``V = exp(-r * T) * E_Q[g(S(T))]``,
estimated here by the sample mean over ``n_paths`` independent draws of
``S(T)``. Because only the terminal value matters for a European payoff, paths
are sampled in a single step (exact, no time discretisation).
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np

from pyfinlib_practice.pricing.payoffs import call_payoff, put_payoff

OptionType = Literal["call", "put"]


class MonteCarloResult(NamedTuple):
    """Result of a Monte Carlo price estimate.

    Attributes
    ----------
    price:
        Discounted sample-mean estimator of the option value.
    std_error:
        Standard error of ``price`` (discounted sample std divided by
        ``sqrt(n_paths)``). A 95% confidence interval is roughly
        ``price +/- 1.96 * std_error``.
    n_paths:
        Number of simulated paths used.
    """

    price: float
    std_error: float
    n_paths: int


def monte_carlo_european_price(
    s0: float,
    strike: float,
    r: float,
    sigma: float,
    t: float,
    n_paths: int,
    *,
    option_type: OptionType = "call",
    q: float = 0.0,
    rng: np.random.Generator | None = None,
) -> MonteCarloResult:
    """Estimate the price of a European call or put by Monte Carlo.

    Parameters
    ----------
    s0, strike:
        Spot and strike, both positive.
    r, sigma, t:
        Risk-free rate, volatility (>= 0), maturity (> 0).
    n_paths:
        Number of simulated terminal prices (> 0).
    option_type:
        ``"call"`` or ``"put"``.
    q:
        Continuous dividend yield.
    rng:
        Random generator; ``None`` creates a fresh non-deterministic one.

    Returns
    -------
    MonteCarloResult
        Price, its standard error, and the path count.
    """
    if s0 <= 0.0 or strike <= 0.0:
        raise ValueError("s0 and strike must be positive")
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")
    if t <= 0.0:
        raise ValueError("t must be positive")
    if n_paths <= 0:
        raise ValueError("n_paths must be a positive integer")
    if rng is None:
        rng = np.random.default_rng()

    z = rng.standard_normal(size=n_paths)
    terminal = s0 * np.exp((r - q - 0.5 * sigma**2) * t + sigma * np.sqrt(t) * z)

    payoff = call_payoff(terminal, strike) if option_type == "call" else put_payoff(
        terminal, strike
    )
    discount = float(np.exp(-r * t))
    discounted = discount * payoff

    price = float(np.mean(discounted))
    # ddof=1 -> unbiased sample variance; SE of the mean = std / sqrt(n).
    std_error = float(np.std(discounted, ddof=1) / np.sqrt(n_paths))
    return MonteCarloResult(price=price, std_error=std_error, n_paths=n_paths)
