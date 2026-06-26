"""Tests for payoff functions and P&L helpers."""

from __future__ import annotations

import numpy as np
import pytest

from pyfinlib_practice.pricing.payoffs import (
    breakeven,
    call_payoff,
    digital_call_payoff,
    digital_put_payoff,
    option_pnl,
    put_payoff,
)


def test_call_payoff_values() -> None:
    spots = np.array([80.0, 100.0, 120.0])
    assert np.array_equal(call_payoff(spots, 100.0), np.array([0.0, 0.0, 20.0]))


def test_put_payoff_values() -> None:
    spots = np.array([80.0, 100.0, 120.0])
    assert np.array_equal(put_payoff(spots, 100.0), np.array([20.0, 0.0, 0.0]))


def test_payoff_accepts_scalar() -> None:
    assert float(call_payoff(120.0, 100.0)) == 20.0
    assert float(put_payoff(80.0, 100.0)) == 20.0


def test_digital_payoffs() -> None:
    spots = np.array([99.0, 100.0, 101.0])
    assert np.array_equal(digital_call_payoff(spots, 100.0), np.array([0.0, 0.0, 1.0]))
    assert np.array_equal(digital_put_payoff(spots, 100.0), np.array([1.0, 0.0, 0.0]))


def test_long_call_pnl() -> None:
    spots = np.array([90.0, 100.0, 110.0, 120.0])
    pnl = option_pnl(spots, 100.0, 5.0, option_type="call", position="long")
    assert np.array_equal(pnl, np.array([-5.0, -5.0, 5.0, 15.0]))


def test_short_call_pnl_is_negated_long() -> None:
    spots = np.array([90.0, 110.0, 130.0])
    long_pnl = option_pnl(spots, 100.0, 5.0, option_type="call", position="long")
    short_pnl = option_pnl(spots, 100.0, 5.0, option_type="call", position="short")
    assert np.array_equal(short_pnl, -long_pnl)


def test_breakevens() -> None:
    assert breakeven(100.0, 5.0, option_type="call") == 105.0
    assert breakeven(100.0, 5.0, option_type="put") == 95.0


def test_pnl_rejects_negative_premium() -> None:
    with pytest.raises(ValueError):
        option_pnl(100.0, 100.0, -1.0, option_type="call", position="long")
