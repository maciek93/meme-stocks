"""Walk-forward backtester. Point-in-time correct — no lookahead."""

import polars as pl
from datetime import date, timedelta
from src.prices import fetch_prices


def compute_forward_return(ticker: str, signal_date: date, hold_days: int = 5) -> float | None:
    """Max forward return over hold_days trading days from the day after signal_date.

    Entry = close on the first trading day after signal_date (realistic fill).
    Return = (max_close_in_window - entry) / entry.
    Returns None if fewer than 2 trading days exist in the window.
    """
    start = signal_date + timedelta(days=1)
    end = signal_date + timedelta(days=hold_days + 10)
    prices = fetch_prices(ticker, start, end)
    if prices.is_empty():
        return None

    prices = prices.sort("date")
    if len(prices) < 2:
        return None

    entry = prices["close"][0]
    hold_window = prices.head(hold_days)
    max_close = hold_window["close"].max()
    return (max_close - entry) / entry


def run_backtest(signals: pl.DataFrame, hold_days: int = 5, moon_threshold: float = 0.20) -> pl.DataFrame:
    """
    signals: DataFrame with columns [ticker, date, ...signal features...]
    Returns signals rows (where signal=True) with forward_return and hit columns added.
    """
    results = []
    for row in signals.filter(pl.col("signal")).iter_rows(named=True):
        fwd = compute_forward_return(row["ticker"], date.fromisoformat(str(row["date"])), hold_days)
        results.append({**row, "forward_return": fwd, "hit": fwd is not None and fwd >= moon_threshold})

    return pl.DataFrame(results) if results else pl.DataFrame()
