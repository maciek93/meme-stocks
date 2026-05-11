"""Walk-forward backtester. Point-in-time correct — no lookahead."""

import polars as pl
from datetime import date, timedelta
from src.prices import fetch_prices


def compute_forward_return(ticker: str, signal_date: date, hold_days: int = 5) -> float | None:
    """Return max forward return over hold_days from signal_date close."""
    start = signal_date + timedelta(days=1)
    end = signal_date + timedelta(days=hold_days + 5)
    prices = fetch_prices(ticker, start, end)
    if prices.is_empty():
        return None
    trading_days = prices.sort("date").head(hold_days)
    if trading_days.is_empty():
        return None
    entry = prices.filter(pl.col("date") == prices["date"].min())["close"][0]
    max_close = trading_days["close"].max()
    return (max_close - entry) / entry


def run_backtest(signals: pl.DataFrame, hold_days: int = 5, moon_threshold: float = 0.20) -> pl.DataFrame:
    """
    signals: DataFrame with columns [ticker, date, ...signal features...]
    Returns signals with forward_return and hit (bool: return >= moon_threshold).
    """
    results = []
    for row in signals.filter(pl.col("signal")).iter_rows(named=True):
        fwd = compute_forward_return(row["ticker"], date.fromisoformat(row["date"]), hold_days)
        results.append({**row, "forward_return": fwd, "hit": fwd is not None and fwd >= moon_threshold})

    return pl.DataFrame(results) if results else pl.DataFrame()
