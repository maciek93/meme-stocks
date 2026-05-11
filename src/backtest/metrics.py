"""Backtest evaluation metrics."""

import polars as pl


def precision_at_k(results: pl.DataFrame, k: int | None = None) -> float:
    df = results.filter(pl.col("forward_return").is_not_null())
    if k:
        df = df.head(k)
    if df.is_empty():
        return 0.0
    return df["hit"].mean()


def hit_rate(results: pl.DataFrame) -> float:
    return precision_at_k(results)


def avg_forward_return(results: pl.DataFrame) -> float:
    df = results.filter(pl.col("forward_return").is_not_null())
    return df["forward_return"].mean() if not df.is_empty() else 0.0


def summary(results: pl.DataFrame) -> dict:
    return {
        "total_signals": len(results),
        "hit_rate": hit_rate(results),
        "avg_forward_return": avg_forward_return(results),
        "precision_at_10": precision_at_k(results, k=10),
    }
