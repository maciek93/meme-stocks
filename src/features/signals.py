"""Aggregate mention tables into daily signal features per ticker."""

import polars as pl


def compute_daily_features(mentions: pl.DataFrame) -> pl.DataFrame:
    """
    Input: mention rows (ticker, date, author, score).
    Output: one row per (ticker, date) with features.
    """
    daily = (
        mentions.group_by(["ticker", "date"])
        .agg([
            pl.len().alias("mentions"),
            pl.col("author").n_unique().alias("unique_authors"),
            pl.col("score").sum().alias("upvote_sum"),
        ])
        .sort(["ticker", "date"])
    )
    return daily


def compute_velocity(daily: pl.DataFrame, window: int = 7) -> pl.DataFrame:
    """Add rolling mean and velocity (ratio to rolling mean)."""
    return (
        daily.with_columns(
            pl.col("mentions")
              .rolling_mean(window_size=window)
              .over("ticker")
              .alias("mentions_rolling_mean")
        )
        .with_columns(
            (pl.col("mentions") / (pl.col("mentions_rolling_mean") + 1e-6))
            .alias("velocity")
        )
    )


def apply_signal(daily: pl.DataFrame, velocity_threshold: float = 3.0, min_mentions: int = 50) -> pl.DataFrame:
    """Flag rows that meet the baseline signal criteria."""
    return daily.with_columns(
        ((pl.col("velocity") >= velocity_threshold) & (pl.col("mentions") >= min_mentions))
        .alias("signal")
    )
