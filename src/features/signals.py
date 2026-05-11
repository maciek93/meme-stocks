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
            pl.col("sentiment").mean().alias("avg_sentiment"),
        ])
        .sort(["ticker", "date"])
    )
    # Derived features
    daily = daily.with_columns([
        (pl.col("mentions").cast(pl.Float64) / (pl.col("unique_authors").cast(pl.Float64) + 1e-6))
            .alias("mentions_per_author"),
        (pl.col("upvote_sum").cast(pl.Float64) / (pl.col("mentions").cast(pl.Float64) + 1e-6))
            .alias("score_per_mention"),
    ])
    return daily


def compute_velocity(daily: pl.DataFrame, window: int = 7) -> pl.DataFrame:
    """Add rolling mean and velocity (ratio to rolling mean)."""
    return (
        daily.with_columns(
            pl.col("mentions")
              .rolling_mean(window_size=window, min_periods=1)
              .over("ticker")
              .alias("mentions_rolling_mean")
        )
        .with_columns(
            (pl.col("mentions") / (pl.col("mentions_rolling_mean") + 1e-6))
            .alias("velocity")
        )
        .with_columns(
            (pl.col("avg_sentiment") * pl.col("velocity"))
            .alias("sentiment_x_velocity")
        )
    )


def apply_signal(daily: pl.DataFrame, velocity_threshold: float = 3.0, min_mentions: int = 50) -> pl.DataFrame:
    """Flag rows that meet the baseline signal criteria."""
    return daily.with_columns(
        ((pl.col("velocity") >= velocity_threshold) & (pl.col("mentions") >= min_mentions))
        .alias("signal")
    )
