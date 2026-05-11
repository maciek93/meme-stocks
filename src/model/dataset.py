"""Build training dataset from backtest results for ML model training."""

import numpy as np
import polars as pl
from datetime import date

FEATURE_COLS = [
    "velocity",
    "avg_sentiment",
    "unique_authors",
    "upvote_sum",
    "mentions",
    "mentions_per_author",
    "score_per_mention",
    "sentiment_x_velocity",
]


def build_dataset(results: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract feature matrix X and label vector y from backtest results.

    Only rows with non-null forward_return are included (i.e. signals where
    price data was available to evaluate).
    """
    df = results.filter(pl.col("forward_return").is_not_null())

    available = [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(available)
    if missing:
        raise ValueError(f"Missing feature columns: {missing}. Re-run pipeline to regenerate.")

    X = df.select(available).to_numpy().astype(np.float32)
    y = df["hit"].cast(pl.Int8).to_numpy()

    return X, y, available


def walk_forward_split(
    results: pl.DataFrame,
    train_months: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split by date — train on first train_months, test on remainder.

    Critically: split is temporal, not random. Random splits leak future
    label information into training data.
    """
    df = results.filter(pl.col("forward_return").is_not_null()).sort("date")
    if df.is_empty():
        raise ValueError("No labelled results to split.")

    dates = df["date"].cast(pl.Utf8).to_list()
    min_date = date.fromisoformat(str(min(dates)))
    cutoff = date(
        min_date.year + (min_date.month + train_months - 1) // 12,
        (min_date.month + train_months - 1) % 12 + 1,
        1,
    )

    train_df = df.filter(pl.col("date").cast(pl.Utf8) < str(cutoff))
    test_df  = df.filter(pl.col("date").cast(pl.Utf8) >= str(cutoff))

    if train_df.is_empty() or test_df.is_empty():
        raise ValueError(
            f"Split produced empty train or test set (cutoff={cutoff}). "
            "Ensure data spans more than train_months."
        )

    X_train, y_train, _ = build_dataset(train_df)
    X_test,  y_test,  _ = build_dataset(test_df)
    return X_train, X_test, y_train, y_test
