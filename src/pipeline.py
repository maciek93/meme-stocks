"""End-to-end pipeline: Reddit data → signals → backtest results."""

import polars as pl
from datetime import date, datetime, timezone
from pathlib import Path

from src.extract.tickers import build_mention_table, load_ticker_whitelist
from src.features.signals import compute_daily_features, compute_velocity, apply_signal
from src.backtest.engine import run_backtest
from src.backtest.metrics import summary

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
TICKERS_DIR = Path(__file__).parent.parent / "data" / "tickers"


def load_raw(raw_dir: Path, start: date, end: date, record_types: list[str] | None = None) -> pl.DataFrame:
    """Load all Parquet files from raw_dir whose date tags overlap [start, end]."""
    record_types = record_types or ["submissions", "comments"]
    frames = []
    for path in sorted(raw_dir.glob("*.parquet")):
        if not any(path.name.startswith(rt) for rt in record_types):
            continue
        df = pl.read_parquet(path)
        if "created_utc" not in df.columns:
            continue
        start_epoch = datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp()
        end_epoch = datetime(end.year, end.month, end.day, 23, 59, 59, tzinfo=timezone.utc).timestamp()
        df = df.filter((pl.col("created_utc") >= start_epoch) & (pl.col("created_utc") <= end_epoch))
        if not df.is_empty():
            frames.append(df)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed").unique(subset=["id"])


def run(
    start: date,
    end: date,
    raw_dir: Path = RAW_DIR,
    tickers_dir: Path = TICKERS_DIR,
    hold_days: int = 5,
    moon_pct: float = 0.20,
    velocity_threshold: float = 3.0,
    min_mentions: int = 50,
    backtest: bool = True,
) -> tuple[pl.DataFrame, dict]:
    """
    Full pipeline from raw Parquet files to backtest results.

    Returns (results_df, metrics_dict). If backtest=False, returns (signals_df, {}).
    """
    print(f"Loading raw data {start} → {end}...")
    df = load_raw(raw_dir, start, end)
    if df.is_empty():
        print("No data found. Run with --fetch first.")
        return pl.DataFrame(), {}

    print(f"Loaded {len(df):,} posts/comments. Extracting tickers...")
    whitelist = load_ticker_whitelist(tickers_dir)
    mentions = build_mention_table(df, whitelist)
    if mentions.is_empty():
        print("No ticker mentions found.")
        return pl.DataFrame(), {}

    print(f"Found {len(mentions):,} mentions across {mentions['ticker'].n_unique()} tickers. Computing signals...")
    daily = compute_daily_features(mentions)
    daily = compute_velocity(daily)
    signals = apply_signal(daily, velocity_threshold=velocity_threshold, min_mentions=min_mentions)

    n_signals = signals.filter(pl.col("signal")).height
    print(f"{n_signals} signals fired.")

    if not backtest:
        return signals, {}

    print("Running backtest...")
    results = run_backtest(signals, hold_days=hold_days, moon_threshold=moon_pct)
    if results.is_empty():
        print("No backtest results (no signals or no price data).")
        return results, {}

    metrics = summary(results)
    return results, metrics
