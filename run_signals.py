"""
Live signal CLI — no backtest. Shows high-velocity tickers right now.

Usage:
    python run_signals.py [--days-back 7] [--fetch]
"""

import argparse
from datetime import date, timedelta

from rich.console import Console
from rich.table import Table

from src.pipeline import run, RAW_DIR
from src.ingest.pullpush import fetch_and_save

console = Console()


def parse_args():
    p = argparse.ArgumentParser(description="Show current high-velocity meme stock signals.")
    p.add_argument("--days-back", type=int, default=7, help="How many days back to scan (default: 7)")
    p.add_argument("--fetch", action="store_true", help="Fetch fresh data from pullpush.io first")
    p.add_argument("--velocity", type=float, default=3.0, help="Velocity threshold (default: 3.0)")
    p.add_argument("--min-mentions", type=int, default=20, help="Minimum daily mentions (default: 20, lower for recency)")
    return p.parse_args()


def main():
    args = parse_args()
    end = date.today()
    start = end - timedelta(days=args.days_back)

    if args.fetch:
        console.print(f"[bold cyan]Fetching Reddit data (last {args.days_back} days)...[/bold cyan]")
        fetch_and_save(RAW_DIR, start, end, record_type="submissions")
        fetch_and_save(RAW_DIR, start, end, record_type="comments")

    signals, _ = run(
        start=start,
        end=end,
        velocity_threshold=args.velocity,
        min_mentions=args.min_mentions,
        backtest=False,
    )

    if signals.is_empty():
        console.print("[yellow]No signals found.[/yellow]")
        return

    fired = signals.filter(signals["signal"]).sort("velocity", descending=True)

    if fired.is_empty():
        console.print(f"[yellow]No signals above velocity={args.velocity} and mentions≥{args.min_mentions}.[/yellow]")
        return

    table = Table(title=f"Live Signals (last {args.days_back} days, as of {end})")
    table.add_column("Ticker", style="bold yellow")
    table.add_column("Date")
    table.add_column("Mentions", justify="right")
    table.add_column("Velocity", justify="right")
    table.add_column("Unique Authors", justify="right")
    table.add_column("Sentiment", justify="right")

    for row in fired.iter_rows(named=True):
        table.add_row(
            row["ticker"],
            str(row["date"]),
            str(row["mentions"]),
            f"{row['velocity']:.1f}x",
            str(row["unique_authors"]),
            f"{row['avg_sentiment']:.2f}",
        )

    console.print(table)
    console.print(f"\n[dim]No hit labels — future hasn't happened yet. "
                  f"Re-run run_backtest.py on this date range in 5+ trading days to score these.[/dim]")


if __name__ == "__main__":
    main()
