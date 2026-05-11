"""
Live signal CLI — no backtest. Shows high-velocity (or model-ranked) tickers right now.

Usage:
    python run_signals.py [--days-back 7] [--fetch] [--model]
"""

import argparse
from datetime import date, timedelta

from rich.console import Console
from rich.table import Table

from src.pipeline import run, RAW_DIR
from src.ingest.pullpush import fetch_and_save

console = Console()


def parse_args():
    p = argparse.ArgumentParser(description="Show current meme stock signals.")
    p.add_argument("--days-back", type=int, default=7)
    p.add_argument("--fetch", action="store_true", help="Fetch fresh data from pullpush.io first")
    p.add_argument("--velocity", type=float, default=3.0)
    p.add_argument("--min-mentions", type=int, default=20)
    p.add_argument("--model", action="store_true", help="Rank by model probability instead of velocity")
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
        use_model=args.model,
    )

    if signals.is_empty():
        console.print("[yellow]No signals found.[/yellow]")
        return

    sort_col = "signal_prob" if (args.model and "signal_prob" in signals.columns) else "velocity"
    # When using model, show all rows scored (not just signal=True), ranked by prob
    if args.model and "signal_prob" in signals.columns:
        fired = signals.sort(sort_col, descending=True).head(20)
    else:
        fired = signals.filter(signals["signal"]).sort(sort_col, descending=True)

    if fired.is_empty():
        console.print(f"[yellow]No signals above velocity={args.velocity} and mentions≥{args.min_mentions}.[/yellow]")
        return

    title = f"{'Model-Ranked' if args.model else 'Live'} Signals (last {args.days_back} days, as of {end})"
    table = Table(title=title)
    table.add_column("Ticker", style="bold yellow")
    table.add_column("Date")
    table.add_column("Mentions", justify="right")
    table.add_column("Velocity", justify="right")
    if args.model and "signal_prob" in fired.columns:
        table.add_column("Moon Prob", justify="right")
    table.add_column("Authors", justify="right")
    table.add_column("Sentiment", justify="right")

    for row in fired.iter_rows(named=True):
        r = [
            row["ticker"],
            str(row["date"]),
            str(row["mentions"]),
            f"{row['velocity']:.1f}x",
        ]
        if args.model and "signal_prob" in row:
            prob = row["signal_prob"]
            colour = "green" if prob >= 0.5 else "yellow" if prob >= 0.3 else "red"
            r.append(f"[{colour}]{prob:.2f}[/{colour}]")
        r += [str(row["unique_authors"]), f"{row['avg_sentiment']:.2f}"]
        table.add_row(*r)

    console.print(table)

    if args.model:
        console.print(f"\n[dim]Moon Prob = model's estimated P(+20% in 5 days). "
                      f"Retrain with: python -m src.model.train --results-path data/backtest_*.parquet[/dim]")
    else:
        console.print(f"\n[dim]No hit labels — re-run run_backtest.py in 5+ trading days to score these.[/dim]")


if __name__ == "__main__":
    main()
