"""
Historical backtest CLI.

Usage:
    python run_backtest.py --start 2025-01-01 --end 2025-12-31 [--fetch] [--save-results] [--model]
"""

import argparse
from datetime import date
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.pipeline import run, RAW_DIR, DATA_DIR
from src.ingest.pullpush import fetch_and_save

console = Console()


def parse_args():
    p = argparse.ArgumentParser(description="Backtest meme stock signals against historical Reddit data.")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    p.add_argument("--fetch", action="store_true", help="Fetch Reddit data from pullpush.io before running")
    p.add_argument("--hold-days", type=int, default=5)
    p.add_argument("--moon-pct", type=float, default=0.20)
    p.add_argument("--velocity", type=float, default=3.0)
    p.add_argument("--min-mentions", type=int, default=50)
    p.add_argument("--window", type=int, default=7, help="Rolling mean window for velocity (default: 7)")
    p.add_argument("--model", action="store_true", help="Score signals with trained model (adds signal_prob column)")
    p.add_argument("--save-results", action="store_true", help="Save results to data/backtest_START_END.parquet")
    return p.parse_args()


def main():
    args = parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    if args.fetch:
        console.print(f"[bold cyan]Fetching Reddit data {start} → {end}...[/bold cyan]")
        fetch_and_save(RAW_DIR, start, end, record_type="submissions")
        fetch_and_save(RAW_DIR, start, end, record_type="comments")

    save_path = DATA_DIR / f"backtest_{args.start}_{args.end}.parquet" if args.save_results else None

    results, metrics = run(
        start=start,
        end=end,
        hold_days=args.hold_days,
        moon_pct=args.moon_pct,
        velocity_threshold=args.velocity,
        min_mentions=args.min_mentions,
        window=args.window,
        use_model=args.model,
        save_results=save_path,
    )

    if results.is_empty():
        console.print("[yellow]No results.[/yellow]")
        return

    sort_col = "signal_prob" if (args.model and "signal_prob" in results.columns) else "velocity"
    display_cols = ["ticker", "date", "mentions", "velocity", "avg_sentiment", "forward_return", "hit"]
    if args.model and "signal_prob" in results.columns:
        display_cols.insert(4, "signal_prob")

    display = results.sort(sort_col, descending=True).head(20).select(display_cols)

    title = f"Top Signals: {start} → {end}  |  hold={args.hold_days}d  moon≥{args.moon_pct:.0%}"
    if args.model:
        title += "  [model ranked]"
    table = Table(title=title)
    table.add_column("Ticker", style="bold")
    table.add_column("Date")
    table.add_column("Mentions", justify="right")
    table.add_column("Velocity", justify="right")
    if args.model and "signal_prob" in results.columns:
        table.add_column("Prob", justify="right")
    table.add_column("Sentiment", justify="right")
    table.add_column("Fwd Return", justify="right")
    table.add_column("Hit", justify="center")

    for row in display.iter_rows(named=True):
        fwd = f"{row['forward_return']:.1%}" if row["forward_return"] is not None else "—"
        hit = "[green]✓[/green]" if row["hit"] else "[red]✗[/red]"
        r = [
            row["ticker"],
            str(row["date"]),
            str(row["mentions"]),
            f"{row['velocity']:.1f}x",
        ]
        if args.model and "signal_prob" in row:
            r.append(f"{row['signal_prob']:.2f}")
        r += [f"{row['avg_sentiment']:.2f}", fwd, hit]
        table.add_row(*r)

    console.print(table)
    console.print(f"\n[bold]Summary[/bold]")
    console.print(f"  Total signals:      {metrics['total_signals']}")
    console.print(f"  Hit rate:           {metrics['hit_rate']:.1%}")
    console.print(f"  Avg forward return: {metrics['avg_forward_return']:.1%}")
    console.print(f"  Precision@10:       {metrics['precision_at_10']:.1%}")
    if save_path:
        console.print(f"\n  Results saved → [cyan]{save_path}[/cyan]")
        console.print(f"  Train model:  python -m src.model.train --results-path {save_path}")


if __name__ == "__main__":
    main()
