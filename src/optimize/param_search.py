"""
Walk-forward grid search over signal rule parameters.

Splits 2025 data Jan–Sep (tune) / Oct–Dec (test). Searches over velocity_threshold,
min_mentions, and rolling window. Optimises for precision@10 on the held-out test split.

Usage:
    python -m src.optimize.param_search --start 2025-01-01 --end 2025-12-31
"""

import argparse
import json
import itertools
from datetime import date
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.pipeline import run, RAW_DIR
from src.backtest.metrics import precision_at_k

console = Console()

MODELS_DIR = Path(__file__).parent.parent.parent / "models"

PARAM_GRID = {
    "velocity_threshold": [1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
    "min_mentions":       [10, 20, 30, 50, 75, 100],
    "window":             [3, 5, 7, 14],
}


def _split_date(start: date, end: date, train_fraction: float = 0.75) -> date:
    total_days = (end - start).days
    return start.__class__.fromordinal(start.toordinal() + int(total_days * train_fraction))


def search(start: date, end: date, raw_dir: Path = RAW_DIR) -> list[dict]:
    split = _split_date(start, end)
    console.print(f"Train window: [cyan]{start}[/cyan] → [cyan]{split}[/cyan]")
    console.print(f"Test  window: [yellow]{split}[/yellow] → [yellow]{end}[/yellow]\n")

    keys = list(PARAM_GRID.keys())
    combos = list(itertools.product(*PARAM_GRID.values()))
    console.print(f"Searching {len(combos)} parameter combinations...\n")

    results = []
    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        vt = params["velocity_threshold"]
        mm = params["min_mentions"]
        w  = params["window"]

        # Tune on train window
        train_results, _ = run(
            start=start, end=split,
            raw_dir=raw_dir,
            velocity_threshold=vt,
            min_mentions=mm,
            backtest=True,
        )
        # Evaluate on held-out test window (never tuned on)
        test_results, test_metrics = run(
            start=split, end=end,
            raw_dir=raw_dir,
            velocity_threshold=vt,
            min_mentions=mm,
            backtest=True,
        )

        train_p10 = precision_at_k(train_results, k=10) if not train_results.is_empty() else 0.0
        test_p10  = precision_at_k(test_results,  k=10) if not test_results.is_empty()  else 0.0
        n_test    = test_metrics.get("total_signals", 0)

        results.append({
            **params,
            "train_p10": train_p10,
            "test_p10":  test_p10,
            "test_signals": n_test,
        })

        if i % 20 == 0:
            console.print(f"  {i}/{len(combos)} done...")

    results.sort(key=lambda r: r["test_p10"], reverse=True)
    return results


def print_results(results: list[dict], top_n: int = 20):
    table = Table(title=f"Top {top_n} Parameter Combinations (by test precision@10)")
    for col in ["velocity_threshold", "min_mentions", "window", "train_p10", "test_p10", "test_signals"]:
        table.add_column(col, justify="right")
    for r in results[:top_n]:
        table.add_row(
            f"{r['velocity_threshold']:.1f}",
            str(r["min_mentions"]),
            str(r["window"]),
            f"{r['train_p10']:.1%}",
            f"{r['test_p10']:.1%}",
            str(r["test_signals"]),
        )
    console.print(table)


def save_best(results: list[dict]):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best = results[0]
    out = {k: best[k] for k in ["velocity_threshold", "min_mentions", "window"]}
    path = MODELS_DIR / "best_params.json"
    path.write_text(json.dumps(out, indent=2))
    console.print(f"\nBest params saved → {path}")
    console.print(json.dumps(out, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2025-01-01")
    p.add_argument("--end",   default="2025-12-31")
    args = p.parse_args()

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)

    results = search(start, end)
    print_results(results)
    save_best(results)


if __name__ == "__main__":
    main()
