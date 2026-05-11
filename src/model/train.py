"""
Train logistic regression on backtest-labelled signals.

Usage:
    python -m src.model.train --results-path data/backtest_2025.parquet
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score

from rich.console import Console
from rich.table import Table

from src.model.dataset import build_dataset, walk_forward_split, FEATURE_COLS

console = Console()
MODELS_DIR = Path(__file__).parent.parent.parent / "models"


def train(results_path: Path, train_months: int = 9) -> None:
    console.print(f"Loading results from [cyan]{results_path}[/cyan]...")
    results = pl.read_parquet(results_path)
    console.print(f"  {len(results)} labelled signals")

    pos_rate = results.filter(pl.col("forward_return").is_not_null())["hit"].mean()
    console.print(f"  Positive rate (hit): [yellow]{pos_rate:.1%}[/yellow]")

    X_train, X_test, y_train, y_test = walk_forward_split(results, train_months=train_months)
    console.print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}\n")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)

    y_prob  = model.predict_proba(X_test_s)[:, 1]
    y_pred  = model.predict(X_test_s)
    auc     = roc_auc_score(y_test, y_prob)
    hit_rate = y_test.mean()

    # precision@K: top-K by predicted probability
    available = [c for c in FEATURE_COLS if c in results.columns]
    def p_at_k(k: int) -> float:
        top_k_idx = np.argsort(y_prob)[-k:]
        return y_test[top_k_idx].mean()

    console.print("[bold]Test set performance[/bold]")
    console.print(f"  AUC-ROC:       {auc:.3f}  (0.5 = random, 1.0 = perfect)")
    console.print(f"  Baseline rate: {hit_rate:.1%}  (predict-all-positive)")
    console.print(f"  Precision@10:  {p_at_k(min(10, len(y_test))):.1%}")
    console.print(f"  Precision@20:  {p_at_k(min(20, len(y_test))):.1%}")
    console.print(f"  Precision@50:  {p_at_k(min(50, len(y_test))):.1%}\n")

    # Coefficient table
    coef_table = Table(title="Feature Coefficients (positive = more likely to moon)")
    coef_table.add_column("Feature")
    coef_table.add_column("Coefficient", justify="right")
    coef_table.add_column("Direction")

    coefs = list(zip(available, model.coef_[0]))
    coefs.sort(key=lambda x: abs(x[1]), reverse=True)
    for feat, coef in coefs:
        direction = "[green]↑ bullish[/green]" if coef > 0 else "[red]↓ bearish[/red]"
        coef_table.add_row(feat, f"{coef:+.3f}", direction)
    console.print(coef_table)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model,  MODELS_DIR / "lr_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(available, MODELS_DIR / "feature_cols.pkl")
    console.print(f"\nModel saved → {MODELS_DIR}/lr_model.pkl")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-path", required=True, help="Path to backtest results Parquet")
    p.add_argument("--train-months", type=int, default=9)
    args = p.parse_args()
    train(Path(args.results_path), train_months=args.train_months)


if __name__ == "__main__":
    main()
