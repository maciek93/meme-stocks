"""Load saved model and score signals with a moon probability."""

import warnings
from pathlib import Path

import numpy as np
import polars as pl

MODELS_DIR = Path(__file__).parent.parent.parent / "models"


def score_signals(signals: pl.DataFrame) -> pl.DataFrame:
    """Add `signal_prob` column (0–1) to a signals DataFrame.

    Falls back to velocity-based rank if no trained model is found.
    """
    model_path   = MODELS_DIR / "lr_model.pkl"
    scaler_path  = MODELS_DIR / "scaler.pkl"
    feature_path = MODELS_DIR / "feature_cols.pkl"

    if not all(p.exists() for p in [model_path, scaler_path, feature_path]):
        warnings.warn(
            "No trained model found. Run `python -m src.model.train` first. "
            "Falling back to velocity rank.",
            stacklevel=2,
        )
        max_v = signals["velocity"].max() or 1.0
        return signals.with_columns(
            (pl.col("velocity") / max_v).alias("signal_prob")
        )

    import joblib
    model        = joblib.load(model_path)
    scaler       = joblib.load(scaler_path)
    feature_cols = joblib.load(feature_path)

    missing = [c for c in feature_cols if c not in signals.columns]
    if missing:
        warnings.warn(f"Missing feature columns for model: {missing}. Falling back to velocity rank.")
        max_v = signals["velocity"].max() or 1.0
        return signals.with_columns(
            (pl.col("velocity") / max_v).alias("signal_prob")
        )

    X = signals.select(feature_cols).to_numpy().astype(np.float32)
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]

    return signals.with_columns(
        pl.Series("signal_prob", probs.astype(float))
    )
