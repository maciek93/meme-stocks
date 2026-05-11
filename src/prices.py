"""yfinance wrapper with local Parquet cache."""

import yfinance as yf
import polars as pl
import pandas as pd
from pathlib import Path
from datetime import date, timedelta


CACHE_DIR = Path(__file__).parent.parent / "data" / "prices"


def fetch_prices(ticker: str, start: date, end: date) -> pl.DataFrame:
    cache_path = CACHE_DIR / f"{ticker}.parquet"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    existing = None
    if cache_path.exists():
        existing = pl.read_parquet(cache_path)
        cached_dates = set(existing["date"].cast(str).to_list())
        needed = [d for d in _date_range(start, end) if str(d) not in cached_dates]
        if not needed:
            return _filter(existing, start, end)
    else:
        needed = list(_date_range(start, end))

    raw = yf.download(ticker, start=str(min(needed)), end=str(max(needed)), auto_adjust=True, progress=False)
    if raw.empty:
        return existing if existing is not None else pl.DataFrame()

    new_df = _parse_yfinance(raw)
    if new_df.is_empty():
        return existing if existing is not None else pl.DataFrame()

    combined = pl.concat([existing, new_df]) if existing is not None else new_df
    combined = combined.unique(subset=["date"]).sort("date")
    combined.write_parquet(cache_path)
    return _filter(combined, start, end)


def _parse_yfinance(raw: pd.DataFrame) -> pl.DataFrame:
    """Handle yfinance v1.x MultiIndex columns and return [date, close, volume]."""
    df = raw.copy()
    # Flatten MultiIndex columns: ('Close', 'AAPL') → 'close'
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df = df.reset_index()
    # Index name may be 'Date' or 'date' or 'Datetime'
    date_col = next((c for c in df.columns if c.lower() in ("date", "datetime")), None)
    if date_col is None:
        return pl.DataFrame()
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return pl.from_pandas(df[["date", "close", "volume"]])


def _filter(df: pl.DataFrame, start: date, end: date) -> pl.DataFrame:
    return df.filter((pl.col("date") >= start) & (pl.col("date") <= end))


def _date_range(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)
