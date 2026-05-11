"""yfinance wrapper with local Parquet cache."""

import yfinance as yf
import polars as pl
from pathlib import Path
from datetime import date


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
        return existing or pl.DataFrame()

    new_df = pl.from_pandas(raw.reset_index()).rename({"Date": "date", "Close": "close", "Volume": "volume"})
    new_df = new_df.select(["date", "close", "volume"])

    combined = pl.concat([existing, new_df]) if existing is not None else new_df
    combined = combined.unique(subset=["date"]).sort("date")
    combined.write_parquet(cache_path)
    return _filter(combined, start, end)


def _filter(df: pl.DataFrame, start: date, end: date) -> pl.DataFrame:
    return df.filter((pl.col("date") >= start) & (pl.col("date") <= end))


def _date_range(start: date, end: date):
    from datetime import timedelta
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)
