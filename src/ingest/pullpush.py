"""
Fetch Reddit submissions and comments via pullpush.io (no auth required).

Handles pagination, rate limiting, and saving to partitioned Parquet.

Usage:
    from src.ingest.pullpush import fetch_submissions
    from datetime import date

    df = fetch_submissions("wallstreetbets", date(2024, 1, 1), date(2024, 1, 31))
"""

import time
import requests
import polars as pl
from datetime import date, datetime, timezone
from pathlib import Path
from tqdm import tqdm

BASE_URL = "https://api.pullpush.io/reddit/search"
SUBS_OF_INTEREST = [
    "wallstreetbets", "stocks", "pennystocks",
    "smallstreetbets", "Superstonk", "options", "investing",
]
PAGE_SIZE = 100
# pullpush.io allows ~60 req/min; 1.1s between requests stays safely under
REQUEST_DELAY = 1.1


def _to_epoch(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())


def _get(endpoint: str, params: dict) -> list[dict]:
    url = f"{BASE_URL}/{endpoint}/"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("data", [])


def fetch_submissions(
    sub: str,
    start: date,
    end: date,
    min_score: int = 0,
) -> pl.DataFrame:
    """Fetch all submissions in [start, end] for a subreddit."""
    rows = []
    after = _to_epoch(start)
    before = _to_epoch(end)

    with tqdm(desc=f"submissions/{sub}", unit="posts") as pbar:
        while True:
            batch = _get("submission", {
                "subreddit": sub,
                "after": after,
                "before": before,
                "size": PAGE_SIZE,
                "sort": "asc",
                "sort_type": "created_utc",
            })
            if not batch:
                break
            for item in batch:
                if item.get("score", 0) >= min_score:
                    rows.append({
                        "id": item.get("id", ""),
                        "sub": sub,
                        "author": item.get("author", ""),
                        "created_utc": item.get("created_utc", 0),
                        "title": item.get("title", ""),
                        "body": item.get("selftext", ""),
                        "score": item.get("score", 0),
                        "num_comments": item.get("num_comments", 0),
                    })
            pbar.update(len(batch))
            after = batch[-1]["created_utc"]
            if len(batch) < PAGE_SIZE:
                break
            time.sleep(REQUEST_DELAY)

    return pl.DataFrame(rows) if rows else _empty_submissions()


def fetch_comments(
    sub: str,
    start: date,
    end: date,
) -> pl.DataFrame:
    """Fetch all comments in [start, end] for a subreddit."""
    rows = []
    after = _to_epoch(start)
    before = _to_epoch(end)

    with tqdm(desc=f"comments/{sub}", unit="comments") as pbar:
        while True:
            batch = _get("comment", {
                "subreddit": sub,
                "after": after,
                "before": before,
                "size": PAGE_SIZE,
                "sort": "asc",
                "sort_type": "created_utc",
            })
            if not batch:
                break
            for item in batch:
                rows.append({
                    "id": item.get("id", ""),
                    "sub": sub,
                    "author": item.get("author", ""),
                    "created_utc": item.get("created_utc", 0),
                    "body": item.get("body", ""),
                    "score": item.get("score", 0),
                    "link_id": item.get("link_id", ""),
                })
            pbar.update(len(batch))
            after = batch[-1]["created_utc"]
            if len(batch) < PAGE_SIZE:
                break
            time.sleep(REQUEST_DELAY)

    return pl.DataFrame(rows) if rows else _empty_comments()


def fetch_and_save(
    out_dir: Path,
    start: date,
    end: date,
    subs: list[str] | None = None,
    record_type: str = "submissions",
):
    """Fetch and save to Parquet, partitioned by sub and year-month."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target_subs = subs or SUBS_OF_INTEREST
    fetch_fn = fetch_submissions if record_type == "submissions" else fetch_comments

    for sub in target_subs:
        print(f"\n--- {sub} ({record_type}) ---")
        df = fetch_fn(sub, start, end)
        if df.is_empty():
            print(f"  no data")
            continue
        tag = f"{start.strftime('%Y%m')}_{end.strftime('%Y%m')}"
        path = out_dir / f"{record_type}_{sub}_{tag}.parquet"
        if path.exists():
            existing = pl.read_parquet(path)
            df = pl.concat([existing, df]).unique(subset=["id"])
        df.write_parquet(path)
        print(f"  saved {len(df)} rows → {path.name}")


def _empty_submissions() -> pl.DataFrame:
    return pl.DataFrame({
        "id": pl.Series([], dtype=pl.Utf8),
        "sub": pl.Series([], dtype=pl.Utf8),
        "author": pl.Series([], dtype=pl.Utf8),
        "created_utc": pl.Series([], dtype=pl.Int64),
        "title": pl.Series([], dtype=pl.Utf8),
        "body": pl.Series([], dtype=pl.Utf8),
        "score": pl.Series([], dtype=pl.Int64),
        "num_comments": pl.Series([], dtype=pl.Int64),
    })


def _empty_comments() -> pl.DataFrame:
    return pl.DataFrame({
        "id": pl.Series([], dtype=pl.Utf8),
        "sub": pl.Series([], dtype=pl.Utf8),
        "author": pl.Series([], dtype=pl.Utf8),
        "created_utc": pl.Series([], dtype=pl.Int64),
        "body": pl.Series([], dtype=pl.Utf8),
        "score": pl.Series([], dtype=pl.Int64),
        "link_id": pl.Series([], dtype=pl.Utf8),
    })
