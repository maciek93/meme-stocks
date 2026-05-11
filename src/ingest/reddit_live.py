"""
Ongoing Reddit collection using pullpush.io (no API credentials required).

Fetches the last N days of submissions across target subreddits and appends
to Parquet files in data/raw/.

Run:
    python -m src.ingest.reddit_live
"""

from datetime import date, timedelta
from pathlib import Path
from src.ingest.pullpush import fetch_and_save, SUBS_OF_INTEREST

RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw"


def collect(days_back: int = 3, subs: list[str] | None = None):
    end = date.today()
    start = end - timedelta(days=days_back)
    fetch_and_save(RAW_DIR, start, end, subs=subs, record_type="submissions")
    fetch_and_save(RAW_DIR, start, end, subs=subs, record_type="comments")


if __name__ == "__main__":
    collect()
