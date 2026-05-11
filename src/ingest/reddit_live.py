"""Live Reddit collection via PRAW. Append to Parquet partitions."""

import os
import praw
import polars as pl
from datetime import datetime, timezone
from pathlib import Path


SUBS_OF_INTEREST = [
    "wallstreetbets", "stocks", "pennystocks",
    "smallstreetbets", "Superstonk", "options", "investing",
]


def get_client() -> praw.Reddit:
    return praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ["REDDIT_USER_AGENT"],
    )


def fetch_hot(reddit: praw.Reddit, sub: str, limit: int = 100) -> list[dict]:
    rows = []
    for post in reddit.subreddit(sub).hot(limit=limit):
        rows.append({
            "id": post.id,
            "sub": sub,
            "author": str(post.author),
            "created_utc": int(post.created_utc),
            "title": post.title,
            "body": post.selftext,
            "score": post.score,
            "num_comments": post.num_comments,
        })
    return rows


def collect_and_save(out_dir: Path):
    reddit = get_client()
    today = datetime.now(timezone.utc).strftime("%Y-%m")
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in SUBS_OF_INTEREST:
        rows = fetch_hot(reddit, sub)
        df = pl.DataFrame(rows)
        path = out_dir / f"live_{sub}_{today}.parquet"
        if path.exists():
            existing = pl.read_parquet(path)
            df = pl.concat([existing, df]).unique(subset=["id"])
        df.write_parquet(path)
        print(f"{sub}: {len(df)} posts saved")
