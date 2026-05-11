"""Extract stock ticker mentions from Reddit text."""

import re
import polars as pl
from pathlib import Path
from src.extract.sentiment import vader_score


# Common words that overlap with valid tickers — extend as needed
STOPWORDS = {
    "A", "I", "FOR", "ALL", "ARE", "BE", "BY", "DO", "GO", "HE", "IF",
    "IN", "IS", "IT", "ME", "MY", "NO", "OF", "ON", "OR", "SO", "TO",
    "UP", "US", "WE", "AT", "AN", "AS", "AM", "DD", "OC", "CEO", "CFO",
    "IPO", "ETF", "GDP", "IMO", "TBH", "LOL", "EPS", "SEC", "NYSE",
    "NOW", "NEW", "THE", "AND", "BUT", "YET", "NOR", "NOT", "TOO",
    "YOLO", "FOMO", "TLDR", "EOD", "EOW", "AH", "PM", "AI", "IT",
}

CASHTAG_RE = re.compile(r"\$([A-Z]{1,5})")
BARE_TICKER_RE = re.compile(r"\b([A-Z]{2,5})\b")


def load_ticker_whitelist(tickers_dir: Path) -> set[str]:
    path = tickers_dir / "nasdaq_nyse_tickers.txt"
    if not path.exists():
        raise FileNotFoundError(f"Ticker list not found: {path}")
    return set(path.read_text().splitlines())


def extract_tickers(text: str, whitelist: set[str]) -> list[str]:
    found = set()
    for m in CASHTAG_RE.finditer(text):
        t = m.group(1)
        if t in whitelist:
            found.add(t)
    for m in BARE_TICKER_RE.finditer(text):
        t = m.group(1)
        if t in whitelist and t not in STOPWORDS:
            found.add(t)
    return list(found)


def build_mention_table(df: pl.DataFrame, whitelist: set[str]) -> pl.DataFrame:
    """Input: posts/comments df with created_utc, author, body/title, score.
    Output: rows of (ticker, date, author, score, sentiment)."""
    rows = []
    for row in df.iter_rows(named=True):
        text = (row.get("title", "") or "") + " " + (row.get("body", "") or "")
        date = str(pl.from_epoch([row["created_utc"]], time_unit="s")[0].date())
        sentiment = vader_score(text)
        for ticker in extract_tickers(text, whitelist):
            rows.append({
                "ticker": ticker,
                "date": date,
                "author": row.get("author", ""),
                "score": row.get("score", 0),
                "sentiment": sentiment,
            })
    return pl.DataFrame(rows) if rows else pl.DataFrame({
        "ticker": pl.Series([], dtype=pl.Utf8),
        "date": pl.Series([], dtype=pl.Utf8),
        "author": pl.Series([], dtype=pl.Utf8),
        "score": pl.Series([], dtype=pl.Int64),
        "sentiment": pl.Series([], dtype=pl.Float64),
    })
