"""
Fetch NASDAQ + NYSE ticker lists from NASDAQ Trader's public symbol directory.

Sources:
  nasdaqlisted.txt  — all NASDAQ-listed securities
  otherlisted.txt   — NYSE, ARCA, BATS, and other exchange securities

Run:
  python -m src.ingest.fetch_tickers
"""

import requests
from pathlib import Path

NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
OUT_PATH = Path(__file__).parent.parent.parent / "data" / "tickers" / "nasdaq_nyse_tickers.txt"


def _fetch(url: str) -> list[str]:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text.splitlines()


def _parse_nasdaq(lines: list[str]) -> set[str]:
    # Header: Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
    tickers = set()
    for line in lines[1:]:  # skip header
        if line.startswith("File Creation Time"):
            break
        parts = line.split("|")
        if len(parts) < 7:
            continue
        symbol, _, _, test_issue, fin_status, _, etf = parts[:7]
        if test_issue == "Y" or etf == "Y" or fin_status not in ("N", ""):
            continue
        if symbol and symbol.isalpha():
            tickers.add(symbol)
    return tickers


def _parse_other(lines: list[str]) -> set[str]:
    # Header: ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
    tickers = set()
    for line in lines[1:]:
        if line.startswith("File Creation Time"):
            break
        parts = line.split("|")
        if len(parts) < 7:
            continue
        act_symbol, _, exchange, _, etf, _, test_issue = parts[:7]
        if test_issue == "Y" or etf == "Y":
            continue
        # Keep NYSE (N), ARCA (A), BATS (Z) — skip OTC/pink sheets
        if exchange not in ("N", "A", "Z", "P"):
            continue
        symbol = act_symbol.strip()
        if symbol and symbol.isalpha():
            tickers.add(symbol)
    return tickers


def fetch_and_save():
    print("Fetching NASDAQ listed...")
    nasdaq_lines = _fetch(NASDAQ_URL)
    nasdaq_tickers = _parse_nasdaq(nasdaq_lines)
    print(f"  {len(nasdaq_tickers)} NASDAQ tickers")

    print("Fetching other (NYSE/ARCA/BATS) listed...")
    other_lines = _fetch(OTHER_URL)
    other_tickers = _parse_other(other_lines)
    print(f"  {len(other_tickers)} NYSE/other tickers")

    combined = nasdaq_tickers | other_tickers
    print(f"  {len(combined)} total unique tickers")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(sorted(combined)))
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    fetch_and_save()
