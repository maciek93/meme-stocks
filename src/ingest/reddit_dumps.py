"""Parse zst-compressed Reddit dump files (Academic Torrents format) into Parquet."""

import zstandard as zstd
import json
import polars as pl
from pathlib import Path
from tqdm import tqdm


SUBS_OF_INTEREST = {
    "wallstreetbets", "stocks", "pennystocks",
    "smallstreetbets", "Superstonk", "options", "investing",
}


def iter_records(zst_path: Path):
    dctx = zstd.ZstdDecompressor()
    with open(zst_path, "rb") as f:
        with dctx.stream_reader(f) as reader:
            buffer = b""
            while chunk := reader.read(2**20):
                buffer += chunk
                lines = buffer.split(b"\n")
                buffer = lines[-1]
                for line in lines[:-1]:
                    if line:
                        yield json.loads(line)
            if buffer:
                yield json.loads(buffer)


def dump_to_parquet(zst_path: Path, out_dir: Path, record_type: str = "submissions"):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rec in tqdm(iter_records(zst_path), desc=str(zst_path.name)):
        sub = rec.get("subreddit", "")
        if sub not in SUBS_OF_INTEREST:
            continue
        if record_type == "submissions":
            rows.append({
                "id": rec.get("id"),
                "sub": sub,
                "author": rec.get("author"),
                "created_utc": rec.get("created_utc"),
                "title": rec.get("title", ""),
                "body": rec.get("selftext", ""),
                "score": rec.get("score", 0),
                "num_comments": rec.get("num_comments", 0),
            })
        else:
            rows.append({
                "id": rec.get("id"),
                "sub": sub,
                "author": rec.get("author"),
                "created_utc": rec.get("created_utc"),
                "body": rec.get("body", ""),
                "score": rec.get("score", 0),
                "link_id": rec.get("link_id", ""),
            })

    if rows:
        df = pl.DataFrame(rows)
        stem = zst_path.stem.replace(".zst", "")
        df.write_parquet(out_dir / f"{stem}.parquet")
        print(f"Wrote {len(df)} rows → {out_dir / stem}.parquet")
