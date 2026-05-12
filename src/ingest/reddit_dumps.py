"""Parse zst-compressed Reddit dump files into Parquet."""

import zstandard as zstd
import json
import polars as pl
from pathlib import Path
from tqdm import tqdm


SUBS_OF_INTEREST = {
    "wallstreetbets", "stocks", "pennystocks",
    "smallstreetbets", "Superstonk", "options", "investing",
    "ValueInvesting",
}


def detect_record_type(zst_path: Path) -> str:
    """Infer submissions vs comments from filename."""
    name = zst_path.stem.lower()
    if "comment" in name:
        return "comments"
    if "submission" in name:
        return "submissions"
    raise ValueError(f"Cannot detect record type from filename: {zst_path.name}. "
                     "Expected 'submissions' or 'comments' in the name.")


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
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue
            if buffer.strip():
                try:
                    yield json.loads(buffer)
                except json.JSONDecodeError:
                    pass


def dump_to_parquet(zst_path: Path, out_dir: Path, record_type: str | None = None) -> Path | None:
    """Parse a .zst dump file and write to Parquet in out_dir.

    Output filename is prefixed by record type so load_raw() can find it:
      submissions_{stem}.parquet  or  comments_{stem}.parquet
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if record_type is None:
        record_type = detect_record_type(zst_path)

    stem = zst_path.stem.replace(".zst", "")
    out_path = out_dir / f"{record_type}_{stem}.parquet"

    rows = []
    for rec in tqdm(iter_records(zst_path), desc=zst_path.name, unit="rec"):
        sub = rec.get("subreddit", "")
        if sub not in SUBS_OF_INTEREST:
            continue
        if record_type == "submissions":
            rows.append({
                "id": str(rec.get("id", "")),
                "sub": sub,
                "author": str(rec.get("author", "")),
                "created_utc": float(rec.get("created_utc", 0)),
                "title": rec.get("title", "") or "",
                "body": rec.get("selftext", "") or "",
                "score": int(rec.get("score", 0) or 0),
                "num_comments": int(rec.get("num_comments", 0) or 0),
            })
        else:
            rows.append({
                "id": str(rec.get("id", "")),
                "sub": sub,
                "author": str(rec.get("author", "")),
                "created_utc": float(rec.get("created_utc", 0)),
                "body": rec.get("body", "") or "",
                "score": int(rec.get("score", 0) or 0),
                "link_id": str(rec.get("link_id", "")),
            })

    if not rows:
        print(f"No matching rows in {zst_path.name}")
        return None

    df = pl.DataFrame(rows)
    df.write_parquet(out_path)
    print(f"  {len(df):,} rows → {out_path.name}")
    return out_path
