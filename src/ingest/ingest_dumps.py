"""
Batch ingest all .zst dump files in a directory into data/raw/ Parquet files.

Usage:
    python -m src.ingest.ingest_dumps                        # uses data/dumps/
    python -m src.ingest.ingest_dumps --dumps-dir /path/to/dumps
    python -m src.ingest.ingest_dumps --submissions-only
"""

import argparse
from pathlib import Path

from src.ingest.reddit_dumps import dump_to_parquet, detect_record_type

DUMPS_DIR = Path(__file__).parent.parent.parent / "data" / "dumps"
RAW_DIR   = Path(__file__).parent.parent.parent / "data" / "raw"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dumps-dir", default=str(DUMPS_DIR), help="Directory containing .zst files")
    p.add_argument("--out-dir",   default=str(RAW_DIR),   help="Output directory for Parquet files")
    p.add_argument("--submissions-only", action="store_true", help="Skip comment files (faster)")
    args = p.parse_args()

    dumps_dir = Path(args.dumps_dir)
    out_dir   = Path(args.out_dir)

    files = sorted(dumps_dir.glob("*.zst"))
    if not files:
        print(f"No .zst files found in {dumps_dir}")
        return

    # Skip double-underscore files (suspected test/partial dumps)
    skipped = [f for f in files if "__" in f.stem]
    files   = [f for f in files if "__" not in f.stem]

    if skipped:
        print(f"Skipping {len(skipped)} double-underscore files: {[f.name for f in skipped]}")

    if args.submissions_only:
        files = [f for f in files if "submission" in f.stem.lower()]
        print("--submissions-only: skipping comment files")

    print(f"\nIngesting {len(files)} files → {out_dir}\n")
    for zst_path in files:
        try:
            record_type = detect_record_type(zst_path)
        except ValueError as e:
            print(f"  Skipping {zst_path.name}: {e}")
            continue

        print(f"[{record_type}] {zst_path.name}  ({zst_path.stat().st_size / 1e6:.0f} MB)")
        dump_to_parquet(zst_path, out_dir, record_type=record_type)

    print(f"\nDone. Parquet files in {out_dir}:")
    for f in sorted(out_dir.glob("*.parquet")):
        print(f"  {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
