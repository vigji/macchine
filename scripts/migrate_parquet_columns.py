"""One-time migration: rename German sensor columns to English in all parquet files.

Usage:
    python scripts/migrate_parquet_columns.py --dry-run   # preview only
    python scripts/migrate_parquet_columns.py              # apply changes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from macchine.harmonize.sensor_map import get_all_mappings

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def migrate_parquet_columns(output_dir: Path, dry_run: bool = False) -> None:
    traces_dir = output_dir / "traces"
    if not traces_dir.exists():
        print(f"Traces directory not found: {traces_dir}")
        return

    mapping = get_all_mappings()
    print(f"Loaded {len(mapping)} German -> English mappings")

    parquet_files = sorted(traces_dir.rglob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files\n")

    renamed_files = 0
    renamed_cols_total = 0
    skipped = 0
    errors = 0

    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
        except Exception as e:
            print(f"  ERROR reading {pf.relative_to(output_dir)}: {e}")
            errors += 1
            continue

        # Build rename dict for columns that exist and have a mapping
        rename = {}
        for col in df.columns:
            if col in mapping:
                rename[col] = mapping[col]

        if not rename:
            skipped += 1
            continue

        renamed_cols_total += len(rename)
        renamed_files += 1

        if dry_run:
            rel = pf.relative_to(output_dir)
            print(f"  [DRY-RUN] {rel}: {len(rename)} columns to rename")
            for old, new in sorted(rename.items()):
                print(f"    {old} -> {new}")
        else:
            df = df.rename(columns=rename)
            df.to_parquet(pf, engine="pyarrow", index=False)

    print(f"\n{'DRY-RUN ' if dry_run else ''}Migration complete:")
    print(f"  Files renamed: {renamed_files}")
    print(f"  Columns renamed: {renamed_cols_total}")
    print(f"  Files skipped (no German columns): {skipped}")
    print(f"  Errors: {errors}")


def main():
    parser = argparse.ArgumentParser(description="Migrate parquet columns from German to English.")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    migrate_parquet_columns(output_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
