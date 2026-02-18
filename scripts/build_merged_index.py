"""Build a deduplicated and merged trace index.

Phase 1: Remove incremental upload duplicates — keep longest trace per (site_id, element_name, start_time).
Phase 2: Merge abutting traces — gap < MERGE_GAP_MINUTES between end of one and start of next.
Phase 3: Generate merge statistics report.

Output: output/metadata/merged_trace_index.parquet
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

from macchine.storage.catalog import get_trace_index

OUTPUT_DIR = Path("output")
MERGE_GAP_MINUTES = 5  # max gap between traces to consider them part of the same session


def load_data() -> pd.DataFrame:
    df = get_trace_index(OUTPUT_DIR)
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df["duration_min"] = df["duration_s"] / 60
    return df


def phase1_dedup(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove incremental upload duplicates: keep longest trace per (site_id, element_name, start_time)."""
    n_before = len(df)

    # Round start_time to the nearest second to handle minor timestamp differences
    df["start_time_rounded"] = df["start_time"].dt.round("s")

    # For each group, keep the row with the longest duration
    idx_keep = df.groupby(["site_id", "element_name", "start_time_rounded"])["duration_s"].idxmax()
    df_dedup = df.loc[idx_keep].copy()
    df_dedup = df_dedup.drop(columns=["start_time_rounded"])

    n_after = len(df_dedup)
    n_removed = n_before - n_after

    # Compute statistics about what was removed
    group_sizes = df.groupby(["site_id", "element_name", "start_time_rounded"]).size()
    n_dup_groups = (group_sizes > 1).sum()

    stats = {
        "n_before": n_before,
        "n_after": n_after,
        "n_removed": n_removed,
        "pct_removed": n_removed / n_before * 100,
        "n_duplicate_groups": int(n_dup_groups),
        "max_group_size": int(group_sizes.max()),
    }
    return df_dedup, stats


def phase2_merge(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Merge abutting traces for the same element with gap < MERGE_GAP_MINUTES.

    This creates session-level metadata: merged traces get combined duration, sensor count
    from the longest trace, and a list of constituent trace_ids.
    """
    n_before = len(df)
    df = df.sort_values(["site_id", "element_name", "start_time"]).reset_index(drop=True)

    # Compute end time and gap to next trace for same element
    df["end_time"] = df["start_time"] + pd.to_timedelta(df["duration_s"], unit="s")

    sessions = []
    current_session = None
    n_merges = 0

    for _, row in df.iterrows():
        if current_session is None:
            current_session = _new_session(row)
            continue

        # Check if same element and gap is small
        same_element = (
            row["site_id"] == current_session["site_id"]
            and row["element_name"] == current_session["element_name"]
        )

        if same_element:
            gap_s = (row["start_time"] - current_session["end_time"]).total_seconds()
            gap_min = gap_s / 60

            if 0 <= gap_min <= MERGE_GAP_MINUTES:
                # Merge: extend session
                current_session["end_time"] = row["end_time"]
                current_session["duration_s"] = (
                    current_session["end_time"] - current_session["start_time"]
                ).total_seconds()
                current_session["duration_min"] = current_session["duration_s"] / 60
                current_session["sample_count"] += row["sample_count"]
                current_session["n_traces"] += 1
                current_session["trace_ids"].append(row["trace_id"])
                # Keep sensor_count from trace with most sensors
                if row["sensor_count"] > current_session["sensor_count"]:
                    current_session["sensor_count"] = row["sensor_count"]
                n_merges += 1
                continue

        # Finalize current session and start new one
        sessions.append(current_session)
        current_session = _new_session(row)

    if current_session is not None:
        sessions.append(current_session)

    # Build merged dataframe
    df_merged = pd.DataFrame(sessions)
    df_merged["trace_ids"] = df_merged["trace_ids"].apply(lambda x: "|".join(x))

    n_after = len(df_merged)
    stats = {
        "n_before": n_before,
        "n_after": n_after,
        "n_merges": n_merges,
        "n_sessions_multi": int((df_merged["n_traces"] > 1).sum()),
        "max_traces_per_session": int(df_merged["n_traces"].max()),
    }
    return df_merged, stats


def _new_session(row) -> dict:
    return {
        "trace_id": row["trace_id"],
        "trace_ids": [row["trace_id"]],
        "element_name": row["element_name"],
        "element_id": row.get("element_id"),
        "site_id": row["site_id"],
        "machine_serial": row["machine_serial"],
        "machine_model": row["machine_model"],
        "machine_number": row["machine_number"],
        "machine_slug": row["machine_slug"],
        "technique": row["technique"],
        "start_time": row["start_time"],
        "end_time": row["start_time"] + pd.Timedelta(seconds=row["duration_s"]),
        "upload_time": row.get("upload_time"),
        "medef_version": row.get("medef_version"),
        "operator": row["operator"],
        "sensor_count": row["sensor_count"],
        "sample_count": row["sample_count"],
        "duration_s": row["duration_s"],
        "duration_min": row["duration_s"] / 60,
        "format": row["format"],
        "source_path": row["source_path"],
        "n_traces": 1,
    }


def print_stats(dedup_stats: dict, merge_stats: dict):
    print("=" * 60)
    print("TRACE MERGING REPORT")
    print("=" * 60)

    print("\n--- Phase 1: Incremental Upload Deduplication ---")
    print(f"  Input traces:           {dedup_stats['n_before']:,}")
    print(f"  Duplicate groups found: {dedup_stats['n_duplicate_groups']:,}")
    print(f"  Max duplicates/group:   {dedup_stats['max_group_size']}")
    print(f"  Traces removed:         {dedup_stats['n_removed']:,} ({dedup_stats['pct_removed']:.1f}%)")
    print(f"  Traces remaining:       {dedup_stats['n_after']:,}")

    print("\n--- Phase 2: Abutting Trace Merge (gap < {MERGE_GAP_MINUTES} min) ---")
    print(f"  Input traces:           {merge_stats['n_before']:,}")
    print(f"  Merges performed:       {merge_stats['n_merges']:,}")
    print(f"  Multi-trace sessions:   {merge_stats['n_sessions_multi']:,}")
    print(f"  Max traces/session:     {merge_stats['max_traces_per_session']}")
    print(f"  Final sessions:         {merge_stats['n_after']:,}")

    total_removed = dedup_stats["n_before"] - merge_stats["n_after"]
    print(f"\n--- Overall ---")
    print(f"  {dedup_stats['n_before']:,} traces → {merge_stats['n_after']:,} sessions")
    print(f"  Reduction: {total_removed:,} ({total_removed / dedup_stats['n_before'] * 100:.1f}%)")


def main():
    print("Loading trace index...")
    df = load_data()
    print(f"  Loaded {len(df):,} traces")

    print("\nPhase 1: Deduplicating incremental uploads...")
    df_dedup, dedup_stats = phase1_dedup(df)

    print("Phase 2: Merging abutting traces...")
    df_merged, merge_stats = phase2_merge(df_dedup)

    print_stats(dedup_stats, merge_stats)

    # Save
    out_path = OUTPUT_DIR / "metadata" / "merged_trace_index.parquet"
    df_merged.to_parquet(out_path, index=False)
    print(f"\nSaved merged index to {out_path}")

    # Also print duration distribution comparison
    print("\n--- Duration Distribution (before vs after) ---")
    for label, d in [("Original", df), ("Merged", df_merged)]:
        print(f"  {label}:")
        print(f"    Count: {len(d):,}")
        print(f"    Median: {d['duration_min'].median():.1f} min")
        print(f"    Mean:   {d['duration_min'].mean():.1f} min")
        print(f"    P95:    {d['duration_min'].quantile(0.95):.1f} min")

    # Per-site summary
    print("\n--- Per-Site Summary ---")
    site_stats = df_merged.groupby("site_id").agg(
        sessions=("trace_id", "count"),
        elements=("element_name", "nunique"),
        total_hours=("duration_min", lambda x: x.sum() / 60),
    ).sort_values("sessions", ascending=False)
    print(site_stats.to_string())


if __name__ == "__main__":
    main()
