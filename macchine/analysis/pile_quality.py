"""Pile/element quality analysis: depth, duration, sensor statistics per element."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from macchine.storage.catalog import get_trace_index


def analyze_pile_quality(
    output_dir: Path,
    site: str | None = None,
    technique: str | None = None,
) -> pd.DataFrame:
    """Analyze element-level quality metrics from the trace index.

    Returns a DataFrame with per-element statistics.
    """
    df = get_trace_index(output_dir)
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")

    if site:
        df = df[df["site_id"] == site]
    if technique:
        df = df[df["technique"] == technique.upper()]

    if df.empty:
        print("No matching traces found.")
        return pd.DataFrame()

    df["duration_min"] = df["duration_s"] / 60

    print("Element Quality Analysis")
    print("=" * 80)

    # Per-technique statistics
    for tech in sorted(df["technique"].unique()):
        t_df = df[df["technique"] == tech]
        print(f"\n  Technique: {tech} ({len(t_df)} elements)")
        print(f"    Duration: mean={t_df['duration_min'].mean():.1f} min, "
              f"median={t_df['duration_min'].median():.1f} min, "
              f"std={t_df['duration_min'].std():.1f} min")
        print(f"    Sensors per element: mean={t_df['sensor_count'].mean():.1f}, "
              f"range={t_df['sensor_count'].min()}-{t_df['sensor_count'].max()}")

        # Identify outliers (elements with unusually short or long duration)
        q1 = t_df["duration_min"].quantile(0.25)
        q3 = t_df["duration_min"].quantile(0.75)
        iqr = q3 - q1
        short = t_df[t_df["duration_min"] < q1 - 1.5 * iqr]
        long = t_df[t_df["duration_min"] > q3 + 1.5 * iqr]
        if len(short) > 0 or len(long) > 0:
            print(f"    Outliers: {len(short)} very short, {len(long)} very long")

    # Per-site element counts and naming quality
    print("\nPer-site element summary:")
    for sid in sorted(df["site_id"].unique()):
        s_df = df[df["site_id"] == sid]
        named = s_df[s_df["element_name"].notna() & (s_df["element_name"] != "")]
        unique_names = named["element_name"].nunique()
        print(
            f"  {sid:25s}: {len(s_df)} elements, {unique_names} unique names, "
            f"techniques: {', '.join(sorted(s_df['technique'].unique()))}"
        )

    return df[["site_id", "element_name", "technique", "machine_slug", "operator",
               "duration_min", "sensor_count", "sample_count", "start_time"]].copy()
