"""Predictive maintenance: feature extraction, anomaly detection, trend analysis.

Uses the merged trace index (deduplicated sessions) and MAD-based anomaly
detection for robustness against skewed distributions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

from macchine.storage.catalog import get_merged_trace_index

# Minimum traces per machine to run anomaly detection
MIN_TRACES_FOR_ANALYSIS = 20


def _mad_zscore(series: pd.Series) -> pd.Series:
    """Compute modified z-scores using Median Absolute Deviation.

    MAD is robust to outliers and skewed distributions, unlike standard z-score.
    Modified z-score = 0.6745 * (x - median) / MAD
    where 0.6745 is the 0.75th quantile of the standard normal distribution.
    """
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0:
        # Fall back to mean absolute deviation if MAD is zero
        mad = (series - median).abs().mean()
    if mad == 0:
        return pd.Series(0.0, index=series.index)
    return 0.6745 * (series - median) / mad


def extract_features(output_dir: Path, machine: str | None = None) -> pd.DataFrame:
    """Extract statistical features per session for predictive analysis.

    Uses the merged trace index (deduplicated + session-merged) to avoid
    analyzing duplicate traces. Features computed from metadata:
    - Duration statistics (MAD-based z-scores)
    - Sensor count deviations
    - Temporal patterns (time of day, day of week)

    Returns a DataFrame with one row per session and extracted features.
    """
    df = get_merged_trace_index(output_dir)
    df = df.dropna(subset=["start_time"]).sort_values("start_time")

    if machine:
        df = df[df["machine_slug"] == machine]

    if df.empty:
        print("No matching traces found.")
        return pd.DataFrame()

    # Basic features
    cols = ["machine_slug", "site_id", "technique", "start_time",
            "duration_s", "sensor_count", "sample_count"]
    # Include trace_id if available
    if "trace_id" in df.columns:
        cols = ["trace_id"] + cols
    # Include source_path if available
    if "source_path" in df.columns:
        cols.append("source_path")

    features = df[[c for c in cols if c in df.columns]].copy()
    features["duration_min"] = features["duration_s"] / 60
    features["hour_of_day"] = features["start_time"].dt.hour
    features["day_of_week"] = features["start_time"].dt.dayofweek
    features["date"] = features["start_time"].dt.date

    # Per-machine MAD-based z-scores
    for slug in features["machine_slug"].unique():
        mask = features["machine_slug"] == slug
        idx = features[mask].index
        n_traces = mask.sum()

        if n_traces < MIN_TRACES_FOR_ANALYSIS:
            features.loc[idx, "duration_zscore"] = 0.0
            features.loc[idx, "sensor_deviation"] = 0.0
            continue

        # MAD-based z-score of duration
        features.loc[idx, "duration_zscore"] = _mad_zscore(
            features.loc[idx, "duration_min"]
        )

        # Sensor count deviation from technique median
        for tech in features.loc[mask, "technique"].unique():
            tech_mask = mask & (features["technique"] == tech)
            median_sensors = features.loc[tech_mask, "sensor_count"].median()
            features.loc[tech_mask, "sensor_deviation"] = (
                features.loc[tech_mask, "sensor_count"] - median_sensors
            )

    return features


def detect_anomalies(
    output_dir: Path,
    machine: str | None = None,
    mad_threshold: float = 3.5,
) -> pd.DataFrame:
    """Detect anomalous sessions using MAD-based modified z-scores.

    Uses median absolute deviation instead of standard z-scores for robustness
    against skewed duration distributions. Only analyzes machines with at least
    MIN_TRACES_FOR_ANALYSIS sessions.

    Returns sessions that deviate significantly from normal behavior.
    """
    features = extract_features(output_dir, machine=machine)

    if features.empty:
        return features

    # Flag anomalies using MAD-based threshold
    duration_anom = features["duration_zscore"].abs() > mad_threshold
    sensor_anom = features.get("sensor_deviation", pd.Series(dtype=float)).abs() > 5

    anomalies = features[duration_anom | sensor_anom].copy()

    print("Anomaly Detection Results (MAD-based)")
    print("=" * 80)
    print(f"Total sessions analyzed: {len(features)}")
    print(f"Machines with >= {MIN_TRACES_FOR_ANALYSIS} sessions: "
          f"{features[features['duration_zscore'] != 0]['machine_slug'].nunique()}")
    print(f"Anomalies detected: {len(anomalies)} ({len(anomalies)/len(features)*100:.1f}%)")

    if not anomalies.empty:
        print(f"\nTop anomalies (MAD threshold = {mad_threshold}):")
        for _, row in anomalies.nlargest(10, "duration_zscore", keep="first").iterrows():
            print(
                f"  {row['machine_slug']} | {row['start_time']} | "
                f"duration={row['duration_min']:.1f}min (MAD-z={row['duration_zscore']:.1f}) | "
                f"{row['technique']}"
            )

    return anomalies


def trend_summary(output_dir: Path) -> None:
    """Print a summary of trends across all machines."""
    features = extract_features(output_dir)

    if features.empty:
        return

    print("Fleet-wide Trend Summary")
    print("=" * 80)

    for slug in sorted(features["machine_slug"].unique()):
        m_df = features[features["machine_slug"] == slug].sort_values("start_time")
        if len(m_df) < MIN_TRACES_FOR_ANALYSIS:
            continue

        # Split into months and compare
        m_df["month"] = m_df["start_time"].dt.to_period("M")
        monthly = m_df.groupby("month").agg(
            traces=("duration_min", "count"),
            avg_duration=("duration_min", "mean"),
        )

        if len(monthly) < 2:
            continue

        first_month = monthly.iloc[0]
        last_month = monthly.iloc[-1]

        print(f"\n  {slug}: {len(m_df)} sessions over {len(monthly)} months")
        print(f"    First month: {monthly.index[0]} — {first_month['traces']} sessions, "
              f"avg {first_month['avg_duration']:.1f} min")
        print(f"    Last month:  {monthly.index[-1]} — {last_month['traces']} sessions, "
              f"avg {last_month['avg_duration']:.1f} min")
