"""Operator comparison: production speed, depth profiles, statistical tests.

Includes both the original broad comparison and a controlled comparison
that accounts for technique, site, and depth confounders.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from macchine.storage.catalog import get_trace_index, get_merged_trace_index


def compare_operators(
    output_dir: Path,
    machine: str | None = None,
    site: str | None = None,
) -> pd.DataFrame:
    """Compare operator performance metrics.

    Returns a DataFrame with per-operator statistics.
    """
    df = get_trace_index(output_dir)
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df = df[df["operator"].notna() & (df["operator"] != "")]

    if machine:
        df = df[df["machine_slug"] == machine]
    if site:
        df = df[df["site_id"] == site]

    if df.empty:
        print("No traces with operator info found.")
        return pd.DataFrame()

    df["duration_min"] = df["duration_s"] / 60

    # Per-operator aggregation
    stats = (
        df.groupby("operator")
        .agg(
            trace_count=("source_path", "count"),
            machines_used=("machine_slug", "nunique"),
            sites=("site_id", "nunique"),
            techniques=("technique", lambda x: ", ".join(sorted(x.unique()))),
            total_hours=("duration_s", lambda x: x.sum() / 3600),
            avg_duration_min=("duration_min", "mean"),
            median_duration_min=("duration_min", "median"),
            avg_sensors=("sensor_count", "mean"),
        )
        .reset_index()
        .sort_values("trace_count", ascending=False)
    )

    print("Operator Comparison")
    print("=" * 100)
    for _, row in stats.iterrows():
        print(
            f"  {row['operator']:20s} | {row['trace_count']:4d} traces | "
            f"{row['total_hours']:.1f}h | "
            f"avg={row['avg_duration_min']:.1f}min med={row['median_duration_min']:.1f}min | "
            f"machines={row['machines_used']} | {row['techniques']}"
        )

    # Statistical comparison: per-machine, compare operator durations
    print("\nPer-machine operator comparison (avg element duration in min):")
    for slug in df["machine_slug"].unique():
        m_df = df[df["machine_slug"] == slug]
        operators = m_df["operator"].unique()
        if len(operators) < 2:
            continue

        print(f"\n  Machine: {slug}")
        op_stats = m_df.groupby("operator")["duration_min"].describe()
        print(op_stats.to_string(float_format="%.1f"))

        # Mann-Whitney U test if scipy is available
        try:
            from scipy.stats import mannwhitneyu

            ops = sorted(operators)
            for i in range(len(ops)):
                for j in range(i + 1, len(ops)):
                    a = m_df[m_df["operator"] == ops[i]]["duration_min"]
                    b = m_df[m_df["operator"] == ops[j]]["duration_min"]
                    if len(a) >= 5 and len(b) >= 5:
                        stat, p = mannwhitneyu(a, b, alternative="two-sided")
                        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                        print(f"    {ops[i]} vs {ops[j]}: U={stat:.0f}, p={p:.4f} {sig}")
        except ImportError:
            pass

    return stats


def controlled_operator_comparison(
    output_dir: Path,
    machine: str,
    technique: str | None = None,
    site: str | None = None,
    depth_tolerance: float = 5.0,
    min_traces_per_operator: int = 10,
) -> pd.DataFrame:
    """Compare operators with controls for technique, site, and depth.

    Filters to a specific machine + technique + site, groups traces into
    depth bins, and compares operator metrics within each depth bin using
    Mann-Whitney U tests.

    Parameters
    ----------
    output_dir : Path to output directory
    machine : Machine slug (required for controlled comparison)
    technique : Technique filter (if None, uses most common)
    site : Site filter (if None, uses most common)
    depth_tolerance : Width of depth bins in meters (default 5m)
    min_traces_per_operator : Minimum traces required per operator

    Returns DataFrame with per-depth-bin, per-operator-pair comparison results.
    """
    df = get_merged_trace_index(output_dir)
    df = df.dropna(subset=["start_time"])
    df = df[df["operator"].notna() & (df["operator"] != "")]
    df = df[df["machine_slug"] == machine]

    if technique:
        df = df[df["technique"] == technique]
    else:
        # Use most common technique
        if df.empty:
            return pd.DataFrame()
        technique = df["technique"].mode().iloc[0]
        df = df[df["technique"] == technique]

    if site:
        df = df[df["site_id"] == site]
    else:
        # Use most common site
        if df.empty:
            return pd.DataFrame()
        site = df["site_id"].mode().iloc[0]
        df = df[df["site_id"] == site]

    if df.empty:
        print(f"No traces found for {machine}/{technique}@{site}")
        return pd.DataFrame()

    df["duration_min"] = df["duration_s"] / 60

    # Filter to operators with enough traces
    op_counts = df["operator"].value_counts()
    valid_ops = op_counts[op_counts >= min_traces_per_operator].index.tolist()

    if len(valid_ops) < 2:
        print(f"Need >=2 operators with >={min_traces_per_operator} traces. "
              f"Found {len(valid_ops)}: {valid_ops}")
        return pd.DataFrame()

    df = df[df["operator"].isin(valid_ops)]

    print(f"\nControlled Operator Comparison: {machine} / {technique} @ {site}")
    print(f"  Operators: {', '.join(valid_ops)} ({len(df)} sessions total)")
    print("=" * 80)

    # Try to load sensor features for depth-normalized metrics
    sensor_features = None
    try:
        from macchine.analysis.sensor_degradation import extract_sensor_features
        sf = extract_sensor_features(output_dir, machine=machine, technique=technique, site=site)
        if not sf.empty and "max_depth" in sf.columns:
            sensor_features = sf[["trace_id", "max_depth"]].dropna()
    except Exception:
        pass

    # Merge depth info if available
    if sensor_features is not None and not sensor_features.empty:
        df = df.merge(sensor_features, on="trace_id", how="left")
    elif "max_depth" not in df.columns:
        # No depth info available; compare without depth binning
        df["max_depth"] = np.nan

    # Try to merge cycle-level summary stats for GRAB
    if technique == "GRAB":
        try:
            from macchine.analysis.cycle_analysis import track_cycle_degradation

            cycle_df = track_cycle_degradation(output_dir, machine, site=site)
            if not cycle_df.empty:
                # Aggregate per session
                cycle_agg = (
                    cycle_df.groupby("trace_id")
                    .agg(
                        n_cycles=("cycle_depth", "count"),
                        med_cycle_depth=("cycle_depth", "median"),
                        med_cycle_duration=("cycle_duration_s", "median"),
                        med_descent_speed=("descent_speed", "median"),
                        med_ascent_speed=("ascent_speed", "median"),
                    )
                    .reset_index()
                )
                df = df.merge(cycle_agg, on="trace_id", how="left")
                print(f"  Merged cycle metrics for {cycle_agg['trace_id'].nunique()} sessions")
        except Exception as e:
            print(f"  Could not load cycle metrics: {e}")

    # Cycle-level metrics available for comparison (if GRAB merge succeeded)
    cycle_metrics = ["med_cycle_duration", "med_descent_speed", "med_ascent_speed", "n_cycles"]

    results = []

    # If we have depth data, do binned comparison
    has_depth = df["max_depth"].notna().sum() > len(df) * 0.5

    if has_depth:
        # Create depth bins
        min_depth = df["max_depth"].min()
        max_depth = df["max_depth"].max()
        bins = np.arange(min_depth, max_depth + depth_tolerance, depth_tolerance)

        if len(bins) >= 2:
            df["depth_bin"] = pd.cut(df["max_depth"], bins=bins)
            print(f"\n  Depth bins: {len(bins)-1} bins of {depth_tolerance}m "
                  f"from {min_depth:.1f}m to {max_depth:.1f}m")

            # Duration per meter (depth-normalized)
            df["duration_per_meter"] = df["duration_min"] / df["max_depth"]

            for bin_label, bin_df in df.groupby("depth_bin", observed=True):
                if len(bin_df) < 10:
                    continue

                bin_metrics = ["duration_min", "duration_per_meter"]
                bin_metrics += [m for m in cycle_metrics if m in bin_df.columns]
                _compare_in_group(
                    bin_df, valid_ops, results,
                    group_label=f"depth={bin_label}",
                    metrics=bin_metrics,
                )
    else:
        print("  No depth data available; comparing duration only (uncontrolled for depth)")

    # Also do overall comparison (all depths)
    overall_metrics = ["duration_min"]
    overall_metrics += [m for m in cycle_metrics if m in df.columns]
    _compare_in_group(
        df, valid_ops, results,
        group_label="all_depths",
        metrics=overall_metrics,
    )

    # Print summary
    results_df = pd.DataFrame(results) if results else pd.DataFrame()

    if not results_df.empty:
        sig_results = results_df[results_df["significant"]]
        print(f"\n  Total comparisons: {len(results_df)}")
        print(f"  Significant (p < 0.05): {len(sig_results)}")

        if not sig_results.empty:
            print("\n  Significant differences:")
            for _, r in sig_results.iterrows():
                print(f"    {r['group']} | {r['operator_a']} vs {r['operator_b']} | "
                      f"metric={r['metric']} | "
                      f"median_a={r['median_a']:.1f} vs median_b={r['median_b']:.1f} | "
                      f"p={r['p_value']:.4f}")
    else:
        print("  No comparisons could be made.")

    return results_df


def _compare_in_group(
    group_df: pd.DataFrame,
    operators: list[str],
    results: list[dict],
    group_label: str,
    metrics: list[str],
):
    """Run Mann-Whitney U between all operator pairs within a group."""
    try:
        from scipy.stats import mannwhitneyu
    except ImportError:
        return

    for metric in metrics:
        if metric not in group_df.columns:
            continue

        for i in range(len(operators)):
            for j in range(i + 1, len(operators)):
                a = group_df[group_df["operator"] == operators[i]][metric].dropna()
                b = group_df[group_df["operator"] == operators[j]][metric].dropna()

                if len(a) < 5 or len(b) < 5:
                    continue

                stat, p = mannwhitneyu(a, b, alternative="two-sided")

                results.append({
                    "group": group_label,
                    "metric": metric,
                    "operator_a": operators[i],
                    "operator_b": operators[j],
                    "n_a": len(a),
                    "n_b": len(b),
                    "median_a": float(a.median()),
                    "median_b": float(b.median()),
                    "u_statistic": float(stat),
                    "p_value": float(p),
                    "significant": p < 0.05,
                })
