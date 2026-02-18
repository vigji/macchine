"""Machine degradation analysis: rolling statistics, baseline drift, performance comparison."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from macchine.storage.catalog import get_trace_index


def analyze_degradation(
    output_dir: Path,
    machine: str | None = None,
    window_days: int = 30,
) -> pd.DataFrame:
    """Analyze machine performance trends over time.

    Computes rolling statistics of key metrics (duration, sensor count)
    to detect degradation or performance changes.

    Returns a DataFrame with time-windowed statistics.
    """
    df = get_trace_index(output_dir)
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df = df.dropna(subset=["start_time"]).sort_values("start_time")

    if machine:
        df = df[df["machine_slug"] == machine]

    if df.empty:
        print("No matching traces found.")
        return pd.DataFrame()

    df["date"] = df["start_time"].dt.date
    df["duration_min"] = df["duration_s"] / 60

    print("Machine Performance Trends")
    print("=" * 80)

    results = []
    for slug in sorted(df["machine_slug"].unique()):
        m_df = df[df["machine_slug"] == slug].copy()
        if len(m_df) < 10:
            continue

        m_df = m_df.set_index("start_time").sort_index()

        # Rolling statistics
        rolling = m_df["duration_min"].rolling(f"{window_days}D", min_periods=5)
        m_df["rolling_mean_duration"] = rolling.mean()
        m_df["rolling_std_duration"] = rolling.std()

        # First and last period comparison
        n = len(m_df)
        first_quarter = m_df.iloc[: n // 4]
        last_quarter = m_df.iloc[3 * n // 4 :]

        print(f"\n  Machine: {slug} ({len(m_df)} traces)")
        print(f"    Date range: {m_df.index.min().date()} to {m_df.index.max().date()}")
        print(f"    First quarter avg duration: {first_quarter['duration_min'].mean():.1f} min")
        print(f"    Last quarter avg duration: {last_quarter['duration_min'].mean():.1f} min")

        duration_change = last_quarter["duration_min"].mean() - first_quarter["duration_min"].mean()
        pct = duration_change / max(first_quarter["duration_min"].mean(), 0.1) * 100
        direction = "increase" if duration_change > 0 else "decrease"
        print(f"    Duration change: {duration_change:+.1f} min ({pct:+.1f}% {direction})")

        # Per-site performance
        m_df_flat = m_df.reset_index()
        for site_id in m_df_flat["site_id"].unique():
            site_data = m_df_flat[m_df_flat["site_id"] == site_id]
            print(
                f"    Site {site_id}: {len(site_data)} traces, "
                f"avg duration {site_data['duration_min'].mean():.1f} min"
            )

        results.append({
            "machine_slug": slug,
            "trace_count": len(m_df),
            "first_q_mean_min": first_quarter["duration_min"].mean(),
            "last_q_mean_min": last_quarter["duration_min"].mean(),
            "duration_change_pct": pct,
        })

    return pd.DataFrame(results)
