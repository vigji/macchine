"""Machine utilization analysis: active hours, elements per day, downtime."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from macchine.storage.catalog import get_trace_index


def compute_utilization(output_dir: Path, machine: str | None = None, site: str | None = None) -> pd.DataFrame:
    """Compute daily utilization metrics.

    Returns a DataFrame with columns: date, machine_slug, site_id, active_hours,
    elements_completed, avg_duration_min.
    """
    df = get_trace_index(output_dir)
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df = df.dropna(subset=["start_time"])

    if machine:
        df = df[df["machine_slug"] == machine]
    if site:
        df = df[df["site_id"] == site]

    if df.empty:
        print("No matching traces found.")
        return pd.DataFrame()

    df["date"] = df["start_time"].dt.date
    df["duration_min"] = df["duration_s"] / 60

    # Daily aggregation per machine per site
    daily = (
        df.groupby(["date", "machine_slug", "site_id"])
        .agg(
            active_hours=("duration_s", lambda x: x.sum() / 3600),
            elements_completed=("element_name", "count"),
            avg_duration_min=("duration_min", "mean"),
        )
        .reset_index()
    )

    # Summary
    print("Utilization Summary")
    print("=" * 80)

    for slug in daily["machine_slug"].unique():
        m_data = daily[daily["machine_slug"] == slug]
        total_days = m_data["date"].nunique()
        total_hours = m_data["active_hours"].sum()
        total_elements = m_data["elements_completed"].sum()
        avg_daily_hours = total_hours / max(total_days, 1)

        print(f"\n  Machine: {slug}")
        print(f"    Active days: {total_days}")
        print(f"    Total recording hours: {total_hours:.1f}")
        print(f"    Elements completed: {total_elements}")
        print(f"    Avg daily recording hours: {avg_daily_hours:.1f}")
        print(f"    Avg elements per active day: {total_elements / max(total_days, 1):.1f}")

        # Detect gaps (downtime)
        dates = sorted(m_data["date"].unique())
        if len(dates) > 1:
            gaps = []
            for i in range(1, len(dates)):
                gap_days = (dates[i] - dates[i - 1]).days
                if gap_days > 3:  # more than 3 days gap
                    gaps.append((dates[i - 1], dates[i], gap_days))
            if gaps:
                print(f"    Downtime gaps (>{3} days):")
                for start, end, days in gaps[:5]:
                    print(f"      {start} to {end}: {days} days")

    return daily
