"""Per-site data quality scoring and reporting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from macchine.storage.catalog import get_trace_index


@dataclass
class SiteQualityReport:
    """Quality scores (0-1) for a single site."""

    site_id: str
    trace_count: int
    machine_id_rate: float  # fraction of files with known machine
    operator_id_rate: float  # fraction of files with operator name
    element_naming_rate: float  # fraction with real element names (not empty/xxxxx/test)
    temporal_continuity: float  # 1 - (gap_days / total_span_days)
    sensor_completeness: float  # avg fraction of expected sensors present
    overall_score: float = 0.0

    def __post_init__(self):
        self.overall_score = (
            self.machine_id_rate * 0.25
            + self.operator_id_rate * 0.20
            + self.element_naming_rate * 0.20
            + self.temporal_continuity * 0.15
            + self.sensor_completeness * 0.20
        )

    def summary_line(self) -> str:
        return (
            f"  {self.site_id:25s} | {self.trace_count:5d} traces | "
            f"machine={self.machine_id_rate:.0%} operator={self.operator_id_rate:.0%} "
            f"naming={self.element_naming_rate:.0%} temporal={self.temporal_continuity:.0%} "
            f"sensors={self.sensor_completeness:.0%} | overall={self.overall_score:.0%}"
        )


def _is_good_element_name(name: str) -> bool:
    """Check if an element name is meaningful (not empty, xxxxx, test, etc.)."""
    if not name or not name.strip():
        return False
    lower = name.strip().lower()
    if all(c == "x" for c in lower):
        return False
    if lower in ("test", "testpile", "unknown", "0", "00"):
        return False
    return True


def _compute_temporal_continuity(dates: pd.Series) -> float:
    """Compute temporal continuity: 1 means evenly spaced, 0 means huge gaps."""
    dates = dates.dropna().sort_values()
    if len(dates) < 2:
        return 0.0

    total_span = (dates.max() - dates.min()).total_seconds() / 86400  # days
    if total_span <= 0:
        return 1.0

    # Count unique active days
    active_days = dates.dt.date.nunique()
    # Expected: 1 day per day of span
    expected_days = total_span + 1
    return min(active_days / expected_days, 1.0)


def compute_site_quality(site_df: pd.DataFrame, site_id: str) -> SiteQualityReport:
    """Compute quality scores for a single site."""
    n = len(site_df)
    if n == 0:
        return SiteQualityReport(site_id=site_id, trace_count=0, machine_id_rate=0, operator_id_rate=0,
                                  element_naming_rate=0, temporal_continuity=0, sensor_completeness=0)

    # Machine identification rate
    has_machine = (site_df["machine_slug"].notna() & (site_df["machine_slug"] != "") & (site_df["machine_slug"] != "unknown")).sum()
    machine_rate = has_machine / n

    # Operator identification rate
    has_operator = (site_df["operator"].notna() & (site_df["operator"] != "")).sum()
    operator_rate = has_operator / n

    # Element naming quality
    good_names = site_df["element_name"].apply(_is_good_element_name).sum()
    naming_rate = good_names / n

    # Temporal continuity
    if "start_time" in site_df.columns:
        times = pd.to_datetime(site_df["start_time"], errors="coerce")
        temporal = _compute_temporal_continuity(times)
    else:
        temporal = 0.0

    # Sensor completeness (relative to median sensor count for this technique)
    if "sensor_count" in site_df.columns and "technique" in site_df.columns:
        # Expected sensor count per technique (from full dataset)
        expected = site_df.groupby("technique")["sensor_count"].transform("median")
        completeness = (site_df["sensor_count"] / expected.clip(lower=1)).clip(upper=1.0).mean()
    else:
        completeness = 0.0

    return SiteQualityReport(
        site_id=site_id,
        trace_count=n,
        machine_id_rate=machine_rate,
        operator_id_rate=operator_rate,
        element_naming_rate=naming_rate,
        temporal_continuity=temporal,
        sensor_completeness=completeness,
    )


def generate_reports(output_dir: Path, site_filter: str | None = None) -> list[SiteQualityReport]:
    """Generate quality reports for all (or one) site(s)."""
    df = get_trace_index(output_dir)

    if "site_id" not in df.columns:
        print("No site_id column in trace index.")
        return []

    sites = df["site_id"].unique()
    if site_filter:
        sites = [s for s in sites if s == site_filter]

    reports = []
    print("Data Quality Reports")
    print("=" * 120)

    for site_id in sorted(sites):
        site_df = df[df["site_id"] == site_id]
        report = compute_site_quality(site_df, site_id)
        reports.append(report)
        print(report.summary_line())

    print("=" * 120)
    if reports:
        avg_score = sum(r.overall_score for r in reports) / len(reports)
        print(f"  Average overall quality score: {avg_score:.0%}")

    return reports
