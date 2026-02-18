"""Core data quality analysis functions.

Pure analysis — no file I/O, no matplotlib.  Takes DataFrames and returns
structured results via dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from macchine.harmonize.calibration import (
    PHYSICAL_RANGES,
    get_sentinel_values,
    is_calibrated,
)
from macchine.storage.catalog import _classify_pile_continuity

# ---------------------------------------------------------------------------
# Zeroing thresholds: sensors that should read ~0 at the start of a trace.
# Maps sensor name prefix → absolute threshold (in calibrated units).
# ---------------------------------------------------------------------------
ZEROING_THRESHOLDS: dict[str, float] = {
    "Tiefe": 0.5,               # m — depth should start near surface
    "Vorschub Tiefe": 0.5,      # m
    "Tiefe Winde 2": 0.5,       # m
    "Tiefe_Hauptwinde_GOK": 0.5,
    "Tiefe_Bohrrohr_GOK": 0.5,
    "Neigung X": 2.0,           # deg — mast inclination near vertical
    "Neigung Y": 2.0,
    "Neigung X Mast": 2.0,
    "Neigung Y Mast": 2.0,
    "Abweichung X": 50.0,       # mm — deviation from target
    "Abweichung Y": 50.0,
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SiteOverview:
    """Aggregate metadata for one site (from merged index only)."""

    site_id: str
    n_machines: int
    machine_slugs: list[str]
    techniques: list[str]
    date_min: str
    date_max: str
    n_traces: int
    operator_rate: float          # fraction with non-empty operator
    naming_rate: float            # fraction with non-empty element_name
    formats: dict[str, int]       # {"json": N, "dat": M}


@dataclass
class CalibrationSummary:
    """Per-machine sensor calibration counts."""

    machine_slug: str
    n_calibrated: int
    n_uncalibrated: int
    calibrated_sensors: list[str]
    uncalibrated_sensors: list[str]


@dataclass
class SensorIssue:
    """A single detected sensor quality issue."""

    sensor: str
    machine_slug: str
    trace_id: str
    issue_type: str               # RANGE_VIOLATION | DEAD | STUCK | ZEROING
    severity: str                 # warning | error
    detail: str = ""


@dataclass
class ElementNameAnalysis:
    """Naming quality analysis for a site."""

    total_traces: int
    unnamed_count: int
    unnamed_rate: float
    duplicate_names: list[dict]   # [{name, n_sessions, technique, continuity}]
    naming_patterns: list[str]    # most common name prefixes


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def compute_site_overview(site_traces: pd.DataFrame, site_id: str) -> SiteOverview:
    """Compute overview statistics from merged index rows for one site."""
    machines = sorted(site_traces["machine_slug"].unique().tolist())
    techniques = sorted(site_traces["technique"].dropna().unique().tolist())

    times = pd.to_datetime(site_traces["start_time"], errors="coerce").dropna()
    date_min = str(times.min().date()) if len(times) else ""
    date_max = str(times.max().date()) if len(times) else ""

    has_operator = site_traces["operator"].fillna("").str.strip().ne("")
    operator_rate = has_operator.mean() if len(site_traces) else 0.0

    has_name = site_traces["element_name"].fillna("").str.strip().ne("")
    naming_rate = has_name.mean() if len(site_traces) else 0.0

    fmt_counts = site_traces["format"].value_counts().to_dict() if "format" in site_traces.columns else {}

    return SiteOverview(
        site_id=site_id,
        n_machines=len(machines),
        machine_slugs=machines,
        techniques=techniques,
        date_min=date_min,
        date_max=date_max,
        n_traces=len(site_traces),
        operator_rate=operator_rate,
        naming_rate=naming_rate,
        formats=fmt_counts,
    )


def compute_calibration_summary(
    sensor_columns: dict[str, list[str]],
) -> list[CalibrationSummary]:
    """Compute calibration status per machine.

    Parameters
    ----------
    sensor_columns : {machine_slug: [sensor_name, ...]} discovered from parquets.
    """
    summaries = []
    for machine_slug, sensors in sorted(sensor_columns.items()):
        cal = [s for s in sensors if is_calibrated(s, machine_slug)]
        uncal = [s for s in sensors if not is_calibrated(s, machine_slug)]
        summaries.append(CalibrationSummary(
            machine_slug=machine_slug,
            n_calibrated=len(cal),
            n_uncalibrated=len(uncal),
            calibrated_sensors=sorted(cal),
            uncalibrated_sensors=sorted(uncal),
        ))
    return summaries


def detect_sensor_issues(
    trace_df: pd.DataFrame,
    machine_slug: str,
    trace_id: str,
) -> list[SensorIssue]:
    """Run quality checks on every numeric column of a single trace."""
    issues: list[SensorIssue] = []
    sentinels = set(get_sentinel_values())

    for col in trace_df.select_dtypes(include=[np.number]).columns:
        if col == "timestamp":
            continue

        raw = trace_df[col].dropna()
        # Clean sentinels for analysis
        clean = raw[~raw.isin(sentinels)]
        n = len(clean)
        if n < 5:
            continue

        calibrated = is_calibrated(col, machine_slug)

        # --- RANGE_VIOLATION (calibrated sensors only) ---
        if calibrated and col in PHYSICAL_RANGES:
            lo, hi = PHYSICAL_RANGES[col]
            out = ((clean < lo) | (clean > hi)).sum()
            pct = 100 * out / n
            if pct > 5:
                issues.append(SensorIssue(
                    sensor=col, machine_slug=machine_slug, trace_id=trace_id,
                    issue_type="RANGE_VIOLATION", severity="error",
                    detail=f"{pct:.1f}% outside [{lo}, {hi}]",
                ))

        # --- DEAD (all values exactly 0 after sentinel cleaning) ---
        if (clean == 0).all():
            issues.append(SensorIssue(
                sensor=col, machine_slug=machine_slug, trace_id=trace_id,
                issue_type="DEAD", severity="warning",
                detail=f"all {n} values are exactly 0",
            ))
            continue  # skip STUCK check if dead

        # --- STUCK (<4 unique values across >100 samples) ---
        if n > 100:
            n_unique = clean.nunique()
            if n_unique < 4:
                issues.append(SensorIssue(
                    sensor=col, machine_slug=machine_slug, trace_id=trace_id,
                    issue_type="STUCK", severity="warning",
                    detail=f"only {n_unique} unique values across {n} samples",
                ))

        # --- ZEROING (first 10 samples offset from 0) ---
        if col in ZEROING_THRESHOLDS and calibrated:
            threshold = ZEROING_THRESHOLDS[col]
            head = clean.iloc[:10] if len(clean) >= 10 else clean
            mean_start = head.mean()
            if abs(mean_start) > threshold:
                issues.append(SensorIssue(
                    sensor=col, machine_slug=machine_slug, trace_id=trace_id,
                    issue_type="ZEROING", severity="warning",
                    detail=f"mean of first 10 samples = {mean_start:.2f} (threshold ±{threshold})",
                ))

    return issues


def analyze_element_names(site_traces: pd.DataFrame) -> ElementNameAnalysis:
    """Analyze naming quality and detect duplicate/reused element names."""
    total = len(site_traces)
    names = site_traces["element_name"].fillna("").str.strip()
    unnamed = (names == "") | (names == "xxxxx")
    unnamed_count = int(unnamed.sum())

    # Duplicate / reused name detection
    named = site_traces[~unnamed].copy()
    duplicates = []
    if not named.empty:
        for name, grp in named.groupby("element_name"):
            if len(grp) <= 1:
                continue
            technique = grp["technique"].mode().iloc[0] if not grp["technique"].mode().empty else ""
            times = pd.to_datetime(grp["start_time"], errors="coerce").dropna()
            if len(times) >= 2:
                span_hours = (times.max() - times.min()).total_seconds() / 3600
            else:
                span_hours = 0.0
            continuity = _classify_pile_continuity(technique, span_hours, len(grp))
            duplicates.append({
                "name": name,
                "n_sessions": len(grp),
                "technique": technique,
                "continuity": continuity,
            })

    # Naming patterns — extract alphabetic prefix
    prefixes: list[str] = []
    for n in names[~unnamed]:
        prefix = ""
        for ch in n:
            if ch.isalpha() or ch == " ":
                prefix += ch
            else:
                break
        prefix = prefix.strip()
        if prefix:
            prefixes.append(prefix)
    pattern_counts = pd.Series(prefixes).value_counts()
    top_patterns = pattern_counts.head(5).index.tolist() if len(pattern_counts) else []

    return ElementNameAnalysis(
        total_traces=total,
        unnamed_count=unnamed_count,
        unnamed_rate=unnamed_count / total if total else 0.0,
        duplicate_names=sorted(duplicates, key=lambda d: d["n_sessions"], reverse=True),
        naming_patterns=top_patterns,
    )


def compute_sensor_coverage_matrix(
    sensor_columns: dict[str, list[str]],
) -> pd.DataFrame:
    """Build a boolean sensor×machine presence matrix.

    Returns DataFrame with sensors as rows, machines as columns, values True/False.
    """
    all_sensors: set[str] = set()
    for sensors in sensor_columns.values():
        all_sensors.update(sensors)

    machines = sorted(sensor_columns.keys())
    sensors = sorted(all_sensors)
    data = {m: [s in sensor_columns[m] for s in sensors] for m in machines}
    return pd.DataFrame(data, index=sensors)


def rank_traces_for_examples(
    all_issues: list[SensorIssue],
    site_traces: pd.DataFrame,
) -> dict[str, str | None]:
    """Pick best and worst trace_ids for sample plots.

    Returns ``{"good": trace_id_or_None, "bad": trace_id_or_None}``.
    """
    # Count issues per trace
    issue_counts: dict[str, int] = {}
    for iss in all_issues:
        issue_counts[iss.trace_id] = issue_counts.get(iss.trace_id, 0) + 1

    analyzed_ids = set(issue_counts.keys())

    # Good: analyzed trace with fewest issues (prefer 0), longest duration
    good_id: str | None = None
    if analyzed_ids:
        candidates = site_traces[site_traces["trace_id"].isin(analyzed_ids)].copy()
        if not candidates.empty:
            candidates["_issues"] = candidates["trace_id"].map(lambda t: issue_counts.get(t, 0))
            candidates = candidates.sort_values(["_issues", "duration_s"], ascending=[True, False])
            good_id = candidates.iloc[0]["trace_id"]

    # Bad: analyzed trace with most issues
    bad_id: str | None = None
    if issue_counts:
        bad_id = max(issue_counts, key=issue_counts.get)

    # Don't use the same trace for both
    if good_id == bad_id:
        bad_id = None

    return {"good": good_id, "bad": bad_id}
