"""Generate per-site data quality reports.

For each site, produces a markdown report with embedded figures covering:
coverage, calibration, plausibility, zeroing, element naming, and sample plots.

Usage:
    python scripts/generate_site_reports.py                     # all sites
    python scripts/generate_site_reports.py --site 1501         # single site
    python scripts/generate_site_reports.py --output-dir /tmp   # custom output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from macchine.harmonize.calibration import (
    clean_sentinels_df,
    get_axis_label,
    get_display_label,
    get_sentinel_values,
    get_unit,
    is_calibrated,
)
from macchine.quality.checks import (
    SensorIssue,
    analyze_element_names,
    compute_calibration_summary,
    compute_sensor_coverage_matrix,
    compute_site_overview,
    detect_sensor_issues,
    rank_traces_for_examples,
)
from macchine.quality.plots import (
    plot_calibration_bar,
    plot_depth_profile,
    plot_duration_histogram,
    plot_sensor_coverage_heatmap,
    plot_trace_example,
)
from macchine.storage.catalog import get_merged_trace_index

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
DEFAULT_REPORT_DIR = OUTPUT_DIR / "reports"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_slug(slug: str) -> str:
    """Map empty or 'unidentified' machine slugs to the 'unknown' directory name."""
    s = str(slug).strip()
    if not s or s == "unidentified":
        return "unknown"
    return s


def _discover_sensor_columns(
    traces_dir: Path, site_id: str, site_traces: pd.DataFrame, max_per_machine: int = 5,
) -> dict[str, list[str]]:
    """Load a few trace parquets per machine and collect numeric column names."""
    sensor_cols: dict[str, set[str]] = {}
    for machine_slug, grp in site_traces.groupby("machine_slug"):
        cols: set[str] = set()
        machine_dir = traces_dir / str(site_id) / _resolve_slug(machine_slug)
        if not machine_dir.exists():
            continue
        # Use trace_ids from the index — they point to actual parquet files
        loaded = 0
        for _, row in grp.iterrows():
            trace_ids = row["trace_ids"].split("|") if isinstance(row.get("trace_ids"), str) else [row["trace_id"]]
            for tid in trace_ids:
                pf = machine_dir / f"{tid}.parquet"
                if not pf.exists():
                    continue
                try:
                    df = pd.read_parquet(pf)
                    for c in df.select_dtypes(include=[np.number]).columns:
                        if c != "timestamp":
                            cols.add(c)
                    loaded += 1
                except Exception:
                    continue
                if loaded >= max_per_machine:
                    break
            if loaded >= max_per_machine:
                break
        sensor_cols[str(machine_slug)] = sorted(cols)
    return {k: v for k, v in sensor_cols.items() if v}


def _sample_traces(site_traces: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Pick a diverse sample of traces for detailed analysis.

    Strategy: at least 1 per machine, 1 per technique, prefer named elements
    and longer traces.
    """
    sampled_ids: set[str] = set()
    pool = site_traces.copy()
    pool["_has_name"] = pool["element_name"].fillna("").str.strip().ne("")

    # 1 per machine
    for _, grp in pool.groupby("machine_slug"):
        best = grp.sort_values(["_has_name", "duration_s"], ascending=[False, False])
        sampled_ids.add(best.iloc[0]["trace_id"])

    # 1 per technique
    for _, grp in pool.groupby("technique"):
        best = grp.sort_values(["_has_name", "duration_s"], ascending=[False, False])
        sampled_ids.add(best.iloc[0]["trace_id"])

    # Fill remaining budget with longest named traces
    remaining = pool[~pool["trace_id"].isin(sampled_ids)]
    remaining = remaining.sort_values(["_has_name", "duration_s"], ascending=[False, False])
    for _, row in remaining.iterrows():
        if len(sampled_ids) >= n:
            break
        sampled_ids.add(row["trace_id"])

    return pool[pool["trace_id"].isin(sampled_ids)]


def _load_trace(traces_dir: Path, site_id: str, row: pd.Series) -> pd.DataFrame | None:
    """Load a single trace parquet, cleaning sentinels."""
    machine_slug = _resolve_slug(row["machine_slug"])
    # For merged sessions, use the first constituent trace_id
    trace_ids = row["trace_ids"].split("|") if isinstance(row.get("trace_ids"), str) else [row["trace_id"]]
    for tid in trace_ids:
        pf = traces_dir / str(site_id) / machine_slug / f"{tid}.parquet"
        if pf.exists():
            try:
                df = pd.read_parquet(pf)
                return clean_sentinels_df(df)
            except Exception:
                continue
    return None


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def _write_report(
    report_dir: Path,
    overview,
    cal_summaries,
    coverage_df,
    all_issues: list[SensorIssue],
    name_analysis,
    site_traces: pd.DataFrame,
    examples: dict,
) -> None:
    """Write the markdown report file."""
    fig_rel = "figures"
    lines: list[str] = []

    def h1(text: str):
        lines.append(f"# {text}\n")

    def h2(text: str):
        lines.append(f"## {text}\n")

    def h3(text: str):
        lines.append(f"### {text}\n")

    def img(filename: str, alt: str = ""):
        lines.append(f"![{alt}]({fig_rel}/{filename})\n")

    def table(headers: list[str], rows: list[list[str]]):
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for r in rows:
            lines.append("| " + " | ".join(str(c) for c in r) + " |")
        lines.append("")

    # ---- Title ----
    h1(f"Data Quality Report — Site {overview.site_id}")
    lines.append(f"*Generated from {overview.n_traces} merged traces*\n")

    # ---- 1. Overview ----
    h2("1. Site Overview")
    table(
        ["Metric", "Value"],
        [
            ["Machines", f"{overview.n_machines} ({', '.join(overview.machine_slugs)})"],
            ["Techniques", ", ".join(overview.techniques)],
            ["Date range", f"{overview.date_min} to {overview.date_max}"],
            ["Traces", str(overview.n_traces)],
            ["Operator recorded", f"{overview.operator_rate:.0%}"],
            ["Element named", f"{overview.naming_rate:.0%}"],
            ["Formats", ", ".join(f"{k}: {v}" for k, v in overview.formats.items())],
        ],
    )

    # ---- 2. Calibration ----
    h2("2. Sensor Calibration Status")
    img("calibration.png", "Calibration status bar chart")
    if cal_summaries:
        table(
            ["Machine", "Calibrated", "Uncalibrated", "Uncalibrated sensors"],
            [
                [
                    s.machine_slug,
                    str(s.n_calibrated),
                    str(s.n_uncalibrated),
                    ", ".join(s.uncalibrated_sensors[:5]) + ("..." if len(s.uncalibrated_sensors) > 5 else ""),
                ]
                for s in cal_summaries
            ],
        )

    # ---- 3. Coverage ----
    h2("3. Sensor Coverage")
    img("coverage.png", "Sensor coverage heatmap")

    # ---- 4. Plausibility ----
    h2("4. Data Plausibility")
    issue_types = {}
    for iss in all_issues:
        issue_types.setdefault(iss.issue_type, []).append(iss)

    if all_issues:
        table(
            ["Issue type", "Count", "Severity breakdown"],
            [
                [
                    itype,
                    str(len(items)),
                    f"{sum(1 for i in items if i.severity == 'error')} errors, "
                    f"{sum(1 for i in items if i.severity == 'warning')} warnings",
                ]
                for itype, items in sorted(issue_types.items())
            ],
        )

        for itype, items in sorted(issue_types.items()):
            h3(f"4.{list(sorted(issue_types.keys())).index(itype)+1}. {itype}")
            detail_rows = []
            for iss in items[:20]:
                detail_rows.append([iss.sensor, iss.machine_slug, iss.trace_id[:20], iss.severity, iss.detail])
            table(["Sensor", "Machine", "Trace", "Severity", "Detail"], detail_rows)
    else:
        lines.append("No plausibility issues detected in sampled traces.\n")

    # ---- 5. Zeroing ----
    h2("5. Zeroing Analysis")
    zeroing = [i for i in all_issues if i.issue_type == "ZEROING"]
    if zeroing:
        lines.append(f"{len(zeroing)} traces had sensors with start-of-trace offset issues:\n")
        table(
            ["Sensor", "Machine", "Trace", "Detail"],
            [[z.sensor, z.machine_slug, z.trace_id[:20], z.detail] for z in zeroing[:15]],
        )
    else:
        lines.append("All sampled traces had acceptable zeroing.\n")

    # ---- 6. Element names ----
    h2("6. Element Name Analysis")
    na = name_analysis
    table(
        ["Metric", "Value"],
        [
            ["Total traces", str(na.total_traces)],
            ["Unnamed", f"{na.unnamed_count} ({na.unnamed_rate:.0%})"],
            ["Naming patterns", ", ".join(na.naming_patterns) if na.naming_patterns else "—"],
        ],
    )
    if na.duplicate_names:
        h3("6.1. Multi-session / reused names")
        dup_rows = []
        for d in na.duplicate_names[:20]:
            dup_rows.append([d["name"], str(d["n_sessions"]), d["technique"], d["continuity"]])
        table(["Name", "Sessions", "Technique", "Continuity"], dup_rows)

    # ---- 7. Duration ----
    h2("7. Duration Distribution")
    img("duration.png", "Duration histogram")

    # ---- 8-9. Sample traces ----
    h2("8. Sample Traces — Good Example")
    if examples.get("good"):
        img("example_good.png", "Good trace example")
    else:
        lines.append("No good example available.\n")

    h2("9. Sample Traces — Bad Example")
    if examples.get("bad"):
        img("example_bad.png", "Bad trace example with issues")
    else:
        lines.append("No bad example available (no issues detected).\n")

    # ---- 10. Depth profile ----
    h2("10. Depth Profile")
    img("depth_profile.png", "Depth profile")

    # Write
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main per-site pipeline
# ---------------------------------------------------------------------------

def generate_site_report(
    site_id: str,
    merged_index: pd.DataFrame,
    report_base: Path,
) -> None:
    """Generate a complete quality report for one site."""
    traces_dir = OUTPUT_DIR / "traces"
    site_traces = merged_index[merged_index["site_id"] == site_id].copy()
    if site_traces.empty:
        print(f"  [{site_id}] No traces found, skipping.")
        return

    report_dir = report_base / site_id
    fig_dir = report_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Overview (index only, no file loading)
    overview = compute_site_overview(site_traces, site_id)
    print(f"  [{site_id}] {overview.n_traces} traces, {overview.n_machines} machines, "
          f"techniques: {', '.join(overview.techniques)}")

    # 2. Discover sensor columns
    sensor_columns = _discover_sensor_columns(traces_dir, site_id, site_traces)
    if not sensor_columns:
        print(f"  [{site_id}] No sensor data found, skipping.")
        return

    # 3. Calibration
    cal_summaries = compute_calibration_summary(sensor_columns)
    plot_calibration_bar(cal_summaries, site_id, fig_dir / "calibration.png")

    # 4. Coverage
    coverage_df = compute_sensor_coverage_matrix(sensor_columns)
    plot_sensor_coverage_heatmap(coverage_df, site_id, fig_dir / "coverage.png")

    # 5. Sample traces for detailed analysis
    sampled = _sample_traces(site_traces)
    all_issues: list[SensorIssue] = []
    loaded_traces: dict[str, tuple[pd.DataFrame, pd.Series]] = {}  # trace_id -> (df, row)

    for _, row in sampled.iterrows():
        tid = row["trace_id"]
        df = _load_trace(traces_dir, site_id, row)
        if df is None:
            continue
        loaded_traces[tid] = (df, row)
        issues = detect_sensor_issues(df, str(row["machine_slug"]), tid)
        all_issues.extend(issues)

    n_issues = len(all_issues)
    print(f"  [{site_id}] Analyzed {len(loaded_traces)} traces, found {n_issues} issues")

    # 6. Element name analysis
    name_analysis = analyze_element_names(site_traces)

    # 7. Pick example traces
    examples = rank_traces_for_examples(all_issues, site_traces)

    # 8. Generate example plots
    # Good example: pick a sensor that is present and interesting
    for label in ("good", "bad"):
        tid = examples.get(label)
        if tid is None or tid not in loaded_traces:
            continue
        df, row = loaded_traces[tid]
        machine_slug = str(row["machine_slug"])
        # Pick a sensor: for good pick depth or torque; for bad pick the one with most issues
        if label == "bad":
            sensor_issues = [i for i in all_issues if i.trace_id == tid]
            if sensor_issues:
                sensor = sensor_issues[0].sensor
            else:
                continue
        else:
            sensor = None
            for candidate in ("depth", "torque", "pump_pressure_1", "rope_force"):
                if candidate in df.columns:
                    sensor = candidate
                    break
            if sensor is None:
                # Fall back to first numeric column
                numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "timestamp"]
                sensor = numeric_cols[0] if numeric_cols else None
            if sensor is None:
                continue

        trace_issues = [i for i in all_issues if i.trace_id == tid]
        plot_trace_example(df, sensor, machine_slug, tid, trace_issues, fig_dir / f"example_{label}.png")

    # 9. Depth profile — pick a named element if possible
    depth_done = False
    for tid, (df, row) in loaded_traces.items():
        name = str(row.get("element_name", ""))
        if not name or name == "xxxxx":
            continue
        machine_slug = str(row["machine_slug"])
        # Pick sensors from different categories
        profile_sensors = []
        for candidates in [
            ["torque", "torque_knm"],
            ["pump_pressure_1", "pump_pressure", "kdk_pressure"],
            ["rope_force", "rope_force_main_winch"],
            ["rotation_speed", "rotation_speed_cutter_left"],
        ]:
            for c in candidates:
                if c in df.columns:
                    profile_sensors.append(c)
                    break
        if profile_sensors:
            plot_depth_profile(df, machine_slug, name, profile_sensors, fig_dir / "depth_profile.png")
            depth_done = True
            break

    if not depth_done and loaded_traces:
        # Fall back to any trace with depth
        for tid, (df, row) in loaded_traces.items():
            if "depth" not in df.columns:
                continue
            machine_slug = str(row["machine_slug"])
            name = str(row.get("element_name", tid[:20]))
            numeric = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ("timestamp", "depth")]
            if numeric:
                plot_depth_profile(df, machine_slug, name, numeric[:4], fig_dir / "depth_profile.png")
                break

    # 10. Duration histogram
    plot_duration_histogram(site_traces, site_id, fig_dir / "duration.png")

    # 11. Write markdown report
    _write_report(report_dir, overview, cal_summaries, coverage_df, all_issues, name_analysis, site_traces, examples)
    print(f"  [{site_id}] Report written to {report_dir / 'report.md'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate per-site data quality reports.")
    parser.add_argument("--site", type=str, default=None, help="Single site ID to process")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for reports")
    args = parser.parse_args()

    report_base = Path(args.output_dir) if args.output_dir else DEFAULT_REPORT_DIR

    print("Loading merged trace index...")
    merged = get_merged_trace_index(OUTPUT_DIR)
    print(f"Loaded {len(merged)} merged traces across {merged['site_id'].nunique()} sites.\n")

    if args.site:
        site_ids = [args.site]
    else:
        site_ids = sorted(merged["site_id"].unique().tolist())

    for sid in site_ids:
        generate_site_report(str(sid), merged, report_base)

    print(f"\nDone. Reports at {report_base}/")


if __name__ == "__main__":
    main()
