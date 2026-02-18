"""Generate per-session data completeness report.

For each of 2,838 sessions:
- Lists all sensors present vs expected for that technique
- Computes % non-NaN, non-sentinel for each sensor
- Flags: complete (>95%), partial (50-95%), poor (<50%), missing

Also lists merged-session constituent files.

Generates: reports/14_data_completeness.md + figures + cached parquet.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

from macchine.harmonize.calibration import clean_sentinels_df, get_sentinel_values
from macchine.analysis.perpile_features import EXPECTED_SENSORS

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

OUTPUT_DIR = Path("output")
FIG_DIR = Path("reports/figures/completeness")
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = Path("reports/14_data_completeness.md")
CACHE_PATH = OUTPUT_DIR / "metadata" / "session_completeness.parquet"

sns.set_theme(style="whitegrid", font_scale=1.0)


def load_merged_index() -> pd.DataFrame:
    path = OUTPUT_DIR / "metadata" / "merged_trace_index.parquet"
    df = pd.read_parquet(path)
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df = df.dropna(subset=["start_time"])
    df["machine_slug"] = df["machine_slug"].replace("", "unidentified")
    return df


def get_trace_path(row) -> Path:
    site = row["site_id"]
    slug = row["machine_slug"] if row["machine_slug"] != "unidentified" else "unknown"
    trace_id = row["trace_id"]
    return OUTPUT_DIR / "traces" / str(site) / slug / f"{trace_id}.parquet"


def assess_sensor_quality(series: pd.Series, sentinels: list[float]) -> tuple[float, str]:
    """Compute quality fraction and flag for a sensor series.

    Returns (fraction_good, flag) where flag is one of:
    complete (>95%), partial (50-95%), poor (<50%).
    """
    if series is None or len(series) == 0:
        return 0.0, "missing"

    total = len(series)
    # Count NaN and sentinel values
    is_nan = series.isna()
    is_sentinel = series.isin(sentinels) if not series.isna().all() else pd.Series(False, index=series.index)
    bad = is_nan | is_sentinel
    good_frac = 1.0 - bad.sum() / total

    if good_frac > 0.95:
        return good_frac, "complete"
    elif good_frac > 0.50:
        return good_frac, "partial"
    else:
        return good_frac, "poor"


def compute_session_completeness(df_index: pd.DataFrame) -> pd.DataFrame:
    """Compute per-session sensor quality for all sessions."""
    sentinels = get_sentinel_values()
    records = []

    for _, row in tqdm(df_index.iterrows(), total=len(df_index), desc="Assessing completeness"):
        trace_path = get_trace_path(row)
        technique = row["technique"]
        expected = EXPECTED_SENSORS.get(technique, [])

        if not trace_path.exists():
            # Mark all expected sensors as missing
            for sensor in expected:
                records.append({
                    "trace_id": row["trace_id"],
                    "site_id": row["site_id"],
                    "machine_slug": row["machine_slug"],
                    "technique": technique,
                    "element_name": row["element_name"],
                    "sensor": sensor,
                    "quality_frac": 0.0,
                    "quality_flag": "missing",
                    "n_samples": 0,
                })
            continue

        try:
            tdf = pd.read_parquet(trace_path)
        except Exception:
            for sensor in expected:
                records.append({
                    "trace_id": row["trace_id"],
                    "site_id": row["site_id"],
                    "machine_slug": row["machine_slug"],
                    "technique": technique,
                    "element_name": row["element_name"],
                    "sensor": sensor,
                    "quality_frac": 0.0,
                    "quality_flag": "missing",
                    "n_samples": 0,
                })
            continue

        for sensor in expected:
            if sensor not in tdf.columns:
                records.append({
                    "trace_id": row["trace_id"],
                    "site_id": row["site_id"],
                    "machine_slug": row["machine_slug"],
                    "technique": technique,
                    "element_name": row["element_name"],
                    "sensor": sensor,
                    "quality_frac": 0.0,
                    "quality_flag": "missing",
                    "n_samples": len(tdf),
                })
            else:
                series = pd.to_numeric(tdf[sensor], errors="coerce")
                frac, flag = assess_sensor_quality(series, sentinels)
                records.append({
                    "trace_id": row["trace_id"],
                    "site_id": row["site_id"],
                    "machine_slug": row["machine_slug"],
                    "technique": technique,
                    "element_name": row["element_name"],
                    "sensor": sensor,
                    "quality_frac": frac,
                    "quality_flag": flag,
                    "n_samples": len(tdf),
                })

    return pd.DataFrame(records)


def plot_technique_heatmap(comp_df: pd.DataFrame, technique: str) -> str | None:
    """Plot sessions x sensors heatmap colored by quality flag for one technique."""
    tc = comp_df[comp_df["technique"] == technique]
    if tc.empty:
        return None

    # Pivot: sessions (trace_id) x sensors
    pivot = tc.pivot_table(index="trace_id", columns="sensor",
                           values="quality_frac", aggfunc="first")
    if pivot.empty or pivot.shape[0] < 3:
        return None

    # Sort sessions by overall completeness
    pivot["_mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("_mean", ascending=False)
    pivot = pivot.drop(columns=["_mean"])

    # Limit to max 100 sessions for readability
    if len(pivot) > 100:
        # Show worst 50 and best 50
        pivot = pd.concat([pivot.head(50), pivot.tail(50)])

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.2),
                                     min(20, max(6, len(pivot) * 0.15))))

    cmap = sns.color_palette(["#e74c3c", "#f39c12", "#f1c40f", "#27ae60"], as_cmap=True)
    sns.heatmap(pivot, cmap=cmap, vmin=0, vmax=1, ax=ax,
                cbar_kws={"label": "Fraction good data"},
                yticklabels=False, linewidths=0.1)
    ax.set_title(f"{technique} — Sensor Completeness per Session "
                 f"({len(tc['trace_id'].unique())} sessions)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Expected Sensor")
    ax.set_ylabel(f"Sessions (sorted by completeness)")
    plt.xticks(rotation=45, ha="right", fontsize=9)

    fig.tight_layout()
    fname = f"completeness_{technique}.png"
    fig.savefig(FIG_DIR / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fname


def plot_overall_summary(comp_df: pd.DataFrame) -> str:
    """Plot overall completeness summary across all techniques."""
    # Per-session: fraction of core sensors that are 'complete'
    session_quality = comp_df.groupby(["trace_id", "technique"]).agg(
        n_sensors=("sensor", "count"),
        n_complete=("quality_flag", lambda x: (x == "complete").sum()),
        avg_frac=("quality_frac", "mean"),
    ).reset_index()
    session_quality["pct_complete"] = session_quality["n_complete"] / session_quality["n_sensors"] * 100

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Distribution of avg quality fraction by technique
    techniques = sorted(session_quality["technique"].unique())
    for tech in techniques:
        t = session_quality[session_quality["technique"] == tech]
        axes[0].hist(t["avg_frac"] * 100, bins=20, alpha=0.5, label=tech, edgecolor="white")
    axes[0].set_xlabel("Avg sensor quality (%)")
    axes[0].set_ylabel("Number of sessions")
    axes[0].set_title("Distribution of Session Quality")
    axes[0].legend(fontsize=8)

    # 2. Stacked bar: quality flags per technique
    flag_counts = comp_df.groupby(["technique", "quality_flag"]).size().unstack(fill_value=0)
    flag_order = ["complete", "partial", "poor", "missing"]
    flag_colors = {"complete": "#27ae60", "partial": "#f39c12", "poor": "#e74c3c", "missing": "#7f8c8d"}
    flag_cols = [f for f in flag_order if f in flag_counts.columns]
    flag_counts[flag_cols].plot.bar(stacked=True, ax=axes[1],
                                     color=[flag_colors[f] for f in flag_cols],
                                     edgecolor="white")
    axes[1].set_xlabel("Technique")
    axes[1].set_ylabel("Sensor-session count")
    axes[1].set_title("Quality Flags by Technique")
    axes[1].legend(fontsize=8)
    axes[1].tick_params(axis="x", rotation=0)

    # 3. Per-sensor missing rate
    sensor_missing = comp_df.groupby("sensor")["quality_flag"].apply(
        lambda x: (x == "missing").mean() * 100
    ).sort_values(ascending=True)
    axes[2].barh(sensor_missing.index, sensor_missing.values, color="#3498db", alpha=0.8)
    axes[2].set_xlabel("% sessions where sensor is missing")
    axes[2].set_title("Sensor Availability")

    fig.suptitle("Data Completeness Overview", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fname = "completeness_overview.png"
    fig.savefig(FIG_DIR / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fname


def generate_report(df_index: pd.DataFrame, comp_df: pd.DataFrame, figures: list[str]):
    """Generate the markdown report."""
    # Compute summary statistics
    session_quality = comp_df.groupby("trace_id").agg(
        n_sensors=("sensor", "count"),
        n_complete=("quality_flag", lambda x: (x == "complete").sum()),
        n_partial=("quality_flag", lambda x: (x == "partial").sum()),
        n_poor=("quality_flag", lambda x: (x == "poor").sum()),
        n_missing=("quality_flag", lambda x: (x == "missing").sum()),
        avg_frac=("quality_frac", "mean"),
    ).reset_index()

    fully_complete = (session_quality["n_complete"] == session_quality["n_sensors"]).sum()
    pct_complete = fully_complete / len(session_quality) * 100

    lines = [
        "# Data Completeness Report\n\n",
        "This report inventories the sensor data quality for each of the ",
        f"{len(df_index):,} sessions in the merged dataset. For each session, we check whether ",
        "the core expected sensors for its technique are present and contain valid data.\n\n",
        "---\n\n",
        "## Summary\n\n",
        f"- **Total sessions**: {len(df_index):,}\n",
        f"- **Sessions with 100% complete core data**: {fully_complete:,} ({pct_complete:.1f}%)\n",
        f"- **Sessions with issues**: {len(session_quality) - fully_complete:,}\n\n",
    ]

    # Per-technique summary
    lines.append("## Per-Technique Completeness\n\n")
    lines.append("| Technique | Sessions | Sensors Checked | Complete | Partial | Poor | Missing | Avg Quality |\n")
    lines.append("|-----------|----------|-----------------|----------|---------|------|---------|-------------|\n")
    for tech in sorted(comp_df["technique"].unique()):
        tc = comp_df[comp_df["technique"] == tech]
        n_sessions = tc["trace_id"].nunique()
        n_checks = len(tc)
        n_complete = (tc["quality_flag"] == "complete").sum()
        n_partial = (tc["quality_flag"] == "partial").sum()
        n_poor = (tc["quality_flag"] == "poor").sum()
        n_missing = (tc["quality_flag"] == "missing").sum()
        avg_q = tc["quality_frac"].mean() * 100
        lines.append(
            f"| {tech} | {n_sessions} | {n_checks} | {n_complete} | "
            f"{n_partial} | {n_poor} | {n_missing} | {avg_q:.1f}% |\n"
        )

    lines.append("\n---\n\n")

    # Figures
    lines.append("## Completeness Figures\n\n")
    for fname in figures:
        if fname:
            lines.append(f"![Completeness](figures/completeness/{fname})\n\n")

    lines.append("---\n\n")

    # Sessions with issues table (worst 50)
    lines.append("## Sessions with Data Quality Issues (worst 50)\n\n")
    worst = session_quality.sort_values("avg_frac").head(50)
    lines.append("| Session ID | Technique | Complete | Partial | Poor | Missing | Avg Quality |\n")
    lines.append("|-----------|-----------|----------|---------|------|---------|-------------|\n")
    for _, row in worst.iterrows():
        # Look up technique
        tech = comp_df[comp_df["trace_id"] == row["trace_id"]]["technique"].iloc[0]
        lines.append(
            f"| {row['trace_id'][:50]}... | {tech} | "
            f"{int(row['n_complete'])} | {int(row['n_partial'])} | "
            f"{int(row['n_poor'])} | {int(row['n_missing'])} | "
            f"{row['avg_frac']*100:.1f}% |\n"
        )

    lines.append("\n---\n\n")

    # Merged session constituent files (Deliverable 5)
    lines.append("## Merged Session Constituent Files\n\n")
    multi = df_index[df_index["n_traces"] > 1].copy()
    n_multi = len(multi)
    if n_multi > 0:
        median_traces = multi["n_traces"].median()
        max_traces = multi["n_traces"].max()
        lines.append(
            f"**{n_multi}** sessions were merged from multiple trace files "
            f"(median: {median_traces:.0f} traces, max: {max_traces} traces).\n\n"
        )
        lines.append("| Session ID | Element | Site | Technique | N Traces | Constituent Trace IDs |\n")
        lines.append("|-----------|---------|------|-----------|----------|----------------------|\n")
        for _, row in multi.sort_values("n_traces", ascending=False).iterrows():
            trace_ids_display = row["trace_ids"]
            if len(trace_ids_display) > 120:
                trace_ids_display = trace_ids_display[:120] + "..."
            lines.append(
                f"| {row['trace_id'][:40]}... | {row['element_name']} | "
                f"{row['site_id']} | {row['technique']} | "
                f"{row['n_traces']} | {trace_ids_display} |\n"
            )
    else:
        lines.append("No sessions with multiple constituent traces.\n")

    lines.append("\n---\n\n")
    lines.append("## Expected Sensors per Technique\n\n")
    for tech, sensors in sorted(EXPECTED_SENSORS.items()):
        lines.append(f"**{tech}**: {', '.join(sensors)}\n\n")

    REPORT_PATH.write_text("".join(lines))
    print(f"Report written to {REPORT_PATH}")


def main():
    print("Loading merged trace index...")
    df_index = load_merged_index()
    print(f"  {len(df_index):,} sessions")

    # Check for cached completeness data
    if CACHE_PATH.exists():
        print(f"Loading cached completeness from {CACHE_PATH}...")
        comp_df = pd.read_parquet(CACHE_PATH)
        print(f"  {len(comp_df):,} sensor-session records loaded from cache")
    else:
        print("Computing per-session sensor completeness (this may take a while)...")
        comp_df = compute_session_completeness(df_index)
        comp_df.to_parquet(CACHE_PATH, index=False)
        print(f"  Cached to {CACHE_PATH}")

    print(f"  {len(comp_df):,} sensor-session records")

    # Generate figures
    figures = []
    print("\nGenerating overview figure...")
    fname = plot_overall_summary(comp_df)
    figures.append(fname)
    print(f"  {fname}")

    print("Generating per-technique heatmaps...")
    for tech in sorted(comp_df["technique"].unique()):
        fname = plot_technique_heatmap(comp_df, tech)
        if fname:
            figures.append(fname)
            print(f"  {fname}")

    print("\nGenerating report...")
    generate_report(df_index, comp_df, figures)
    print("Done!")


if __name__ == "__main__":
    main()
