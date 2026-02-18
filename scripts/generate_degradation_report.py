"""Generate sensor-level degradation analysis report.

Runs sensor degradation tracking for all controlled groups (machine + technique + site)
and generates per-machine timeseries plots and a summary report.

Output: reports/06_degradation.md + reports/figures/degradation/*.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

from macchine.storage.catalog import get_merged_trace_index
from macchine.analysis.sensor_degradation import (
    extract_sensor_features,
    track_degradation,
    track_cut_health,
)
from macchine.harmonize.calibration import (
    is_calibrated,
    get_display_label,
    get_unit,
)
from macchine.analysis.plot_utils import plot_with_gaps, add_site_markers

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

OUTPUT_DIR = Path("output")
FIG_DIR = Path("reports/figures/degradation")
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = Path("reports/06_degradation.md")

sns.set_theme(style="whitegrid", font_scale=0.9)

# Priority machines for analysis (most data, best controls)
PRIORITY_MACHINES = [
    ("gb50_601", "GRAB", "1514"),       # 282 GRAB, single site
    ("bg33v_5610", "KELLY", "1508"),     # KELLY+SCM at 1508
    ("bg33v_5610", "GRAB", "LignanoSabbiadoro"),  # GRAB at Lignano
    ("bg45v_3923", "CUT", "Paris L18.3 OA20"),  # CUT with all health sensors calibrated
    ("mc86_621", "CUT", None),           # CUT across sites
    ("bg42v_5925", "KELLY", None),       # KELLY across sites
    ("bg28h_6061", "GRAB", None),        # GRAB across sites
]


def plot_sensor_degradation(
    features_df: pd.DataFrame,
    machine: str,
    technique: str,
    site: str | None,
) -> list[str]:
    """Generate timeseries plots for sensor degradation metrics."""
    if features_df.empty:
        return []

    features_df = features_df.sort_values("start_time")

    # Find metric columns (sensor__stat format)
    metric_cols = [c for c in features_df.columns if "__median" in c]
    if not metric_cols:
        return []

    # Limit to 8 panels max
    metric_cols = metric_cols[:8]
    n_panels = len(metric_cols)

    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3 * n_panels),
                              squeeze=False, sharex=True)

    for i, col in enumerate(metric_cols):
        ax = axes[i, 0]
        sensor_name = col.split("__")[0]
        stat_name = col.split("__")[1]

        vals = features_df[["start_time", col]].dropna()
        if vals.empty:
            ax.set_visible(False)
            continue

        # Scatter points
        ax.scatter(vals["start_time"], vals[col], s=12, alpha=0.4, color="#3498db")

        # Rolling mean (30-session window)
        if len(vals) > 30:
            rolling = vals[col].rolling(30, min_periods=15).mean()
            ax.plot(vals["start_time"], rolling, color="red", linewidth=1.5,
                    alpha=0.8, label="30-session rolling mean")

        # Site markers if multi-site
        if "site_id" in features_df.columns:
            add_site_markers(ax, features_df, site_col="site_id", time_col="start_time")

        display = get_display_label(sensor_name)
        unit = get_unit(sensor_name, machine) if is_calibrated(sensor_name, machine) else "arb."
        ax.set_ylabel(f"{display}\n{stat_name} [{unit}]", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[-1, 0].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45, ha="right")

    site_label = f" @ {site}" if site else ""
    fig.suptitle(
        f"Sensor Degradation: {machine} ({technique}{site_label}) — "
        f"{len(features_df)} sessions",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    safe_site = str(site or "all").replace("/", "_").replace(" ", "_")
    fname = f"{machine}_{technique}_{safe_site}_degradation.png"
    fig.savefig(FIG_DIR / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return [fname]


def generate_report(
    all_results: list[dict],
    cut_results: list[dict],
    all_figures: dict[str, list[str]],
):
    """Generate the degradation analysis markdown report."""
    lines = [
        "# Sensor-Level Degradation Analysis\n\n",
        "This report tracks machine health indicators extracted from actual sensor time-series data, ",
        "**controlled by machine, technique, and site**. Unlike the previous duration-only analysis, ",
        "this tracks calibrated sensor values (pressure, temperature, torque, oil condition) over time.\n\n",
        "**Methodology**: For each controlled group (machine + technique + site), we extract per-session ",
        "summary statistics from parquet trace files, compute rolling baselines, and test for systematic ",
        "drift using linear regression. Only calibrated sensors are used.\n\n",
        "---\n\n",
    ]

    # Summary table
    lines.append("## Degradation Summary\n\n")
    lines.append("| Machine | Technique | Site | Sessions | Drifting Metrics | Key Finding |\n")
    lines.append("|---------|-----------|------|----------|------------------|-------------|\n")

    for result in all_results:
        machine = result["machine"]
        technique = result.get("technique", "all")
        site = result.get("site", "all")
        n = result["n_sessions"]

        drifting = [name for name, m in result["metrics"].items()
                    if m["direction"] != "STABLE"]
        n_drift = len(drifting)

        if drifting:
            key_metric = drifting[0]
            key_info = result["metrics"][key_metric]
            finding = f"{key_metric.split('__')[0]}: {key_info['direction']}"
        else:
            finding = "All metrics stable"

        lines.append(
            f"| {machine} | {technique} | {site} | {n} | "
            f"{n_drift} | {finding} |\n"
        )

    lines.append("\n---\n\n")

    # Per-machine detailed results
    for result in all_results:
        machine = result["machine"]
        technique = result.get("technique", "all")
        site = result.get("site", "all")
        key = f"{machine}_{technique}_{site}"

        lines.append(f"## {machine} — {technique} @ {site}\n\n")
        lines.append(f"**Sessions**: {result['n_sessions']}")
        if result.get("date_range"):
            lines.append(f" | **Date range**: {result['date_range'][0]} to {result['date_range'][1]}")
        lines.append("\n\n")

        if result["metrics"]:
            lines.append("| Metric | Direction | Slope/session | Median | Q1 Avg | Q4 Avg | p-value |\n")
            lines.append("|--------|-----------|---------------|--------|--------|--------|--------|\n")

            for name, m in sorted(result["metrics"].items()):
                p_str = f"{m['p_value']:.4f}" if m.get("p_value") is not None else "N/A"
                lines.append(
                    f"| {name} | **{m['direction']}** | {m['slope_per_session']:.4f} | "
                    f"{m['median']:.2f} | {m['first_quarter_mean']:.2f} | "
                    f"{m['last_quarter_mean']:.2f} | {p_str} |\n"
                )
        else:
            lines.append("*No sensor metrics available (insufficient calibrated data)*\n\n")

        # Include figures
        figs = all_figures.get(key, [])
        for fname in figs:
            lines.append(f"\n![{key}](figures/degradation/{fname})\n")

        lines.append("\n---\n\n")

    # CUT-specific health section
    if cut_results:
        lines.append("## CUT Machine Health Summary\n\n")
        for cr in cut_results:
            machine = cr["machine"]
            site = cr.get("site", "all")
            lines.append(f"### {machine} @ {site} ({cr['n_sessions']} sessions)\n\n")

            if cr["health"]:
                for indicator, info in cr["health"].items():
                    lines.append(
                        f"- **{indicator}**: {info.get('first_quarter_avg', 'N/A'):.2f} -> "
                        f"{info.get('last_quarter_avg', 'N/A'):.2f} — **{info['direction']}** "
                        f"({info.get('change_pct', 0):.1f}% change, {info['n_sessions']} sessions)\n"
                    )
            else:
                lines.append("*No CUT health indicators available*\n")
            lines.append("\n")

    # Methodology
    lines.append("---\n\n")
    lines.append("## Methodology\n\n")
    lines.append("1. **Controlled groups**: Each analysis runs within (machine, technique, site) to avoid confounders\n")
    lines.append("2. **Calibration-aware**: Only calibrated sensors are used; uncalibrated channels are excluded\n")
    lines.append("3. **Physical validation**: Sensor values are validated against physical ranges before analysis\n")
    lines.append("4. **Active phase extraction**: Pressure/torque stats use only samples where depth is changing\n")
    lines.append("5. **Drift detection**: Linear regression on session-ordered values; p < 0.05 and relative slope > 0.1%/session = drift\n")
    lines.append("6. **CUT health**: Temperature, pressure asymmetry, oil pressure, and leakage tracked as specific wear indicators\n")

    REPORT_PATH.write_text("".join(lines))
    print(f"Report written to {REPORT_PATH}")


def main():
    print("Loading merged trace index...")
    df_index = get_merged_trace_index(OUTPUT_DIR)
    df_index = df_index.dropna(subset=["start_time"])
    print(f"  {len(df_index):,} sessions, {df_index['machine_slug'].nunique()} machines")

    all_results = []
    cut_results = []
    all_figures = {}

    for machine, technique, site in PRIORITY_MACHINES:
        label = f"{machine}/{technique}@{site or 'all'}"

        # Check if this combination exists in the index
        mask = df_index["machine_slug"] == machine
        if technique:
            mask = mask & (df_index["technique"] == technique)
        if site:
            mask = mask & (df_index["site_id"] == site)
        n_available = mask.sum()

        if n_available < 10:
            print(f"\n--- {label}: only {n_available} sessions, skipping ---")
            continue

        print(f"\n--- {label} ({n_available} sessions) ---")

        # Extract sensor features
        print("  Extracting sensor features...")
        features_df = extract_sensor_features(
            OUTPUT_DIR, machine=machine, technique=technique, site=site
        )

        if features_df.empty:
            print("  No features extracted, skipping")
            continue

        print(f"  Extracted features from {len(features_df)} sessions")

        # Track degradation (pass pre-computed features to avoid re-extraction)
        print("  Tracking degradation...")
        result = track_degradation(
            OUTPUT_DIR, machine=machine, technique=technique, site=site,
            features_df=features_df,
        )
        all_results.append(result)

        # Generate plots
        print("  Generating plots...")
        key = f"{machine}_{technique}_{site or 'all'}"
        fnames = plot_sensor_degradation(features_df, machine, technique, site)
        if fnames:
            all_figures[key] = fnames
            for f in fnames:
                print(f"    {f}")

        # CUT-specific health analysis (pass pre-computed features)
        if technique == "CUT":
            print("  Running CUT health analysis...")
            cut_health = track_cut_health(OUTPUT_DIR, machine, site=site,
                                          features_df=features_df)
            cut_results.append(cut_health)

    print("\nGenerating report...")
    generate_report(all_results, cut_results, all_figures)
    print(f"\nDone! Figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
