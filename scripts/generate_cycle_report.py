"""Generate GRAB cycle-level analysis report.

Runs cycle detection and feature extraction on priority GRAB machines,
generates per-machine timeseries plots, and writes a summary report.

Output: reports/19_cycle_analysis.md + reports/figures/cycle/*.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import warnings

from macchine.analysis.cycle_analysis import track_cycle_degradation
from macchine.analysis.plot_utils import plot_with_gaps, add_site_markers

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

OUTPUT_DIR = Path("output")
FIG_DIR = Path("reports/figures/cycle")
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = Path("reports/19_cycle_analysis.md")

# Priority GRAB machines
PRIORITY_MACHINES = [
    ("gb50_601", "1514"),           # 282 GRAB, single site, jaw pressure available
    ("bg33v_5610", "LignanoSabbiadoro"),  # GRAB at Lignano
    ("bg28h_6061", None),           # GRAB across sites
]

# Cycle features to plot
CYCLE_METRICS = [
    ("cycle_depth", "Cycle Depth", "m"),
    ("cycle_duration_s", "Cycle Duration", "s"),
    ("descent_speed", "Descent Speed", "m/s"),
    ("ascent_speed", "Ascent Speed", "m/s"),
    ("soil_weight_proxy", "Soil Weight Proxy", "force units"),
    ("jaw_pressure_at_bottom", "Jaw Pressure at Bottom", "bar"),
    ("Druck Pumpe 1_mean", "Pump 1 Pressure (mean)", "bar"),
]


def plot_cycle_metrics(
    cycle_df: pd.DataFrame,
    machine: str,
    site: str | None,
) -> list[str]:
    """Generate timeseries plots for per-cycle metrics."""
    if cycle_df.empty:
        return []

    # Find which metrics are available
    available = [
        (col, label, unit) for col, label, unit in CYCLE_METRICS
        if col in cycle_df.columns and cycle_df[col].notna().sum() > 10
    ]
    if not available:
        return []

    n_panels = len(available)
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3 * n_panels),
                              squeeze=False, sharex=True)

    # Assign session-level timestamp to each cycle
    cycle_df = cycle_df.sort_values("start_time")

    for i, (col, label, unit) in enumerate(available):
        ax = axes[i, 0]
        vals = cycle_df[["start_time", col, "depth_at_bottom"]].dropna(subset=[col])
        if vals.empty:
            ax.set_visible(False)
            continue

        # Color by depth
        depths = vals["depth_at_bottom"].values
        depth_norm = (depths - depths.min()) / (depths.max() - depths.min() + 1e-6)

        scatter = ax.scatter(
            vals["start_time"], vals[col],
            c=depth_norm, cmap="viridis_r", s=10, alpha=0.5,
        )

        # Rolling median
        if len(vals) > 50:
            rolling = vals[col].rolling(50, min_periods=25).median()
            ax.plot(vals["start_time"], rolling, color="red", linewidth=1.5,
                    alpha=0.8, label="50-cycle rolling median")

        # Site markers
        if "site_id" in cycle_df.columns and cycle_df["site_id"].nunique() > 1:
            add_site_markers(ax, cycle_df, site_col="site_id", time_col="start_time")

        ax.set_ylabel(f"{label}\n[{unit}]", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    # Colorbar for depth
    if n_panels > 0:
        cbar = fig.colorbar(scatter, ax=axes[-1, 0], orientation="horizontal",
                            pad=0.15, aspect=40, shrink=0.5)
        cbar.set_label("Relative depth (dark = deeper)", fontsize=8)

    axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[-1, 0].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45, ha="right")

    site_label = f" @ {site}" if site else ""
    fig.suptitle(
        f"GRAB Cycle Analysis: {machine}{site_label} — "
        f"{len(cycle_df)} cycles from {cycle_df['trace_id'].nunique()} sessions",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    safe_site = str(site or "all").replace("/", "_").replace(" ", "_")
    fname = f"{machine}_{safe_site}_cycles.png"
    fig.savefig(FIG_DIR / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return [fname]


def plot_depth_vs_feature(
    cycle_df: pd.DataFrame,
    machine: str,
    site: str | None,
) -> list[str]:
    """Plot cycle features vs depth to detect depth-dependent degradation."""
    if cycle_df.empty or len(cycle_df) < 20:
        return []

    # Features to plot against depth
    features = [
        ("cycle_duration_s", "Cycle Duration [s]"),
        ("soil_weight_proxy", "Soil Weight Proxy"),
        ("jaw_pressure_at_bottom", "Jaw Pressure at Bottom [bar]"),
    ]
    available = [(c, l) for c, l in features
                 if c in cycle_df.columns and cycle_df[c].notna().sum() > 10]

    if not available:
        return []

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    # Color by session order (time)
    times = cycle_df["start_time"]
    time_ord = (times - times.min()).dt.total_seconds()
    time_norm = time_ord / (time_ord.max() + 1e-6)

    for i, (col, label) in enumerate(available):
        ax = axes[0, i]
        vals = cycle_df[["depth_at_bottom", col]].dropna()
        if vals.empty:
            continue

        # Match time_norm to valid rows
        valid_idx = vals.index
        t_norm = time_norm.loc[valid_idx]

        ax.scatter(vals["depth_at_bottom"], vals[col],
                   c=t_norm, cmap="coolwarm", s=8, alpha=0.4)
        ax.set_xlabel("Depth at bottom [m]", fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.3)

    site_label = f" @ {site}" if site else ""
    fig.suptitle(f"Depth vs Features: {machine}{site_label}", fontsize=11, fontweight="bold")
    fig.tight_layout()

    safe_site = str(site or "all").replace("/", "_").replace(" ", "_")
    fname = f"{machine}_{safe_site}_depth_features.png"
    fig.savefig(FIG_DIR / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return [fname]


def generate_report(all_results: list[dict], all_figures: dict[str, list[str]]):
    """Generate the cycle analysis markdown report."""
    lines = [
        "# GRAB Cycle-Level Analysis\n\n",
        "This report extracts individual grab cycles from the depth sawtooth pattern ",
        "and tracks per-cycle features over time. Each cycle represents one descent-ascent ",
        "pair where the grab descends into the trench, closes its jaws to capture soil, ",
        "then ascends to dump the material.\n\n",
        "**Key degradation signals**:\n",
        "- Declining soil weight proxy at similar depth → grab jaw wear\n",
        "- Slowing cycle times → winch degradation\n",
        "- Rising pump pressure at similar load → hydraulic wear\n",
        "- Declining jaw pressure at bottom → jaw seal degradation\n\n",
        "---\n\n",
    ]

    # Summary table
    lines.append("## Summary\n\n")
    lines.append("| Machine | Site | Sessions | Cycles | Cycles/Session | "
                 "Med. Depth | Med. Duration | Med. Jaw P. |\n")
    lines.append("|---------|------|----------|--------|----------------|"
                 "-----------|---------------|-------------|\n")

    for result in all_results:
        m = result["machine"]
        s = result.get("site", "all")
        n_sess = result["n_sessions"]
        n_cyc = result["n_cycles"]
        cps = f"{n_cyc / n_sess:.1f}" if n_sess > 0 else "0"
        med_d = f"{result['med_depth']:.1f}" if result.get("med_depth") is not None else "N/A"
        med_dur = f"{result['med_duration']:.0f}" if result.get("med_duration") is not None else "N/A"
        med_jaw = f"{result['med_jaw']:.0f}" if result.get("med_jaw") is not None else "N/A"
        lines.append(f"| {m} | {s} | {n_sess} | {n_cyc} | {cps} | {med_d}m | {med_dur}s | {med_jaw} |\n")

    lines.append("\n---\n\n")

    # Per-machine details
    for result in all_results:
        m = result["machine"]
        s = result.get("site", "all")
        key = f"{m}_{s}"
        lines.append(f"## {m} @ {s}\n\n")
        lines.append(f"**Sessions with cycles**: {result['n_sessions']} | "
                     f"**Total cycles**: {result['n_cycles']}")
        if result.get("date_range"):
            lines.append(f" | **Date range**: {result['date_range'][0]} to {result['date_range'][1]}")
        lines.append("\n\n")

        if result.get("stats"):
            lines.append("| Metric | Median | Mean | Std | P5 | P95 |\n")
            lines.append("|--------|--------|------|-----|-----|-----|\n")
            for metric, stats in sorted(result["stats"].items()):
                lines.append(
                    f"| {metric} | {stats['median']:.2f} | {stats['mean']:.2f} | "
                    f"{stats['std']:.2f} | {stats['p5']:.2f} | {stats['p95']:.2f} |\n"
                )

        figs = all_figures.get(key, [])
        for fname in figs:
            lines.append(f"\n![{key}](figures/cycle/{fname})\n")

        lines.append("\n---\n\n")

    # Methodology
    lines.append("## Methodology\n\n")
    lines.append("1. **Cycle detection**: Depth signal smoothed (10-sample uniform filter), "
                 "peaks and troughs found via scipy `find_peaks` with prominence >= 5m\n")
    lines.append("2. **Cycle pairing**: Peak-trough-peak triplets form one complete cycle "
                 "(descent then ascent)\n")
    lines.append("3. **Feature extraction**: Per-cycle force (descent vs ascent), speed, "
                 "pump pressure, jaw closing pressure\n")
    lines.append("4. **Soil weight proxy**: loaded_force (ascent) - unloaded_force (descent); "
                 "declining proxy at similar depth suggests jaw wear\n")
    lines.append("5. **Jaw pressure**: Schließzylinderdruck near cycle bottom; reflects "
                 "soil resistance and jaw closing force\n")

    REPORT_PATH.write_text("".join(lines))
    print(f"Report written to {REPORT_PATH}")


def main():
    all_results = []
    all_figures = {}

    for machine, site in PRIORITY_MACHINES:
        label = f"{machine}@{site or 'all'}"
        print(f"\n--- {label} ---")

        print("  Running cycle detection...")
        cycle_df = track_cycle_degradation(OUTPUT_DIR, machine, site=site)

        if cycle_df.empty:
            print("  No cycles detected, skipping")
            continue

        n_cycles = len(cycle_df)
        n_sessions = cycle_df["trace_id"].nunique()
        print(f"  Detected {n_cycles} cycles across {n_sessions} sessions")

        # Compute summary stats
        stats = {}
        for col in ["cycle_depth", "cycle_duration_s", "descent_speed", "ascent_speed",
                     "soil_weight_proxy", "jaw_pressure_at_bottom", "jaw_pressure_mean",
                     "Druck Pumpe 1_mean"]:
            if col in cycle_df.columns:
                vals = cycle_df[col].dropna()
                if len(vals) > 5:
                    stats[col] = {
                        "median": float(vals.median()),
                        "mean": float(vals.mean()),
                        "std": float(vals.std()),
                        "p5": float(vals.quantile(0.05)),
                        "p95": float(vals.quantile(0.95)),
                    }

        result = {
            "machine": machine,
            "site": site or "all",
            "n_sessions": n_sessions,
            "n_cycles": n_cycles,
            "med_depth": float(cycle_df["depth_at_bottom"].median()) if "depth_at_bottom" in cycle_df.columns else None,
            "med_duration": float(cycle_df["cycle_duration_s"].median()) if "cycle_duration_s" in cycle_df.columns else None,
            "med_jaw": float(cycle_df["jaw_pressure_at_bottom"].median()) if "jaw_pressure_at_bottom" in cycle_df.columns and cycle_df["jaw_pressure_at_bottom"].notna().any() else None,
            "stats": stats,
        }

        if not cycle_df.empty:
            result["date_range"] = (
                str(cycle_df["start_time"].min().date()),
                str(cycle_df["start_time"].max().date()),
            )

        all_results.append(result)

        # Generate plots
        key = f"{machine}_{site or 'all'}"
        print("  Generating plots...")
        fnames = plot_cycle_metrics(cycle_df, machine, site)
        fnames += plot_depth_vs_feature(cycle_df, machine, site)
        if fnames:
            all_figures[key] = fnames
            for f in fnames:
                print(f"    {f}")

    print("\nGenerating report...")
    generate_report(all_results, all_figures)
    print(f"Done! Figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
