"""Generate controlled operator comparison report.

Runs both the broad operator overview and controlled comparisons
(controlling for machine, technique, site, and depth) on priority machines.

Output: reports/04_operator_comparison.md + reports/figures/operator/*.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

from macchine.analysis.operator_comparison import (
    compare_operators,
    controlled_operator_comparison,
)
from macchine.storage.catalog import get_merged_trace_index

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

OUTPUT_DIR = Path("output")
FIG_DIR = Path("reports/figures/operator")
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = Path("reports/04_operator_comparison.md")

# Machines with multiple operators and enough data for controlled comparison
CONTROLLED_MACHINES = [
    ("gb50_601", "GRAB", "1514"),        # 4 operators, single site, GRAB only
    ("bg33v_5610", "KELLY", "1508"),      # KELLY operators at site 1508
    ("bg33v_5610", "GRAB", "LignanoSabbiadoro"),  # GRAB operators at Lignano
]


def plot_operator_duration_boxplots(
    df: pd.DataFrame,
    machine: str,
    technique: str,
    site: str,
    operators: list[str],
) -> str | None:
    """Generate duration boxplots per operator."""
    fig, ax = plt.subplots(figsize=(10, 5))

    data = []
    labels = []
    for op in operators:
        vals = df[df["operator"] == op]["duration_min"].dropna()
        if len(vals) >= 5:
            data.append(vals.values)
            labels.append(f"{op}\n(n={len(vals)})")

    if len(data) < 2:
        plt.close(fig)
        return None

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
    colors = plt.cm.Set2(np.linspace(0, 1, len(data)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Session Duration [min]")
    ax.set_title(f"Operator Duration: {machine} / {technique} @ {site}")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    safe_site = site.replace("/", "_").replace(" ", "_")
    fname = f"{machine}_{technique}_{safe_site}_duration_boxplot.png"
    fig.savefig(FIG_DIR / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fname


def plot_operator_cycle_metrics(
    df: pd.DataFrame,
    machine: str,
    site: str,
    operators: list[str],
) -> list[str]:
    """Generate per-operator cycle metric boxplots for GRAB."""
    cycle_cols = [
        ("n_cycles", "Cycles per Session"),
        ("med_cycle_duration", "Median Cycle Duration [s]"),
        ("med_descent_speed", "Median Descent Speed [m/s]"),
        ("med_ascent_speed", "Median Ascent Speed [m/s]"),
    ]
    available = [(c, l) for c, l in cycle_cols if c in df.columns and df[c].notna().sum() > 10]

    if not available:
        return []

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), squeeze=False)

    colors = plt.cm.Set2(np.linspace(0, 1, len(operators)))

    for i, (col, label) in enumerate(available):
        ax = axes[0, i]
        data = []
        labels = []
        for op in operators:
            vals = df[df["operator"] == op][col].dropna()
            if len(vals) >= 5:
                data.append(vals.values)
                labels.append(f"{op}\n(n={len(vals)})")

        if len(data) < 2:
            ax.set_visible(False)
            continue

        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="x", labelsize=8)

    safe_site = site.replace("/", "_").replace(" ", "_")
    fig.suptitle(f"GRAB Cycle Metrics by Operator: {machine} @ {site}", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fname = f"{machine}_{safe_site}_cycle_by_operator.png"
    fig.savefig(FIG_DIR / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return [fname]


def generate_report(
    broad_stats: pd.DataFrame,
    controlled_results: list[dict],
    figures: dict[str, list[str]],
):
    """Write the operator comparison markdown report."""
    lines = [
        "# Operator Comparison\n\n",
        "This report provides both a broad overview of operator activity and controlled ",
        "statistical comparisons that account for machine, technique, site, and depth confounders.\n\n",
        "---\n\n",
    ]

    # Broad overview
    lines.append("## Operator Summary (All Machines)\n\n")
    lines.append("| Operator | Traces | Total Hours | Avg Duration | Median Duration | "
                 "Machines | Techniques |\n")
    lines.append("|----------|--------|-------------|--------------|-----------------|"
                 "----------|------------|\n")

    for _, row in broad_stats.iterrows():
        lines.append(
            f"| {row['operator']} | {row['trace_count']:,} | {row['total_hours']:,.0f} | "
            f"{row['avg_duration_min']:.1f} min | {row['median_duration_min']:.1f} min | "
            f"{row['machines_used']} | {row['techniques']} |\n"
        )

    lines.append("\n---\n\n")

    # Controlled comparisons
    lines.append("## Controlled Comparisons\n\n")
    lines.append("Each comparison below controls for machine, technique, and site. "
                 "Where depth data is available, comparisons are also done within depth bins. "
                 "Statistical significance is assessed with Mann-Whitney U tests (p < 0.05).\n\n")

    for cr in controlled_results:
        key = cr["key"]
        machine = cr["machine"]
        technique = cr["technique"]
        site = cr["site"]
        results_df = cr["results_df"]
        operators = cr["operators"]
        n_sessions = cr["n_sessions"]

        lines.append(f"### {machine} / {technique} @ {site}\n\n")
        lines.append(f"**Operators**: {', '.join(operators)} | "
                     f"**Sessions**: {n_sessions}\n\n")

        # Operator summary stats
        if cr.get("op_stats"):
            lines.append("| Operator | N | Med Duration | Avg Duration |")
            extra_cols = []
            if cr.get("has_cycles"):
                lines[-1] = lines[-1] + " Med Cycles | Med Cycle Dur | Med Desc Speed |"
                extra_cols = ["n_cycles", "med_cycle_duration", "med_descent_speed"]
            lines.append("\n")
            lines.append("|----------|---|-------------|-------------|")
            if extra_cols:
                lines.append("------------|---------------|----------------|")
            lines.append("\n")

            for op, stats in sorted(cr["op_stats"].items()):
                line = (f"| {op} | {stats['n']} | "
                        f"{stats['med_duration']:.1f} min | "
                        f"{stats['avg_duration']:.1f} min |")
                if cr.get("has_cycles"):
                    line += (f" {stats.get('med_n_cycles', 'N/A')} | "
                             f"{stats.get('med_cycle_dur', 'N/A')} | "
                             f"{stats.get('med_desc_speed', 'N/A')} |")
                lines.append(line + "\n")
            lines.append("\n")

        # Significant results
        if not results_df.empty:
            sig = results_df[results_df["significant"]]
            lines.append(f"**Comparisons**: {len(results_df)} total, "
                         f"{len(sig)} significant (p < 0.05)\n\n")

            if not sig.empty:
                lines.append("| Group | Metric | Operator A | Operator B | "
                             "Median A | Median B | p-value |\n")
                lines.append("|-------|--------|------------|------------|"
                             "----------|----------|----------|\n")
                for _, r in sig.iterrows():
                    lines.append(
                        f"| {r['group']} | {r['metric']} | "
                        f"{r['operator_a']} | {r['operator_b']} | "
                        f"{r['median_a']:.2f} | {r['median_b']:.2f} | "
                        f"{r['p_value']:.4f} |\n"
                    )
                lines.append("\n")
        else:
            lines.append("No comparisons could be made (insufficient data per operator).\n\n")

        # Figures
        figs = figures.get(key, [])
        for fname in figs:
            lines.append(f"![{key}](figures/operator/{fname})\n\n")

        lines.append("---\n\n")

    # Methodology
    lines.append("## Methodology\n\n")
    lines.append("1. **Broad comparison**: Per-operator aggregation across all machines/techniques. "
                 "Numbers here are descriptive only and confounded by task assignment.\n")
    lines.append("2. **Controlled comparison**: Filters to a single machine + technique + site. "
                 "Uses merged session index (abutting traces combined). "
                 "Mann-Whitney U test for each operator pair.\n")
    lines.append("3. **Depth binning**: Where depth data is available, traces are grouped "
                 "into 5m depth bins. Comparisons within bins control for depth.\n")
    lines.append("4. **Cycle metrics** (GRAB only): Per-session median of cycle-level features "
                 "(duration, descent/ascent speed) from cycle detection analysis.\n")
    lines.append("5. **Minimum thresholds**: Operators need >= 10 traces; "
                 "each comparison needs >= 5 samples per operator.\n")

    REPORT_PATH.write_text("".join(lines))
    print(f"Report written to {REPORT_PATH}")


def main():
    print("=== Broad Operator Overview ===")
    broad_stats = compare_operators(OUTPUT_DIR)

    controlled_results = []
    all_figures = {}

    for machine, technique, site in CONTROLLED_MACHINES:
        key = f"{machine}_{technique}_{site}".replace(" ", "_").replace("/", "_")
        print(f"\n=== Controlled: {machine} / {technique} @ {site} ===")

        results_df = controlled_operator_comparison(
            OUTPUT_DIR, machine=machine, technique=technique, site=site,
        )

        # Get the filtered dataframe for plots
        df = get_merged_trace_index(OUTPUT_DIR)
        df = df.dropna(subset=["start_time"])
        df = df[df["operator"].notna() & (df["operator"] != "")]
        df = df[df["machine_slug"] == machine]
        df = df[df["technique"] == technique]
        df = df[df["site_id"] == site]
        df["duration_min"] = df["duration_s"] / 60

        op_counts = df["operator"].value_counts()
        valid_ops = op_counts[op_counts >= 10].index.tolist()

        if len(valid_ops) < 2:
            print(f"  Skipping plots — fewer than 2 qualifying operators")
            continue

        df_filtered = df[df["operator"].isin(valid_ops)]

        # Merge cycle data for GRAB (used for both stats and plots)
        has_cycles = False
        if technique == "GRAB":
            try:
                from macchine.analysis.cycle_analysis import track_cycle_degradation
                cycle_df = track_cycle_degradation(OUTPUT_DIR, machine, site=site)
                if not cycle_df.empty:
                    cycle_agg = (
                        cycle_df.groupby("trace_id")
                        .agg(
                            n_cycles=("cycle_depth", "count"),
                            med_cycle_duration=("cycle_duration_s", "median"),
                            med_descent_speed=("descent_speed", "median"),
                            med_ascent_speed=("ascent_speed", "median"),
                        )
                        .reset_index()
                    )
                    df_filtered = df_filtered.merge(cycle_agg, on="trace_id", how="left")
                    has_cycles = True
            except Exception as e:
                print(f"  Could not load cycle metrics: {e}")

        # Compute per-operator summary stats
        op_stats = {}
        for op in valid_ops:
            op_df = df_filtered[df_filtered["operator"] == op]
            stats = {
                "n": len(op_df),
                "med_duration": float(op_df["duration_min"].median()),
                "avg_duration": float(op_df["duration_min"].mean()),
            }
            if has_cycles and "n_cycles" in df_filtered.columns:
                n_cyc = op_df["n_cycles"].dropna()
                cyc_dur = op_df["med_cycle_duration"].dropna()
                desc_spd = op_df["med_descent_speed"].dropna()
                stats["med_n_cycles"] = f"{n_cyc.median():.0f}" if len(n_cyc) > 0 else "N/A"
                stats["med_cycle_dur"] = f"{cyc_dur.median():.0f}s" if len(cyc_dur) > 0 else "N/A"
                stats["med_desc_speed"] = f"{desc_spd.median():.2f}" if len(desc_spd) > 0 else "N/A"
            op_stats[op] = stats

        cr = {
            "key": key,
            "machine": machine,
            "technique": technique,
            "site": site,
            "results_df": results_df,
            "operators": valid_ops,
            "n_sessions": len(df_filtered),
            "op_stats": op_stats,
            "has_cycles": has_cycles,
        }
        controlled_results.append(cr)

        # Generate plots
        figs = []
        fname = plot_operator_duration_boxplots(
            df_filtered, machine, technique, site, valid_ops,
        )
        if fname:
            figs.append(fname)

        if has_cycles:
            cycle_figs = plot_operator_cycle_metrics(
                df_filtered, machine, site, valid_ops,
            )
            figs.extend(cycle_figs)

        if figs:
            all_figures[key] = figs
            for f in figs:
                print(f"  Plot: {f}")

    print("\n=== Generating Report ===")
    generate_report(broad_stats, controlled_results, all_figures)
    print(f"Done! Figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
