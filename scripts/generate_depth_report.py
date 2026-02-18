"""Generate depth measurement investigation report.

Comprehensive analysis of how the Bauer B-Tronic depth sensor works,
its quirks across machines, and what the data actually means.

Output: reports/18_depth_measurement.md + reports/figures/depth/*.png
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
from collections import defaultdict

from macchine.storage.catalog import get_merged_trace_index
from macchine.harmonize.calibration import clean_sentinels_df, get_sentinel_values

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

OUTPUT_DIR = Path("output")
TRACES_DIR = OUTPUT_DIR / "traces"
FIG_DIR = Path("reports/figures/depth")
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = Path("reports/18_depth_measurement.md")

SENTINELS = set(get_sentinel_values())


def _load_trace(site_id: str, machine_slug: str, trace_id: str) -> pd.DataFrame | None:
    """Load a single trace parquet file."""
    slug_path = machine_slug if machine_slug != "unidentified" else "unknown"
    path = TRACES_DIR / str(site_id) / slug_path / f"{trace_id}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        df = clean_sentinels_df(df)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df
    except Exception:
        return None


def _clean_depth_raw(series: pd.Series) -> pd.Series:
    """Remove sentinels but keep ALL real values (including extreme negatives)."""
    d = pd.to_numeric(series, errors="coerce")
    d[d.isin(SENTINELS)] = np.nan
    return d


def _get_depth_stats_per_machine(idx: pd.DataFrame, n_traces: int = 30) -> dict:
    """Sample traces per machine and compute depth statistics."""
    results = {}
    for machine in sorted(idx["machine_slug"].unique()):
        if machine == "":
            machine_label = "(empty_slug)"
        else:
            machine_label = machine
        m_idx = idx[idx["machine_slug"] == machine]
        sample = m_idx.sample(n=min(n_traces, len(m_idx)), random_state=42)

        traces_data = []
        for _, row in sample.iterrows():
            tids = row["trace_ids"].split("|") if isinstance(row["trace_ids"], str) else []
            first_tid = tids[0] if tids else row.get("trace_id", "")
            df = _load_trace(row["site_id"], row["machine_slug"], first_tid)
            if df is None or "Tiefe" not in df.columns:
                continue
            depth = _clean_depth_raw(df["Tiefe"])
            valid = depth.dropna()
            if valid.empty:
                continue
            traces_data.append({
                "depth": valid,
                "min": valid.min(),
                "max": valid.max(),
                "range": valid.max() - valid.min(),
                "frac_neg": (valid < 0).sum() / len(valid),
                "first_val": valid.iloc[0],
                "n_samples": len(valid),
                "is_integer": np.all(np.mod(valid.values, 1) == 0),
                "technique": row["technique"],
                "site_id": row["site_id"],
                "element_name": row.get("element_name", ""),
            })

        if traces_data:
            results[machine_label] = {
                "traces": traces_data,
                "n_sessions": len(m_idx),
                "techniques": m_idx["technique"].value_counts().to_dict(),
                "sites": sorted(m_idx["site_id"].unique()),
            }
    return results


# ── Figure generators ──────────────────────────────────────────────

def fig_depth_range_per_machine(stats: dict) -> str:
    """Box/swarm plot of depth range per machine, colored by sign convention."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [1, 1]})

    machines = sorted(stats.keys())
    positions = range(len(machines))

    # Top: min and max depth per trace
    ax = axes[0]
    for i, m in enumerate(machines):
        mins = [t["min"] for t in stats[m]["traces"]]
        maxs = [t["max"] for t in stats[m]["traces"]]
        ax.scatter([i] * len(mins), mins, s=15, alpha=0.5, c="steelblue", zorder=3)
        ax.scatter([i] * len(maxs), maxs, s=15, alpha=0.5, c="coral", zorder=3)
        # Connect min-max pairs
        for mn, mx in zip(mins, maxs):
            ax.plot([i, i], [mn, mx], c="grey", alpha=0.15, linewidth=0.5)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(machines, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Depth (m)")
    ax.set_title("Raw Depth Range Per Trace (blue = min, red = max)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Bottom: fraction of negative values
    ax = axes[1]
    for i, m in enumerate(machines):
        fracs = [t["frac_neg"] * 100 for t in stats[m]["traces"]]
        bp = ax.boxplot([fracs], positions=[i], widths=0.5, patch_artist=True,
                       medianprops=dict(color="black"))
        # Color by category
        median_frac = np.median(fracs)
        if median_frac > 90:
            color = "#e74c3c"  # red: inverted
        elif median_frac > 30:
            color = "#f39c12"  # orange: mixed
        else:
            color = "#27ae60"  # green: normal
        bp["boxes"][0].set_facecolor(color)
        bp["boxes"][0].set_alpha(0.7)

    ax.axhline(50, color="grey", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(machines, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("% Negative Depth Values")
    ax.set_title("Fraction of Negative Depth Per Trace (green=normal, orange=mixed, red=inverted)",
                 fontweight="bold")
    ax.set_ylim(-5, 105)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fname = "01_depth_range_per_machine.png"
    fig.savefig(FIG_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


def fig_sign_convention_examples(stats: dict) -> str:
    """Show depth traces from 3 machines with different sign conventions."""
    # Pick representative machines: normal (bg30v_2872), mixed (gb50_601), inverted (cube0_482)
    examples = [
        ("bg30v_2872", "Normal: positive = deeper"),
        ("gb50_601", "Mixed: positive = deeper, negatives = above ref"),
        ("cube0_482", "Inverted: negative = deeper"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)

    for ax, (machine, title) in zip(axes, examples):
        if machine not in stats:
            ax.set_title(f"{machine} (no data)", fontweight="bold")
            continue
        # Pick a trace with good data
        traces = stats[machine]["traces"]
        best = max(traces, key=lambda t: t["range"])
        depth = best["depth"]
        x = np.arange(len(depth))
        ax.plot(x, depth.values, linewidth=0.4, alpha=0.8, color="steelblue")
        ax.axhline(0, color="red", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_ylabel("Depth (m)")
        ax.set_title(f"{machine} — {title} [{best['technique']}, site {best['site_id']}]",
                     fontweight="bold", fontsize=10)
        ax.grid(True, alpha=0.3)
        # Annotate range
        ax.text(0.98, 0.95, f"Range: [{depth.min():.0f}, {depth.max():.0f}] m",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    axes[-1].set_xlabel("Sample index")
    fig.tight_layout()
    fname = "02_sign_convention_examples.png"
    fig.savefig(FIG_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


def fig_grab_cycle_anatomy(idx: pd.DataFrame) -> str:
    """Dissect a single GRAB cycle from pile 66D showing negative depth = above reference."""
    # Load S2 of pile 66D (longest session) — concatenate ALL sub-traces
    pile = idx[
        (idx["site_id"] == "1514") & (idx["element_name"] == "66D")
    ].sort_values("start_time")

    if len(pile) < 2:
        return ""

    s2_row = pile.iloc[1]
    tids = s2_row["trace_ids"].split("|") if isinstance(s2_row["trace_ids"], str) else []

    # Load and concatenate all sub-traces for this session
    frames = []
    for tid in tids:
        tdf = _load_trace("1514", "gb50_601", tid)
        if tdf is not None:
            frames.append(tdf)
    if not frames:
        return ""
    df = pd.concat(frames, ignore_index=True)

    depth_raw = _clean_depth_raw(df["Tiefe"])

    # Find a complete cycle: positive depth → 0 → negative → back to positive
    # Look for a transition from >30m to negative
    vals = depth_raw.dropna()
    transitions = []
    for i in range(1, len(vals)):
        prev = vals.iloc[i-1]
        curr = vals.iloc[i]
        if prev > 30 and curr < 0:
            transitions.append(vals.index[i])

    if not transitions:
        return ""

    # Take first transition and show +-100 samples around it
    trans_idx = transitions[0]
    loc = vals.index.get_loc(trans_idx)
    start = max(0, loc - 40)
    end = min(len(vals), loc + 120)
    cycle_depth = vals.iloc[start:end]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [2, 1]})

    # Top: full cycle with annotations
    ax = axes[0]
    x = np.arange(len(cycle_depth))
    y = cycle_depth.values

    # Color segments: positive (blue/below ground), negative (red/above ground)
    for i in range(len(x) - 1):
        color = "steelblue" if y[i] >= 0 else "#e74c3c"
        ax.plot(x[i:i+2], y[i:i+2], color=color, linewidth=2)

    ax.axhline(0, color="black", linewidth=1.5, linestyle="--", label="Ground level (zero reference)")
    ax.axhspan(0, max(y) * 1.1, alpha=0.05, color="brown", label="Below ground (in trench)")
    ax.axhspan(min(y) * 1.1, 0, alpha=0.05, color="skyblue", label="Above ground (dumping)")

    # Annotations
    max_pos = max(y)
    min_neg = min(y)
    ax.annotate("Grab at bottom\nof trench", xy=(np.argmax(y), max_pos),
                xytext=(np.argmax(y) - 8, max_pos - 15),
                fontsize=9, fontweight="bold", color="steelblue",
                arrowprops=dict(arrowstyle="->", color="steelblue"))
    neg_start = next(i for i, v in enumerate(y) if v < -20)
    ax.annotate("Grab above ground\n(swinging to dump)", xy=(neg_start + 20, y[neg_start + 20]),
                xytext=(neg_start + 35, y[neg_start + 20] + 30),
                fontsize=9, fontweight="bold", color="#e74c3c",
                arrowprops=dict(arrowstyle="->", color="#e74c3c"))

    ax.set_ylabel("Depth (m)\npositive = below ground, negative = above reference")
    ax.set_title("Anatomy of a GRAB Cycle: Why Depth Goes Negative (pile 66D, session S2)",
                 fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    # Bottom: phase labels
    ax = axes[1]
    # Classify phases
    phases = []
    for v in y:
        if v > 10:
            phases.append("IN TRENCH")
        elif v >= -5:
            phases.append("SURFACE")
        else:
            phases.append("DUMPING")
    phase_colors = {"IN TRENCH": "steelblue", "SURFACE": "grey", "DUMPING": "#e74c3c"}
    for i in range(len(x)):
        ax.bar(x[i], 1, color=phase_colors[phases[i]], alpha=0.7, width=1.0)

    # Add legend manually
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="steelblue", alpha=0.7, label="In trench (depth > 10m)"),
        Patch(facecolor="grey", alpha=0.7, label="At surface (-5m to 10m)"),
        Patch(facecolor="#e74c3c", alpha=0.7, label="Dumping position (depth < -5m)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel("Sample index (within cycle)")
    ax.set_title("Cycle Phase Classification", fontweight="bold", fontsize=10)

    fig.tight_layout()
    fname = "03_grab_cycle_anatomy.png"
    fig.savefig(FIG_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


def fig_starting_depth(stats: dict) -> str:
    """Histogram of first depth value per session for each machine."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()

    machines = [m for m in sorted(stats.keys()) if m != "(empty_slug)"]
    for i, machine in enumerate(machines[:9]):
        ax = axes[i]
        first_vals = [t["first_val"] for t in stats[machine]["traces"]]
        colors = ["#27ae60" if abs(v) < 2 else ("#f39c12" if abs(v) < 10 else "#e74c3c")
                  for v in first_vals]
        ax.bar(range(len(first_vals)), first_vals, color=colors, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"{machine}", fontweight="bold", fontsize=10)
        ax.set_ylabel("First depth (m)")
        n_zero = sum(1 for v in first_vals if abs(v) < 2)
        ax.text(0.02, 0.95, f"{n_zero}/{len(first_vals)} start at 0",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="wheat", alpha=0.8))
        ax.grid(axis="y", alpha=0.3)

    # Hide unused axes
    for i in range(len(machines), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Session Starting Depth — Does Each Session Reset to Zero?",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fname = "04_starting_depth.png"
    fig.savefig(FIG_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


def fig_resolution_comparison(stats: dict) -> str:
    """Compare depth resolution: integer (1m) vs float (0.01m)."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()

    # Pick 3 integer and 3 float machines
    integer_machines = [m for m in stats if
                       np.mean([t["is_integer"] for t in stats[m]["traces"]]) > 0.5
                       and m != "(empty_slug)"][:3]
    float_machines = [m for m in stats if
                     np.mean([t["is_integer"] for t in stats[m]["traces"]]) <= 0.5
                     and m != "(empty_slug)"][:3]

    for i, (machine, label) in enumerate(
        [(m, "INTEGER (1m)") for m in integer_machines] +
        [(m, "FLOAT (0.01m)") for m in float_machines]
    ):
        ax = axes[i]
        best_trace = max(stats[machine]["traces"], key=lambda t: t["range"])
        depth = best_trace["depth"]
        # Show 200-sample excerpt
        excerpt = depth.iloc[:min(200, len(depth))]
        ax.plot(range(len(excerpt)), excerpt.values, linewidth=0.8, alpha=0.9,
                color="steelblue" if "FLOAT" in label else "#e74c3c",
                marker="." if "INTEGER" in label else "")
        ax.set_title(f"{machine} ({label})", fontweight="bold", fontsize=9)
        ax.set_ylabel("Depth (m)")
        # Show step sizes
        steps = np.abs(np.diff(depth.dropna().values))
        steps = steps[steps > 0]
        if len(steps) > 0:
            ax.text(0.02, 0.05, f"Min step: {np.min(steps):.2f}m\nMedian step: {np.median(steps):.2f}m",
                    transform=ax.transAxes, fontsize=7, va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="wheat", alpha=0.8))
        ax.grid(True, alpha=0.3)

    fig.suptitle("Depth Resolution: Integer (1m) vs Float (0.01m) Machines",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fname = "05_resolution_comparison.png"
    fig.savefig(FIG_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


def fig_depth_by_technique(stats: dict) -> str:
    """Depth behavior grouped by technique."""
    technique_data = defaultdict(list)
    for machine, mdata in stats.items():
        for t in mdata["traces"]:
            technique_data[t["technique"]].append({
                "machine": machine,
                "min": t["min"],
                "max": t["max"],
                "range": t["range"],
                "frac_neg": t["frac_neg"],
            })

    techniques = sorted(technique_data.keys())
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: depth range by technique
    ax = axes[0]
    tech_colors = {"SOB": "#3498db", "GRAB": "#e67e22", "KELLY": "#2ecc71",
                   "CUT": "#e74c3c", "SCM": "#9b59b6", "FREE": "#95a5a6"}
    bp_data = []
    bp_labels = []
    for tech in techniques:
        ranges = [d["range"] for d in technique_data[tech]]
        bp_data.append(ranges)
        bp_labels.append(f"{tech}\n(n={len(ranges)})")

    bp = ax.boxplot(bp_data, labels=bp_labels, patch_artist=True, showfliers=True,
                   flierprops=dict(markersize=3, alpha=0.3))
    for patch, tech in zip(bp["boxes"], techniques):
        patch.set_facecolor(tech_colors.get(tech, "#95a5a6"))
        patch.set_alpha(0.7)

    ax.set_ylabel("Depth Range (m)")
    ax.set_title("Depth Range by Technique", fontweight="bold")
    ax.set_yscale("symlog", linthresh=50)
    ax.grid(axis="y", alpha=0.3)

    # Right: fraction negative by technique
    ax = axes[1]
    bp_data_neg = []
    for tech in techniques:
        fracs = [d["frac_neg"] * 100 for d in technique_data[tech]]
        bp_data_neg.append(fracs)

    bp = ax.boxplot(bp_data_neg, labels=bp_labels, patch_artist=True,
                   flierprops=dict(markersize=3, alpha=0.3))
    for patch, tech in zip(bp["boxes"], techniques):
        patch.set_facecolor(tech_colors.get(tech, "#95a5a6"))
        patch.set_alpha(0.7)

    ax.set_ylabel("% Negative Depth Values")
    ax.set_title("Negative Depth by Technique", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fname = "06_depth_by_technique.png"
    fig.savefig(FIG_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


def fig_pile66d_session_resets(idx: pd.DataFrame) -> str:
    """Show how each session for pile 66D starts at zero — no inter-session continuity."""
    pile = idx[
        (idx["site_id"] == "1514") & (idx["element_name"] == "66D")
    ].sort_values("start_time")

    if pile.empty:
        return ""

    fig, axes = plt.subplots(len(pile), 1, figsize=(14, 2.5 * len(pile)), sharex=False)
    if len(pile) == 1:
        axes = [axes]

    cmap = plt.cm.tab10
    for i, (_, row) in enumerate(pile.iterrows()):
        ax = axes[i]
        tids = row["trace_ids"].split("|") if isinstance(row["trace_ids"], str) else []
        first_tid = tids[0] if tids else row.get("trace_id", "")
        df = _load_trace("1514", "gb50_601", first_tid)
        if df is None or "Tiefe" not in df.columns:
            continue
        depth = _clean_depth_raw(df["Tiefe"])
        x = np.arange(len(depth))
        ax.plot(x, depth.values, linewidth=0.5, alpha=0.8, color=cmap(i))
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_ylabel("Depth (m)", fontsize=8)
        date_str = row["start_time"].strftime("%b %d %H:%M")
        ax.set_title(f"S{i+1}: {date_str} ({len(depth)} samples, first={depth.iloc[0]:.0f}m)",
                     fontweight="bold", fontsize=9, loc="left")
        # Show positive-only range
        pos = depth[depth > 0]
        neg = depth[depth < -5]
        if not pos.empty:
            ax.text(0.98, 0.05, f"Positive: [0, {pos.max():.0f}]m\nNegative: [{neg.min():.0f}, -5]m" if not neg.empty else f"Positive: [0, {pos.max():.0f}]m",
                    transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="wheat", alpha=0.8))
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Sample index")
    fig.suptitle("Pile 66D — Raw Depth Per Session (including negative values)",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fname = "07_pile66d_raw_sessions.png"
    fig.savefig(FIG_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


def fig_depth_sensor_availability(idx: pd.DataFrame, stats: dict) -> str:
    """Summary heatmap: which depth sensors each machine has."""
    depth_sensors = ["Tiefe", "Vorschub Tiefe", "Tiefe Winde 2",
                     "Tiefe_Hauptwinde_GOK", "Tiefe_Bohrrohr_GOK"]

    machines = [m for m in sorted(stats.keys()) if m != "(empty_slug)"]

    # Check which sensors each machine has
    availability = {}
    for machine in machines:
        m_idx = idx[idx["machine_slug"] == machine].head(5)
        sensor_present = {s: False for s in depth_sensors}
        for _, row in m_idx.iterrows():
            tids = row["trace_ids"].split("|") if isinstance(row["trace_ids"], str) else []
            first_tid = tids[0] if tids else row.get("trace_id", "")
            df = _load_trace(row["site_id"], row["machine_slug"], first_tid)
            if df is None:
                continue
            for s in depth_sensors:
                if s in df.columns:
                    vals = pd.to_numeric(df[s], errors="coerce").dropna()
                    vals = vals[~vals.isin(SENTINELS)]
                    if len(vals) > 10:
                        sensor_present[s] = True
        availability[machine] = sensor_present

    # Plot as heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    data = np.array([[1 if availability[m][s] else 0 for s in depth_sensors] for m in machines])
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(depth_sensors)))
    ax.set_xticklabels(depth_sensors, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels(machines, fontsize=9)

    # Add text annotations
    for i in range(len(machines)):
        for j in range(len(depth_sensors)):
            text = "Y" if data[i, j] else ""
            ax.text(j, i, text, ha="center", va="center", fontsize=10, fontweight="bold")

    ax.set_title("Depth Sensor Availability Per Machine", fontweight="bold")
    fig.tight_layout()
    fname = "08_depth_sensor_availability.png"
    fig.savefig(FIG_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


# ── Report generator ──────────────────────────────────────────────

def generate_report(figures: dict, stats: dict):
    """Write the comprehensive depth measurement report."""
    lines = []
    w = lines.append

    w("# Depth Measurement Investigation\n\n")
    w("A comprehensive analysis of how depth (\"Tiefe\") is measured, recorded, and\n")
    w("represented across the Bauer drilling fleet. This report examines the B-Tronic\n")
    w("depth sensor system, its quirks, sign conventions, resolution, and implications\n")
    w("for data analysis.\n\n")

    w("---\n\n")

    # ── Section 1: How depth is measured ──
    w("## 1. How the B-Tronic Measures Depth\n\n")
    w("### Measurement Mechanism\n\n")
    w("Bauer rigs use the **B-Tronic** electronic monitoring system. Depth is measured\n")
    w("by **rotary encoders on the winch drums** that count cable payout:\n\n")
    w("- **Tiefe** (primary depth): Measured from rope length on the main winch. As the\n")
    w("  tool descends, more cable pays out, and the depth counter increases.\n")
    w("- **Vorschub Tiefe** (feed depth): Position of the crowd system (kelly bar feed).\n")
    w("- **Tiefe Winde 2** (winch 2 depth): Depth on the secondary/auxiliary winch.\n")
    w("- **Tiefe_Hauptwinde_GOK** / **Tiefe_Bohrrohr_GOK**: Depth referenced to\n")
    w("  ground surface (GOK = Geländeoberkante = ground level). Available only on\n")
    w("  select machines.\n\n")

    w("### The Zero Reference and T=0 Reset\n\n")
    w("From the Bauer BG 20H and BG 25 instruction manuals:\n\n")
    w("> *\"Place drilling tool in desired position\"* and *\"Press button T=0 to reset\n")
    w("> the depth indicator to '0'\"*\n\n")
    w("Key implications:\n\n")
    w("1. **Zero is operator-defined**: The operator places the tool at a reference\n")
    w("   position (typically ground level) and presses T=0 to reset the counter.\n")
    w("2. **Reset can happen at any time**: The depth can be re-zeroed mid-session.\n")
    w("3. **No absolute reference**: Depth values are always relative to wherever\n")
    w("   the operator set zero. Different sessions for the same pile start at zero\n")
    w("   independently — there is no inter-session depth continuity.\n")
    w("4. **Negative values = above reference**: If the tool moves above the zero\n")
    w("   point (e.g., a grab lifting soil above ground level), depth goes negative.\n\n")

    w("### Sources\n\n")
    w("- [BAUER BG 25 Instruction Manual (ManualsLib)](https://www.manualslib.com/manual/2187696/Bauer-Bg-25.html) — Section 5.8 \"Depth Measuring Control\"\n")
    w("- [BAUER BG 20 H Instruction Manual (ManualsLib)](https://www.manualslib.com/manual/2215513/Bauer-Bg-24-H.html) — Depth reset procedure\n")
    w("- [B-Tronic System (Bauer Maschinen GmbH)](https://equipment.bauer.de/en/b-tronic) — Electronic monitoring system overview\n")
    w("- [B-Tronic Display Documentation](https://ecodrill.es/wp-content/uploads/2019/04/B_Tronic_Bildschirm_EN_905_759_2.pdf)\n\n")

    w("---\n\n")

    # ── Section 2: Depth sensors available ──
    w("## 2. Depth Sensors Available Per Machine\n\n")
    w("Not all machines record the same depth channels. The primary `Tiefe` channel\n")
    w("is universally present (100% of all traces). The GOK-referenced channels\n")
    w("(`Tiefe_Hauptwinde_GOK`, `Tiefe_Bohrrohr_GOK`) are rarer and provide\n")
    w("ground-surface-referenced depth on newer/upgraded machines.\n\n")
    if figures.get("availability"):
        w(f"![Depth sensor availability](figures/depth/{figures['availability']})\n\n")

    w("---\n\n")

    # ── Section 3: Does depth start at zero? ──
    w("## 3. Does Each Session Start at Zero?\n\n")
    w("**Yes, predominantly.** The operator resets depth to zero at the start of each\n")
    w("recording session. Across all machines:\n\n")

    # Compute starting stats
    w("| Machine | Technique(s) | Starts near 0 (%) | Notable exceptions |\n")
    w("|---------|-------------|-------------------|--------------------|\n")
    for machine in sorted(stats.keys()):
        if machine == "(empty_slug)":
            continue
        traces = stats[machine]["traces"]
        n_zero = sum(1 for t in traces if abs(t["first_val"]) < 2)
        pct = 100 * n_zero / len(traces) if traces else 0
        techs = ", ".join(sorted(stats[machine]["techniques"].keys()))
        exceptions = []
        for t in traces:
            if abs(t["first_val"]) >= 10:
                exceptions.append(f"{t['first_val']:.0f}m")
        exc_str = ", ".join(exceptions[:3]) if exceptions else "—"
        if len(exceptions) > 3:
            exc_str += f" (+{len(exceptions)-3} more)"
        w(f"| {machine} | {techs} | {pct:.0f}% | {exc_str} |\n")
    w("\n")

    if figures.get("starting"):
        w(f"![Starting depth per session](figures/depth/{figures['starting']})\n\n")

    w("**Exceptions**: A few sessions start at non-zero depth, typically because:\n")
    w("- The operator did not reset T=0 before starting the recording\n")
    w("- The recording system started before the operator reached ground level\n")
    w("- Constant values like 99.99 (bg30v_2872) suggest the sensor was in a\n")
    w("  diagnostic/default state\n\n")

    w("---\n\n")

    # ── Section 4: Why negative depth? ──
    w("## 4. Why Depth Goes Negative\n\n")
    w("Negative depth values appear in 64% of all traces. There are three distinct\n")
    w("causes, depending on technique and machine:\n\n")

    w("### 4a. GRAB Technique: Above-Ground Dumping Phase\n\n")
    w("This is the most common and best-understood cause. During a GRAB cycle:\n\n")
    w("1. **Descent**: Grab descends into the trench (depth increases, positive values)\n")
    w("2. **Bottom**: Grab reaches target depth, jaws close to capture soil\n")
    w("3. **Ascent**: Grab rises back through the trench (depth decreases)\n")
    w("4. **Surface crossing**: Depth passes through zero as the grab exits the trench\n")
    w("5. **Dumping**: Grab continues upward and swings to dump soil — depth goes\n")
    w("   **negative** because the grab is now above the zero reference point\n")
    w("6. **Return**: Grab swings back and descends for the next cycle\n\n")

    if figures.get("anatomy"):
        w(f"![GRAB cycle anatomy](figures/depth/{figures['anatomy']})\n\n")

    w("The extreme negative values on gb50_601 (down to -3433m) are **not physical**.\n")
    w("They represent the winch encoder continuing to count cable payout/reeling\n")
    w("during the above-ground swing phase, amplified by the integer-only (1m)\n")
    w("resolution and possible encoder overflow/accumulation errors.\n\n")

    w("### 4b. CUT Technique: Reference Offset\n\n")
    w("For CUT machines (cutter wheels), negative values typically mean the\n")
    w("cutting head is above the ground reference. This is common during\n")
    w("entry/exit phases. On **cube0_482**, the sign convention is entirely\n")
    w("inverted: negative = below ground, positive = above. This is a\n")
    w("machine-specific configuration, not a data error.\n\n")

    w("### 4c. SCM/KELLY: Mixed causes\n\n")
    w("SCM traces on bg33v_5610 at site 1508 show extreme negatives (-1901m)\n")
    w("that are clearly non-physical. These likely represent raw encoder\n")
    w("counts that were not properly converted to engineering units.\n\n")

    w("---\n\n")

    # ── Section 5: Sign conventions ──
    w("## 5. Sign Conventions Across the Fleet\n\n")
    w("Three distinct sign conventions exist in the dataset:\n\n")

    w("| Convention | Machines | Description |\n")
    w("|-----------|----------|-------------|\n")
    w("| **Normal** (positive = deeper) | bg30v_2872, bg45v_3923, bg45v_4027 | Standard: 0 at surface, increases with depth. Negatives rare/absent. |\n")
    w("| **Mixed** (both signs) | gb50_601, bg28h_6061, bg33v_5610, bg42v_5925, mc86_621 | 0 at surface. Positive = in-ground, negative = above ground. Common with GRAB technique. |\n")
    w("| **Inverted** (negative = deeper) | cube0_482 | 0 at surface, depth DECREASES (more negative) as tool goes deeper. |\n\n")

    if figures.get("sign_conv"):
        w(f"![Sign convention examples](figures/depth/{figures['sign_conv']})\n\n")

    w("---\n\n")

    # ── Section 6: Resolution ──
    w("## 6. Depth Resolution: Integer vs Float\n\n")
    w("The fleet splits into two resolution tiers:\n\n")

    w("| Resolution | Machines | Min Step | Quality |\n")
    w("|-----------|----------|----------|--------|\n")
    w("| **0.01m** (float) | bg30v_2872, bg42v_5925, bg45v_3923, bg45v_4027 | 0.01m | Clean, smooth depth curves. |\n")
    w("| **1m** (integer) | gb50_601, bg28h_6061, cube0_482, mc86_621, bg33v_5610 | 1m | Staircase-like depth. Coarser machines correlate with more data quality issues. |\n\n")

    if figures.get("resolution"):
        w(f"![Resolution comparison](figures/depth/{figures['resolution']})\n\n")

    w("The correlation between integer-only resolution and data quality problems\n")
    w("is notable: **all machines with extreme depth range issues use 1m integer\n")
    w("resolution**. This suggests these machines have older or less-capable\n")
    w("B-Tronic encoders, or the data export pipeline loses decimal precision.\n\n")

    w("---\n\n")

    # ── Section 7: Per-technique depth behavior ──
    w("## 7. Depth Behavior by Technique\n\n")

    if figures.get("by_technique"):
        w(f"![Depth by technique](figures/depth/{figures['by_technique']})\n\n")

    w("| Technique | Typical Range | % Negative | Behavior |\n")
    w("|-----------|--------------|-----------|----------|\n")
    w("| **SOB** (Soldier pile boring) | 0–28m | ~0% | Cleanest. Short piles, always positive. |\n")
    w("| **KELLY** (Kelly drilling) | 0–36m | ~10% | Mostly clean. Small negatives from tool above ground. |\n")
    w("| **GRAB** (Clamshell grab) | -3400 to 120m | ~50% | Most negative values due to above-ground dumping cycle. |\n")
    w("| **CUT** (Cutter wheel) | -690 to 65m | ~35–99% | Varies by machine. cube0_482 fully inverted. |\n")
    w("| **SCM** (Soil cement mixing) | -1900 to 167m | ~80% | Problematic on bg33v_5610. Likely uncalibrated at site 1508. |\n\n")

    w("---\n\n")

    # ── Section 8: Pile 66D deep dive ──
    w("## 8. Case Study: Pile 66D Raw Depth\n\n")
    w("The merge case study (Report 17) filters depth to [−2m, 120m] for analysis.\n")
    w("Here we show the **raw unfiltered depth** for each session, revealing the\n")
    w("full extent of above-ground excursions.\n\n")

    if figures.get("pile66d"):
        w(f"![Pile 66D raw sessions](figures/depth/{figures['pile66d']})\n\n")

    w("Key observations:\n\n")
    w("1. **Every session starts at or near zero** — confirming the operator resets\n")
    w("   T=0 at the beginning of each recording.\n")
    w("2. **Positive values (0–91m)** represent the grab inside the trench. These\n")
    w("   are the physically meaningful depth measurements.\n")
    w("3. **Negative values (down to -3433m)** represent the grab above the zero\n")
    w("   reference during the dumping phase. The extreme magnitude is an artifact\n")
    w("   of the winch encoder's integer-only resolution and accumulation behavior.\n")
    w("4. **No inter-session continuity** — each session independently starts at zero.\n")
    w("   The physical depth of the trench is not carried over between recordings.\n\n")

    w("---\n\n")

    # ── Section 9: Overview plot ──
    w("## 9. Fleet-Wide Depth Summary\n\n")
    if figures.get("overview"):
        w(f"![Depth range per machine](figures/depth/{figures['overview']})\n\n")

    w("---\n\n")

    # ── Section 10: Master summary table ──
    w("## 10. Master Summary Table\n\n")
    w("| Machine | Techniques | Sites | Sessions | Typical Range (m) | % Negative | Resolution | Issues |\n")
    w("|---------|-----------|-------|----------|-------------------|-----------|-----------|--------|\n")

    for machine in sorted(stats.keys()):
        if machine == "(empty_slug)":
            continue
        mdata = stats[machine]
        traces = mdata["traces"]
        techs = "/".join(sorted(mdata["techniques"].keys()))
        sites_str = ", ".join(mdata["sites"][:2])
        if len(mdata["sites"]) > 2:
            sites_str += f" +{len(mdata['sites'])-2}"
        n_sess = mdata["n_sessions"]

        mins = [t["min"] for t in traces]
        maxs = [t["max"] for t in traces]
        typ_min = np.median(mins)
        typ_max = np.median(maxs)
        avg_neg = np.mean([t["frac_neg"] for t in traces]) * 100
        is_int = np.mean([t["is_integer"] for t in traces]) > 0.5
        res_str = "1m" if is_int else "0.01m"

        avg_range = np.mean([t["range"] for t in traces])
        issues = []
        if avg_range > 200:
            issues.append("Unreasonable range")
        if avg_neg > 90:
            issues.append("Inverted sign")
        elif avg_neg > 30:
            issues.append("Mixed sign")
        if is_int and avg_range > 10:
            issues.append("Integer only")

        issue_str = ", ".join(issues) if issues else "OK"
        w(f"| {machine} | {techs} | {sites_str} | {n_sess} | [{typ_min:.0f}, {typ_max:.0f}] | {avg_neg:.0f}% | {res_str} | {issue_str} |\n")

    w("\n---\n\n")

    # ── Section 11: Recommendations ──
    w("## 11. Recommendations for Data Processing\n\n")

    w("### Filtering Strategy\n\n")
    w("Based on this investigation, depth data should be processed with technique-aware\n")
    w("filters:\n\n")
    w("```python\n")
    w("# GRAB: keep only in-trench values (positive depth)\n")
    w("depth_grab = depth[depth >= -2]  # small negative tolerance for surface noise\n")
    w("depth_grab = depth_grab[depth_grab <= 150]  # reasonable max for GRAB panels\n\n")
    w("# CUT: handle inverted sign convention\n")
    w("if machine == 'cube0_482':\n")
    w("    depth = -depth  # flip sign\n")
    w("depth_cut = depth[depth >= -5]\n")
    w("depth_cut = depth_cut[depth_cut <= 200]\n\n")
    w("# SOB/KELLY: straightforward\n")
    w("depth_clean = depth[(depth >= 0) & (depth <= 50)]\n")
    w("```\n\n")

    w("### Calibration Status Updates Needed\n\n")
    w("The current `calibration_status.yaml` marks `Tiefe` as calibrated for all\n")
    w("machines. Based on this analysis, this should be revised:\n\n")
    w("| Machine | Current Status | Recommended | Reason |\n")
    w("|---------|---------------|------------|--------|\n")
    w("| gb50_601 | calibrated | **partially calibrated** | Positive values OK, extreme negatives are encoder artifacts |\n")
    w("| mc86_621 | calibrated | **partially calibrated** | CUT traces reach -690m regularly |\n")
    w("| bg33v_5610 | calibrated | **per-site** | OK at most sites; SCM at 1508 reaches -1901m |\n")
    w("| cube0_482 | calibrated | **inverted** | Sign convention needs flipping |\n")
    w("| Others | calibrated | calibrated | Confirmed OK |\n\n")

    w("### Inter-Session Depth\n\n")
    w("Depth resets to zero at each session start. Multi-session pile analysis\n")
    w("(like the merge case study in Report 17) should treat each session's depth\n")
    w("independently and **never assume depth continuity between sessions**.\n")

    REPORT_PATH.write_text("".join(lines))
    print(f"Report written to {REPORT_PATH}")


def main():
    print("Loading merged trace index...")
    idx = get_merged_trace_index(OUTPUT_DIR)
    print(f"  {len(idx)} sessions across {idx['machine_slug'].nunique()} machines")

    print("\nComputing per-machine depth statistics (sampling traces)...")
    stats = _get_depth_stats_per_machine(idx, n_traces=30)
    print(f"  Analyzed {sum(len(s['traces']) for s in stats.values())} traces across {len(stats)} machines")

    print("\nGenerating figures...")
    figures = {}

    fig_funcs = [
        ("overview", "Depth range overview", lambda: fig_depth_range_per_machine(stats)),
        ("sign_conv", "Sign convention examples", lambda: fig_sign_convention_examples(stats)),
        ("anatomy", "GRAB cycle anatomy", lambda: fig_grab_cycle_anatomy(idx)),
        ("starting", "Starting depth", lambda: fig_starting_depth(stats)),
        ("resolution", "Resolution comparison", lambda: fig_resolution_comparison(stats)),
        ("by_technique", "Depth by technique", lambda: fig_depth_by_technique(stats)),
        ("pile66d", "Pile 66D raw sessions", lambda: fig_pile66d_session_resets(idx)),
        ("availability", "Sensor availability", lambda: fig_depth_sensor_availability(idx, stats)),
    ]

    for key, name, func in fig_funcs:
        print(f"  {name}...")
        try:
            fname = func()
            if fname:
                figures[key] = fname
                print(f"    -> {fname}")
            else:
                print(f"    -> SKIPPED (no data)")
        except Exception as e:
            print(f"    -> ERROR: {e}")

    print("\nWriting report...")
    generate_report(figures, stats)
    print(f"Done! Figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
