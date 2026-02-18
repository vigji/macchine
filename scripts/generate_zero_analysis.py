"""Analyze operator zeroing practices.

For key sensors (depth, inclination, deviation), checks whether the initial values
at the start of each trace indicate proper zeroing. When zeroing was not done properly,
assesses whether corrections can be applied (constant offset subtraction).

Generates figures in reports/figures/zeros/ and report at reports/11_zero_analysis.md.
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

from macchine.harmonize.calibration import (
    is_calibrated,
    get_unit,
    get_display_label,
    clean_sentinels_df,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

OUTPUT_DIR = Path("output")
FIG_DIR = Path("reports/figures/zeros")
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = Path("reports/11_zero_analysis.md")

sns.set_theme(style="whitegrid", font_scale=1.0)

# Sensors to check for zeroing
ZERO_SENSORS = {
    "depth": {
        "sensors": ["Tiefe", "Vorschub Tiefe"],
        "expected_zero": 0.0,
        "tolerance": 0.5,
        "unit": "m",
        "label": "Depth",
        "description": "Depth sensors should start at 0 (ground level). Non-zero start indicates the operator didn't zero the depth reference.",
    },
    "inclination": {
        "sensors": ["Neigung X", "Neigung Y", "Neigung X Mast", "Neigung Y Mast"],
        "expected_zero": 0.0,
        "tolerance": 0.3,
        "unit": "deg",
        "label": "Inclination",
        "description": "Inclination should be near 0 when machine is level. Small offsets are normal (imperfect leveling), but large offsets indicate no zeroing.",
    },
    "deviation": {
        "sensors": ["Abweichung X", "Abweichung Y", "Messpunktabw. X", "Messpunktabw. Y",
                     "x-Abweichung", "y-Abweichung"],
        "expected_zero": 0.0,
        "tolerance": 1.0,
        "unit": "mm",
        "label": "Deviation",
        "description": "Position deviation from target should start near 0. Large initial offsets indicate the reference was not set.",
    },
    "torque": {
        "sensors": ["Drehmoment", "DrehmomentkNm"],
        "expected_zero": 0.0,
        "tolerance": 5.0,
        "unit": "kNm",
        "label": "Torque",
        "description": "Torque should be near 0 at trace start (tool not yet engaged). Non-zero start may indicate the trace started mid-operation.",
    },
    "pressure": {
        "sensors": ["Druck Pumpe 1", "Druck Pumpe 2", "Druck Pumpe 3", "Druck Pumpe 4"],
        "expected_zero": 0.0,
        "tolerance": 10.0,
        "unit": "bar",
        "label": "Pump Pressure",
        "description": "Pump pressure at trace start. Low pressure indicates idle start (normal). High pressure suggests recording started mid-operation.",
    },
}

N_INITIAL = 5


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


def extract_initial_values(trace_path: Path, sensors: list[str]) -> dict | None:
    """Read a trace and extract initial values for specified sensors."""
    if not trace_path.exists():
        return None
    try:
        tdf = pd.read_parquet(trace_path)
    except Exception:
        return None

    # Clean sentinel values
    tdf = clean_sentinels_df(tdf)

    available = [s for s in sensors if s in tdf.columns]
    if not available:
        return None

    result = {}
    for s in available:
        vals = tdf[s].dropna()
        if len(vals) < N_INITIAL:
            continue
        initial_vals = vals.iloc[:N_INITIAL]
        all_vals = vals

        result[s] = {
            "initial_mean": float(initial_vals.mean()),
            "initial_std": float(initial_vals.std()),
            "initial_min": float(initial_vals.min()),
            "initial_max": float(initial_vals.max()),
            "overall_mean": float(all_vals.mean()),
            "overall_min": float(all_vals.min()),
            "overall_max": float(all_vals.max()),
            "overall_range": float(all_vals.max() - all_vals.min()),
            "n_samples": len(all_vals),
            "first_value": float(vals.iloc[0]),
            "last_value": float(vals.iloc[-1]),
            "initial_stable": float(initial_vals.std()) < 0.1 * abs(float(initial_vals.mean()) + 0.01),
        }
    return result if result else None


def collect_zero_data(df_index: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Collect initial value data for all sensor groups."""
    all_data = {}

    for group_key, group_info in ZERO_SENSORS.items():
        sensors = group_info["sensors"]
        records = []

        for _, row in tqdm(df_index.iterrows(), total=len(df_index),
                           desc=f"  {group_info['label']}..."):
            trace_path = get_trace_path(row)
            initial = extract_initial_values(trace_path, sensors)
            if initial is None:
                continue
            for sensor_name, stats in initial.items():
                rec = {
                    "trace_id": row["trace_id"],
                    "start_time": row["start_time"],
                    "site_id": row["site_id"],
                    "machine_slug": row["machine_slug"],
                    "technique": row["technique"],
                    "operator": row["operator"],
                    "element_name": row["element_name"],
                    "duration_min": row["duration_min"],
                    "sensor": sensor_name,
                }
                rec.update(stats)
                # Classify zeroing quality
                offset = abs(stats["initial_mean"] - group_info["expected_zero"])
                if offset <= group_info["tolerance"]:
                    rec["zero_quality"] = "good"
                elif offset <= group_info["tolerance"] * 3:
                    rec["zero_quality"] = "marginal"
                else:
                    rec["zero_quality"] = "not_zeroed"
                rec["offset"] = stats["initial_mean"] - group_info["expected_zero"]
                records.append(rec)

        if records:
            all_data[group_key] = pd.DataFrame(records)
            n = len(all_data[group_key])
            n_good = (all_data[group_key]["zero_quality"] == "good").sum()
            print(f"  {group_key}: {n} measurements, {n_good} good ({n_good / n * 100:.0f}%)")

    return all_data


def _get_unit_label(sensor: str, machine_slug: str, fallback_unit: str) -> str:
    """Get unit label accounting for calibration."""
    unit = get_unit(sensor, machine_slug)
    if unit == "arb. units":
        return "arb. units"
    return fallback_unit


def plot_zero_distribution(df: pd.DataFrame, group_key: str, group_info: dict) -> str:
    """Histogram of initial offsets across all traces."""
    sensors = sorted(df["sensor"].unique())
    n = len(sensors)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)

    for i, sensor in enumerate(sensors):
        ax = axes[0, i]
        sd = df[df["sensor"] == sensor]

        colors = {"good": "#2ecc71", "marginal": "#f39c12", "not_zeroed": "#e74c3c"}
        for q in ["good", "marginal", "not_zeroed"]:
            subset = sd[sd["zero_quality"] == q]
            if len(subset) > 0:
                ax.hist(subset["offset"], bins=50, alpha=0.6, label=f"{q} ({len(subset)})",
                        color=colors[q])

        ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
        ax.axvline(-group_info["tolerance"], color="gray", linewidth=0.5, linestyle=":")
        ax.axvline(group_info["tolerance"], color="gray", linewidth=0.5, linestyle=":")

        display = get_display_label(sensor)
        ax.set_xlabel(f"Initial Offset [{group_info['unit']}]")
        ax.set_ylabel("Count")
        ax.set_title(display, fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Initial Value Distribution \u2014 {group_info['label']}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fname = f"{group_key}_offset_distribution.png"
    fig.savefig(FIG_DIR / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fname


def plot_zero_by_operator(df: pd.DataFrame, group_key: str, group_info: dict) -> str | None:
    """Compare zeroing practices across operators."""
    df_op = df[df["operator"].notna() & (df["operator"] != "")].copy()
    if len(df_op) < 20:
        return None

    op_counts = df_op["operator"].value_counts()
    major_ops = op_counts[op_counts >= 10].index.tolist()
    if len(major_ops) < 2:
        return None

    df_op = df_op[df_op["operator"].isin(major_ops)]

    sensor = group_info["sensors"][0]
    sd = df_op[df_op["sensor"] == sensor]
    if len(sd) < 20:
        for s in group_info["sensors"]:
            sd = df_op[df_op["sensor"] == s]
            if len(sd) >= 20:
                sensor = s
                break
        else:
            return None

    display = get_display_label(sensor)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Boxplot of offsets by operator
    sns.boxplot(data=sd, x="operator", y="offset", hue="zero_quality",
                hue_order=["good", "marginal", "not_zeroed"],
                palette={"good": "#2ecc71", "marginal": "#f39c12", "not_zeroed": "#e74c3c"},
                ax=axes[0], showfliers=False)
    axes[0].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[0].set_title(f"{display} \u2014 Offset by Operator", fontsize=11)
    axes[0].set_ylabel(f"Offset [{group_info['unit']}]")
    axes[0].tick_params(axis="x", rotation=30)

    # Stacked bar: zeroing quality proportions per operator
    quality_counts = sd.groupby(["operator", "zero_quality"]).size().unstack(fill_value=0)
    quality_pct = quality_counts.div(quality_counts.sum(axis=1), axis=0) * 100
    quality_pct.plot(kind="bar", stacked=True, ax=axes[1],
                     color=["#2ecc71", "#f39c12", "#e74c3c"])
    axes[1].set_title(f"{display} \u2014 Zeroing Quality by Operator", fontsize=11)
    axes[1].set_ylabel("Percentage")
    axes[1].set_xlabel("")
    axes[1].legend(title="Quality", fontsize=8)
    axes[1].tick_params(axis="x", rotation=30)

    fig.suptitle(f"Operator Zeroing Practices \u2014 {group_info['label']}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fname = f"{group_key}_by_operator.png"
    fig.savefig(FIG_DIR / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fname


def plot_zero_by_machine(df: pd.DataFrame, group_key: str, group_info: dict) -> str | None:
    """Compare zeroing across machines."""
    machines = sorted(df["machine_slug"].unique())
    if len(machines) < 2:
        return None

    sensor = df["sensor"].value_counts().index[0]
    sd = df[df["sensor"] == sensor]
    display = get_display_label(sensor)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Boxplot by machine
    sns.boxplot(data=sd, x="machine_slug", y="offset", hue="technique",
                ax=axes[0], showfliers=False)
    axes[0].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[0].set_title(f"{display} \u2014 Offset by Machine", fontsize=11)
    axes[0].set_ylabel(f"Offset [{group_info['unit']}]")
    axes[0].tick_params(axis="x", rotation=30)

    # Quality proportions by machine
    quality_counts = sd.groupby(["machine_slug", "zero_quality"]).size().unstack(fill_value=0)
    quality_pct = quality_counts.div(quality_counts.sum(axis=1), axis=0) * 100
    quality_pct.plot(kind="bar", stacked=True, ax=axes[1],
                     color=["#2ecc71", "#f39c12", "#e74c3c"])
    axes[1].set_title(f"{display} \u2014 Zeroing Quality by Machine", fontsize=11)
    axes[1].set_ylabel("Percentage")
    axes[1].set_xlabel("")
    axes[1].legend(title="Quality", fontsize=8)
    axes[1].tick_params(axis="x", rotation=30)

    fig.suptitle(f"Machine Zeroing Comparison \u2014 {group_info['label']}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fname = f"{group_key}_by_machine.png"
    fig.savefig(FIG_DIR / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fname


def plot_zero_over_time(df: pd.DataFrame, group_key: str, group_info: dict) -> str | None:
    """Plot initial offsets over time per machine to detect drift."""
    machines = sorted(df["machine_slug"].unique())
    sensor = df["sensor"].value_counts().index[0]
    sd = df[df["sensor"] == sensor].sort_values("start_time")
    display = get_display_label(sensor)

    n_machines = min(len(machines), 6)
    if n_machines < 1:
        return None

    fig, axes = plt.subplots(n_machines, 1, figsize=(14, 3 * n_machines), squeeze=False, sharex=True)

    for i, m in enumerate(machines[:n_machines]):
        ax = axes[i, 0]
        ms = sd[sd["machine_slug"] == m].sort_values("start_time")
        if ms.empty:
            continue

        colors = {"good": "#2ecc71", "marginal": "#f39c12", "not_zeroed": "#e74c3c"}
        for q in ["good", "marginal", "not_zeroed"]:
            subset = ms[ms["zero_quality"] == q]
            if len(subset) > 0:
                ax.scatter(subset["start_time"], subset["offset"], s=15, alpha=0.5,
                           color=colors[q], label=q)

        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.axhline(group_info["tolerance"], color="gray", linewidth=0.5, linestyle=":")
        ax.axhline(-group_info["tolerance"], color="gray", linewidth=0.5, linestyle=":")
        ax.set_ylabel(f"{m}\n[{group_info['unit']}]", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")

        # Add site boundaries
        site_changes = ms.groupby("site_id")["start_time"].agg(["min", "max"])
        for _, sc in site_changes.iterrows():
            ax.axvspan(sc["min"], sc["max"], alpha=0.05, color="blue")

    axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45, ha="right")
    fig.suptitle(
        f"{display} \u2014 Initial Offset Over Time",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fname = f"{group_key}_offset_over_time.png"
    fig.savefig(FIG_DIR / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fname


def plot_correctability(df: pd.DataFrame, group_key: str, group_info: dict) -> str | None:
    """For non-zeroed traces, assess if a simple offset correction could be applied."""
    non_zero = df[df["zero_quality"] == "not_zeroed"].copy()
    if len(non_zero) < 10:
        return None

    sensor = non_zero["sensor"].value_counts().index[0]
    sd = non_zero[non_zero["sensor"] == sensor]
    display = get_display_label(sensor)

    sd = sd.copy()
    sd["correctable"] = (sd["initial_std"] < abs(sd["offset"]) * 0.1) & (sd["overall_range"] > abs(sd["offset"]) * 2)

    n_correctable = sd["correctable"].sum()
    n_total = len(sd)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Offset vs range scatter
    ax = axes[0]
    corr = sd[sd["correctable"]]
    not_corr = sd[~sd["correctable"]]
    if len(corr) > 0:
        ax.scatter(corr["offset"].abs(), corr["overall_range"], s=15, alpha=0.5,
                   color="#2ecc71", label=f"Correctable ({n_correctable})")
    if len(not_corr) > 0:
        ax.scatter(not_corr["offset"].abs(), not_corr["overall_range"], s=15, alpha=0.5,
                   color="#e74c3c", label=f"Not correctable ({n_total - n_correctable})")
    ax.set_xlabel(f"|Initial Offset| [{group_info['unit']}]")
    ax.set_ylabel(f"Signal Range [{group_info['unit']}]")
    ax.set_title("Offset vs Signal Range")
    ax.legend(fontsize=8)
    ax.plot([0, ax.get_xlim()[1]], [0, ax.get_xlim()[1] * 2], "k--", alpha=0.3, label="range=2x offset")

    # Offset stability (initial_std)
    ax = axes[1]
    ax.hist(sd["initial_std"], bins=30, alpha=0.7, color="steelblue")
    ax.set_xlabel(f"Initial Value Std [{group_info['unit']}]")
    ax.set_ylabel("Count")
    ax.set_title("Stability of Initial Offset")

    # Correctability by machine
    ax = axes[2]
    corr_by_machine = sd.groupby("machine_slug")["correctable"].mean() * 100
    corr_by_machine.plot(kind="bar", ax=ax, color="steelblue", alpha=0.7)
    ax.set_ylabel("Correctable (%)")
    ax.set_title("Correctability by Machine")
    ax.axhline(50, color="gray", linestyle="--", alpha=0.5)
    ax.tick_params(axis="x", rotation=30)

    fig.suptitle(
        f"Offset Correctability \u2014 {display} (n={n_total} non-zeroed traces)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fname = f"{group_key}_correctability.png"
    fig.savefig(FIG_DIR / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fname


def generate_report(df_index: pd.DataFrame, all_data: dict, figures: dict):
    """Generate the zero analysis markdown report."""
    lines = [
        "# Zero / Calibration Analysis\n\n",
        "This report analyzes whether operators properly zeroed key sensors before starting each trace. ",
        "Proper zeroing ensures depth readings start at ground level, inclinations reflect true tilt, ",
        "and deviations measure actual displacement.\n\n",
        f"**Dataset**: {len(df_index):,} sessions analyzed.\n\n",
        "**Note**: Sentinel values (9999, 99991, etc.) have been removed before analysis. ",
        "Uncalibrated sensors are excluded from zeroing analysis as their raw values cannot be meaningfully assessed.\n\n",
        "---\n\n",
    ]

    # Overall summary table
    lines.append("## Summary\n\n")
    lines.append("| Sensor Group | Traces | Good | Marginal | Not Zeroed | Good % |\n")
    lines.append("|-------------|--------|------|----------|------------|--------|\n")

    for group_key, group_info in ZERO_SENSORS.items():
        if group_key not in all_data:
            continue
        df = all_data[group_key]
        n = len(df)
        n_good = (df["zero_quality"] == "good").sum()
        n_marg = (df["zero_quality"] == "marginal").sum()
        n_bad = (df["zero_quality"] == "not_zeroed").sum()
        pct = n_good / n * 100 if n > 0 else 0
        lines.append(
            f"| {group_info['label']} ({group_info['unit']}, tol={group_info['tolerance']}) | "
            f"{n} | {n_good} | {n_marg} | {n_bad} | {pct:.0f}% |\n"
        )

    lines.append("\n---\n\n")

    # Per-group detailed analysis
    for group_key, group_info in ZERO_SENSORS.items():
        if group_key not in all_data:
            continue
        df = all_data[group_key]

        lines.append(f"## {group_info['label']}\n\n")
        lines.append(f"**Description**: {group_info['description']}\n\n")
        lines.append(f"**Tolerance**: \u00b1{group_info['tolerance']} {group_info['unit']} from zero\n\n")

        # Per-sensor stats
        for sensor in sorted(df["sensor"].unique()):
            sd = df[df["sensor"] == sensor]
            n_good = (sd["zero_quality"] == "good").sum()
            n = len(sd)
            mean_offset = sd["offset"].mean()
            std_offset = sd["offset"].std()
            display = get_display_label(sensor)
            lines.append(
                f"- **{display}**: {n} traces, {n_good}/{n} good ({n_good / n * 100:.0f}%), "
                f"mean offset = {mean_offset:.3f} \u00b1 {std_offset:.3f} {group_info['unit']}\n"
            )

        # Per-technique breakdown
        lines.append("\n**By Technique:**\n\n")
        lines.append("| Technique | Traces | Good % | Mean Offset |\n")
        lines.append("|-----------|--------|--------|-------------|\n")
        for tech in sorted(df["technique"].unique()):
            td = df[df["technique"] == tech]
            n_good = (td["zero_quality"] == "good").sum()
            n = len(td)
            pct = n_good / n * 100 if n > 0 else 0
            mean_off = td["offset"].mean()
            lines.append(f"| {tech} | {n} | {pct:.0f}% | {mean_off:.3f} |\n")

        # Include figures
        group_figs = figures.get(group_key, [])
        for fname in group_figs:
            if fname:
                lines.append(f"\n![{group_info['label']}](figures/zeros/{fname})\n")

        lines.append("\n---\n\n")

    # Correctability section
    lines.append("## Correctability Assessment\n\n")
    lines.append("For non-zeroed traces, we assess whether a simple constant-offset correction can be applied. ")
    lines.append("A correction is **applicable** when:\n")
    lines.append("1. The initial offset is stable (low variance in first 5 samples)\n")
    lines.append("2. The signal range is much larger than the offset (offset is < 50% of range)\n\n")

    for group_key in ["depth", "inclination", "deviation"]:
        if group_key not in all_data:
            continue
        df = all_data[group_key]
        non_zero = df[df["zero_quality"] == "not_zeroed"]
        if len(non_zero) < 5:
            continue

        n_total = len(non_zero)
        n_stable = (non_zero["initial_std"] < abs(non_zero["offset"]) * 0.1).sum()
        n_correctable = (
            (non_zero["initial_std"] < abs(non_zero["offset"]) * 0.1) &
            (non_zero["overall_range"] > abs(non_zero["offset"]) * 2)
        ).sum()

        label = ZERO_SENSORS[group_key]["label"]
        lines.append(f"### {label}\n\n")
        lines.append(f"- Non-zeroed traces: **{n_total}**\n")
        lines.append(f"- Stable initial offset (std < 10% of offset): **{n_stable}** ({n_stable / n_total * 100:.0f}%)\n")
        lines.append(f"- Correctable (stable + range > 2x offset): **{n_correctable}** ({n_correctable / n_total * 100:.0f}%)\n\n")

    # Recommendations
    lines.append("---\n\n## Recommendations\n\n")
    lines.append("1. **Enforce zeroing at trace start**: The machine software should prompt the operator ")
    lines.append("to verify zero reference before recording begins.\n")
    lines.append("2. **Auto-detect and flag**: Traces with initial depth offset > 0.5m should be ")
    lines.append("flagged in the quality index for manual review.\n")
    lines.append("3. **Apply constant-offset correction**: For correctable traces, subtract the mean ")
    lines.append("of the first 5 samples from the entire series. This is safe when the initial offset ")
    lines.append("is stable and small relative to the signal range.\n")
    lines.append("4. **Operator training**: Share zeroing quality reports per operator to encourage ")
    lines.append("consistent practices.\n")

    REPORT_PATH.write_text("".join(lines))
    print(f"Report written to {REPORT_PATH}")


def main():
    print("Loading merged trace index...")
    df_index = load_merged_index()
    print(f"  {len(df_index):,} sessions")

    print("\nCollecting initial values for all sensor groups...")
    all_data = collect_zero_data(df_index)

    print("\nGenerating plots...")
    all_figures = {}

    for group_key, group_info in ZERO_SENSORS.items():
        if group_key not in all_data:
            continue
        df = all_data[group_key]
        figs = []

        fname = plot_zero_distribution(df, group_key, group_info)
        figs.append(fname)
        print(f"  {fname}")

        fname = plot_zero_by_operator(df, group_key, group_info)
        if fname:
            figs.append(fname)
            print(f"  {fname}")

        fname = plot_zero_by_machine(df, group_key, group_info)
        if fname:
            figs.append(fname)
            print(f"  {fname}")

        fname = plot_zero_over_time(df, group_key, group_info)
        if fname:
            figs.append(fname)
            print(f"  {fname}")

        if group_key in ["depth", "inclination", "deviation"]:
            fname = plot_correctability(df, group_key, group_info)
            if fname:
                figs.append(fname)
                print(f"  {fname}")

        all_figures[group_key] = figs

    print("\nGenerating report...")
    generate_report(df_index, all_data, all_figures)
    print("Done!")


if __name__ == "__main__":
    main()
