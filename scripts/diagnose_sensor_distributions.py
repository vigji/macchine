"""Sensor distribution diagnostics across all (site, machine, technique, format) combos.

Produces visual and tabular outputs to detect calibration/divisor issues:
- sensor_stats.csv: percentile stats for every (combo, sensor) pair
- per_sensor/{sensor}.png: horizontal box plots across all combos
- dat_vs_json/{machine}_{sensor}.png: side-by-side DAT vs JSON where divergent
- anomaly_heatmap.png: sensors × combos colored by log10(p95/range_max)
- flags_report.txt: text summary of flagged anomalies

Usage:
    python scripts/diagnose_sensor_distributions.py
    python scripts/diagnose_sensor_distributions.py --max-traces 5
    python scripts/diagnose_sensor_distributions.py --sensors depth,torque
    python scripts/diagnose_sensor_distributions.py --output-dir /tmp/diag
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from macchine.harmonize.calibration import (
    PHYSICAL_RANGES,
    SENSOR_UNITS,
    clean_and_flag_df,
    get_sentinel_values,
    qc_summary,
    QC_SENTINEL,
    QC_OUT_OF_RANGE,
)
from macchine.storage.catalog import get_merged_trace_index

_FIG_DPI = 120
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_DIR = _PROJECT_ROOT / "output"
_TRACES_DIR = _OUTPUT_DIR / "traces"

# Maximum raw values to keep per (combo, sensor) for box plots
_MAX_VALUES_PER_COMBO = 2000
# Sensors to always skip
_SKIP_SENSORS = {"timestamp", "device_status", "inclination_valid"}


# ── 1. Data collection ────────────────────────────────────────────────────────

def collect_stats(
    index_df: pd.DataFrame,
    max_traces: int = 10,
    sensor_filter: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    """Load sampled traces and compute per-(combo, sensor) percentile stats.

    Uses clean_and_flag_df for sentinel detection — sentinels are removed
    and quality flags (sentinel / out-of-range counts) are tracked per combo.

    Returns:
        stats_df: DataFrame with one row per (combo, sensor)
        raw_values: {(combo_label, sensor): np.ndarray} for box plots
    """
    # Group by (site_id, machine_slug, technique, format)
    groups = index_df.groupby(["site_id", "machine_slug", "technique", "format"])
    print(f"Found {len(groups)} combos across {len(index_df)} sessions")

    # {(combo_label, sensor_name): list[float]}
    raw_collector: dict[tuple[str, str], list[float]] = defaultdict(list)
    # {(combo_label, sensor_name): {"sentinel": int, "out_of_range": int, "total": int}}
    qc_collector: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"sentinel": 0, "out_of_range": 0, "total": 0}
    )
    rows = []

    for (site_id, machine_slug, technique, fmt), group in groups:
        combo_label = f"{site_id}/{machine_slug}/{technique}/{fmt}"
        sampled = group.head(max_traces)

        for _, row in sampled.iterrows():
            # Resolve individual trace files
            trace_ids_raw = row.get("trace_ids", "")
            if isinstance(trace_ids_raw, str) and trace_ids_raw:
                trace_ids = trace_ids_raw.split("|")
            else:
                trace_ids = [row["trace_id"]]

            slug = str(machine_slug) if machine_slug and machine_slug != "unidentified" else "unknown"

            for tid in trace_ids:
                path = _TRACES_DIR / str(site_id) / slug / f"{tid}.parquet"
                if not path.exists():
                    continue
                try:
                    df = pd.read_parquet(path)
                except Exception:
                    continue

                # Clean sentinels and get quality flags in one pass
                cleaned, qc_flags = clean_and_flag_df(df, machine_slug=str(machine_slug))

                for col in cleaned.select_dtypes(include=[np.number]).columns:
                    if col in _SKIP_SENSORS:
                        continue
                    if sensor_filter and col not in sensor_filter:
                        continue

                    key = (combo_label, col)

                    # Accumulate QC counts
                    if col in qc_flags.columns:
                        qc_col = qc_flags[col]
                        qc_collector[key]["sentinel"] += int((qc_col == QC_SENTINEL).sum())
                        qc_collector[key]["out_of_range"] += int((qc_col == QC_OUT_OF_RANGE).sum())
                        qc_collector[key]["total"] += len(qc_col)

                    # Cap values per combo-sensor
                    if len(raw_collector[key]) >= _MAX_VALUES_PER_COMBO:
                        continue

                    vals = cleaned[col].dropna()
                    if vals.empty:
                        continue

                    remaining = _MAX_VALUES_PER_COMBO - len(raw_collector[key])
                    raw_collector[key].extend(vals.iloc[:remaining].tolist())

    # Compute stats from collected values
    for (combo_label, sensor), values_list in raw_collector.items():
        arr = np.array(values_list)
        if len(arr) < 5:
            continue

        parts = combo_label.split("/")
        site_id, machine_slug, technique, fmt = parts[0], parts[1], parts[2], parts[3]

        pct_zero = 100 * np.sum(arr == 0) / len(arr)
        pct_integer = 100 * np.sum(arr == np.floor(arr)) / len(arr)

        # QC stats for this combo-sensor
        qc = qc_collector.get((combo_label, sensor), {"sentinel": 0, "out_of_range": 0, "total": 0})
        total = qc["total"] or 1

        rows.append({
            "combo": combo_label,
            "site_id": site_id,
            "machine_slug": machine_slug,
            "technique": technique,
            "format": fmt,
            "sensor": sensor,
            "min": float(np.min(arr)),
            "p5": float(np.percentile(arr, 5)),
            "p25": float(np.percentile(arr, 25)),
            "median": float(np.median(arr)),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(np.max(arr)),
            "pct_zero": round(pct_zero, 1),
            "pct_integer": round(pct_integer, 1),
            "n_unique": int(len(np.unique(arr))),
            "n_values": len(arr),
            "n_sentinel": qc["sentinel"],
            "pct_sentinel": round(100 * qc["sentinel"] / total, 1),
            "n_out_of_range": qc["out_of_range"],
            "pct_out_of_range": round(100 * qc["out_of_range"] / total, 1),
        })

    stats_df = pd.DataFrame(rows)

    # Convert raw_collector to numpy arrays
    raw_values = {k: np.array(v) for k, v in raw_collector.items() if len(v) >= 5}

    print(f"Collected stats for {len(stats_df)} (combo, sensor) pairs "
          f"across {len(stats_df['sensor'].unique()) if not stats_df.empty else 0} sensors")

    return stats_df, raw_values


# ── 2. Per-sensor box plots ──────────────────────────────────────────────────

def plot_per_sensor(
    stats_df: pd.DataFrame,
    raw_values: dict[tuple[str, str], np.ndarray],
    out_dir: Path,
) -> None:
    """Horizontal box plots for each sensor across all combos."""
    sensor_dir = out_dir / "per_sensor"
    sensor_dir.mkdir(parents=True, exist_ok=True)

    sensors = sorted(stats_df["sensor"].unique())
    print(f"Generating {len(sensors)} per-sensor box plots...")

    for sensor in sensors:
        sensor_rows = stats_df[stats_df["sensor"] == sensor].sort_values("combo")
        if sensor_rows.empty:
            continue

        combos = sensor_rows["combo"].tolist()
        box_data = []
        for combo in combos:
            key = (combo, sensor)
            if key in raw_values:
                box_data.append(raw_values[key])
            else:
                box_data.append(np.array([]))

        n = len(combos)
        fig_height = max(3, n * 0.4 + 1)
        fig, ax = plt.subplots(figsize=(10, fig_height))

        # Use bxp with precomputed stats for efficiency, but raw data gives real whiskers
        bp = ax.boxplot(
            box_data,
            vert=False,
            patch_artist=True,
            widths=0.6,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.5),
        )

        # Color by format
        for i, combo in enumerate(combos):
            fmt = combo.split("/")[-1]
            color = "#aec7e8" if fmt == "json" else "#ffbb78"
            bp["boxes"][i].set_facecolor(color)
            bp["boxes"][i].set_alpha(0.7)

        ax.set_yticks(range(1, n + 1))
        ax.set_yticklabels(combos, fontsize=7)

        # Physical range reference lines
        if sensor in PHYSICAL_RANGES:
            lo, hi = PHYSICAL_RANGES[sensor]
            ax.axvline(lo, color="green", linestyle="--", alpha=0.5, linewidth=1)
            ax.axvline(hi, color="red", linestyle="--", alpha=0.5, linewidth=1)
            ax.axvspan(lo, hi, alpha=0.05, color="green")

        unit = SENSOR_UNITS.get(sensor, "")
        unit_str = f" [{unit}]" if unit else ""
        ax.set_xlabel(f"{sensor.replace('_', ' ').title()}{unit_str}")
        ax.set_title(f"{sensor} — distribution across combos")

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#aec7e8", alpha=0.7, label="JSON"),
            Patch(facecolor="#ffbb78", alpha=0.7, label="DAT"),
        ]
        if sensor in PHYSICAL_RANGES:
            legend_elements.append(Patch(facecolor="green", alpha=0.15, label="Physical range"))
        ax.legend(handles=legend_elements, fontsize=7, loc="lower right")

        fig.tight_layout()
        fig.savefig(sensor_dir / f"{sensor}.png", dpi=_FIG_DPI)
        plt.close(fig)


# ── 3. DAT vs JSON comparison plots ─────────────────────────────────────────

def plot_dat_vs_json(
    stats_df: pd.DataFrame,
    raw_values: dict[tuple[str, str], np.ndarray],
    out_dir: Path,
) -> None:
    """Side-by-side box plots for DAT vs JSON where median ratio diverges."""
    dvj_dir = out_dir / "dat_vs_json"
    dvj_dir.mkdir(parents=True, exist_ok=True)

    if stats_df.empty:
        return

    # Find machines with both formats
    machine_formats = stats_df.groupby("machine_slug")["format"].apply(set)
    dual_machines = [m for m, fmts in machine_formats.items() if "dat" in fmts and "json" in fmts]

    if not dual_machines:
        print("No machines with both DAT and JSON formats — skipping DAT vs JSON plots")
        return

    n_plots = 0
    for machine in dual_machines:
        machine_stats = stats_df[stats_df["machine_slug"] == machine]
        sensors_by_format = machine_stats.groupby(["sensor", "format"])["median"].first().unstack("format")

        for sensor in sensors_by_format.index:
            if "dat" not in sensors_by_format.columns or "json" not in sensors_by_format.columns:
                continue
            dat_med = sensors_by_format.loc[sensor, "dat"]
            json_med = sensors_by_format.loc[sensor, "json"]
            if pd.isna(dat_med) or pd.isna(json_med):
                continue
            if json_med == 0 and dat_med == 0:
                continue

            # Compute ratio — skip if within 1.5x
            if json_med != 0:
                ratio = abs(dat_med / json_med)
            elif dat_med != 0:
                ratio = float("inf")
            else:
                continue

            if 0.67 <= ratio <= 1.5:
                continue

            # Collect box data for this machine+sensor, split by format
            dat_data = []
            json_data = []
            dat_labels = []
            json_labels = []

            for _, row in machine_stats[machine_stats["sensor"] == sensor].iterrows():
                key = (row["combo"], sensor)
                if key not in raw_values:
                    continue
                if row["format"] == "dat":
                    dat_data.append(raw_values[key])
                    dat_labels.append(row["combo"])
                else:
                    json_data.append(raw_values[key])
                    json_labels.append(row["combo"])

            if not dat_data or not json_data:
                continue

            all_data = json_data + dat_data
            all_labels = [f"JSON: {l}" for l in json_labels] + [f"DAT: {l}" for l in dat_labels]
            n = len(all_data)

            fig_height = max(3, n * 0.5 + 1.5)
            fig, ax = plt.subplots(figsize=(10, fig_height))

            bp = ax.boxplot(
                all_data,
                vert=False,
                patch_artist=True,
                widths=0.6,
                showfliers=False,
                medianprops=dict(color="black", linewidth=1.5),
            )

            for i in range(len(all_data)):
                color = "#aec7e8" if i < len(json_data) else "#ffbb78"
                bp["boxes"][i].set_facecolor(color)
                bp["boxes"][i].set_alpha(0.7)

            ax.set_yticks(range(1, n + 1))
            ax.set_yticklabels(all_labels, fontsize=7)

            if sensor in PHYSICAL_RANGES:
                lo, hi = PHYSICAL_RANGES[sensor]
                ax.axvline(lo, color="green", linestyle="--", alpha=0.5)
                ax.axvline(hi, color="red", linestyle="--", alpha=0.5)

            unit = SENSOR_UNITS.get(sensor, "")
            unit_str = f" [{unit}]" if unit else ""
            ax.set_xlabel(f"{sensor.replace('_', ' ').title()}{unit_str}")
            ax.set_title(f"{machine} — {sensor} — DAT vs JSON (ratio={ratio:.1f}×)")

            fig.tight_layout()
            fig.savefig(dvj_dir / f"{machine}_{sensor}.png", dpi=_FIG_DPI)
            plt.close(fig)
            n_plots += 1

    print(f"Generated {n_plots} DAT-vs-JSON comparison plots")


# ── 4. Anomaly heatmap ──────────────────────────────────────────────────────

def plot_anomaly_heatmap(stats_df: pd.DataFrame, out_dir: Path) -> None:
    """Heatmap: sensors × combos, colored by log10(p95/range_max)."""
    if stats_df.empty:
        return

    # Only include sensors with known physical ranges
    sensors_with_range = sorted(
        s for s in stats_df["sensor"].unique() if s in PHYSICAL_RANGES
    )
    if not sensors_with_range:
        print("No sensors with physical ranges — skipping anomaly heatmap")
        return

    combos = sorted(stats_df["combo"].unique())

    # Build matrix: rows=sensors, cols=combos
    matrix = np.full((len(sensors_with_range), len(combos)), np.nan)
    sensor_idx = {s: i for i, s in enumerate(sensors_with_range)}
    combo_idx = {c: i for i, c in enumerate(combos)}

    for _, row in stats_df.iterrows():
        sensor = row["sensor"]
        if sensor not in sensor_idx:
            continue
        combo = row["combo"]
        p95_abs = max(abs(row["p95"]), abs(row["p5"]))
        lo, hi = PHYSICAL_RANGES[sensor]
        range_max = max(abs(lo), abs(hi))
        if range_max == 0:
            continue

        score = np.log10(max(p95_abs / range_max, 1e-6))
        matrix[sensor_idx[sensor], combo_idx[combo]] = score

    fig_width = max(8, len(combos) * 0.5 + 3)
    fig_height = max(6, len(sensors_with_range) * 0.3 + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # TwoSlopeNorm: green around 0, white at 0, red at +2
    norm = mcolors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
    cmap = plt.cm.RdYlGn_r  # green (low) → yellow → red (high)

    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    ax.set_xticks(range(len(combos)))
    ax.set_xticklabels(combos, rotation=90, fontsize=5, ha="center")
    ax.set_yticks(range(len(sensors_with_range)))
    ax.set_yticklabels(sensors_with_range, fontsize=6)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("log₁₀(p95 / range_max)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title("Anomaly heatmap — sensor range violations", fontsize=10)
    # Mark NaN cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isnan(matrix[i, j]):
                ax.plot(j, i, marker="x", color="lightgray", markersize=3)

    fig.tight_layout()
    fig.savefig(out_dir / "anomaly_heatmap.png", dpi=_FIG_DPI)
    plt.close(fig)
    print("Generated anomaly heatmap")


# ── 5. Flags report ─────────────────────────────────────────────────────────

def generate_flags_report(stats_df: pd.DataFrame, out_dir: Path) -> None:
    """Write text summary of flagged anomalies."""
    if stats_df.empty:
        (out_dir / "flags_report.txt").write_text("No data collected.\n")
        return

    lines = [
        "SENSOR DISTRIBUTION FLAGS REPORT",
        "=" * 80,
        "",
    ]

    # ── Range violations ──
    lines.append("1. RANGE VIOLATIONS (p95 > 2× physical range max)")
    lines.append("-" * 60)
    range_flags = []
    for _, row in stats_df.iterrows():
        sensor = row["sensor"]
        if sensor not in PHYSICAL_RANGES:
            continue
        lo, hi = PHYSICAL_RANGES[sensor]
        range_max = max(abs(lo), abs(hi))
        p95_abs = max(abs(row["p95"]), abs(row["p5"]))
        if p95_abs > range_max * 2:
            # Suggest divisor
            suggested = None
            for div in [10, 100, 1000]:
                if p95_abs / div <= range_max:
                    suggested = div
                    break
            range_flags.append((row, p95_abs, range_max, suggested))

    if range_flags:
        for row, p95_abs, range_max, suggested in sorted(
            range_flags, key=lambda x: -x[1] / x[2]
        ):
            ratio = p95_abs / range_max
            div_str = f"  → suggested divisor: {suggested}" if suggested else ""
            lines.append(
                f"  {row['combo']:50s} {row['sensor']:30s} "
                f"p95={row['p95']:>10.1f}  range_max={range_max:>8.0f}  "
                f"ratio={ratio:>6.1f}x{div_str}"
            )
    else:
        lines.append("  (none)")
    lines.append("")

    # ── DAT/JSON inconsistencies ──
    lines.append("2. DAT/JSON INCONSISTENCIES (median ratio > 3× or < 0.33×)")
    lines.append("-" * 60)
    djson_flags = []

    if not stats_df.empty:
        # Group by machine+sensor, compare formats
        for (machine, sensor), grp in stats_df.groupby(["machine_slug", "sensor"]):
            formats_present = set(grp["format"])
            if "dat" not in formats_present or "json" not in formats_present:
                continue
            dat_med = grp[grp["format"] == "dat"]["median"].median()
            json_med = grp[grp["format"] == "json"]["median"].median()
            if pd.isna(dat_med) or pd.isna(json_med):
                continue
            if json_med == 0 and dat_med == 0:
                continue
            if json_med != 0:
                ratio = dat_med / json_med
            elif dat_med != 0:
                ratio = float("inf")
            else:
                continue
            if abs(ratio) > 3 or (abs(ratio) < 0.33 and abs(ratio) > 0):
                djson_flags.append((machine, sensor, json_med, dat_med, ratio))

    if djson_flags:
        for machine, sensor, json_med, dat_med, ratio in sorted(djson_flags):
            lines.append(
                f"  {machine:20s} {sensor:30s} "
                f"JSON_median={json_med:>10.2f}  DAT_median={dat_med:>10.2f}  "
                f"ratio={ratio:>8.1f}x"
            )
    else:
        lines.append("  (none)")
    lines.append("")

    # ── Raw ADC counts ──
    lines.append("3. SUSPECTED RAW ADC COUNTS (100% integer + out of physical range)")
    lines.append("-" * 60)
    adc_flags = []
    for _, row in stats_df.iterrows():
        sensor = row["sensor"]
        if row["pct_integer"] < 99.9:
            continue
        if sensor not in PHYSICAL_RANGES:
            continue
        lo, hi = PHYSICAL_RANGES[sensor]
        range_max = max(abs(lo), abs(hi))
        p95_abs = max(abs(row["p95"]), abs(row["p5"]))
        if p95_abs > range_max * 2:
            adc_flags.append(row)

    if adc_flags:
        for row in sorted(adc_flags, key=lambda r: (r["combo"], r["sensor"])):
            lines.append(
                f"  {row['combo']:50s} {row['sensor']:30s} "
                f"range=[{row['min']:.0f}, {row['max']:.0f}]  "
                f"pct_int={row['pct_integer']:.0f}%  n_unique={row['n_unique']}"
            )
    else:
        lines.append("  (none)")
    lines.append("")

    # ── Sentinel contamination ──
    has_qc_cols = "n_sentinel" in stats_df.columns
    sentinel_flags = []
    if has_qc_cols:
        lines.append("4. SENTINEL CONTAMINATION (>5% sentinel values detected)")
        lines.append("-" * 60)
        for _, row in stats_df.iterrows():
            if row.get("pct_sentinel", 0) > 5:
                sentinel_flags.append(row)

        if sentinel_flags:
            for row in sorted(sentinel_flags, key=lambda r: -r["pct_sentinel"]):
                oor_str = ""
                if row.get("pct_out_of_range", 0) > 0:
                    oor_str = f"  OOR={row['pct_out_of_range']:.0f}%"
                lines.append(
                    f"  {row['combo']:50s} {row['sensor']:30s} "
                    f"sentinel={row['pct_sentinel']:>5.1f}%  "
                    f"({row['n_sentinel']} of {row['n_sentinel'] + row['n_values']}){oor_str}"
                )
        else:
            lines.append("  (none)")
        lines.append("")

    # ── Summary ──
    lines.append("=" * 80)
    sentinel_summary = f", {len(sentinel_flags)} sentinel-contaminated" if has_qc_cols else ""
    lines.append(
        f"SUMMARY: {len(range_flags)} range violations, "
        f"{len(djson_flags)} DAT/JSON mismatches, "
        f"{len(adc_flags)} raw ADC suspects{sentinel_summary}"
    )
    lines.append(
        f"Total (combo, sensor) pairs analyzed: {len(stats_df)}"
    )

    report_text = "\n".join(lines) + "\n"
    report_path = out_dir / "flags_report.txt"
    report_path.write_text(report_text)
    print(f"Wrote flags report: {len(range_flags)} range violations, "
          f"{len(djson_flags)} DAT/JSON mismatches, {len(adc_flags)} raw ADC suspects")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Diagnose sensor distributions across all data combos"
    )
    parser.add_argument(
        "--max-traces", type=int, default=10,
        help="Max traces to sample per combo (default: 10)",
    )
    parser.add_argument(
        "--sensors", type=str, default=None,
        help="Comma-separated sensor names to analyze (default: all)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: output/diagnostics)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else _OUTPUT_DIR / "diagnostics"

    # Clean previous outputs to avoid stale leftovers
    import shutil
    if out_dir.exists():
        print(f"Cleaning previous outputs in {out_dir}...")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sensor_filter = None
    if args.sensors:
        sensor_filter = set(args.sensors.split(","))
        print(f"Filtering to sensors: {sensor_filter}")

    # Load index
    print("Loading merged trace index...")
    index_df = get_merged_trace_index(_OUTPUT_DIR)
    print(f"  {len(index_df)} merged sessions")

    # Collect stats
    print("\nCollecting sensor distributions...")
    stats_df, raw_values = collect_stats(
        index_df, max_traces=args.max_traces, sensor_filter=sensor_filter
    )

    if stats_df.empty:
        print("ERROR: No data collected. Check that traces exist in output/traces/")
        sys.exit(1)

    # Save stats CSV
    csv_path = out_dir / "sensor_stats.csv"
    stats_df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path} ({len(stats_df)} rows)")

    # Generate plots
    print("\nGenerating per-sensor box plots...")
    plot_per_sensor(stats_df, raw_values, out_dir)

    print("\nGenerating DAT vs JSON comparison plots...")
    plot_dat_vs_json(stats_df, raw_values, out_dir)

    print("\nGenerating anomaly heatmap...")
    plot_anomaly_heatmap(stats_df, out_dir)

    print("\nGenerating flags report...")
    generate_flags_report(stats_df, out_dir)

    print(f"\nDone! All outputs in {out_dir}")


if __name__ == "__main__":
    main()
