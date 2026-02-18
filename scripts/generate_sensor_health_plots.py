"""Generate sensor health monitoring analysis.

For each machine, tracks key health-related sensors over time:
- Hydraulic pressure trends (pump pressures, gearbox oil pressures)
- Temperature trends (CUT machines)
- Oil condition (water content for CUT)
- Torque/force patterns
- Inclination consistency

Generates figures in reports/figures/sensor_health/ and report at reports/10_sensor_health.md.
Uses the merged trace index to avoid duplicate analysis.
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
import yaml
import warnings

from macchine.harmonize.calibration import (
    is_calibrated,
    get_unit,
    get_display_label,
    get_axis_label,
    clean_sentinels,
    clean_sentinels_df,
    get_sentinel_values,
)
from macchine.analysis.plot_utils import plot_with_gaps, add_site_markers

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

OUTPUT_DIR = Path("output")
FIG_DIR = Path("reports/figures/sensor_health")
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = Path("reports/10_sensor_health.md")

sns.set_theme(style="whitegrid", font_scale=1.0)

# Sensor groups for health monitoring
HEALTH_SENSORS = {
    "pump_pressure": {
        "sensors": ["Druck Pumpe 1", "Druck Pumpe 2", "Druck Pumpe 3", "Druck Pumpe 4", "Druck Pumpe"],
        "label": "Pump Pressure",
        "unit": "bar",
        "description": "Main hydraulic pump outlet pressures",
    },
    "cutter_pressure": {
        "sensors": ["Druck FRL", "Druck FRR"],
        "label": "Cutter Hydraulic Pressure",
        "unit": "bar",
        "description": "Left/right cutter hydraulic pressures (CUT only)",
    },
    "gearbox_oil_pressure": {
        "sensors": ["Oeldruck Getriebe rechts", "Oeldruck Getriebe links"],
        "label": "Gearbox Oil Pressure",
        "unit": "bar",
        "description": "Left/right gearbox bearing oil pressure",
    },
    "leakage_pressure": {
        "sensors": ["Leckagedruck Getriebe rechts", "Leckagedruck Getriebe links", "Leckoeldruck"],
        "label": "Leakage / Seal Pressure",
        "unit": "bar",
        "description": "Gearbox leakage + leak oil pressure — rising indicates seal wear",
    },
    "temperature": {
        "sensors": ["Temperatur FRL", "Temperatur FRR", "Temp. Verteilergetriebe"],
        "label": "Temperature",
        "unit": "\u00b0C",
        "description": "Cutter and gearbox temperatures",
    },
    "water_content": {
        "sensors": ["Wassergehalt Getriebeoel FRL", "Wassergehalt Getriebeoel FRR"],
        "label": "Water Content in Gearbox Oil",
        "unit": "%",
        "description": "Water ingress into gearbox oil — rising indicates seal degradation",
    },
    "torque": {
        "sensors": ["Drehmoment", "DrehmomentkNm", "DrehmomentProzent"],
        "label": "Torque",
        "unit": "mixed",
        "description": "Drilling/rotation torque",
    },
    "rope_force": {
        "sensors": ["Seilkraft Hauptwinde", "Seilkraft Hilfswinde", "Seilkraft Fr\u00e4swinde", "Seilkraft"],
        "label": "Rope / Winch Force",
        "unit": "t",
        "description": "Main and auxiliary winch rope forces",
    },
    "inclination": {
        "sensors": ["Neigung X", "Neigung Y", "Neigung X Mast", "Neigung Y Mast"],
        "label": "Inclination",
        "unit": "deg",
        "description": "Machine and mast tilt angles",
    },
    "kdk": {
        "sensors": ["KDK Druck", "KDK Drehzahl"],
        "label": "Rotary Drive (KDK)",
        "unit": "bar / rpm",
        "description": "Rotary drive pressure and speed",
    },
}


def load_merged_index() -> pd.DataFrame:
    path = OUTPUT_DIR / "metadata" / "merged_trace_index.parquet"
    df = pd.read_parquet(path)
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df = df.dropna(subset=["start_time"])
    df["machine_slug"] = df["machine_slug"].replace("", "unidentified")
    return df


def extract_trace_stats(trace_path: Path, sensors: list[str]) -> dict | None:
    """Read a trace parquet and extract summary statistics for specified sensors."""
    if not trace_path.exists():
        return None
    try:
        tdf = pd.read_parquet(trace_path)
    except Exception:
        return None

    # Clean sentinel values from all numeric columns
    tdf = clean_sentinels_df(tdf)

    available = [s for s in sensors if s in tdf.columns]
    if not available:
        return None

    stats = {}
    for s in available:
        vals = tdf[s].dropna()
        if len(vals) < 10:
            continue
        # Remove first 10 samples (startup transient) if trace is long enough
        if len(vals) > 30:
            vals_work = vals.iloc[10:]
        else:
            vals_work = vals
        stats[s] = {
            "mean": float(vals_work.mean()),
            "median": float(vals_work.median()),
            "std": float(vals_work.std()),
            "p5": float(vals_work.quantile(0.05)),
            "p25": float(vals_work.quantile(0.25)),
            "p75": float(vals_work.quantile(0.75)),
            "p95": float(vals_work.quantile(0.95)),
            "max": float(vals_work.max()),
            "min": float(vals_work.min()),
            "initial": float(vals.iloc[0]),
            "final": float(vals.iloc[-1]),
            "zero_frac": float((vals_work == 0).mean()),
            "n_samples": len(vals),
        }
    return stats if stats else None


def get_trace_path(row) -> Path:
    """Resolve the parquet trace file path from trace_id and metadata."""
    site = row["site_id"]
    slug = row["machine_slug"] if row["machine_slug"] != "unidentified" else "unknown"
    trace_id = row["trace_id"]
    return OUTPUT_DIR / "traces" / str(site) / slug / f"{trace_id}.parquet"


def collect_sensor_timelines(
    df_index: pd.DataFrame, sensor_group: dict, machine_filter: str | None = None
) -> pd.DataFrame:
    """Collect per-trace summary stats for a group of sensors across all traces."""
    sensors = sensor_group["sensors"]

    if machine_filter:
        df_sub = df_index[df_index["machine_slug"] == machine_filter]
    else:
        df_sub = df_index

    records = []
    for _, row in df_sub.iterrows():
        trace_path = get_trace_path(row)
        stats = extract_trace_stats(trace_path, sensors)
        if stats is None:
            continue
        for sensor_name, sensor_stats in stats.items():
            rec = {
                "trace_id": row["trace_id"],
                "start_time": row["start_time"],
                "site_id": row["site_id"],
                "machine_slug": row["machine_slug"],
                "technique": row["technique"],
                "element_name": row["element_name"],
                "duration_min": row["duration_min"],
                "sensor": sensor_name,
            }
            rec.update(sensor_stats)
            records.append(rec)

    return pd.DataFrame(records) if records else pd.DataFrame()


def _sensor_ylabel(sensor_name: str, machine_slug: str) -> str:
    """Build a y-axis label with English name and calibration-aware unit."""
    display = get_display_label(sensor_name)
    unit = get_unit(sensor_name, machine_slug)
    if unit:
        return f"{display}\n[{unit}]"
    return display


def plot_sensor_timeline(
    df_stats: pd.DataFrame,
    sensor_group: dict,
    machine_slug: str,
    site_id: str | None = None,
) -> str | None:
    """Plot sensor summary statistics over time for one machine (optionally one site)."""
    df_sub = df_stats[df_stats["machine_slug"] == machine_slug].copy()
    if site_id:
        df_sub = df_sub[df_sub["site_id"] == site_id]
    if df_sub.empty:
        return None

    df_sub = df_sub.sort_values("start_time")
    sensors_present = df_sub["sensor"].unique()

    n_sensors = len(sensors_present)
    if n_sensors == 0:
        return None

    fig, axes = plt.subplots(n_sensors, 1, figsize=(14, 3.5 * n_sensors), squeeze=False, sharex=True)

    for i, sensor in enumerate(sorted(sensors_present)):
        ax = axes[i, 0]
        sd = df_sub[df_sub["sensor"] == sensor].sort_values("start_time")

        # Color by site
        sites = sd["site_id"].unique()
        colors = sns.color_palette("Set2", max(len(sites), 3))
        site_color = {s: colors[j % len(colors)] for j, s in enumerate(sites)}

        for s in sites:
            ss = sd[sd["site_id"] == s]
            ax.scatter(ss["start_time"], ss["median"], s=12, alpha=0.4,
                       color=site_color[s], label=f"Site {s} (median)")
            # P5-P95 band
            ax.fill_between(ss["start_time"], ss["p5"], ss["p95"],
                            alpha=0.08, color=site_color[s])

        # Rolling trend (14-day window) — gap-aware to avoid interpolating across inactive periods
        if len(sd) > 15:
            sd_roll = sd.set_index("start_time").resample("7D").agg({"median": "mean", "p95": "mean"}).dropna()
            if len(sd_roll) > 2:
                plot_with_gaps(ax, sd_roll.index, sd_roll["median"],
                               max_gap_days=14, color="k", linestyle="-",
                               linewidth=1.5, alpha=0.7, label="7-day avg")
                plot_with_gaps(ax, sd_roll.index, sd_roll["p95"],
                               max_gap_days=14, color="r", linestyle="--",
                               linewidth=1.0, alpha=0.5, label="7-day P95 avg")

        # Site boundary markers
        add_site_markers(ax, sd, site_col="site_id", time_col="start_time")

        # Use English label with calibration-aware unit
        ylabel = _sensor_ylabel(sensor, machine_slug)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        # Add visual warning if uncalibrated
        if not is_calibrated(sensor, machine_slug):
            ax.set_facecolor("#fff8f0")
            ax.text(0.01, 0.95, "UNCALIBRATED", transform=ax.transAxes,
                    fontsize=8, color="#cc6600", fontweight="bold", va="top", alpha=0.7)

    axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[-1, 0].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45, ha="right")

    title_suffix = f" \u2014 Site {site_id}" if site_id else ""
    fig.suptitle(
        f"{sensor_group['label']} Timeline \u2014 {machine_slug}{title_suffix}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    safe_label = sensor_group["label"].replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
    fname = f"{machine_slug}_{safe_label}"
    if site_id:
        fname += f"_{site_id}"
    fname += ".png"
    fig.savefig(FIG_DIR / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fname


def plot_sensor_boxplot_by_site(
    df_stats: pd.DataFrame, sensor_group: dict, machine_slug: str
) -> str | None:
    """Boxplot comparison of sensor values across sites for one machine."""
    df_sub = df_stats[df_stats["machine_slug"] == machine_slug].copy()
    if df_sub.empty or df_sub["site_id"].nunique() < 2:
        return None

    sensors_present = df_sub["sensor"].unique()
    n = len(sensors_present)
    if n == 0:
        return None

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)

    for i, sensor in enumerate(sorted(sensors_present)):
        ax = axes[0, i]
        sd = df_sub[df_sub["sensor"] == sensor]
        if len(sd) < 5:
            ax.set_visible(False)
            continue
        sns.boxplot(data=sd, x="site_id", y="median", hue="technique",
                    ax=ax, showfliers=False)
        ax.set_title(get_display_label(sensor), fontsize=10)
        unit = get_unit(sensor, machine_slug)
        ax.set_ylabel(f"[{unit}]" if unit else "")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)

        if not is_calibrated(sensor, machine_slug):
            ax.set_facecolor("#fff8f0")

    fig.suptitle(
        f"{sensor_group['label']} by Site \u2014 {machine_slug}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    safe_label = sensor_group["label"].replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
    fname = f"{machine_slug}_{safe_label}_by_site.png"
    fig.savefig(FIG_DIR / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fname


def plot_cross_machine_comparison(df_stats: pd.DataFrame, sensor_group: dict) -> list[str]:
    """Compare sensor distributions across machines, grouped by technique.

    Only compares machines that use the same technique to avoid misleading
    cross-technique comparisons (e.g. CUT pump pressure vs KELLY pump pressure).
    Returns a list of generated filenames (one per technique with >=2 machines).
    """
    if df_stats.empty:
        return []

    sensors_present = sorted(df_stats["sensor"].unique())
    if not sensors_present:
        return []

    # Pick the most common sensor for the overview
    sensor = df_stats["sensor"].value_counts().index[0]
    sd = df_stats[df_stats["sensor"] == sensor]
    if len(sd) < 10:
        return []

    fnames = []
    techniques = sorted(sd["technique"].unique())

    for technique in techniques:
        sd_tech = sd[sd["technique"] == technique]
        machines = sorted(sd_tech["machine_slug"].unique())
        if len(machines) < 2:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        cal_machines = [m for m in machines if is_calibrated(sensor, m)]
        uncal_machines = [m for m in machines if not is_calibrated(sensor, m)]

        # Box plot by machine (within technique)
        sns.boxplot(data=sd_tech, x="machine_slug", y="median",
                    ax=axes[0], showfliers=False)
        display_name = get_display_label(sensor)
        axes[0].set_title(f"{display_name} \u2014 Median per Trace", fontsize=11)

        if uncal_machines and cal_machines:
            axes[0].set_ylabel(f"{sensor_group['unit']} (cal.) / arb. units (uncal.)")
        elif uncal_machines:
            axes[0].set_ylabel("[arb. units]")
        else:
            axes[0].set_ylabel(f"[{sensor_group['unit']}]")
        axes[0].tick_params(axis="x", rotation=30)

        for label in axes[0].get_xticklabels():
            if label.get_text() in uncal_machines:
                label.set_color("#cc6600")

        # P95 over time for machines in this technique
        for m in machines:
            ms = sd_tech[sd_tech["machine_slug"] == m].sort_values("start_time")
            if len(ms) > 5:
                marker = "x" if not is_calibrated(sensor, m) else "o"
                axes[1].scatter(ms["start_time"], ms["p95"], s=10, alpha=0.3,
                                label=m, marker=marker)
        axes[1].set_title(f"{display_name} \u2014 P95 Over Time", fontsize=11)
        if uncal_machines and cal_machines:
            axes[1].set_ylabel(f"[{sensor_group['unit']}] / [arb. units]")
        elif uncal_machines:
            axes[1].set_ylabel("[arb. units]")
        else:
            axes[1].set_ylabel(f"[{sensor_group['unit']}]")
        axes[1].legend(fontsize=7, ncol=2)
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.xticks(rotation=45, ha="right")

        fig.suptitle(
            f"Cross-Machine ({technique}): {sensor_group['label']}\n"
            f"Note: values depend on ground conditions, pile depth, and site \u2014 "
            f"cross-site differences are expected.",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout()
        safe_label = sensor_group["label"].replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        fname = f"cross_machine_{technique}_{safe_label}.png"
        fig.savefig(FIG_DIR / fname, dpi=120, bbox_inches="tight")
        plt.close(fig)
        fnames.append(fname)

    return fnames


def plot_working_vs_idle_pressure(df_stats: pd.DataFrame) -> str | None:
    """Analyze pump pressure distributions: working vs idle states."""
    sensor_cols = [s for s in ["Druck Pumpe 1", "Druck Pumpe 2", "Druck Pumpe 3", "Druck Pumpe 4"]
                   if s in df_stats["sensor"].values]
    if not sensor_cols:
        return None

    sd = df_stats[df_stats["sensor"].isin(sensor_cols)].copy()
    if sd.empty:
        return None

    machines = sorted(sd["machine_slug"].unique())
    n_machines = len(machines)
    if n_machines < 1:
        return None

    fig, axes = plt.subplots(1, min(n_machines, 4), figsize=(5 * min(n_machines, 4), 5), squeeze=False)
    for i, m in enumerate(machines[:4]):
        ax = axes[0, i]
        ms = sd[sd["machine_slug"] == m].sort_values("start_time")
        for sensor in sensor_cols:
            ss = ms[ms["sensor"] == sensor]
            if len(ss) > 3:
                label = get_display_label(sensor)
                ax.scatter(ss["start_time"], ss["zero_frac"] * 100, s=10, alpha=0.4, label=label)
        ax.set_title(m, fontsize=10)
        ax.set_ylabel("Zero-pressure fraction (%)")
        ax.set_ylim(-5, 105)
        ax.legend(fontsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

    fig.suptitle("Pump Idle Fraction Over Time", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fname = "pump_idle_fraction.png"
    fig.savefig(FIG_DIR / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fname


def generate_report(
    df_index: pd.DataFrame, all_stats: dict[str, pd.DataFrame], figures: dict
):
    """Generate the markdown report."""
    lines = [
        "# Sensor Health Monitoring Report\n\n",
        "This report analyzes key machine health indicators extracted from time-series sensor data. ",
        "For each sensor group, we extract per-trace summary statistics (median, P5, P95, etc.) ",
        "and track them chronologically to detect trends, degradation, or anomalies.\n\n",
        f"**Dataset**: {len(df_index):,} sessions (after dedup + merge) across ",
        f"{df_index['machine_slug'].nunique()} machines and {df_index['site_id'].nunique()} sites.\n\n",
        "**Calibration note**: Sensors marked as *uncalibrated* report raw ADC counts or uncalibrated ",
        "scaling factors. Their absolute values are not in engineering units, but trends over time are ",
        "still meaningful. These are labelled with `[arb. units]` on axes and highlighted with an ",
        "orange background on plots.\n\n",
        "**Important caveats for cross-machine comparisons**: Sensor values (pressures, torques, forces) "
        "depend heavily on ground conditions, pile depth, and construction technique. Differences between "
        "sites are expected and do not necessarily indicate machine degradation. Cross-machine comparisons "
        "are only shown within the same technique to avoid misleading conclusions.\n\n",
        "All figures are in `reports/figures/sensor_health/`.\n\n",
        "---\n\n",
    ]

    # Machine overview
    lines.append("## Machine Sensor Coverage\n\n")
    lines.append("| Machine | Technique(s) | Sites | Sessions | Pump Pressure | Temperature | Oil Condition | Torque |\n")
    lines.append("|---------|-------------|-------|----------|---------------|-------------|---------------|--------|\n")
    for slug in sorted(df_index["machine_slug"].unique()):
        m_df = df_index[df_index["machine_slug"] == slug]
        techs = ", ".join(sorted(m_df["technique"].unique()))
        sites = ", ".join(sorted(m_df["site_id"].unique()))
        n = len(m_df)

        def has_sensor(group_key):
            if group_key not in all_stats:
                return False
            gs = all_stats[group_key]
            return not gs[gs["machine_slug"] == slug].empty

        pump = "YES" if has_sensor("pump_pressure") else "no"
        temp = "YES" if has_sensor("temperature") else "no"
        oil = "YES" if has_sensor("water_content") else "no"
        torque = "YES" if has_sensor("torque") else "no"
        lines.append(f"| {slug} | {techs} | {sites} | {n} | {pump} | {temp} | {oil} | {torque} |\n")

    lines.append("\n---\n\n")

    # Per-sensor-group analysis
    for group_key, group_info in HEALTH_SENSORS.items():
        if group_key not in all_stats or all_stats[group_key].empty:
            continue

        gs = all_stats[group_key]
        lines.append(f"## {group_info['label']}\n\n")
        lines.append(f"**Description**: {group_info['description']}\n\n")

        # Overall statistics table — with calibration column
        sensor_summary = gs.groupby("sensor").agg(
            n_traces=("trace_id", "count"),
            machines=("machine_slug", "nunique"),
            median_of_medians=("median", "median"),
            p95_of_medians=("median", lambda x: x.quantile(0.95)),
            avg_std=("std", "mean"),
        ).round(2)
        lines.append("\n| Sensor | English Name | Traces | Machines | Median | P95 | Avg Std | Calibration |\n")
        lines.append("|--------|-------------|--------|----------|--------|-----|---------|-------------|\n")
        for sensor_name, row in sensor_summary.iterrows():
            english = get_display_label(sensor_name)
            # Check calibration across machines that have this sensor
            machines_with = gs[gs["sensor"] == sensor_name]["machine_slug"].unique()
            cal_status = []
            for m in machines_with:
                if is_calibrated(sensor_name, m):
                    cal_status.append(f"{m}: calibrated")
                else:
                    cal_status.append(f"{m}: **uncalibrated**")
            all_cal = all(is_calibrated(sensor_name, m) for m in machines_with)
            unit = group_info["unit"] if all_cal else "mixed"
            cal_text = "All calibrated" if all_cal else "; ".join(cal_status)
            lines.append(
                f"| {sensor_name} | {english} | {int(row['n_traces'])} | {int(row['machines'])} | "
                f"{row['median_of_medians']:.1f} | {row['p95_of_medians']:.1f} | "
                f"{row['avg_std']:.1f} | {cal_text} |\n"
            )

        # Include generated figures
        group_figs = figures.get(group_key, [])
        for fname in group_figs:
            if fname:
                lines.append(f"\n![{group_info['label']}](figures/sensor_health/{fname})\n")

        lines.append("\n---\n\n")

    # Interpretation sections
    lines.append("## Key Findings\n\n")

    # Check for temperature trends
    if "temperature" in all_stats and not all_stats["temperature"].empty:
        temp_df = all_stats["temperature"]
        lines.append("### Temperature Monitoring (CUT Machines)\n\n")
        for m in sorted(temp_df["machine_slug"].unique()):
            ms = temp_df[temp_df["machine_slug"] == m].sort_values("start_time")
            for sensor in ms["sensor"].unique():
                ss = ms[ms["sensor"] == sensor]
                if len(ss) < 10:
                    continue
                cal_note = "" if is_calibrated(sensor, m) else " **[uncalibrated \u2014 values are raw]**"
                q1 = ss.iloc[:len(ss) // 4]["median"].mean()
                q4 = ss.iloc[-len(ss) // 4:]["median"].mean()
                trend = "RISING" if q4 > q1 * 1.1 else ("FALLING" if q4 < q1 * 0.9 else "STABLE")
                unit = get_unit(sensor, m)
                display = get_display_label(sensor)
                lines.append(
                    f"- **{m}** / {display}: First-quarter avg {q1:.1f} \u2192 Last-quarter avg {q4:.1f} [{unit}] \u2192 **{trend}**{cal_note}\n"
                )
        lines.append("\n")

    # Check for oil condition trends
    if "water_content" in all_stats and not all_stats["water_content"].empty:
        wc_df = all_stats["water_content"]
        lines.append("### Oil Condition (Water Content)\n\n")
        for m in sorted(wc_df["machine_slug"].unique()):
            ms = wc_df[wc_df["machine_slug"] == m].sort_values("start_time")
            for sensor in ms["sensor"].unique():
                ss = ms[ms["sensor"] == sensor]
                if len(ss) < 5:
                    continue
                cal_note = "" if is_calibrated(sensor, m) else " **[uncalibrated \u2014 raw sensor values]**"
                q1 = ss.iloc[:max(1, len(ss) // 4)]["median"].mean()
                q4 = ss.iloc[-max(1, len(ss) // 4):]["median"].mean()
                unit = get_unit(sensor, m)
                display = get_display_label(sensor)
                lines.append(
                    f"- **{m}** / {display}: First-quarter avg {q1:.1f} \u2192 Last-quarter avg {q4:.1f} [{unit}]\n{cal_note}"
                )
        lines.append("\n")

    # Check for pressure trends
    if "pump_pressure" in all_stats and not all_stats["pump_pressure"].empty:
        pp_df = all_stats["pump_pressure"]
        lines.append("### Pump Pressure Trends\n\n")
        for m in sorted(pp_df["machine_slug"].unique()):
            ms = pp_df[pp_df["machine_slug"] == m].sort_values("start_time")
            sensors_here = sorted(ms["sensor"].unique())
            sensors_english = [get_display_label(s) for s in sensors_here]
            avg_medians = ms.groupby("site_id")["median"].mean().round(1)
            avg_dict = {k: float(v) for k, v in avg_medians.items()}
            lines.append(
                f"- **{m}**: {len(ms)} traces, sensors: {', '.join(sensors_english)}. "
                f"Avg median by site: {avg_dict}\n"
            )
        lines.append("\n")

    # Leakage analysis
    if "leakage_pressure" in all_stats and not all_stats["leakage_pressure"].empty:
        lk_df = all_stats["leakage_pressure"]
        lines.append("### Leakage / Seal Pressure\n\n")
        for m in sorted(lk_df["machine_slug"].unique()):
            ms = lk_df[lk_df["machine_slug"] == m].sort_values("start_time")
            for sensor in sorted(ms["sensor"].unique()):
                ss = ms[ms["sensor"] == sensor]
                if len(ss) < 5:
                    continue
                cal_note = "" if is_calibrated(sensor, m) else " **[uncalibrated]**"
                q1 = ss.iloc[:max(1, len(ss) // 4)]["median"].mean()
                q4 = ss.iloc[-max(1, len(ss) // 4):]["median"].mean()
                trend = "RISING" if q4 > q1 * 1.15 else "STABLE"
                unit = get_unit(sensor, m)
                display = get_display_label(sensor)
                lines.append(
                    f"- **{m}** / {display}: {q1:.3f} \u2192 {q4:.3f} [{unit}] \u2014 **{trend}**{cal_note}\n"
                )
        lines.append("\n")

    # Data quality warnings
    lines.append("---\n\n")
    lines.append("## Data Quality / Calibration Notes\n\n")
    lines.append("Several sensor channels report raw ADC (analog-to-digital converter) counts rather than ")
    lines.append("engineering units. These are marked with `[arb. units]` on plot axes and highlighted ")
    lines.append("with an orange background. Specifically:\n\n")
    lines.append("| Sensor | Calibrated Machines | Uncalibrated Machines | Issue |\n")
    lines.append("|--------|--------------------|-----------------------|-------|\n")
    lines.append("| Temperature Cutter Right (Temperatur FRR) | bg45v_3923 | cube0_482, mc86_621 | Raw values 0\u20131300 instead of \u00b0C |\n")
    lines.append("| Water Content Gearbox Oil (Wassergehalt) | bg45v_3923 | cube0_482, mc86_621 | Raw capacitance, not % |\n")
    lines.append("| Leakage Pressure Gearbox (Leckagedruck) | bg45v_3923 | cube0_482, mc86_621 | Raw values 0\u20133000 instead of bar |\n")
    lines.append("| Gearbox Oil Pressure (Oeldruck Getriebe) | bg45v_3923 | cube0_482, mc86_621 | Raw values 0\u2013880 instead of bar |\n")
    lines.append("| Cutter Winch Force (Seilkraft Fr\u00e4swinde) | bg45v_3923 | cube0_482, mc86_621 | Raw values 0\u20133620 instead of tonnes |\n")
    lines.append("| Cutter Rotation (Drehzahl FRL/FRR) | bg45v_3923 | cube0_482, mc86_621 | Raw values 0\u20139232 instead of rpm |\n")
    lines.append("| Inclination Y Mast (Neigung Y Mast) | bg42v_5925, bg45v_4027 | bg30v_2872, bg33v_5610 | Raw values 0\u201314000 |\n")
    lines.append("| Torque kNm (DrehmomentkNm) | bg42v_5925 | bg28h_6061, bg33v_5610, bg45v_4027 | Raw values 0\u20139023 |\n")
    lines.append("| Auxiliary Winch Force (Seilkraft Hilfswinde) | bg30v_2872, bg42v_5925 | bg28h_6061, bg33v_5610, bg45v_4027 | Raw values 0\u20139900 |\n")
    lines.append("\n")
    lines.append("**Impact**: Uncalibrated channels still provide useful information for **trend analysis** ")
    lines.append("(relative changes over time). A rising trend in raw ADC counts for leakage pressure still ")
    lines.append("indicates increasing seal wear. However, absolute comparisons between calibrated and ")
    lines.append("uncalibrated machines are not meaningful.\n\n")
    lines.append("**Recommendation**: Obtain sensor calibration curves from Bauer to convert raw values ")
    lines.append("to engineering units.\n\n")

    lines.append("---\n\n")
    lines.append("## Recommendations\n\n")
    lines.append("1. **CUT machines**: Monitor temperature and oil water content trends closely. ")
    lines.append("Rising temperatures or water content indicate bearing/seal wear.\n")
    lines.append("2. **All machines**: Track pump pressure P95 over time. A gradual rise in P95 ")
    lines.append("at constant load may indicate pump wear or system restriction.\n")
    lines.append("3. **Leakage pressures**: Any sustained rise in gearbox leakage pressure is an ")
    lines.append("early indicator of seal degradation \u2014 schedule inspection before failure.\n")
    lines.append("4. **Multi-site machines**: Compare baseline pressures across sites to separate ")
    lines.append("machine degradation from site-specific load differences.\n")
    lines.append("5. **Sensor calibration**: Obtain calibration data from Bauer for raw-value channels ")
    lines.append("(especially cube0_482 and mc86_621 which have many uncalibrated sensors).\n")
    lines.append("6. **bg45v_3923 attention**: Temperature shows RISING trend on both cutters, ")
    lines.append("and left gearbox leakage pressure shows upward trend. This machine should be inspected.\n")

    REPORT_PATH.write_text("".join(lines))
    print(f"Report written to {REPORT_PATH}")


def main():
    print("Loading merged trace index...")
    df_index = load_merged_index()
    print(f"  {len(df_index):,} sessions, {df_index['machine_slug'].nunique()} machines")

    machines = sorted(df_index["machine_slug"].unique())
    all_stats = {}
    all_figures = {}

    for group_key, group_info in HEALTH_SENSORS.items():
        print(f"\nCollecting {group_info['label']}...")
        group_figures = []

        df_stats = collect_sensor_timelines(df_index, group_info)
        if df_stats.empty:
            print(f"  No data found for {group_info['label']}")
            continue

        all_stats[group_key] = df_stats
        n_traces = df_stats["trace_id"].nunique()
        n_machines = df_stats["machine_slug"].nunique()
        print(f"  Found data in {n_traces} traces across {n_machines} machines")

        # Per-machine timeline plots
        for slug in machines:
            fname = plot_sensor_timeline(df_stats, group_info, slug)
            if fname:
                group_figures.append(fname)
                print(f"    {fname}")

        # Per-machine boxplot by site
        for slug in machines:
            fname = plot_sensor_boxplot_by_site(df_stats, group_info, slug)
            if fname:
                group_figures.append(fname)
                print(f"    {fname}")

        # Cross-machine comparison (grouped by technique)
        cross_fnames = plot_cross_machine_comparison(df_stats, group_info)
        for fname in cross_fnames:
            group_figures.append(fname)
            print(f"    {fname}")

        all_figures[group_key] = group_figures

    # Pump idle fraction plot
    if "pump_pressure" in all_stats:
        fname = plot_working_vs_idle_pressure(all_stats["pump_pressure"])
        if fname:
            all_figures.setdefault("pump_pressure", []).append(fname)
            print(f"  {fname}")

    print("\nGenerating report...")
    generate_report(df_index, all_stats, all_figures)
    print("Done!")


if __name__ == "__main__":
    main()
