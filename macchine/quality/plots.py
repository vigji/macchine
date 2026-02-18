"""Matplotlib plotting functions for site quality reports.

Each function saves one PNG and closes the figure.  Uses the Agg backend
so no display is needed.  Axis labels use ``get_display_label`` /
``get_unit`` from the calibration module.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from macchine.harmonize.calibration import get_axis_label, get_display_label, get_unit
from macchine.quality.checks import SensorIssue

# Consistent style
_FIG_DPI = 120
_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def plot_trace_example(
    trace_df: pd.DataFrame,
    sensor_name: str,
    machine_slug: str,
    trace_id: str,
    issues: list[SensorIssue],
    save_path: Path,
) -> None:
    """Time-series plot of one sensor, annotating any detected issues."""
    if sensor_name not in trace_df.columns:
        return

    series = trace_df[sensor_name].dropna()
    if series.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 3.5))

    x = np.arange(len(series))
    ax.plot(x, series.values, linewidth=0.6, color=_COLORS[0])

    ax.set_xlabel("Sample index")
    ax.set_ylabel(get_axis_label(sensor_name, machine_slug))
    ax.set_title(f"{get_display_label(sensor_name)} — {machine_slug} / {trace_id}")

    # Annotate issues for this sensor
    sensor_issues = [i for i in issues if i.sensor == sensor_name]
    for si in sensor_issues:
        ax.annotate(
            f"{si.issue_type}",
            xy=(0.98, 0.95),
            xycoords="axes fraction",
            ha="right", va="top",
            fontsize=8,
            color="red",
            bbox=dict(boxstyle="round,pad=0.3", fc="mistyrose", ec="red", alpha=0.8),
        )

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=_FIG_DPI)
    plt.close(fig)


def plot_depth_profile(
    trace_df: pd.DataFrame,
    machine_slug: str,
    element_name: str,
    sensors: list[str],
    save_path: Path,
) -> None:
    """Sensor values vs depth (y-axis inverted), up to 4 subplots."""
    depth_col = None
    for candidate in ("Tiefe", "Vorschub Tiefe", "Tiefe_Hauptwinde_GOK", "Tiefe_Bohrrohr_GOK"):
        if candidate in trace_df.columns:
            depth_col = candidate
            break
    if depth_col is None:
        return

    # Pick sensors that exist, excluding the depth column itself
    plot_sensors = [s for s in sensors if s in trace_df.columns and s != depth_col][:4]
    if not plot_sensors:
        return

    n = len(plot_sensors)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 6), sharey=True)
    if n == 1:
        axes = [axes]

    depth = pd.to_numeric(trace_df[depth_col], errors="coerce").values

    for ax, sensor in zip(axes, plot_sensors):
        vals = pd.to_numeric(trace_df[sensor], errors="coerce").values
        mask = np.isfinite(depth) & np.isfinite(vals)
        ax.plot(vals[mask], depth[mask], linewidth=0.6)
        ax.set_xlabel(get_axis_label(sensor, machine_slug), fontsize=8)
        ax.tick_params(labelsize=7)

    axes[0].set_ylabel(get_axis_label(depth_col, machine_slug))
    axes[0].invert_yaxis()
    fig.suptitle(f"Depth profile — {element_name} ({machine_slug})", fontsize=10)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=_FIG_DPI)
    plt.close(fig)


def plot_sensor_coverage_heatmap(
    coverage_df: pd.DataFrame,
    site_id: str,
    save_path: Path,
) -> None:
    """Binary heatmap: sensors (rows) × machines (columns)."""
    if coverage_df.empty:
        return

    fig, ax = plt.subplots(figsize=(max(4, len(coverage_df.columns) * 1.2), max(4, len(coverage_df) * 0.3)))
    data = coverage_df.values.astype(float)
    ax.imshow(data, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")

    ax.set_xticks(range(len(coverage_df.columns)))
    ax.set_xticklabels(coverage_df.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(coverage_df.index)))
    ax.set_yticklabels(coverage_df.index, fontsize=7)
    ax.set_title(f"Sensor coverage — site {site_id}")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=_FIG_DPI)
    plt.close(fig)


def plot_calibration_bar(
    summaries: list,
    site_id: str,
    save_path: Path,
) -> None:
    """Stacked bar chart: calibrated / uncalibrated sensor count per machine."""
    if not summaries:
        return

    machines = [s.machine_slug for s in summaries]
    cal = [s.n_calibrated for s in summaries]
    uncal = [s.n_uncalibrated for s in summaries]

    x = np.arange(len(machines))
    fig, ax = plt.subplots(figsize=(max(4, len(machines) * 1.5), 4))
    ax.bar(x, cal, label="Calibrated", color=_COLORS[0])
    ax.bar(x, uncal, bottom=cal, label="Uncalibrated", color=_COLORS[3])

    ax.set_xticks(x)
    ax.set_xticklabels(machines, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Number of sensors")
    ax.set_title(f"Calibration status — site {site_id}")
    ax.legend(fontsize=8)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=_FIG_DPI)
    plt.close(fig)


def plot_duration_histogram(
    site_traces: pd.DataFrame,
    site_id: str,
    save_path: Path,
) -> None:
    """Histogram of trace durations (minutes), coloured by technique."""
    if site_traces.empty or "duration_s" not in site_traces.columns:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    techniques = sorted(site_traces["technique"].dropna().unique())

    for i, tech in enumerate(techniques):
        durations = site_traces.loc[site_traces["technique"] == tech, "duration_s"].dropna() / 60
        if durations.empty:
            continue
        ax.hist(durations, bins=30, alpha=0.6, label=tech, color=_COLORS[i % len(_COLORS)])

    ax.set_xlabel("Duration [min]")
    ax.set_ylabel("Count")
    ax.set_title(f"Trace duration distribution — site {site_id}")
    if techniques:
        ax.legend(fontsize=8)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=_FIG_DPI)
    plt.close(fig)
