"""Generate event detection report.

Scans the fleet for anomalous events within individual traces and generates
a summary report with event timelines.

Output: reports/16_event_detection.md + reports/figures/events/*.png
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
import warnings

from macchine.analysis.event_detection import scan_fleet_events
from macchine.analysis.plot_utils import plot_with_gaps, add_site_markers

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

OUTPUT_DIR = Path("output")
FIG_DIR = Path("reports/figures/events")
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = Path("reports/16_event_detection.md")

sns.set_theme(style="whitegrid", font_scale=0.9)


def plot_event_timeline(events_df: pd.DataFrame) -> list[str]:
    """Generate event timeline plots grouped by machine and event type."""
    if events_df.empty:
        return []

    fnames = []

    # Overview: event count per machine per type
    pivot = events_df.groupby(["machine_slug", "event_type"]).size().unstack(fill_value=0)

    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(12, max(4, len(pivot) * 0.5 + 2)))
        pivot.plot(kind="barh", stacked=True, ax=ax, alpha=0.8)
        ax.set_xlabel("Number of Events")
        ax.set_title("Detected Events by Machine and Type", fontweight="bold")
        ax.legend(title="Event Type", fontsize=8, bbox_to_anchor=(1.02, 1))
        fig.tight_layout()
        fname = "event_overview.png"
        fig.savefig(FIG_DIR / fname, dpi=120, bbox_inches="tight")
        plt.close(fig)
        fnames.append(fname)

    # Per-machine timeline
    for machine in events_df["machine_slug"].unique():
        m_events = events_df[events_df["machine_slug"] == machine].copy()
        if len(m_events) < 3:
            continue

        event_types = m_events["event_type"].unique()
        n_types = len(event_types)

        fig, axes = plt.subplots(n_types, 1, figsize=(14, 3 * n_types),
                                  squeeze=False, sharex=True)

        severity_colors = {"high": "red", "medium": "orange", "low": "#3498db"}

        for i, etype in enumerate(sorted(event_types)):
            ax = axes[i, 0]
            et = m_events[m_events["event_type"] == etype]

            for severity, color in severity_colors.items():
                s_events = et[et["severity"] == severity]
                if not s_events.empty:
                    ax.scatter(s_events["start_time"], [1] * len(s_events),
                               s=30, c=color, alpha=0.6, label=f"{severity} ({len(s_events)})")

            ax.set_ylabel(etype.replace("_", "\n"), fontsize=8)
            ax.set_yticks([])
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.3)

            if "site_id" in et.columns:
                add_site_markers(ax, et, site_col="site_id", time_col="start_time")

        axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.xticks(rotation=45, ha="right")

        fig.suptitle(f"Event Timeline: {machine} ({len(m_events)} events)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        fname = f"{machine}_event_timeline.png"
        fig.savefig(FIG_DIR / fname, dpi=120, bbox_inches="tight")
        plt.close(fig)
        fnames.append(fname)

    return fnames


def generate_report(events_df: pd.DataFrame, figures: list[str]):
    """Generate the event detection markdown report."""
    lines = [
        "# Event Detection Report\n\n",
        "This report identifies specific anomalous events within individual traces ",
        "using technique-specific detectors. Events are detected from actual sensor ",
        "time-series data (not metadata).\n\n",
    ]

    if events_df.empty:
        lines.append("**No events detected in the scanned traces.**\n")
        REPORT_PATH.write_text("".join(lines))
        return

    n_events = len(events_df)
    n_traces = events_df["trace_id"].nunique()
    n_machines = events_df["machine_slug"].nunique()

    lines.append(f"**Summary**: {n_events} events detected across {n_traces} traces "
                 f"and {n_machines} machines.\n\n")
    lines.append("---\n\n")

    # Event type summary
    lines.append("## Event Types\n\n")
    lines.append("| Event Type | Count | High | Medium | Low | Machines | Description |\n")
    lines.append("|------------|-------|------|--------|-----|----------|-------------|\n")

    type_descriptions = {
        "concrete_pressure_loss": "Sudden concrete pressure drop during concreting (SOB)",
        "pressure_asymmetry": "Sustained L/R cutter pressure imbalance (CUT)",
        "temperature_excursion": "Temperature exceeding baseline (CUT)",
        "incomplete_grab_cycle": "Grab cycle with no soil captured (GRAB)",
        "torque_anomaly": "Sudden torque drop during active drilling (KELLY)",
    }

    for etype, edf in events_df.groupby("event_type"):
        n = len(edf)
        n_high = len(edf[edf["severity"] == "high"])
        n_med = len(edf[edf["severity"] == "medium"])
        n_low = len(edf[edf["severity"] == "low"])
        machines = ", ".join(sorted(edf["machine_slug"].unique()))
        desc = type_descriptions.get(etype, etype)
        lines.append(f"| {etype} | {n} | {n_high} | {n_med} | {n_low} | {machines} | {desc} |\n")

    lines.append("\n---\n\n")

    # Per-machine details
    lines.append("## Per-Machine Event Details\n\n")
    for machine in sorted(events_df["machine_slug"].unique()):
        m_events = events_df[events_df["machine_slug"] == machine]
        lines.append(f"### {machine} ({len(m_events)} events)\n\n")

        for etype in sorted(m_events["event_type"].unique()):
            et = m_events[m_events["event_type"] == etype]
            lines.append(f"**{etype}** ({len(et)} events):\n\n")

            # Show top 10 by severity
            for severity in ["high", "medium", "low"]:
                s_events = et[et["severity"] == severity]
                if s_events.empty:
                    continue
                sample = s_events.head(5)
                for _, evt in sample.iterrows():
                    lines.append(
                        f"- [{severity.upper()}] {evt['start_time'].strftime('%Y-%m-%d')} "
                        f"({evt.get('element_name', 'N/A')}): {evt['description']}\n"
                    )
                if len(s_events) > 5:
                    lines.append(f"- ... and {len(s_events) - 5} more {severity} events\n")

            lines.append("\n")

        lines.append("---\n\n")

    # Figures
    if figures:
        lines.append("## Event Timeline Plots\n\n")
        for fname in figures:
            lines.append(f"![Events](figures/events/{fname})\n\n")

    # Methodology
    lines.append("## Methodology\n\n")
    lines.append("### Detectors\n\n")
    lines.append("| Technique | Detector | What It Detects |\n")
    lines.append("|-----------|----------|-----------------|\n")
    lines.append("| SOB | concrete_pressure_loss | Sudden Betondruck drop during concreting |\n")
    lines.append("| CUT | pressure_asymmetry | Sustained |FRL-FRR| > 50 bar during cutting |\n")
    lines.append("| CUT | temperature_excursion | Temperature > baseline + 3*std |\n")
    lines.append("| GRAB | incomplete_grab_cycle | Ascent force ~= descent force (empty grab) |\n")
    lines.append("| KELLY | torque_anomaly | Torque drop > 80% during active drilling |\n\n")
    lines.append("All detectors use only calibrated sensors and validate physical ranges.\n")

    REPORT_PATH.write_text("".join(lines))
    print(f"Report written to {REPORT_PATH}")


def main():
    print("Scanning fleet for events...")
    events_df = scan_fleet_events(OUTPUT_DIR)

    if events_df.empty:
        print("No events detected.")
    else:
        print(f"\nDetected {len(events_df)} events across {events_df['trace_id'].nunique()} traces")
        print("\nEvent type breakdown:")
        for etype, count in events_df["event_type"].value_counts().items():
            print(f"  {etype}: {count}")

    print("\nGenerating plots...")
    figures = plot_event_timeline(events_df)
    for f in figures:
        print(f"  {f}")

    print("\nGenerating report...")
    generate_report(events_df, figures)
    print(f"\nDone! Figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
