"""Generate per-pile feature analysis with site-level summaries and representative dashboards.

Tier 1 — Site-level summary grids: depth profiles, cross-pile comparison,
         verticality summary, depth-vs-sensor scatter (one figure per site).
Tier 2 — Representative pile dashboards: multi-panel time-series for
         3-5 interesting piles per site.

Generates figures in reports/figures/perpile/ and report at reports/13_perpile_analysis.md.
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
    clean_sentinels_df, get_display_label, get_unit, is_calibrated,
)
from macchine.analysis.perpile_features import (
    TECHNIQUE_SENSORS, extract_pile_features, depth_profile_features,
    verticality_features, get_deviation_trajectory, select_representative_piles,
    _get_col, _to_numeric,
)
from macchine.analysis.plot_utils import plot_with_gaps

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

OUTPUT_DIR = Path("output")
FIG_DIR = Path("reports/figures/perpile")
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = Path("reports/13_perpile_analysis.md")

sns.set_theme(style="whitegrid", font_scale=0.9)

MAX_PILES_PER_SITE_GRID = 40  # Max piles to show in grid plots


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


def load_trace(row) -> pd.DataFrame | None:
    """Load and clean a trace parquet file."""
    path = get_trace_path(row)
    if not path.exists():
        return None
    try:
        tdf = pd.read_parquet(path)
        tdf = clean_sentinels_df(tdf)
        return tdf
    except Exception:
        return None


# ── Feature extraction for all piles in a site ──────────────────────────────

def extract_site_features(site_df: pd.DataFrame) -> pd.DataFrame:
    """Extract features for all piles in a site."""
    records = []
    for _, row in site_df.iterrows():
        tdf = load_trace(row)
        if tdf is None:
            continue
        features = extract_pile_features(tdf, row["technique"], row["machine_slug"])
        features["trace_id"] = row["trace_id"]
        features["element_name"] = row["element_name"]
        features["technique"] = row["technique"]
        features["machine_slug"] = row["machine_slug"]
        features["site_id"] = row["site_id"]
        features["start_time"] = row["start_time"]
        records.append(features)
    return pd.DataFrame(records)


# ── TIER 1: Site-level summary figures ───────────────────────────────────────

def fig_depth_profiles_grid(site_id: str, site_df: pd.DataFrame) -> str | None:
    """Depth vs time profiles, one subplot per pile (grid layout)."""
    named = site_df[site_df["element_name"].notna() &
                    (site_df["element_name"] != "") &
                    (site_df["element_name"] != "xxxxx")]
    elements = sorted(named["element_name"].unique())
    if len(elements) < 3:
        return None

    # Limit to manageable number
    if len(elements) > MAX_PILES_PER_SITE_GRID:
        elements = elements[:MAX_PILES_PER_SITE_GRID]

    ncols = min(6, len(elements))
    nrows = (len(elements) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows),
                              squeeze=False, sharex=False)

    for i, elem in enumerate(elements):
        ax = axes[i // ncols, i % ncols]
        elem_rows = named[named["element_name"] == elem]

        for _, row in elem_rows.iterrows():
            tdf = load_trace(row)
            if tdf is None:
                continue
            depth = _get_col(tdf, "Tiefe")
            if depth is None:
                continue
            if "timestamp" in tdf.columns:
                ts = pd.to_datetime(tdf["timestamp"], errors="coerce")
                elapsed_min = (ts - ts.min()).dt.total_seconds() / 60
                ax.plot(elapsed_min, depth, linewidth=0.8, alpha=0.7)
                ax.set_xlabel("min", fontsize=7)
            else:
                ax.plot(depth.values, linewidth=0.8, alpha=0.7)

        ax.set_title(elem, fontsize=8, fontweight="bold")
        ax.invert_yaxis()
        if i % ncols == 0:
            ax.set_ylabel("Depth (m)", fontsize=7)
        ax.tick_params(labelsize=6)

    # Hide unused subplots
    for j in range(len(elements), nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)

    fig.suptitle(f"Site {site_id} — Depth Profiles ({len(elements)} piles)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fname = f"{site_id}_depth_profiles.png"
    fig.savefig(FIG_DIR / fname, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return fname


def fig_cross_pile_comparison(site_id: str, features_df: pd.DataFrame) -> str | None:
    """6-panel bar chart: max depth, mean pressure, mean torque, max deviation, duration, n_cycles."""
    if features_df.empty or len(features_df) < 3:
        return None

    df = features_df.sort_values("element_name")
    if len(df) > MAX_PILES_PER_SITE_GRID:
        df = df.head(MAX_PILES_PER_SITE_GRID)

    metrics = [
        ("max_depth", "Max Depth (m)"),
        ("mean_pressure", "Mean Pressure"),
        ("mean_torque", "Mean Torque"),
        ("max_deviation", "Max Deviation"),
        ("duration_active_s", "Duration (s)"),
        ("n_cycles", "N Cycles"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    palette = sns.color_palette("Set2", len(df))

    for idx, (col, label) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        if col not in df.columns:
            ax.set_visible(False)
            continue
        vals = df[col].fillna(0)
        bars = ax.barh(df["element_name"], vals, color=palette[:len(df)], alpha=0.8)
        ax.set_xlabel(label, fontsize=9)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_title(label, fontweight="bold", fontsize=10)

    fig.suptitle(f"Site {site_id} — Cross-Pile Comparison ({len(df)} piles)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fname = f"{site_id}_cross_pile.png"
    fig.savefig(FIG_DIR / fname, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return fname


def fig_verticality_summary(site_id: str, site_df: pd.DataFrame,
                            features_df: pd.DataFrame) -> str | None:
    """Grid of inclination X-Y scatter + deviation per pile."""
    named = features_df.dropna(subset=["inclination_x_mean", "inclination_y_mean"])
    if len(named) < 3:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Inclination X vs Y scatter (one point per pile)
    ax = axes[0]
    ax.scatter(named["inclination_x_mean"], named["inclination_y_mean"],
               s=30, alpha=0.6, c="#3498db", edgecolors="white")
    # Add circle at 1 degree threshold
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), "r--", alpha=0.5, label="1 deg")
    ax.plot(2 * np.cos(theta), 2 * np.sin(theta), "r:", alpha=0.3, label="2 deg")
    ax.set_xlabel("Inclination X (deg)")
    ax.set_ylabel("Inclination Y (deg)")
    ax.set_title("Inclination X vs Y (per pile)")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Max deviation histogram
    ax = axes[1]
    dev = features_df["max_deviation"].dropna()
    if len(dev) > 0:
        ax.hist(dev, bins=min(30, len(dev)), color="#e67e22", alpha=0.7, edgecolor="white")
        ax.axvline(dev.median(), color="red", linestyle="--", label=f"Median: {dev.median():.1f}")
        ax.set_xlabel("Max Radial Deviation")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Max Deviation")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No deviation data", ha="center", va="center",
                transform=ax.transAxes)

    fig.suptitle(f"Site {site_id} — Verticality Summary ({len(named)} piles with data)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fname = f"{site_id}_verticality.png"
    fig.savefig(FIG_DIR / fname, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return fname


def fig_depth_vs_sensor(site_id: str, site_df: pd.DataFrame) -> str | None:
    """Sensor vs depth scatter, colored by pile, for primary sensors."""
    # Determine technique
    technique = site_df["technique"].mode().iloc[0] if not site_df.empty else None
    if technique is None:
        return None

    tech_cfg = TECHNIQUE_SENSORS.get(technique, {})
    primary_sensors = tech_cfg.get("primary", [])[:3]  # max 3 sensors
    depth_col = tech_cfg.get("depth", "Tiefe")

    if not primary_sensors:
        return None

    named = site_df[site_df["element_name"].notna() &
                    (site_df["element_name"] != "") &
                    (site_df["element_name"] != "xxxxx")]
    elements = sorted(named["element_name"].unique())
    if len(elements) < 3:
        return None

    # Sample max 20 piles for readability
    if len(elements) > 20:
        elements = elements[:20]

    n_sensors = len(primary_sensors)
    fig, axes = plt.subplots(1, n_sensors, figsize=(6 * n_sensors, 7), squeeze=False)
    colors = sns.color_palette("husl", len(elements))

    for si, sensor in enumerate(primary_sensors):
        ax = axes[0, si]
        for ei, elem in enumerate(elements):
            elem_rows = named[named["element_name"] == elem]
            for _, row in elem_rows.iterrows():
                tdf = load_trace(row)
                if tdf is None:
                    continue
                depth = _get_col(tdf, depth_col)
                val = _get_col(tdf, sensor)
                if depth is None or val is None:
                    continue
                # Subsample for performance
                step = max(1, len(depth) // 200)
                ax.scatter(val.iloc[::step], depth.iloc[::step],
                           s=2, alpha=0.15, color=colors[ei % len(colors)])

        ax.invert_yaxis()
        ax.set_xlabel(get_display_label(sensor), fontsize=9)
        if si == 0:
            ax.set_ylabel("Depth (m)")
        ax.set_title(sensor, fontweight="bold", fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Site {site_id} — Depth vs Sensor ({technique}, {len(elements)} piles)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fname = f"{site_id}_depth_vs_sensor.png"
    fig.savefig(FIG_DIR / fname, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return fname


# ── TIER 2: Representative pile dashboards ──────────────────────────────────

def fig_pile_dashboard(site_id: str, element_name: str,
                       elem_df: pd.DataFrame) -> str | None:
    """Multi-panel time-series dashboard for one pile."""
    if elem_df.empty:
        return None

    # Load the first (or only) trace
    row = elem_df.iloc[0]
    tdf = load_trace(row)
    if tdf is None:
        return None

    technique = row["technique"]
    tech_cfg = TECHNIQUE_SENSORS.get(technique, {})
    depth_col = tech_cfg.get("depth", "Tiefe")
    machine_slug = row["machine_slug"]

    # Build list of sensor panels
    panels = []

    # Panel 1: Depth
    depth = _get_col(tdf, depth_col)
    if depth is not None:
        panels.append((depth_col, depth, "Depth (m)", True))  # True = invert y

    # Panel 2-3: Primary sensors
    for sensor in tech_cfg.get("primary", [])[:3]:
        val = _get_col(tdf, sensor)
        if val is not None:
            unit = get_unit(sensor, machine_slug)
            label = f"{get_display_label(sensor)} [{unit}]"
            panels.append((sensor, val, label, False))

    # Panel 4: Pressure
    for sensor in tech_cfg.get("pressure", [])[:2]:
        if sensor not in [p[0] for p in panels]:  # avoid duplicates
            val = _get_col(tdf, sensor)
            if val is not None:
                unit = get_unit(sensor, machine_slug)
                label = f"{get_display_label(sensor)} [{unit}]"
                panels.append((sensor, val, label, False))

    # Panel 5: Verticality
    for sensor in tech_cfg.get("verticality", [])[:2]:
        val = _get_col(tdf, sensor)
        if val is not None:
            unit = get_unit(sensor, machine_slug)
            label = f"{get_display_label(sensor)} [{unit}]"
            panels.append((sensor, val, label, False))

    if not panels:
        return None

    # Limit to 6 panels max
    panels = panels[:6]
    n_panels = len(panels)

    # X axis: elapsed time in minutes
    if "timestamp" in tdf.columns:
        ts = pd.to_datetime(tdf["timestamp"], errors="coerce")
        x = (ts - ts.min()).dt.total_seconds() / 60
        xlabel = "Elapsed Time (min)"
    else:
        x = pd.Series(range(len(tdf)), dtype=float)
        xlabel = "Sample Index"

    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 2.5 * n_panels),
                              squeeze=False, sharex=True)

    for i, (sensor_name, values, label, invert) in enumerate(panels):
        ax = axes[i, 0]
        ax.plot(x, values, linewidth=0.6, alpha=0.8, color="#2c3e50")
        ax.set_ylabel(label, fontsize=8)
        if invert:
            ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

        if not is_calibrated(sensor_name, machine_slug):
            ax.set_facecolor("#fff8f0")
            ax.text(0.01, 0.95, "UNCALIBRATED", transform=ax.transAxes,
                    fontsize=7, color="#cc6600", fontweight="bold", va="top", alpha=0.6)

    axes[-1, 0].set_xlabel(xlabel, fontsize=9)

    fig.suptitle(
        f"Site {site_id} — Pile {element_name} ({technique}, {machine_slug})",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    safe_elem = element_name.replace("/", "_").replace(" ", "_")
    fname = f"{site_id}_{safe_elem}_dashboard.png"
    fig.savefig(FIG_DIR / fname, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return fname


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading merged trace index...")
    df_index = load_merged_index()
    print(f"  {len(df_index):,} sessions across {df_index['site_id'].nunique()} sites")

    # Filter to named elements only
    df_named = df_index[
        df_index["element_name"].notna() &
        (df_index["element_name"] != "") &
        (df_index["element_name"] != "xxxxx")
    ]
    print(f"  {len(df_named):,} named sessions")

    sites = sorted(df_named["site_id"].unique())
    all_figures = []
    all_features = []

    for site_id in sites:
        site_df = df_named[df_named["site_id"] == site_id]
        n_elements = site_df["element_name"].nunique()
        if n_elements < 3:
            continue

        technique = site_df["technique"].mode().iloc[0]
        print(f"\n--- Site {site_id} ({len(site_df)} sessions, "
              f"{n_elements} elements, {technique}) ---")

        # Extract features for all piles
        print("  Extracting features...")
        features_df = extract_site_features(site_df)
        if features_df.empty:
            print("  No features extracted, skipping")
            continue
        all_features.append(features_df)

        # TIER 1: Site-level summary figures
        print("  Generating Tier 1 figures...")

        fname = fig_depth_profiles_grid(site_id, site_df)
        if fname:
            all_figures.append(fname)
            print(f"    {fname}")

        fname = fig_cross_pile_comparison(site_id, features_df)
        if fname:
            all_figures.append(fname)
            print(f"    {fname}")

        fname = fig_verticality_summary(site_id, site_df, features_df)
        if fname:
            all_figures.append(fname)
            print(f"    {fname}")

        fname = fig_depth_vs_sensor(site_id, site_df)
        if fname:
            all_figures.append(fname)
            print(f"    {fname}")

        # TIER 2: Representative pile dashboards
        print("  Generating Tier 2 dashboards...")
        reps = select_representative_piles(site_df, features_df, n=5)
        for elem in reps:
            elem_df = site_df[site_df["element_name"] == elem]
            fname = fig_pile_dashboard(site_id, elem, elem_df)
            if fname:
                all_figures.append(fname)
                print(f"    {fname}")

    # Generate report
    print("\nGenerating report...")
    generate_report(df_index, all_features, all_figures, sites)
    print(f"\nDone! {len(all_figures)} figures saved to {FIG_DIR}/")


def generate_report(df_index: pd.DataFrame, all_features: list[pd.DataFrame],
                    all_figures: list[str], sites: list[str]):
    """Generate the markdown report."""
    lines = [
        "# Per-Pile Feature Analysis\n\n",
        "This report provides per-pile analysis across all sites: depth profiles, ",
        "cross-pile feature comparisons, verticality assessments, and depth-vs-sensor ",
        "scatter plots. For each site, representative pile dashboards are generated.\n\n",
        f"**Dataset**: {len(df_index):,} sessions across {df_index['site_id'].nunique()} sites.\n\n",
        "---\n\n",
    ]

    # Combine all features
    if all_features:
        features_all = pd.concat(all_features, ignore_index=True)

        lines.append("## Fleet-wide Pile Statistics\n\n")
        lines.append("| Metric | Mean | Median | Std | Min | Max |\n")
        lines.append("|--------|------|--------|-----|-----|-----|\n")
        for col, label in [
            ("max_depth", "Max Depth (m)"),
            ("mean_pressure", "Mean Pressure"),
            ("mean_torque", "Mean Torque"),
            ("duration_active_s", "Duration (s)"),
            ("max_deviation", "Max Deviation"),
            ("inclination_rms", "Inclination RMS (deg)"),
        ]:
            if col in features_all.columns:
                vals = features_all[col].dropna()
                if len(vals) > 0:
                    lines.append(
                        f"| {label} | {vals.mean():.1f} | {vals.median():.1f} | "
                        f"{vals.std():.1f} | {vals.min():.1f} | {vals.max():.1f} |\n"
                    )

        lines.append("\n---\n\n")

    # Per-site sections with figures
    for site_id in sites:
        site_figs = [f for f in all_figures if f.startswith(f"{site_id}_")]
        if not site_figs:
            continue

        site_df = df_index[df_index["site_id"] == site_id]
        technique = site_df["technique"].mode().iloc[0] if not site_df.empty else "?"
        n_elements = site_df["element_name"].nunique()

        lines.append(f"## Site {site_id}\n\n")
        lines.append(f"**Technique**: {technique} | **Elements**: {n_elements} | "
                     f"**Sessions**: {len(site_df)}\n\n")

        for fname in site_figs:
            lines.append(f"![{site_id}](figures/perpile/{fname})\n\n")

        lines.append("---\n\n")

    lines.append("## Methodology\n\n")
    lines.append("### Tier 1 — Site-Level Summary Grids\n\n")
    lines.append("- **Depth Profiles**: One subplot per pile showing depth vs elapsed time\n")
    lines.append("- **Cross-Pile Comparison**: Bar charts of max depth, mean pressure, "
                 "torque, deviation, duration, cycles\n")
    lines.append("- **Verticality Summary**: Inclination X-Y scatter and max deviation distribution\n")
    lines.append("- **Depth-vs-Sensor Scatter**: Primary sensors plotted against depth, colored by pile\n\n")
    lines.append("### Tier 2 — Representative Pile Dashboards\n\n")
    lines.append("For each site, 3-5 representative piles are selected: deepest, shallowest, "
                 "longest, shortest, most inclined. Each gets a multi-panel time-series dashboard.\n\n")
    lines.append("All figures are in `reports/figures/perpile/`.\n")

    REPORT_PATH.write_text("".join(lines))
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
