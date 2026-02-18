"""Generate per-site pile/element distribution plots."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
import numpy as np

from macchine.storage.catalog import get_trace_index

OUTPUT_DIR = Path("output")
FIG_DIR = Path("reports/figures/piles")
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.0)


def load_data() -> pd.DataFrame:
    df = get_trace_index(OUTPUT_DIR)
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df = df.dropna(subset=["start_time"])
    df["duration_min"] = df["duration_s"] / 60
    df["date"] = df["start_time"].dt.date
    df["machine_slug"] = df["machine_slug"].replace("", "unidentified")
    # Filter out xxxxx and empty
    df["has_name"] = df["element_name"].notna() & (df["element_name"] != "") & (df["element_name"] != "xxxxx")
    return df


def parse_grid_name(name: str) -> tuple[str | None, int | None]:
    """Extract (row_letter, col_number) from grid-style names like H39, A35, C46."""
    if not isinstance(name, str):
        return None, None
    m = re.match(r"^([A-Za-z])\s*[-]?\s*(\d+)", name.strip())
    if m:
        return m.group(1).upper(), int(m.group(2))
    return None, None


# ── Per-element aggregation ───────────────────────────────────────────────────

def get_element_stats(site_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate traces per unique element within a site."""
    named = site_df[site_df["has_name"]].copy()
    if named.empty:
        return pd.DataFrame()

    stats = named.groupby("element_name").agg(
        traces=("source_path", "count"),
        techniques=("technique", lambda x: ", ".join(sorted(x.unique()))),
        primary_technique=("technique", lambda x: x.value_counts().index[0]),
        total_duration_min=("duration_min", "sum"),
        avg_duration_min=("duration_min", "mean"),
        first_time=("start_time", "min"),
        last_time=("start_time", "max"),
        sensors_avg=("sensor_count", "mean"),
        operator=("operator", lambda x: x.value_counts().index[0] if (x != "").any() else ""),
    ).reset_index()

    stats["first_date"] = stats["first_time"].dt.date
    stats["sequence_order"] = stats["first_time"].rank(method="first").astype(int)
    return stats


# ── FIGURE A: Site overview — element production timeline ─────────────────────

def fig_production_timeline(site_id: str, site_df: pd.DataFrame, stats: pd.DataFrame):
    """Timeline showing when each element was constructed, colored by technique."""
    if stats.empty or len(stats) < 3:
        return

    tech_colors = {"SOB": "#2ecc71", "KELLY": "#3498db", "CUT": "#e74c3c",
                   "GRAB": "#9b59b6", "SCM": "#f39c12", "FREE": "#95a5a6", "DMS": "#1abc9c"}

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Element construction sequence (first trace time)
    ax = axes[0, 0]
    ordered = stats.sort_values("first_time")
    colors = [tech_colors.get(t, "#7f8c8d") for t in ordered["primary_technique"]]
    ax.scatter(ordered["first_time"], range(len(ordered)), c=colors, s=15, alpha=0.7)
    ax.set_ylabel("Element # (construction order)")
    ax.set_xlabel("Date")
    ax.set_title("Element Construction Sequence")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    # Add technique legend
    for tech, color in tech_colors.items():
        if tech in ordered["primary_technique"].values:
            ax.scatter([], [], c=color, s=30, label=tech)
    ax.legend(fontsize=8, loc="upper left")

    # 2. Daily element completion rate
    ax = axes[0, 1]
    daily = site_df[site_df["has_name"]].groupby("date")["element_name"].nunique()
    dates = pd.to_datetime(pd.Series(daily.index))
    ax.bar(dates, daily.values, width=1.2, color="#3498db", alpha=0.7)
    # 7-day rolling average
    rolling = daily.rolling(7, min_periods=1, center=True).mean()
    ax.plot(pd.to_datetime(pd.Series(rolling.index)), rolling.values,
            color="red", linewidth=2, label="7-day avg")
    ax.set_ylabel("Unique Elements per Day")
    ax.set_xlabel("Date")
    ax.set_title("Daily Production Rate")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # 3. Duration distribution per element (total time spent)
    ax = axes[1, 0]
    for tech in sorted(stats["primary_technique"].unique()):
        t_stats = stats[stats["primary_technique"] == tech]
        ax.hist(t_stats["total_duration_min"], bins=30, alpha=0.6,
                color=tech_colors.get(tech, "#7f8c8d"), label=tech, edgecolor="white")
    ax.set_xlabel("Total Duration per Element (min)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Total Time per Element")
    ax.legend(fontsize=8)

    # 4. Traces per element distribution
    ax = axes[1, 1]
    trace_counts = stats["traces"].value_counts().sort_index()
    ax.bar(trace_counts.index, trace_counts.values, color="#e67e22", alpha=0.8, edgecolor="white")
    ax.set_xlabel("Number of Traces per Element")
    ax.set_ylabel("Number of Elements")
    ax.set_title("Traces per Element (phases/re-recordings)")
    ax.axvline(x=stats["traces"].median(), color="red", linestyle="--",
               label=f"Median={stats['traces'].median():.0f}")
    ax.legend(fontsize=8)

    fig.suptitle(f"Site {site_id} — Element Production Overview ({len(stats)} elements)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{site_id}_production_timeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {site_id}_production_timeline.png")


# ── FIGURE B: Grid-based spatial layout ───────────────────────────────────────

def fig_spatial_grid(site_id: str, stats: pd.DataFrame):
    """For sites with grid-style element names, plot a 2D spatial map."""
    stats = stats.copy()
    stats["row"], stats["col"] = zip(*stats["element_name"].apply(parse_grid_name))
    grid = stats.dropna(subset=["row", "col"]).copy()

    if len(grid) < 5:
        return

    rows = sorted(grid["row"].unique())
    row_map = {r: i for i, r in enumerate(rows)}
    grid["row_idx"] = grid["row"].map(row_map)

    tech_colors = {"SOB": "#2ecc71", "KELLY": "#3498db", "CUT": "#e74c3c",
                   "GRAB": "#9b59b6", "SCM": "#f39c12", "FREE": "#95a5a6"}

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # 1. Spatial layout colored by technique
    ax = axes[0]
    for tech in sorted(grid["primary_technique"].unique()):
        t = grid[grid["primary_technique"] == tech]
        ax.scatter(t["col"], t["row_idx"], c=tech_colors.get(tech, "#7f8c8d"),
                   s=60, alpha=0.8, label=tech, edgecolors="white", linewidth=0.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_xlabel("Column Number")
    ax.set_ylabel("Row")
    ax.set_title("Spatial Layout — by Technique")
    ax.legend(fontsize=8)
    ax.invert_yaxis()

    # 2. Spatial layout colored by construction order (time)
    ax = axes[1]
    sc = ax.scatter(grid["col"], grid["row_idx"], c=grid["sequence_order"],
                    cmap="viridis", s=60, alpha=0.8, edgecolors="white", linewidth=0.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_xlabel("Column Number")
    ax.set_ylabel("Row")
    ax.set_title("Construction Sequence")
    ax.invert_yaxis()
    plt.colorbar(sc, ax=ax, label="Construction Order")

    # 3. Spatial layout colored by total duration
    ax = axes[2]
    vmax = grid["total_duration_min"].quantile(0.95)
    sc = ax.scatter(grid["col"], grid["row_idx"], c=grid["total_duration_min"],
                    cmap="YlOrRd", s=60, alpha=0.8, edgecolors="white", linewidth=0.5,
                    vmin=0, vmax=vmax)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_xlabel("Column Number")
    ax.set_ylabel("Row")
    ax.set_title("Total Duration per Element")
    ax.invert_yaxis()
    plt.colorbar(sc, ax=ax, label="Duration (min)")

    fig.suptitle(f"Site {site_id} — Spatial Distribution ({len(grid)} elements on grid)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{site_id}_spatial_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {site_id}_spatial_grid.png")


# ── FIGURE C: Per-element duration heatmap (for grid sites) ──────────────────

def fig_grid_heatmap(site_id: str, stats: pd.DataFrame):
    """Create a row×col heatmap of element duration for grid-named sites."""
    stats = stats.copy()
    stats["row"], stats["col"] = zip(*stats["element_name"].apply(parse_grid_name))
    grid = stats.dropna(subset=["row", "col"]).copy()

    if len(grid) < 10:
        return

    # Pivot to row × col
    pivot = grid.pivot_table(index="row", columns="col",
                             values="total_duration_min", aggfunc="sum")
    if pivot.shape[0] < 2 or pivot.shape[1] < 3:
        return

    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns) * 0.4), max(4, len(pivot) * 0.8)))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, linewidths=0.3, linecolor="white",
                cbar_kws={"label": "Total Duration (min)"}, annot=pivot.shape[1] <= 30,
                fmt=".0f" if pivot.shape[1] <= 30 else "")
    ax.set_title(f"Site {site_id} — Element Duration Heatmap (Row × Column)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{site_id}_duration_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {site_id}_duration_heatmap.png")


# ── FIGURE D: Operator per element (who built what) ──────────────────────────

def fig_operator_elements(site_id: str, site_df: pd.DataFrame, stats: pd.DataFrame):
    """Show which operator worked on which elements, spatially if possible."""
    named = site_df[site_df["has_name"] & (site_df["operator"] != "")].copy()
    if named.empty or named["operator"].nunique() < 2:
        return

    # Parse grid coords
    named["row"], named["col"] = zip(*named["element_name"].apply(parse_grid_name))
    grid = named.dropna(subset=["row", "col"])

    operators = named["operator"].value_counts()
    top_ops = operators[operators >= 5].index.tolist()
    if len(top_ops) < 2:
        return

    op_colors = dict(zip(top_ops, sns.color_palette("Set1", len(top_ops))))

    if len(grid) >= 10:
        # Spatial plot
        rows = sorted(grid["row"].unique())
        row_map = {r: i for i, r in enumerate(rows)}
        grid = grid.copy()
        grid["row_idx"] = grid["row"].map(row_map)

        fig, ax = plt.subplots(figsize=(14, 7))
        for op in top_ops:
            op_data = grid[grid["operator"] == op]
            ax.scatter(op_data["col"], op_data["row_idx"],
                       c=[op_colors[op]], s=50, alpha=0.7, label=op,
                       edgecolors="white", linewidth=0.5)
        other = grid[~grid["operator"].isin(top_ops)]
        if len(other) > 0:
            ax.scatter(other["col"], other["row_idx"],
                       c="gray", s=20, alpha=0.3, label="Other")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows)
        ax.set_xlabel("Column Number")
        ax.set_ylabel("Row")
        ax.invert_yaxis()
        ax.legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.set_title(f"Site {site_id} — Operator Spatial Assignment")
    else:
        # Bar chart: elements per operator
        fig, ax = plt.subplots(figsize=(12, 6))
        op_elements = named.groupby("operator")["element_name"].nunique().sort_values(ascending=True)
        op_elements = op_elements[op_elements >= 2]
        colors = [op_colors.get(op, "gray") for op in op_elements.index]
        ax.barh(op_elements.index, op_elements.values, color=colors)
        ax.set_xlabel("Unique Elements")
        ax.set_title(f"Site {site_id} — Elements per Operator")

    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{site_id}_operator_elements.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {site_id}_operator_elements.png")


# ── FIGURE E: Construction progress curve ─────────────────────────────────────

def fig_progress_curve(site_id: str, stats: pd.DataFrame):
    """Cumulative element completion over time."""
    if len(stats) < 5:
        return

    ordered = stats.sort_values("first_time")
    cumulative = range(1, len(ordered) + 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ordered["first_time"], cumulative, linewidth=2, color="#2c3e50")
    ax.fill_between(ordered["first_time"], 0, cumulative, alpha=0.1, color="#2c3e50")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Elements Completed")
    ax.set_title(f"Site {site_id} — Construction Progress ({len(stats)} elements)",
                 fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # Annotate rate changes
    dates = ordered["first_time"].values
    if len(dates) > 20:
        mid = len(dates) // 2
        first_half_days = (dates[mid] - dates[0]) / np.timedelta64(1, "D")
        second_half_days = (dates[-1] - dates[mid]) / np.timedelta64(1, "D")
        if first_half_days > 0 and second_half_days > 0:
            rate1 = mid / first_half_days
            rate2 = (len(dates) - mid) / second_half_days
            ax.text(0.02, 0.95,
                    f"First half: {rate1:.1f} elements/day\nSecond half: {rate2:.1f} elements/day",
                    transform=ax.transAxes, fontsize=10, va="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{site_id}_progress_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {site_id}_progress_curve.png")


# ── FIGURE F: Element duration vs construction order ──────────────────────────

def fig_duration_vs_order(site_id: str, stats: pd.DataFrame):
    """Does duration change as the project progresses?"""
    if len(stats) < 10:
        return

    tech_colors = {"SOB": "#2ecc71", "KELLY": "#3498db", "CUT": "#e74c3c",
                   "GRAB": "#9b59b6", "SCM": "#f39c12", "FREE": "#95a5a6"}

    ordered = stats.sort_values("first_time").copy()
    ordered["order"] = range(1, len(ordered) + 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    for tech in sorted(ordered["primary_technique"].unique()):
        t = ordered[ordered["primary_technique"] == tech]
        ax.scatter(t["order"], t["avg_duration_min"],
                   c=tech_colors.get(tech, "#7f8c8d"), s=20, alpha=0.5, label=tech)

    # Rolling average across all
    roll = ordered["avg_duration_min"].rolling(max(5, len(ordered) // 20),
                                               min_periods=3, center=True).mean()
    ax.plot(ordered["order"], roll, color="black", linewidth=2, label="Rolling avg")
    ax.set_xlabel("Element # (construction order)")
    ax.set_ylabel("Avg Duration per Element (min)")
    ax.set_title(f"Site {site_id} — Duration vs Construction Order")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{site_id}_duration_vs_order.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {site_id}_duration_vs_order.png")


# ── Main ──────────────────────────────────────────────────────────────────────

# Sites with enough data for meaningful analysis
SITES_TO_PLOT = [
    "3096", "5028", "1514", "LignanoSabbiadoro", "1508", "CS-Antwerpen",
    "1502", "1454", "1461", "1511", "PISA", "1501", "Paris L18.3 OA20",
    "1427",
]

def main():
    print("Loading data...")
    df = load_data()
    print(f"  {len(df)} traces loaded\n")

    for site_id in SITES_TO_PLOT:
        site_df = df[df["site_id"] == site_id]
        if len(site_df) < 5:
            continue

        stats = get_element_stats(site_df)
        if stats.empty:
            continue

        print(f"\n--- Site {site_id} ({len(site_df)} traces, {len(stats)} named elements) ---")

        fig_production_timeline(site_id, site_df, stats)
        fig_spatial_grid(site_id, stats)
        fig_grid_heatmap(site_id, stats)
        fig_operator_elements(site_id, site_df, stats)
        fig_progress_curve(site_id, stats)
        fig_duration_vs_order(site_id, stats)

    n_figs = len(list(FIG_DIR.glob("*.png")))
    print(f"\nDone! {n_figs} figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
