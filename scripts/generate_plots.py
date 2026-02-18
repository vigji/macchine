"""Generate all report figures from the converted dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np

from macchine.storage.catalog import get_trace_index
from macchine.analysis.plot_utils import plot_with_gaps, add_site_markers

OUTPUT_DIR = Path("output")
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Style
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("deep", 12)
MACHINE_COLORS = {}


def load_data() -> pd.DataFrame:
    df = get_trace_index(OUTPUT_DIR)
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df = df.dropna(subset=["start_time"])
    df["duration_min"] = df["duration_s"] / 60
    df["month"] = df["start_time"].dt.to_period("M")
    df["date"] = df["start_time"].dt.date
    # Replace empty machine_slug with "unidentified"
    df["machine_slug"] = df["machine_slug"].replace("", "unidentified")
    # Build color map
    slugs = sorted(df["machine_slug"].unique())
    for i, s in enumerate(slugs):
        MACHINE_COLORS[s] = PALETTE[i % len(PALETTE)]
    return df


# ── Figure 1: Technique distribution ──────────────────────────────────────────

def fig_technique_distribution(df: pd.DataFrame):
    tech_counts = df["technique"].value_counts()
    tech_hours = df.groupby("technique")["duration_min"].sum() / 60

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Trace counts
    colors = sns.color_palette("Set2", len(tech_counts))
    bars = axes[0].barh(tech_counts.index, tech_counts.values, color=colors)
    axes[0].set_xlabel("Number of Traces")
    axes[0].set_title("Traces per Technique")
    axes[0].invert_yaxis()
    for bar, val in zip(bars, tech_counts.values):
        axes[0].text(val + 20, bar.get_y() + bar.get_height() / 2,
                     f"{val:,}", va="center", fontsize=10)

    # Recording hours
    bars = axes[1].barh(tech_hours.index, tech_hours.values, color=colors)
    axes[1].set_xlabel("Recording Hours")
    axes[1].set_title("Recording Hours per Technique")
    axes[1].invert_yaxis()
    for bar, val in zip(bars, tech_hours.values):
        axes[1].text(val + 10, bar.get_y() + bar.get_height() / 2,
                     f"{val:,.0f}h", va="center", fontsize=10)

    fig.suptitle("Technique Distribution", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_technique_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  01_technique_distribution.png")


# ── Figure 2: Machine utilization heatmap ─────────────────────────────────────

def fig_utilization_heatmap(df: pd.DataFrame):
    # Filter to main activity period (2024+)
    df2 = df[df["start_time"] >= "2024-01-01"].copy()
    df2["month_str"] = df2["start_time"].dt.to_period("M").astype(str)

    pivot = df2.groupby(["machine_slug", "month_str"]).size().unstack(fill_value=0)
    # Sort machines by total traces
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", linewidths=0.5,
                ax=ax, cbar_kws={"label": "Traces"})
    ax.set_title("Monthly Trace Count per Machine (2024–2026)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Machine")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_utilization_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  02_utilization_heatmap.png")


# ── Figure 3: Duration distributions by technique ────────────────────────────

def fig_duration_by_technique(df: pd.DataFrame):
    # Cap at 500 min for readability
    df2 = df[df["technique"].isin(["SOB", "KELLY", "CUT", "GRAB", "SCM"])].copy()
    df2["duration_capped"] = df2["duration_min"].clip(upper=500)

    fig, ax = plt.subplots(figsize=(12, 6))
    order = ["SOB", "SCM", "GRAB", "CUT", "KELLY"]
    sns.violinplot(data=df2, x="technique", y="duration_capped", order=order,
                   hue="technique", hue_order=order, inner="quartile",
                   palette="Set2", ax=ax, cut=0, legend=False)
    ax.set_xlabel("Technique")
    ax.set_ylabel("Duration (minutes)")
    ax.set_title("Element Duration Distribution by Technique", fontsize=14, fontweight="bold")
    ax.axhline(y=60, color="gray", linestyle="--", alpha=0.5, label="1 hour")
    ax.axhline(y=180, color="gray", linestyle=":", alpha=0.5, label="3 hours")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_duration_by_technique.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  03_duration_by_technique.png")


# ── Figure 4: Fleet timeline (Gantt-style) ───────────────────────────────────

def fig_fleet_timeline(df: pd.DataFrame):
    df2 = df[df["start_time"] >= "2023-01-01"].copy()

    machines = sorted(df2["machine_slug"].unique())

    fig, ax = plt.subplots(figsize=(16, 7))
    for i, slug in enumerate(machines):
        m_df = df2[df2["machine_slug"] == slug]
        # Group by site to show site assignments
        for site_id in m_df["site_id"].unique():
            s_df = m_df[m_df["site_id"] == site_id]
            dates = s_df["start_time"].sort_values()
            if len(dates) < 2:
                ax.scatter(dates.iloc[0], i, s=30, color=MACHINE_COLORS.get(slug, "gray"), zorder=3)
            else:
                ax.barh(i, (dates.max() - dates.min()).days,
                        left=dates.min(), height=0.6,
                        color=MACHINE_COLORS.get(slug, "gray"), alpha=0.7, edgecolor="white")
                # Label the site
                mid = dates.min() + (dates.max() - dates.min()) / 2
                ax.text(mid, i, site_id, ha="center", va="center", fontsize=7,
                        fontweight="bold", color="white")

    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels(machines)
    ax.set_xlabel("Date")
    ax.set_title("Fleet Deployment Timeline", fontsize=14, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha="right")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_fleet_timeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  04_fleet_timeline.png")


# ── Figure 5: Data quality radar/bar chart ────────────────────────────────────

def fig_quality_scores(df: pd.DataFrame):
    # Recompute quality scores from data
    sites = sorted(df["site_id"].unique())
    records = []
    for sid in sites:
        s_df = df[df["site_id"] == sid]
        n = len(s_df)
        machine_pct = (s_df["machine_slug"] != "unidentified").sum() / n * 100
        operator_pct = (s_df["operator"].notna() & (s_df["operator"] != "")).sum() / n * 100
        naming_pct = (s_df["element_name"].notna() & (s_df["element_name"] != "")).sum() / n * 100
        sensor_pct = min(100, (s_df["sensor_count"] > 0).sum() / n * 100)
        overall = 0.25 * machine_pct + 0.20 * operator_pct + 0.20 * naming_pct + 0.20 * sensor_pct
        # Temporal not easy to recompute, use approximate
        records.append({
            "site_id": sid, "traces": n,
            "machine": machine_pct, "operator": operator_pct,
            "naming": naming_pct, "sensors": sensor_pct,
            "overall": overall,
        })

    qdf = pd.DataFrame(records).sort_values("overall", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ["#e74c3c" if v < 70 else "#f39c12" if v < 85 else "#27ae60" for v in qdf["overall"]]
    bars = ax.barh(qdf["site_id"], qdf["overall"], color=colors)
    ax.set_xlabel("Quality Score (%)")
    ax.set_title("Data Quality Score by Site", fontsize=14, fontweight="bold")
    ax.axvline(x=85, color="green", linestyle="--", alpha=0.5, label="Good (85%)")
    ax.axvline(x=70, color="orange", linestyle="--", alpha=0.5, label="Acceptable (70%)")
    ax.legend()
    for bar, val in zip(bars, qdf["overall"]):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}%", va="center", fontsize=9)
    ax.set_xlim(0, 105)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_quality_scores.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  05_quality_scores.png")


# ── Figure 6: Monthly production timeline ─────────────────────────────────────

def fig_monthly_production(df: pd.DataFrame):
    df2 = df[df["start_time"] >= "2024-01-01"].copy()
    df2["month_str"] = df2["start_time"].dt.to_period("M").astype(str)

    monthly = df2.groupby(["month_str", "machine_slug"]).agg(
        traces=("source_path", "count"),
        hours=("duration_min", lambda x: x.sum() / 60),
    ).reset_index()

    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # Stacked bar: trace counts
    pivot_traces = monthly.pivot_table(index="month_str", columns="machine_slug",
                                        values="traces", fill_value=0)
    pivot_traces.plot.bar(stacked=True, ax=axes[0], color=[MACHINE_COLORS.get(c, "gray") for c in pivot_traces.columns],
                          width=0.8)
    axes[0].set_ylabel("Traces")
    axes[0].set_title("Monthly Trace Production by Machine", fontsize=13, fontweight="bold")
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    # Stacked bar: recording hours
    pivot_hours = monthly.pivot_table(index="month_str", columns="machine_slug",
                                       values="hours", fill_value=0)
    pivot_hours.plot.bar(stacked=True, ax=axes[1], color=[MACHINE_COLORS.get(c, "gray") for c in pivot_hours.columns],
                         width=0.8)
    axes[1].set_ylabel("Recording Hours")
    axes[1].set_title("Monthly Recording Hours by Machine", fontsize=13, fontweight="bold")
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    # Format x ticks
    axes[1].set_xticklabels(pivot_traces.index, rotation=45, ha="right")
    axes[1].set_xlabel("Month")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_monthly_production.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  06_monthly_production.png")


# ── Figure 7: Operator comparison ─────────────────────────────────────────────

def fig_operator_comparison(df: pd.DataFrame):
    df2 = df[df["operator"].notna() & (df["operator"] != "")].copy()
    op_stats = df2.groupby("operator").agg(
        traces=("source_path", "count"),
        total_hours=("duration_s", lambda x: x.sum() / 3600),
        avg_dur=("duration_min", "mean"),
        median_dur=("duration_min", "median"),
    ).reset_index()
    # Only operators with >= 50 traces
    op_stats = op_stats[op_stats["traces"] >= 50].sort_values("total_hours", ascending=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    # Total hours
    axes[0].barh(op_stats["operator"], op_stats["total_hours"], color=PALETTE[0])
    axes[0].set_xlabel("Total Recording Hours")
    axes[0].set_title("Total Hours", fontweight="bold")

    # Trace count
    axes[1].barh(op_stats["operator"], op_stats["traces"], color=PALETTE[1])
    axes[1].set_xlabel("Number of Traces")
    axes[1].set_title("Trace Count", fontweight="bold")

    # Avg vs median duration
    y = range(len(op_stats))
    axes[2].barh([i - 0.15 for i in y], op_stats["avg_dur"], height=0.3,
                 label="Mean", color=PALETTE[2], alpha=0.8)
    axes[2].barh([i + 0.15 for i in y], op_stats["median_dur"], height=0.3,
                 label="Median", color=PALETTE[3], alpha=0.8)
    axes[2].set_yticks(list(y))
    axes[2].set_yticklabels(op_stats["operator"])
    axes[2].set_xlabel("Duration (min)")
    axes[2].set_title("Avg vs Median Duration", fontweight="bold")
    axes[2].legend()

    fig.suptitle("Operator Comparison (operators with 50+ traces)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_operator_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  07_operator_comparison.png")


# ── Figure 8: Machine daily recording intensity ──────────────────────────────

def fig_daily_intensity(df: pd.DataFrame):
    df2 = df[df["start_time"] >= "2024-01-01"].copy()
    # Exclude unidentified
    df2 = df2[df2["machine_slug"] != "unidentified"]

    daily = df2.groupby(["date", "machine_slug"]).agg(
        hours=("duration_min", lambda x: x.sum() / 60),
        elements=("source_path", "count"),
    ).reset_index()

    top_machines = daily.groupby("machine_slug")["hours"].sum().nlargest(6).index.tolist()
    daily = daily[daily["machine_slug"].isin(top_machines)]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey="row")

    for i, slug in enumerate(top_machines):
        ax = axes[i // 3, i % 3]
        m_df = daily[daily["machine_slug"] == slug].sort_values("date")
        dates = pd.to_datetime(m_df["date"])
        ax.bar(dates, m_df["hours"], width=1.5,
               color=MACHINE_COLORS.get(slug, "gray"), alpha=0.7)
        ax.set_title(slug, fontweight="bold")
        ax.set_ylabel("Hours" if i % 3 == 0 else "")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.set_ylim(0, 24)
        ax.axhline(y=8, color="gray", linestyle="--", alpha=0.4)
        ax.axhline(y=16, color="gray", linestyle=":", alpha=0.3)

    fig.suptitle("Daily Recording Intensity (hours/day)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_daily_intensity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  08_daily_intensity.png")


# ── Figure 9: Site comparison ─────────────────────────────────────────────────

def fig_site_comparison(df: pd.DataFrame):
    site_stats = df.groupby("site_id").agg(
        traces=("source_path", "count"),
        hours=("duration_s", lambda x: x.sum() / 3600),
        techniques=("technique", "nunique"),
        machines=("machine_slug", "nunique"),
    ).reset_index().sort_values("traces", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Traces + hours
    axes[0].barh(site_stats["site_id"], site_stats["traces"], color=PALETTE[0], alpha=0.8, label="Traces")
    axes[0].set_xlabel("Count")
    axes[0].set_title("Traces per Site", fontweight="bold")
    for _, row in site_stats.iterrows():
        axes[0].text(row["traces"] + 10, row["site_id"],
                     f'{row["traces"]:,}', va="center", fontsize=8)

    axes[1].barh(site_stats["site_id"], site_stats["hours"], color=PALETTE[2], alpha=0.8)
    axes[1].set_xlabel("Hours")
    axes[1].set_title("Recording Hours per Site", fontweight="bold")
    for _, row in site_stats.iterrows():
        axes[1].text(row["hours"] + 5, row["site_id"],
                     f'{row["hours"]:.0f}h', va="center", fontsize=8)

    fig.suptitle("Site Comparison", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_site_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  09_site_comparison.png")


# ── Figure 10: Anomaly scatter ────────────────────────────────────────────────

def fig_anomaly_scatter(df: pd.DataFrame):
    df2 = df[df["technique"].isin(["SOB", "KELLY", "CUT", "GRAB", "SCM"])].copy()
    # Compute z-scores per machine
    df2["duration_zscore"] = 0.0
    for slug in df2["machine_slug"].unique():
        mask = df2["machine_slug"] == slug
        mean = df2.loc[mask, "duration_min"].mean()
        std = df2.loc[mask, "duration_min"].std()
        if std > 0:
            df2.loc[mask, "duration_zscore"] = (df2.loc[mask, "duration_min"] - mean) / std

    df2["is_anomaly"] = df2["duration_zscore"].abs() > 3

    fig, ax = plt.subplots(figsize=(14, 7))
    normal = df2[~df2["is_anomaly"]]
    anomalous = df2[df2["is_anomaly"]]

    ax.scatter(normal["start_time"], normal["duration_min"],
               s=3, alpha=0.15, c="steelblue", label=f"Normal ({len(normal):,})")
    ax.scatter(anomalous["start_time"], anomalous["duration_min"],
               s=15, alpha=0.6, c="red", marker="x", label=f"Anomaly ({len(anomalous):,})")

    ax.set_xlabel("Date")
    ax.set_ylabel("Duration (minutes)")
    ax.set_title("Trace Duration Over Time — Anomaly Detection", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 800)
    ax.legend(markerscale=3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "10_anomaly_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  10_anomaly_scatter.png")


# ── Figure 11: Sensor count distribution ──────────────────────────────────────

def fig_sensor_distribution(df: pd.DataFrame):
    df2 = df[df["technique"].isin(["SOB", "KELLY", "CUT", "GRAB", "SCM"])].copy()

    fig, ax = plt.subplots(figsize=(12, 6))
    order = ["GRAB", "KELLY", "SOB", "SCM", "CUT"]
    sns.boxplot(data=df2, x="technique", y="sensor_count", order=order,
                hue="technique", hue_order=order, palette="Set2",
                ax=ax, fliersize=2, legend=False)
    ax.set_xlabel("Technique")
    ax.set_ylabel("Sensor Count per Trace")
    ax.set_title("Sensor Count Distribution by Technique", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "11_sensor_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  11_sensor_distribution.png")


# ── Figure 12: Machine degradation (rolling avg duration) ────────────────────

def fig_rolling_degradation(df: pd.DataFrame):
    df2 = df[df["start_time"] >= "2024-01-01"].copy()
    df2 = df2[df2["machine_slug"] != "unidentified"]

    top_machines = df2.groupby("machine_slug").size().nlargest(6).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for i, slug in enumerate(top_machines):
        ax = axes[i // 3, i % 3]
        m_df = df2[df2["machine_slug"] == slug].sort_values("start_time").copy()

        # Add site-change markers so apparent "degradation" from site changes is visible
        if "site_id" in m_df.columns:
            add_site_markers(ax, m_df, site_col="site_id", time_col="start_time")

        m_df = m_df.set_index("start_time")

        # Plot raw durations as dots
        ax.scatter(m_df.index, m_df["duration_min"], s=5, alpha=0.2,
                   color=MACHINE_COLORS.get(slug, "gray"))

        # Rolling mean (30-day window) — gap-aware to avoid interpolating across inactive periods
        rolling = m_df["duration_min"].rolling("30D", min_periods=10)
        rmean = rolling.mean()
        rstd = rolling.std()
        plot_with_gaps(ax, rmean.index, rmean.values, max_gap_days=14,
                       color="red", linewidth=2, label="30-day rolling mean")
        # Rolling std band — also gap-aware
        ax.fill_between(m_df.index, rmean - rstd, rmean + rstd,
                        color="red", alpha=0.1)

        ax.set_title(slug, fontweight="bold")
        ax.set_ylabel("Duration (min)" if i % 3 == 0 else "")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.legend(fontsize=8)

    fig.suptitle("Machine Performance: 30-Day Rolling Average Duration", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "12_rolling_degradation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  12_rolling_degradation.png")


# ── Figure 13: Technique per site heatmap ─────────────────────────────────────

def fig_technique_site_heatmap(df: pd.DataFrame):
    pivot = df.groupby(["site_id", "technique"]).size().unstack(fill_value=0)
    # Sort by total
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    # Keep only main techniques
    main_techs = ["SOB", "KELLY", "CUT", "GRAB", "SCM"]
    pivot = pivot[[t for t in main_techs if t in pivot.columns]]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="Blues", linewidths=0.5, ax=ax,
                cbar_kws={"label": "Traces"})
    ax.set_title("Technique Usage by Site", fontsize=14, fontweight="bold")
    ax.set_xlabel("Technique")
    ax.set_ylabel("Site")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "13_technique_site_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  13_technique_site_heatmap.png")


# ── Figure 14: Format distribution (JSON vs DAT) ─────────────────────────────

def fig_format_distribution(df: pd.DataFrame):
    fmt_site = df.groupby(["site_id", "format"]).size().unstack(fill_value=0)
    fmt_site = fmt_site.loc[fmt_site.sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(12, 7))
    fmt_site.plot.barh(stacked=True, ax=ax, color={"json": PALETTE[0], "dat": PALETTE[1]})
    ax.set_xlabel("Number of Traces")
    ax.set_title("File Format Distribution by Site", fontsize=14, fontweight="bold")
    ax.legend(title="Format")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "14_format_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  14_format_distribution.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    df = load_data()
    print(f"  {len(df)} traces loaded\n")

    print("Generating figures...")
    fig_technique_distribution(df)
    fig_utilization_heatmap(df)
    fig_duration_by_technique(df)
    fig_fleet_timeline(df)
    fig_quality_scores(df)
    fig_monthly_production(df)
    fig_operator_comparison(df)
    fig_daily_intensity(df)
    fig_site_comparison(df)
    fig_anomaly_scatter(df)
    fig_sensor_distribution(df)
    fig_rolling_degradation(df)
    fig_technique_site_heatmap(df)
    fig_format_distribution(df)
    print(f"\nDone! {len(list(FIG_DIR.glob('*.png')))} figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
