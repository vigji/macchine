"""Analyze actual machine sensor dynamics from Parquet trace files."""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np

from macchine.storage.catalog import get_trace_index

OUTPUT_DIR = Path("output")
TRACES_DIR = OUTPUT_DIR / "traces"
FIG_DIR = Path("reports/figures/dynamics")
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.0)

# Sensor name translation (German → English)
SENSOR_EN = {
    "Tiefe": "Depth", "Drehmoment": "Torque", "DrehmomentkNm": "Torque (kNm)",
    "DrehmomentProzent": "Torque (%)", "Drehzahl": "Rotation Speed",
    "Drehzahl Rohr": "Pipe Rotation", "Drehrate": "Rotation Rate",
    "KDK Drehzahl": "Kelly Drive Speed", "Drehrichtung": "Rotation Direction",
    "Seilkraft Hauptwinde": "Main Winch Force", "Seilkraft Hilfswinde": "Aux Winch Force",
    "Seilkraft": "Rope Force", "Seilkraft Fräswinde": "Cutter Winch Force",
    "Seilkraft Winde 2": "Winch 2 Force", "Hakenlast Hauptwinde": "Main Winch Hook Load",
    "Vorschub-KraftPM": "Feed Force", "Vorschub-Kraft": "Feed Force",
    "Vorschubgeschwindigkeit": "Feed Speed", "Vorschub Tiefe": "Feed Depth",
    "Eindringwiderstand": "Penetration Resistance",
    "Druck Pumpe 1": "Pump 1 Pressure", "Druck Pumpe 2": "Pump 2 Pressure",
    "Druck Pumpe 3": "Pump 3 Pressure", "Druck Pumpe 4": "Pump 4 Pressure",
    "Druck Pumpe": "Pump Pressure", "Druck FRL": "Cutter L Pressure",
    "Druck FRR": "Cutter R Pressure", "KDK Druck": "Kelly Drive Pressure",
    "Neigung X": "Incl. X", "Neigung Y": "Incl. Y",
    "Neigung X Mast": "Mast Incl. X", "Neigung Y Mast": "Mast Incl. Y",
    "Neigung gueltig": "Incl. Valid",
    "Betondruck": "Concrete Pressure", "Betonmenge": "Concrete Volume",
    "Gesamtbetonmenge": "Total Concrete", "Betondurchfluss": "Concrete Flow",
    "Betondruck unten": "Bottom Concrete Press.",
    "Susp.-Druck": "Slurry Pressure", "Susp.-Durchfl.": "Slurry Flow",
    "Susp.-Mg": "Slurry Volume", "Susp.-Druck unten": "Bottom Slurry Press.",
    "Susp.-Druck2": "Slurry Pressure 2", "Susp.-Durchfl.2": "Slurry Flow 2",
    "Susp.-Mg2": "Slurry Volume 2", "Susp.-Mg1+2": "Slurry Vol 1+2",
    "Temperatur FRL": "Cutter L Temp.", "Temperatur FRR": "Cutter R Temp.",
    "Temp. Verteilergetriebe": "Gearbox Temp.",
    "Drehzahl FRL": "Cutter L RPM", "Drehzahl FRR": "Cutter R RPM",
    "Drehzahl Pumpe": "Pump RPM", "Durchfluss Pumpe": "Pump Flow",
    "Durchfluss Vorlauf": "Feed Flow", "Durchfluss Ruecklauf": "Return Flow",
    "Fräszeit": "Cutting Time", "Auflast": "Surcharge",
    "Drehrichtung FRL": "Cutter L Dir.", "Drehrichtung FRR": "Cutter R Dir.",
    "Winkel DWS absolut": "DWS Angle Abs.", "Winkel DWS Messung": "DWS Angle Meas.",
    "Abweichung X": "Deviation X", "Abweichung Y": "Deviation Y",
    "Messpunktabw. X": "Meas. Dev. X", "Messpunktabw. Y": "Meas. Dev. Y",
    "Regenerationszeit": "Regen. Time",
    "Schliessgrad": "Closing Degree", "Schließzylinderdruck": "Close Cyl. Press.",
    "Verdrehwinkel": "Twist Angle",
    "Status Geraet": "Device Status", "Status DWS": "DWS Status",
    "Drehmomentstufen": "Torque Step", "Pitch": "Pitch",
    "Relativverschiebung": "Relative Displacement",
    "Menge pro Meter": "Volume per Meter", "Spuelluft_Druck": "Flush Air Pressure",
    "Rohrlaenge": "Casing Length", "Bohrgrenze": "Drill Limit",
    "Tiefe_Hauptwinde_GOK": "Main Winch Depth", "Tiefe_Bohrrohr_GOK": "Casing Depth",
    "Diff_Werkzeug_Bohrrohrunterkante": "Tool-Casing Diff",
    "Hauptwindengeschwindigkeit": "Main Winch Speed",
    "Stufenschalter_Hauptwinde": "Winch Gear Step",
    "x-Abweichung": "X Deviation", "y-Abweichung": "Y Deviation",
    "Tiefe Winde 2": "Winch 2 Depth",
    "Oeldruck Getriebe links": "Gearbox Oil L", "Oeldruck Getriebe rechts": "Gearbox Oil R",
    "Leckagedruck Getriebe links": "Leak Press. L", "Leckagedruck Getriebe rechts": "Leak Press. R",
    "Wassergehalt Getriebeoel FRL": "Water Content L", "Wassergehalt Getriebeoel FRR": "Water Content R",
    "Leckoeldruck": "Leak Oil Pressure", "Klappenprogrammnummer": "Flap Program",
    "Fraesenfortschritt": "Cutter Progress",
    "Drehboden": "Rotary Table", "Ausladung": "Boom Reach",
    "Strom Kanal 1": "Current Ch.1",
}

def en(name: str) -> str:
    return SENSOR_EN.get(name, name)


def load_index() -> pd.DataFrame:
    df = get_trace_index(OUTPUT_DIR)
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df["duration_min"] = df["duration_s"] / 60
    df["machine_slug"] = df["machine_slug"].replace("", "unidentified")
    return df


def find_trace_file(row: pd.Series) -> Path | None:
    """Find the parquet trace file for an index row."""
    site_dir = TRACES_DIR / row["site_id"] / row["machine_slug"]
    if not site_dir.exists():
        # Try unidentified
        site_dir = TRACES_DIR / row["site_id"]
        if not site_dir.exists():
            return None
    stem = Path(row["source_path"]).stem
    pq = site_dir / f"{stem}.parquet"
    if pq.exists():
        return pq
    # Search recursively
    for p in site_dir.rglob(f"{stem}.parquet"):
        return p
    return None


def sample_traces(idx: pd.DataFrame, machine: str, technique: str, n: int = 20) -> list[pd.DataFrame]:
    """Load n random trace DataFrames for a machine+technique combo."""
    subset = idx[(idx["machine_slug"] == machine) & (idx["technique"] == technique)]
    subset = subset[subset["duration_s"] > 60]  # At least 1 min
    if len(subset) == 0:
        return []
    sample_rows = subset.sample(min(n, len(subset)), random_state=42)
    traces = []
    for _, row in sample_rows.iterrows():
        pq = find_trace_file(row)
        if pq is not None:
            try:
                tdf = pd.read_parquet(pq)
                if len(tdf) > 10:
                    tdf["_machine"] = machine
                    tdf["_technique"] = technique
                    tdf["_element"] = row.get("element_name", "")
                    tdf["_duration_min"] = row["duration_min"]
                    traces.append(tdf)
            except Exception:
                pass
    return traces


# ── Key sensors per machine type ──────────────────────────────────────────────

# Rotary rigs (BG-series): depth, torque, rotation, feed force, pressures
ROTARY_SENSORS = ["Tiefe", "Drehmoment", "DrehmomentkNm", "KDK Drehzahl",
                  "Vorschub-KraftPM", "Vorschubgeschwindigkeit",
                  "Druck Pumpe 1", "Druck Pumpe 2",
                  "Seilkraft Hauptwinde", "Neigung X Mast", "Neigung Y Mast"]

# CUT machines (MC-86, CUBE0): depth, cutter speeds, pressures, temperatures
CUT_SENSORS = ["Tiefe", "Drehzahl FRL", "Drehzahl FRR", "Druck FRL", "Druck FRR",
               "Durchfluss Pumpe", "Seilkraft Fräswinde", "Vorschubgeschwindigkeit",
               "Temperatur FRL", "Temperatur FRR", "Temp. Verteilergetriebe",
               "Neigung X", "Neigung Y", "Abweichung X", "Abweichung Y"]

# GRAB (GB-50): depth, rope forces, closing degree, deviation
GRAB_SENSORS = ["Tiefe", "Seilkraft", "Hakenlast Hauptwinde", "Seilkraft Winde 2",
                "Schliessgrad", "Schließzylinderdruck", "Verdrehwinkel",
                "x-Abweichung", "y-Abweichung", "Neigung X", "Neigung Y",
                "Druck Pumpe 1", "Druck Pumpe 2"]

# SOB/SCM additional: concrete/slurry flow
SOB_EXTRA = ["Betondruck", "Betonmenge", "Gesamtbetonmenge", "Betondurchfluss"]
SCM_EXTRA = ["Susp.-Druck", "Susp.-Durchfl.", "Susp.-Mg"]


# ── Figure 1: Example trace profiles per machine ─────────────────────────────

def fig_example_traces(idx: pd.DataFrame):
    """Show one representative trace per machine model, with key sensors over time."""

    configs = [
        ("bg33v_5610", "KELLY", ROTARY_SENSORS, "BG-33-V — Kelly Drilling"),
        ("bg33v_5610", "SOB", ROTARY_SENSORS + SOB_EXTRA, "BG-33-V — CFA (SOB)"),
        ("bg33v_5610", "SCM", ROTARY_SENSORS + SCM_EXTRA, "BG-33-V — Soil Mixing (SCM)"),
        ("bg42v_5925", "KELLY", ROTARY_SENSORS, "BG-42-V — Kelly Drilling"),
        ("bg45v_4027", "KELLY", ROTARY_SENSORS, "BG-45-V — Kelly Drilling"),
        ("mc86_621", "CUT", CUT_SENSORS, "MC-86 — Diaphragm Wall Cutter"),
        ("cube0_482", "CUT", CUT_SENSORS, "CUBE0 — Diaphragm Wall Cutter"),
        ("gb50_601", "GRAB", GRAB_SENSORS, "GB-50 — Grab"),
        ("bg28h_6061", "GRAB", ROTARY_SENSORS, "BG-28H — Grab (rotary rig)"),
    ]

    for machine, tech, sensors, title in configs:
        traces = sample_traces(idx, machine, tech, n=5)
        if not traces:
            continue

        # Pick the trace with most data points
        trace = max(traces, key=len)
        available = [s for s in sensors if s in trace.columns]
        if not available:
            continue

        # Select up to 8 most interesting sensors (prefer ones with variation)
        interesting = []
        for s in available:
            vals = trace[s].dropna()
            if len(vals) > 10 and vals.std() > 0.01:
                interesting.append((s, vals.std() / max(abs(vals.mean()), 0.01)))
        interesting.sort(key=lambda x: x[1], reverse=True)
        plot_sensors = [s for s, _ in interesting[:8]]

        if len(plot_sensors) < 3:
            continue

        n_panels = len(plot_sensors)
        fig, axes = plt.subplots(n_panels, 1, figsize=(14, 2.2 * n_panels), sharex=True)
        if n_panels == 1:
            axes = [axes]

        time_min = np.arange(len(trace)) / 60  # assuming 1Hz

        for i, sensor in enumerate(plot_sensors):
            ax = axes[i]
            vals = trace[sensor].values
            ax.plot(time_min, vals, linewidth=0.6, color="steelblue")
            ax.set_ylabel(en(sensor), fontsize=9)
            ax.tick_params(labelsize=8)
            # Add range annotation
            valid = trace[sensor].dropna()
            if len(valid) > 0:
                ax.text(0.98, 0.95,
                        f"range: {valid.min():.1f}–{valid.max():.1f}",
                        transform=ax.transAxes, fontsize=7, ha="right", va="top",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.8))

        axes[-1].set_xlabel("Time (minutes)")
        elem = trace["_element"].iloc[0] if "_element" in trace.columns else ""
        dur = trace["_duration_min"].iloc[0] if "_duration_min" in trace.columns else 0
        fig.suptitle(f"{title}\n({elem} — {dur:.0f} min, {len(trace)} samples)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        fname = f"01_trace_{machine}_{tech}.png"
        fig.savefig(FIG_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {fname}")


# ── Figure 2: Sensor operating ranges per machine ────────────────────────────

def fig_sensor_ranges(idx: pd.DataFrame):
    """Boxplots of key sensor statistics across many traces per machine."""

    configs = [
        ("bg33v_5610", "KELLY", "BG-33-V Kelly"),
        ("bg45v_4027", "KELLY", "BG-45-V Kelly"),
        ("bg42v_5925", "KELLY", "BG-42-V Kelly"),
        ("bg33v_5610", "SOB", "BG-33-V SOB"),
        ("mc86_621", "CUT", "MC-86 CUT"),
        ("cube0_482", "CUT", "CUBE0 CUT"),
        ("gb50_601", "GRAB", "GB-50 GRAB"),
    ]

    # Key metrics to extract from each trace
    def extract_metrics(trace: pd.DataFrame) -> dict:
        m = {}
        for col in trace.columns:
            if col.startswith("_") or col == "timestamp":
                continue
            vals = trace[col].dropna()
            if len(vals) < 10:
                continue
            m[f"{col}__max"] = vals.max()
            m[f"{col}__mean"] = vals.mean()
            m[f"{col}__std"] = vals.std()
        return m

    all_records = []
    for machine, tech, label in configs:
        traces = sample_traces(idx, machine, tech, n=50)
        for t in traces:
            metrics = extract_metrics(t)
            metrics["_label"] = label
            metrics["_machine"] = machine
            metrics["_tech"] = tech
            all_records.append(metrics)

    if not all_records:
        return

    mdf = pd.DataFrame(all_records)

    # ── Depth comparison ──
    depth_col = "Tiefe__max"
    if depth_col in mdf.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        labels_with_data = [l for l in [c[2] for c in configs] if l in mdf["_label"].values]
        data = [mdf[mdf["_label"] == l][depth_col].dropna() for l in labels_with_data]
        data = [d for d, l in zip(data, labels_with_data) if len(d) > 0]
        labels_with_data = [l for d, l in zip([mdf[mdf["_label"] == l][depth_col].dropna() for l in labels_with_data], labels_with_data) if len(d) > 0]
        if data:
            bp = ax.boxplot(data, labels=labels_with_data, patch_artist=True)
            colors = sns.color_palette("Set2", len(data))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
            ax.set_ylabel("Max Depth Reached (m or sensor units)")
            ax.set_title("Maximum Depth per Trace — by Machine × Technique", fontsize=13, fontweight="bold")
            plt.xticks(rotation=30, ha="right")
            fig.tight_layout()
            fig.savefig(FIG_DIR / "02_depth_comparison.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print("  02_depth_comparison.png")

    # ── Torque comparison (rotary rigs) ──
    torque_col = "Drehmoment__max"
    if torque_col in mdf.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        rotary = mdf[mdf["_tech"].isin(["KELLY", "SOB", "SCM"])]
        labels_with_data = sorted(rotary["_label"].unique())
        data = [rotary[rotary["_label"] == l][torque_col].dropna() for l in labels_with_data]
        data_filt = [(d, l) for d, l in zip(data, labels_with_data) if len(d) > 0]
        if data_filt:
            data, labels_with_data = zip(*data_filt)
            bp = ax.boxplot(data, labels=labels_with_data, patch_artist=True)
            colors = sns.color_palette("Set2", len(data))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
            ax.set_ylabel("Max Torque (sensor units)")
            ax.set_title("Peak Torque per Trace — Rotary Rigs", fontsize=13, fontweight="bold")
            plt.xticks(rotation=30, ha="right")
            fig.tight_layout()
            fig.savefig(FIG_DIR / "02_torque_comparison.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print("  02_torque_comparison.png")

    # ── Pressure comparison ──
    press_col = "Druck Pumpe 1__max"
    if press_col in mdf.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        labels_with_data = sorted(mdf["_label"].unique())
        data = [mdf[mdf["_label"] == l][press_col].dropna() for l in labels_with_data]
        data_filt = [(d, l) for d, l in zip(data, labels_with_data) if len(d) > 0]
        if data_filt:
            data, labels_with_data = zip(*data_filt)
            bp = ax.boxplot(data, labels=labels_with_data, patch_artist=True)
            colors = sns.color_palette("Set2", len(data))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
            ax.set_ylabel("Max Pump 1 Pressure (sensor units)")
            ax.set_title("Peak Hydraulic Pressure (Pump 1) — by Machine", fontsize=13, fontweight="bold")
            plt.xticks(rotation=30, ha="right")
            fig.tight_layout()
            fig.savefig(FIG_DIR / "02_pressure_comparison.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print("  02_pressure_comparison.png")


# ── Figure 3: Depth-torque profiles ──────────────────────────────────────────

def fig_depth_torque(idx: pd.DataFrame):
    """Plot torque vs depth curves for rotary rigs — the 'fingerprint' of drilling."""

    configs = [
        ("bg33v_5610", "KELLY"),
        ("bg45v_4027", "KELLY"),
        ("bg42v_5925", "KELLY"),
        ("bg33v_5610", "SOB"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for i, (machine, tech) in enumerate(configs):
        ax = axes[i // 2, i % 2]
        traces = sample_traces(idx, machine, tech, n=15)
        if not traces:
            ax.set_title(f"{machine} {tech} — no data")
            continue

        for t in traces:
            depth_col = "Tiefe" if "Tiefe" in t.columns else None
            torque_col = "Drehmoment" if "Drehmoment" in t.columns else (
                "DrehmomentkNm" if "DrehmomentkNm" in t.columns else None)
            if depth_col and torque_col:
                d = pd.to_numeric(t[depth_col], errors="coerce").values
                tq = pd.to_numeric(t[torque_col], errors="coerce").values
                mask = np.isfinite(d) & np.isfinite(tq)
                if mask.sum() > 10:
                    ax.plot(tq[mask], d[mask], linewidth=0.5, alpha=0.4)

        ax.set_xlabel(f"Torque ({en(torque_col) if torque_col else '?'})")
        ax.set_ylabel("Depth")
        ax.invert_yaxis()
        ax.set_title(f"{machine.upper()} — {tech}", fontweight="bold")

    fig.suptitle("Depth vs Torque Profiles (each line = one element)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_depth_torque.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  03_depth_torque.png")


# ── Figure 4: CUT machine dynamics (cutter-specific) ─────────────────────────

def fig_cutter_dynamics(idx: pd.DataFrame):
    """MC-86 and CUBE0 cutter-specific sensor profiles."""
    for machine, label in [("mc86_621", "MC-86"), ("cube0_482", "CUBE0")]:
        traces = sample_traces(idx, machine, "CUT", n=10)
        if not traces:
            continue

        trace = max(traces, key=len)

        sensors = [
            ("Tiefe", "Depth"),
            ("Drehzahl FRL", "Cutter L RPM"), ("Drehzahl FRR", "Cutter R RPM"),
            ("Druck FRL", "Cutter L Pressure"), ("Druck FRR", "Cutter R Pressure"),
            ("Temperatur FRL", "Cutter L Temp"), ("Temperatur FRR", "Cutter R Temp"),
            ("Durchfluss Pumpe", "Pump Flow"), ("Seilkraft Fräswinde", "Cutter Winch Force"),
            ("Vorschubgeschwindigkeit", "Feed Speed"),
        ]
        available = [(s, lbl) for s, lbl in sensors if s in trace.columns]
        if len(available) < 4:
            continue

        n = len(available)
        fig, axes = plt.subplots(n, 1, figsize=(14, 2 * n), sharex=True)
        time_min = np.arange(len(trace)) / 60

        for i, (sensor, lbl) in enumerate(available):
            ax = axes[i]
            vals = trace[sensor].values
            ax.plot(time_min, vals, linewidth=0.5, color="steelblue")
            ax.set_ylabel(lbl, fontsize=9)
            valid = trace[sensor].dropna()
            if len(valid) > 0:
                ax.text(0.98, 0.92,
                        f"range: {valid.min():.1f}–{valid.max():.1f}",
                        transform=ax.transAxes, fontsize=7, ha="right", va="top",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.8))

        axes[-1].set_xlabel("Time (minutes)")
        elem = trace["_element"].iloc[0] if "_element" in trace.columns else ""
        dur = trace["_duration_min"].iloc[0] if "_duration_min" in trace.columns else 0
        fig.suptitle(f"{label} Cutter Dynamics — {elem} ({dur:.0f} min)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"04_cutter_{machine}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  04_cutter_{machine}.png")


# ── Figure 5: GB-50 grab dynamics ─────────────────────────────────────────────

def fig_grab_dynamics(idx: pd.DataFrame):
    """GB-50 grab-specific profiles: cyclic grab-lift-dump pattern."""
    traces = sample_traces(idx, "gb50_601", "GRAB", n=10)
    if not traces:
        return

    trace = max(traces, key=len)

    sensors = [
        ("Tiefe", "Depth"), ("Hakenlast Hauptwinde", "Hook Load"),
        ("Seilkraft", "Rope Force"), ("Seilkraft Winde 2", "Winch 2 Force"),
        ("Schliessgrad", "Closing Degree"), ("Schließzylinderdruck", "Close Cyl Press."),
        ("Verdrehwinkel", "Twist Angle"),
        ("Druck Pumpe 1", "Pump 1 Pressure"),
        ("x-Abweichung", "X Deviation"), ("y-Abweichung", "Y Deviation"),
    ]
    available = [(s, lbl) for s, lbl in sensors if s in trace.columns]
    if len(available) < 3:
        return

    n = len(available)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2 * n), sharex=True)
    time_min = np.arange(len(trace)) / 60

    for i, (sensor, lbl) in enumerate(available):
        ax = axes[i]
        vals = trace[sensor].values
        ax.plot(time_min, vals, linewidth=0.5, color="steelblue")
        ax.set_ylabel(lbl, fontsize=9)
        valid = trace[sensor].dropna()
        if len(valid) > 0:
            ax.text(0.98, 0.92,
                    f"range: {valid.min():.1f}–{valid.max():.1f}",
                    transform=ax.transAxes, fontsize=7, ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.8))

    axes[-1].set_xlabel("Time (minutes)")
    elem = trace["_element"].iloc[0] if "_element" in trace.columns else ""
    dur = trace["_duration_min"].iloc[0] if "_duration_min" in trace.columns else 0
    fig.suptitle(f"GB-50 Grab Dynamics — {elem} ({dur:.0f} min)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_grab_gb50.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  05_grab_gb50.png")


# ── Figure 6: Sensor stats evolution over project life ────────────────────────

def fig_sensor_evolution(idx: pd.DataFrame):
    """For each machine, track how key sensor stats evolve over months."""

    configs = [
        ("bg33v_5610", "KELLY", ["Drehmoment", "Druck Pumpe 1", "Tiefe"]),
        ("mc86_621", "CUT", ["Druck FRL", "Drehzahl FRL", "Tiefe"]),
        ("gb50_601", "GRAB", ["Seilkraft", "Druck Pumpe 1", "Tiefe"]),
    ]

    for machine, tech, sensors in configs:
        sub = idx[(idx["machine_slug"] == machine) & (idx["technique"] == tech)].copy()
        sub = sub.sort_values("start_time")
        if len(sub) < 20:
            continue

        # Sample up to 200 traces spread across time
        if len(sub) > 200:
            sub = sub.iloc[np.linspace(0, len(sub) - 1, 200, dtype=int)]

        records = []
        for _, row in sub.iterrows():
            pq = find_trace_file(row)
            if pq is None:
                continue
            try:
                t = pd.read_parquet(pq)
            except Exception:
                continue
            rec = {"start_time": row["start_time"], "duration_min": row["duration_min"]}
            for s in sensors:
                if s in t.columns:
                    vals = t[s].dropna()
                    if len(vals) > 0:
                        rec[f"{s}_max"] = vals.max()
                        rec[f"{s}_mean"] = vals.mean()
                        rec[f"{s}_p95"] = vals.quantile(0.95)
            records.append(rec)

        if len(records) < 10:
            continue
        rdf = pd.DataFrame(records)

        n = len(sensors)
        fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), sharex=True)
        if n == 1:
            axes = [axes]

        for i, s in enumerate(sensors):
            ax = axes[i]
            max_col = f"{s}_max"
            mean_col = f"{s}_mean"
            p95_col = f"{s}_p95"
            if max_col in rdf.columns:
                ax.scatter(rdf["start_time"], rdf[max_col], s=10, alpha=0.4, label="Max", color="red")
            if p95_col in rdf.columns:
                ax.scatter(rdf["start_time"], rdf[p95_col], s=10, alpha=0.4, label="P95", color="orange")
            if mean_col in rdf.columns:
                ax.scatter(rdf["start_time"], rdf[mean_col], s=10, alpha=0.4, label="Mean", color="steelblue")
                # Rolling trend
                sorted_rdf = rdf.sort_values("start_time")
                rolling = sorted_rdf[mean_col].rolling(20, min_periods=5, center=True).mean()
                ax.plot(sorted_rdf["start_time"], rolling, color="navy", linewidth=2, label="20-trace rolling mean")
            ax.set_ylabel(en(s))
            ax.legend(fontsize=8, loc="upper right")

        axes[-1].set_xlabel("Date")
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        fig.suptitle(f"{machine.upper()} ({tech}) — Sensor Statistics Over Time",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"06_evolution_{machine}_{tech}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  06_evolution_{machine}_{tech}.png")


# ── Figure 7: Machine sensor fingerprint comparison ───────────────────────────

def fig_sensor_fingerprint(idx: pd.DataFrame):
    """Compare sensor distributions across machines doing the same technique."""

    # Compare all KELLY machines
    kelly_machines = ["bg33v_5610", "bg45v_4027", "bg42v_5925", "bg30v_2872"]
    common_sensors = ["Tiefe", "Drehmoment", "Druck Pumpe 1", "KDK Drehzahl",
                      "Vorschub-KraftPM", "Seilkraft Hauptwinde"]

    all_data = []
    for machine in kelly_machines:
        traces = sample_traces(idx, machine, "KELLY", n=30)
        for t in traces:
            for s in common_sensors:
                if s in t.columns:
                    vals = t[s].dropna()
                    if len(vals) > 10:
                        all_data.append({
                            "machine": machine, "sensor": en(s),
                            "max": vals.max(), "mean": vals.mean(),
                            "std": vals.std(), "p95": vals.quantile(0.95),
                        })

    if not all_data:
        return

    adf = pd.DataFrame(all_data)

    n_sensors = adf["sensor"].nunique()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, sensor in enumerate(sorted(adf["sensor"].unique())):
        if i >= 6:
            break
        ax = axes[i]
        s_df = adf[adf["sensor"] == sensor]
        machines = sorted(s_df["machine"].unique())
        data = [s_df[s_df["machine"] == m]["max"].dropna().values for m in machines]
        data = [d for d in data if len(d) > 0]
        machines = [m for m, d in zip(machines, [s_df[s_df["machine"] == m]["max"].dropna().values for m in machines]) if len(d) > 0]
        if data:
            bp = ax.boxplot(data, labels=machines, patch_artist=True)
            colors = sns.color_palette("Set2", len(data))
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c)
        ax.set_title(sensor, fontweight="bold")
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Kelly Drilling — Sensor Max Values Across Machines\n(each box = 30 traces)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_kelly_fingerprint.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  07_kelly_fingerprint.png")


# ── Figure 8: Operational phases within a single trace ────────────────────────

def fig_operational_phases(idx: pd.DataFrame):
    """Show how a single KELLY trace breaks into drilling/lifting/concreting phases."""

    # Find a good KELLY trace (medium duration, many sensors)
    sub = idx[(idx["machine_slug"] == "bg33v_5610") & (idx["technique"] == "KELLY")]
    sub = sub[(sub["duration_s"] > 3600) & (sub["duration_s"] < 10800)]  # 1-3 hours
    if sub.empty:
        sub = idx[(idx["machine_slug"] == "bg33v_5610") & (idx["technique"] == "KELLY")]
        sub = sub[sub["duration_s"] > 1800]

    traces = sample_traces(idx.loc[sub.index], "bg33v_5610", "KELLY", n=5)
    if not traces:
        return

    trace = max(traces, key=len)

    sensors = [
        ("Tiefe", "Depth (m)"),
        ("Drehmoment", "Torque"),
        ("KDK Drehzahl", "Kelly Drive RPM"),
        ("Vorschub-KraftPM", "Feed Force"),
        ("Druck Pumpe 1", "Pump 1 Pressure"),
        ("Seilkraft Hauptwinde", "Main Winch Force"),
    ]
    available = [(s, lbl) for s, lbl in sensors if s in trace.columns and trace[s].std() > 0.01]

    if len(available) < 4:
        return

    fig, axes = plt.subplots(len(available), 1, figsize=(16, 2.5 * len(available)), sharex=True)
    time_min = np.arange(len(trace)) / 60

    for i, (sensor, lbl) in enumerate(available):
        ax = axes[i]
        vals = trace[sensor].values
        ax.plot(time_min, vals, linewidth=0.5, color="steelblue")
        ax.set_ylabel(lbl, fontsize=10)

        # Highlight "active" periods (depth changing)
        if sensor == "Tiefe":
            depth = trace[sensor].values
            depth_diff = np.abs(np.diff(depth, prepend=depth[0]))
            active = depth_diff > 0.01
            for a in axes:
                for start, end in _contiguous_regions(active):
                    a.axvspan(time_min[start], time_min[min(end, len(time_min) - 1)],
                              alpha=0.05, color="green")

    axes[-1].set_xlabel("Time (minutes)")
    fig.suptitle(
        f"BG-33-V Kelly Trace — Operational Phases\n"
        f"({trace['_element'].iloc[0]}, {trace['_duration_min'].iloc[0]:.0f} min, green shading = depth changing)",
        fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_operational_phases.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  08_operational_phases.png")


def _contiguous_regions(mask):
    """Find contiguous True regions in a boolean array."""
    d = np.diff(mask.astype(int))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1
    if mask[0]:
        starts = np.concatenate([[0], starts])
    if mask[-1]:
        ends = np.concatenate([ends, [len(mask)]])
    return list(zip(starts, ends))


# ── Figure 9: Inclination stability per machine ──────────────────────────────

def fig_inclination_stats(idx: pd.DataFrame):
    """Compare mast inclination stability across machines — proxy for setup quality."""

    machines = ["bg33v_5610", "bg45v_4027", "bg42v_5925", "bg30v_2872", "bg28h_6061"]
    incl_x_col = "Neigung X Mast"
    incl_y_col = "Neigung Y Mast"

    records = []
    for machine in machines:
        traces = sample_traces(idx, machine, "KELLY", n=40)
        if not traces:
            traces = sample_traces(idx, machine, "GRAB", n=40)
        for t in traces:
            for col, axis in [(incl_x_col, "X"), (incl_y_col, "Y")]:
                if col in t.columns:
                    vals = t[col].dropna()
                    if len(vals) > 10:
                        records.append({
                            "machine": machine, "axis": axis,
                            "mean_incl": vals.mean(), "max_incl": vals.abs().max(),
                            "std_incl": vals.std(),
                        })

    if not records:
        return

    rdf = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    for i, (metric, label) in enumerate([
        ("mean_incl", "Mean Inclination"), ("max_incl", "Max |Inclination|"), ("std_incl", "Incl. Std Dev")
    ]):
        ax = axes[i]
        machines_present = sorted(rdf["machine"].unique())
        for j, m in enumerate(machines_present):
            m_data = rdf[rdf["machine"] == m]
            x_vals = m_data[m_data["axis"] == "X"][metric]
            y_vals = m_data[m_data["axis"] == "Y"][metric]
            offset = j * 2
            if len(x_vals) > 0:
                bp = ax.boxplot([x_vals.values], positions=[offset], widths=0.6,
                                patch_artist=True, boxprops=dict(facecolor="skyblue"))
            if len(y_vals) > 0:
                bp = ax.boxplot([y_vals.values], positions=[offset + 0.7], widths=0.6,
                                patch_artist=True, boxprops=dict(facecolor="salmon"))

        ax.set_xticks([j * 2 + 0.35 for j in range(len(machines_present))])
        ax.set_xticklabels(machines_present, rotation=30, ha="right", fontsize=9)
        ax.set_title(label, fontweight="bold")
        if i == 0:
            ax.legend(handles=[
                plt.Rectangle((0, 0), 1, 1, fc="skyblue", label="X axis"),
                plt.Rectangle((0, 0), 1, 1, fc="salmon", label="Y axis"),
            ], fontsize=8)

    fig.suptitle("Mast Inclination Stability by Machine (blue=X, red=Y)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_inclination_stability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  09_inclination_stability.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading index...")
    idx = load_index()
    print(f"  {len(idx)} traces\n")

    print("Generating machine dynamics figures...")
    fig_example_traces(idx)
    fig_sensor_ranges(idx)
    fig_depth_torque(idx)
    fig_cutter_dynamics(idx)
    fig_grab_dynamics(idx)
    fig_sensor_evolution(idx)
    fig_sensor_fingerprint(idx)
    fig_operational_phases(idx)
    fig_inclination_stats(idx)

    n = len(list(FIG_DIR.glob("*.png")))
    print(f"\nDone! {n} figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
