"""Generate a Technique Atlas: detailed explanation of each construction technique
with annotated example traces showing all available sensors.

For each technique (SOB, KELLY, GRAB, CUT, SCM):
- Explain the physical process
- Show 1-2 exemplary traces with all sensors plotted
- Annotate construction phases on the plots
- Group sensors by category for clarity

Output: reports/12_technique_atlas.md + reports/figures/techniques/
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np
import yaml
import warnings

from macchine.harmonize.calibration import (
    is_calibrated,
    get_unit,
    get_display_label,
    get_axis_label,
    clean_sentinels_df,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

OUTPUT_DIR = Path("output")
FIG_DIR = Path("reports/figures/techniques")
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = Path("reports/12_technique_atlas.md")

sns.set_theme(style="whitegrid", font_scale=0.9)

# Load sensor definitions
SENSOR_DEFS_PATH = Path("macchine/harmonize/sensor_definitions.yaml")
with open(SENSOR_DEFS_PATH) as f:
    SENSOR_DEFS = yaml.safe_load(f)

# Map sensor name -> (canonical, category)
SENSOR_MAP = {}
for name, info in SENSOR_DEFS.items():
    SENSOR_MAP[name] = (info["canonical"], info["category"])

# Sensor display order by category
CATEGORY_ORDER = [
    "position", "torque", "rotation", "force", "pressure",
    "flow", "speed", "inclination", "deviation",
    "temperature", "lubrication", "cutter", "status", "electrical", "other",
]

CATEGORY_LABELS = {
    "position": "Depth / Position",
    "torque": "Torque",
    "rotation": "Rotation",
    "force": "Force / Load",
    "pressure": "Hydraulic Pressure",
    "flow": "Flow / Volume",
    "speed": "Speed",
    "inclination": "Inclination / Tilt",
    "deviation": "Deviation / Accuracy",
    "temperature": "Temperature",
    "lubrication": "Oil / Lubrication",
    "cutter": "Cutter-Specific",
    "status": "Status / Control",
    "electrical": "Electrical",
    "other": "Other",
}

CATEGORY_COLORS = {
    "position": "#1f77b4",
    "torque": "#d62728",
    "rotation": "#ff7f0e",
    "force": "#2ca02c",
    "pressure": "#9467bd",
    "flow": "#17becf",
    "speed": "#8c564b",
    "inclination": "#e377c2",
    "deviation": "#7f7f7f",
    "temperature": "#bcbd22",
    "lubrication": "#aec7e8",
    "cutter": "#ffbb78",
    "status": "#98df8a",
    "electrical": "#ff9896",
    "other": "#c5b0d5",
}

# Exemplary traces for each technique (chosen for good data, typical duration)
EXEMPLARY_TRACES = {
    "SOB": {
        "trace_id": "C2-2019_202405030753",
        "site_id": "1461",
        "machine_slug": "bg45v_4027",
        "description": "CFA pile C2-2019 at site 1461 (Invitalia), BG-45-V #4027, 88 min, depth 0-28 m",
    },
    "KELLY": {
        "trace_id": "01K00033511_bg45v_4027_20250717_110733_00000200_P21083",
        "site_id": "1427",
        "machine_slug": "bg45v_4027",
        "description": "Kelly pile P21083 at site 1427, BG-45-V #4027, 54 min",
    },
    "GRAB": {
        "trace_id": "01K00047045_bg28h_6061_20250627_071628_00001336_A40",
        "site_id": "5028",
        "machine_slug": "bg28h_6061",
        "description": "Grab panel A40 at site 5028, BG-28H #6061, 190 min",
    },
    "CUT": {
        "trace_id": "bc5x_482_20260119_164903_00001249_dpw003_",
        "site_id": "CS-Antwerpen",
        "machine_slug": "cube0_482",
        "description": "Diaphragm wall panel DPW003 at CS-Antwerpen, CUBE0 #482, 244 min",
    },
    "SCM": {
        "trace_id": "bg33v_5610_20240529_114910_00002531_palo_280_dms,_",
        "site_id": "1508",
        "machine_slug": "bg33v_5610",
        "description": "Soil cement column palo 280 at site 1508, BG-33-V #5610, 51 min",
    },
}


def load_trace(trace_info: dict) -> pd.DataFrame | None:
    """Load a trace parquet file and clean sentinel values."""
    path = OUTPUT_DIR / "traces" / trace_info["site_id"] / trace_info["machine_slug"] / f"{trace_info['trace_id']}.parquet"
    if not path.exists():
        print(f"  WARNING: Trace not found at {path}")
        return None
    tdf = pd.read_parquet(path)
    return clean_sentinels_df(tdf)


def categorize_sensors(columns: list[str]) -> dict[str, list[str]]:
    """Group sensor columns by category."""
    groups = {}
    for col in columns:
        if col == "timestamp":
            continue
        if col in SENSOR_MAP:
            _, cat = SENSOR_MAP[col]
        else:
            cat = "other"
        groups.setdefault(cat, []).append(col)
    return groups


def _sensor_label_for_plot(sensor_name: str, machine_slug: str) -> str:
    """Build label: 'English Name (German) [unit]'."""
    display = get_display_label(sensor_name)
    unit = get_unit(sensor_name, machine_slug)
    if unit:
        return f"{display} [{unit}]"
    return display


def plot_technique_overview(
    tdf: pd.DataFrame,
    technique: str,
    trace_info: dict,
) -> str:
    """Create a comprehensive multi-panel plot showing all sensors grouped by category."""
    machine_slug = trace_info["machine_slug"]
    sensor_groups = categorize_sensors([c for c in tdf.columns if c != "timestamp"])

    # Filter to non-empty groups with actual variation
    active_groups = {}
    for cat, sensors in sensor_groups.items():
        active = []
        for s in sensors:
            vals = tdf[s].dropna()
            if len(vals) > 10 and vals.std() > 0.001:
                active.append(s)
        if active:
            active_groups[cat] = active

    # Sort by category order
    ordered_cats = [c for c in CATEGORY_ORDER if c in active_groups]

    n_panels = len(ordered_cats)
    if n_panels == 0:
        return ""

    # Calculate time axis
    if "timestamp" in tdf.columns:
        time_axis = (tdf["timestamp"] - tdf["timestamp"].iloc[0]).dt.total_seconds() / 60
        time_label = "Time (minutes)"
    else:
        time_axis = np.arange(len(tdf)) / 60
        time_label = "Time (minutes, estimated)"

    fig, axes = plt.subplots(n_panels, 1, figsize=(16, 2.8 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    for i, cat in enumerate(ordered_cats):
        ax = axes[i]
        sensors = active_groups[cat]
        base_color = CATEGORY_COLORS.get(cat, "#333333")

        n_sensors = len(sensors)
        if n_sensors <= 3:
            colors = sns.color_palette([base_color], n_sensors)
        else:
            colors = sns.color_palette("husl", n_sensors)

        for j, sensor in enumerate(sorted(sensors)):
            vals = tdf[sensor]
            label = _sensor_label_for_plot(sensor, machine_slug)

            alpha = 0.8 if n_sensors <= 4 else 0.6
            lw = 1.0 if n_sensors <= 4 else 0.7
            ax.plot(time_axis, vals, label=label, alpha=alpha, linewidth=lw,
                    color=colors[j % len(colors)])

        cat_label = CATEGORY_LABELS.get(cat, cat)

        # Check if any sensor in this category is uncalibrated
        any_uncal = any(not is_calibrated(s, machine_slug) for s in sensors)
        if any_uncal:
            cat_label += " [mixed cal.]"
            ax.set_facecolor("#fff8f0")

        ax.set_ylabel(cat_label, fontsize=9, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right", ncol=min(3, n_sensors),
                  framealpha=0.8, handlelength=1)
        ax.grid(True, alpha=0.2)

        if i % 2 == 1 and not any_uncal:
            ax.set_facecolor("#f8f8f8")

    axes[-1].set_xlabel(time_label, fontsize=10)
    fig.suptitle(
        f"{technique} \u2014 Full Sensor Overview\n{trace_info['description']}",
        fontsize=14, fontweight="bold", y=1.005,
    )
    fig.tight_layout()

    fname = f"{technique}_full_overview.png"
    fig.savefig(FIG_DIR / fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return fname


def plot_technique_key_sensors(
    tdf: pd.DataFrame,
    technique: str,
    trace_info: dict,
    key_sensors: list[tuple[str, str]],
    title_suffix: str = "",
) -> str:
    """Plot a focused view of key sensors with more detail and annotations."""
    machine_slug = trace_info["machine_slug"]
    available = [(s, desc) for s, desc in key_sensors if s in tdf.columns]
    if not available:
        return ""

    n = len(available)
    fig, axes = plt.subplots(n, 1, figsize=(16, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    if "timestamp" in tdf.columns:
        time_axis = (tdf["timestamp"] - tdf["timestamp"].iloc[0]).dt.total_seconds() / 60
    else:
        time_axis = np.arange(len(tdf)) / 60

    for i, (sensor, desc) in enumerate(available):
        ax = axes[i]
        vals = tdf[sensor]

        cat = SENSOR_MAP.get(sensor, ("", "other"))[1]
        ax.plot(time_axis, vals, linewidth=1.2, alpha=0.8,
                color=CATEGORY_COLORS.get(cat, "#333"))

        # Add statistics annotations
        valid = vals.dropna()
        if len(valid) > 0:
            mean_val = valid.mean()
            max_val = valid.max()
            ax.axhline(mean_val, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
            max_idx = valid.idxmax()
            if time_axis.iloc[max_idx] > 0:
                ax.annotate(f"max={max_val:.1f}", xy=(time_axis.iloc[max_idx], max_val),
                            fontsize=7, color="red", alpha=0.7,
                            xytext=(5, 5), textcoords="offset points")

        # English label with unit
        display = get_display_label(sensor)
        unit = get_unit(sensor, machine_slug)
        if unit:
            ylabel = f"{display}\n[{unit}]"
        else:
            ylabel = display
        ax.set_ylabel(ylabel, fontsize=8, fontweight="bold")
        ax.set_title(desc, fontsize=8, style="italic", loc="left", pad=2)
        ax.grid(True, alpha=0.2)

        if not is_calibrated(sensor, machine_slug):
            ax.set_facecolor("#fff8f0")
            ax.text(0.99, 0.95, "UNCALIBRATED", transform=ax.transAxes,
                    fontsize=7, color="#cc6600", fontweight="bold", va="top", ha="right", alpha=0.7)

    axes[-1].set_xlabel("Time (minutes)", fontsize=10)
    fig.suptitle(
        f"{technique} \u2014 Key Sensors{title_suffix}\n{trace_info['description']}",
        fontsize=13, fontweight="bold", y=1.005,
    )
    fig.tight_layout()

    safe_suffix = title_suffix.replace(" ", "_").replace("/", "_")
    fname = f"{technique}_key_sensors{safe_suffix}.png"
    fig.savefig(FIG_DIR / fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return fname


def plot_depth_vs_sensor(
    tdf: pd.DataFrame,
    technique: str,
    trace_info: dict,
    sensors: list[str],
    depth_col: str = "Tiefe",
) -> str:
    """Plot sensors as a function of depth (depth profile) instead of time."""
    machine_slug = trace_info["machine_slug"]
    if depth_col not in tdf.columns:
        return ""

    available = [s for s in sensors if s in tdf.columns]
    if not available:
        return ""

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 8), sharey=True)
    if n == 1:
        axes = [axes]

    depth = tdf[depth_col]

    for i, sensor in enumerate(available):
        ax = axes[i]
        vals = tdf[sensor]
        cat = SENSOR_MAP.get(sensor, ("", "other"))[1]
        color = CATEGORY_COLORS.get(cat, "#333")

        ax.plot(vals, depth, linewidth=0.8, alpha=0.7, color=color)

        # English label with unit
        display = get_display_label(sensor)
        unit = get_unit(sensor, machine_slug)
        xlabel = f"{display}\n[{unit}]" if unit else display
        ax.set_xlabel(xlabel, fontsize=9)
        if i == 0:
            ax.set_ylabel("Depth (m)", fontsize=10)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.2)

        if not is_calibrated(sensor, machine_slug):
            ax.set_facecolor("#fff8f0")
            ax.text(0.5, 0.01, "UNCALIBRATED", transform=ax.transAxes,
                    fontsize=7, color="#cc6600", fontweight="bold", ha="center", alpha=0.7)

    fig.suptitle(
        f"{technique} \u2014 Depth Profiles\n{trace_info['description']}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fname = f"{technique}_depth_profiles.png"
    fig.savefig(FIG_DIR / fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return fname


# -- Technique-specific key sensor definitions ---------------------------------

SOB_KEY_SENSORS = [
    ("Tiefe", "Depth: tool penetration below ground level (drilling down, then extraction up)"),
    ("Drehmoment", "Torque: resistance of the auger \u2014 peaks indicate hard soil layers"),
    ("Betondruck", "Concrete pressure: injection pressure during auger extraction phase"),
    ("Betondurchfluss", "Concrete flow: volume rate of concrete pumped through the hollow stem"),
    ("Gesamtbetonmenge", "Total concrete: cumulative volume \u2014 should match theoretical pile volume"),
    ("Vorschub-KraftPM", "Feed force: downward push force on the auger (crowd force)"),
    ("Druck Pumpe 1", "Main pump pressure: hydraulic system load indicator"),
    ("Eindringwiderstand", "Penetration resistance: soil resistance to auger advance"),
    ("Neigung X Mast", "Mast inclination X: mast verticality during drilling"),
]

SOB_DEPTH_SENSORS = [
    "Drehmoment", "Betondruck", "Betondurchfluss", "Vorschub-KraftPM",
    "Eindringwiderstand", "Druck Pumpe 1",
]

KELLY_KEY_SENSORS = [
    ("Tiefe", "Depth: Kelly bar penetration \u2014 extends telescopically as hole deepens"),
    ("Drehmoment", "Torque: cutting resistance \u2014 high in rock/dense soil, lower in soft layers"),
    ("DrehmomentkNm", "Torque (kNm): absolute torque value on the Kelly drive"),
    ("Seilkraft Hauptwinde", "Main winch force: tension in the main rope \u2014 tracks tool weight + resistance"),
    ("Druck Pumpe 1", "Pump pressure 1: main hydraulic circuit load"),
    ("KDK Druck", "Rotary drive pressure: KDK (rotary drive head) hydraulic pressure"),
    ("KDK Drehzahl", "Rotary drive speed: Kelly bar rotation rate"),
    ("Vorschub-KraftPM", "Feed force: downward push on the drill string"),
    ("Neigung X Mast", "Mast inclination: verticality monitoring"),
]

KELLY_DEPTH_SENSORS = [
    "Drehmoment", "DrehmomentkNm", "Seilkraft Hauptwinde",
    "KDK Druck", "Vorschub-KraftPM",
]

GRAB_KEY_SENSORS = [
    ("Tiefe", "Depth: grab position \u2014 sawtooth pattern shows repeated grab cycles (down-up-down)"),
    ("Seilkraft Hauptwinde", "Main winch force: rope tension \u2014 high on ascent (loaded grab), low on descent"),
    ("Seilkraft Hilfswinde", "Auxiliary winch force: secondary rope supporting the grab"),
    ("Druck Pumpe 1", "Pump pressure 1: hydraulic system load during grab operation"),
    ("Druck Pumpe 2", "Pump pressure 2: second pump circuit"),
    ("Neigung X Mast", "Mast inclination X: vertical alignment of the grab"),
]

GRAB_DEPTH_SENSORS = ["Seilkraft Hauptwinde", "Druck Pumpe 1", "Druck Pumpe 2"]

CUT_KEY_SENSORS = [
    ("Tiefe", "Depth: cutter position \u2014 steady descent as cutting wheels excavate"),
    ("Druck FRL", "Left cutter pressure: hydraulic pressure driving the left cutting wheel"),
    ("Druck FRR", "Right cutter pressure: hydraulic pressure driving the right cutting wheel"),
    ("Drehzahl FRL", "Left cutter speed: rotation rate of the left cutting drum"),
    ("Drehzahl FRR", "Right cutter speed: rotation rate of the right cutting drum"),
    ("Temperatur FRL", "Left cutter temperature: thermal load on the left hydraulic motor"),
    ("Temperatur FRR", "Right cutter temperature: thermal load on the right hydraulic motor"),
    ("Seilkraft Fr\u00e4swinde", "Cutter winch force: suspension cable tension (tool weight + cutting force)"),
    ("Durchfluss Pumpe", "Pump flow: reverse-circulation slurry extraction rate"),
    ("Oeldruck Getriebe links", "Left gearbox oil pressure: bearing lubrication health indicator"),
    ("Oeldruck Getriebe rechts", "Right gearbox oil pressure: bearing lubrication health indicator"),
    ("Abweichung X", "X deviation: horizontal deviation from target position"),
    ("Abweichung Y", "Y deviation: horizontal deviation from target position"),
]

CUT_DEPTH_SENSORS = [
    "Druck FRL", "Druck FRR", "Drehzahl FRL", "Drehzahl FRR",
    "Temperatur FRL", "Temperatur FRR", "Durchfluss Pumpe",
]

SCM_KEY_SENSORS = [
    ("Tiefe", "Depth: mixing tool position \u2014 descends during penetration, ascends during withdrawal"),
    ("Drehmoment", "Torque: mixing resistance \u2014 indicates soil density and mixing intensity"),
    ("Drehzahl", "Rotation speed: mixing tool RPM \u2014 determines blade rotation number (BRN)"),
    ("Susp.-Druck", "Suspension pressure: grout/binder injection pressure at the nozzles"),
    ("Susp.-Durchfl.", "Suspension flow: binder injection rate (l/min)"),
    ("Susp.-Mg", "Suspension volume: cumulative binder volume injected"),
    ("Vorschub-Kraft", "Feed force: push/pull force on the mixing tool"),
    ("Druck Pumpe 1", "Pump pressure 1: main hydraulic system load"),
    ("Menge pro Meter", "Volume per meter: binder consumption per depth unit \u2014 quality indicator"),
]

SCM_DEPTH_SENSORS = [
    "Drehmoment", "Drehzahl", "Susp.-Druck", "Susp.-Durchfl.",
    "Vorschub-Kraft", "Menge pro Meter",
]


def generate_technique_section(
    technique: str,
    trace_info: dict,
    key_sensors: list[tuple[str, str]],
    depth_sensors: list[str],
    description: str,
    process_steps: str,
    sensor_interpretation: str,
) -> tuple[str, list[str]]:
    """Generate markdown + figures for one technique."""
    figures = []
    machine_slug = trace_info["machine_slug"]

    tdf = load_trace(trace_info)
    if tdf is None:
        return f"## {technique}\n\nNo example trace available.\n\n", []

    print(f"\n  Generating {technique} overview...")
    fname = plot_technique_overview(tdf, technique, trace_info)
    if fname:
        figures.append(fname)
        print(f"    {fname}")

    print(f"  Generating {technique} key sensors...")
    fname = plot_technique_key_sensors(tdf, technique, trace_info, key_sensors)
    if fname:
        figures.append(fname)
        print(f"    {fname}")

    print(f"  Generating {technique} depth profiles...")
    fname = plot_depth_vs_sensor(tdf, technique, trace_info, depth_sensors)
    if fname:
        figures.append(fname)
        print(f"    {fname}")

    # Sensor inventory table
    sensor_groups = categorize_sensors([c for c in tdf.columns if c != "timestamp"])
    sensor_table = "| Category | Sensor (German) | English Name | Range in This Trace | Unit | Calibrated? |\n"
    sensor_table += "|----------|----------------|--------------|---------------------|------|------------|\n"
    for cat in CATEGORY_ORDER:
        if cat not in sensor_groups:
            continue
        for s in sorted(sensor_groups[cat]):
            vals = tdf[s].dropna()
            if len(vals) == 0:
                continue
            english = get_display_label(s)
            cat_label = CATEGORY_LABELS.get(cat, cat)
            vmin, vmax = vals.min(), vals.max()
            unit = get_unit(s, machine_slug)
            cal = "Yes" if is_calibrated(s, machine_slug) else "**No** (raw)"
            sensor_table += f"| {cat_label} | {s} | {english} | {vmin:.1f} \u2013 {vmax:.1f} | {unit} | {cal} |\n"

    # Build markdown
    md = f"## {technique}\n\n"
    md += description + "\n\n"
    md += "### Construction Process\n\n"
    md += process_steps + "\n\n"

    # Example trace info
    md += f"### Example Trace\n\n"
    md += f"**{trace_info['description']}**\n\n"
    md += f"- Duration: {tdf['timestamp'].iloc[-1] - tdf['timestamp'].iloc[0] if 'timestamp' in tdf.columns else 'N/A'}\n"
    md += f"- Samples: {len(tdf):,} at ~1 Hz\n"
    md += f"- Active sensors: {len([c for c in tdf.columns if c != 'timestamp'])}\n\n"

    # Calibration summary for this trace
    all_sensors = [c for c in tdf.columns if c != "timestamp"]
    n_cal = sum(1 for s in all_sensors if is_calibrated(s, machine_slug))
    n_uncal = len(all_sensors) - n_cal
    if n_uncal > 0:
        md += f"**Calibration**: {n_cal} sensors calibrated, {n_uncal} uncalibrated (raw values, shown with [arb. units])\n\n"
    else:
        md += f"**Calibration**: All {n_cal} sensors calibrated\n\n"

    # Full overview figure
    if figures:
        md += f"#### Full Sensor Overview\n\n"
        md += f"![{technique} Overview](figures/techniques/{figures[0]})\n\n"

    if len(figures) > 1:
        md += f"#### Key Sensors \u2014 Annotated\n\n"
        md += f"![{technique} Key Sensors](figures/techniques/{figures[1]})\n\n"

    md += "### Sensor Interpretation\n\n"
    md += sensor_interpretation + "\n\n"

    if len(figures) > 2:
        md += f"#### Depth Profiles\n\n"
        md += f"![{technique} Depth Profiles](figures/techniques/{figures[2]})\n\n"

    md += "### Sensor Inventory\n\n"
    md += sensor_table + "\n\n"
    md += "---\n\n"

    return md, figures


# -- Technique descriptions ----------------------------------------------------

SOB_DESCRIPTION = """\
**SOB (Schneckenortbeton) = Continuous Flight Auger (CFA) Piling**

The CFA technique uses a continuous-flight auger with a hollow central stem to create cast-in-place concrete piles. \
The auger is rotated continuously into the ground without ever leaving an open borehole. As the auger is extracted, \
concrete is pumped under pressure through the hollow stem, filling the void left by the retreating flights. \
Reinforcement is then pushed into the fresh concrete. This method is fast, quiet, and vibration-free, making it \
ideal for urban sites and sensitive environments.\
"""

SOB_PROCESS = """\
1. **Setup**: The BG rig positions its mast vertically over the pile location. The mast inclination sensors confirm verticality.

2. **Drilling phase (descent)**: The auger rotates clockwise and advances into the ground. \
The soil is carried upward by the flights but not removed \u2014 it stays on the auger. \
During this phase you see: **depth increasing steadily**, **torque rising** as the tool encounters resistance \
(peaks at hard layers), **feed force** pushing the auger down, and **penetration resistance** reflecting ground conditions. \
The **pump pressures** show the hydraulic load on the machine.

3. **Target depth reached**: The auger reaches the design depth. Depth value hits its maximum. \
Torque and feed force may spike at the bottom if hitting a hard bearing layer.

4. **Concreting phase (extraction)**: This is the critical phase. The auger is slowly extracted while concrete is \
pumped through the hollow stem. **Concrete pressure** rises sharply as pumping begins. **Concrete flow rate** shows \
the volume being injected per minute. **Total concrete volume** ramps up steadily. The **depth** now decreases \
back toward zero as the auger is withdrawn. The extraction rate must be carefully matched to concrete flow to avoid voids.

5. **Completion**: Auger fully extracted, pile is concreted. Reinforcement cage is pushed in while concrete is still fresh.\
"""

SOB_SENSORS = """\
- **Depth (Tiefe)**: The signature CFA trace shows a V-shape \u2014 steady descent to target depth, then ascent during extraction. \
The descent is the drilling phase; the ascent is concreting.
- **Torque (Drehmoment)**: High during drilling (soil resistance), drops during extraction (no cutting). \
Torque spikes indicate encounters with hard layers (gravel, dense sand, boulders).
- **Concrete Pressure (Betondruck)**: Zero during drilling, rises abruptly when concreting begins. \
Should remain steady during extraction. Pressure drops may indicate voids or poor seal.
- **Concrete Flow (Betondurchfluss)**: Active only during extraction. Pulsating pattern from the concrete pump.
- **Total Concrete Volume (Gesamtbetonmenge)**: Monotonically increasing during extraction. Final value should be \
close to theoretical pile volume (slightly higher due to overbreak).
- **Feed Force (Vorschub-KraftPM)**: Positive during descent (pushing down), may reverse during extraction.
- **Penetration Resistance (Eindringwiderstand)**: Reflects soil strength at the auger tip. Useful for geological profiling.
- **Pump Pressures (Druck Pumpe)**: Reflect overall hydraulic system load. Higher during drilling than extraction.\
"""

KELLY_DESCRIPTION = """\
**KELLY = Kelly Bar Drilling (Rotary Boring)**

Kelly drilling uses a telescopic square or hexagonal steel bar (the "Kelly bar") to transmit torque from a rotary \
drive unit (KDK = Kraftdrehkopf, i.e. rotary drive head) at the top of the mast to a drill bucket or auger at the bottom. The Kelly bar \
extends as the hole deepens. Drilling produces loose material (cuttings) that fills the bucket, which is then \
lifted out and emptied. This cycle repeats until target depth is reached. The borehole may be supported by a \
temporary steel casing, by bentonite slurry, or by the natural stability of the soil. After drilling, \
reinforcement is placed and concrete is poured via tremie pipe.\
"""

KELLY_PROCESS = """\
1. **Setup**: Mast positioned vertically. If casing is needed, it is oscillated or rotated into the ground first \
using a casing drive unit.

2. **Drilling cycles**: The Kelly bar rotates the drill bucket into the soil. After filling, \
the bucket is lifted out and discharged at the surface, then lowered back for the next pass. Each cycle shows: \
**depth increasing in steps** (advance per pass), **torque peaks** during cutting, **main winch force** varying \
(low when lowering, high when lifting a full bucket), and **rotary drive pressure/speed** showing the rotary drive working.

3. **Intermediate stages**: At certain depths, the casing may be advanced deeper. Bentonite slurry may be added \
for borehole stability. The depth trace shows a staircase pattern \u2014 each step is one bucket cycle.

4. **Borehole cleaning**: At final depth, a cleaning bucket or air-lift is used to remove all cuttings from the bottom.

5. **Reinforcement**: The steel cage is lowered into the cleaned borehole.

6. **Concreting**: Concrete is placed via tremie pipe from the bottom up, displacing any slurry. \
The casing (if used) is extracted as concrete rises.\
"""

KELLY_SENSORS = """\
- **Depth (Tiefe)**: Shows a characteristic **staircase pattern** \u2014 each step down is a drilling pass, \
each step up is the bucket being lifted out. The overall envelope increases to target depth.
- **Torque (Drehmoment / DrehmomentkNm)**: Active during each drilling pass (Kelly rotating), \
zero when lifting/emptying the bucket. Torque magnitude indicates ground hardness.
- **Main Winch Force (Seilkraft Hauptwinde)**: Follows the drill cycle \u2014 low when lowering the tool, \
high when lifting a full bucket. The difference between loaded and unloaded lifts indicates how much material was excavated.
- **Rotary Drive Pressure (KDK Druck)**: Hydraulic pressure to the Kelly drive. Proportional to torque demand. \
High in hard ground, low in soft soil.
- **Rotary Drive Speed (KDK Drehzahl)**: Kelly rotation rate in RPM. May slow down under high torque \
(power-limited operation).
- **Feed Force (Vorschub-KraftPM)**: Downward force on the drill string. Higher in hard ground.
- **Pump Pressures (Druck Pumpe)**: Overall hydraulic system load indicators.\
"""

GRAB_DESCRIPTION = """\
**GRAB = Grab Excavation (for Diaphragm Walls)**

Grab excavation is a cyclic method for constructing diaphragm walls. A heavy hydraulic or mechanical clamshell grab \
is suspended from the machine's main winch and lowered into a trench filled with bentonite or polymer slurry \
(which prevents collapse). The grab opens, descends to the trench bottom, closes to capture a bite of soil, \
and is lifted out. The soil is discharged at the surface and the cycle repeats. The grab progressively deepens \
the panel until design depth is reached. After excavation, reinforcement cages are lowered in and concrete \
is placed via tremie method, displacing the slurry.\
"""

GRAB_PROCESS = """\
1. **Guide walls**: Pre-cast or cast-in-place concrete guide walls at surface level define the panel alignment.

2. **Slurry fill**: The panel excavation area is filled with bentonite or polymer slurry to provide hydrostatic \
support against the soil.

3. **Grab cycles**: Each cycle consists of:
   - **Descent (open)**: Grab opens and is lowered to the bottom of the current excavation
   - **Bite (close)**: Grab jaws close, capturing soil
   - **Ascent (closed)**: Grab is lifted through the slurry with its load
   - **Discharge**: Grab opens at surface, releasing soil into a spoil area

   The **depth trace** shows a distinctive **sawtooth pattern**: each tooth is one grab cycle. \
   The envelope of maximum depths gradually increases as the panel deepens.

4. **Panel completion**: Continue cycles until target depth is reached. A final cleaning pass ensures a level bottom.

5. **Reinforcement + concreting**: Cage lowered in; concrete placed via tremie from bottom up.\
"""

GRAB_SENSORS = """\
- **Depth (Tiefe)**: The defining feature is a **sawtooth/zigzag pattern**. Each downward tooth is the grab descending \
to the trench bottom; each upward tooth is the grab being lifted out with soil. The lower envelope gradually deepens \
as the panel progresses.
- **Main Winch Force (Seilkraft Hauptwinde)**: Follows the grab cycle \u2014 **low during descent** (grab hanging by its own weight), \
**high during ascent** (grab + soil load). The difference between loaded and unloaded ascent indicates the \
weight of soil captured per bite.
- **Pump Pressures (Druck Pumpe)**: Reflect hydraulic load for grab opening/closing and winch operation. \
Pressure spikes when the grab jaws close (crushing and capturing soil).
- **Mast Inclination (Neigung X Mast)**: Should remain stable throughout \u2014 any drift indicates the machine is shifting or the ground is yielding.

The grab technique is slower than the cutter technique (CUT) because each cycle involves physically lifting material \
out of the trench. Cycle time increases with depth since the grab must travel further.\
"""

CUT_DESCRIPTION = """\
**CUT = Diaphragm Wall Cutter (Hydrofraise / Trench Cutter)**

The trench cutter is a continuous excavation tool for constructing diaphragm walls. Unlike the grab (which works in cycles), \
the cutter works continuously: two counter-rotating cutting drums with tungsten carbide teeth are mounted on a heavy frame \
that slowly descends into a bentonite-filled trench. The drums break up soil and rock, and a reverse-circulation pump \
inside the cutter body extracts the slurry-soil mixture, sending it to a desanding plant at the surface. Clean slurry \
is returned to the trench. This method is more efficient than grabs for deep panels and hard ground, and provides \
better verticality due to the heavy frame acting as a plumb weight.\
"""

CUT_PROCESS = """\
1. **Setup**: Cutter is positioned at panel start. Trench is pre-filled with bentonite slurry.

2. **Continuous cutting descent**: The two drums rotate in opposite directions. Teeth break up the soil/rock. \
The reverse-circulation pump extracts the spoil-laden slurry and sends it to a surface desanding plant. \
Clean slurry returns to the trench. The cutter descends slowly and steadily \u2014 **depth increases monotonically** \
(unlike the grab's sawtooth). Cutting rate depends on ground hardness.

3. **Sensors during cutting**:
   - **Cutter pressures (Druck FRL/FRR)**: reflect cutting resistance \u2014 peaks at hard layers
   - **Cutter speeds (Drehzahl FRL/FRR)**: rotation rates of left and right drums
   - **Temperatures (Temperatur FRL/FRR)**: rise during sustained cutting (thermal load on hydraulic motors)
   - **Pump flow**: extraction rate of the reverse-circulation system
   - **Deviation X/Y**: lateral drift from target position
   - **Gearbox oil pressures**: health monitoring for the cutter's mechanical components

4. **Panel completion**: Cutter reaches target depth. Continue circulating to clean the bottom.

5. **Cutter extraction**: Lift the cutter out. Install reinforcement and concrete via tremie.\
"""

CUT_SENSORS = """\
- **Depth (Tiefe)**: Smooth, monotonic descent \u2014 no cycles like grab. Rate of descent indicates cutting speed.
- **Cutter Pressure Left/Right (Druck FRL / Druck FRR)**: Hydraulic pressure to left and right cutting motors. \
These are the primary indicators of cutting load. High pressure = hard ground. Left and right should be roughly equal; \
large asymmetry may indicate uneven wear or a tilted cutter.
- **Cutter Speed Left/Right (Drehzahl FRL / Drehzahl FRR)**: Drum rotation rates. May drop under heavy load (pressure-limited).
- **Cutter Temperature Left/Right (Temperatur FRL / FRR)**: Rise during sustained cutting. Important for machine health \u2014 \
excessive temperatures (>80\u00b0C actual) indicate potential bearing or seal issues.
- **Cutter Winch Force (Seilkraft Fr\u00e4swinde)**: Tension in the suspension cables. Increases with depth \
(more cable weight) and during active cutting.
- **Pump Flow (Durchfluss Pumpe)**: Reverse-circulation extraction rate. Should be steady during cutting.
- **Gearbox Oil Pressure (Oeldruck Getriebe)**: Bearing lubrication pressure. Rising values over time indicate wear.
- **Deviation X/Y (Abweichung X/Y)**: Lateral drift from the target vertical line. Must stay within tolerance (~3mm/m).\
"""

SCM_DESCRIPTION = """\
**SCM (Single Column Mixing) = Soil Cement Mixing / Deep Mixing**

SCM is an in-situ ground improvement technique. A mixing tool with specialized blades is rotated into the ground \
while cement-based grout (binder) is injected through nozzles on the tool. The blades disaggregate the native soil \
and mix it with the binder, creating soil-cement columns that harden over time. Unlike piling, no soil is removed \u2014 \
the ground itself becomes the construction material. SCM is used for foundation support, cut-off walls (seepage barriers), \
slope stabilization, and liquefaction prevention.\
"""

SCM_PROCESS = """\
1. **Penetration (downstroke)**: The mixing tool rotates and advances into the ground. Binder (cement slurry) \
is injected during descent through nozzles on the tool. **Depth increases steadily**, **torque** reflects soil \
resistance, **suspension pressure and flow** show binder injection rate.

2. **Bottom mixing**: At target depth, the tool continues rotating to ensure thorough mixing at the column base.

3. **Withdrawal (upstroke)**: The tool is extracted while continuing to rotate. Additional binder may be injected. \
**Depth decreases** back to zero. The **volume per meter** sensor tracks how much binder was placed per depth unit \u2014 \
this is a key quality indicator.

4. **Completion**: The tool is fully extracted, leaving a soil-cement column that will cure (typically tested at 28 days).

The depth trace shows a characteristic **tent shape**: steady descent, then steady ascent. \
Unlike CFA (SOB), there is no concrete phase \u2014 the binder is injected continuously during both descent and ascent.\
"""

SCM_SENSORS = """\
- **Depth (Tiefe)**: Tent-shaped profile \u2014 steady descent during penetration, then steady ascent during withdrawal.
- **Torque (Drehmoment)**: Active throughout \u2014 reflects soil mixing resistance. May increase with depth as soil becomes denser.
- **Rotation Speed (Drehzahl)**: Mixing tool RPM. Combined with advance rate, determines the Blade Rotation Number (BRN) \u2014 \
a key quality metric for mixing intensity.
- **Suspension Pressure (Susp.-Druck)**: Injection pressure at the nozzles. Should be steady and sufficient \
to ensure full penetration of the binder into the soil.
- **Suspension Flow (Susp.-Durchfl.)**: Binder injection rate in l/min. Active during both descent and ascent.
- **Suspension Volume (Susp.-Mg)**: Cumulative binder volume. Should increase monotonically.
- **Feed Force (Vorschub-Kraft)**: Push/pull force on the mixing tool.
- **Volume per Meter (Menge pro Meter)**: Binder consumption per depth unit \u2014 lower values indicate efficient mixing; \
high values may indicate soil washout or excessive grout consumption.\
"""


def main():
    print("Generating Technique Atlas...")

    report_sections = []

    # Header
    report_sections.append("# Technique Atlas\n\n")
    report_sections.append(
        "This report provides a detailed explanation of each construction technique in the dataset, "
        "illustrated with real sensor data from exemplary traces. For each technique, we show:\n\n"
        "- **What the technique does** physically\n"
        "- **Step-by-step construction process** with sensor correlations\n"
        "- **Full sensor overview** \u2014 all available channels plotted over time\n"
        "- **Key sensor interpretation** \u2014 what each sensor tells you about the operation\n"
        "- **Depth profiles** \u2014 sensors plotted against depth rather than time\n"
        "- **Sensor inventory** \u2014 complete list with value ranges and calibration status\n\n"
        "**Note on calibration**: Some sensors on certain machines report raw ADC values rather than "
        "engineering units. These are marked as `[arb. units]` on plots and highlighted with an orange "
        "background. See the calibration column in sensor inventory tables for details.\n\n"
        "All figures are in `reports/figures/techniques/`.\n\n"
        "---\n\n"
    )

    techniques = [
        ("SOB", SOB_DESCRIPTION, SOB_PROCESS, SOB_SENSORS, SOB_KEY_SENSORS, SOB_DEPTH_SENSORS),
        ("KELLY", KELLY_DESCRIPTION, KELLY_PROCESS, KELLY_SENSORS, KELLY_KEY_SENSORS, KELLY_DEPTH_SENSORS),
        ("GRAB", GRAB_DESCRIPTION, GRAB_PROCESS, GRAB_SENSORS, GRAB_KEY_SENSORS, GRAB_DEPTH_SENSORS),
        ("CUT", CUT_DESCRIPTION, CUT_PROCESS, CUT_SENSORS, CUT_KEY_SENSORS, CUT_DEPTH_SENSORS),
        ("SCM", SCM_DESCRIPTION, SCM_PROCESS, SCM_SENSORS, SCM_KEY_SENSORS, SCM_DEPTH_SENSORS),
    ]

    all_figures = {}
    for tech, desc, process, sensor_interp, key_sensors, depth_sensors in techniques:
        print(f"\n{'='*60}")
        print(f"  {tech}")
        print(f"{'='*60}")

        trace_info = EXEMPLARY_TRACES[tech]
        md, figs = generate_technique_section(
            tech, trace_info, key_sensors, depth_sensors,
            desc, process, sensor_interp,
        )
        report_sections.append(md)
        all_figures[tech] = figs

    # Write report
    REPORT_PATH.write_text("".join(report_sections))
    print(f"\nReport written to {REPORT_PATH}")

    total_figs = sum(len(f) for f in all_figures.values())
    print(f"Total figures: {total_figs}")
    print("Done!")


if __name__ == "__main__":
    main()
