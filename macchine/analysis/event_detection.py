"""Event detection in time series.

Per-technique detectors for anomalous events within individual traces:
- SOB: concrete pressure loss during concreting
- CUT: pressure asymmetry, temperature excursion
- GRAB: incomplete grab cycles
- KELLY: torque anomaly during drilling
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from macchine.storage.catalog import get_merged_trace_index
from macchine.harmonize.calibration import (
    clean_sentinels_df,
    is_calibrated,
    validate_physical_range,
)
from macchine.analysis.cycle_analysis import detect_grab_cycles


def detect_concrete_pressure_loss(
    trace_df: pd.DataFrame,
    machine_slug: str,
    min_drop_bar: float = 1.0,
    window: int = 10,
) -> list[dict]:
    """SOB: Detect sudden drops in concrete pressure during concreting phase.

    Flags potential tremie disconnect or pump failure. Only analyzes the
    ascending (concreting) phase where depth is decreasing.
    """
    if "Betondruck" not in trace_df.columns:
        return []
    if not is_calibrated("Betondruck", machine_slug):
        return []
    if "Tiefe" not in trace_df.columns:
        return []

    depth = pd.to_numeric(trace_df["Tiefe"], errors="coerce")
    depth = validate_physical_range(depth, "Tiefe", machine_slug)
    pressure = pd.to_numeric(trace_df["Betondruck"], errors="coerce")
    pressure = validate_physical_range(pressure, "Betondruck", machine_slug)

    # Concreting phase: depth is ascending (decreasing values)
    depth_diff = depth.diff()
    concreting_mask = depth_diff < -0.01
    if concreting_mask.sum() < 20:
        return []

    p_concreting = pressure[concreting_mask].dropna()
    if len(p_concreting) < 20:
        return []

    # Look for sudden drops (rolling max - current > threshold)
    rolling_max = p_concreting.rolling(window, min_periods=5).max()
    drops = rolling_max - p_concreting

    events = []
    # Find drops exceeding threshold
    drop_mask = drops > min_drop_bar
    if not drop_mask.any():
        return []

    # Group consecutive drop samples into events
    drop_indices = drop_mask[drop_mask].index.tolist()
    event_start = drop_indices[0]
    prev_idx = drop_indices[0]

    for idx in drop_indices[1:]:
        if idx - prev_idx > 5:  # Gap > 5 samples = new event
            events.append({
                "event_type": "concrete_pressure_loss",
                "sample_index": int(event_start),
                "duration_samples": int(prev_idx - event_start),
                "severity": "high" if drops.loc[event_start:prev_idx].max() > min_drop_bar * 3 else "medium",
                "description": f"Concrete pressure drop of {drops.loc[event_start:prev_idx].max():.1f} bar during concreting",
            })
            event_start = idx
        prev_idx = idx

    # Final event
    events.append({
        "event_type": "concrete_pressure_loss",
        "sample_index": int(event_start),
        "duration_samples": int(prev_idx - event_start),
        "severity": "high" if drops.loc[event_start:prev_idx].max() > min_drop_bar * 3 else "medium",
        "description": f"Concrete pressure drop of {drops.loc[event_start:prev_idx].max():.1f} bar during concreting",
    })

    return events


def detect_pressure_asymmetry(
    trace_df: pd.DataFrame,
    machine_slug: str,
    threshold_bar: float = 50.0,
    min_duration_samples: int = 30,
) -> list[dict]:
    """CUT: Detect sustained left/right cutter pressure asymmetry.

    Flags uneven wear when |Druck FRL - Druck FRR| exceeds threshold
    for a sustained period during active cutting.
    """
    frl, frr = "Druck FRL", "Druck FRR"
    if frl not in trace_df.columns or frr not in trace_df.columns:
        return []
    if not is_calibrated(frl, machine_slug) or not is_calibrated(frr, machine_slug):
        return []

    p_left = pd.to_numeric(trace_df[frl], errors="coerce")
    p_right = pd.to_numeric(trace_df[frr], errors="coerce")
    p_left = validate_physical_range(p_left, frl, machine_slug)
    p_right = validate_physical_range(p_right, frr, machine_slug)

    # Active cutting: both pressures above noise floor (> 20 bar)
    active = (p_left > 20) & (p_right > 20)
    if active.sum() < 50:
        return []

    asym = (p_left - p_right).abs()
    asym_active = asym[active]

    # Find sustained periods above threshold
    above = asym_active > threshold_bar
    if not above.any():
        return []

    events = []
    above_indices = above[above].index.tolist()
    if not above_indices:
        return []

    event_start = above_indices[0]
    prev_idx = above_indices[0]

    for idx in above_indices[1:]:
        if idx - prev_idx > 10:
            duration = prev_idx - event_start
            if duration >= min_duration_samples:
                max_asym = float(asym_active.loc[event_start:prev_idx].max())
                events.append({
                    "event_type": "pressure_asymmetry",
                    "sample_index": int(event_start),
                    "duration_samples": int(duration),
                    "severity": "high" if max_asym > threshold_bar * 2 else "medium",
                    "description": f"L/R pressure asymmetry of {max_asym:.0f} bar for {duration}s",
                })
            event_start = idx
        prev_idx = idx

    # Final event
    duration = prev_idx - event_start
    if duration >= min_duration_samples:
        max_asym = float(asym_active.loc[event_start:prev_idx].max())
        events.append({
            "event_type": "pressure_asymmetry",
            "sample_index": int(event_start),
            "duration_samples": int(duration),
            "severity": "high" if max_asym > threshold_bar * 2 else "medium",
            "description": f"L/R pressure asymmetry of {max_asym:.0f} bar for {duration}s",
        })

    return events


def detect_temperature_excursion(
    trace_df: pd.DataFrame,
    machine_slug: str,
    n_std: float = 3.0,
) -> list[dict]:
    """CUT: Detect temperature exceeding baseline + N*std.

    Only for calibrated temperature sensors. Flags overheating events.
    """
    events = []
    for sensor in ["Temperatur FRL", "Temperatur FRR"]:
        if sensor not in trace_df.columns:
            continue
        if not is_calibrated(sensor, machine_slug):
            continue

        temp = pd.to_numeric(trace_df[sensor], errors="coerce")
        temp = validate_physical_range(temp, sensor, machine_slug)
        temp = temp.dropna()

        if len(temp) < 50:
            continue

        # Baseline: first 20% of trace (warmup period excluded)
        n_baseline = max(20, len(temp) // 5)
        baseline = temp.iloc[:n_baseline]
        baseline_mean = baseline.mean()
        baseline_std = baseline.std()

        if baseline_std < 0.1:
            continue  # No variation in baseline

        threshold = baseline_mean + n_std * baseline_std
        excursion_mask = temp > threshold

        if not excursion_mask.any():
            continue

        max_temp = float(temp.max())
        max_idx = int(temp.idxmax())

        events.append({
            "event_type": "temperature_excursion",
            "sample_index": max_idx,
            "duration_samples": int(excursion_mask.sum()),
            "severity": "high" if max_temp > threshold + baseline_std * 2 else "medium",
            "description": (
                f"{sensor}: peak {max_temp:.1f}C vs baseline "
                f"{baseline_mean:.1f}+/-{baseline_std:.1f}C (threshold={threshold:.1f}C)"
            ),
        })

    return events


def detect_incomplete_cycle(
    trace_df: pd.DataFrame,
    machine_slug: str,
    force_ratio_threshold: float = 1.1,
) -> list[dict]:
    """GRAB: Detect cycles where grab didn't capture soil.

    A complete cycle should have ascent force > descent force (soil weight).
    If ascent force ~= descent force, the grab came up empty.
    """
    if "Tiefe" not in trace_df.columns:
        return []

    depth = pd.to_numeric(trace_df["Tiefe"], errors="coerce")
    depth = validate_physical_range(depth, "Tiefe", machine_slug)

    cycles = detect_grab_cycles(depth)
    if not cycles:
        return []

    # Find force sensor
    force_sensor = None
    for s in ["Seilkraft", "Seilkraft Hauptwinde"]:
        if s in trace_df.columns and is_calibrated(s, machine_slug):
            force_sensor = s
            break

    if force_sensor is None:
        return []

    force = pd.to_numeric(trace_df[force_sensor], errors="coerce")
    force = validate_physical_range(force, force_sensor, machine_slug)

    events = []
    for cycle in cycles:
        start = cycle["start_idx"]
        bottom = cycle["bottom_idx"]
        end = cycle["end_idx"]

        descent_force = force.iloc[start:bottom].dropna()
        ascent_force = force.iloc[bottom:end].dropna()

        if len(descent_force) < 3 or len(ascent_force) < 3:
            continue

        descent_mean = descent_force.mean()
        ascent_mean = ascent_force.mean()

        # If ratio < threshold, grab came up empty
        if descent_mean > 0 and ascent_mean / descent_mean < force_ratio_threshold:
            events.append({
                "event_type": "incomplete_grab_cycle",
                "sample_index": int(start),
                "duration_samples": int(end - start),
                "severity": "low",
                "description": (
                    f"Grab cycle at depth {cycle['depth_at_bottom']:.1f}m: "
                    f"ascent/descent force ratio = {ascent_mean/descent_mean:.2f} "
                    f"(< {force_ratio_threshold})"
                ),
            })

    return events


def detect_torque_anomaly(
    trace_df: pd.DataFrame,
    machine_slug: str,
    drop_threshold_pct: float = 80.0,
    min_active_torque: float = 5.0,
) -> list[dict]:
    """KELLY: Detect sudden torque drops during active drilling.

    A sudden drop to near-zero during active drilling may indicate
    tool breakage, encountering a cavity, or loss of ground contact.
    """
    # Find torque sensor
    torque_sensor = None
    for s in ["DrehmomentkNm", "Drehmoment", "DrehmomentProzent"]:
        if s in trace_df.columns and is_calibrated(s, machine_slug):
            torque_sensor = s
            break

    if torque_sensor is None:
        return []
    if "Tiefe" not in trace_df.columns:
        return []

    depth = pd.to_numeric(trace_df["Tiefe"], errors="coerce")
    depth = validate_physical_range(depth, "Tiefe", machine_slug)
    torque = pd.to_numeric(trace_df[torque_sensor], errors="coerce")
    torque = validate_physical_range(torque, torque_sensor, machine_slug)

    # Active drilling: depth is increasing (going deeper)
    depth_diff = depth.diff()
    drilling = depth_diff > 0.01
    if drilling.sum() < 50:
        return []

    torque_drilling = torque[drilling].dropna()
    if len(torque_drilling) < 50:
        return []

    # Rolling baseline
    rolling_mean = torque_drilling.rolling(30, min_periods=15).mean()

    # Detect drops
    events = []
    for i in range(len(torque_drilling)):
        if i < 30:
            continue
        baseline = rolling_mean.iloc[i]
        current = torque_drilling.iloc[i]

        if baseline < min_active_torque:
            continue

        drop_pct = (baseline - current) / baseline * 100
        if drop_pct > drop_threshold_pct:
            events.append({
                "event_type": "torque_anomaly",
                "sample_index": int(torque_drilling.index[i]),
                "duration_samples": 1,
                "severity": "high" if drop_pct > 95 else "medium",
                "description": (
                    f"Torque drop of {drop_pct:.0f}% during active drilling "
                    f"(baseline={baseline:.1f}, current={current:.1f})"
                ),
            })

    # Deduplicate: merge events within 30 samples
    if len(events) > 1:
        merged = [events[0]]
        for evt in events[1:]:
            if evt["sample_index"] - merged[-1]["sample_index"] < 30:
                merged[-1]["duration_samples"] = (
                    evt["sample_index"] - merged[-1]["sample_index"]
                )
            else:
                merged.append(evt)
        events = merged

    return events


# Technique -> list of detector functions
TECHNIQUE_DETECTORS = {
    "SOB": [detect_concrete_pressure_loss],
    "CUT": [detect_pressure_asymmetry, detect_temperature_excursion],
    "GRAB": [detect_incomplete_cycle],
    "KELLY": [detect_torque_anomaly],
}


def scan_trace_events(
    trace_df: pd.DataFrame,
    technique: str,
    machine_slug: str,
) -> list[dict]:
    """Run all technique-appropriate detectors on a single trace."""
    detectors = TECHNIQUE_DETECTORS.get(technique, [])
    all_events = []
    for detector in detectors:
        events = detector(trace_df, machine_slug)
        all_events.extend(events)
    return all_events


def scan_fleet_events(
    output_dir: Path,
    machine: str | None = None,
    technique: str | None = None,
    max_traces: int | None = None,
) -> pd.DataFrame:
    """Iterate over all merged sessions and run technique-appropriate detectors.

    Returns DataFrame of detected events with: trace_id, event_type,
    sample_index, severity, description.
    """
    df = get_merged_trace_index(output_dir)
    df = df.dropna(subset=["start_time"]).sort_values("start_time")

    if machine:
        df = df[df["machine_slug"] == machine]
    if technique:
        df = df[df["technique"] == technique]

    # Only scan techniques with detectors
    df = df[df["technique"].isin(TECHNIQUE_DETECTORS.keys())]

    if df.empty:
        return pd.DataFrame()

    if max_traces and len(df) > max_traces:
        df = df.head(max_traces)

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scanning for events"):
        slug = row["machine_slug"]
        site = row["site_id"]
        trace_id = row["trace_id"]
        tech = row["technique"]

        slug_path = slug if slug != "unidentified" else "unknown"
        trace_path = output_dir / "traces" / str(site) / slug_path / f"{trace_id}.parquet"

        if not trace_path.exists():
            continue

        try:
            tdf = pd.read_parquet(trace_path)
        except Exception:
            continue

        tdf = clean_sentinels_df(tdf)
        events = scan_trace_events(tdf, tech, slug)

        for evt in events:
            evt["trace_id"] = trace_id
            evt["machine_slug"] = slug
            evt["site_id"] = site
            evt["technique"] = tech
            evt["start_time"] = row["start_time"]
            evt["element_name"] = row.get("element_name", "")
            records.append(evt)

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records)
    result = result.sort_values("start_time").reset_index(drop=True)
    return result
