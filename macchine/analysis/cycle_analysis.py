"""GRAB cycle-level analysis.

Detects individual grab cycles from the depth sawtooth pattern and extracts
per-cycle features (loaded/unloaded force, descent/ascent speed, soil weight
proxy, pump and jaw pressure). Tracks these features over sessions to detect
degradation (jaw wear, winch degradation, hydraulic wear).

Depth convention: negative values = below ground level.  A cycle is a
descent (depth becoming more negative) followed by an ascent (depth returning
toward zero).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from tqdm import tqdm

from macchine.harmonize.calibration import (
    clean_sentinels_df,
    is_calibrated,
    validate_physical_range,
)
from macchine.storage.catalog import get_merged_trace_index

# Minimum depth excursion (meters) to consider a cycle real
MIN_CYCLE_DEPTH = 5.0
# Minimum samples between peaks/troughs (avoids noise)
MIN_PEAK_DISTANCE = 20
# Smoothing window (samples, ~seconds at 1 Hz)
SMOOTH_WINDOW = 10
# Force sensor candidates for GRAB machines (in priority order)
FORCE_SENSORS = ["Seilkraft", "Seilkraft Hauptwinde", "Hakenlast Hauptwinde"]
# Jaw closing pressure sensor
JAW_SENSOR = "Schließzylinderdruck"
# Pump pressure sensors
PUMP_SENSORS = ["Druck Pumpe 1", "Druck Pumpe 2"]


def detect_grab_cycles(
    depth_series: pd.Series,
    min_depth: float = MIN_CYCLE_DEPTH,
    smooth_window: int = SMOOTH_WINDOW,
) -> list[dict]:
    """Detect individual grab cycles from a depth time series.

    Uses scipy peak detection with prominence filtering for robust
    cycle extraction even in noisy data.

    Parameters
    ----------
    depth_series : Depth values (negative = below ground).
    min_depth : Minimum depth excursion (meters) to count as a real cycle.
    smooth_window : Uniform filter window for smoothing.

    Returns list of cycle dicts with keys:
        start_idx, bottom_idx, end_idx, depth_at_bottom, depth_at_top,
        cycle_depth (positive meters), descent_samples, ascent_samples,
        cycle_samples.
    """
    depth = pd.to_numeric(depth_series, errors="coerce")
    valid_mask = ~depth.isna()
    if valid_mask.sum() < 50:
        return []

    # Interpolate NaN gaps and smooth
    arr = depth.values.copy().astype(float)
    valid_idx = np.where(valid_mask.values)[0]
    if len(valid_idx) < 2:
        return []
    arr = np.interp(np.arange(len(arr)), valid_idx, arr[valid_idx])
    arr_smooth = uniform_filter1d(arr, size=smooth_window)

    # Find peaks (near surface = local maxima) and troughs (bottom = local minima)
    # Prominence ensures only significant excursions are detected
    peaks, _ = find_peaks(arr_smooth, prominence=min_depth, distance=MIN_PEAK_DISTANCE)
    troughs, _ = find_peaks(-arr_smooth, prominence=min_depth, distance=MIN_PEAK_DISTANCE)

    if len(peaks) < 2 or len(troughs) < 1:
        return []

    # Build cycles by pairing: peak -> trough -> next_peak
    events = [(int(p), "peak") for p in peaks] + [(int(t), "trough") for t in troughs]
    events.sort(key=lambda x: x[0])

    cycles = []
    i = 0
    while i < len(events) - 2:
        idx0, type0 = events[i]
        idx1, type1 = events[i + 1]
        idx2, type2 = events[i + 2]

        if type0 == "peak" and type1 == "trough" and type2 == "peak":
            top_depth = float(arr_smooth[idx0])
            bottom_depth = float(arr_smooth[idx1])
            # cycle_depth is always positive (meters of excursion)
            cycle_depth = top_depth - bottom_depth

            if cycle_depth >= min_depth:
                cycles.append({
                    "start_idx": idx0,
                    "bottom_idx": idx1,
                    "end_idx": idx2,
                    "depth_at_bottom": bottom_depth,
                    "depth_at_top": top_depth,
                    "cycle_depth": cycle_depth,
                    "descent_samples": idx1 - idx0,
                    "ascent_samples": idx2 - idx1,
                    "cycle_samples": idx2 - idx0,
                })
            i += 2
        else:
            i += 1

    return cycles


def extract_cycle_features(
    trace_df: pd.DataFrame,
    cycles: list[dict],
    machine_slug: str,
) -> list[dict]:
    """Extract per-cycle sensor features from a GRAB trace.

    For each cycle extracts:
    - loaded_force: mean winch force during ascent (grab full of soil)
    - unloaded_force: mean winch force during descent (empty grab)
    - soil_weight_proxy: loaded_force - unloaded_force
    - descent_speed: depth range / descent duration
    - ascent_speed: depth range / ascent duration
    - pump_pressure_mean, pump_pressure_p95: pump pressure during cycle
    - jaw_pressure_mean, jaw_pressure_at_bottom: grab jaw closing pressure

    Only uses calibrated sensors where available, falls back to raw values
    for force sensors when no calibrated option exists.
    """
    if not cycles:
        return []

    # Find best force sensor
    force_col = None
    for s in FORCE_SENSORS:
        if s in trace_df.columns and is_calibrated(s, machine_slug):
            force_col = s
            break
    if force_col is None:
        for s in FORCE_SENSORS:
            if s in trace_df.columns:
                force_col = s
                break

    # Find pump pressure sensors
    pump_cols = [c for c in PUMP_SENSORS if c in trace_df.columns]

    # Jaw pressure
    jaw_col = JAW_SENSOR if JAW_SENSOR in trace_df.columns else None

    records = []
    for cycle in cycles:
        s = cycle["start_idx"]
        b = cycle["bottom_idx"]
        e = cycle["end_idx"]

        if e >= len(trace_df):
            continue

        rec = {
            "start_idx": s,
            "bottom_idx": b,
            "end_idx": e,
            "depth_at_bottom": cycle["depth_at_bottom"],
            "depth_at_top": cycle["depth_at_top"],
            "cycle_depth": cycle["cycle_depth"],
            "descent_samples": cycle["descent_samples"],
            "ascent_samples": cycle["ascent_samples"],
            "cycle_samples": cycle["cycle_samples"],
            "descent_duration_s": float(cycle["descent_samples"]),
            "ascent_duration_s": float(cycle["ascent_samples"]),
            "cycle_duration_s": float(cycle["cycle_samples"]),
        }

        # Speeds (m/s)
        if cycle["descent_samples"] > 0:
            rec["descent_speed"] = cycle["cycle_depth"] / cycle["descent_samples"]
        if cycle["ascent_samples"] > 0:
            rec["ascent_speed"] = cycle["cycle_depth"] / cycle["ascent_samples"]

        # Force features
        if force_col is not None:
            force = pd.to_numeric(trace_df[force_col], errors="coerce")
            force = validate_physical_range(force, force_col, machine_slug)

            descent_force = force.iloc[s:b].dropna()
            ascent_force = force.iloc[b:e].dropna()

            if len(descent_force) > 3:
                rec["unloaded_force"] = float(descent_force.mean())
            if len(ascent_force) > 3:
                rec["loaded_force"] = float(ascent_force.mean())
            if "unloaded_force" in rec and "loaded_force" in rec:
                rec["soil_weight_proxy"] = rec["loaded_force"] - rec["unloaded_force"]

        # Pump pressure
        for pcol in pump_cols:
            pressure = pd.to_numeric(trace_df[pcol].iloc[s:e], errors="coerce")
            pressure = validate_physical_range(pressure, pcol, machine_slug)
            clean = pressure.dropna()
            if len(clean) > 3:
                rec[f"{pcol}_mean"] = float(clean.mean())
                rec[f"{pcol}_p95"] = float(clean.quantile(0.95))

        # Jaw pressure
        if jaw_col is not None:
            jaw = pd.to_numeric(trace_df[jaw_col].iloc[s:e], errors="coerce")
            clean = jaw.dropna()
            if len(clean) > 3:
                rec["jaw_pressure_mean"] = float(clean.mean())
                # Jaw pressure near the bottom (when jaws close to grab soil)
                window = max(1, (b - s) // 5)
                jaw_at_bottom = pd.to_numeric(
                    trace_df[jaw_col].iloc[max(s, b - window) : min(b + window, e)],
                    errors="coerce",
                ).dropna()
                if len(jaw_at_bottom) > 0:
                    rec["jaw_pressure_at_bottom"] = float(jaw_at_bottom.max())

        records.append(rec)

    return records


def _get_trace_path(output_dir: Path, row: pd.Series) -> Path:
    """Resolve parquet path from merged index row."""
    site = row["site_id"]
    slug = row["machine_slug"] if row["machine_slug"] != "unidentified" else "unknown"
    trace_id = row["trace_id"]
    return output_dir / "traces" / str(site) / slug / f"{trace_id}.parquet"


def track_cycle_degradation(
    output_dir: Path,
    machine: str,
    site: str | None = None,
    max_sessions: int | None = None,
) -> pd.DataFrame:
    """Track per-cycle metrics over time for a GRAB machine.

    For each GRAB session, detects cycles and extracts per-cycle features.
    Returns a DataFrame with one row per cycle, including session metadata.

    Key degradation signals:
    - Declining soil_weight_proxy at similar depth → grab jaw wear
    - Slowing cycle times → winch degradation
    - Rising pump pressure at similar load → hydraulic wear
    """
    df = get_merged_trace_index(output_dir)
    df = df.dropna(subset=["start_time"]).sort_values("start_time")
    df = df[df["machine_slug"] == machine]
    df = df[df["technique"] == "GRAB"]
    if site:
        df = df[df["site_id"] == site]

    if df.empty:
        return pd.DataFrame()

    if max_sessions and len(df) > max_sessions:
        df = df.head(max_sessions)

    all_cycles = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Analyzing GRAB cycles for {machine}"):
        trace_path = _get_trace_path(output_dir, row)
        if not trace_path.exists():
            continue

        try:
            tdf = pd.read_parquet(trace_path)
        except Exception:
            continue

        tdf = clean_sentinels_df(tdf)

        if "Tiefe" not in tdf.columns:
            continue

        depth = pd.to_numeric(tdf["Tiefe"], errors="coerce")
        depth = validate_physical_range(depth, "Tiefe", machine)

        cycles = detect_grab_cycles(depth)
        if not cycles:
            continue

        features = extract_cycle_features(tdf, cycles, machine)

        for feat in features:
            feat["trace_id"] = row["trace_id"]
            feat["start_time"] = row["start_time"]
            feat["site_id"] = row["site_id"]
            feat["element_name"] = row.get("element_name", "")
            feat["operator"] = row.get("operator", "")
            all_cycles.append(feat)

    return pd.DataFrame(all_cycles) if all_cycles else pd.DataFrame()
