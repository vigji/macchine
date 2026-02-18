"""Sensor-level degradation analysis.

Extracts per-trace sensor statistics from actual parquet files and tracks
them over time within controlled groups (same machine + technique + site).
Includes CUT-specific health analysis for temperature, pressure asymmetry,
gearbox oil, and leakage monitoring.
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
    get_display_label,
    get_unit,
)

# Sensors to extract per technique
TECHNIQUE_SENSOR_GROUPS = {
    "SOB": {
        "pressure": ["Druck Pumpe 1", "Druck Pumpe 2"],
        "torque": ["Drehmoment"],
        "depth": ["Tiefe"],
        "verticality": ["Neigung X Mast", "Neigung Y Mast"],
    },
    "KELLY": {
        "pressure": ["Druck Pumpe 1", "Druck Pumpe 2", "KDK Druck"],
        "torque": ["DrehmomentkNm"],
        "depth": ["Tiefe"],
        "force": ["Seilkraft Hauptwinde"],
        "verticality": ["Neigung X Mast", "Neigung Y Mast"],
    },
    "CUT": {
        "pressure": ["Druck FRL", "Druck FRR"],
        "temperature": ["Temperatur FRL", "Temperatur FRR"],
        "oil_pressure": ["Oeldruck Getriebe links", "Oeldruck Getriebe rechts"],
        "leakage": ["Leckagedruck Getriebe links", "Leckagedruck Getriebe rechts"],
        "depth": ["Tiefe"],
        "rotation": ["Drehzahl FRL", "Drehzahl FRR"],
        "force": ["Seilkraft Fräswinde"],
    },
    "GRAB": {
        "pressure": ["Druck Pumpe 1", "Druck Pumpe 2"],
        "depth": ["Tiefe"],
        "force": ["Seilkraft", "Seilkraft Hauptwinde"],
        "verticality": ["Neigung X", "Neigung Y"],
    },
    "SCM": {
        "pressure": ["Druck Pumpe 1"],
        "torque": ["Drehmoment"],
        "depth": ["Tiefe"],
        "verticality": ["Neigung X Mast", "Neigung Y Mast"],
    },
}


def _get_trace_path(output_dir: Path, row: pd.Series) -> Path:
    """Resolve the parquet trace file path from a merged index row."""
    site = row["site_id"]
    slug = row["machine_slug"] if row["machine_slug"] != "unidentified" else "unknown"
    trace_id = row["trace_id"]
    return output_dir / "traces" / str(site) / slug / f"{trace_id}.parquet"


def _extract_active_phase(tdf: pd.DataFrame, depth_col: str = "Tiefe") -> pd.DataFrame:
    """Extract the active drilling phase where depth is changing.

    Returns subset of dataframe where |d(depth)/dt| > threshold.
    Falls back to full trace if depth not available.
    """
    if depth_col not in tdf.columns:
        return tdf

    depth = pd.to_numeric(tdf[depth_col], errors="coerce")
    depth_diff = depth.diff().abs()
    # Active = depth changing by more than 0.01m per sample
    active_mask = depth_diff > 0.01
    # Include some buffer around active periods
    active_mask = active_mask.rolling(5, center=True, min_periods=1).max().astype(bool)

    if active_mask.sum() < 10:
        return tdf  # Too few active samples, use full trace
    return tdf[active_mask]


def _sensor_stats(series: pd.Series) -> dict:
    """Compute summary statistics for a sensor series."""
    clean = series.dropna()
    if len(clean) < 10:
        return {}
    return {
        "median": float(clean.median()),
        "mean": float(clean.mean()),
        "std": float(clean.std()),
        "p5": float(clean.quantile(0.05)),
        "p25": float(clean.quantile(0.25)),
        "p75": float(clean.quantile(0.75)),
        "p95": float(clean.quantile(0.95)),
        "max": float(clean.max()),
        "min": float(clean.min()),
    }


def extract_sensor_features(
    output_dir: Path,
    machine: str | None = None,
    technique: str | None = None,
    site: str | None = None,
    max_traces: int | None = None,
) -> pd.DataFrame:
    """Extract per-trace sensor statistics from parquet trace files.

    Loads the merged trace index, filters by parameters, then for each session
    loads the parquet file and computes summary stats for technique-relevant sensors.
    Only uses calibrated sensors.

    Returns DataFrame with one row per session, columns = sensor features + metadata.
    """
    df = get_merged_trace_index(output_dir)
    df = df.dropna(subset=["start_time"]).sort_values("start_time")

    if machine:
        df = df[df["machine_slug"] == machine]
    if technique:
        df = df[df["technique"] == technique]
    if site:
        df = df[df["site_id"] == site]

    if df.empty:
        return pd.DataFrame()

    if max_traces and len(df) > max_traces:
        df = df.head(max_traces)

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting sensor features"):
        trace_path = _get_trace_path(output_dir, row)
        if not trace_path.exists():
            continue

        try:
            tdf = pd.read_parquet(trace_path)
        except Exception:
            continue

        tdf = clean_sentinels_df(tdf)
        tech = row["technique"]
        slug = row["machine_slug"]
        sensor_groups = TECHNIQUE_SENSOR_GROUPS.get(tech, {})

        # Get active phase for pressure/torque extraction
        active_tdf = _extract_active_phase(tdf)

        rec = {
            "trace_id": row["trace_id"],
            "start_time": row["start_time"],
            "site_id": row["site_id"],
            "machine_slug": slug,
            "technique": tech,
            "element_name": row.get("element_name", ""),
            "operator": row.get("operator", ""),
            "duration_s": row["duration_s"],
        }

        # Extract depth features (always)
        for sensor in sensor_groups.get("depth", []):
            if sensor in tdf.columns:
                vals = pd.to_numeric(tdf[sensor], errors="coerce")
                vals = validate_physical_range(vals, sensor, slug)
                clean = vals.dropna()
                if not clean.empty:
                    rec["max_depth"] = float(clean.max())
                    rec["depth_range"] = float(clean.max() - clean.min())

        # Extract all sensor group stats
        for group_name, sensors in sensor_groups.items():
            if group_name == "depth":
                continue  # Already handled

            for sensor in sensors:
                if sensor not in tdf.columns:
                    continue
                if not is_calibrated(sensor, slug):
                    continue

                # Use active phase for pressure/torque, full trace for temperature/oil
                src = active_tdf if group_name in ("pressure", "torque", "force") else tdf

                vals = pd.to_numeric(src[sensor] if sensor in src.columns else tdf[sensor],
                                     errors="coerce")
                vals = validate_physical_range(vals, sensor, slug)
                stats = _sensor_stats(vals)

                for stat_name, stat_val in stats.items():
                    col_name = f"{sensor}__{stat_name}"
                    rec[col_name] = stat_val

        # CUT-specific: pressure asymmetry
        if tech == "CUT":
            frl = "Druck FRL"
            frr = "Druck FRR"
            if (frl in active_tdf.columns and frr in active_tdf.columns
                    and is_calibrated(frl, slug) and is_calibrated(frr, slug)):
                l_vals = pd.to_numeric(active_tdf[frl], errors="coerce")
                r_vals = pd.to_numeric(active_tdf[frr], errors="coerce")
                l_vals = validate_physical_range(l_vals, frl, slug)
                r_vals = validate_physical_range(r_vals, frr, slug)
                asym = (l_vals - r_vals).abs()
                asym_clean = asym.dropna()
                if len(asym_clean) > 10:
                    rec["pressure_asymmetry_median"] = float(asym_clean.median())
                    rec["pressure_asymmetry_p95"] = float(asym_clean.quantile(0.95))

        records.append(rec)

    return pd.DataFrame(records) if records else pd.DataFrame()


def track_degradation(
    output_dir: Path,
    machine: str,
    technique: str | None = None,
    site: str | None = None,
    window: int = 30,
    features_df: pd.DataFrame | None = None,
) -> dict:
    """Track sensor degradation for a machine within controlled groups.

    Computes rolling baselines and detects systematic drift using linear
    regression over time for each controlled group.

    Parameters
    ----------
    output_dir : Path to output directory
    machine : Machine slug to analyze
    technique : Optional technique filter
    site : Optional site filter for within-site analysis
    window : Rolling window size (number of sessions)
    features_df : Pre-computed features DataFrame. If None, will be extracted.

    Returns dict with drift results per sensor metric.
    """
    if features_df is None:
        features_df = extract_sensor_features(
            output_dir, machine=machine, technique=technique, site=site
        )

    if features_df.empty or len(features_df) < window:
        return {"machine": machine, "n_sessions": len(features_df), "metrics": {}}

    features_df = features_df.sort_values("start_time").reset_index(drop=True)

    # Find sensor metric columns (those with __ separator)
    metric_cols = [c for c in features_df.columns if "__" in c]

    # Add derived metrics
    if "max_depth" in features_df.columns and "duration_s" in features_df.columns:
        depth = features_df["max_depth"]
        dur = features_df["duration_s"]
        mask = depth > 1.0
        features_df.loc[mask, "duration_per_meter"] = dur[mask] / depth[mask]
        metric_cols.append("duration_per_meter")

    if "pressure_asymmetry_median" in features_df.columns:
        metric_cols.append("pressure_asymmetry_median")
    if "pressure_asymmetry_p95" in features_df.columns:
        metric_cols.append("pressure_asymmetry_p95")

    results = {
        "machine": machine,
        "technique": technique or "all",
        "site": site or "all",
        "n_sessions": len(features_df),
        "date_range": (
            str(features_df["start_time"].min().date()),
            str(features_df["start_time"].max().date()),
        ),
        "metrics": {},
    }

    # Compute trend for each metric
    for col in metric_cols:
        vals = features_df[col].dropna()
        if len(vals) < window:
            continue

        # Rolling baseline
        rolling_mean = vals.rolling(window, min_periods=window // 2).mean()

        # Linear regression: y = slope * x + intercept
        x = np.arange(len(vals), dtype=float)
        y = vals.values

        # Remove NaN for regression
        valid = ~np.isnan(y)
        if valid.sum() < window:
            continue

        x_valid = x[valid]
        y_valid = y[valid]

        try:
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(x_valid, y_valid)
        except ImportError:
            # Fallback: simple numpy polyfit
            coeffs = np.polyfit(x_valid, y_valid, 1)
            slope = coeffs[0]
            p_value = np.nan  # Can't compute without scipy

        # Relative slope: change per session as fraction of median
        median_val = np.median(y_valid)
        rel_slope = slope / abs(median_val) if abs(median_val) > 1e-6 else 0.0

        # Classify drift
        if p_value < 0.05 and abs(rel_slope) > 0.001:
            direction = "RISING" if slope > 0 else "FALLING"
        else:
            direction = "STABLE"

        results["metrics"][col] = {
            "slope_per_session": float(slope),
            "relative_slope": float(rel_slope),
            "p_value": float(p_value) if not np.isnan(p_value) else None,
            "r_squared": float(r_value ** 2) if not np.isnan(p_value) else None,
            "direction": direction,
            "median": float(median_val),
            "first_quarter_mean": float(np.mean(y_valid[:len(y_valid) // 4])),
            "last_quarter_mean": float(np.mean(y_valid[-len(y_valid) // 4:])),
        }

    return results


def track_cut_health(
    output_dir: Path,
    machine: str,
    site: str | None = None,
    features_df: pd.DataFrame | None = None,
) -> dict:
    """CUT-specific health analysis tracking temperature, pressure asymmetry,
    gearbox oil pressure, and leakage pressure trends.

    Parameters
    ----------
    output_dir : Path to output directory
    machine : Machine slug (must be CUT machine)
    site : Optional site filter
    features_df : Pre-computed features DataFrame. If None, will be extracted.

    Returns dict with CUT-specific health indicators.
    """
    if features_df is None:
        features_df = extract_sensor_features(
            output_dir, machine=machine, technique="CUT", site=site
        )

    if features_df.empty:
        return {"machine": machine, "n_sessions": 0, "health": {}}

    features_df = features_df.sort_values("start_time").reset_index(drop=True)
    n = len(features_df)
    health = {}

    def _trend(col_pattern: str) -> dict | None:
        """Compute trend for columns matching pattern."""
        matching = [c for c in features_df.columns if col_pattern in c and "__median" in c]
        if not matching:
            return None
        col = matching[0]
        vals = features_df[col].dropna()
        if len(vals) < 10:
            return None
        q1 = vals.iloc[:max(1, len(vals) // 4)].mean()
        q4 = vals.iloc[-max(1, len(vals) // 4):].mean()
        change_pct = ((q4 - q1) / abs(q1) * 100) if abs(q1) > 1e-6 else 0
        return {
            "first_quarter_avg": float(q1),
            "last_quarter_avg": float(q4),
            "change_pct": float(change_pct),
            "direction": "RISING" if change_pct > 10 else ("FALLING" if change_pct < -10 else "STABLE"),
            "n_sessions": len(vals),
        }

    # Temperature baselines
    for sensor_key in ["Temperatur FRL", "Temperatur FRR"]:
        trend = _trend(sensor_key)
        if trend:
            health[f"temperature_{sensor_key}"] = trend

    # Pressure asymmetry
    if "pressure_asymmetry_median" in features_df.columns:
        asym = features_df["pressure_asymmetry_median"].dropna()
        if len(asym) >= 10:
            q1 = asym.iloc[:max(1, len(asym) // 4)].mean()
            q4 = asym.iloc[-max(1, len(asym) // 4):].mean()
            health["pressure_asymmetry"] = {
                "first_quarter_avg": float(q1),
                "last_quarter_avg": float(q4),
                "direction": "GROWING" if q4 > q1 * 1.15 else "STABLE",
                "n_sessions": len(asym),
            }

    # Gearbox oil pressure
    for sensor_key in ["Oeldruck Getriebe links", "Oeldruck Getriebe rechts"]:
        trend = _trend(sensor_key)
        if trend:
            health[f"oil_pressure_{sensor_key}"] = trend

    # Leakage pressure
    for sensor_key in ["Leckagedruck Getriebe links", "Leckagedruck Getriebe rechts"]:
        trend = _trend(sensor_key)
        if trend:
            health[f"leakage_{sensor_key}"] = trend

    return {
        "machine": machine,
        "site": site or "all",
        "n_sessions": n,
        "health": health,
    }
