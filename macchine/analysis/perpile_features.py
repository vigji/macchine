"""Per-pile feature extraction library.

Extracts depth profiles, verticality metrics, and summary features from
individual pile/element trace data for cross-pile comparison and dashboards.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from macchine.harmonize.calibration import (
    clean_sentinels_df,
    is_calibrated,
    validate_physical_range,
)


# ── Technique-specific sensor definitions ─────────────────────────────────────

TECHNIQUE_SENSORS = {
    "SOB": {
        "depth": "Tiefe",
        "primary": ["Drehmoment", "Betondruck", "Vorschub-KraftPM"],
        "pressure": ["Druck Pumpe 1", "Druck Pumpe 2"],
        "verticality": ["Neigung X Mast", "Neigung Y Mast"],
        "extra": ["Betondurchfluss", "Betonmenge", "Eindringwiderstand"],
    },
    "KELLY": {
        "depth": "Tiefe",
        "primary": ["DrehmomentkNm", "KDK Druck", "Seilkraft Hauptwinde"],
        "pressure": ["Druck Pumpe 1", "Druck Pumpe 2"],
        "verticality": ["Neigung X Mast", "Neigung Y Mast"],
        "extra": ["KDK Drehzahl", "Drehzahl Rohr"],
    },
    "CUT": {
        "depth": "Tiefe",
        "primary": ["Druck FRL", "Druck FRR", "Drehzahl FRL", "Drehzahl FRR"],
        "pressure": ["Druck FRL", "Druck FRR"],
        "verticality": ["Abweichung X", "Abweichung Y"],
        "extra": ["Temperatur FRL", "Temperatur FRR", "Seilkraft Fräswinde"],
    },
    "GRAB": {
        "depth": "Tiefe",
        "primary": ["Druck Pumpe 1", "Seilkraft"],
        "pressure": ["Druck Pumpe 1", "Druck Pumpe 2"],
        "verticality": ["Neigung X", "Neigung Y", "Abweichung X", "Abweichung Y"],
        "extra": ["Seilkraft Hauptwinde"],
    },
    "SCM": {
        "depth": "Tiefe",
        "primary": ["Drehmoment", "Susp.-Druck", "Susp.-Durchfl."],
        "pressure": ["Druck Pumpe 1"],
        "verticality": ["Neigung X Mast", "Neigung Y Mast"],
        "extra": ["Drehzahl", "Susp.-Mg"],
    },
}

# Core expected sensors per technique (for completeness checking)
EXPECTED_SENSORS = {
    "SOB": ["Tiefe", "Drehmoment", "Druck Pumpe 1", "Betondurchfluss",
            "Neigung X Mast", "Neigung Y Mast"],
    "KELLY": ["Tiefe", "DrehmomentkNm", "KDK Druck", "Seilkraft Hauptwinde",
              "Druck Pumpe 1"],
    "CUT": ["Tiefe", "Druck FRL", "Druck FRR", "Drehzahl FRL", "Drehzahl FRR",
            "Temperatur FRL", "Temperatur FRR", "Seilkraft Fräswinde"],
    "GRAB": ["Tiefe", "Seilkraft", "Druck Pumpe 1", "Druck Pumpe 2",
             "Neigung X", "Neigung Y"],
    "SCM": ["Tiefe", "Drehmoment", "Drehzahl", "Susp.-Druck", "Susp.-Durchfl."],
}


def _to_numeric(series: pd.Series) -> pd.Series:
    """Coerce a series to numeric, handling object-typed columns."""
    if series.dtype == object:
        return pd.to_numeric(series, errors="coerce")
    return series


def _get_col(df: pd.DataFrame, name: str) -> pd.Series | None:
    """Get a numeric column from df, or None if unavailable."""
    if name not in df.columns:
        return None
    s = _to_numeric(df[name]).copy()
    s = s.replace([np.inf, -np.inf], np.nan)
    if s.dropna().empty:
        return None
    return s


def extract_pile_features(trace_df: pd.DataFrame, technique: str,
                          machine_slug: str = "") -> dict:
    """Extract summary features from a single pile trace.

    Calibration-aware: skips uncalibrated sensors for verticality and pressure
    metrics, applies physical range validation to depth/deviation/inclination,
    and computes depth-normalized metrics.

    Returns dict with keys: max_depth, mean_pressure, mean_torque,
    max_deviation, depth_range, duration_active_s, n_cycles,
    inclination_rms, verticality_ok, pressure_per_meter, torque_per_meter,
    duration_per_meter.
    """
    tdf = clean_sentinels_df(trace_df)
    tech_cfg = TECHNIQUE_SENSORS.get(technique, {})
    depth_col = tech_cfg.get("depth", "Tiefe")

    features = {}

    # Depth features — always validate physical range
    depth = _get_col(tdf, depth_col)
    if depth is not None:
        depth = validate_physical_range(depth, depth_col, machine_slug)
        depth = depth.dropna()
    if depth is not None and not depth.empty:
        features["max_depth"] = float(depth.max())
        features["min_depth"] = float(depth.min())
        features["depth_range"] = features["max_depth"] - features["min_depth"]
    else:
        features["max_depth"] = np.nan
        features["min_depth"] = np.nan
        features["depth_range"] = np.nan

    # Pressure features — only use calibrated sensors
    pressure_vals = []
    for p_sensor in tech_cfg.get("pressure", []):
        if not is_calibrated(p_sensor, machine_slug):
            continue
        p = _get_col(tdf, p_sensor)
        if p is not None:
            p = validate_physical_range(p, p_sensor, machine_slug)
            p_clean = p.dropna()
            if not p_clean.empty:
                pressure_vals.append(float(p_clean.mean()))
    features["mean_pressure"] = float(np.mean(pressure_vals)) if pressure_vals else np.nan

    # Torque features — only use calibrated sensors
    torque_sensors = [s for s in tech_cfg.get("primary", [])
                      if "moment" in s.lower() or "drehmoment" in s.lower()]
    torque_vals = []
    for t_sensor in torque_sensors:
        if not is_calibrated(t_sensor, machine_slug):
            continue
        t = _get_col(tdf, t_sensor)
        if t is not None:
            t = validate_physical_range(t, t_sensor, machine_slug)
            t_clean = t.dropna()
            if not t_clean.empty:
                torque_vals.append(float(t_clean.mean()))
    features["mean_torque"] = float(np.mean(torque_vals)) if torque_vals else np.nan

    # Duration
    if "timestamp" in tdf.columns:
        ts = pd.to_datetime(tdf["timestamp"], errors="coerce").dropna()
        if len(ts) >= 2:
            features["duration_active_s"] = (ts.max() - ts.min()).total_seconds()
        else:
            features["duration_active_s"] = np.nan
    else:
        features["duration_active_s"] = len(tdf)  # approx at 1Hz

    # Cycle detection (for GRAB/KELLY: depth goes up and down)
    if depth is not None and not depth.empty and technique in ("GRAB", "KELLY"):
        # Count sign changes in depth derivative
        ddepth = depth.diff().dropna()
        if len(ddepth) > 1:
            sign_changes = (ddepth.iloc[:-1].values * ddepth.iloc[1:].values < 0).sum()
            features["n_cycles"] = int(sign_changes // 2)
        else:
            features["n_cycles"] = 0
    else:
        features["n_cycles"] = 0

    # Verticality features — pass machine_slug for calibration checking
    vert = verticality_features(tdf, tech_cfg, machine_slug=machine_slug)
    features.update(vert)

    # Depth-normalized metrics
    depth_range = features.get("depth_range", np.nan)
    if pd.notna(depth_range) and depth_range > 1.0:
        features["pressure_per_meter"] = features["mean_pressure"] / depth_range if pd.notna(features["mean_pressure"]) else np.nan
        features["torque_per_meter"] = features["mean_torque"] / depth_range if pd.notna(features["mean_torque"]) else np.nan
        dur = features.get("duration_active_s", np.nan)
        features["duration_per_meter"] = dur / depth_range if pd.notna(dur) else np.nan
    else:
        features["pressure_per_meter"] = np.nan
        features["torque_per_meter"] = np.nan
        features["duration_per_meter"] = np.nan

    return features


def depth_profile_features(trace_df: pd.DataFrame, sensor: str,
                           depth_col: str = "Tiefe",
                           bin_size: float = 1.0) -> dict:
    """Compute binned sensor-vs-depth profiles.

    Returns dict with:
    - depth_bins: list of bin centers
    - sensor_mean: mean sensor value per bin
    - sensor_std: std per bin
    - sensor_median: median per bin
    """
    tdf = clean_sentinels_df(trace_df)
    depth = _get_col(tdf, depth_col)
    vals = _get_col(tdf, sensor)

    if depth is None or vals is None:
        return {"depth_bins": [], "sensor_mean": [], "sensor_std": [],
                "sensor_median": []}

    combined = pd.DataFrame({"depth": depth, "value": vals}).dropna()
    if combined.empty:
        return {"depth_bins": [], "sensor_mean": [], "sensor_std": [],
                "sensor_median": []}

    # Create depth bins
    min_d, max_d = combined["depth"].min(), combined["depth"].max()
    bins = np.arange(min_d, max_d + bin_size, bin_size)
    if len(bins) < 2:
        return {"depth_bins": [], "sensor_mean": [], "sensor_std": [],
                "sensor_median": []}

    combined["depth_bin"] = pd.cut(combined["depth"], bins=bins, labels=False)
    binned = combined.groupby("depth_bin")["value"].agg(["mean", "std", "median"])
    bin_centers = bins[:-1] + bin_size / 2

    valid = binned.dropna(subset=["mean"])
    valid_centers = bin_centers[valid.index.astype(int)]

    return {
        "depth_bins": valid_centers.tolist(),
        "sensor_mean": valid["mean"].tolist(),
        "sensor_std": valid["std"].fillna(0).tolist(),
        "sensor_median": valid["median"].tolist(),
    }


def verticality_features(trace_df: pd.DataFrame,
                         tech_cfg: dict | None = None,
                         machine_slug: str = "") -> dict:
    """Extract verticality/inclination features from a trace.

    Calibration-aware: skips uncalibrated deviation/inclination sensors and
    applies physical range validation on calibrated ones.

    Returns dict with: max_deviation, inclination_rms, inclination_x_mean,
    inclination_y_mean, verticality_ok.
    """
    tdf = trace_df  # already cleaned by caller if needed

    # Determine which columns to use
    vert_sensors = (tech_cfg or {}).get("verticality", [])

    result = {
        "max_deviation": np.nan,
        "inclination_rms": np.nan,
        "inclination_x_mean": np.nan,
        "inclination_y_mean": np.nan,
        "verticality_ok": True,
    }

    # Look for X/Y inclination pairs
    x_col = None
    y_col = None
    dev_x_col = None
    dev_y_col = None

    for s in vert_sensors:
        if "X" in s and "Abweichung" in s:
            dev_x_col = s
        elif "Y" in s and "Abweichung" in s:
            dev_y_col = s
        elif "X" in s:
            x_col = s
        elif "Y" in s:
            y_col = s

    # Deviation (mm) — direct measure of verticality
    # Skip uncalibrated deviation sensors
    dev_x = None
    dev_y = None
    if dev_x_col and is_calibrated(dev_x_col, machine_slug):
        dev_x = _get_col(tdf, dev_x_col)
        if dev_x is not None:
            dev_x = validate_physical_range(dev_x, dev_x_col, machine_slug)
    if dev_y_col and is_calibrated(dev_y_col, machine_slug):
        dev_y = _get_col(tdf, dev_y_col)
        if dev_y is not None:
            dev_y = validate_physical_range(dev_y, dev_y_col, machine_slug)

    if dev_x is not None and dev_y is not None:
        radial = np.sqrt(dev_x**2 + dev_y**2)
        max_rad = radial.max()
        if pd.notna(max_rad):
            result["max_deviation"] = float(max_rad)
    elif dev_x is not None:
        max_val = dev_x.abs().max()
        if pd.notna(max_val):
            result["max_deviation"] = float(max_val)
    elif dev_y is not None:
        max_val = dev_y.abs().max()
        if pd.notna(max_val):
            result["max_deviation"] = float(max_val)

    # Inclination (degrees) — mast tilt
    # Skip uncalibrated inclination sensors
    inc_x = None
    inc_y = None
    if x_col and is_calibrated(x_col, machine_slug):
        inc_x = _get_col(tdf, x_col)
        if inc_x is not None:
            inc_x = validate_physical_range(inc_x, x_col, machine_slug)
    if y_col and is_calibrated(y_col, machine_slug):
        inc_y = _get_col(tdf, y_col)
        if inc_y is not None:
            inc_y = validate_physical_range(inc_y, y_col, machine_slug)

    if inc_x is not None and inc_y is not None:
        ix_clean = inc_x.dropna()
        iy_clean = inc_y.dropna()
        if not ix_clean.empty and not iy_clean.empty:
            result["inclination_x_mean"] = float(ix_clean.mean())
            result["inclination_y_mean"] = float(iy_clean.mean())
            rms = float(np.sqrt((ix_clean**2 + iy_clean**2).mean()))
            result["inclination_rms"] = rms
            result["verticality_ok"] = rms < 2.0
    elif inc_x is not None:
        ix_clean = inc_x.dropna()
        if not ix_clean.empty:
            result["inclination_x_mean"] = float(ix_clean.mean())
            result["inclination_rms"] = float(np.sqrt((ix_clean**2).mean()))

    return result


def get_deviation_trajectory(trace_df: pd.DataFrame,
                             tech_cfg: dict | None = None) -> pd.DataFrame | None:
    """Extract X/Y deviation trajectory over depth for verticality plots.

    Returns DataFrame with columns: depth, dev_x, dev_y (or None if unavailable).
    """
    tdf = clean_sentinels_df(trace_df)
    vert_sensors = (tech_cfg or {}).get("verticality", [])

    depth = _get_col(tdf, "Tiefe")
    if depth is None:
        return None

    # Find deviation or inclination columns
    x_col = None
    y_col = None
    for s in vert_sensors:
        if "X" in s:
            x_col = s
        elif "Y" in s:
            y_col = s

    if x_col is None and y_col is None:
        return None

    result = pd.DataFrame({"depth": depth})
    if x_col:
        result["dev_x"] = _get_col(tdf, x_col)
    if y_col:
        result["dev_y"] = _get_col(tdf, y_col)

    return result.dropna(subset=["depth"])


def select_representative_piles(site_df: pd.DataFrame,
                                features_df: pd.DataFrame,
                                n: int = 5) -> list[str]:
    """Select N representative/interesting piles from a site.

    Selection criteria: deepest, shallowest, longest duration,
    shortest duration, most inclined.

    Parameters
    ----------
    site_df : DataFrame of merged trace index rows for one site
    features_df : DataFrame with pile features (from extract_pile_features)
        Must have columns: element_name, max_depth, duration_active_s, inclination_rms
    n : max number of piles to return

    Returns list of element_name strings.
    """
    if features_df.empty:
        return []

    selected = set()

    # Deepest pile
    if "max_depth" in features_df.columns:
        valid = features_df.dropna(subset=["max_depth"])
        if not valid.empty:
            selected.add(valid.loc[valid["max_depth"].idxmax(), "element_name"])

    # Shallowest pile (among those with some depth)
    if "max_depth" in features_df.columns:
        valid = features_df[features_df["max_depth"] > 0].dropna(subset=["max_depth"])
        if not valid.empty:
            selected.add(valid.loc[valid["max_depth"].idxmin(), "element_name"])

    # Longest duration
    if "duration_active_s" in features_df.columns:
        valid = features_df.dropna(subset=["duration_active_s"])
        if not valid.empty:
            selected.add(valid.loc[valid["duration_active_s"].idxmax(), "element_name"])

    # Shortest duration
    if "duration_active_s" in features_df.columns:
        valid = features_df[features_df["duration_active_s"] > 0].dropna(subset=["duration_active_s"])
        if not valid.empty:
            selected.add(valid.loc[valid["duration_active_s"].idxmin(), "element_name"])

    # Most inclined
    if "inclination_rms" in features_df.columns:
        valid = features_df.dropna(subset=["inclination_rms"])
        if not valid.empty:
            selected.add(valid.loc[valid["inclination_rms"].idxmax(), "element_name"])

    # Fill up with random piles if we haven't reached n
    remaining = features_df[~features_df["element_name"].isin(selected)]
    while len(selected) < n and not remaining.empty:
        pick = remaining.sample(1)
        selected.add(pick["element_name"].iloc[0])
        remaining = remaining[~remaining["element_name"].isin(selected)]

    return list(selected)[:n]
