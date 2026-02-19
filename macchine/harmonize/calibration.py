"""Calibration status utilities.

Loads calibration_status.yaml and provides helpers to:
- Check if a sensor is calibrated for a given machine
- Get proper axis labels (engineering units vs 'arb. units')
- Clean sentinel values (replace with NaN)
- Format display labels as 'English Name (German Original)'
"""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import yaml

# Physical plausible ranges for calibrated sensors (min, max).
# Values outside these ranges indicate uncalibrated or corrupted data.
PHYSICAL_RANGES: dict[str, tuple[float, float]] = {
    "depth": (-200, 200),
    "feed_depth": (-200, 200),
    "depth_winch_2": (-200, 200),
    "depth_main_winch_gok": (-200, 200),
    "depth_casing_gok": (-200, 200),
    "inclination_x": (-90, 90),
    "inclination_y": (-90, 90),
    "inclination_x_mast": (-90, 90),
    "inclination_y_mast": (-90, 90),
    "deviation_x": (-5000, 5000),
    "deviation_y": (-5000, 5000),
    "measuring_point_deviation_x": (-50000, 50000),
    "measuring_point_deviation_y": (-50000, 50000),
    "pump_pressure": (0, 1000),
    "pump_pressure_1": (0, 1000),
    "pump_pressure_2": (0, 1000),
    "pump_pressure_3": (0, 1000),
    "pump_pressure_4": (0, 1000),
    "pressure_cutter_left": (0, 500),
    "pressure_cutter_right": (0, 500),
    "kdk_pressure": (0, 1000),
    "concrete_pressure": (0, 50),
    "torque": (0, 120),
    "torque_knm": (0, 1000),
    "torque_percent": (0, 120),
    "rope_force": (0, 200),
    "rope_force_main_winch": (0, 1000),
    "rope_force_auxiliary_winch": (0, 200),
    "rope_force_cutter_winch": (0, 200),
    "rope_force_winch_2": (0, 200),
    "temperature_cutter_left": (-20, 150),
    "temperature_cutter_right": (-20, 150),
    "temperature_distribution_gearbox": (-20, 200),
    "oil_pressure_gearbox_left": (0, 50),
    "oil_pressure_gearbox_right": (0, 50),
    "leakage_pressure_gearbox_left": (0, 20),
    "leakage_pressure_gearbox_right": (0, 20),
    "leak_oil_pressure": (0, 50),
    "water_content_gearbox_oil_left": (0, 100),
    "water_content_gearbox_oil_right": (0, 100),
    "rotation_speed": (0, 100),
    "rotation_speed_cutter_left": (0, 100),
    "rotation_speed_cutter_right": (0, 100),
    "kdk_rotation_speed": (0, 100),
    "surcharge_load": (0, 200),
    "feed_force": (-200, 200),
    "feed_force_scm": (-200, 200),
    "pitch": (-90, 90),
    "rotation_rate": (-360, 360),
    "twist_angle": (-360, 360),
}

_CALIBRATION_PATH = Path(__file__).parent / "calibration_status.yaml"
_DEFINITIONS_PATH = Path(__file__).parent / "sensor_definitions.yaml"
_DAT_CORRECTIONS_PATH = Path(__file__).parent / "dat_corrections.yaml"

# Known units for calibrated sensors (sensor_name -> unit string)
SENSOR_UNITS = {
    "depth": "m",
    "feed_depth": "m",
    "depth_winch_2": "m",
    "depth_main_winch_gok": "m",
    "depth_casing_gok": "m",
    "torque": "%",
    "torque_knm": "kNm",
    "torque_percent": "%",
    "torque_stages": "stage",
    "rotation_direction": "",
    "rotation_direction_cutter_left": "",
    "rotation_direction_cutter_right": "",
    "pressure_cutter_left": "bar",
    "pressure_cutter_right": "bar",
    "pump_pressure": "bar",
    "pump_pressure_1": "bar",
    "pump_pressure_2": "bar",
    "pump_pressure_3": "bar",
    "pump_pressure_4": "bar",
    "kdk_pressure": "bar",
    "concrete_pressure": "bar",
    "suspension_pressure": "bar",
    "suspension_pressure_2": "bar",
    "suspension_pressure_bottom": "bar",
    "oil_pressure_gearbox_left": "bar",
    "oil_pressure_gearbox_right": "bar",
    "leakage_pressure_gearbox_left": "bar",
    "leakage_pressure_gearbox_right": "bar",
    "leak_oil_pressure": "bar",
    "rope_force": "t",
    "rope_force_main_winch": "t",
    "rope_force_auxiliary_winch": "t",
    "rope_force_cutter_winch": "t",
    "rope_force_winch_2": "t",
    "inclination_x": "deg",
    "inclination_y": "deg",
    "inclination_x_mast": "deg",
    "inclination_y_mast": "deg",
    "surcharge_load": "t",
    "temperature_cutter_left": "\u00b0C",
    "temperature_cutter_right": "\u00b0C",
    "temperature_distribution_gearbox": "\u00b0C",
    "water_content_gearbox_oil_left": "%",
    "water_content_gearbox_oil_right": "%",
    "deviation_x": "mm",
    "deviation_y": "mm",
    "measuring_point_deviation_x": "mm",
    "measuring_point_deviation_y": "mm",
    "rotation_speed": "rpm",
    "rotation_speed_cutter_left": "rpm",
    "rotation_speed_cutter_right": "rpm",
    "kdk_rotation_speed": "rpm",
    "feed_force": "kN",
    "feed_force_scm": "kN",
    "penetration_resistance": "kN/m",
    "suspension_flow": "l/min",
    "suspension_flow_2": "l/min",
    "suspension_volume": "m\u00b3",
    "suspension_volume_2": "m\u00b3",
    "suspension_volume_total": "m\u00b3",
    "volume_per_meter": "l/m",
    "concrete_flow": "l/min",
    "concrete_volume": "m\u00b3",
    "total_concrete_volume": "m\u00b3",
    "pump_flow": "l/min",
    "return_flow": "l/min",
    "feed_flow": "l/min",
    "feed_speed": "mm/s",
    "current_channel_1": "A",
    "drilling_limit": "m",
    "casing_length": "m",
    "tool_casing_bottom_diff": "m",
    "pitch": "deg",
    "rotation_rate": "deg/s",
    "twist_angle": "deg",
}


@lru_cache(maxsize=1)
def _load_calibration_yaml() -> dict:
    with open(_CALIBRATION_PATH) as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def _load_sensor_defs() -> dict:
    with open(_DEFINITIONS_PATH) as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def _load_dat_corrections() -> dict[str, dict[str, float]]:
    """Load empirical DAT correction divisors from YAML.

    Returns {machine_slug: {sensor_name: divisor}}.
    """
    if not _DAT_CORRECTIONS_PATH.exists():
        return {}
    with open(_DAT_CORRECTIONS_PATH) as f:
        data = yaml.safe_load(f)
    return data or {}


def get_dat_correction(machine_slug: str, sensor_name: str) -> float:
    """Get the empirical correction divisor for a DAT-parsed sensor.

    Returns the divisor to apply AFTER the header-declared divisor.
    Returns 1.0 if no correction is needed.
    """
    corrections = _load_dat_corrections()
    machine_corr = corrections.get(machine_slug, {})
    return float(machine_corr.get(sensor_name, 1.0))


def get_calibrated_set() -> set[str]:
    """Return set of sensor names that are calibrated on ALL machines."""
    data = _load_calibration_yaml()
    return set(data.get("calibrated", []))


def get_per_machine_status() -> dict[str, dict[str, str]]:
    """Return {sensor: {machine: 'calibrated'|'uncalibrated'}} for mixed sensors."""
    data = _load_calibration_yaml()
    return data.get("per_machine", {})


def get_sentinel_values() -> list[float]:
    """Return list of sentinel values to treat as NaN."""
    data = _load_calibration_yaml()
    return [float(v) for v in data.get("sentinel_values", [])]


def is_calibrated(sensor_name: str, machine_slug: str) -> bool:
    """Check if a sensor is calibrated for a given machine.

    Returns True if:
    - sensor is in the 'calibrated' list (calibrated on all machines), OR
    - sensor is in 'per_machine' with status 'calibrated' for this machine
    Returns False if:
    - sensor is in 'per_machine' with status 'uncalibrated' for this machine
    - sensor is not listed at all (assume uncalibrated / unknown)
    """
    calibrated_all = get_calibrated_set()
    if sensor_name in calibrated_all:
        return True

    per_machine = get_per_machine_status()
    if sensor_name in per_machine:
        machine_statuses = per_machine[sensor_name]
        status = machine_statuses.get(machine_slug, None)
        if status == "calibrated":
            return True
        if status == "uncalibrated":
            return False
        # Machine not listed — check if any machine is calibrated
        # If so, assume unknown machines might be uncalibrated
        return False

    # Sensor not listed in calibration file at all — assume calibrated
    # (most unlisted sensors have reasonable values)
    return True


def get_unit(sensor_name: str, machine_slug: str) -> str:
    """Get the display unit for a sensor on a given machine.

    Returns the engineering unit if calibrated, 'arb. units' if not.
    """
    if is_calibrated(sensor_name, machine_slug):
        return SENSOR_UNITS.get(sensor_name, "")
    return "arb. units"


def get_english_name(sensor_name: str) -> str:
    """Get a human-readable title from a canonical English sensor name."""
    return sensor_name.replace("_", " ").title()


def get_display_label(sensor_name: str) -> str:
    """Format sensor name as a human-readable title for plot labels."""
    return sensor_name.replace("_", " ").title()


def get_axis_label(sensor_name: str, machine_slug: str) -> str:
    """Get a full axis label: 'Display Name\n[unit]'."""
    display = get_display_label(sensor_name)
    unit = get_unit(sensor_name, machine_slug)
    if unit:
        return f"{display}\n[{unit}]"
    return display


def clean_sentinels(series: pd.Series) -> pd.Series:
    """Replace sentinel values with NaN in a pandas Series."""
    sentinels = get_sentinel_values()
    mask = series.isin(sentinels)
    if mask.any():
        series = series.copy()
        series[mask] = np.nan
    return series


def clean_sentinels_df(df: pd.DataFrame) -> pd.DataFrame:
    """Replace sentinel values with NaN in all numeric columns of a DataFrame."""
    sentinels = get_sentinel_values()
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        mask = df[col].isin(sentinels)
        if mask.any():
            df.loc[mask, col] = np.nan
    return df


# ── Quality flags ─────────────────────────────────────────────────────────

# Quality flag values (int8 for memory efficiency)
QC_OK = np.int8(0)
QC_SENTINEL = np.int8(1)
QC_OUT_OF_RANGE = np.int8(2)


def clean_and_flag_df(
    df: pd.DataFrame,
    machine_slug: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean sentinels and flag data quality for every numeric cell.

    Returns
    -------
    cleaned_df : DataFrame
        Copy of *df* with sentinel values replaced by NaN.
    qc_df : DataFrame
        Same shape as *df* (numeric columns only), dtype int8.
        Cell values: QC_OK (0), QC_SENTINEL (1), QC_OUT_OF_RANGE (2).
        Out-of-range flags are only set for sensors with known
        PHYSICAL_RANGES; the values are NOT replaced in cleaned_df.
    """
    sentinels = set(get_sentinel_values())
    cleaned = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude timestamp from QC
    numeric_cols = [c for c in numeric_cols if c != "timestamp"]

    qc = pd.DataFrame(QC_OK, index=df.index, columns=numeric_cols, dtype=np.int8)

    for col in numeric_cols:
        series = df[col]

        # Sentinel detection
        sentinel_mask = series.isin(sentinels)
        if sentinel_mask.any():
            qc.loc[sentinel_mask, col] = QC_SENTINEL
            cleaned.loc[sentinel_mask, col] = np.nan

        # Out-of-range detection (only for sensors with known ranges)
        if col in PHYSICAL_RANGES:
            lo, hi = PHYSICAL_RANGES[col]
            non_sentinel = ~sentinel_mask & series.notna()
            oor_mask = non_sentinel & ((series < lo) | (series > hi))
            if oor_mask.any():
                qc.loc[oor_mask, col] = QC_OUT_OF_RANGE

    return cleaned, qc


def qc_summary(qc_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise a QC flags DataFrame into one row per sensor.

    Returns a DataFrame with columns:
        sensor, n_total, n_ok, n_sentinel, n_out_of_range,
        pct_sentinel, pct_out_of_range
    """
    rows = []
    for col in qc_df.columns:
        counts = qc_df[col].value_counts()
        n_total = len(qc_df)
        n_sent = int(counts.get(QC_SENTINEL, 0))
        n_oor = int(counts.get(QC_OUT_OF_RANGE, 0))
        n_ok = n_total - n_sent - n_oor
        rows.append({
            "sensor": col,
            "n_total": n_total,
            "n_ok": n_ok,
            "n_sentinel": n_sent,
            "n_out_of_range": n_oor,
            "pct_sentinel": round(100 * n_sent / n_total, 1) if n_total else 0,
            "pct_out_of_range": round(100 * n_oor / n_total, 1) if n_total else 0,
        })
    return pd.DataFrame(rows)


def detect_dat_divisor(values: list[float | None], sensor_name: str) -> int:
    """DEPRECATED: heuristic-based divisor detection.

    This function is superseded by the empirical correction table in
    dat_corrections.yaml (loaded via get_dat_correction). The heuristic
    was unreliable — it over-corrected some sensors and under-corrected
    others. Kept for backward compatibility with tests.

    Returns 1 if no correction is needed or the sensor has no known range.
    """
    if sensor_name not in PHYSICAL_RANGES:
        return 1

    lo, hi = PHYSICAL_RANGES[sensor_name]
    range_max = max(abs(lo), abs(hi))

    # Filter to valid, non-zero values (keeping sign for step analysis)
    raw_valid = [v for v in values if v is not None and not np.isnan(v)]
    abs_valid = [abs(v) for v in raw_valid if v != 0]
    if len(abs_valid) < 5:
        return 1

    abs_valid.sort()
    p90 = abs_valid[int(len(abs_valid) * 0.9)]

    # If the p90 is within the range, check step size as a secondary signal
    if p90 <= range_max:
        # Values are within range, but might still be sub-units if steps are too large.
        # Only refine if data is all-integer (typical of DAT sub-unit encoding).
        all_integer = all(v == int(v) for v in raw_valid[:200])
        if not all_integer or len(raw_valid) < 20:
            return 1

        # Compute median non-zero step (consecutive differences)
        diffs = [abs(raw_valid[i + 1] - raw_valid[i]) for i in range(min(500, len(raw_valid) - 1))]
        nonzero_diffs = [d for d in diffs if d > 0]
        if len(nonzero_diffs) < 5:
            return 1
        nonzero_diffs.sort()
        median_step = nonzero_diffs[len(nonzero_diffs) // 2]

        # If the median step is ≥ 2 integer units, values might be in sub-units.
        # A correct meter-resolution sensor has step ~1; a centimeter sensor
        # read as meters has step ~2+ (because real movement is ~2cm/step).
        if median_step < 2:
            return 1

        # Try divisors: pick the one where corrected step is 0.01–0.5 (typical float resolution)
        for divisor in [10, 100, 1000]:
            corrected_step = median_step / divisor
            corrected_p90 = p90 / divisor
            if 0.005 <= corrected_step <= 0.5 and corrected_p90 <= range_max:
                return divisor

        return 1

    # p90 exceeds range — definitely need correction
    # Collect all divisors that bring p90 within physical range
    candidates = []
    for divisor in [10, 100, 1000]:
        corrected_p90 = p90 / divisor
        if corrected_p90 <= range_max:
            candidates.append((divisor, corrected_p90))

    if not candidates:
        return 1

    # Use step size to pick the best divisor if data is all-integer.
    # Try divisors from LARGEST to SMALLEST — prefer stronger corrections
    # when the step size confirms sub-unit encoding. Require corrected p90 >= 5
    # to prevent over-correction (e.g., ÷1000 turning 3500 rpm into 3.5 rpm).
    all_integer = all(v == int(v) for v in raw_valid[:200])
    if all_integer and len(raw_valid) >= 20:
        diffs = [abs(raw_valid[i + 1] - raw_valid[i]) for i in range(min(500, len(raw_valid) - 1))]
        nonzero_diffs = [d for d in diffs if d > 0]
        if len(nonzero_diffs) >= 5:
            nonzero_diffs.sort()
            median_step = nonzero_diffs[len(nonzero_diffs) // 2]
            for divisor in [1000, 100, 10]:  # largest first
                corrected_step = median_step / divisor
                corrected_p90 = p90 / divisor
                if (
                    0.005 <= corrected_step <= 0.5
                    and corrected_p90 <= range_max
                    and corrected_p90 >= 5
                ):
                    return divisor

    # Fallback: prefer the smallest divisor that puts p90 below half the range
    half_range = range_max / 2
    for divisor, corrected_p90 in candidates:
        if corrected_p90 <= half_range:
            return divisor

    return candidates[0][0]


def correct_dat_values_df(
    df: pd.DataFrame, machine_slug: str = "", format_hint: str = ""
) -> pd.DataFrame:
    """Auto-correct DAT divisor issues in a DataFrame of sensor values.

    For each numeric column with a known physical range, detects whether
    values appear to be in raw sub-units (cm instead of m, 0.1 bar instead
    of bar, etc.) and divides by the detected divisor.

    Only applies to columns where the sensor is considered calibrated.
    Skips correction if ``format_hint`` is ``"json"`` (JSON values are correct).

    Returns the corrected DataFrame (copy).
    """
    if format_hint == "json":
        return df

    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        if col in ("timestamp",):
            continue
        # Only correct calibrated sensors (uncalibrated have arbitrary scale)
        if machine_slug and not is_calibrated(col, machine_slug):
            continue
        if col not in PHYSICAL_RANGES:
            continue

        values = df[col].dropna().tolist()
        divisor = detect_dat_divisor(values, col)
        if divisor > 1:
            df[col] = df[col] / divisor

    return df


def validate_physical_range(
    series: pd.Series, sensor_name: str, machine_slug: str = ""
) -> pd.Series:
    """NaN-ify values outside the physical plausible range for calibrated sensors.

    Only applies validation if the sensor is calibrated (or no machine_slug given)
    and the sensor has a known physical range. Returns the cleaned series.
    """
    if sensor_name not in PHYSICAL_RANGES:
        return series
    # Only validate calibrated sensors — uncalibrated values have arbitrary scale
    if machine_slug and not is_calibrated(sensor_name, machine_slug):
        return series
    lo, hi = PHYSICAL_RANGES[sensor_name]
    series = series.copy()
    mask = (series < lo) | (series > hi)
    if mask.any():
        series[mask] = np.nan
    return series
