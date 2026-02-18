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
    "Tiefe": (-200, 200),
    "Vorschub Tiefe": (-200, 200),
    "Tiefe Winde 2": (-200, 200),
    "Tiefe_Hauptwinde_GOK": (-200, 200),
    "Tiefe_Bohrrohr_GOK": (-200, 200),
    "Neigung X": (-90, 90),
    "Neigung Y": (-90, 90),
    "Neigung X Mast": (-90, 90),
    "Neigung Y Mast": (-90, 90),
    "Abweichung X": (-5000, 5000),
    "Abweichung Y": (-5000, 5000),
    "Messpunktabw. X": (-50000, 50000),
    "Messpunktabw. Y": (-50000, 50000),
    "Druck Pumpe": (0, 1000),
    "Druck Pumpe 1": (0, 1000),
    "Druck Pumpe 2": (0, 1000),
    "Druck Pumpe 3": (0, 1000),
    "Druck Pumpe 4": (0, 1000),
    "Druck FRL": (0, 500),
    "Druck FRR": (0, 500),
    "KDK Druck": (0, 500),
    "Betondruck": (0, 50),
    "Drehmoment": (0, 120),
    "DrehmomentkNm": (0, 1000),
    "DrehmomentProzent": (0, 120),
    "Seilkraft": (0, 200),
    "Seilkraft Hauptwinde": (0, 1000),
    "Seilkraft Hilfswinde": (0, 200),
    "Seilkraft Fräswinde": (0, 200),
    "Seilkraft Winde 2": (0, 200),
    "Temperatur FRL": (-20, 150),
    "Temperatur FRR": (-20, 150),
    "Temp. Verteilergetriebe": (-20, 200),
    "Oeldruck Getriebe links": (0, 50),
    "Oeldruck Getriebe rechts": (0, 50),
    "Leckagedruck Getriebe links": (0, 20),
    "Leckagedruck Getriebe rechts": (0, 20),
    "Leckoeldruck": (0, 50),
    "Wassergehalt Getriebeoel FRL": (0, 100),
    "Wassergehalt Getriebeoel FRR": (0, 100),
    "Drehzahl": (0, 100),
    "Drehzahl FRL": (0, 100),
    "Drehzahl FRR": (0, 100),
    "KDK Drehzahl": (0, 100),
    "Auflast": (0, 200),
    "Vorschub-KraftPM": (-200, 200),
    "Vorschub-Kraft": (-200, 200),
    "Pitch": (-90, 90),
    "Drehrate": (-360, 360),
    "Verdrehwinkel": (-360, 360),
}

_CALIBRATION_PATH = Path(__file__).parent / "calibration_status.yaml"
_DEFINITIONS_PATH = Path(__file__).parent / "sensor_definitions.yaml"

# Known units for calibrated sensors (sensor_name -> unit string)
SENSOR_UNITS = {
    "Tiefe": "m",
    "Vorschub Tiefe": "m",
    "Tiefe Winde 2": "m",
    "Tiefe_Hauptwinde_GOK": "m",
    "Tiefe_Bohrrohr_GOK": "m",
    "Drehmoment": "%",
    "DrehmomentkNm": "kNm",
    "DrehmomentProzent": "%",
    "Drehmomentstufen": "stage",
    "Drehrichtung": "",
    "Drehrichtung FRL": "",
    "Drehrichtung FRR": "",
    "Druck FRL": "bar",
    "Druck FRR": "bar",
    "Druck Pumpe": "bar",
    "Druck Pumpe 1": "bar",
    "Druck Pumpe 2": "bar",
    "Druck Pumpe 3": "bar",
    "Druck Pumpe 4": "bar",
    "KDK Druck": "bar",
    "Betondruck": "bar",
    "Susp.-Druck": "bar",
    "Susp.-Druck2": "bar",
    "Susp.-Druck unten": "bar",
    "Oeldruck Getriebe links": "bar",
    "Oeldruck Getriebe rechts": "bar",
    "Leckagedruck Getriebe links": "bar",
    "Leckagedruck Getriebe rechts": "bar",
    "Leckoeldruck": "bar",
    "Seilkraft": "t",
    "Seilkraft Hauptwinde": "t",
    "Seilkraft Hilfswinde": "t",
    "Seilkraft Fräswinde": "t",
    "Seilkraft Winde 2": "t",
    "Neigung X": "deg",
    "Neigung Y": "deg",
    "Neigung X Mast": "deg",
    "Neigung Y Mast": "deg",
    "Auflast": "t",
    "Temperatur FRL": "\u00b0C",
    "Temperatur FRR": "\u00b0C",
    "Temp. Verteilergetriebe": "\u00b0C",
    "Wassergehalt Getriebeoel FRL": "%",
    "Wassergehalt Getriebeoel FRR": "%",
    "Abweichung X": "mm",
    "Abweichung Y": "mm",
    "Messpunktabw. X": "mm",
    "Messpunktabw. Y": "mm",
    "Drehzahl": "rpm",
    "Drehzahl FRL": "rpm",
    "Drehzahl FRR": "rpm",
    "KDK Drehzahl": "rpm",
    "Vorschub-KraftPM": "kN",
    "Vorschub-Kraft": "kN",
    "Eindringwiderstand": "kN/m",
    "Susp.-Durchfl.": "l/min",
    "Susp.-Durchfl.2": "l/min",
    "Susp.-Mg": "m\u00b3",
    "Susp.-Mg2": "m\u00b3",
    "Susp.-Mg1+2": "m\u00b3",
    "Menge pro Meter": "l/m",
    "Betondurchfluss": "l/min",
    "Betonmenge": "m\u00b3",
    "Gesamtbetonmenge": "m\u00b3",
    "Durchfluss Pumpe": "l/min",
    "Durchfluss Ruecklauf": "l/min",
    "Durchfluss Vorlauf": "l/min",
    "Vorschubgeschwindigkeit": "mm/s",
    "Strom Kanal 1": "A",
    "Bohrgrenze": "m",
    "Rohrlaenge": "m",
    "Diff_Werkzeug_Bohrrohrunterkante": "m",
    "Pitch": "deg",
    "Drehrate": "deg/s",
    "Verdrehwinkel": "deg",
}


@lru_cache(maxsize=1)
def _load_calibration_yaml() -> dict:
    with open(_CALIBRATION_PATH) as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def _load_sensor_defs() -> dict:
    with open(_DEFINITIONS_PATH) as f:
        return yaml.safe_load(f)


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
    """Get the English canonical name for a German sensor name."""
    defs = _load_sensor_defs()
    entry = defs.get(sensor_name)
    if entry:
        # Convert snake_case to Title Case for display
        canonical = entry["canonical"]
        return canonical.replace("_", " ").title()
    return sensor_name


def get_display_label(sensor_name: str) -> str:
    """Format sensor name as 'English Name (German)' for plot labels."""
    english = get_english_name(sensor_name)
    if english != sensor_name:
        return f"{english} ({sensor_name})"
    return sensor_name


def get_axis_label(sensor_name: str, machine_slug: str) -> str:
    """Get a full axis label: 'English Name (German)\n[unit]'."""
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


def detect_dat_divisor(values: list[float | None], sensor_name: str) -> int:
    """Detect the likely correct divisor for a DAT-parsed sensor based on physical ranges.

    The DAT format header often declares ``divisor=1`` for sensors whose raw
    integer values are actually in sub-units (centimeters, decibar, etc.).
    This function checks whether the parsed values exceed the expected physical
    range and, if so, returns the divisor (10, 100, or 1000) that brings them
    well within range.

    Uses two signals:
    1. **Magnitude**: the 90th-percentile absolute value vs the physical range
    2. **Step size**: for integer-valued data, the median step between consecutive
       values. A step of 2+ in a sensor with typical sub-unit resolution (like Tiefe
       in meters) suggests the values are in sub-units.

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
