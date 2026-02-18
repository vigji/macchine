"""Sensor name harmonization: German → English, unit normalization."""

from __future__ import annotations

from pathlib import Path

import yaml

_SENSOR_MAP: dict[str, dict] | None = None
_DEFINITIONS_PATH = Path(__file__).parent / "sensor_definitions.yaml"

# Unit normalization table
UNIT_MAP = {
    "U/min": "rpm",
    "Grad": "deg",
    "°": "deg",
    "°C": "degC",
    "cbm/h": "m3_per_h",
    "m³/h": "m3_per_h",
    "cbm": "m3",
    "m³": "m3",
    "l/min": "L_per_min",
    "l/m": "L_per_m",
    "cm/min": "cm_per_min",
    "mm/U": "mm_per_rev",
    "K=1;M=2;G=3": "stage",
    "R; ;L": "direction",
    "L/R": "direction",
    "Nr.": "number",
}


def _load_sensor_map() -> dict[str, dict]:
    global _SENSOR_MAP
    if _SENSOR_MAP is not None:
        return _SENSOR_MAP

    with open(_DEFINITIONS_PATH) as f:
        _SENSOR_MAP = yaml.safe_load(f)
    return _SENSOR_MAP


def get_canonical_name(sensor_name: str) -> str:
    """Get the English canonical name for a German sensor name."""
    sensor_map = _load_sensor_map()
    entry = sensor_map.get(sensor_name)
    if entry:
        return entry["canonical"]
    return sensor_name


def get_category(sensor_name: str) -> str:
    """Get the sensor category."""
    sensor_map = _load_sensor_map()
    entry = sensor_map.get(sensor_name)
    if entry:
        return entry.get("category", "unknown")
    return "unknown"


def normalize_unit(unit: str) -> str:
    """Normalize a unit string to a standard form."""
    return UNIT_MAP.get(unit, unit)


def get_all_mappings() -> dict[str, str]:
    """Return all German → English mappings."""
    sensor_map = _load_sensor_map()
    return {name: info["canonical"] for name, info in sensor_map.items()}


def get_categories() -> dict[str, list[str]]:
    """Return sensors grouped by category."""
    sensor_map = _load_sensor_map()
    categories: dict[str, list[str]] = {}
    for name, info in sensor_map.items():
        cat = info.get("category", "unknown")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(info["canonical"])
    return categories
