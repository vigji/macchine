"""Technique code mapping."""

from __future__ import annotations

TECHNIQUE_MAP = {
    "SOB": "Continuous Flight Auger",
    "KELLY": "Kelly Drilling",
    "CUT": "Diaphragm Wall Cutter",
    "SCM": "Soil Cement Mixing",
    "GRAB": "Grab",
    "DMS": "Deep Mixing",
    "FREE": "Free Recording",
}


def get_technique_name(code: str) -> str:
    """Get the full English name for a technique code."""
    return TECHNIQUE_MAP.get(code.upper(), code)


def get_all_techniques() -> dict[str, str]:
    """Return all technique code → name mappings."""
    return dict(TECHNIQUE_MAP)
