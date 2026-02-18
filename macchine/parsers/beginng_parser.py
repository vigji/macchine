"""Parser for the BEGINNG section of DAT files (153-char fixed-width)."""

from __future__ import annotations

import re
from datetime import datetime


def parse_beginng(section: str) -> dict:
    """Parse a 153-char BEGINNG section.

    Layout (empirically determined):
        [0:7]    "BEGINN" + variant char (A, 1, 9, etc.)
        [33:48]  Element name (left-padded with spaces, right-padded to 15 chars)
        [48:60]  Timestamp DDMMYYHHMMSS
        [80:100] "Bauer" + technique string (e.g. "BauerSOB", "BauerFIRE 1470")
        [123:153] Machine code (right-aligned, e.g. "BG33V_5610", "BC5X_482")
    """
    result = {}

    if len(section) < 153:
        return result

    # Variant
    result["variant"] = section[6:7]

    # Element name (around positions 33-47, strip padding)
    element_raw = section[33:48].strip()
    if element_raw and element_raw != "0":
        result["element_name"] = element_raw

    # Timestamp: DDMMYYHHMMSS
    ts_raw = section[48:60].strip()
    if len(ts_raw) == 12:
        try:
            result["timestamp"] = datetime.strptime(ts_raw, "%d%m%y%H%M%S")
        except ValueError:
            pass

    # Technique: extract from "BauerXXX" pattern
    tech_area = section[80:100].strip()
    if tech_area.startswith("Bauer"):
        tech_str = tech_area[5:].strip()
        # Map technique strings to codes
        result["technique_raw"] = tech_str
        result["technique"] = _map_technique(tech_str)

    # Machine code at end (right-aligned)
    machine_raw = section[123:153].strip()
    if machine_raw:
        result["machine_code"] = machine_raw
        machine_info = _parse_machine_code(machine_raw)
        result.update(machine_info)

    return result


def _map_technique(tech_str: str) -> str:
    """Map BEGINNG technique strings to standard codes."""
    upper = tech_str.upper()
    if "SOB" in upper or "CFA" in upper:
        return "SOB"
    if "KELLY" in upper:
        return "KELLY"
    if "CUT" in upper or "SCHLITZ" in upper:
        return "CUT"
    if "FIRE" in upper:
        # FIRE can be SCM, KELLY, or SOB depending on context
        # The FIRE/BauerFIRE string appears for multiple techniques
        return "KELLY"  # default — will be refined by header content
    if "GRAB" in upper or "GREIFER" in upper:
        return "GRAB"
    if "DMS" in upper:
        return "DMS"
    if "SCM" in upper:
        return "SCM"
    if "FREE" in upper:
        return "FREE"
    return tech_str


def _parse_machine_code(code: str) -> dict:
    """Parse machine code like 'BG33V_5610' or 'BC5X_482'."""
    m = re.match(r"^([A-Za-z0-9]+)_(\d+)$", code)
    if m:
        model = m.group(1).lower()
        number = int(m.group(2))
        return {"machine_model_code": model, "machine_number": number}
    return {}
