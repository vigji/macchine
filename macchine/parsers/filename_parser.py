"""Extract metadata from file paths and filenames."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

# Model alias table: normalize model codes found in filenames
MODEL_ALIASES: dict[str, str] = {
    "bc5x": "cube0",
    "c_gb50": "gb50",
}

# Known serial → (model, number) mapping, populated at runtime
_SERIAL_REGISTRY: dict[str, tuple[str, int]] = {}

# Regex for machine directory: "BG-33-V #5610 | 01K00044171" or "MC-86-null 621 | 01K00046811"
_MACHINE_DIR_RE = re.compile(
    r"^(?P<model>[A-Za-z0-9\-]+?)(?:-null)?\s+(?:#?)(?P<number>\d+)\s*\|\s*(?P<serial>\w+)$"
)

# Regex for site directory: "2026-02-16_1508 - VICENZA" or "2026-02-16_1427TriesteFerrier"
_SITE_DIR_RE = re.compile(
    r"^(?:\d{4}-\d{2}-\d{2}_)"  # date prefix
    r"(?P<site_id>\d+)"  # numeric site ID
    r"(?:\s*-?\s*(?P<name>.+))?$"  # optional name
)

# Fallback for non-numeric site IDs: "2026-02-16_CS-Antwerpen", "2026-02-16_LignanoSabbiadoro"
_SITE_DIR_NONNUMERIC_RE = re.compile(
    r"^(?:\d{4}-\d{2}-\d{2}_)(?P<site_id>.+)$"
)

# JSON filename: "{element_name}_{YYYYMMDDHHMI}.json"
_JSON_FILENAME_RE = re.compile(
    r"^(?P<element>.+?)_(?P<date>\d{12})\.json$"
)

# DAT filename pattern A (with serial): "{serial}_{model}_{number}_{YYYYMMDD}_{HHMMSS}_{seqnum}_{element}.dat"
_DAT_FILENAME_A_RE = re.compile(
    r"^(?P<serial>[A-Za-z0-9]+)_(?P<model>[a-z0-9]+)_(?P<number>\d+)_"
    r"(?P<date>\d{8})_(?P<time>\d{6})_(?P<seq>\d+)_(?P<element>.+?)\.dat$",
    re.IGNORECASE,
)

# DAT filename pattern B (without serial): "{model}_{number}_{YYYYMMDD}_{HHMMSS}_{seqnum}_{element}_.dat"
_DAT_FILENAME_B_RE = re.compile(
    r"^(?P<model>[a-z][a-z0-9]+)_(?P<number>\d+)_"
    r"(?P<date>\d{8})_(?P<time>\d{6})_(?P<seq>\d+)_(?P<element>.+?)_?\.dat$",
    re.IGNORECASE,
)


def register_serial(serial: str, model: str, number: int) -> None:
    """Register a known serial → (model, number) mapping."""
    _SERIAL_REGISTRY[serial] = (model, number)


def normalize_model(model_code: str) -> str:
    """Normalize a model code from a filename."""
    lower = model_code.lower().replace("-", "")
    return MODEL_ALIASES.get(lower, lower)


def parse_site_dir(dirname: str) -> dict:
    """Parse a site directory name into site_id and name.

    Examples:
        "2026-02-16_1508 - VICENZA" → {"site_id": "1508", "name": "VICENZA"}
        "2026-02-16_1427TriesteFerrier" → {"site_id": "1427", "name": "TriesteFerrier"}
        "2026-02-16_CS-Antwerpen" → {"site_id": "CS-Antwerpen", "name": "Antwerpen"}
    """
    m = _SITE_DIR_RE.match(dirname)
    if m:
        name = (m.group("name") or "").strip().strip("-").strip()
        return {"site_id": m.group("site_id"), "name": name}

    m = _SITE_DIR_NONNUMERIC_RE.match(dirname)
    if m:
        site_id = m.group("site_id").strip()
        return {"site_id": site_id, "name": site_id}

    return {"site_id": "", "name": dirname}


def parse_machine_dir(dirname: str) -> dict:
    """Parse a machine directory name.

    Examples:
        "BG-33-V #5610 | 01K00044171" → {"machine_model": "BG-33-V", "machine_number": 5610, "machine_serial": "01K00044171"}
        "Unidentified" → {}
        "01K00033511" → {"machine_serial": "01K00033511"}
    """
    if dirname == "Unidentified" or dirname.startswith("2026-"):
        return {}

    m = _MACHINE_DIR_RE.match(dirname)
    if m:
        model = m.group("model")
        number = int(m.group("number"))
        serial = m.group("serial")
        slug = f"{normalize_model(model)}_{number}"
        register_serial(serial, normalize_model(model), number)
        return {
            "machine_model": model,
            "machine_number": number,
            "machine_serial": serial,
            "machine_slug": slug,
        }

    # Bare serial ID directory (e.g. "01K00033511")
    if re.match(r"^[A-Za-z0-9]+$", dirname) and len(dirname) > 5:
        result = {"machine_serial": dirname}
        if dirname in _SERIAL_REGISTRY:
            model, number = _SERIAL_REGISTRY[dirname]
            result["machine_model"] = model
            result["machine_number"] = number
            result["machine_slug"] = f"{model}_{number}"
        return result

    return {}


def parse_json_filename(filename: str) -> dict:
    """Parse a JSON filename for element name and timestamp.

    Example: "palo 96 dms_202407180938.json" → {"element_name": "palo 96 dms", "timestamp": datetime(...)}
    """
    m = _JSON_FILENAME_RE.match(filename)
    if not m:
        return {"element_name": Path(filename).stem}

    element = m.group("element")
    date_str = m.group("date")
    try:
        ts = datetime.strptime(date_str, "%Y%m%d%H%M")
    except ValueError:
        ts = None

    return {"element_name": element, "filename_timestamp": ts}


def parse_dat_filename(filename: str) -> dict:
    """Parse a DAT filename for machine info, timestamp, and element.

    Pattern A: "01K00044171_bg33v_5610_20251022_135857_00003177_J40.dat"
    Pattern B: "bg33v_5610_20240613_140339_00002593_palo_167_dms_.dat"
    """
    # Try pattern A (with serial) first
    m = _DAT_FILENAME_A_RE.match(filename)
    if m:
        model = normalize_model(m.group("model"))
        number = int(m.group("number"))
        serial = m.group("serial")
        register_serial(serial, model, number)
        element = m.group("element").strip("_").replace("_", " ")
        try:
            ts = datetime.strptime(f"{m.group('date')}_{m.group('time')}", "%Y%m%d_%H%M%S")
        except ValueError:
            ts = None
        return {
            "machine_serial": serial,
            "machine_model": model,
            "machine_number": number,
            "machine_slug": f"{model}_{number}",
            "element_name": element,
            "filename_timestamp": ts,
            "sequence_number": int(m.group("seq")),
        }

    # Try pattern B (without serial)
    m = _DAT_FILENAME_B_RE.match(filename)
    if m:
        model = normalize_model(m.group("model"))
        number = int(m.group("number"))
        element = m.group("element").strip("_").replace("_", " ")
        try:
            ts = datetime.strptime(f"{m.group('date')}_{m.group('time')}", "%Y%m%d_%H%M%S")
        except ValueError:
            ts = None
        result = {
            "machine_model": model,
            "machine_number": number,
            "machine_slug": f"{model}_{number}",
            "element_name": element,
            "filename_timestamp": ts,
            "sequence_number": int(m.group("seq")),
        }
        # Try to resolve serial from registry
        for serial, (reg_model, reg_number) in _SERIAL_REGISTRY.items():
            if reg_model == model and reg_number == number:
                result["machine_serial"] = serial
                break
        return result

    return {"element_name": Path(filename).stem}


def parse_json_path(path: Path) -> dict:
    """Parse a full JSON file path to extract site, machine, and element info.

    Expected structure: {raw_dir}/{site_dir}/{machine_dir}/{filename}.json
    Or for Pisa-like: {raw_dir}/{site_dir}/{sub_dir}/{filename}.json (no machine in dir)
    Or flat: {raw_dir}/{site_dir}/{filename}.json
    """
    parts = path.parts
    result = parse_json_filename(path.name)

    # Walk up from file to find site and machine directories
    # The file is in parts[-1], parent in parts[-2], grandparent in parts[-3], etc.
    if len(parts) >= 3:
        parent = parts[-2]
        grandparent = parts[-3] if len(parts) >= 3 else ""

        # Check if parent is a machine directory
        machine_info = parse_machine_dir(parent)
        if machine_info:
            result.update(machine_info)
            # Grandparent should be the site
            site_info = parse_site_dir(grandparent)
            result["site_id"] = site_info.get("site_id", "")
            result["site_name"] = site_info.get("name", "")
        else:
            # Parent might be a sub-site dir (Pisa) or the site itself
            site_info = parse_site_dir(parent)
            if site_info.get("site_id"):
                result["site_id"] = site_info["site_id"]
                result["site_name"] = site_info.get("name", "")
            else:
                # Try grandparent as site
                site_info = parse_site_dir(grandparent)
                result["site_id"] = site_info.get("site_id", "")
                result["site_name"] = site_info.get("name", "")

    return result


def parse_dat_path(path: Path) -> dict:
    """Parse a full DAT file path to extract site, machine, and element info."""
    parts = path.parts
    result = parse_dat_filename(path.name)

    if len(parts) >= 3:
        parent = parts[-2]
        grandparent = parts[-3] if len(parts) >= 3 else ""

        machine_info = parse_machine_dir(parent)
        if machine_info:
            # Merge, preferring filename-derived machine info if present
            for k, v in machine_info.items():
                if k not in result or not result[k]:
                    result[k] = v
            site_info = parse_site_dir(grandparent)
            result["site_id"] = site_info.get("site_id", "")
            result["site_name"] = site_info.get("name", "")
        else:
            site_info = parse_site_dir(parent)
            if site_info.get("site_id"):
                result["site_id"] = site_info["site_id"]
                result["site_name"] = site_info.get("name", "")
            else:
                site_info = parse_site_dir(grandparent)
                result["site_id"] = site_info.get("site_id", "")
                result["site_name"] = site_info.get("name", "")

    return result
