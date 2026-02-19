"""Investigate whether MEDEF spec-driven parsing can replace empirical corrections.

Goes back to raw DAT files, parses from scratch using spec info, and compares
against JSON ground truth to determine which approach produces correct values.

Four approaches tested per sensor:
  A (header only):   raw_int / header_divisor
  B (spec divisor):  raw_int / spec_expected_divisor
  C (fully spec):    raw_int_spec_layout / spec_expected_divisor  (if layout matches)
  D (header+corr):   raw_int / header_divisor / empirical_correction

Usage:
    uv run python scripts/investigate_spec_parsing.py
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from macchine.parsers.dat_parser import (
    DatFieldDef,
    _estimate_field_widths,
    _parse_header,
)
from macchine.parsers.beginng_parser import parse_beginng

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SPECS_PATH = ROOT / "specs_data" / "medef_specs.yaml"
CORRECTIONS_PATH = ROOT / "macchine" / "harmonize" / "dat_corrections.yaml"
RAW_DIR = ROOT / "output" / "raw_downloads"
OUTPUT_DIR = ROOT / "output" / "diagnostics"

# Test cases: each entry is (project_dir_pattern, machine_slug, description)
# A machine can appear in multiple projects with different techniques.
TEST_CASES = [
    ("LignanoSabbiadoro", "bg33v_5610", "KELLY DAT + KELLY JSON"),
    ("VICENZA", "bg33v_5610", "CSM/DMS DAT + SCM JSON + KELLY JSON"),
    ("Catania", "gb50_601", "Seilgreifer DAT + GRAB JSON"),
]

# Map JSON technique names to spec technique prefixes
JSON_TECHNIQUE_TO_SPEC = {
    "KELLY": "KELLY",
    "SOB": "SOB",
    "GRAB": "Seilgreifer",
    "CUT": "CUT",
    "SCM": "CSM",
    "CSM": "CSM",
    "FDP": "FDP",
}

# Unicode ligatures from PDF parsing that may appear in spec sensor names
LIGATURE_MAP = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
}


# ── Data classes ───────────────────────────────────────────────────────────────
@dataclass
class SensorResult:
    sensor_name: str
    unit: str
    machine_slug: str
    technique: str
    in_spec: bool
    spec_key: str | None
    spec_chars: int | None
    spec_decimal: int | None
    spec_divisor: float | None
    spec_signed: bool | None
    header_divisor: int
    header_meta2: int
    header_width: int
    correction: float
    raw_median: float | None
    raw_median_spec_layout: float | None
    json_median: float | None
    approach_a: float | None
    approach_b: float | None
    approach_c: float | None
    approach_d: float | None
    error_a: float | None
    error_b: float | None
    error_c: float | None
    error_d: float | None
    winner: str


# ── Loading ────────────────────────────────────────────────────────────────────
def load_medef_specs() -> dict:
    with open(SPECS_PATH) as f:
        return yaml.safe_load(f)


def load_corrections() -> dict[str, dict[str, float]]:
    with open(CORRECTIONS_PATH) as f:
        data = yaml.safe_load(f)
    return data or {}


def normalize_spec_name(name: str) -> str:
    """Replace unicode ligatures in spec sensor names."""
    for lig, replacement in LIGATURE_MAP.items():
        name = name.replace(lig, replacement)
    return name


# ── File discovery ─────────────────────────────────────────────────────────────
def find_project_dir(pattern: str) -> Path | None:
    for d in sorted(RAW_DIR.iterdir()):
        if d.is_dir() and pattern in d.name:
            return d
    return None


def find_dat_json_files(project_dir: Path) -> tuple[list[Path], list[Path]]:
    dat_files = sorted(project_dir.rglob("*.dat"))
    json_files = sorted(project_dir.rglob("*.json"))
    return dat_files, json_files


# ── DAT parsing (raw integers) ────────────────────────────────────────────────
def parse_dat_raw(path: Path) -> dict | None:
    """Parse a DAT file and return raw (pre-divisor) information.

    Returns dict with field_defs, technique info, widths, raw_medians, etc.
    """
    try:
        with open(path, "rb") as f:
            raw = f.read()
        text = raw.decode("iso-8859-1")
    except Exception:
        return None

    sections = text.split("$")
    if len(sections) < 2:
        return None

    field_defs = _parse_header(sections[0])
    if not field_defs:
        return None

    beginng_info = {}
    for section in sections[1:5]:
        if section.startswith("BEGINN"):
            beginng_info = parse_beginng(section)
            break

    dat_sections = [s for s in sections if s.startswith("DAT")]
    if not dat_sections:
        return None

    data_length = len(dat_sections[0]) - 3

    widths = _estimate_field_widths(field_defs, data_length)
    if not widths:
        return None

    # Also infer technique from sensor names (more reliable than BEGINNG for
    # machines like bg33v_5610 that report "FIRE" for multiple techniques)
    sensor_names = {f.name.lower() for f in field_defs}
    inferred_technique = ""
    if any("fräs" in n or "fraes" in n or "dws" in n for n in sensor_names):
        inferred_technique = "CUT"
    elif any("susp" in n for n in sensor_names):
        inferred_technique = "SCM"
    elif "betondruck" in sensor_names or "betonmenge" in sensor_names:
        if any("seilkraft hauptwinde" in n for n in sensor_names):
            inferred_technique = "KELLY"
        else:
            inferred_technique = "SOB"
    elif any("seilkraft hauptwinde" in n for n in sensor_names):
        # KELLY without concrete sensors (e.g. LignanoSabbiadoro)
        inferred_technique = "KELLY"
    elif any("seilkraft" in n for n in sensor_names):
        inferred_technique = "GRAB"

    # Use inferred technique if available, otherwise BEGINNG
    technique = inferred_technique or beginng_info.get("technique", "")

    # Extract raw integers
    num_fields = len(field_defs)
    all_raw: list[list[int | None]] = [[] for _ in range(num_fields)]

    for dat_section in dat_sections:
        data = dat_section[3:]
        if len(data) != data_length:
            continue

        pos = 0
        for i, (fdef, w) in enumerate(zip(field_defs, widths)):
            if pos + w > len(data):
                break
            raw_str = data[pos : pos + w]
            pos += w

            if fdef.is_direction:
                all_raw[i].append(None)
            else:
                try:
                    all_raw[i].append(int(raw_str))
                except ValueError:
                    all_raw[i].append(None)

    raw_medians = {}
    for i, fdef in enumerate(field_defs):
        values = [v for v in all_raw[i] if v is not None and v != 0]
        if values:
            raw_medians[fdef.name] = statistics.median(values)

    return {
        "path": path,
        "field_defs": field_defs,
        "technique_beginng": beginng_info.get("technique", ""),
        "technique_raw": beginng_info.get("technique_raw", ""),
        "technique_inferred": inferred_technique,
        "technique": technique,
        "machine_code": beginng_info.get("machine_code", ""),
        "widths": widths,
        "data_length": data_length,
        "raw_medians": raw_medians,
        "num_dat_sections": len(dat_sections),
    }


def compute_spec_widths(
    field_defs: list[DatFieldDef], specs: dict, spec_key: str
) -> list[int] | None:
    """Compute field widths using spec characters instead of header meta[2]."""
    if spec_key not in specs:
        return None

    spec_sensors = specs[spec_key].get("sensors", {})
    spec_lookup = {}
    for name, info in spec_sensors.items():
        spec_lookup[normalize_spec_name(name)] = info

    widths = []
    for fdef in field_defs:
        if fdef.is_direction:
            widths.append(1)
            continue

        spec = spec_lookup.get(fdef.name)
        if spec is None:
            return None

        chars = spec["characters"]
        if spec.get("signed", False):
            chars += 1
        widths.append(chars)

    return widths


def parse_dat_with_spec_widths(
    path: Path, spec_widths: list[int], field_defs: list[DatFieldDef]
) -> dict[str, float]:
    """Re-parse a DAT file using spec-derived field widths."""
    with open(path, "rb") as f:
        raw = f.read()
    text = raw.decode("iso-8859-1")
    sections = text.split("$")

    dat_sections = [s for s in sections if s.startswith("DAT")]
    if not dat_sections:
        return {}

    data_length = len(dat_sections[0]) - 3
    if sum(spec_widths) != data_length:
        return {}

    num_fields = len(field_defs)
    all_raw: list[list[int | None]] = [[] for _ in range(num_fields)]

    for dat_section in dat_sections:
        data = dat_section[3:]
        if len(data) != data_length:
            continue

        pos = 0
        for i, (fdef, w) in enumerate(zip(field_defs, spec_widths)):
            if pos + w > len(data):
                break
            raw_str = data[pos : pos + w]
            pos += w

            if fdef.is_direction:
                all_raw[i].append(None)
            else:
                try:
                    all_raw[i].append(int(raw_str))
                except ValueError:
                    all_raw[i].append(None)

    raw_medians = {}
    for i, fdef in enumerate(field_defs):
        values = [v for v in all_raw[i] if v is not None and v != 0]
        if values:
            raw_medians[fdef.name] = statistics.median(values)

    return raw_medians


# ── JSON parsing (ground truth) ───────────────────────────────────────────────
def parse_json_files_by_technique(
    json_files: list[Path], max_per_technique: int = 20
) -> dict[str, dict]:
    """Parse JSON files, grouped by technique.

    Returns dict[technique -> {sensor_medians, sensor_units, medef_versions}].
    """
    # First pass: group files by technique
    files_by_technique: dict[str, list[Path]] = defaultdict(list)
    for jpath in json_files:
        try:
            with open(jpath) as f:
                d = json.load(f)
            technique = d.get("technique", "UNKNOWN")
            files_by_technique[technique].append(jpath)
        except Exception:
            continue

    result = {}
    for technique, paths in files_by_technique.items():
        sensor_values: dict[str, list[float]] = defaultdict(list)
        sensor_units: dict[str, str] = {}
        medef_versions: set[int] = set()

        for jpath in paths[:max_per_technique]:
            try:
                with open(jpath) as f:
                    d = json.load(f)
            except Exception:
                continue

            version = d.get("medefVersion")
            if version is not None:
                medef_versions.add(version)

            tsb = d.get("timeSeriesBlock", {})
            svf = tsb.get("serialValuesFree", [])

            for item in svf:
                name = item.get("seriesName", "")
                if not name:
                    continue
                unit = item.get("unitOfMeasurement", "")
                sensor_units[name] = unit

                values = item.get("values", [])
                nonzero = [v for v in values if isinstance(v, (int, float)) and v != 0.0]
                if nonzero:
                    sensor_values[name].append(statistics.median(nonzero))

        sensor_medians = {}
        for name, file_medians in sensor_values.items():
            sensor_medians[name] = statistics.median(file_medians)

        result[technique] = {
            "sensor_medians": sensor_medians,
            "sensor_units": sensor_units,
            "medef_versions": medef_versions,
            "num_files": len(paths),
            "num_used": min(len(paths), max_per_technique),
        }

    return result


# ── Spec matching ─────────────────────────────────────────────────────────────
def find_best_spec_key(
    technique: str,
    medef_version: int | None,
    machine_slug: str,
    specs: dict,
) -> str | None:
    """Find the best spec key for a given technique and version."""
    spec_prefix = JSON_TECHNIQUE_TO_SPEC.get(technique)
    if spec_prefix is None:
        return None

    # Try exact version match
    if medef_version is not None:
        key = f"{spec_prefix}_v{medef_version}"
        if key in specs:
            return key

    # Try from machine_technique_map
    machine_map = specs.get("machine_technique_map", {})
    mapped_keys = machine_map.get(machine_slug, [])
    for key in mapped_keys:
        if key.startswith(spec_prefix):
            return key

    # Try any version
    for key in specs:
        if key == "machine_technique_map":
            continue
        if key.startswith(spec_prefix + "_"):
            return key

    return None


def get_spec_for_sensor(
    sensor_name: str, spec_keys: list[str], specs: dict
) -> tuple[dict | None, str | None]:
    """Look up a sensor across multiple spec keys, handling ligature normalization.

    Returns (spec_info, spec_key) or (None, None).
    """
    for spec_key in spec_keys:
        if spec_key not in specs:
            continue
        spec_sensors = specs[spec_key].get("sensors", {})

        # Direct match
        if sensor_name in spec_sensors:
            return spec_sensors[sensor_name], spec_key

        # Normalized match
        for spec_name, info in spec_sensors.items():
            if normalize_spec_name(spec_name) == sensor_name:
                return info, spec_key

    return None, None


# ── Error metric ──────────────────────────────────────────────────────────────
def log_error(approach_val: float | None, json_val: float | None) -> float | None:
    """Compute |log10(|approach| / |json|)|. None if either is zero/None."""
    if approach_val is None or json_val is None:
        return None
    a = abs(approach_val)
    j = abs(json_val)
    if a == 0 or j == 0:
        return None
    try:
        return abs(math.log10(a / j))
    except (ValueError, ZeroDivisionError):
        return None


# ── Main investigation per technique group ─────────────────────────────────────
def investigate_technique_group(
    machine_slug: str,
    dat_technique: str,
    json_technique: str,
    dat_parsed_list: list[dict],
    json_ground_truth: dict,
    specs: dict,
    corrections: dict,
) -> tuple[list[SensorResult], list[str]]:
    """Compare DAT raw values against JSON for one technique group."""
    lines: list[str] = []
    results: list[SensorResult] = []

    machine_corrections = corrections.get(machine_slug, {})

    json_medians = json_ground_truth["sensor_medians"]
    json_units = json_ground_truth["sensor_units"]
    json_versions = json_ground_truth["medef_versions"]

    # Determine best spec key
    primary_version = max(json_versions) if json_versions else None
    spec_key = find_best_spec_key(json_technique, primary_version, machine_slug, specs)

    # Also collect all mapped spec keys for this machine
    machine_map = specs.get("machine_technique_map", {})
    all_spec_keys = machine_map.get(machine_slug, [])
    # Prioritize the best-match key
    spec_keys_to_try = []
    if spec_key:
        spec_keys_to_try.append(spec_key)
    for k in all_spec_keys:
        if k not in spec_keys_to_try:
            spec_keys_to_try.append(k)

    lines.append(f"  Technique: DAT={dat_technique}, JSON={json_technique}")
    lines.append(f"  Matched spec key: {spec_key or 'NONE'}")
    lines.append(f"  All spec keys tried: {spec_keys_to_try}")
    lines.append(f"  JSON MEDEF versions: {json_versions}")
    lines.append(f"  DAT files used: {len(dat_parsed_list)}")
    lines.append(f"  JSON ground truth sensors: {len(json_medians)}")
    lines.append("")

    # Aggregate raw medians across DAT files
    sensor_raw_per_file: dict[str, list[float]] = defaultdict(list)
    for parsed in dat_parsed_list:
        for name, val in parsed["raw_medians"].items():
            sensor_raw_per_file[name].append(val)

    raw_medians: dict[str, float] = {}
    for name, vals in sensor_raw_per_file.items():
        raw_medians[name] = statistics.median(vals)

    # Reference parsed file for field defs and widths
    ref_parsed = dat_parsed_list[0]
    field_defs = ref_parsed["field_defs"]
    widths = ref_parsed["widths"]
    data_length = ref_parsed["data_length"]

    # ── Field width comparison ──
    lines.append(f"  {'Sensor':<35s} {'meta[2]':>7s} {'hdr_w':>5s} "
                 f"{'spec_ch':>7s} {'spec_w':>6s} {'match':>5s}")
    lines.append("  " + "-" * 70)

    spec_widths_feasible = True
    spec_widths: list[int] = []

    for i, fdef in enumerate(field_defs):
        meta2 = fdef.meta[2] if len(fdef.meta) > 2 else "?"
        hdr_w = widths[i]

        spec_info, _ = get_spec_for_sensor(fdef.name, spec_keys_to_try, specs)
        if spec_info:
            sp_ch = spec_info["characters"]
            sp_signed = spec_info.get("signed", False)
            sp_w = sp_ch + (1 if sp_signed else 0)
            if fdef.is_direction:
                sp_w = 1
            spec_widths.append(sp_w)
            match = "YES" if sp_w == hdr_w else "NO"
        else:
            sp_ch = "-"
            sp_w = "-"
            match = "n/a"
            spec_widths_feasible = False
            spec_widths.append(hdr_w)

        lines.append(f"    {fdef.name:<33s} {str(meta2):>7s} {hdr_w:>5d} "
                     f"{str(sp_ch):>7s} {str(sp_w):>6s} {match:>5s}")

    spec_width_total = sum(spec_widths) if spec_widths_feasible else None
    lines.append("  " + "-" * 70)
    lines.append(f"  Header width total: {sum(widths)} vs data_length: {data_length}")
    if spec_width_total is not None:
        lines.append(f"  Spec width total:   {spec_width_total} vs data_length: {data_length}")
        if spec_width_total == data_length:
            lines.append("    -> Spec layout IS feasible (approach C possible)")
        else:
            lines.append(f"    -> Spec layout MISMATCH (off by {spec_width_total - data_length})")
            spec_widths_feasible = False
    else:
        lines.append("  Spec width total:   N/A (not all sensors in spec)")
        spec_widths_feasible = False
    lines.append("")

    # ── Try approach C if feasible ──
    raw_medians_spec_layout: dict[str, float] = {}
    if spec_widths_feasible and spec_width_total == data_length:
        spec_raw_per_file: dict[str, list[float]] = defaultdict(list)
        for parsed in dat_parsed_list[:10]:
            spec_raw = parse_dat_with_spec_widths(parsed["path"], spec_widths, field_defs)
            for name, val in spec_raw.items():
                spec_raw_per_file[name].append(val)
        for name, vals in spec_raw_per_file.items():
            raw_medians_spec_layout[name] = statistics.median(vals)

    # ── Sensor-by-sensor comparison ──
    common_sensors = set(raw_medians.keys()) & set(json_medians.keys())
    all_dat_sensors = set(raw_medians.keys())
    all_json_sensors = set(json_medians.keys())

    lines.append(f"  DAT sensors: {len(all_dat_sensors)}")
    lines.append(f"  JSON sensors: {len(all_json_sensors)}")
    lines.append(f"  Common (exact name match): {len(common_sensors)}")
    dat_only = sorted(all_dat_sensors - all_json_sensors)
    json_only = sorted(all_json_sensors - all_dat_sensors)
    if dat_only:
        lines.append(f"  DAT-only: {dat_only}")
    if json_only:
        lines.append(f"  JSON-only: {json_only}")
    lines.append("")

    for sensor_name in sorted(common_sensors):
        raw_val = raw_medians.get(sensor_name)
        json_val = json_medians.get(sensor_name)
        json_unit = json_units.get(sensor_name, "?")

        if raw_val is None or json_val is None:
            continue

        # Find field def
        fdef = None
        fdef_idx = None
        for idx, fd in enumerate(field_defs):
            if fd.name == sensor_name:
                fdef = fd
                fdef_idx = idx
                break

        if fdef is None or fdef.is_direction:
            continue

        header_div = fdef.divisor
        meta2 = fdef.meta[2] if len(fdef.meta) > 2 else 0
        hdr_w = widths[fdef_idx] if fdef_idx is not None else 0
        correction = machine_corrections.get(sensor_name, 1.0)

        # Get spec info
        spec_info, found_spec_key = get_spec_for_sensor(
            sensor_name, spec_keys_to_try, specs
        )

        in_spec = spec_info is not None
        spec_div = spec_info["expected_divisor"] if spec_info else None
        spec_chars = spec_info["characters"] if spec_info else None
        spec_decimal = spec_info.get("decimal") if spec_info else None
        spec_signed = spec_info.get("signed") if spec_info else None

        # ── Compute 4 approaches ──
        approach_a = raw_val / header_div if header_div != 0 else None
        approach_b = raw_val / spec_div if spec_div and spec_div != 0 else None

        raw_spec = raw_medians_spec_layout.get(sensor_name)
        approach_c = raw_spec / spec_div if raw_spec and spec_div and spec_div != 0 else None

        if header_div != 0 and correction != 0:
            approach_d = raw_val / header_div / correction
        else:
            approach_d = None

        # ── Errors ──
        error_a = log_error(approach_a, json_val)
        error_b = log_error(approach_b, json_val)
        error_c = log_error(approach_c, json_val)
        error_d = log_error(approach_d, json_val)

        # ── Winner ──
        candidates = {}
        if error_a is not None:
            candidates["A"] = error_a
        if error_b is not None:
            candidates["B"] = error_b
        if error_c is not None:
            candidates["C"] = error_c
        if error_d is not None:
            candidates["D"] = error_d

        if candidates:
            best_error = min(candidates.values())
            best_approaches = [k for k, v in candidates.items() if abs(v - best_error) < 0.01]
            winner = "+".join(sorted(best_approaches))
        else:
            winner = "none"

        result = SensorResult(
            sensor_name=sensor_name,
            unit=json_unit,
            machine_slug=machine_slug,
            technique=json_technique,
            in_spec=in_spec,
            spec_key=found_spec_key,
            spec_chars=spec_chars,
            spec_decimal=spec_decimal,
            spec_divisor=spec_div,
            spec_signed=spec_signed,
            header_divisor=header_div,
            header_meta2=meta2,
            header_width=hdr_w,
            correction=correction,
            raw_median=raw_val,
            raw_median_spec_layout=raw_spec,
            json_median=json_val,
            approach_a=approach_a,
            approach_b=approach_b,
            approach_c=approach_c,
            approach_d=approach_d,
            error_a=error_a,
            error_b=error_b,
            error_c=error_c,
            error_d=error_d,
            winner=winner,
        )
        results.append(result)

        # ── Format ──
        lines.append(f"  Sensor: {sensor_name} [{json_unit}]")
        lines.append(f"    Raw median: {raw_val}")
        if raw_spec is not None and raw_spec != raw_val:
            lines.append(f"    Raw median (spec layout): {raw_spec}")
        if in_spec:
            lines.append(f"    Spec: chars={spec_chars}, dec={spec_decimal}, "
                         f"div={spec_div}, signed={spec_signed} ({found_spec_key})")
        else:
            lines.append(f"    NOT IN SPEC (tried: {spec_keys_to_try})")
        lines.append(f"    Header: div={header_div}, meta[2]={meta2}, width={hdr_w}")
        if correction != 1.0:
            lines.append(f"    Empirical correction: {correction}")
        lines.append(f"    JSON median: {json_val}")

        def fmt(label, val, err):
            if val is None:
                return f"      {label}: N/A"
            marker = " <-- WINNER" if label[0] in winner else ""
            err_str = f"err={err:.3f}" if err is not None else "err=N/A"
            return f"      {label}: {val:>12.3f}  {err_str}{marker}"

        lines.append(fmt("A (header only) ", approach_a, error_a))
        lines.append(fmt("B (spec divisor)", approach_b, error_b))
        lines.append(fmt("C (spec layout) ", approach_c, error_c))
        lines.append(fmt("D (hdr+correct) ", approach_d, error_d))
        lines.append(f"      JSON reference: {json_val:>12.3f}")

        # Verdict
        if "B" in winner and "D" in winner:
            verdict = "SPEC = CORRECTION (both work)"
        elif "B" in winner:
            verdict = "SPEC DIVISOR WORKS"
        elif "A" in winner and "D" not in winner:
            verdict = "HEADER ALREADY CORRECT"
        elif "D" in winner:
            verdict = "EMPIRICAL CORRECTION NEEDED"
        elif "C" in winner:
            verdict = "SPEC LAYOUT WORKS"
        elif winner == "none":
            verdict = "NO COMPARISON"
        else:
            verdict = f"WINNER: {winner}"
        lines.append(f"    VERDICT: {verdict}")
        lines.append("")

    return results, lines


# ── Main ───────────────────────────────────────────────────────────────────────
def investigate_machine(
    machine_slug: str,
    project_name: str,
    dat_files: list[Path],
    json_files: list[Path],
    specs: dict,
    corrections: dict,
) -> tuple[list[SensorResult], list[str]]:
    """Run investigation for one project directory."""
    lines: list[str] = []
    results: list[SensorResult] = []

    lines.append(f"MACHINE: {machine_slug} (project: {project_name})")
    lines.append(f"DAT files: {len(dat_files)}, JSON files: {len(json_files)}")
    lines.append("")

    # ── Parse DAT files, group by inferred technique ──
    dat_by_technique: dict[str, list[dict]] = defaultdict(list)
    parse_failures = 0

    for dpath in dat_files[:50]:
        parsed = parse_dat_raw(dpath)
        if parsed is None:
            parse_failures += 1
            continue
        dat_by_technique[parsed["technique"]].append(parsed)

    lines.append(f"DAT parsed: {sum(len(v) for v in dat_by_technique.values())} "
                 f"(failures: {parse_failures})")
    for tech, plist in sorted(dat_by_technique.items()):
        lines.append(f"  DAT technique '{tech}': {len(plist)} files")
    lines.append("")

    # ── Parse JSON files, group by technique ──
    json_by_technique = parse_json_files_by_technique(json_files)

    for tech, gt in sorted(json_by_technique.items()):
        lines.append(f"  JSON technique '{tech}': {gt['num_files']} files "
                     f"(used {gt['num_used']}), {len(gt['sensor_medians'])} sensors")
    lines.append("")

    # ── Match DAT technique groups to JSON technique groups ──
    # Build a mapping from DAT technique to JSON technique
    # DAT "SCM" maps to JSON "SCM"; DAT "KELLY" maps to JSON "KELLY"; etc.
    # DAT "GRAB" may appear as JSON "GRAB"; DAT "SOB" on a Seilgreifer machine maps to JSON "GRAB"

    # Try each DAT technique group against the best-matching JSON technique
    matched_pairs: list[tuple[str, str, list[dict], dict]] = []

    for dat_tech, dat_list in dat_by_technique.items():
        # Direct match first
        if dat_tech in json_by_technique:
            matched_pairs.append((dat_tech, dat_tech, dat_list, json_by_technique[dat_tech]))
            continue

        # Try known equivalences
        equivalences = {
            "SCM": ["SCM", "CSM"],
            "CSM": ["SCM", "CSM"],
            "GRAB": ["GRAB", "Seilgreifer"],
            "SOB": ["SOB", "GRAB"],  # SOB BEGINNG on a grab machine
            "KELLY": ["KELLY"],
            "CUT": ["CUT"],
        }
        found = False
        for json_tech_candidate in equivalences.get(dat_tech, []):
            if json_tech_candidate in json_by_technique:
                matched_pairs.append((dat_tech, json_tech_candidate, dat_list,
                                      json_by_technique[json_tech_candidate]))
                found = True
                break

        if not found:
            # Last resort: if only one JSON technique, try matching anyway
            if len(json_by_technique) == 1:
                json_tech = list(json_by_technique.keys())[0]
                matched_pairs.append((dat_tech, json_tech, dat_list,
                                      json_by_technique[json_tech]))
            else:
                lines.append(f"  WARNING: No JSON match for DAT technique '{dat_tech}'")

    lines.append(f"Matched technique pairs: {len(matched_pairs)}")
    for dat_tech, json_tech, dat_list, _ in matched_pairs:
        lines.append(f"  DAT '{dat_tech}' <-> JSON '{json_tech}' ({len(dat_list)} DAT files)")
    lines.append("")

    # ── Run investigation per matched pair ──
    for dat_tech, json_tech, dat_list, json_gt in matched_pairs:
        lines.append("-" * 80)
        lines.append(f"TECHNIQUE GROUP: DAT={dat_tech} -> JSON={json_tech}")
        lines.append("-" * 80)

        group_results, group_lines = investigate_technique_group(
            machine_slug, dat_tech, json_tech, dat_list, json_gt,
            specs, corrections,
        )
        results.extend(group_results)
        lines.extend(group_lines)

    return results, lines


def main():
    print("Loading specs and corrections...")
    specs = load_medef_specs()
    corrections = load_corrections()

    num_specs = sum(1 for k in specs if k != "machine_technique_map")
    print(f"  Specs: {num_specs} technique definitions")
    print(f"  Corrections: {sum(len(v) for v in corrections.values())} across "
          f"{len(corrections)} machines")

    all_lines: list[str] = []
    all_lines.append("SPEC-DRIVEN DAT PARSING INVESTIGATION")
    all_lines.append("=" * 80)
    all_lines.append("")

    all_results: list[SensorResult] = []

    for project_pattern, machine_slug, desc in TEST_CASES:
        print(f"\n--- {project_pattern} ({machine_slug}): {desc} ---")
        project_dir = find_project_dir(project_pattern)
        if project_dir is None:
            print(f"  WARNING: No project directory found for '{project_pattern}'")
            continue

        dat_files, json_files = find_dat_json_files(project_dir)
        print(f"  DAT: {len(dat_files)}, JSON: {len(json_files)}")

        if not dat_files or not json_files:
            print("  WARNING: Missing DAT or JSON files — skipping")
            continue

        results, report_lines = investigate_machine(
            machine_slug, project_pattern, dat_files, json_files, specs, corrections,
        )
        all_results.extend(results)
        all_lines.append("")
        all_lines.extend(report_lines)
        all_lines.append("")
        all_lines.append("~" * 80)
        all_lines.append("")

    # ── Global summary ──
    all_lines.append("")
    all_lines.append("=" * 80)
    all_lines.append("GLOBAL SUMMARY ACROSS ALL MACHINES / TECHNIQUES")
    all_lines.append("=" * 80)

    if all_results:
        total = len(all_results)

        # Quality threshold: error < 0.5 means within ~3x (good match)
        GOOD_THRESHOLD = 0.5

        # ── Raw winner analysis ──
        spec_wins = sum(1 for r in all_results if "B" in r.winner or "C" in r.winner)
        corr_wins = sum(1 for r in all_results
                        if "D" in r.winner and "B" not in r.winner)
        header_wins = sum(1 for r in all_results
                          if "A" in r.winner and "B" not in r.winner
                          and "D" not in r.winner)

        all_lines.append(f"Total sensors compared: {total}")
        all_lines.append("")
        all_lines.append("--- WHO WINS (least error, regardless of quality) ---")
        all_lines.append(f"  Spec divisor wins (B or C):      {spec_wins:>3d} / {total} "
                         f"({100*spec_wins/total:.0f}%)")
        all_lines.append(f"  Correction needed (D only):      {corr_wins:>3d} / {total} "
                         f"({100*corr_wins/total:.0f}%)")
        all_lines.append(f"  Header already correct (A only): {header_wins:>3d} / {total} "
                         f"({100*header_wins/total:.0f}%)")
        all_lines.append("")

        winner_counts: dict[str, int] = defaultdict(int)
        for r in all_results:
            winner_counts[r.winner] += 1
        all_lines.append("  Winner breakdown:")
        for w, cnt in sorted(winner_counts.items(), key=lambda x: -x[1]):
            all_lines.append(f"    {w:>10s}: {cnt}")

        # ── Quality-filtered analysis ──
        # A "good" result means the winning approach has error < threshold
        all_lines.append("")
        all_lines.append(f"--- QUALITY CHECK (error < {GOOD_THRESHOLD} = within "
                         f"{10**GOOD_THRESHOLD:.1f}x) ---")

        def best_error(r: SensorResult) -> float | None:
            errs = [e for e in [r.error_a, r.error_b, r.error_c, r.error_d]
                    if e is not None]
            return min(errs) if errs else None

        good_results = [r for r in all_results
                        if (be := best_error(r)) is not None and be < GOOD_THRESHOLD]
        bad_results = [r for r in all_results
                       if (be := best_error(r)) is not None and be >= GOOD_THRESHOLD]
        no_data = [r for r in all_results if best_error(r) is None]

        all_lines.append(f"  Good matches (best error < {GOOD_THRESHOLD}): "
                         f"{len(good_results)} / {total}")
        all_lines.append(f"  Poor matches (best error >= {GOOD_THRESHOLD}): "
                         f"{len(bad_results)} / {total}")
        if no_data:
            all_lines.append(f"  No data: {len(no_data)}")

        if good_results:
            spec_good = sum(1 for r in good_results
                            if "B" in r.winner or "C" in r.winner)
            corr_good = sum(1 for r in good_results
                            if "D" in r.winner and "B" not in r.winner)
            header_good = sum(1 for r in good_results
                              if "A" in r.winner and "B" not in r.winner
                              and "D" not in r.winner)
            n = len(good_results)
            all_lines.append(f"  Among good matches:")
            all_lines.append(f"    Spec correct:      {spec_good:>3d} / {n} "
                             f"({100*spec_good/n:.0f}%)")
            all_lines.append(f"    Correction needed: {corr_good:>3d} / {n} "
                             f"({100*corr_good/n:.0f}%)")
            all_lines.append(f"    Header correct:    {header_good:>3d} / {n} "
                             f"({100*header_good/n:.0f}%)")

        if bad_results:
            all_lines.append(f"  Poor matches (no approach gets within "
                             f"{10**GOOD_THRESHOLD:.1f}x):")
            for r in sorted(bad_results, key=lambda x: x.sensor_name):
                be = best_error(r)
                all_lines.append(f"    {r.sensor_name:<30s} {r.machine_slug:<15s} "
                                 f"{r.technique:<8s} best_err={be:.2f} "
                                 f"({10**be:.0f}x off)")

        # ── Pattern analysis: when does spec work vs not? ──
        all_lines.append("")
        all_lines.append("--- PATTERN ANALYSIS ---")
        all_lines.append("Sensors where spec divisor > 1 (spec says scaling needed):")
        spec_scaling = [r for r in all_results
                        if r.spec_divisor is not None and r.spec_divisor > 1]
        if spec_scaling:
            spec_right = sum(1 for r in spec_scaling
                             if r.error_b is not None and r.error_b < GOOD_THRESHOLD)
            all_lines.append(f"  {len(spec_scaling)} sensors with spec_divisor > 1")
            all_lines.append(f"  Of these, spec divisor gives good result: "
                             f"{spec_right} ({100*spec_right/len(spec_scaling):.0f}%)")

        all_lines.append("")
        all_lines.append("Sensors where header_divisor == 1 AND spec_divisor == 1 "
                         "(both say no scaling):")
        both_one = [r for r in all_results
                    if r.header_divisor == 1
                    and r.spec_divisor is not None and r.spec_divisor == 1]
        if both_one:
            needs_corr = sum(1 for r in both_one if r.correction != 1.0)
            all_lines.append(f"  {len(both_one)} sensors where header=1 AND spec=1")
            all_lines.append(f"  Of these, {needs_corr} still need empirical correction")
            all_lines.append(f"  -> Firmware encodes at HIGHER resolution than spec for "
                             f"these {needs_corr} sensors")

        # ── Conclusion ──
        all_lines.append("")
        all_lines.append("=" * 80)
        all_lines.append("CONCLUSION")
        all_lines.append("=" * 80)

        if good_results:
            n = len(good_results)
            spec_pct = 100 * sum(1 for r in good_results
                                 if "B" in r.winner or "C" in r.winner) / n
            corr_pct = 100 * sum(1 for r in good_results
                                 if "D" in r.winner and "B" not in r.winner) / n
        else:
            spec_pct = 0
            corr_pct = 0

        if spec_pct > 80:
            all_lines.append("Spec-driven parsing CAN largely replace empirical corrections.")
        elif spec_pct > 40:
            all_lines.append("HYBRID APPROACH needed:")
            all_lines.append(f"  - Spec divisor works for ~{spec_pct:.0f}% of sensors")
            all_lines.append(f"  - Empirical corrections still needed for ~{corr_pct:.0f}%")
            all_lines.append("  - Main gap: firmware encodes pressure/force sensors at higher")
            all_lines.append("    resolution than MEDEF spec declares (e.g. centibar vs bar)")
        else:
            all_lines.append("Spec is unreliable — empirical corrections remain necessary.")
            all_lines.append(f"Spec only correct for {spec_pct:.0f}% of well-matched sensors.")

        # ── Full detail table ──
        all_lines.append("")
        all_lines.append("=" * 80)
        all_lines.append("FULL DETAIL TABLE")
        all_lines.append("=" * 80)
        all_lines.append(f"{'Sensor':<35s} {'Machine':<15s} {'Tech':<8s} "
                         f"{'Winner':>8s} {'Spec÷':>8s} {'Hdr÷':>6s} "
                         f"{'Corr':>8s} {'ErrB':>6s} {'ErrD':>6s} {'Quality':>7s}")
        all_lines.append("-" * 120)
        for r in sorted(all_results, key=lambda x: (x.winner, x.sensor_name)):
            err_b = f"{r.error_b:.3f}" if r.error_b is not None else "-"
            err_d = f"{r.error_d:.3f}" if r.error_d is not None else "-"
            spec_d = str(r.spec_divisor) if r.spec_divisor is not None else "-"
            be = best_error(r)
            quality = "GOOD" if be is not None and be < GOOD_THRESHOLD else "POOR"
            all_lines.append(
                f"  {r.sensor_name:<33s} {r.machine_slug:<15s} {r.technique:<8s} "
                f"{r.winner:>8s} {spec_d:>8s} {r.header_divisor:>6d} "
                f"{r.correction:>8.3f} {err_b:>6s} {err_d:>6s} {quality:>7s}"
            )
    else:
        all_lines.append("No results.")

    # ── Write report ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "spec_parsing_investigation.txt"
    with open(output_path, "w") as f:
        f.write("\n".join(all_lines) + "\n")

    print(f"\nReport: {output_path}")
    print(f"Total sensors compared: {len(all_results)}")

    # Print summary
    for line in all_lines[-30:]:
        print(line)


if __name__ == "__main__":
    main()
