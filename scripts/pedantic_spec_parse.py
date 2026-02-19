"""Pedantic spec-only DAT parsing — follow the MEDEF format specification EXACTLY.

The MEDEF PDF specifies for each sensor:
  - characters: number of INTEGER digits (before the decimal point)
  - decimal: number of FRACTIONAL digits (after the decimal point)
  - signed: whether there is a +/- prefix
  - format: exact display notation (e.g. "XXX", "+/-XX,X", "XX,XX")

From these, the specification defines:
  - Field width in data = (1 if signed else 0) + characters + decimal
  - Divisor = 10^decimal
  - Max value = (10^(characters+decimal) - 1) / 10^decimal

This script:
  1. Reads a DAT file header to get field ORDER and NAMES
  2. Validates field layout against ALL spec versions (v7 AND v8) using sign anchors
  3. Uses whichever spec version's widths match the data
  4. Extracts raw values using the validated spec widths
  5. Divides by spec divisor (10^decimal)
  6. Compares to JSON ground truth
  7. Checks physical plausibility

Usage:
    uv run python scripts/pedantic_spec_parse.py
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from macchine.parsers.dat_parser import _parse_header

SPECS_PATH = Path(__file__).resolve().parent.parent / "specs_data" / "medef_specs.yaml"
CORRECTIONS_PATH = (
    Path(__file__).resolve().parent.parent / "macchine" / "harmonize" / "dat_corrections.yaml"
)
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "diagnostics"

# Physical plausibility limits (generous — ABSOLUTE maximums)
PHYSICAL_LIMITS = {
    "m": (-300, 300),
    "t": (-200, 200),
    "Grad": (-90, 90),
    "°": (-90, 90),
    "bar": (-10, 600),
    "kNm": (-500, 500),
    "%": (-200, 300),
    "cm": (-5000, 5000),
    "cm/min": (-50000, 50000),
    "°C": (-50, 200),
    "U/min": (-200, 200),
    "l/min": (0, 5000),
    "cbm": (0, 100),
    "cbm/h": (0, 500),
    "U": (0, 99999),
    "mm/U": (0, 99999),
    "U/m": (0, 999),
    "Nr.": (0, 99),
}


def load_specs():
    with open(SPECS_PATH) as f:
        return yaml.safe_load(f)


def load_corrections():
    with open(CORRECTIONS_PATH) as f:
        return yaml.safe_load(f) or {}


def normalize_spec_name(name: str) -> str:
    return name.replace("\ufb00", "ff").replace("\ufb01", "fi").replace("\ufb02", "fl")


def spec_field_width(spec_entry: dict) -> int:
    chars = spec_entry["characters"]
    dec = spec_entry["decimal"]
    signed = spec_entry.get("signed", False)
    return (1 if signed else 0) + chars + dec


def spec_max_value(spec_entry: dict) -> float:
    chars = spec_entry["characters"]
    dec = spec_entry["decimal"]
    total_digits = chars + dec
    max_raw = 10**total_digits - 1
    return max_raw / (10**dec) if dec > 0 else max_raw


def find_sensor_in_spec(sensor_name: str, spec: dict) -> dict | None:
    sensors = spec.get("sensors", {})
    if sensor_name in sensors:
        return dict(sensors[sensor_name])
    for spec_sensor, spec_info in sensors.items():
        if normalize_spec_name(spec_sensor) == normalize_spec_name(sensor_name):
            return dict(spec_info)
    return None


def scan_sign_positions(data_lines: list[str]) -> dict[int, float]:
    """Scan data lines for sign character (+/-) positions and their frequency."""
    n = len(data_lines)
    if n == 0:
        return {}
    pos_counts = defaultdict(int)
    for line in data_lines:
        for i, c in enumerate(line):
            if c in "+-":
                pos_counts[i] += 1
    return {pos: count / n for pos, count in sorted(pos_counts.items())}


def validate_spec_against_sign_anchors(
    field_defs: list, spec: dict, data_lines: list[str], data_length: int
) -> tuple[list[int] | None, list[str], float]:
    """Check if a spec version's field widths match the sign-anchor positions.

    Strategy:
    1. Compute spec widths for header fields (+ trailing spec-only fields)
    2. Check that widths sum to data_length
    3. Check that ALL observed sign anchors match expected signed-field positions
       (not the reverse — some signed fields may have zero values with no sign char)

    Returns (widths, diagnostics, score).
    """
    diag = []

    # Get sign anchor positions (>50% frequency = reliable anchors)
    sign_freq = scan_sign_positions(data_lines[:500])
    anchors = {pos for pos, freq in sign_freq.items() if freq > 0.5}

    diag.append(f"  Sign anchors (>50%): {sorted(anchors)}")

    # Build spec widths for each header field
    widths = []
    spec_sensors_used = set()
    for fdef in field_defs:
        spec_entry = find_sensor_in_spec(fdef.name, spec)
        if spec_entry:
            w = spec_field_width(spec_entry)
            widths.append(w)
            spec_sensors_used.add(fdef.name)
            # Also track normalized name
            for sname in spec.get("sensors", {}):
                if normalize_spec_name(sname) == normalize_spec_name(fdef.name):
                    spec_sensors_used.add(sname)
        elif fdef.is_direction:
            widths.append(1)
        else:
            widths.append(None)  # placeholder

    # Check for spec sensors NOT in header (trailing fields)
    trailing_spec_widths = []
    for sname, sinfo in spec.get("sensors", {}).items():
        norm = normalize_spec_name(sname)
        if sname not in spec_sensors_used and norm not in {normalize_spec_name(n) for n in spec_sensors_used}:
            w = spec_field_width(sinfo)
            trailing_spec_widths.append((sname, w))

    # Compute known total
    known_total = sum(w for w in widths if w is not None) + sum(w for _, w in trailing_spec_widths)
    unknown_count = sum(1 for w in widths if w is None)

    # Fill unknown widths
    if unknown_count > 0:
        remaining = data_length - known_total
        if remaining > 0:
            fill_width = remaining // unknown_count
            remainder = remaining % unknown_count
            idx = 0
            for i in range(len(widths)):
                if widths[i] is None:
                    widths[i] = fill_width + (1 if idx < remainder else 0)
                    idx += 1
        elif remaining == 0:
            # All unknowns must be 0 width — impossible, set to fallback
            for i in range(len(widths)):
                if widths[i] is None:
                    widths[i] = 0
        else:
            # Negative remaining — spec is already too wide
            for i in range(len(widths)):
                if widths[i] is None:
                    widths[i] = 0

    # Add trailing spec widths (not in header but in data)
    total_header = sum(widths)
    total_with_trailing = total_header + sum(w for _, w in trailing_spec_widths)

    diag.append(f"  Header fields width: {total_header}")
    if trailing_spec_widths:
        diag.append(f"  Trailing spec fields: {', '.join(f'{n}({w})' for n, w in trailing_spec_widths)}")
    diag.append(f"  Total (header + trailing): {total_with_trailing}  (data length: {data_length})")

    # Check total width
    if total_with_trailing != data_length:
        diag.append(f"  WIDTH MISMATCH: {total_with_trailing} != {data_length}")
        return None, diag, 0.0

    # Compute expected signed positions
    pos = 0
    expected_sign_positions = set()
    for i, (fdef, w) in enumerate(zip(field_defs, widths)):
        spec_entry = find_sensor_in_spec(fdef.name, spec)
        if spec_entry and spec_entry.get("signed", False):
            expected_sign_positions.add(pos)
        pos += w
    # Add trailing
    for sname, w in trailing_spec_widths:
        sinfo = spec["sensors"].get(sname, {})
        if sinfo.get("signed", False):
            expected_sign_positions.add(pos)
        pos += w

    diag.append(f"  Expected sign positions (spec): {sorted(expected_sign_positions)}")

    # KEY CHECK: every observed anchor must be at an expected sign position
    anchors_matched = anchors & expected_sign_positions
    anchors_unexpected = anchors - expected_sign_positions

    if anchors:
        anchor_precision = len(anchors_matched) / len(anchors)
    else:
        anchor_precision = 1.0  # no anchors to contradict

    diag.append(f"  Anchors matching expected positions: {len(anchors_matched)}/{len(anchors)} = {anchor_precision:.0%}")
    if anchors_unexpected:
        diag.append(f"  UNEXPECTED anchors: {sorted(anchors_unexpected)} — "
                    f"these sign chars don't match any spec signed field!")

    # Score: anchor precision (all anchors should match) weighted by anchor count
    score = anchor_precision * min(1.0, len(anchors) / 3)  # need at least 3 anchors for confidence

    return widths, diag, score


def extract_with_widths(data_line: str, widths: list[int]) -> list[str]:
    result = []
    pos = 0
    for w in widths:
        if pos + w > len(data_line):
            result.append(None)
        else:
            result.append(data_line[pos:pos + w])
        pos += w
    return result


def parse_raw_integer(raw_str: str) -> int | None:
    """Parse a raw field string as integer, handling +/- signs and spaces."""
    if raw_str is None:
        return None
    s = raw_str.strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def load_json_medians(json_dir: Path, max_files: int = 50) -> dict[str, float]:
    """Load JSON ground truth and compute medians per sensor."""
    if not json_dir.exists():
        return {}

    json_files = sorted(json_dir.glob("*.json"))[:max_files]
    sensor_values = defaultdict(list)

    for jf in json_files:
        try:
            with open(jf) as f:
                data = json.load(f)
            ts_block = data.get("timeSeriesBlock", {})
            for sensor in ts_block.get("serialValuesFree", []):
                name = sensor.get("seriesName") or sensor.get("description", "")
                values = sensor.get("values", [])
                nums = [v for v in values if isinstance(v, (int, float)) and v != 0]
                if nums and name:
                    sensor_values[name].extend(nums)
        except Exception:
            continue

    medians = {}
    for name, vals in sensor_values.items():
        if vals:
            medians[name] = statistics.median(vals)
    return medians


def analyze_machine(
    machine_slug: str,
    label: str,
    dat_dir: Path,
    json_dir: Path | None,
    specs: dict,
    corrections: dict,
    out,
    max_dat_files: int = 30,
):
    """Full analysis for one machine."""
    out.write(f"\n{'#'*80}\n")
    out.write(f"# MACHINE: {machine_slug} — {label}\n")
    out.write(f"{'#'*80}\n\n")

    # Find DAT files
    dat_files = sorted(dat_dir.glob("*.dat"))
    if not dat_files:
        out.write("No DAT files found.\n")
        return []

    # Parse first file's header
    first_path = dat_files[0]
    with open(first_path, "rb") as f:
        raw = f.read()
    text = raw.decode("iso-8859-1")
    sections = text.split("$")
    field_defs = _parse_header(sections[0])

    dat_sections_first = [s for s in sections if s.startswith("DAT")]
    if not dat_sections_first:
        out.write("No DAT sections in first file.\n")
        return []

    data_length = len(dat_sections_first[0]) - 3  # minus "DAT"
    out.write(f"Header: {len(field_defs)} fields, data line: {data_length} chars\n")
    out.write(f"DAT files: {len(dat_files)}\n\n")

    # Show header fields
    out.write("HEADER FIELDS:\n")
    for i, fdef in enumerate(field_defs):
        meta2 = fdef.meta[2] if len(fdef.meta) > 2 else "?"
        sign = "S" if fdef.signed else ("D" if fdef.is_direction else "U")
        out.write(f"  {i+1:>2}. {fdef.name:<30s}  div={fdef.divisor:<6}  "
                  f"meta[2]={meta2:<3}  {sign}\n")
    out.write("\n")

    # Collect data lines from multiple files
    all_data_lines = []
    for dp in dat_files[:max_dat_files]:
        try:
            with open(dp, "rb") as f:
                raw = f.read()
            text = raw.decode("iso-8859-1")
            for sec in text.split("$"):
                if sec.startswith("DAT"):
                    dl = sec[3:]
                    if len(dl) == data_length:
                        all_data_lines.append(dl)
        except Exception:
            continue

    out.write(f"Collected {len(all_data_lines)} data lines from {min(len(dat_files), max_dat_files)} files\n\n")

    # Try BOTH v7 and v8 specs for this machine's technique
    machine_map = specs.get("machine_technique_map", {})
    mapped_keys = machine_map.get(machine_slug, [])

    # Get all technique types and try both v7 and v8
    techniques = set()
    for key in mapped_keys:
        tech = specs.get(key, {}).get("technique", "")
        if tech:
            techniques.add(tech)

    # Build candidate spec keys: mapped + alternative versions
    candidate_keys = list(mapped_keys)
    for tech in techniques:
        for key_name, spec_data in specs.items():
            if isinstance(spec_data, dict) and spec_data.get("technique") == tech:
                if key_name not in candidate_keys:
                    candidate_keys.append(key_name)

    out.write(f"SPEC VERSION VALIDATION (testing which version matches data layout)\n")
    out.write("=" * 80 + "\n\n")

    best_widths = None
    best_spec_key = None
    best_score = 0

    for spec_key in candidate_keys:
        if spec_key not in specs or spec_key == "machine_technique_map":
            continue
        spec_candidate = specs[spec_key]
        out.write(f"  Testing {spec_key}:\n")
        widths, diag, score = validate_spec_against_sign_anchors(
            field_defs, spec_candidate, all_data_lines, data_length
        )
        for d in diag:
            out.write(f"    {d}\n")

        if widths is not None:
            out.write(f"    >>> VALID: total={sum(widths) + sum(spec_field_width(spec_candidate['sensors'][sn]) for sn in spec_candidate.get('sensors', {}) if sn not in {f.name for f in field_defs} and normalize_spec_name(sn) not in {normalize_spec_name(f.name) for f in field_defs})}, score={score:.2f}\n")
            if score > best_score:
                best_score = score
                best_widths = widths
                best_spec_key = spec_key
        else:
            out.write(f"    >>> INVALID (score={score:.2f})\n")
        out.write("\n")

    if best_widths is None:
        out.write("NO SPEC VERSION MATCHES THE DATA!\n")
        out.write("Cannot proceed with spec-only parsing.\n")
        return []

    out.write(f"BEST MATCH: {best_spec_key} (score: {best_score:.2f})\n\n")
    spec = specs[best_spec_key]

    # Show validated field layout
    out.write("VALIDATED FIELD LAYOUT:\n")
    out.write("-" * 80 + "\n")
    pos = 0
    for i, (fdef, w) in enumerate(zip(field_defs, best_widths)):
        se = find_sensor_in_spec(fdef.name, spec)
        if se:
            fmt = se.get("format", "?")
            div = se["expected_divisor"]
            maxv = spec_max_value(se)
            unit = se.get("unit", "")
            out.write(f"  {i+1:>2}. pos {pos:>3}-{pos+w:>3} ({w}ch)  "
                      f"{fdef.name:<30s}  spec: {fmt:<12s} div={div:<6} max={maxv:<10.2f} {unit}\n")
        else:
            out.write(f"  {i+1:>2}. pos {pos:>3}-{pos+w:>3} ({w}ch)  "
                      f"{fdef.name:<30s}  [NOT IN SPEC]\n")
        pos += w
    out.write("-" * 80 + "\n\n")

    # Extract raw values using validated widths
    all_raw = [[] for _ in range(len(field_defs))]
    parse_fail_count = [0 for _ in range(len(field_defs))]

    for dl in all_data_lines:
        slices = extract_with_widths(dl, best_widths)
        for i, (fdef, raw_str) in enumerate(zip(field_defs, slices)):
            if fdef.is_direction:
                continue
            val = parse_raw_integer(raw_str)
            if val is not None:
                all_raw[i].append(val)
            else:
                parse_fail_count[i] += 1

    # Load JSON ground truth
    json_medians = {}
    if json_dir and json_dir.exists():
        json_medians = load_json_medians(json_dir)
        out.write(f"JSON ground truth: {len(json_medians)} sensors from {json_dir.name}\n\n")

    # Load empirical corrections for this machine
    machine_corrections = corrections.get(machine_slug, {})

    # SENSOR-BY-SENSOR ANALYSIS
    out.write("SENSOR-BY-SENSOR ANALYSIS\n")
    out.write("=" * 80 + "\n")
    out.write("For each sensor: raw → spec value → plausibility check → JSON comparison\n\n")

    results = []

    for i, fdef in enumerate(field_defs):
        if fdef.is_direction:
            continue

        se = find_sensor_in_spec(fdef.name, spec)
        raw_vals = all_raw[i]

        if not raw_vals:
            out.write(f"  {fdef.name}: NO VALID DATA ({parse_fail_count[i]} failures)\n\n")
            continue

        # Raw statistics
        raw_median = statistics.median(raw_vals)
        raw_min = min(raw_vals)
        raw_max = max(raw_vals)
        sorted_vals = sorted(raw_vals)
        raw_p5 = sorted_vals[max(0, len(raw_vals) * 5 // 100)]
        raw_p95 = sorted_vals[min(len(raw_vals) - 1, len(raw_vals) * 95 // 100)]
        # Active median: median of non-zero values (for fair JSON comparison)
        nonzero_vals = [v for v in raw_vals if v != 0]
        raw_active_median = statistics.median(nonzero_vals) if nonzero_vals else 0

        # Spec-only physical value
        if se:
            sd = se["expected_divisor"]
            unit = se.get("unit", "")
            fmt = se.get("format", "?")
            maxv = spec_max_value(se)
            chars = se["characters"]
            dec = se["decimal"]
            s_signed = se.get("signed", False)
        else:
            # Not in spec — use header divisor
            sd = fdef.divisor
            unit = fdef.unit
            fmt = "?"
            maxv = None
            chars = None
            dec = None
            s_signed = None

        phys_median = raw_median / sd if sd else raw_median
        phys_min = raw_min / sd if sd else raw_min
        phys_max = raw_max / sd if sd else raw_max
        phys_p5 = raw_p5 / sd if sd else raw_p5
        phys_p95 = raw_p95 / sd if sd else raw_p95
        phys_active_median = raw_active_median / sd if sd else raw_active_median

        # Corrected value (for comparison)
        corr = machine_corrections.get(fdef.name, 1.0)
        corr_median = (raw_median / fdef.divisor / corr) if fdef.divisor else raw_median
        corr_active_median = (raw_active_median / fdef.divisor / corr) if fdef.divisor else raw_active_median

        # JSON comparison
        json_val = json_medians.get(fdef.name)

        # Plausibility check
        phys_limit = PHYSICAL_LIMITS.get(unit)
        implausible = False
        reason = ""
        if phys_limit and unit not in ("", "None", None):
            lo, hi = phys_limit
            if phys_p5 < lo * 2 or phys_p95 > hi * 2:  # 2x margin
                implausible = True
                reason = f"p5={phys_p5:.1f}, p95={phys_p95:.1f} outside [{lo},{hi}] {unit}"

        exceeds_range = False
        if maxv is not None:
            if abs(phys_p95) > maxv * 1.01 or abs(phys_p5) > maxv * 1.01:
                exceeds_range = True

        verdict = "PLAUSIBLE"
        if exceeds_range and implausible:
            verdict = "UTTERLY IMPLAUSIBLE"
        elif exceeds_range:
            verdict = "EXCEEDS SPEC RANGE"
        elif implausible:
            verdict = "PHYSICALLY IMPLAUSIBLE"

        # JSON match quality (using active medians for fair comparison)
        json_match = ""
        if json_val is not None and json_val != 0 and phys_active_median != 0:
            ratio_spec = abs(phys_active_median / json_val)
            ratio_corr = abs(corr_active_median / json_val) if corr_active_median != 0 else 0
            if 0.1 <= ratio_spec <= 10:
                json_match = f"SPEC CLOSE (active_ratio={ratio_spec:.2f})"
            else:
                json_match = f"SPEC OFF (active_ratio={ratio_spec:.1f})"
            if ratio_corr and 0.1 <= ratio_corr <= 10:
                json_match += f" | CORR CLOSE (active_ratio={ratio_corr:.2f})"
            elif ratio_corr:
                json_match += f" | CORR OFF (active_ratio={ratio_corr:.1f})"
        elif json_val is not None and json_val != 0 and phys_active_median == 0:
            json_match = "DAT all zeros (no active data)"

        out.write(f"  SENSOR: {fdef.name}")
        if se:
            out.write(f"  [{unit}]  spec: {fmt}  (chars={chars}, dec={dec}, "
                      f"signed={s_signed}, div={sd})\n")
        else:
            out.write(f"  [{unit}]  NOT IN SPEC\n")

        out.write(f"    Raw: median={raw_median:.0f}  range=[{raw_min}, {raw_max}]  "
                  f"p5/p95=[{raw_p5}, {raw_p95}]  "
                  f"({len(raw_vals)} values, {parse_fail_count[i]} failures)\n")

        out.write(f"    SPEC VALUE (median):  {raw_median:.0f} / {sd} = {phys_median:.4f} {unit}\n")
        out.write(f"    SPEC VALUE (active):  {raw_active_median:.0f} / {sd} = {phys_active_median:.4f} {unit}"
                  f"  ({len(nonzero_vals)} non-zero values)\n")
        out.write(f"    SPEC VALUE range: [{phys_min:.4f}, {phys_max:.4f}] {unit}\n")

        if maxv is not None:
            out.write(f"    Spec max representable: ±{maxv:.2f} {unit}\n")

        if corr != 1.0:
            out.write(f"    EMPIRICAL CORRECTION value: "
                      f"{raw_median:.0f} / {fdef.divisor} / {corr} = {corr_median:.4f} {unit}\n")

        if json_val is not None:
            out.write(f"    JSON ground truth: {json_val:.4f} {unit}\n")
            out.write(f"    {json_match}\n")

        if verdict != "PLAUSIBLE":
            out.write(f"    *** {verdict}: {reason} ***\n")

        out.write(f"    VERDICT: {verdict}\n\n")

        results.append({
            "sensor": fdef.name,
            "label": label,
            "unit": unit,
            "spec_key": best_spec_key,
            "raw_median": raw_median,
            "raw_active_median": raw_active_median,
            "spec_divisor": sd,
            "spec_value": phys_median,
            "spec_active": phys_active_median,
            "spec_p5": phys_p5,
            "spec_p95": phys_p95,
            "spec_max": maxv,
            "correction": corr,
            "corr_value": corr_median,
            "corr_active": corr_active_median,
            "json_value": json_val,
            "verdict": verdict,
            "reason": reason,
        })

    return results


def main():
    specs = load_specs()
    corrections = load_corrections()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "pedantic_spec_parse.txt"

    raw_dl = Path(__file__).resolve().parent.parent / "output/raw_downloads"

    test_cases = [
        {
            "machine_slug": "gb50_601",
            "label": "Catania (Seilgreifer/GRAB)",
            "dat_dir": raw_dl / "2026-02-16_1514 Catania/Unidentified",
            "json_dir": raw_dl / "2026-02-16_1514 Catania/GB-50-null 601 | 01K00047564",
        },
        {
            "machine_slug": "bg33v_5610",
            "label": "LignanoSabbiadoro (KELLY)",
            "dat_dir": raw_dl / "2026-02-16_LignanoSabbiadoro/Unidentified",
            "json_dir": raw_dl / "2026-02-16_LignanoSabbiadoro/BG-33-V #5610 | 01K00044171",
        },
        {
            "machine_slug": "bg33v_5610",
            "label": "VICENZA (SCM/CSM)",
            "dat_dir": raw_dl / "2026-02-16_1508 - VICENZA/Unidentified",
            "json_dir": raw_dl / "2026-02-16_1508 - VICENZA/BG-33-V #5610 | 01K00044171",
        },
    ]

    with open(out_path, "w") as out:
        out.write("PEDANTIC SPEC-ONLY DAT PARSING (v2)\n")
        out.write("Follow the MEDEF format specification EXACTLY. No heuristics.\n")
        out.write("=" * 80 + "\n\n")
        out.write("KEY IMPROVEMENT: Validate spec version by checking field widths against\n")
        out.write("sign-anchor positions in the actual data. The CORRECT spec version will\n")
        out.write("have sign characters (+/-) at exactly the right positions.\n\n")

        all_results = []

        for tc in test_cases:
            try:
                results = analyze_machine(
                    machine_slug=tc["machine_slug"],
                    label=tc["label"],
                    dat_dir=tc["dat_dir"],
                    json_dir=tc.get("json_dir"),
                    specs=specs,
                    corrections=corrections,
                    out=out,
                )
                all_results.extend(results)
            except Exception as e:
                out.write(f"\n  ERROR: {e}\n")
                import traceback
                out.write(traceback.format_exc())

        # Final summary
        out.write(f"\n\n{'='*80}\n")
        out.write("FINAL SUMMARY\n")
        out.write(f"{'='*80}\n\n")

        # Count by verdict
        plausible = [r for r in all_results if r["verdict"] == "PLAUSIBLE"]
        implausible = [r for r in all_results if r["verdict"] != "PLAUSIBLE"]

        out.write(f"Total sensors analyzed: {len(all_results)}\n")
        out.write(f"  Plausible with spec divisor: {len(plausible)}\n")
        out.write(f"  Implausible: {len(implausible)}\n\n")

        # JSON comparison using active (non-zero) medians
        with_json = [r for r in all_results
                     if r["json_value"] is not None and r["json_value"] != 0
                     and r["spec_active"] != 0]
        if with_json:
            # Group by label
            labels = sorted(set(r["label"] for r in with_json))
            for lbl in labels:
                subset = [r for r in with_json if r["label"] == lbl]
                spec_close = 0
                corr_close = 0
                spec_wins = 0
                corr_wins = 0

                for r in subset:
                    jv = abs(r["json_value"])
                    sv = abs(r["spec_active"])
                    cv = abs(r["corr_active"])
                    if jv > 0:
                        sr = sv / jv
                        cr = cv / jv if cv > 0 else 0
                        if 0.1 <= sr <= 10:
                            spec_close += 1
                        if cr > 0 and 0.1 <= cr <= 10:
                            corr_close += 1
                        spec_err = abs(sr - 1)
                        corr_err = abs(cr - 1) if cr > 0 else 999
                        if spec_err < corr_err:
                            spec_wins += 1
                        elif corr_err < spec_err:
                            corr_wins += 1

                out.write(f"\nJSON COMPARISON — {lbl} ({len(subset)} sensors with active DAT + JSON):\n")
                out.write(f"  Spec (active) within 10x of JSON:    {spec_close}/{len(subset)}\n")
                out.write(f"  Corrected (active) within 10x of JSON: {corr_close}/{len(subset)}\n")
                out.write(f"  Spec closer to JSON:                 {spec_wins}/{len(subset)}\n")
                out.write(f"  Correction closer to JSON:           {corr_wins}/{len(subset)}\n\n")

                out.write(f"  {'Sensor':<30s}  {'SpecActive':>12s}  {'CorrActive':>12s}  {'JSON':>12s}  {'Winner':>6s}  Spec/JSON  Corr/JSON\n")
                out.write("  " + "-" * 110 + "\n")
                for r in sorted(subset, key=lambda x: x["sensor"]):
                    jv = r["json_value"]
                    sv = r["spec_active"]
                    cv = r["corr_active"]
                    if jv != 0:
                        sr = abs(sv / jv)
                        cr = abs(cv / jv) if cv != 0 else 0
                        spec_err = abs(sr - 1)
                        corr_err = abs(cr - 1) if cr > 0 else 999
                        winner = "SPEC" if spec_err < corr_err else "CORR"
                        if spec_err < 0.5 and corr_err < 0.5:
                            winner = "BOTH"
                    else:
                        sr = cr = 0
                        winner = "?"
                    out.write(f"  {r['sensor']:<30s}  {sv:>12.4f}  {cv:>12.4f}  {jv:>12.4f}  {winner:>6s}  {sr:>9.2f}  {cr:>9.2f}\n")

        if implausible:
            out.write(f"\nIMPLAUSIBLE VALUES:\n")
            for r in implausible:
                out.write(f"  {r['sensor']:<30s}  spec_value={r['spec_value']:.4f} {r['unit']}  "
                          f"verdict={r['verdict']}  {r['reason']}\n")

    print(f"Wrote {out_path}")
    print(f"Total: {len(all_results)} sensors, "
          f"{len([r for r in all_results if r['verdict'] == 'PLAUSIBLE'])} plausible, "
          f"{len([r for r in all_results if r['verdict'] != 'PLAUSIBLE'])} implausible")


if __name__ == "__main__":
    main()
