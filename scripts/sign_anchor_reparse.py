"""Re-parse DAT data using sign-anchor-based field width correction.

Proves/disproves whether field width misestimation is the reason spec
divisors appeared unreliable.

Strategy:
  1. Parse header normally to get field definitions
  2. Scan ALL data lines for +/- character positions (sign anchors)
  3. Map anchors to signed fields to get ground-truth field boundaries
  4. Re-extract raw integers with corrected boundaries
  5. Test: corrected_raw / spec_divisor — does it match JSON?

Usage:
    uv run python scripts/sign_anchor_reparse.py
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from macchine.parsers.dat_parser import DatFieldDef, _estimate_field_widths, _parse_header
from macchine.parsers.beginng_parser import parse_beginng

ROOT = Path(__file__).resolve().parent.parent
SPECS_PATH = ROOT / "specs_data" / "medef_specs.yaml"
CORRECTIONS_PATH = ROOT / "macchine" / "harmonize" / "dat_corrections.yaml"
RAW_DIR = ROOT / "output" / "raw_downloads"
OUTPUT_DIR = ROOT / "output" / "diagnostics"

LIGATURE_MAP = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl"}

JSON_TECHNIQUE_TO_SPEC = {
    "KELLY": "KELLY", "SOB": "SOB", "GRAB": "Seilgreifer",
    "CUT": "CUT", "SCM": "CSM", "CSM": "CSM", "FDP": "FDP",
}


def normalize_spec_name(name: str) -> str:
    for lig, repl in LIGATURE_MAP.items():
        name = name.replace(lig, repl)
    return name


def load_specs() -> dict:
    with open(SPECS_PATH) as f:
        return yaml.safe_load(f)


def load_corrections() -> dict:
    with open(CORRECTIONS_PATH) as f:
        return yaml.safe_load(f) or {}


def infer_technique(field_defs: list[DatFieldDef]) -> str:
    """Infer technique from sensor names. Check specifics before generics."""
    names = {f.name.lower() for f in field_defs}
    # CUT: milling/cutter sensors
    if any("fräs" in n or "fraes" in n or "dws" in n for n in names):
        return "CUT"
    # SCM/CSM: suspension (slurry) sensors — check BEFORE kelly/grab
    if any("susp" in n for n in names):
        return "SCM"
    # SOB/KELLY: concrete sensors
    if "betondruck" in names or "betonmenge" in names:
        if any("seilkraft hauptwinde" in n for n in names):
            return "KELLY"
        return "SOB"
    # KELLY without concrete (e.g., LignanoSabbiadoro)
    if any("seilkraft hauptwinde" in n for n in names):
        return "KELLY"
    # GRAB: generic grab
    if any("seilkraft" in n for n in names):
        return "GRAB"
    return "UNKNOWN"


# ── Sign anchor analysis ─────────────────────────────────────────────────────

def scan_sign_positions(dat_sections: list[str], data_length: int) -> dict[int, float]:
    """Scan data lines for +/- positions. Returns {position: frequency}."""
    counts = Counter()
    n_lines = 0
    for section in dat_sections:
        data = section[3:]
        if len(data) != data_length:
            continue
        n_lines += 1
        for pos, ch in enumerate(data):
            if ch in "+-":
                counts[pos] += 1
    if n_lines == 0:
        return {}
    return {pos: count / n_lines for pos, count in counts.items()}


def map_anchors_to_fields(
    sign_freqs: dict[int, float],
    field_defs: list[DatFieldDef],
    current_widths: list[int],
    min_freq: float = 0.95,
) -> dict[int, int]:
    """Map high-frequency sign positions to signed field indices.
    Returns {field_index: sign_position}.
    """
    reliable = sorted(pos for pos, freq in sign_freqs.items() if freq >= min_freq)
    signed_indices = [i for i, f in enumerate(field_defs) if f.signed and not f.is_direction]

    cumulative = []
    pos = 0
    for w in current_widths:
        cumulative.append(pos)
        pos += w

    mapping = {}
    used_anchors = set()
    for idx in signed_indices:
        expected_pos = cumulative[idx]
        best_anchor = None
        best_dist = float("inf")
        for anchor in reliable:
            if anchor in used_anchors:
                continue
            dist = abs(anchor - expected_pos)
            if dist < best_dist and dist <= 8:
                best_dist = dist
                best_anchor = anchor
        if best_anchor is not None:
            mapping[idx] = best_anchor
            used_anchors.add(best_anchor)
    return mapping


def compute_corrected_widths(
    field_defs: list[DatFieldDef],
    current_widths: list[int],
    anchor_mapping: dict[int, int],
    data_length: int,
) -> list[int] | None:
    """Compute corrected field widths using sign anchors as fixed points.

    Strategy: within each segment between anchors, start with meta[2]+sign
    allocation and adjust to match segment width by adding/subtracting from
    largest fields.
    """
    n = len(field_defs)
    anchored = sorted(anchor_mapping.items())

    boundary_points = [(0, 0)]
    for field_idx, sign_pos in anchored:
        boundary_points.append((sign_pos, field_idx))
    boundary_points.append((data_length, n))

    corrected = list(current_widths)

    for seg_i in range(len(boundary_points) - 1):
        seg_start_pos, seg_start_field = boundary_points[seg_i]
        seg_end_pos, seg_end_field = boundary_points[seg_i + 1]
        seg_width = seg_end_pos - seg_start_pos

        fields_in_seg = list(range(seg_start_field, seg_end_field))
        if not fields_in_seg:
            continue

        # Use meta[2]+sign as base allocation (same as current parser)
        base = []
        for fi in fields_in_seg:
            fdef = field_defs[fi]
            if fdef.is_direction:
                base.append(1)
            else:
                m2 = fdef.meta[2] if len(fdef.meta) > 2 else 3
                base.append(m2 + (1 if fdef.signed else 0))

        diff = seg_width - sum(base)

        if diff == 0:
            for i, fi in enumerate(fields_in_seg):
                corrected[fi] = base[i]
        else:
            # Adjust: add to or subtract from fields
            allocated = list(base)
            step = 1 if diff > 0 else -1

            # Sort indices by size (largest first if adding, smallest non-1 first if subtracting)
            if diff > 0:
                # Add to largest unsigned fields (they're most likely to need extra chars)
                adjustable = sorted(
                    range(len(allocated)),
                    key=lambda i: allocated[i],
                    reverse=True,
                )
            else:
                # Subtract from largest fields that can spare it (min width = 1)
                adjustable = sorted(
                    [i for i in range(len(allocated)) if allocated[i] > 1],
                    key=lambda i: allocated[i],
                    reverse=True,
                )

            for j in range(abs(diff)):
                if not adjustable:
                    break
                idx = adjustable[j % len(adjustable)]
                allocated[idx] += step
                # Ensure minimum width of 1
                if allocated[idx] < 1:
                    allocated[idx] = 1

            for i, fi in enumerate(fields_in_seg):
                corrected[fi] = allocated[i]

    if sum(corrected) != data_length:
        return None
    return corrected


# ── Raw extraction ────────────────────────────────────────────────────────────

def extract_raw_with_widths(
    dat_sections: list[str],
    field_defs: list[DatFieldDef],
    widths: list[int],
    data_length: int,
) -> dict[str, float]:
    """Extract raw integers using given widths. Returns {sensor: median_raw}."""
    num_fields = len(field_defs)
    all_raw: list[list[int | None]] = [[] for _ in range(num_fields)]

    for section in dat_sections:
        data = section[3:]
        if len(data) != data_length:
            continue
        pos = 0
        for i, (fdef, w) in enumerate(zip(field_defs, widths)):
            if pos + w > len(data):
                break
            raw_str = data[pos:pos + w]
            pos += w
            if fdef.is_direction:
                all_raw[i].append(None)
            else:
                try:
                    all_raw[i].append(int(raw_str))
                except ValueError:
                    all_raw[i].append(None)

    result = {}
    for i, fdef in enumerate(field_defs):
        values = [v for v in all_raw[i] if v is not None and v != 0]
        if values:
            result[fdef.name] = statistics.median(values)
    return result


def extract_raw_samples(
    dat_sections: list[str],
    field_defs: list[DatFieldDef],
    widths: list[int],
    data_length: int,
    max_lines: int = 5,
) -> dict[str, list[str]]:
    """Extract raw string samples per field (for debugging)."""
    samples: dict[str, list[str]] = {f.name: [] for f in field_defs}
    count = 0
    for section in dat_sections:
        data = section[3:]
        if len(data) != data_length:
            continue
        pos = 0
        for i, (fdef, w) in enumerate(zip(field_defs, widths)):
            if pos + w > len(data):
                break
            samples[fdef.name].append(data[pos:pos + w])
            pos += w
        count += 1
        if count >= max_lines:
            break
    return samples


def count_parse_failures(
    dat_sections: list[str],
    field_defs: list[DatFieldDef],
    widths: list[int],
    data_length: int,
) -> dict[str, float]:
    """Count parse failure rate per field (what fraction can't be parsed as int)."""
    num_fields = len(field_defs)
    total = 0
    failures = [0] * num_fields

    for section in dat_sections[:1000]:
        data = section[3:]
        if len(data) != data_length:
            continue
        total += 1
        pos = 0
        for i, (fdef, w) in enumerate(zip(field_defs, widths)):
            if pos + w > len(data):
                break
            raw_str = data[pos:pos + w]
            pos += w
            if not fdef.is_direction:
                try:
                    int(raw_str)
                except ValueError:
                    failures[i] += 1

    if total == 0:
        return {}
    return {field_defs[i].name: failures[i] / total for i in range(num_fields)}


# ── JSON ground truth ─────────────────────────────────────────────────────────

def load_json_ground_truth(json_dir: Path, technique: str, max_files: int = 30) -> dict[str, float]:
    """Load JSON ground truth medians for a specific technique."""
    sensor_values: dict[str, list[float]] = defaultdict(list)
    json_files = sorted(json_dir.rglob("*.json"))
    used = 0
    for jpath in json_files:
        try:
            with open(jpath) as f:
                d = json.load(f)
        except Exception:
            continue
        if d.get("technique") != technique:
            continue
        tsb = d.get("timeSeriesBlock", {})
        for item in tsb.get("serialValuesFree", []):
            name = item.get("seriesName", "")
            if not name:
                continue
            values = item.get("values", [])
            nonzero = [v for v in values if isinstance(v, (int, float)) and v != 0.0]
            if nonzero:
                sensor_values[name].append(statistics.median(nonzero))
        used += 1
        if used >= max_files:
            break
    return {name: statistics.median(meds) for name, meds in sensor_values.items() if meds}


# ── Spec lookup ───────────────────────────────────────────────────────────────

def get_spec_info(sensor_name: str, spec_keys: list[str], specs: dict) -> dict | None:
    for key in spec_keys:
        if key not in specs:
            continue
        sensors = specs[key].get("sensors", {})
        if sensor_name in sensors:
            return sensors[sensor_name]
        for sn, info in sensors.items():
            if normalize_spec_name(sn) == sensor_name:
                return info
    return None


def log_error(val: float | None, ref: float | None) -> float | None:
    if val is None or ref is None:
        return None
    a, r = abs(val), abs(ref)
    if a == 0 or r == 0:
        return None
    try:
        return abs(math.log10(a / r))
    except (ValueError, ZeroDivisionError):
        return None


# ── Main analysis ─────────────────────────────────────────────────────────────

def analyze_machine(
    dat_files: list[Path],
    json_dir: Path,
    machine_slug: str,
    specs: dict,
    corrections: dict,
    max_dat_files: int = 20,
) -> list[str]:
    """Full sign-anchor analysis across multiple DAT files for one machine."""
    lines = []

    # Parse first file to get header structure
    ref_path = None
    ref_field_defs = None
    ref_data_length = None
    all_dat_sections = []

    for dpath in dat_files[:max_dat_files]:
        try:
            with open(dpath, "rb") as f:
                raw = f.read()
            text = raw.decode("iso-8859-1")
        except Exception:
            continue

        sections = text.split("$")
        field_defs = _parse_header(sections[0])
        if not field_defs:
            continue

        dat_secs = [s for s in sections if s.startswith("DAT")]
        if not dat_secs:
            continue

        dl = len(dat_secs[0]) - 3

        if ref_field_defs is None:
            ref_path = dpath
            ref_field_defs = field_defs
            ref_data_length = dl

        # Only merge files with same structure
        if dl == ref_data_length and len(field_defs) == len(ref_field_defs):
            all_dat_sections.extend(dat_secs)

    if ref_field_defs is None:
        return ["ERROR: No parseable DAT files"]

    field_defs = ref_field_defs
    data_length = ref_data_length

    current_widths = _estimate_field_widths(field_defs, data_length)
    if not current_widths:
        return ["ERROR: Could not estimate field widths"]

    technique = infer_technique(field_defs)

    lines.append(f"Machine: {machine_slug}, Technique: {technique}")
    lines.append(f"Fields: {len(field_defs)}, Data length: {data_length}")
    lines.append(f"DAT files used: {min(len(dat_files), max_dat_files)}")
    lines.append(f"Total DAT sections: {len(all_dat_sections)}")
    lines.append("")

    # ── Step 1: Sign position analysis ──
    lines.append("=" * 80)
    lines.append("STEP 1: SIGN POSITION ANALYSIS")
    lines.append("=" * 80)

    sign_freqs = scan_sign_positions(all_dat_sections, data_length)

    reliable = [(pos, sign_freqs[pos]) for pos in sorted(sign_freqs) if sign_freqs[pos] >= 0.95]
    semi = [(pos, sign_freqs[pos]) for pos in sorted(sign_freqs) if 0.50 <= sign_freqs[pos] < 0.95]

    lines.append(f"Total positions with signs: {len(sign_freqs)}")
    lines.append(f"Reliable (>95%): {len(reliable)}")
    for pos, freq in reliable:
        lines.append(f"  Position {pos:>3d}: {freq:.1%}")
    if semi:
        lines.append(f"Semi-reliable (50-95%): {len(semi)}")
        for pos, freq in semi:
            lines.append(f"  Position {pos:>3d}: {freq:.1%}")
    lines.append("")

    # ── Step 2: Map anchors & compute corrected widths ──
    lines.append("=" * 80)
    lines.append("STEP 2: MAP ANCHORS & CORRECT WIDTHS")
    lines.append("=" * 80)

    anchor_map = map_anchors_to_fields(sign_freqs, field_defs, current_widths, 0.95)
    # Also try with semi-reliable for more anchor points
    anchor_map_50 = map_anchors_to_fields(sign_freqs, field_defs, current_widths, 0.50)
    best_anchors = anchor_map_50 if len(anchor_map_50) > len(anchor_map) else anchor_map

    for fi, pos in sorted(best_anchors.items()):
        expected = sum(current_widths[:fi])
        lines.append(f"  Field {fi:>2d} ({field_defs[fi].name:<35s}): "
                     f"sign at {pos:>3d} (expected {expected:>3d}, delta={pos - expected:+d})")
    lines.append("")

    corrected_widths = compute_corrected_widths(field_defs, current_widths, best_anchors, data_length)

    if corrected_widths is None:
        lines.append("WARNING: Could not compute corrected widths, using current")
        corrected_widths = current_widths

    # Show width comparison
    lines.append(f"{'#':>2s} {'Field':<35s} {'meta2':>5s} {'CurW':>5s} {'CorrW':>5s} {'Delta':>5s}")
    lines.append("-" * 65)
    cum_cur = 0
    cum_corr = 0
    for i, (fdef, cw, ow) in enumerate(zip(field_defs, corrected_widths, current_widths)):
        m2 = fdef.meta[2] if len(fdef.meta) > 2 else "?"
        delta = cw - ow
        delta_str = f"{delta:+d}" if delta != 0 else ""
        lines.append(f"{i:>2d} {fdef.name:<35s} {str(m2):>5s} {ow:>5d} {cw:>5d} {delta_str:>5s}")
        cum_cur += ow
        cum_corr += cw
    lines.append(f"   {'TOTAL':>35s} {'':>5s} {cum_cur:>5d} {cum_corr:>5d}  target={data_length}")
    lines.append("")

    # ── Step 3: Parse failure analysis (validates widths) ──
    lines.append("=" * 80)
    lines.append("STEP 3: PARSE FAILURE RATES (validates width correctness)")
    lines.append("=" * 80)

    fail_cur = count_parse_failures(all_dat_sections, field_defs, current_widths, data_length)
    fail_corr = count_parse_failures(all_dat_sections, field_defs, corrected_widths, data_length)

    lines.append(f"{'Field':<35s} {'CurFail':>8s} {'CorrFail':>8s} {'Better':>6s}")
    lines.append("-" * 65)
    cur_total_fail = 0
    corr_total_fail = 0
    for fdef in field_defs:
        if fdef.is_direction:
            continue
        cf = fail_cur.get(fdef.name, 0)
        rf = fail_corr.get(fdef.name, 0)
        cur_total_fail += cf
        corr_total_fail += rf
        better = "CUR" if cf < rf else ("CORR" if rf < cf else "")
        lines.append(f"{fdef.name:<35s} {cf:>7.1%} {rf:>7.1%} {better:>6s}")
    lines.append(f"{'TOTAL fail rate':>35s} {cur_total_fail / max(len(field_defs), 1):>7.1%} "
                 f"{corr_total_fail / max(len(field_defs), 1):>7.1%}")
    lines.append("")

    # ── Step 4: Extract raw values with both widths ──
    lines.append("=" * 80)
    lines.append("STEP 4: RAW VALUE COMPARISON")
    lines.append("=" * 80)

    raw_current = extract_raw_with_widths(all_dat_sections, field_defs, current_widths, data_length)
    raw_corrected = extract_raw_with_widths(all_dat_sections, field_defs, corrected_widths, data_length)

    # Samples for debugging
    samples_cur = extract_raw_samples(all_dat_sections, field_defs, current_widths, data_length, 3)
    samples_corr = extract_raw_samples(all_dat_sections, field_defs, corrected_widths, data_length, 3)

    lines.append(f"{'Field':<35s} {'CurRaw':>12s} {'CorrRaw':>12s} {'Changed':>7s}  Samples(cur|corr)")
    lines.append("-" * 110)
    for fdef in field_defs:
        if fdef.is_direction:
            continue
        cur = raw_current.get(fdef.name)
        corr = raw_corrected.get(fdef.name)
        changed = "YES" if cur != corr else ""
        cs = f"{cur:.1f}" if cur is not None else "N/A"
        rs = f"{corr:.1f}" if corr is not None else "N/A"
        sc = samples_cur.get(fdef.name, [])[:2]
        sr = samples_corr.get(fdef.name, [])[:2]
        lines.append(f"{fdef.name:<35s} {cs:>12s} {rs:>12s} {changed:>7s}  {sc} | {sr}")
    lines.append("")

    # ── Step 5: Load JSON & compare ──
    lines.append("=" * 80)
    lines.append("STEP 5: COMPARISON VS JSON GROUND TRUTH")
    lines.append("=" * 80)

    json_medians = load_json_ground_truth(json_dir, technique)
    lines.append(f"JSON ground truth: {len(json_medians)} sensors for technique={technique}")

    machine_map = specs.get("machine_technique_map", {})
    spec_keys = machine_map.get(machine_slug, [])
    machine_corrections = corrections.get(machine_slug, {})

    common = set(raw_current.keys()) & set(json_medians.keys())
    lines.append(f"Common sensors (DAT ∩ JSON): {len(common)}")
    lines.append("")

    lines.append("Approaches:")
    lines.append("  A: current_raw / header_divisor")
    lines.append("  B: current_raw / spec_divisor")
    lines.append("  D: current_raw / header_divisor / correction")
    lines.append("  E: corrected_raw / spec_divisor  *** THE KEY TEST ***")
    lines.append("  F: corrected_raw / header_divisor / correction")
    lines.append("")

    results = []

    for sensor_name in sorted(common):
        cur_raw = raw_current.get(sensor_name)
        corr_raw = raw_corrected.get(sensor_name)
        json_val = json_medians.get(sensor_name)
        if cur_raw is None or json_val is None:
            continue

        fdef = next((f for f in field_defs if f.name == sensor_name), None)
        if fdef is None or fdef.is_direction:
            continue

        header_div = fdef.divisor
        correction = machine_corrections.get(sensor_name, 1.0)
        spec = get_spec_info(sensor_name, spec_keys, specs)
        spec_div = spec["expected_divisor"] if spec else None

        a = cur_raw / header_div if header_div else None
        b = cur_raw / spec_div if spec_div else None
        d = cur_raw / header_div / correction if header_div and correction else None
        e = corr_raw / spec_div if corr_raw and spec_div else None
        f_val = corr_raw / header_div / correction if corr_raw and header_div and correction else None

        err_a = log_error(a, json_val)
        err_b = log_error(b, json_val)
        err_d = log_error(d, json_val)
        err_e = log_error(e, json_val)
        err_f = log_error(f_val, json_val)

        candidates = {}
        if err_a is not None: candidates["A"] = err_a
        if err_b is not None: candidates["B"] = err_b
        if err_d is not None: candidates["D"] = err_d
        if err_e is not None: candidates["E"] = err_e
        if err_f is not None: candidates["F"] = err_f

        if candidates:
            best = min(candidates.values())
            winner = "+".join(sorted(k for k, v in candidates.items() if abs(v - best) < 0.01))
        else:
            winner = "none"

        results.append({
            "sensor": sensor_name,
            "cur_raw": cur_raw, "corr_raw": corr_raw,
            "raw_changed": cur_raw != corr_raw,
            "json": json_val,
            "header_div": header_div, "spec_div": spec_div,
            "correction": correction,
            "A": a, "B": b, "D": d, "E": e, "F": f_val,
            "err_A": err_a, "err_B": err_b, "err_D": err_d,
            "err_E": err_e, "err_F": err_f,
            "winner": winner,
        })

        lines.append(f"  Sensor: {sensor_name}")
        lines.append(f"    CurRaw={cur_raw}, CorrRaw={corr_raw}"
                     f"{'  <-- CHANGED' if cur_raw != corr_raw else ''}")
        lines.append(f"    HdrDiv={header_div}, SpecDiv={spec_div}, Corr={correction}")
        lines.append(f"    JSON={json_val}")

        def fmt(label, val, err):
            if val is None:
                return f"      {label}: N/A"
            marker = " <-- WINNER" if label.strip()[0] in winner else ""
            e_str = f"err={err:.3f}" if err is not None else "err=N/A"
            return f"      {label}: {val:>14.4f}  {e_str}{marker}"

        lines.append(fmt("A (cur/hdr)      ", a, err_a))
        lines.append(fmt("B (cur/spec)     ", b, err_b))
        lines.append(fmt("D (cur/hdr/corr) ", d, err_d))
        lines.append(fmt("E (corr/spec) ** ", e, err_e))
        lines.append(fmt("F (corr/hdr/corr)", f_val, err_f))
        lines.append("")

    # ── Summary ──
    lines.append("=" * 80)
    lines.append("SUMMARY")
    lines.append("=" * 80)

    total = len(results)
    if total == 0:
        lines.append("No comparable sensors found.")
        return lines

    n_changed = sum(1 for r in results if r["raw_changed"])
    lines.append(f"Sensors compared: {total}")
    lines.append(f"Raw values changed by correction: {n_changed} / {total}")
    lines.append("")

    # Winner counts
    wc = Counter(r["winner"] for r in results)
    lines.append("Winner breakdown:")
    for w, cnt in sorted(wc.items(), key=lambda x: -x[1]):
        lines.append(f"  {w:>15s}: {cnt}")
    lines.append("")

    e_wins = sum(1 for r in results if "E" in r["winner"])
    d_wins = sum(1 for r in results if "D" in r["winner"] and "E" not in r["winner"])
    b_wins = sum(1 for r in results if "B" in r["winner"] and "E" not in r["winner"])

    lines.append(f"E wins (corrected raw + spec div):   {e_wins:>3d} / {total}")
    lines.append(f"D wins (current raw + correction):   {d_wins:>3d} / {total}")
    lines.append(f"B wins (current raw + spec div):     {b_wins:>3d} / {total}")
    lines.append("")

    T = 0.5
    good_e = sum(1 for r in results if r["err_E"] is not None and r["err_E"] < T)
    good_d = sum(1 for r in results if r["err_D"] is not None and r["err_D"] < T)
    lines.append(f"E within {10**T:.1f}x of JSON: {good_e} / {total}")
    lines.append(f"D within {10**T:.1f}x of JSON: {good_d} / {total}")
    lines.append("")

    # Detail table
    lines.append(f"{'Sensor':<35s} {'CurRaw':>10s} {'CorrRaw':>10s} {'Chg':>3s} "
                 f"{'JSON':>10s} {'SpecDiv':>7s} {'Corr':>8s} "
                 f"{'ErrE':>6s} {'ErrD':>6s} {'Winner':>10s}")
    lines.append("-" * 115)
    for r in sorted(results, key=lambda x: x["sensor"]):
        cr = f"{r['cur_raw']:.0f}" if r['cur_raw'] else "-"
        rr = f"{r['corr_raw']:.0f}" if r['corr_raw'] else "-"
        chg = "Y" if r["raw_changed"] else ""
        jv = f"{r['json']:.3f}"
        sd = str(r['spec_div']) if r['spec_div'] else "-"
        co = f"{r['correction']}" if r['correction'] != 1.0 else "-"
        ee = f"{r['err_E']:.3f}" if r['err_E'] is not None else "-"
        ed = f"{r['err_D']:.3f}" if r['err_D'] is not None else "-"
        lines.append(f"  {r['sensor']:<33s} {cr:>10s} {rr:>10s} {chg:>3s} "
                     f"{jv:>10s} {sd:>7s} {co:>8s} "
                     f"{ee:>6s} {ed:>6s} {r['winner']:>10s}")

    return lines


def main():
    print("Loading specs and corrections...")
    specs = load_specs()
    corrections = load_corrections()

    all_lines = ["SIGN-ANCHOR FIELD WIDTH ANALYSIS", "=" * 80, "",
                 "Goal: determine if field width misestimation is why spec divisors",
                 "appeared unreliable in the previous investigation.", "",
                 "Key test: Approach E (corrected widths + spec divisor) vs",
                 "Approach D (current widths + empirical correction)", ""]

    # Test cases
    test_cases = [
        ("LignanoSabbiadoro", "bg33v_5610", "KELLY"),
        ("Catania", "gb50_601", "GRAB"),
        ("VICENZA", "bg33v_5610", "SCM"),
    ]

    global_results_e = 0
    global_results_d = 0
    global_results_total = 0

    for project_pattern, machine_slug, expected_tech in test_cases:
        project_dir = None
        for d in sorted(RAW_DIR.iterdir()):
            if d.is_dir() and project_pattern in d.name:
                project_dir = d
                break

        if not project_dir:
            continue

        dat_files = sorted(project_dir.rglob("*.dat"))
        if not dat_files:
            continue

        print(f"\nAnalyzing {project_pattern} ({machine_slug}, expected={expected_tech})...")
        print(f"  {len(dat_files)} DAT files found")

        all_lines.extend(["~" * 80,
                          f"TEST: {machine_slug} {expected_tech} ({project_pattern})",
                          "~" * 80, ""])

        result_lines = analyze_machine(
            dat_files, project_dir, machine_slug, specs, corrections, max_dat_files=30
        )
        all_lines.extend(result_lines)
        all_lines.append("")

        # Extract counts from the summary
        for line in result_lines:
            if "E wins" in line:
                try:
                    n = int(line.split(":")[1].strip().split("/")[0])
                    global_results_e += n
                except Exception:
                    pass
            elif "D wins" in line and "E" not in line.split("D wins")[0]:
                try:
                    n = int(line.split(":")[1].strip().split("/")[0])
                    global_results_d += n
                except Exception:
                    pass
            elif "Sensors compared:" in line:
                try:
                    n = int(line.split(":")[1].strip())
                    global_results_total += n
                except Exception:
                    pass

    # ── Global conclusion ──
    all_lines.extend(["", "=" * 80, "GLOBAL CONCLUSION", "=" * 80, ""])
    all_lines.append(f"Across all machines/techniques:")
    all_lines.append(f"  E wins (corrected + spec): {global_results_e}")
    all_lines.append(f"  D wins (current + corr):   {global_results_d}")
    all_lines.append(f"  Total sensors:             {global_results_total}")
    all_lines.append("")

    if global_results_total > 0:
        e_pct = 100 * global_results_e / global_results_total
        d_pct = 100 * global_results_d / global_results_total
        if e_pct > d_pct:
            all_lines.append(f"CONCLUSION: Corrected widths + spec divisor (E) wins {e_pct:.0f}%")
            all_lines.append("of sensors. The previous conclusion was WRONG — spec-driven")
            all_lines.append("parsing CAN work if field widths are correctly estimated.")
        elif e_pct > 20:
            all_lines.append(f"CONCLUSION: Spec helps for {e_pct:.0f}% of sensors, but")
            all_lines.append(f"empirical corrections still needed for {d_pct:.0f}%.")
            all_lines.append("The truth is mixed: width correction helps SOME sensors,")
            all_lines.append("but firmware genuinely encodes some sensors differently than spec.")
        else:
            all_lines.append(f"CONCLUSION: Even with corrected widths, spec divisors only win")
            all_lines.append(f"{e_pct:.0f}% of sensors. Empirical corrections remain essential.")

    # Write
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "sign_anchor_analysis.txt"
    with open(output_path, "w") as f:
        f.write("\n".join(all_lines) + "\n")

    print(f"\nReport: {output_path}")
    for line in all_lines[-20:]:
        print(line)


if __name__ == "__main__":
    main()
