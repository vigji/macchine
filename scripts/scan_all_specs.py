"""Scan ALL DAT files and validate spec matching for pedantic parsing.

For each unique (machine, project, technique) combination:
  1. Parse header + BEGINNG to identify technique
  2. Try ALL available spec versions (not just mapped ones)
  3. Validate via sign-anchor + width matching
  4. Report matches and gaps

Usage:
    uv run python scripts/scan_all_specs.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from macchine.parsers.beginng_parser import parse_beginng
from macchine.parsers.dat_parser import _parse_header

SPECS_PATH = Path(__file__).resolve().parent.parent / "specs_data" / "medef_specs.yaml"
RAW_DIR = Path(__file__).resolve().parent.parent / "output" / "raw_downloads"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "diagnostics"


def load_specs():
    with open(SPECS_PATH) as f:
        return yaml.safe_load(f)


def normalize_spec_name(name: str) -> str:
    return name.replace("\ufb00", "ff").replace("\ufb01", "fi").replace("\ufb02", "fl")


def spec_field_width(spec_entry: dict) -> int:
    chars = spec_entry["characters"]
    dec = spec_entry["decimal"]
    signed = spec_entry.get("signed", False)
    return (1 if signed else 0) + chars + dec


def find_sensor_in_spec(sensor_name: str, spec: dict) -> dict | None:
    sensors = spec.get("sensors", {})
    if sensor_name in sensors:
        return dict(sensors[sensor_name])
    for spec_sensor, spec_info in sensors.items():
        if normalize_spec_name(spec_sensor) == normalize_spec_name(sensor_name):
            return dict(spec_info)
    return None


def scan_sign_positions(data_lines: list[str]) -> dict[int, float]:
    n = len(data_lines)
    if n == 0:
        return {}
    pos_counts = defaultdict(int)
    for line in data_lines:
        for i, c in enumerate(line):
            if c in "+-":
                pos_counts[i] += 1
    return {pos: count / n for pos, count in sorted(pos_counts.items())}


def validate_spec(field_defs: list, spec: dict, data_lines: list[str],
                  data_length: int) -> tuple[float, str, list[int] | None]:
    """Validate a spec against data. Returns (score, reason, widths)."""
    sign_freq = scan_sign_positions(data_lines[:500])
    anchors = {pos for pos, freq in sign_freq.items() if freq > 0.5}

    # Build spec widths for header fields
    widths = []
    spec_sensors_used = set()
    for fdef in field_defs:
        spec_entry = find_sensor_in_spec(fdef.name, spec)
        if spec_entry:
            w = spec_field_width(spec_entry)
            widths.append(w)
            spec_sensors_used.add(fdef.name)
            for sname in spec.get("sensors", {}):
                if normalize_spec_name(sname) == normalize_spec_name(fdef.name):
                    spec_sensors_used.add(sname)
        elif fdef.is_direction:
            widths.append(1)
        else:
            widths.append(None)

    # Trailing spec fields (in spec but not in header)
    trailing_width = 0
    trailing_fields = []
    for sname, sinfo in spec.get("sensors", {}).items():
        norm = normalize_spec_name(sname)
        if sname not in spec_sensors_used and norm not in {normalize_spec_name(n) for n in spec_sensors_used}:
            w = spec_field_width(sinfo)
            trailing_width += w
            trailing_fields.append((sname, w))

    known_total = sum(w for w in widths if w is not None) + trailing_width
    unknown_count = sum(1 for w in widths if w is None)

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
        else:
            for i in range(len(widths)):
                if widths[i] is None:
                    widths[i] = 0

    total = sum(widths) + trailing_width
    if total != data_length:
        return 0.0, f"WIDTH {total} != {data_length}", None

    # Check sign anchors
    pos = 0
    expected_sign = set()
    for fdef, w in zip(field_defs, widths):
        se = find_sensor_in_spec(fdef.name, spec)
        if se and se.get("signed", False):
            expected_sign.add(pos)
        pos += w
    for sname, w in trailing_fields:
        sinfo = spec["sensors"].get(sname, {})
        if sinfo.get("signed", False):
            expected_sign.add(pos)
        pos += w

    if anchors:
        matched = anchors & expected_sign
        unexpected = anchors - expected_sign
        precision = len(matched) / len(anchors)
        score = precision * min(1.0, len(anchors) / 3)
    else:
        score = 0.5  # no anchors to validate — width match only
        precision = 1.0
        unexpected = set()

    n_in_spec = sum(1 for f in field_defs if find_sensor_in_spec(f.name, spec) is not None)
    coverage = n_in_spec / len(field_defs) if field_defs else 0

    reason = (f"width={total} anchors={len(anchors)} "
              f"precision={precision:.0%} coverage={coverage:.0%}")
    if unexpected:
        reason += f" UNEXPECTED={sorted(unexpected)}"

    return score, reason, widths


def parse_dat_file(dat_path: Path):
    """Parse a single DAT file and return header info + data lines."""
    with open(dat_path, "rb") as f:
        raw = f.read()
    text = raw.decode("iso-8859-1")
    sections = text.split("$")

    field_defs = _parse_header(sections[0])

    # Get technique from BEGINNG
    technique = None
    technique_raw = None
    for sec in sections:
        if sec.startswith("BEGINN"):
            info = parse_beginng(sec)
            technique = info.get("technique")
            technique_raw = info.get("technique_raw")
            break

    # Get data lines
    data_lines = []
    data_length = None
    for sec in sections:
        if sec.startswith("DAT"):
            dl = sec[3:]
            if data_length is None:
                data_length = len(dl)
            if len(dl) == data_length:
                data_lines.append(dl)

    return field_defs, technique, technique_raw, data_lines, data_length


def scan_project(project_dir: Path, specs: dict) -> list[dict]:
    """Scan a project directory for all DAT files and validate specs."""
    results = []

    # Find all DAT files (in subdirs or top-level)
    dat_files = sorted(project_dir.rglob("*.dat"))
    if not dat_files:
        return results

    # Group by machine slug (from filename)
    machine_groups = defaultdict(list)
    for df in dat_files:
        # Filename pattern: SERIAL_machineslug_DATE_...
        parts = df.stem.split("_")
        if len(parts) >= 2:
            # Try to find machine slug — it's usually part 2 (after serial)
            # Pattern: 01K00044171_bg33v_5610_... or bg33v_5610_...
            slug = None
            for i in range(len(parts) - 1):
                candidate = f"{parts[i]}_{parts[i+1]}"
                # Machine slugs look like: bg33v_5610, gb50_601, mc86_621, etc.
                if any(c.isalpha() for c in parts[i]) and parts[i+1].isdigit():
                    slug = candidate
                    break
            if slug is None:
                slug = "unknown"
            machine_groups[slug].append(df)
        else:
            machine_groups["unknown"].append(df)

    project_name = project_dir.name

    for machine_slug, dat_paths in sorted(machine_groups.items()):
        # Sample a few files
        sample = dat_paths[:5]

        # Parse first file for header
        try:
            field_defs, technique, technique_raw, data_lines, data_length = \
                parse_dat_file(sample[0])
        except Exception as e:
            results.append({
                "project": project_name,
                "machine": machine_slug,
                "n_dat": len(dat_paths),
                "error": str(e),
            })
            continue

        if not field_defs or data_length is None:
            results.append({
                "project": project_name,
                "machine": machine_slug,
                "n_dat": len(dat_paths),
                "error": "No header or data sections",
            })
            continue

        # Collect more data lines from additional files
        for dp in sample[1:]:
            try:
                _, _, _, extra_lines, dl = parse_dat_file(dp)
                if dl == data_length:
                    data_lines.extend(extra_lines)
            except Exception:
                continue

        # Try ALL spec keys
        best_score = 0
        best_key = None
        best_reason = ""
        best_widths = None
        all_attempts = []

        for spec_key, spec_data in sorted(specs.items()):
            if spec_key == "machine_technique_map":
                continue
            if not isinstance(spec_data, dict) or "sensors" not in spec_data:
                continue

            score, reason, widths = validate_spec(
                field_defs, spec_data, data_lines, data_length
            )
            all_attempts.append((spec_key, score, reason))

            if score > best_score:
                best_score = score
                best_key = spec_key
                best_reason = reason
                best_widths = widths

        # Also show width-matching candidates (even if anchor score is low)
        width_matches = [(k, s, r) for k, s, r in all_attempts
                         if "WIDTH" not in r]

        results.append({
            "project": project_name,
            "machine": machine_slug,
            "n_dat": len(dat_paths),
            "n_fields": len(field_defs),
            "data_length": data_length,
            "technique": technique,
            "technique_raw": technique_raw,
            "n_data_lines": len(data_lines),
            "best_spec": best_key,
            "best_score": best_score,
            "best_reason": best_reason,
            "width_matches": width_matches,
            "field_names": [f.name for f in field_defs if not f.is_direction],
        })

    return results


def main():
    specs = load_specs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "spec_scan_all.txt"

    all_results = []

    with open(out_path, "w") as out:
        out.write("COMPREHENSIVE SPEC SCAN — ALL DAT FILES\n")
        out.write("=" * 90 + "\n\n")
        out.write("For each (project, machine): try ALL spec versions, report best match.\n")
        out.write("Score 1.0 = perfect match (all sign anchors valid, width correct).\n")
        out.write("Score 0.0 = no match.\n\n")

        for project_dir in sorted(RAW_DIR.iterdir()):
            if not project_dir.is_dir():
                continue

            results = scan_project(project_dir, specs)
            if not results:
                continue

            all_results.extend(results)

            out.write(f"\n{'#' * 90}\n")
            out.write(f"# PROJECT: {project_dir.name}\n")
            out.write(f"{'#' * 90}\n\n")

            for r in results:
                if "error" in r:
                    out.write(f"  {r['machine']}: ERROR — {r['error']}\n\n")
                    continue

                status = ""
                if r["best_score"] >= 0.8:
                    status = "MATCH"
                elif r["best_score"] > 0:
                    status = "PARTIAL"
                else:
                    status = "NO MATCH"

                out.write(f"  Machine: {r['machine']}\n")
                out.write(f"    DAT files: {r['n_dat']}, "
                          f"fields: {r['n_fields']}, "
                          f"data_length: {r['data_length']}, "
                          f"technique: {r['technique']} "
                          f"(raw: {r['technique_raw']})\n")
                out.write(f"    Data lines sampled: {r['n_data_lines']}\n")
                out.write(f"    Best spec: {r['best_spec']} "
                          f"(score={r['best_score']:.2f}) — {status}\n")
                out.write(f"    Detail: {r['best_reason']}\n")

                if r["width_matches"]:
                    out.write(f"    Width-matching specs:\n")
                    for k, s, reason in sorted(r["width_matches"],
                                               key=lambda x: -x[1]):
                        out.write(f"      {k:<25s}  score={s:.2f}  {reason}\n")
                else:
                    out.write(f"    NO SPEC MATCHES DATA WIDTH ({r['data_length']} chars)\n")

                # Show header field names (for diagnosis)
                out.write(f"    Header sensors: {', '.join(r['field_names'][:10])}")
                if len(r["field_names"]) > 10:
                    out.write(f" ... (+{len(r['field_names'])-10} more)")
                out.write("\n\n")

        # GLOBAL SUMMARY
        out.write(f"\n\n{'=' * 90}\n")
        out.write("GLOBAL SUMMARY\n")
        out.write(f"{'=' * 90}\n\n")

        valid = [r for r in all_results if "error" not in r]
        matched = [r for r in valid if r["best_score"] >= 0.8]
        partial = [r for r in valid if 0 < r["best_score"] < 0.8]
        no_match = [r for r in valid if r["best_score"] == 0]

        out.write(f"Total (project, machine) combinations: {len(all_results)}\n")
        out.write(f"  MATCH (score >= 0.8):    {len(matched)}\n")
        out.write(f"  PARTIAL (0 < score < 0.8): {len(partial)}\n")
        out.write(f"  NO MATCH (score = 0):    {len(no_match)}\n\n")

        if matched:
            out.write("MATCHED:\n")
            for r in sorted(matched, key=lambda x: (x["machine"], x["project"])):
                out.write(f"  {r['machine']:<20s}  {r['project']:<35s}  "
                          f"→ {r['best_spec']:<20s}  score={r['best_score']:.2f}  "
                          f"({r['n_dat']} files)\n")
            out.write("\n")

        if partial:
            out.write("PARTIAL MATCHES (wrong spec, but width happens to fit):\n")
            for r in sorted(partial, key=lambda x: (x["machine"], x["project"])):
                out.write(f"  {r['machine']:<20s}  {r['project']:<35s}  "
                          f"→ {r['best_spec']:<20s}  score={r['best_score']:.2f}  "
                          f"technique={r['technique']}  data_len={r['data_length']}  "
                          f"({r['n_dat']} files)\n")
            out.write("\n")

        if no_match:
            out.write("NO MATCH — NEED NEW SPECS:\n")
            for r in sorted(no_match, key=lambda x: (x["machine"], x["project"])):
                wm = r.get("width_matches", [])
                wm_str = f"  (width-matching: {', '.join(k for k, _, _ in wm)})" if wm else ""
                out.write(f"  {r['machine']:<20s}  {r['project']:<35s}  "
                          f"technique={r['technique']}  data_len={r['data_length']}  "
                          f"fields={r['n_fields']}  ({r['n_dat']} files){wm_str}\n")
            out.write("\n")

        # Suggestions for missing specs
        if no_match or partial:
            out.write("\nSUGGESTED SPECS TO DOWNLOAD:\n")
            out.write("-" * 60 + "\n")
            seen = set()
            for r in no_match + partial:
                key = (r["machine"], r["technique"], r["data_length"])
                if key in seen:
                    continue
                seen.add(key)
                tech = r["technique"] or r.get("technique_raw", "?")
                out.write(f"\n  Machine: {r['machine']}, "
                          f"technique: {tech} (raw: {r.get('technique_raw', '?')}), "
                          f"data_length: {r['data_length']}, "
                          f"fields: {r['n_fields']}\n")
                # Suggest based on technique
                suggestions = _suggest_spec(tech, r)
                for s in suggestions:
                    out.write(f"    → Try: {s}\n")

    print(f"Wrote {out_path}")
    print(f"Total: {len(all_results)} combos, "
          f"{len(matched)} matched, {len(partial)} partial, {len(no_match)} no match")


def _suggest_spec(technique: str, result: dict) -> list[str]:
    """Suggest specs to download based on technique and field names."""
    suggestions = []
    tech_upper = (technique or "").upper()
    fields = result.get("field_names", [])

    if tech_upper in ("KELLY", ""):
        suggestions.append("KELLY [G] (latest version)")
    if tech_upper == "SOB" or tech_upper == "CFA":
        suggestions.append("SOB [1] (latest version)")
        suggestions.append("SOBmHW [1] (SOB with main winch)")
    if tech_upper == "GRAB" or any("Seilkraft Winde" in f or "Seilkraft" == f for f in fields):
        suggestions.append("Seilgreifer [H] (latest version)")
    if tech_upper in ("CUT", "SCHLITZ"):
        suggestions.append("CUT [9] (latest version)")
    if tech_upper == "SCM" or any("Susp." in f for f in fields):
        suggestions.append("CSM [C] or CSM [5] (latest version)")
        suggestions.append("SCM [5] or SCM [A]")
    if tech_upper == "DMS":
        suggestions.append("DKS [1] or DKSmBTM [1] (Diaphragm wall)")
    if tech_upper == "FREE":
        suggestions.append("Check BEGINNG raw — may be any technique")
    if any("Fräs" in f or "FRL" in f or "FRR" in f for f in fields):
        suggestions.append("Fräse HDS [9] (latest cutter version)")
        suggestions.append("Fräse HSS [9] or Fräse HTS [9]")
    if any("FDP" in f or "Durchf" in f for f in fields):
        suggestions.append("FDP [1] or FDPmHW [1]")

    if not suggestions:
        suggestions.append(f"Unknown technique '{technique}' — check available spec list manually")

    return suggestions


if __name__ == "__main__":
    main()
