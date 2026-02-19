"""Hybrid DAT parsing: header shifted-meta widths + MEDEF spec divisors.

KEY DISCOVERY: meta[2] of field N contains the TOTAL data width (chars + decimal +
sign) of field N+1 — i.e., the metadata is shifted by one field position.

This means:
  - width of field 0 = data_length - sum(ALL meta[2] values)
  - width of field i (i>0) = meta[2] of field i-1
  - meta[2] of last field = width of trailing spec field (not in header)

Two approaches tested:
  A) pure_spec:    widths from MEDEF spec (chars + decimal + sign)
  B) shifted_meta: width[i] = meta[2] of field[i-1], width[0] = remainder

All approaches use spec divisor (10^decimal) to convert raw integers to physical values.

Usage:
    uv run python scripts/hybrid_spec_parse.py
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
RAW_DL = Path(__file__).resolve().parent.parent / "output" / "raw_downloads"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "diagnostics"


def load_specs():
    with open(SPECS_PATH) as f:
        return yaml.safe_load(f)


def load_corrections():
    with open(CORRECTIONS_PATH) as f:
        return yaml.safe_load(f) or {}


def normalize_spec_name(name: str) -> str:
    return name.replace("\ufb00", "ff").replace("\ufb01", "fi").replace("\ufb02", "fl")


def find_sensor_in_spec(sensor_name: str, spec: dict) -> dict | None:
    sensors = spec.get("sensors", {})
    if sensor_name in sensors:
        return dict(sensors[sensor_name])
    for spec_sensor, spec_info in sensors.items():
        if normalize_spec_name(spec_sensor) == normalize_spec_name(sensor_name):
            return dict(spec_info)
    return None


def spec_field_width(spec_entry: dict) -> int:
    return (1 if spec_entry.get("signed", False) else 0) + spec_entry["characters"] + spec_entry["decimal"]


def read_dat_file(path: Path) -> tuple[list, list[str], int]:
    with open(path, "rb") as f:
        raw = f.read()
    text = raw.decode("iso-8859-1")
    sections = text.split("$")
    field_defs = _parse_header(sections[0])

    data_lines = []
    data_length = 0
    for s in sections:
        if s.startswith("DAT"):
            dl = s[3:]
            if data_length == 0:
                data_length = len(dl)
            if len(dl) == data_length:
                data_lines.append(dl)
    return field_defs, data_lines, data_length


def scan_sign_positions(data_lines: list[str]) -> dict[int, float]:
    n = min(len(data_lines), 500)
    if n == 0:
        return {}
    pos_counts = defaultdict(int)
    for line in data_lines[:n]:
        for i, c in enumerate(line):
            if c in "+-":
                pos_counts[i] += 1
    return {pos: count / n for pos, count in sorted(pos_counts.items())}


def compute_shifted_meta_widths(field_defs, data_length):
    """Width[i] = meta[2] of field[i-1]. Width[0] = remainder.

    meta[2] of the LAST field gives trailing field width (space in data line
    after the last header field, occupied by spec-only sensors).

    data_length = width[0] + sum(meta[2] of fields 0..N-2) + trailing
    trailing = meta[2] of last field
    So: width[0] = data_length - sum(ALL meta[2])

    Returns (widths, trailing_width) or (None, 0).
    """
    n = len(field_defs)
    if n == 0:
        return None, 0

    all_meta2 = []
    for fdef in field_defs:
        if len(fdef.meta) > 2:
            all_meta2.append(fdef.meta[2])
        else:
            return None, 0

    # widths[i] for i>0 = meta[2] of previous field
    widths = [0] * n
    for i in range(1, n):
        widths[i] = all_meta2[i - 1]

    # trailing = meta[2] of last field
    trailing = all_meta2[-1]

    # width[0] = data_length - sum(widths[1:]) - trailing
    widths[0] = data_length - sum(widths[1:]) - trailing

    # NOTE: Do NOT override direction fields to width=1 here.
    # "Direction" fields (meta[0]=2) like Status DWS can be 4+ chars wide.
    # The shifted_meta relationship holds for ALL fields including direction.
    # Direction-specific handling (skip numeric parsing) is in extract_values().

    # Verify: sum(widths) + trailing should equal data_length
    if sum(widths) + trailing == data_length and widths[0] > 0:
        return widths, trailing

    # Fallback: maybe there's no trailing field
    widths2 = [0] * n
    for i in range(1, n):
        widths2[i] = all_meta2[i - 1]
    widths2[0] = data_length - sum(widths2[1:])
    if sum(widths2) == data_length and widths2[0] > 0:
        return widths2, 0

    return None, 0


def compute_pure_spec_widths(field_defs, spec, data_length):
    """Widths from MEDEF spec, including trailing spec fields not in header."""
    widths = []
    header_names_norm = set()
    for fdef in field_defs:
        header_names_norm.add(fdef.name)
        header_names_norm.add(normalize_spec_name(fdef.name))

        # Direction fields (meta[0]=2) can be multi-char (e.g. Status DWS = 4 chars).
        # Look up spec width like any other field.
        se = find_sensor_in_spec(fdef.name, spec)
        if se:
            widths.append(spec_field_width(se))
        elif fdef.is_direction:
            # Direction field not in spec — assume 1 char (L/ /R)
            widths.append(1)
        else:
            return None, 0

    header_total = sum(widths)

    # Trailing spec fields
    trailing_width = 0
    for sname, sinfo in spec.get("sensors", {}).items():
        norm = normalize_spec_name(sname)
        if sname not in header_names_norm and norm not in header_names_norm:
            trailing_width += spec_field_width(sinfo)

    total = header_total + trailing_width
    return widths, total


def validate_with_signs(widths, data_lines, field_defs):
    sign_freq = scan_sign_positions(data_lines)
    anchors = {pos for pos, freq in sign_freq.items() if freq > 0.5}

    if not anchors:
        return 1.0, "no sign anchors"

    expected = set()
    pos = 0
    for fdef, w in zip(field_defs, widths):
        if fdef.signed:
            expected.add(pos)
        pos += w

    matched = anchors & expected
    unexpected = anchors - expected
    precision = len(matched) / len(anchors) if anchors else 1.0

    detail = f"{len(matched)}/{len(anchors)} anchors match"
    if unexpected:
        detail += f", {len(unexpected)} unexpected"
    return precision, detail


def load_json_medians(json_dir: Path, max_files: int = 50) -> dict[str, float]:
    if not json_dir or not json_dir.exists():
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
    return {name: statistics.median(vals) for name, vals in sensor_values.items() if vals}


def extract_values(data_lines, widths, field_defs):
    all_raw = [[] for _ in range(len(field_defs))]
    for dl in data_lines:
        pos = 0
        for i, (fdef, w) in enumerate(zip(field_defs, widths)):
            if pos + w > len(dl):
                break
            raw_str = dl[pos:pos + w]
            pos += w
            if fdef.is_direction:
                continue
            s = raw_str.strip()
            if not s:
                continue
            try:
                all_raw[i].append(int(s))
            except ValueError:
                pass
    return all_raw


def find_best_spec(field_defs, data_lines, data_length, specs):
    """Find best spec for this machine. Try pure_spec first, then shifted_meta."""
    candidates = []
    shifted_widths, trailing = compute_shifted_meta_widths(field_defs, data_length)

    for spec_key, spec_data in specs.items():
        if spec_key == "machine_technique_map" or not isinstance(spec_data, dict):
            continue
        if "sensors" not in spec_data:
            continue

        n_in_spec = sum(
            1 for f in field_defs if find_sensor_in_spec(f.name, spec_data) is not None
        )
        coverage = n_in_spec / max(len(field_defs), 1)
        if coverage < 0.3:
            continue

        # Try pure spec
        spec_widths, spec_total = compute_pure_spec_widths(field_defs, spec_data, data_length)
        if spec_widths is not None and data_length - spec_total == 0:
            score, detail = validate_with_signs(spec_widths, data_lines, field_defs)
            candidates.append({
                "spec_key": spec_key, "approach": "pure_spec",
                "widths": spec_widths, "sign_score": score,
                "coverage": coverage, "quality": 3.0 + score + coverage,
                "detail": detail,
            })

        # Try shifted_meta
        if shifted_widths:
            score, detail = validate_with_signs(shifted_widths, data_lines, field_defs)
            candidates.append({
                "spec_key": spec_key, "approach": "shifted_meta",
                "widths": shifted_widths, "sign_score": score,
                "coverage": coverage, "quality": 1.0 + coverage + score * 0.5,
                "detail": detail,
            })

    candidates.sort(key=lambda c: c["quality"], reverse=True)
    return candidates


def analyze_machine(machine_slug, label, dat_files, json_dir, specs, corrections, out,
                    max_dat=30):
    """Full analysis for one machine."""
    out.write(f"\n{'#'*80}\n")
    out.write(f"# {label} / {machine_slug}\n")
    out.write(f"{'#'*80}\n\n")

    try:
        field_defs, data_lines, data_length = read_dat_file(dat_files[0])
    except Exception as e:
        out.write(f"  ERROR: {e}\n")
        return None

    if not field_defs or not data_lines:
        out.write(f"  No data.\n")
        return None

    for dp in dat_files[1:max_dat]:
        try:
            _, more, dl = read_dat_file(dp)
            if dl == data_length:
                data_lines.extend(more)
        except Exception:
            pass

    out.write(f"  {len(field_defs)} fields, {data_length} chars/line, "
              f"{len(dat_files)} files, {len(data_lines)} lines sampled\n\n")

    # Show header
    out.write("  HEADER:\n")
    for i, fdef in enumerate(field_defs):
        m2 = fdef.meta[2] if len(fdef.meta) > 2 else "?"
        s = "S" if fdef.signed else ("D" if fdef.is_direction else "U")
        out.write(f"    {i:>2}. {fdef.name:<35s}  div={fdef.divisor:<6}  m2={str(m2):<3}  {s}\n")

    # Find best approach
    candidates = find_best_spec(field_defs, data_lines, data_length, specs)

    if not candidates:
        out.write("\n  NO MATCHING SPEC\n")
        # Show shifted meta widths for reference
        shifted, trail = compute_shifted_meta_widths(field_defs, data_length)
        if shifted:
            out.write(f"  Shifted meta widths sum={sum(shifted)}+{trail}={sum(shifted)+trail}, data_length={data_length}\n")
        # Show closest specs
        for sk, sd in sorted(specs.items()):
            if sk == "machine_technique_map" or not isinstance(sd, dict):
                continue
            if "sensors" not in sd:
                continue
            cov = sum(1 for f in field_defs if find_sensor_in_spec(f.name, sd)) / len(field_defs)
            if cov > 0.3:
                sw, st = compute_pure_spec_widths(field_defs, sd, data_length)
                gap = data_length - st if sw else "incomplete"
                out.write(f"    {sk:<20s}  cov={cov:.2f}  spec_total={st if sw else '?'}  gap={gap}\n")
        out.write("\n")
        return None

    # Show candidates
    out.write("\n  CANDIDATES:\n")
    for c in candidates[:5]:
        marker = " <<<" if c is candidates[0] else ""
        out.write(f"    {c['spec_key']:<20s}  {c['approach']:<14s}  "
                  f"sign={c['sign_score']:.2f}  cov={c['coverage']:.2f}  "
                  f"q={c['quality']:.2f}{marker}  ({c['detail']})\n")
    out.write("\n")

    best = candidates[0]
    widths = best["widths"]
    spec_data = specs[best["spec_key"]]

    out.write(f"  BEST: {best['spec_key']} via {best['approach']}\n\n")

    # Field layout
    out.write(f"  {'#':>3}  {'Field':<35s}  {'w':>3}  {'sw':>3}  "
              f"{'div':>6}  {'unit':<6}  notes\n")
    out.write("  " + "-" * 80 + "\n")
    for i, (fdef, w) in enumerate(zip(field_defs, widths)):
        se = find_sensor_in_spec(fdef.name, spec_data)
        if se:
            sw = spec_field_width(se)
            sd = se["expected_divisor"]
            unit = se.get("unit", "") or ""
            note = "" if w == sw else f"w≠sw({sw})"
        else:
            sw = "-"
            sd = fdef.divisor
            unit = fdef.unit or ""
            note = "[NO SPEC]"
        out.write(f"  {i:>3}  {fdef.name:<35s}  {w:>3}  {str(sw):>3}  "
                  f"{sd:>6}  {unit:<6}  {note}\n")
    out.write("  " + "-" * 80 + "\n\n")

    # Extract values
    all_raw = extract_values(data_lines, widths, field_defs)
    json_medians = load_json_medians(json_dir)
    machine_corr = corrections.get(machine_slug, {})

    if json_medians:
        out.write(f"  JSON: {len(json_medians)} sensors\n\n")

    # Comparison table
    out.write(f"  {'Sensor':<30s}  {'Spec':>9}  {'Corr':>9}  {'JSON':>9}  "
              f"{'S/J':>7}  {'C/J':>7}  {'Win':>5}\n")
    out.write("  " + "-" * 90 + "\n")

    sw = cw = tw = sc = cc = tc = 0

    for i, fdef in enumerate(field_defs):
        if fdef.is_direction:
            continue
        raw = all_raw[i]
        if not raw:
            continue
        nz = [v for v in raw if v != 0]
        if not nz:
            continue
        active = statistics.median(nz)

        se = find_sensor_in_spec(fdef.name, spec_data)
        sdiv = se["expected_divisor"] if se else 1
        sval = active / sdiv

        hdiv = fdef.divisor if fdef.divisor else 1
        cfac = machine_corr.get(fdef.name, 1.0)
        cval = active / hdiv / cfac

        jval = json_medians.get(fdef.name)
        sr_s = cr_s = winner = ""

        if jval and jval != 0 and sval != 0:
            tc += 1
            sr = abs(sval / jval)
            cr = abs(cval / jval) if cval else 0
            sr_s = f"{sr:.2f}"
            cr_s = f"{cr:.2f}" if cr else "-"

            if 0.1 <= sr <= 10:
                sc += 1
            if cr and 0.1 <= cr <= 10:
                cc += 1

            # Check if spec and corr give same value (tie)
            max_abs = max(abs(sval), abs(cval), 1e-10)
            is_tie = abs(sval - cval) < 0.001 * max_abs

            se_err = abs(sr - 1)
            ce_err = abs(cr - 1) if cr else 999
            if is_tie:
                winner = "TIE"
                tw += 1
            elif se_err < ce_err:
                winner = "SPEC"
                sw += 1
            else:
                winner = "CORR"
                cw += 1

        jstr = f"{jval:.2f}" if jval else "-"
        out.write(f"  {fdef.name:<30s}  {sval:>9.3f}  {cval:>9.3f}  {jstr:>9}  "
                  f"{sr_s:>7}  {cr_s:>7}  {winner:>5}\n")

    out.write("  " + "-" * 90 + "\n")

    if tc > 0:
        out.write(f"\n  SUMMARY ({tc} sensors):\n")
        out.write(f"    Spec within 10x:  {sc}/{tc} ({100*sc//tc}%)\n")
        out.write(f"    Corr within 10x:  {cc}/{tc} ({100*cc//tc}%)\n")
        out.write(f"    Spec closer:      {sw}/{tc} ({100*sw//tc}%)\n")
        out.write(f"    Corr closer:      {cw}/{tc} ({100*cw//tc}%)\n")
        out.write(f"    Tied (same val):  {tw}/{tc} ({100*tw//tc}%)\n")

    out.write("\n")
    return {
        "label": label, "machine": machine_slug, "spec_key": best["spec_key"],
        "approach": best["approach"], "sign_score": best["sign_score"],
        "coverage": best["coverage"], "n_fields": len(field_defs),
        "total_compared": tc, "spec_wins": sw, "corr_wins": cw, "ties": tw,
        "spec_close": sc, "corr_close": cc,
    }


def discover_machines(raw_dl):
    """Scan raw_downloads for unique (project, machine) combos."""
    machines = {}
    for proj_dir in sorted(raw_dl.iterdir()):
        if not proj_dir.is_dir():
            continue

        dat_dir = proj_dir / "Unidentified"
        if not dat_dir.exists():
            continue

        dat_files = sorted(dat_dir.glob("*.dat"))
        if not dat_files:
            continue

        # Group by machine slug from filename
        groups = defaultdict(list)
        for dp in dat_files:
            parts = dp.stem.split("_")
            # Handle both formats:
            # 01K00044171_bg33v_5610_... → slug = bg33v_5610
            # mc86_621_20240926_... → slug = mc86_621
            # bg33v_5610_20240221_... → slug = bg33v_5610
            if len(parts) >= 4 and parts[0].startswith("01K"):
                slug = f"{parts[1]}_{parts[2]}"
            elif len(parts) >= 3:
                # Check if parts[2] looks like a date (8 digits)
                if len(parts[2]) == 8 and parts[2].isdigit():
                    slug = f"{parts[0]}_{parts[1]}"
                else:
                    slug = f"{parts[1]}_{parts[2]}"
            else:
                slug = "unknown"
            groups[slug].append(dp)

        # Find JSON directories
        json_dir = None
        for subdir in proj_dir.iterdir():
            if subdir.is_dir() and subdir.name != "Unidentified":
                if list(subdir.glob("*.json"))[:1]:
                    json_dir = subdir
                    break

        for slug, files in groups.items():
            key = (proj_dir.name, slug)
            if key not in machines:
                machines[key] = {
                    "machine_slug": slug, "label": proj_dir.name,
                    "dat_files": files, "json_dir": json_dir,
                }
            else:
                machines[key]["dat_files"].extend(files)

    return list(machines.values())


def main():
    specs = load_specs()
    corrections = load_corrections()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "hybrid_spec_parse.txt"

    machines = discover_machines(RAW_DL)

    with open(out_path, "w") as out:
        out.write("HYBRID SPEC PARSING v2: shifted-meta widths + spec divisors\n")
        out.write("=" * 80 + "\n\n")
        out.write("KEY DISCOVERY: meta[2] of field N = total width of field N+1.\n")
        out.write("So width[0] = data_length - sum(all meta[2]).\n\n")
        out.write(f"Machines: {len(machines)}\n\n")

        results = []
        for m in machines:
            try:
                r = analyze_machine(
                    m["machine_slug"], m["label"], m["dat_files"],
                    m["json_dir"], specs, corrections, out,
                )
                if r:
                    results.append(r)
            except Exception as e:
                out.write(f"\nERROR: {m['label']}/{m['machine_slug']}: {e}\n")
                import traceback
                out.write(traceback.format_exc())

        # Summary
        out.write(f"\n{'='*80}\n")
        out.write("GRAND SUMMARY\n")
        out.write(f"{'='*80}\n\n")
        out.write(f"Matched: {len(results)}/{len(machines)} machines\n\n")

        by_approach = defaultdict(list)
        for r in results:
            by_approach[r["approach"]].append(r)
        for ap, rs in sorted(by_approach.items()):
            out.write(f"{ap}: {len(rs)} machines\n")
            for r in rs:
                line = f"  {r['label']}/{r['machine']} → {r['spec_key']}"
                if r["total_compared"] > 0:
                    line += (f"  JSON: spec={r['spec_wins']} corr={r['corr_wins']} "
                             f"tie={r['ties']}/{r['total_compared']}")
                out.write(line + "\n")
            out.write("\n")

        tj = sum(r["total_compared"] for r in results)
        ts = sum(r["spec_wins"] for r in results)
        tc = sum(r["corr_wins"] for r in results)
        tt = sum(r["ties"] for r in results)
        tsc = sum(r["spec_close"] for r in results)
        tcc = sum(r["corr_close"] for r in results)

        if tj > 0:
            non_tie = tj - tt
            out.write(f"JSON TOTALS ({tj} sensors, {tt} ties):\n")
            out.write(f"  Spec within 10x:  {tsc}/{tj} ({100*tsc//tj}%)\n")
            out.write(f"  Corr within 10x:  {tcc}/{tj} ({100*tcc//tj}%)\n")
            out.write(f"  Spec closer:      {ts}/{tj} ({100*ts//tj}%)\n")
            out.write(f"  Corr closer:      {tc}/{tj} ({100*tc//tj}%)\n")
            out.write(f"  Tied (same val):  {tt}/{tj} ({100*tt//tj}%)\n")
            if non_tie > 0:
                out.write(f"\n  Excluding ties ({non_tie} sensors where spec ≠ corr):\n")
                out.write(f"    Spec closer:  {ts}/{non_tie} ({100*ts//non_tie}%)\n")
                out.write(f"    Corr closer:  {tc}/{non_tie} ({100*tc//non_tie}%)\n")

    print(f"Wrote {out_path}")
    print(f"Matched {len(results)}/{len(machines)}")
    if tj > 0:
        non_tie = tj - tt
        print(f"JSON: spec_close={tsc}/{tj} ({100*tsc//tj}%), "
              f"corr_close={tcc}/{tj} ({100*tcc//tj}%)")
        print(f"      spec_wins={ts}, corr_wins={tc}, ties={tt} / {tj}")
        if non_tie > 0:
            print(f"      Excluding ties: spec={ts}/{non_tie} ({100*ts//non_tie}%), "
                  f"corr={tc}/{non_tie} ({100*tc//non_tie}%)")


if __name__ == "__main__":
    main()
