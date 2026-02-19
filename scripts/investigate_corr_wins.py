"""Deep-dive investigation of the 10 sensors where corrections beat spec.

For each sensor, examine:
1. Full raw value distribution across multiple DAT files (not just median)
2. Full JSON value distribution across multiple JSON files
3. Whether data range / zero-inflation / sign issues explain the discrepancy
4. Per-file comparisons (DAT vs JSON from same time periods)

Usage:
    uv run python scripts/investigate_corr_wins.py
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

# The 10 CORR-winning sensors, deduplicated to unique (machine, sensor, spec) combos
CASES = [
    {
        "label": "Paris mc86_621 Neigung Y",
        "project_pattern": "Paris",
        "machine_slug": "mc86_621",
        "sensor": "Neigung Y",
        "spec_key": "Fraese_HDS_v7",
    },
    {
        "label": "bg30v_2872 Druck Pumpe 1",
        "project_pattern": "1501",
        "machine_slug": "bg30v_2872",
        "sensor": "Druck Pumpe 1",
        "spec_key": "KELLY_v8",
    },
    {
        "label": "VICENZA bg33v_5610 Ausladung",
        "project_pattern": "VICENZA",
        "machine_slug": "bg33v_5610",
        "sensor": "Ausladung",
        "spec_key": "FDP_v7",
    },
    {
        "label": "Roma bg42v_5925 Tiefe_Hauptwinde_GOK",
        "project_pattern": "Roma",
        "machine_slug": "bg42v_5925",
        "sensor": "Tiefe_Hauptwinde_GOK",
        "spec_key": "KELLY_v8",
    },
    {
        "label": "Catania gb50_601 Verdrehwinkel",
        "project_pattern": "Catania",
        "machine_slug": "gb50_601",
        "sensor": "Verdrehwinkel",
        "spec_key": "Seilgreifer_v7",
    },
    {
        "label": "Paris mc86_621 Messpunktabw. Y",
        "project_pattern": "Paris",
        "machine_slug": "mc86_621",
        "sensor": "Messpunktabw. Y",
        "spec_key": "Fraese_HDS_v7",
    },
    {
        "label": "Paris Fräse Auflast",
        "project_pattern": "Paris",
        "machine_slug": "mc86_621",
        "sensor": "Auflast",
        "spec_key": "Fraese_HDS_v7",
    },
]


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


def compute_shifted_meta_widths(field_defs, data_length):
    n = len(field_defs)
    if n == 0:
        return None, 0
    all_meta2 = []
    for fdef in field_defs:
        if len(fdef.meta) > 2:
            all_meta2.append(fdef.meta[2])
        else:
            return None, 0
    widths = [0] * n
    for i in range(1, n):
        widths[i] = all_meta2[i - 1]
    trailing = all_meta2[-1]
    widths[0] = data_length - sum(widths[1:]) - trailing
    if sum(widths) + trailing == data_length and widths[0] > 0:
        return widths, trailing
    return None, 0


def read_dat_file(path: Path):
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


def extract_sensor_values(data_lines, widths, field_defs, sensor_idx):
    """Extract raw integer values for a specific sensor from data lines."""
    values = []
    for dl in data_lines:
        pos = 0
        for i, (fdef, w) in enumerate(zip(field_defs, widths)):
            if pos + w > len(dl):
                break
            if i == sensor_idx:
                raw_str = dl[pos:pos + w]
                s = raw_str.strip()
                if s:
                    try:
                        values.append(int(s))
                    except ValueError:
                        pass
                break
            pos += w
    return values


def load_json_sensor_values(json_dir: Path, sensor_name: str, max_files: int = 50):
    """Load all values for a specific sensor from JSON files. Return per-file stats."""
    if not json_dir or not json_dir.exists():
        return [], {}
    json_files = sorted(json_dir.glob("*.json"))[:max_files]
    all_values = []
    per_file = {}
    for jf in json_files:
        try:
            with open(jf) as f:
                data = json.load(f)
            ts_block = data.get("timeSeriesBlock", {})
            for sensor in ts_block.get("serialValuesFree", []):
                name = sensor.get("seriesName") or sensor.get("description", "")
                if name == sensor_name:
                    values = sensor.get("values", [])
                    nums = [v for v in values if isinstance(v, (int, float))]
                    non_zero = [v for v in nums if v != 0]
                    if nums:
                        per_file[jf.name] = {
                            "n": len(nums),
                            "n_nonzero": len(non_zero),
                            "min": min(nums),
                            "max": max(nums),
                            "median": statistics.median(nums),
                            "median_nz": statistics.median(non_zero) if non_zero else 0,
                        }
                        all_values.extend(nums)
        except Exception:
            continue
    return all_values, per_file


def pct(values, p):
    """Percentile."""
    if not values:
        return 0
    s = sorted(values)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s) - 1)]


def describe(values, label=""):
    """Describe a distribution."""
    if not values:
        return f"  {label}: NO DATA"
    nz = [v for v in values if v != 0]
    lines = []
    lines.append(f"  {label}: n={len(values)}, n_nonzero={len(nz)}, "
                 f"zero%={100*(len(values)-len(nz))/len(values):.0f}%")
    lines.append(f"    all:     min={min(values):.4f}  p5={pct(values,5):.4f}  "
                 f"median={statistics.median(values):.4f}  p95={pct(values,95):.4f}  "
                 f"max={max(values):.4f}")
    if nz:
        lines.append(f"    nonzero: min={min(nz):.4f}  p5={pct(nz,5):.4f}  "
                     f"median={statistics.median(nz):.4f}  p95={pct(nz,95):.4f}  "
                     f"max={max(nz):.4f}")
    return "\n".join(lines)


def find_project_dirs(pattern, machine_slug):
    """Find project directories matching pattern."""
    dirs = []
    for d in sorted(RAW_DL.iterdir()):
        if not d.is_dir():
            continue
        if pattern in d.name:
            dirs.append(d)
    return dirs


def investigate_case(case, specs, corrections, out):
    out.write(f"\n{'#' * 80}\n")
    out.write(f"# {case['label']}\n")
    out.write(f"{'#' * 80}\n\n")

    spec_data = specs.get(case["spec_key"], {})
    spec_entry = find_sensor_in_spec(case["sensor"], spec_data)
    machine_corr = corrections.get(case["machine_slug"], {})
    corr_factor = machine_corr.get(case["sensor"], 1.0)

    out.write(f"  Sensor: {case['sensor']}\n")
    out.write(f"  Machine: {case['machine_slug']}\n")
    out.write(f"  Spec: {case['spec_key']}\n")
    if spec_entry:
        out.write(f"  Spec entry: chars={spec_entry['characters']}, "
                  f"dec={spec_entry['decimal']}, signed={spec_entry.get('signed',False)}, "
                  f"div={spec_entry['expected_divisor']}, unit={spec_entry.get('unit','?')}\n")
    else:
        out.write(f"  Spec entry: NOT FOUND IN SPEC\n")
    out.write(f"  Empirical correction: {corr_factor}\n\n")

    # Find project directories
    proj_dirs = find_project_dirs(case["project_pattern"], case["machine_slug"])
    if not proj_dirs:
        out.write("  NO PROJECT DIRECTORIES FOUND\n\n")
        return

    # Collect DAT data across all project dirs
    all_dat_raw = []
    dat_per_file = {}
    n_dat_files = 0

    for proj_dir in proj_dirs:
        dat_dir = proj_dir / "Unidentified"
        if not dat_dir.exists():
            continue

        dat_files = sorted(dat_dir.glob("*.dat"))
        # Filter by machine slug
        machine_files = []
        for dp in dat_files:
            parts = dp.stem.split("_")
            if len(parts) >= 4 and parts[0].startswith("01K"):
                slug = f"{parts[1]}_{parts[2]}"
            elif len(parts) >= 3:
                if len(parts[2]) == 8 and parts[2].isdigit():
                    slug = f"{parts[0]}_{parts[1]}"
                else:
                    slug = f"{parts[0]}_{parts[1]}"
            else:
                slug = dp.stem
            if slug == case["machine_slug"]:
                machine_files.append(dp)

        for dp in machine_files[:20]:  # Up to 20 files
            try:
                field_defs, data_lines, data_length = read_dat_file(dp)
            except Exception as e:
                continue

            if not data_lines:
                continue

            widths, _ = compute_shifted_meta_widths(field_defs, data_length)
            if widths is None:
                continue

            # Find sensor index
            sensor_idx = None
            for i, fdef in enumerate(field_defs):
                if fdef.name == case["sensor"]:
                    sensor_idx = i
                    break
            if sensor_idx is None:
                continue

            raw_ints = extract_sensor_values(data_lines, widths, field_defs, sensor_idx)
            if not raw_ints:
                continue

            n_dat_files += 1
            all_dat_raw.extend(raw_ints)

            # Per-file stats
            nz = [v for v in raw_ints if v != 0]
            hdr_div = field_defs[sensor_idx].divisor or 1
            dat_per_file[dp.name] = {
                "n": len(raw_ints),
                "n_nz": len(nz),
                "raw_med": statistics.median(raw_ints),
                "raw_med_nz": statistics.median(nz) if nz else 0,
                "raw_min": min(raw_ints),
                "raw_max": max(raw_ints),
                "hdr_div": hdr_div,
            }

    out.write(f"  DAT FILES: {n_dat_files} files, {len(all_dat_raw)} total samples\n")

    if not all_dat_raw:
        out.write("  NO DAT DATA FOUND\n\n")
        return

    # Show raw integer distribution
    out.write(describe([float(v) for v in all_dat_raw], "Raw integers") + "\n")

    # Show spec-scaled distribution
    spec_div = spec_entry["expected_divisor"] if spec_entry else 1
    spec_vals = [v / spec_div for v in all_dat_raw]
    out.write(describe(spec_vals, f"Spec-scaled (÷{spec_div})") + "\n")

    # Show correction-scaled distribution
    hdr_div = 1  # Get from first file
    if dat_per_file:
        hdr_div = list(dat_per_file.values())[0]["hdr_div"]
    corr_vals = [v / hdr_div / corr_factor for v in all_dat_raw]
    out.write(describe(corr_vals, f"Corr-scaled (÷{hdr_div}÷{corr_factor})") + "\n")

    # Per-file breakdown (first 5 files)
    out.write(f"\n  Per-file raw values (up to 5 files):\n")
    for fname, info in list(dat_per_file.items())[:5]:
        out.write(f"    {fname[:60]:60s}  n={info['n']:5d}  nz={info['n_nz']:5d}  "
                  f"raw=[{info['raw_min']}, {info['raw_med_nz']:.0f}, {info['raw_max']}]  "
                  f"hdr_div={info['hdr_div']}\n")

    # Collect JSON data
    out.write(f"\n  JSON FILES:\n")
    all_json_vals = []
    json_per_file = {}

    for proj_dir in proj_dirs:
        # Look for JSON in machine-slug subdirectory
        json_dir = proj_dir / case["machine_slug"]
        if not json_dir.exists():
            # Try other subdirs
            for subdir in sorted(proj_dir.iterdir()):
                if subdir.is_dir() and subdir.name != "Unidentified":
                    jvals, jpf = load_json_sensor_values(subdir, case["sensor"])
                    if jvals:
                        all_json_vals.extend(jvals)
                        json_per_file.update(jpf)
        else:
            jvals, jpf = load_json_sensor_values(json_dir, case["sensor"])
            all_json_vals.extend(jvals)
            json_per_file.update(jpf)

    out.write(f"    {len(json_per_file)} files, {len(all_json_vals)} total samples\n")

    if all_json_vals:
        out.write(describe(all_json_vals, "JSON values") + "\n")

        # Per-file JSON breakdown
        out.write(f"\n  Per-file JSON (up to 5 files):\n")
        for fname, info in list(json_per_file.items())[:5]:
            out.write(f"    {fname[:60]:60s}  n={info['n']:5d}  nz={info['n_nonzero']:5d}  "
                      f"[{info['min']:.3f}, {info['median_nz']:.3f}, {info['max']:.3f}]\n")

    # VERDICT
    out.write(f"\n  ANALYSIS:\n")

    if all_json_vals and all_dat_raw:
        json_nz = [v for v in all_json_vals if v != 0]
        raw_nz = [v for v in all_dat_raw if v != 0]

        if json_nz and raw_nz:
            json_med = statistics.median(json_nz)
            raw_med = statistics.median(raw_nz)

            # What divisor would make DAT match JSON?
            if json_med != 0:
                implied_div = raw_med / json_med
                out.write(f"    Implied divisor (raw_median_nz / json_median_nz): "
                          f"{raw_med:.2f} / {json_med:.2f} = {implied_div:.4f}\n")
                out.write(f"    Spec divisor: {spec_div}\n")
                out.write(f"    Header divisor × correction: {hdr_div} × {corr_factor} = {hdr_div * corr_factor}\n")

                # Check spec at different percentiles
                out.write(f"\n    Comparison at different percentiles:\n")
                for p_label, p in [("p5", 5), ("p25", 25), ("median", 50), ("p75", 75), ("p95", 95)]:
                    raw_p = pct(raw_nz, p)
                    json_p = pct(json_nz, p) if json_nz else 0
                    spec_p = raw_p / spec_div if spec_div else 0
                    corr_p = raw_p / hdr_div / corr_factor if corr_factor else 0

                    if json_p != 0:
                        out.write(f"      {p_label:8s}: raw={raw_p:>10.1f}  "
                                  f"spec={spec_p:>10.3f}  corr={corr_p:>10.3f}  "
                                  f"json={json_p:>10.3f}  "
                                  f"spec/json={spec_p/json_p:.3f}  corr/json={corr_p/json_p:.3f}\n")
                    else:
                        out.write(f"      {p_label:8s}: raw={raw_p:>10.1f}  "
                                  f"spec={spec_p:>10.3f}  corr={corr_p:>10.3f}  "
                                  f"json={json_p:>10.3f}\n")

            # Check if the issue is sign convention
            dat_signs = sum(1 for v in raw_nz if v < 0) / len(raw_nz)
            json_signs = sum(1 for v in json_nz if v < 0) / len(json_nz)
            out.write(f"\n    Sign convention: DAT negative={dat_signs:.0%}, JSON negative={json_signs:.0%}\n")

            # Check zero inflation
            dat_zero_pct = 100 * (len(all_dat_raw) - len(raw_nz)) / len(all_dat_raw) if all_dat_raw else 0
            json_zero_pct = 100 * (len(all_json_vals) - len(json_nz)) / len(all_json_vals) if all_json_vals else 0
            out.write(f"    Zero inflation: DAT={dat_zero_pct:.0f}%, JSON={json_zero_pct:.0f}%\n")

            # Check value range overlap
            if abs(implied_div - spec_div) / max(spec_div, 1) < 0.5:
                out.write(f"\n    VERDICT: Implied divisor ({implied_div:.2f}) MATCHES spec ({spec_div}) "
                          f"— DATA RANGE artifact, not spec error\n")
            elif corr_factor != 1.0 and abs(implied_div - hdr_div * corr_factor) / max(hdr_div * corr_factor, 1) < 0.5:
                out.write(f"\n    VERDICT: Implied divisor ({implied_div:.2f}) matches correction "
                          f"({hdr_div * corr_factor}) — GENUINE spec deviation\n")
            else:
                out.write(f"\n    VERDICT: Neither spec ({spec_div}) nor correction ({hdr_div * corr_factor}) "
                          f"matches implied divisor ({implied_div:.2f}) — NEEDS INVESTIGATION\n")
    else:
        out.write("    Insufficient data for comparison\n")

    out.write("\n")


def main():
    specs = load_specs()
    corrections = load_corrections()

    out_path = Path(__file__).resolve().parent.parent / "output" / "diagnostics" / "corr_wins_investigation.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as out:
        out.write("DEEP INVESTIGATION: 10 Sensors Where Corrections Beat Spec\n")
        out.write("=" * 80 + "\n")
        out.write(f"Goal: Determine if these are genuine spec errors or data range artifacts\n\n")

        for case in CASES:
            investigate_case(case, specs, corrections, out)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
