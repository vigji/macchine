"""Cross-validate MEDEF spec vs DAT header divisors vs empirical corrections.

Three-way comparison for each (machine, sensor):
  1. MEDEF spec (specs_data/medef_specs.yaml): expected_divisor = 10^decimal
  2. DAT file headers: actually declared divisor
  3. dat_corrections.yaml: our empirical additional correction

Generates:
  - output/diagnostics/spec_validation_report.txt  (human-readable)
  - output/diagnostics/spec_validation_report.csv  (machine-readable)

Usage:
    uv run python scripts/cross_validate_corrections.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from macchine.harmonize.sensor_map import get_canonical_name
from macchine.parsers.dat_parser import _parse_header
from macchine.storage.catalog import get_trace_index

SPECS_PATH = Path(__file__).resolve().parent.parent / "specs_data" / "medef_specs.yaml"
CORRECTIONS_PATH = (
    Path(__file__).resolve().parent.parent / "macchine" / "harmonize" / "dat_corrections.yaml"
)
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "diagnostics"


def load_medef_specs() -> dict:
    """Load parsed MEDEF specs."""
    with open(SPECS_PATH) as f:
        return yaml.safe_load(f)


def load_corrections() -> dict[str, dict[str, float]]:
    """Load empirical DAT corrections."""
    with open(CORRECTIONS_PATH) as f:
        data = yaml.safe_load(f)
    return data or {}


def get_header_divisors(dat_path: Path) -> dict[str, int]:
    """Parse a DAT file header and return {sensor_name: divisor}."""
    try:
        with open(dat_path, "rb") as f:
            raw = f.read()
        text = raw.decode("iso-8859-1")
        sections = text.split("$")
        if len(sections) < 2:
            return {}
        field_defs = _parse_header(sections[0])
        return {f.name: f.divisor for f in field_defs}
    except Exception as e:
        print(f"  WARNING: Could not parse header from {dat_path}: {e}")
        return {}


def get_representative_dat_headers(machine_slug: str) -> dict[str, int]:
    """Get header divisors from representative DAT files for a machine.

    Parses up to 3 DAT files and takes the most common divisor for each sensor.
    """
    output_dir = Path(__file__).resolve().parent.parent / "output"
    try:
        df = get_trace_index(output_dir)
    except FileNotFoundError:
        print("WARNING: Trace index not found. Cannot read DAT headers.")
        return {}

    dat_traces = df[(df["machine_slug"] == machine_slug) & (df["format"] == "dat")]
    if dat_traces.empty:
        return {}

    # Parse up to 3 files
    all_divisors: dict[str, list[int]] = defaultdict(list)
    count = 0
    for _, row in dat_traces.iterrows():
        source_path = Path(str(row.get("source_path", "")))
        if not source_path.exists():
            continue
        divs = get_header_divisors(source_path)
        if divs:
            for name, div in divs.items():
                all_divisors[name].append(div)
            count += 1
            if count >= 3:
                break

    # Take most common divisor for each sensor
    result = {}
    for name, divs in all_divisors.items():
        # Most common value
        result[name] = max(set(divs), key=divs.count)

    return result


def find_sensor_in_specs(sensor_name: str, spec_keys: list[str], specs: dict) -> dict | None:
    """Find a sensor in the MEDEF specs across multiple technique specs.

    Returns the spec entry or None.
    """
    for key in spec_keys:
        if key not in specs:
            continue
        sensors = specs[key].get("sensors", {})
        if sensor_name in sensors:
            entry = dict(sensors[sensor_name])
            entry["spec_key"] = key
            return entry
    return None


def cross_validate():
    """Main cross-validation logic."""
    specs = load_medef_specs()
    corrections = load_corrections()
    machine_map = specs.get("machine_technique_map", {})

    print("MEDEF SPEC vs EMPIRICAL CORRECTIONS CROSS-VALIDATION")
    print("=" * 70)
    print(f"Specs loaded: {len(specs) - 1} technique definitions")  # -1 for machine_map
    print(f"Corrections loaded: {sum(len(v) for v in corrections.values())} across {len(corrections)} machines")
    print()

    # Collect results
    results = []  # List of dicts for CSV output
    consistent = []
    inconsistent = []
    uncorrected = []
    not_in_spec = []

    for machine_slug in sorted(set(list(corrections.keys()) + list(machine_map.keys()))):
        spec_keys = machine_map.get(machine_slug, [])
        machine_corrections = corrections.get(machine_slug, {})

        if not spec_keys:
            # Machine not in technique map — all its corrections are "not in spec"
            for sensor_name, correction in machine_corrections.items():
                canonical = get_canonical_name(sensor_name)
                entry = {
                    "machine": machine_slug,
                    "sensor_german": sensor_name,
                    "sensor_english": canonical,
                    "technique": "unknown",
                    "header_divisor": "",
                    "correction": correction,
                    "effective_divisor": "",
                    "spec_divisor": "",
                    "spec_unit": "",
                    "match": "NO_SPEC_MAP",
                    "ratio": "",
                }
                results.append(entry)
                not_in_spec.append(entry)
            continue

        # Get header divisors from actual DAT files
        header_divs = get_representative_dat_headers(machine_slug)

        print(f"\n--- {machine_slug} (techniques: {', '.join(spec_keys)}) ---")
        if header_divs:
            print(f"  DAT header sensors: {len(header_divs)}")
        else:
            print(f"  WARNING: No DAT header data available")

        # 1. Check corrected sensors against spec
        for sensor_name, correction in sorted(machine_corrections.items()):
            canonical = get_canonical_name(sensor_name)
            spec_entry = find_sensor_in_specs(sensor_name, spec_keys, specs)
            header_div = header_divs.get(sensor_name, None)

            if spec_entry:
                spec_divisor = spec_entry["expected_divisor"]
                spec_unit = spec_entry.get("unit", "")
                technique = spec_entry["spec_key"]

                if header_div is not None:
                    effective = header_div * correction
                    if abs(spec_divisor) > 0 and abs(effective) > 0:
                        ratio = effective / spec_divisor
                    else:
                        ratio = None

                    entry = {
                        "machine": machine_slug,
                        "sensor_german": sensor_name,
                        "sensor_english": canonical,
                        "technique": technique,
                        "header_divisor": header_div,
                        "correction": correction,
                        "effective_divisor": effective,
                        "spec_divisor": spec_divisor,
                        "spec_unit": spec_unit,
                        "match": "",
                        "ratio": ratio if ratio else "",
                    }

                    if ratio and 0.5 <= abs(ratio) <= 2.0:
                        entry["match"] = "MATCH"
                        consistent.append(entry)
                    else:
                        if ratio and abs(ratio) > 1:
                            entry["match"] = f"FINER_{abs(ratio):.0f}x"
                        elif ratio and 0 < abs(ratio) < 1:
                            entry["match"] = f"COARSER_{1/abs(ratio):.0f}x"
                        else:
                            entry["match"] = "MISMATCH"
                        inconsistent.append(entry)
                    results.append(entry)
                else:
                    # No header data, but sensor is in spec
                    entry = {
                        "machine": machine_slug,
                        "sensor_german": sensor_name,
                        "sensor_english": canonical,
                        "technique": technique,
                        "header_divisor": "N/A",
                        "correction": correction,
                        "effective_divisor": "N/A",
                        "spec_divisor": spec_divisor,
                        "spec_unit": spec_unit,
                        "match": "NO_HEADER",
                        "ratio": "",
                    }
                    results.append(entry)
            else:
                # Sensor not in any spec for this machine's techniques
                entry = {
                    "machine": machine_slug,
                    "sensor_german": sensor_name,
                    "sensor_english": canonical,
                    "technique": ",".join(spec_keys),
                    "header_divisor": header_divs.get(sensor_name, "N/A"),
                    "correction": correction,
                    "effective_divisor": "",
                    "spec_divisor": "",
                    "spec_unit": "",
                    "match": "NOT_IN_SPEC",
                    "ratio": "",
                }
                results.append(entry)
                not_in_spec.append(entry)

        # 2. Check uncorrected sensors: in spec but no correction
        seen_uncorrected = set()  # Deduplicate across technique specs
        for key in spec_keys:
            if key not in specs:
                continue
            for sensor_name, spec_info in specs[key].get("sensors", {}).items():
                if sensor_name in machine_corrections:
                    continue  # Already handled above
                if sensor_name not in header_divs:
                    continue  # Not present in actual DAT files
                dedup_key = (machine_slug, sensor_name)
                if dedup_key in seen_uncorrected:
                    continue
                seen_uncorrected.add(dedup_key)

                canonical = get_canonical_name(sensor_name)
                header_div = header_divs[sensor_name]
                spec_divisor = spec_info["expected_divisor"]
                spec_unit = spec_info.get("unit", "")

                # Check if the header divisor matches the spec
                if spec_divisor > 0 and header_div > 0:
                    ratio = header_div / spec_divisor
                else:
                    ratio = None

                match_status = "UNCORRECTED"
                if ratio and 0.5 <= abs(ratio) <= 2.0:
                    match_status = "UNCORRECTED_OK"
                elif ratio:
                    match_status = f"UNCORRECTED_MISMATCH"

                entry = {
                    "machine": machine_slug,
                    "sensor_german": sensor_name,
                    "sensor_english": canonical,
                    "technique": key,
                    "header_divisor": header_div,
                    "correction": 1.0,
                    "effective_divisor": header_div,
                    "spec_divisor": spec_divisor,
                    "spec_unit": spec_unit,
                    "match": match_status,
                    "ratio": ratio if ratio else "",
                }
                results.append(entry)
                uncorrected.append(entry)

    return results, consistent, inconsistent, uncorrected, not_in_spec


def write_txt_report(
    results, consistent, inconsistent, uncorrected, not_in_spec
):
    """Write human-readable report."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "spec_validation_report.txt"

    with open(path, "w") as f:
        f.write("MEDEF SPEC vs EMPIRICAL CORRECTIONS CROSS-VALIDATION\n")
        f.write("=" * 70 + "\n\n")

        # Summary
        f.write(f"Total entries: {len(results)}\n")
        f.write(f"  Consistent (effective ≈ spec):     {len(consistent)}\n")
        f.write(f"  Inconsistent (firmware diverges):   {len(inconsistent)}\n")
        f.write(f"  Uncorrected (in spec, no fix):      {len(uncorrected)}\n")
        f.write(f"  Not in spec:                        {len(not_in_spec)}\n\n")

        # Consistent
        f.write("-" * 70 + "\n")
        f.write("CONSISTENT (effective_divisor ≈ spec_divisor)\n")
        f.write("-" * 70 + "\n")
        if consistent:
            for e in sorted(consistent, key=lambda x: (x["machine"], x["sensor_german"])):
                f.write(
                    f"  {e['machine']:15s}  {e['sensor_german']:30s}  "
                    f"header={e['header_divisor']}  x correction={e['correction']}  "
                    f"= {e['effective_divisor']}  spec={e['spec_divisor']}  "
                    f"[{e['spec_unit']}]  OK\n"
                )
        else:
            f.write("  (none)\n")
        f.write("\n")

        # Inconsistent
        f.write("-" * 70 + "\n")
        f.write("INCONSISTENT (firmware encodes at different resolution than spec)\n")
        f.write("-" * 70 + "\n")
        if inconsistent:
            for e in sorted(inconsistent, key=lambda x: (x["machine"], x["sensor_german"])):
                if isinstance(e['ratio'], (int, float)):
                    if abs(e['ratio']) > 1:
                        ratio_str = f"{abs(e['ratio']):.0f}x finer"
                    elif abs(e['ratio']) > 0:
                        ratio_str = f"{1/abs(e['ratio']):.0f}x coarser"
                    else:
                        ratio_str = "?"
                else:
                    ratio_str = "?"
                f.write(
                    f"  {e['machine']:15s}  {e['sensor_german']:30s}  "
                    f"header={e['header_divisor']}  x correction={e['correction']}  "
                    f"= {e['effective_divisor']}  spec={e['spec_divisor']}  "
                    f"[{e['spec_unit']}]  {ratio_str}\n"
                )
        else:
            f.write("  (none)\n")
        f.write("\n")

        # Uncorrected
        f.write("-" * 70 + "\n")
        f.write("UNCORRECTED SENSORS (in spec, present in DAT, no correction applied)\n")
        f.write("-" * 70 + "\n")
        if uncorrected:
            # Group by match status
            ok = [e for e in uncorrected if e["match"] == "UNCORRECTED_OK"]
            mismatch = [e for e in uncorrected if e["match"] != "UNCORRECTED_OK"]

            if ok:
                f.write("  Header matches spec (no correction needed):\n")
                for e in sorted(ok, key=lambda x: (x["machine"], x["sensor_german"])):
                    f.write(
                        f"    {e['machine']:15s}  {e['sensor_german']:30s}  "
                        f"header={e['header_divisor']}  spec={e['spec_divisor']}  "
                        f"[{e['spec_unit']}]\n"
                    )
            if mismatch:
                f.write("\n  Header DOES NOT match spec (potential missing correction!):\n")
                for e in sorted(mismatch, key=lambda x: (x["machine"], x["sensor_german"])):
                    ratio_str = f"{e['ratio']:.1f}x" if isinstance(e['ratio'], (int, float)) else "?"
                    f.write(
                        f"    {e['machine']:15s}  {e['sensor_german']:30s}  "
                        f"header={e['header_divisor']}  spec={e['spec_divisor']}  "
                        f"[{e['spec_unit']}]  ratio={ratio_str}\n"
                    )
        else:
            f.write("  (none)\n")
        f.write("\n")

        # Not in spec
        f.write("-" * 70 + "\n")
        f.write("SENSORS NOT IN SPEC (in corrections but no MEDEF entry found)\n")
        f.write("-" * 70 + "\n")
        if not_in_spec:
            for e in sorted(not_in_spec, key=lambda x: (x["machine"], x["sensor_german"])):
                f.write(
                    f"  {e['machine']:15s}  {e['sensor_german']:30s}  "
                    f"correction={e['correction']}  ({e['match']})\n"
                )
        else:
            f.write("  (none)\n")

    print(f"\nWrote {path}")


def write_csv_report(results):
    """Write machine-readable CSV report."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "spec_validation_report.csv"

    fieldnames = [
        "machine", "sensor_german", "sensor_english", "technique",
        "header_divisor", "correction", "effective_divisor",
        "spec_divisor", "spec_unit", "match", "ratio",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(results, key=lambda x: (x["machine"], x["sensor_german"])):
            writer.writerow(row)

    print(f"Wrote {path}")


def main():
    results, consistent, inconsistent, uncorrected, not_in_spec = cross_validate()

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total entries:                        {len(results)}")
    print(f"  Consistent (effective ≈ spec):       {len(consistent)}")
    print(f"  Inconsistent (firmware diverges):    {len(inconsistent)}")
    print(f"  Uncorrected (in spec, no fix):       {len(uncorrected)}")
    print(f"  Not in spec:                         {len(not_in_spec)}")

    write_txt_report(results, consistent, inconsistent, uncorrected, not_in_spec)
    write_csv_report(results)


if __name__ == "__main__":
    main()
