"""Compute empirical DAT correction factors using JSON data as ground truth.

For each (machine, sensor) pair that has both JSON and DAT traces:
1. Load JSON parquet data -> compute per-sensor medians (ground truth)
2. Parse raw DAT files with header divisor ONLY (no heuristic)
3. Compare and compute the needed correction factor
4. Output macchine/harmonize/dat_corrections.yaml

This replaces the unreliable heuristic in detect_dat_divisor() with
deterministic, provable correction factors.
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from macchine.harmonize.calibration import get_sentinel_values
from macchine.storage.catalog import get_trace_index

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
CORRECTIONS_PATH = (
    Path(__file__).resolve().parent.parent
    / "macchine"
    / "harmonize"
    / "dat_corrections.yaml"
)


def parse_dat_header_only(path: Path) -> list[tuple[str, list[float]]]:
    """Parse a DAT file applying ONLY the header-declared divisor.

    Returns list of (sensor_name, values) tuples.
    Bypasses the heuristic by calling internal parsing functions directly.
    """
    # Import internal functions from dat_parser
    from macchine.parsers.dat_parser import _parse_data_lines, _parse_header
    from macchine.parsers.beginng_parser import parse_beginng

    with open(path, "rb") as f:
        raw = f.read()
    text = raw.decode("iso-8859-1")
    sections = text.split("$")

    if len(sections) < 2:
        return []

    field_defs = _parse_header(sections[0])
    if not field_defs:
        return []

    # Get timestamp from BEGINNG (needed for _parse_data_lines)
    beginng_info = {}
    for section in sections[1:5]:
        if section.startswith("BEGINN"):
            beginng_info = parse_beginng(section)
            break

    start_time = beginng_info.get("timestamp")

    # Parse data lines — this applies header divisor but NOT the heuristic
    sensors = _parse_data_lines(sections, field_defs, start_time)

    result = []
    for s in sensors:
        result.append((s.sensor_name, s.values))
    return result


def collect_json_medians(
    traces_dir: Path,
    json_traces: pd.DataFrame,
    sentinels: set,
    max_files: int = 15,
) -> dict[str, float]:
    """Load JSON parquet files and compute per-sensor medians."""
    sensor_values: dict[str, list[float]] = defaultdict(list)
    count = 0

    for _, row in json_traces.head(max_files).iterrows():
        site_id = str(row["site_id"])
        slug = str(row.get("machine_slug", "unknown"))
        trace_id = str(row.get("trace_id", ""))

        path = traces_dir / site_id / slug / f"{trace_id}.parquet"
        if not path.exists():
            continue

        try:
            df = pd.read_parquet(path)
            count += 1
        except Exception:
            continue

        for col in df.select_dtypes(include=[np.number]).columns:
            if col in ("timestamp",):
                continue
            vals = df[col].dropna()
            vals = vals[~vals.isin(sentinels)]
            if len(vals) < 10:
                continue
            sensor_values[col].extend(vals.tolist()[:500])

    medians = {}
    for sensor, vals in sensor_values.items():
        if vals:
            medians[sensor] = float(np.median(vals))

    return medians


def collect_dat_raw_medians(
    dat_traces: pd.DataFrame,
    sentinels: set,
    max_files: int = 15,
) -> dict[str, float]:
    """Re-parse raw DAT files with header-only divisor, compute per-sensor medians."""
    sensor_values: dict[str, list[float]] = defaultdict(list)
    count = 0

    for _, row in dat_traces.head(max_files * 2).iterrows():
        source_path = Path(str(row.get("source_path", "")))
        if not source_path.exists():
            continue

        try:
            sensors = parse_dat_header_only(source_path)
            if not sensors:
                continue
            count += 1
        except Exception:
            continue

        if count > max_files:
            break

        for name, values in sensors:
            valid = [
                v
                for v in values
                if v is not None and not math.isnan(v) and v not in sentinels
            ]
            if len(valid) < 10:
                continue
            sensor_values[name].extend(valid[:500])

    medians = {}
    for sensor, vals in sensor_values.items():
        if vals:
            medians[sensor] = float(np.median(vals))

    return medians


def find_best_divisor(ratio: float) -> float | None:
    """Find the nearest clean divisor for a DAT/JSON ratio.

    Returns the divisor D such that dat_value / D ≈ json_value.
    Only returns factors that are clean multiples/fractions of powers of 10.

    Returns None if no clean factor matches.
    """
    abs_ratio = abs(ratio)

    # Standard divisors to check
    candidates = [
        0.001, 0.01, 0.1,  # header over-divides
        10, 100, 1000, 10000, 100000, 1000000,  # header under-divides
    ]

    for d in candidates:
        # Check if the ratio is close to this divisor (within factor of 3)
        if abs_ratio < 1e-10:
            continue
        if 0.33 * d <= abs_ratio <= 3.0 * d:
            return d

    return None


def compute_corrections() -> dict[str, dict[str, dict]]:
    """Main computation: compare DAT header-only values against JSON ground truth."""
    output_dir = OUTPUT_DIR
    traces_dir = output_dir / "traces"
    sentinels = set(get_sentinel_values())

    # Use basic trace index (has one entry per raw file with source_path)
    df = get_trace_index(output_dir)

    # Find machines with both formats
    machines_both = []
    for slug, g in df.groupby("machine_slug"):
        if not slug:
            continue
        fmts = g["format"].value_counts()
        if "json" in fmts and "dat" in fmts:
            machines_both.append(slug)

    print(f"Machines with both JSON and DAT formats: {machines_both}")

    all_corrections: dict[str, dict[str, dict]] = {}

    for machine_slug in sorted(machines_both):
        print(f"\n{'=' * 60}")
        print(f"Machine: {machine_slug}")
        print(f"{'=' * 60}")

        machine_traces = df[df["machine_slug"] == machine_slug]
        json_traces = machine_traces[machine_traces["format"] == "json"]
        dat_traces = machine_traces[machine_traces["format"] == "dat"]

        print(f"  JSON traces: {len(json_traces)}, DAT traces: {len(dat_traces)}")

        # Step 1: JSON medians (ground truth)
        json_medians = collect_json_medians(traces_dir, json_traces, sentinels)
        print(f"  JSON sensors with data: {len(json_medians)}")

        # Step 2: DAT medians (header divisor only, no heuristic)
        dat_medians = collect_dat_raw_medians(dat_traces, sentinels)
        print(f"  DAT sensors with data: {len(dat_medians)}")

        # Step 3: Compare
        common = sorted(set(json_medians) & set(dat_medians))
        machine_corrections = {}

        for sensor in common:
            jmed = json_medians[sensor]
            dmed = dat_medians[sensor]

            # Skip if either is zero (can't compute ratio)
            if abs(jmed) < 1e-10 or abs(dmed) < 1e-10:
                continue

            ratio = dmed / jmed

            # If close to 1, no correction needed
            if 0.5 <= abs(ratio) <= 2.0:
                continue

            divisor = find_best_divisor(ratio)
            if divisor is not None and divisor != 1.0:
                machine_corrections[sensor] = {
                    "divisor": divisor,
                    "json_median": round(jmed, 4),
                    "dat_raw_median": round(dmed, 4),
                    "ratio": round(ratio, 2),
                }
                direction = "÷" if divisor > 1 else "×"
                factor = divisor if divisor > 1 else int(1 / divisor)
                print(
                    f"  {sensor}: JSON={jmed:.3f}, DAT_raw={dmed:.3f}, "
                    f"ratio={ratio:.1f} → {direction}{factor}"
                )
            else:
                print(
                    f"  {sensor}: JSON={jmed:.3f}, DAT_raw={dmed:.3f}, "
                    f"ratio={ratio:.1f} → NO CLEAN FACTOR"
                )

        if machine_corrections:
            all_corrections[machine_slug] = machine_corrections

    return all_corrections


def write_corrections_yaml(corrections: dict[str, dict[str, dict]]) -> None:
    """Write the corrections to a YAML file."""
    lines = [
        "# Empirical DAT correction divisors.",
        "# Computed by comparing DAT values (header-declared divisor only)",
        "# against JSON ground-truth medians.",
        "#",
        "# For each (machine_slug, sensor_name): the additional divisor to apply",
        "# to DAT values AFTER the header-declared divisor.",
        "#   divisor > 1  → DAT values are too large, divide by this",
        "#   divisor < 1  → DAT values are too small (header over-divides),",
        "#                   effectively multiply by 1/divisor",
        "#",
        "# Generated by: python scripts/compute_dat_corrections.py",
        "",
    ]

    for machine in sorted(corrections):
        lines.append(f"{machine}:")
        for sensor in sorted(corrections[machine]):
            info = corrections[machine][sensor]
            d = info["divisor"]
            # Format divisor nicely
            if d >= 1:
                d_str = str(int(d))
            else:
                d_str = str(d)
            comment = (
                f"JSON={info['json_median']}, "
                f"DAT_raw={info['dat_raw_median']}, "
                f"ratio={info['ratio']}"
            )
            # Quote sensor names with special chars
            if any(c in sensor for c in ".-+äöüß "):
                lines.append(f'  "{sensor}": {d_str}  # {comment}')
            else:
                lines.append(f"  {sensor}: {d_str}  # {comment}")
        lines.append("")

    CORRECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CORRECTIONS_PATH, "w") as f:
        f.write("\n".join(lines))

    print(f"\nWrote corrections to {CORRECTIONS_PATH}")


def main():
    corrections = compute_corrections()

    total = sum(len(s) for s in corrections.values())
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {total} corrections across {len(corrections)} machines")
    print(f"{'=' * 60}")

    for machine, sensors in sorted(corrections.items()):
        print(f"\n  {machine}: {len(sensors)} corrections")
        for sensor, info in sorted(sensors.items()):
            print(f"    {sensor}: ÷{info['divisor']} (ratio {info['ratio']})")

    write_corrections_yaml(corrections)


if __name__ == "__main__":
    main()
