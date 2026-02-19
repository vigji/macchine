"""Data quality audit script.

Performs deterministic, provable checks on the dataset:
1. Check for German vs English labels in imported parquet data
2. Report calibration status per (machine, site, technique)
3. Look for gain/divisor issues (like the depth problem) across ALL sensors
4. Physical range validation on all calibrated sensors
5. Verify that DAT-imported data doesn't lose calibration info
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from macchine.config import KNOWN_MACHINES
from macchine.harmonize.calibration import (
    PHYSICAL_RANGES,
    SENSOR_UNITS,
    _load_calibration_yaml,
    _load_sensor_defs,
    get_sentinel_values,
    is_calibrated,
)
from macchine.storage.catalog import get_merged_trace_index, get_trace_index

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def load_sensor_translation_table() -> dict[str, dict]:
    """Load sensor definitions and return German→English mapping."""
    defs = _load_sensor_defs()
    table = {}
    for german_name, info in defs.items():
        table[german_name] = {
            "english": info["canonical"],
            "category": info.get("category", ""),
        }
    return table


def check_column_labels(output_dir: Path) -> dict[str, list[str]]:
    """Check what column names appear in the actual parquet trace files.

    Returns mapping of column_name -> list of example file paths where it appears.
    """
    traces_dir = output_dir / "traces"
    all_columns = defaultdict(list)

    for site_dir in sorted(traces_dir.iterdir()):
        if not site_dir.is_dir():
            continue
        for machine_dir in sorted(site_dir.iterdir()):
            if not machine_dir.is_dir():
                continue
            parquets = list(machine_dir.glob("*.parquet"))
            # Sample up to 3 files per machine/site combo
            for pf in parquets[:3]:
                try:
                    df = pd.read_parquet(pf)
                    for col in df.columns:
                        if len(all_columns[col]) < 3:
                            all_columns[col].append(str(pf.relative_to(output_dir)))
                except Exception:
                    continue

    return dict(all_columns)


def audit_gain_factors(output_dir: Path) -> list[dict]:
    """Check ALL calibrated sensors across all machines for potential gain/divisor issues.

    For each (sensor, machine) pair, loads a sample of traces and checks:
    - Are values within the expected physical range?
    - Is the median step size consistent with the expected resolution?
    - Are values suspiciously integer-only (suggesting raw ADC counts)?
    - Is the range suspiciously wide or narrow?

    Returns list of findings.
    """
    traces_dir = output_dir / "traces"
    findings = []
    sentinels = set(get_sentinel_values())

    # Collect sensor stats per (machine, sensor)
    sensor_stats = defaultdict(lambda: {"values": [], "files": 0})

    for site_dir in sorted(traces_dir.iterdir()):
        if not site_dir.is_dir():
            continue
        for machine_dir in sorted(site_dir.iterdir()):
            if not machine_dir.is_dir():
                continue
            machine_slug = machine_dir.name
            parquets = list(machine_dir.glob("*.parquet"))

            # Sample up to 10 files per machine
            for pf in parquets[:10]:
                try:
                    df = pd.read_parquet(pf)
                except Exception:
                    continue

                for col in df.select_dtypes(include=[np.number]).columns:
                    if col in ("timestamp",):
                        continue
                    key = (machine_slug, col)
                    if sensor_stats[key]["files"] >= 10:
                        continue

                    vals = df[col].dropna()
                    # Remove sentinels
                    vals = vals[~vals.isin(sentinels)]
                    if len(vals) < 10:
                        continue

                    sensor_stats[key]["values"].extend(vals.tolist()[:500])
                    sensor_stats[key]["files"] += 1

    # Now analyze each (machine, sensor) pair
    for (machine_slug, sensor_name), stats in sorted(sensor_stats.items()):
        values = np.array(stats["values"])
        if len(values) < 20:
            continue

        calibrated = is_calibrated(sensor_name, machine_slug)

        # Basic stats
        p5 = np.percentile(values, 5)
        p50 = np.percentile(values, 50)
        p95 = np.percentile(values, 95)
        vmin = np.min(values)
        vmax = np.max(values)

        finding = {
            "machine": machine_slug,
            "sensor": sensor_name,
            "calibrated": calibrated,
            "n_values": len(values),
            "n_files": stats["files"],
            "min": float(vmin),
            "p5": float(p5),
            "median": float(p50),
            "p95": float(p95),
            "max": float(vmax),
            "issues": [],
        }

        # Check 1: Physical range violation for calibrated sensors
        if calibrated and sensor_name in PHYSICAL_RANGES:
            lo, hi = PHYSICAL_RANGES[sensor_name]
            out_of_range = np.sum((values < lo) | (values > hi))
            pct_out = 100 * out_of_range / len(values)
            if pct_out > 1:
                finding["issues"].append(
                    f"RANGE_VIOLATION: {pct_out:.1f}% of values outside [{lo}, {hi}]"
                )

        # Check 2: Suspected wrong divisor - values are all integers and
        # suspiciously large for the physical range
        if calibrated and sensor_name in PHYSICAL_RANGES:
            lo, hi = PHYSICAL_RANGES[sensor_name]
            range_max = max(abs(lo), abs(hi))

            # Check if values suggest a missing divisor
            all_int = np.all(values == np.floor(values))
            if all_int and abs(p95) > range_max * 2:
                # Try common divisors
                for divisor in [10, 100, 1000]:
                    corrected_p95 = abs(p95) / divisor
                    if corrected_p95 <= range_max:
                        finding["issues"].append(
                            f"SUSPECTED_DIVISOR: values look like raw integers; "
                            f"÷{divisor} would put p95={corrected_p95:.1f} within range [{lo},{hi}]"
                        )
                        break

        # Check 3: Uncalibrated sensor - just report the range for human review
        if not calibrated and sensor_name in PHYSICAL_RANGES:
            lo, hi = PHYSICAL_RANGES[sensor_name]
            range_max = max(abs(lo), abs(hi))
            if abs(p95) > range_max * 5:
                finding["issues"].append(
                    f"UNCALIBRATED_RAW: range [{vmin:.0f}, {vmax:.0f}] >> physical [{lo}, {hi}]; "
                    f"likely raw ADC counts"
                )

        # Check 4: All-zero sensor (dead sensor)
        nonzero = np.sum(values != 0)
        if nonzero == 0:
            finding["issues"].append("DEAD_SENSOR: all values are exactly 0")
        elif nonzero < len(values) * 0.01:
            finding["issues"].append(f"NEAR_DEAD: only {nonzero}/{len(values)} non-zero values")

        # Check 5: Constant value (stuck sensor)
        unique = len(np.unique(values))
        if unique <= 3 and len(values) > 100:
            finding["issues"].append(f"STUCK_SENSOR: only {unique} unique values across {len(values)} samples")

        findings.append(finding)

    return findings


def report_calibration_by_combination(output_dir: Path) -> pd.DataFrame:
    """Report calibration status for each (machine, site, technique) combination.

    For each combination, loads sample traces and reports:
    - Which sensors are present
    - Which are calibrated vs uncalibrated
    - Whether lack of calibration might be from DAT import issues
    """
    try:
        df = get_merged_trace_index(output_dir)
    except FileNotFoundError:
        df = get_trace_index(output_dir)

    traces_dir = output_dir / "traces"
    rows = []

    # Group by (machine, site, technique)
    for (machine_slug, site_id, technique), group in df.groupby(
        ["machine_slug", "site_id", "technique"]
    ):
        n_traces = len(group)
        formats = group["format"].value_counts().to_dict() if "format" in group.columns else {}

        # Sample some trace files to check actual sensor columns
        sensors_found = set()
        sensors_calibrated = set()
        sensors_uncalibrated = set()
        sample_count = 0

        # Find trace files for this combination
        machine_dir = traces_dir / str(site_id) / str(machine_slug)
        if machine_dir.exists():
            parquets = list(machine_dir.glob("*.parquet"))[:5]
            for pf in parquets:
                try:
                    tdf = pd.read_parquet(pf)
                    for col in tdf.select_dtypes(include=[np.number]).columns:
                        if col in ("timestamp",):
                            continue
                        sensors_found.add(col)
                        if is_calibrated(col, str(machine_slug)):
                            sensors_calibrated.add(col)
                        else:
                            sensors_uncalibrated.add(col)
                    sample_count += 1
                except Exception:
                    continue

        rows.append({
            "machine": machine_slug,
            "site": site_id,
            "technique": technique,
            "n_traces": n_traces,
            "formats": str(formats),
            "n_sensors": len(sensors_found),
            "n_calibrated": len(sensors_calibrated),
            "n_uncalibrated": len(sensors_uncalibrated),
            "calibrated_sensors": ", ".join(sorted(sensors_calibrated)),
            "uncalibrated_sensors": ", ".join(sorted(sensors_uncalibrated)),
            "sampled_files": sample_count,
        })

    return pd.DataFrame(rows)


def check_dat_vs_json_consistency(output_dir: Path) -> list[dict]:
    """Check if DAT-imported data has different characteristics than JSON-imported data.

    For machines that have both DAT and JSON traces, compare the value distributions
    for common sensors to detect if DAT import is losing calibration or applying wrong divisors.
    """
    try:
        df = get_merged_trace_index(output_dir)
    except FileNotFoundError:
        df = get_trace_index(output_dir)

    traces_dir = output_dir / "traces"
    findings = []

    # Find machines with both formats
    if "format" not in df.columns:
        return findings

    for machine_slug, group in df.groupby("machine_slug"):
        format_counts = group["format"].value_counts()
        if "json" not in format_counts or "dat" not in format_counts:
            continue

        # We have both formats for this machine
        json_traces = group[group["format"] == "json"]
        dat_traces = group[group["format"] == "dat"]

        # Sample traces from each format
        json_stats = _sample_sensor_stats(traces_dir, json_traces.head(5))
        dat_stats = _sample_sensor_stats(traces_dir, dat_traces.head(5))

        # Compare common sensors
        common_sensors = set(json_stats.keys()) & set(dat_stats.keys())
        for sensor in sorted(common_sensors):
            j = json_stats[sensor]
            d = dat_stats[sensor]

            # Check if medians differ by a factor
            if j["median"] != 0 and d["median"] != 0:
                ratio = d["median"] / j["median"]
                if ratio > 5 or ratio < 0.2:
                    findings.append({
                        "machine": machine_slug,
                        "sensor": sensor,
                        "json_median": j["median"],
                        "dat_median": d["median"],
                        "ratio_dat_json": ratio,
                        "issue": f"DAT/JSON median ratio = {ratio:.1f}; "
                                f"possible divisor mismatch",
                    })

    return findings


def _sample_sensor_stats(traces_dir: Path, traces_df: pd.DataFrame) -> dict[str, dict]:
    """Load a few traces and compute basic stats per sensor."""
    stats = {}
    sentinels = set(get_sentinel_values())

    for _, row in traces_df.iterrows():
        site_id = row.get("site_id", "")
        machine_slug = row.get("machine_slug", "")
        trace_id = row.get("trace_id", "")

        path = traces_dir / str(site_id) / str(machine_slug) / f"{trace_id}.parquet"
        if not path.exists():
            # Try finding the trace in merged trace_ids
            continue

        try:
            df = pd.read_parquet(path)
        except Exception:
            continue

        for col in df.select_dtypes(include=[np.number]).columns:
            if col in ("timestamp",):
                continue
            vals = df[col].dropna()
            vals = vals[~vals.isin(sentinels)]
            if len(vals) < 10:
                continue

            if col not in stats:
                stats[col] = {"values": []}
            stats[col]["values"].extend(vals.tolist()[:200])

    # Compute summary stats
    result = {}
    for sensor, data in stats.items():
        values = np.array(data["values"])
        if len(values) > 0:
            result[sensor] = {
                "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
    return result


def main():
    output_dir = OUTPUT_DIR
    if not output_dir.exists():
        print(f"ERROR: Output directory {output_dir} not found")
        sys.exit(1)

    print("=" * 80)
    print("DATA QUALITY AUDIT")
    print("=" * 80)

    # ── 1. Translation table ────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("1. GERMAN → ENGLISH SENSOR TRANSLATION TABLE")
    print("─" * 80)
    translation = load_sensor_translation_table()
    print(f"\n{len(translation)} sensors mapped in sensor_definitions.yaml")

    # Check which columns in actual data use English canonical names
    print("\nChecking actual parquet column labels...")
    actual_columns = check_column_labels(output_dir)
    english_names = {info["english"] for info in translation.values()}
    german_names = set(translation.keys())

    english_cols = []
    german_cols = []
    unknown_cols = []
    for col in sorted(actual_columns.keys()):
        if col in ("timestamp",):
            continue
        if col in english_names:
            english_cols.append(col)
        elif col in german_names:
            german_cols.append(col)
        else:
            unknown_cols.append(col)

    print(f"\nColumns found in data: {len(actual_columns)}")
    print(f"  English canonical: {len(english_cols)}")
    print(f"  Still German: {len(german_cols)}")
    print(f"  Unknown: {len(unknown_cols)}")
    if german_cols:
        print("\n  GERMAN COLUMNS STILL IN DATA (need migration):")
        for col in german_cols:
            examples = actual_columns[col]
            print(f"    '{col}' — found in {examples[0]}")
    if unknown_cols:
        print("\n  UNKNOWN COLUMNS (not in translation table):")
        for col in unknown_cols:
            examples = actual_columns[col]
            print(f"    '{col}' — found in {examples[0]}")

    all_english = len(german_cols) == 0 and len(unknown_cols) == 0
    print(f"\n  Data columns are in: {'ENGLISH (canonical)' if all_english else 'MIXED — run migration script'}")

    # ── 2. Calibration status per combination ───────────────────────────────
    print("\n" + "─" * 80)
    print("2. CALIBRATION STATUS PER (MACHINE, SITE, TECHNIQUE)")
    print("─" * 80)
    cal_df = report_calibration_by_combination(output_dir)

    if not cal_df.empty:
        for _, row in cal_df.iterrows():
            print(f"\n  {row['machine']} @ {row['site']} [{row['technique']}]")
            print(f"    Traces: {row['n_traces']}, Formats: {row['formats']}")
            print(f"    Sensors: {row['n_sensors']} total, "
                  f"{row['n_calibrated']} calibrated, "
                  f"{row['n_uncalibrated']} uncalibrated")
            if row['uncalibrated_sensors']:
                print(f"    UNCALIBRATED: {row['uncalibrated_sensors']}")

        # Summary table
        print("\n\n  SUMMARY TABLE:")
        summary = cal_df[["machine", "site", "technique", "n_traces", "n_sensors",
                          "n_calibrated", "n_uncalibrated"]].to_string(index=False)
        print(f"  {summary}")

    # ── 3. Gain factor / divisor audit ──────────────────────────────────────
    print("\n" + "─" * 80)
    print("3. GAIN FACTOR / DIVISOR AUDIT (all sensors)")
    print("─" * 80)
    print("\nScanning all trace files for potential gain/divisor issues...")

    findings = audit_gain_factors(output_dir)

    # Report only findings with issues
    issues = [f for f in findings if f["issues"]]
    clean = [f for f in findings if not f["issues"]]

    print(f"\nTotal (machine, sensor) pairs checked: {len(findings)}")
    print(f"  With issues: {len(issues)}")
    print(f"  Clean: {len(clean)}")

    if issues:
        print("\n  ISSUES FOUND:")
        for f in sorted(issues, key=lambda x: (x["machine"], x["sensor"])):
            print(f"\n  {f['machine']} / {f['sensor']} "
                  f"[{'calibrated' if f['calibrated'] else 'UNCALIBRATED'}]")
            print(f"    Range: [{f['min']:.2f}, {f['max']:.2f}], "
                  f"median={f['median']:.2f}, p5={f['p5']:.2f}, p95={f['p95']:.2f}")
            print(f"    Samples: {f['n_values']} from {f['n_files']} files")
            for issue in f["issues"]:
                print(f"    >>> {issue}")

    # ── 4. DAT vs JSON consistency ──────────────────────────────────────────
    print("\n" + "─" * 80)
    print("4. DAT vs JSON FORMAT CONSISTENCY CHECK")
    print("─" * 80)
    format_findings = check_dat_vs_json_consistency(output_dir)
    if format_findings:
        print(f"\n  {len(format_findings)} inconsistencies found:")
        for f in format_findings:
            print(f"\n  {f['machine']} / {f['sensor']}")
            print(f"    JSON median: {f['json_median']:.2f}")
            print(f"    DAT median:  {f['dat_median']:.2f}")
            print(f"    >>> {f['issue']}")
    else:
        print("\n  No DAT/JSON inconsistencies detected (or no machines with both formats)")

    # ── 5. Physical range summary ───────────────────────────────────────────
    print("\n" + "─" * 80)
    print("5. SENSOR PHYSICAL RANGE REFERENCE")
    print("─" * 80)
    print(f"\n{'Sensor':<35} {'Min':>8} {'Max':>8} {'Unit':>10}")
    print("─" * 65)
    for sensor, (lo, hi) in sorted(PHYSICAL_RANGES.items()):
        unit = SENSOR_UNITS.get(sensor, "?")
        print(f"  {sensor:<33} {lo:>8.0f} {hi:>8.0f} {unit:>10}")

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
