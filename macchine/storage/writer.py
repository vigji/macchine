"""Convert parsed traces to Parquet files."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from macchine.config import METADATA_DIR, TRACE_INDEX_FILE, TRACES_DIR
from macchine.harmonize.sensor_map import get_canonical_name
from macchine.models.core import SensorSeries, TraceMetadata
from macchine.parsers.json_parser import parse_json_file

logger = logging.getLogger(__name__)


def trace_to_dataframe(metadata: TraceMetadata, sensors: list[SensorSeries]) -> pd.DataFrame | None:
    """Convert sensor series to a wide-format DataFrame with a timestamp column."""
    if not sensors:
        return None

    max_len = max(len(s.values) for s in sensors)
    if max_len == 0:
        return None

    # Use the first sensor's start_time and interval for timestamps
    ref = sensors[0]
    if ref.start_time:
        timestamps = pd.date_range(start=ref.start_time, periods=max_len, freq=f"{ref.interval_ms}ms")
    else:
        timestamps = pd.RangeIndex(max_len)

    data = {"timestamp": timestamps}
    for s in sensors:
        col_name = get_canonical_name(s.sensor_name)
        # Handle duplicate sensor names by appending unit
        if col_name in data:
            col_name = f"{col_name}_{s.unit}"
        values = s.values
        if len(values) < max_len:
            values = values + [None] * (max_len - len(values))
        data[col_name] = values

    return pd.DataFrame(data)


def write_trace_parquet(
    output_dir: Path,
    metadata: TraceMetadata,
    sensors: list[SensorSeries],
) -> Path | None:
    """Write a single trace as a Parquet file."""
    df = trace_to_dataframe(metadata, sensors)
    if df is None:
        return None

    site_id = metadata.site_id or "unknown"
    machine_slug = metadata.machine_slug or "unknown"
    trace_id = metadata.trace_id

    out_path = output_dir / TRACES_DIR / site_id / machine_slug / f"{trace_id}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, engine="pyarrow", index=False)
    return out_path


def metadata_to_dict(meta: TraceMetadata) -> dict:
    """Convert TraceMetadata to a flat dict for the trace index."""
    return {
        "source_path": str(meta.source_path),
        "format": meta.format,
        "trace_id": meta.trace_id,
        "element_name": meta.element_name,
        "element_id": meta.element_id,
        "site_id": meta.site_id,
        "machine_serial": meta.machine_serial,
        "machine_model": meta.machine_model,
        "machine_number": meta.machine_number,
        "machine_slug": meta.machine_slug,
        "technique": meta.technique,
        "start_time": meta.start_time,
        "upload_time": meta.upload_time,
        "medef_version": meta.medef_version,
        "operator": meta.operator,
        "sensor_count": meta.sensor_count,
        "sample_count": meta.sample_count,
        "duration_s": meta.duration_s,
    }


def write_trace_index(output_dir: Path, records: list[dict]) -> Path:
    """Write the trace index as a single Parquet file."""
    index_path = output_dir / METADATA_DIR / TRACE_INDEX_FILE
    index_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_parquet(index_path, engine="pyarrow", index=False)
    return index_path


def discover_json_files(raw_dir: Path) -> list[Path]:
    """Find all JSON files in the raw data directory."""
    return sorted(raw_dir.rglob("*.json"))


def discover_dat_files(raw_dir: Path) -> list[Path]:
    """Find all DAT files in the raw data directory."""
    return sorted(raw_dir.rglob("*.dat"))


def run_conversion(
    raw_dir: Path,
    output_dir: Path,
    skip_existing: bool = True,
    fmt: str = "all",
) -> None:
    """Run the full conversion pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files: list[Path] = []
    dat_files: list[Path] = []
    if fmt in ("json", "all"):
        json_files = discover_json_files(raw_dir)
    if fmt in ("dat", "all"):
        dat_files = discover_dat_files(raw_dir)

    logger.info("Found %d JSON files and %d DAT files", len(json_files), len(dat_files))

    index_records = []
    errors = []

    # Convert JSON files
    for path in tqdm(json_files, desc="Converting JSON files"):
        try:
            metadata, sensors = parse_json_file(path)
            write_trace_parquet(output_dir, metadata, sensors)
            index_records.append(metadata_to_dict(metadata))
        except Exception as e:
            errors.append((path, str(e)))
            logger.error("Error parsing %s: %s", path, e)

    # Convert DAT files
    if dat_files:
        from macchine.parsers.dat_parser import parse_dat_file

        for path in tqdm(dat_files, desc="Converting DAT files"):
            try:
                metadata, sensors = parse_dat_file(path)
                if sensors:
                    write_trace_parquet(output_dir, metadata, sensors)
                index_records.append(metadata_to_dict(metadata))
            except Exception as e:
                errors.append((path, str(e)))
                logger.error("Error parsing DAT %s: %s", path, e)

    if index_records:
        index_path = write_trace_index(output_dir, index_records)
        logger.info("Wrote trace index with %d records to %s", len(index_records), index_path)

    # Write fleet registry
    _build_and_save_fleet(output_dir, index_records)

    # Summary
    print(f"\nConversion complete:")
    print(f"  JSON files processed: {len(json_files)}")
    print(f"  DAT files processed: {len(dat_files)}")
    print(f"  Traces indexed: {len(index_records)}")
    print(f"  Errors: {len(errors)}")
    if errors:
        print("\nErrors:")
        for path, err in errors[:20]:
            print(f"  {path.name}: {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")


def _build_and_save_fleet(output_dir: Path, index_records: list[dict]) -> None:
    """Build fleet registry from trace index and save."""
    from macchine.models.fleet import FleetAssignment, FleetRegistry
    from macchine.models.core import Machine, MachineType, Site
    from macchine.config import KNOWN_MACHINES, FLEET_REGISTRY_FILE

    registry = FleetRegistry()

    # Collect machines and sites from traces
    machines_seen: dict[str, dict] = {}  # serial → info
    sites_seen: dict[str, dict] = {}  # site_id → info
    assignments_map: dict[tuple[str, str], list] = {}  # (serial, site_id) → timestamps

    for rec in index_records:
        serial = rec.get("machine_serial", "")
        model = rec.get("machine_model", "")
        number = rec.get("machine_number")
        site_id = rec.get("site_id", "")
        start_time = rec.get("start_time")
        technique = rec.get("technique", "")

        if serial and serial not in machines_seen:
            machines_seen[serial] = {"model": model, "number": number or 0}
        elif serial and number and not machines_seen[serial].get("number"):
            machines_seen[serial] = {"model": model, "number": number}

        if site_id and site_id not in sites_seen:
            sites_seen[site_id] = {"techniques": set(), "json_count": 0, "dat_count": 0}
        if site_id:
            sites_seen[site_id]["techniques"].add(technique)
            if rec.get("format") == "json":
                sites_seen[site_id]["json_count"] += 1
            else:
                sites_seen[site_id]["dat_count"] += 1

        if serial and site_id:
            key = (serial, site_id)
            if key not in assignments_map:
                assignments_map[key] = []
            if start_time:
                assignments_map[key].append(start_time)

    # Also add known machines that might not appear in traces
    for serial, (model, number) in KNOWN_MACHINES.items():
        if serial not in machines_seen:
            machines_seen[serial] = {"model": model, "number": number}

    # Build registry
    for serial, info in machines_seen.items():
        m = Machine(
            serial_id=serial,
            model_name=info["model"],
            machine_number=info["number"] or 0,
        )
        registry.add_machine(m)

    for site_id, info in sites_seen.items():
        s = Site(
            site_id=site_id,
            name=site_id,
            raw_dir_name="",
            json_count=info["json_count"],
            dat_count=info["dat_count"],
            techniques=sorted(info["techniques"]),
        )
        registry.add_site(s)

    for (serial, site_id), timestamps in assignments_map.items():
        machine = registry.machines.get(serial)
        site = registry.sites.get(site_id)
        if machine and site:
            start = min(timestamps) if timestamps else None
            end = max(timestamps) if timestamps else None
            registry.add_assignment(FleetAssignment(
                machine=machine,
                site=site,
                start_date=start,
                end_date=end,
                trace_count=len(timestamps),
            ))

    # Save as JSON
    fleet_path = output_dir / METADATA_DIR / FLEET_REGISTRY_FILE
    fleet_path.parent.mkdir(parents=True, exist_ok=True)
    fleet_data = {
        "machines": {
            s: {"model": m.model_name, "number": m.machine_number, "type": m.machine_type.value, "slug": m.slug}
            for s, m in registry.machines.items()
        },
        "sites": {
            s_id: {"name": s.name, "json_count": s.json_count, "dat_count": s.dat_count, "techniques": s.techniques}
            for s_id, s in registry.sites.items()
        },
        "assignments": [
            {
                "machine_serial": a.machine.serial_id,
                "site_id": a.site.site_id,
                "start_date": a.start_date.isoformat() if a.start_date else None,
                "end_date": a.end_date.isoformat() if a.end_date else None,
                "trace_count": a.trace_count,
            }
            for a in registry.assignments
        ],
    }
    with open(fleet_path, "w") as f:
        json.dump(fleet_data, f, indent=2)

    print(f"\n{registry.summary()}")
