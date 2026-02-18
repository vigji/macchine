"""Parser for MEDEF v7/v8 JSON trace files."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from macchine.models.core import SensorSeries, TraceMetadata
from macchine.parsers.filename_parser import parse_json_path


def parse_json_file(path: Path) -> tuple[TraceMetadata, list[SensorSeries]]:
    """Parse a MEDEF JSON file into metadata and sensor series.

    Args:
        path: Path to the JSON file.

    Returns:
        Tuple of (TraceMetadata, list of SensorSeries).
    """
    with open(path) as f:
        data = json.load(f)

    metadata = _extract_metadata(data, path)
    sensors = _extract_sensors(data)

    metadata.sensor_count = len(sensors)
    if sensors:
        metadata.sample_count = max(s.sample_count for s in sensors)
        metadata.duration_s = max(s.duration_s for s in sensors)

    return metadata, sensors


def _parse_timestamp(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        # Handle "2024-07-18T08:56:41Z" and "2024-07-18T08:56:41"
        ts = ts.rstrip("Z")
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def _extract_metadata(data: dict, path: Path) -> TraceMetadata:
    """Extract metadata from JSON content and file path."""
    # Parse what we can from the file path
    path_info = parse_json_path(path)

    # Extract operator
    operator = ""
    operators = data.get("operators", [])
    if operators and isinstance(operators, list):
        op = operators[0]
        if isinstance(op, dict):
            operator = op.get("operatorName", "")

    start_time = _parse_timestamp(data.get("startTimeOfProduction"))
    upload_time = _parse_timestamp(data.get("fileUploadTime"))

    element_name = data.get("name", "")
    # Treat "xxxxx" names as unnamed
    if element_name and all(c == "x" for c in element_name):
        element_name = ""

    return TraceMetadata(
        source_path=path,
        format="json",
        element_name=element_name or path_info.get("element_name", ""),
        element_id=data.get("elementId"),
        site_id=path_info.get("site_id", ""),
        machine_serial=path_info.get("machine_serial", ""),
        machine_model=path_info.get("machine_model", ""),
        machine_number=path_info.get("machine_number"),
        machine_slug=path_info.get("machine_slug", ""),
        technique=data.get("technique", ""),
        start_time=start_time,
        upload_time=upload_time,
        medef_version=data.get("medefVersion"),
        operator=operator,
    )


def _extract_sensors(data: dict) -> list[SensorSeries]:
    """Extract all sensor time series from JSON."""
    sensors = []
    tsb = data.get("timeSeriesBlock")
    if not tsb:
        return sensors

    for block_name in ("serialValuesFree", "serialValuesPredefinedCfa", "serialValuesPredefinedKelly"):
        for series_data in tsb.get(block_name, []):
            sensor = _parse_series(series_data)
            if sensor is not None:
                sensors.append(sensor)

    return sensors


def _parse_series(series_data: dict) -> SensorSeries | None:
    """Parse a single sensor series from JSON."""
    values = series_data.get("values", [])
    if not values:
        return None

    return SensorSeries(
        sensor_name=series_data.get("seriesName", ""),
        unit=series_data.get("unitOfMeasurement", ""),
        start_time=_parse_timestamp(series_data.get("startTime")),
        interval_ms=series_data.get("timeInterval", 1000),
        values=values,
    )
