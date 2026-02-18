"""Parser for Bauer DAT (fixed-width) trace files.

DAT files use `$` as a section delimiter with the structure:
    {header}$BEGINNG...${EVT...}*${UTC...}$...${DAT...}*${ED...}$...

- Header: comma-separated field definitions (names, units, divisors, metadata)
- BEGINNG: 153-char fixed-width section with element, timestamp, technique, machine
- EVT: event records (operator, machine ID, etc.)
- DAT: fixed-width numeric data lines (one per time step)
- ED: end markers
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from macchine.models.core import SensorSeries, TraceMetadata
from macchine.parsers.beginng_parser import parse_beginng
from macchine.parsers.filename_parser import normalize_model, parse_dat_path

logger = logging.getLogger(__name__)


@dataclass
class DatFieldDef:
    """A field definition from the DAT header."""

    name: str
    unit: str
    divisor: int = 1
    signed: bool = False
    is_direction: bool = False
    meta: list[int] = field(default_factory=list)


def parse_dat_file(path: Path) -> tuple[TraceMetadata, list[SensorSeries]]:
    """Parse a DAT file into metadata and (best-effort) sensor series.

    Returns:
        Tuple of (TraceMetadata, list of SensorSeries).
        The sensor series may be empty if data line parsing fails.
    """
    with open(path, "rb") as f:
        raw = f.read()

    text = raw.decode("iso-8859-1")
    sections = text.split("$")

    if len(sections) < 2:
        raise ValueError(f"DAT file has too few sections: {path}")

    # Parse header
    field_defs = _parse_header(sections[0])

    # Parse BEGINNG
    beginng_info = {}
    for section in sections[1:5]:
        if section.startswith("BEGINN"):
            beginng_info = parse_beginng(section)
            break

    # Parse EVT records for operator and machine info
    evt_info = _parse_events(sections)

    # Merge path info
    path_info = parse_dat_path(path)

    # Build metadata
    metadata = _build_metadata(path, field_defs, beginng_info, evt_info, path_info)

    # Parse DAT data lines (best-effort)
    sensors = _parse_data_lines(sections, field_defs, metadata.start_time)

    # Auto-correct DAT divisor issues: some DAT files declare divisor=1
    # for sensors whose raw values are in sub-units (cm, 0.1 bar, etc.)
    sensors = _apply_scaling_corrections(sensors, metadata.machine_slug)

    metadata.sensor_count = len(field_defs)
    if sensors:
        metadata.sample_count = max(s.sample_count for s in sensors) if sensors else 0
        metadata.duration_s = max(s.duration_s for s in sensors) if sensors else 0.0

    return metadata, sensors


def _parse_header(header_text: str) -> list[DatFieldDef]:
    """Parse the comma-separated header into field definitions."""
    # Tokenize respecting quoted strings
    tokens = []
    in_quote = False
    current = ""
    for c in header_text:
        if c == "'":
            in_quote = not in_quote
            current += c
        elif c == "," and not in_quote:
            tokens.append(current.strip())
            current = ""
        else:
            current += c
    if current.strip():
        tokens.append(current.strip())

    # Remove copyright tail
    if tokens and "copyright" in tokens[-1].lower():
        tokens = tokens[:-1]

    if len(tokens) < 8:
        return []

    num_fields = int(tokens[2])
    fields = []

    i = 7  # skip preamble (buffer_size, version, num_fields, 4 more values)
    while i + 8 < len(tokens) and len(fields) < num_fields:
        if tokens[i + 1].startswith("'"):
            divisor = int(tokens[i])
            name = tokens[i + 1].strip("'")
            unit = tokens[i + 2].strip("'").strip("[]")
            meta = []
            for j in range(i + 3, min(i + 9, len(tokens))):
                try:
                    meta.append(int(tokens[j]))
                except ValueError:
                    break

            signed = meta[0] == 1 if meta else False
            is_direction = meta[0] == 2 if meta else False

            fields.append(DatFieldDef(
                name=name,
                unit=unit,
                divisor=divisor,
                signed=signed,
                is_direction=is_direction,
                meta=meta,
            ))
            i += 9
        else:
            i += 1

    return fields


def _parse_events(sections: list[str]) -> dict:
    """Extract operator and machine info from EVT sections."""
    result = {}
    for section in sections:
        if not section.startswith("EVT"):
            continue
        # EVT format: "EVT{DDMMYYHHMMSS}{value_padded}{event_code}"
        # Event code 10200 = operator name
        # Event code 10210 = machine code
        if len(section) < 45:
            continue
        event_code = section[-5:]
        value = section[15:-5].strip()

        if event_code == "10200" and value:
            result["operator"] = value
        elif event_code == "10210" and value:
            result["machine_event_code"] = value
    return result


def _build_metadata(
    path: Path,
    field_defs: list[DatFieldDef],
    beginng: dict,
    evt: dict,
    path_info: dict,
) -> TraceMetadata:
    """Build TraceMetadata from all parsed sources."""
    # Machine info: prefer filename > BEGINNG > EVT
    machine_serial = path_info.get("machine_serial", "")
    machine_model = path_info.get("machine_model", "")
    machine_number = path_info.get("machine_number")
    machine_slug = path_info.get("machine_slug", "")

    if not machine_model and beginng.get("machine_model_code"):
        machine_model = beginng["machine_model_code"]
    if not machine_number and beginng.get("machine_number"):
        machine_number = beginng["machine_number"]
    if machine_model and machine_number and not machine_slug:
        machine_slug = f"{normalize_model(machine_model)}_{machine_number}"

    # Element name: prefer BEGINNG (from inside the file) over filename-derived
    element_name = beginng.get("element_name", "") or path_info.get("element_name", "")

    # Technique: header fields give the best signal; BEGINNG as fallback
    technique_from_fields = _infer_technique_from_fields(field_defs)
    technique_from_beginng = beginng.get("technique", "")
    # BauerFIRE is generic (used for KELLY, SOB, SCM), so prefer field inference
    technique = technique_from_fields or technique_from_beginng

    # Timestamp
    start_time = beginng.get("timestamp") or path_info.get("filename_timestamp")

    # Operator
    operator = evt.get("operator", "")

    return TraceMetadata(
        source_path=path,
        format="dat",
        element_name=element_name,
        site_id=path_info.get("site_id", ""),
        machine_serial=machine_serial,
        machine_model=machine_model,
        machine_number=machine_number,
        machine_slug=machine_slug,
        technique=technique,
        start_time=start_time,
        operator=operator,
    )


def _infer_technique_from_fields(field_defs: list[DatFieldDef]) -> str:
    """Infer technique from the set of sensor fields present."""
    names = {f.name.lower() for f in field_defs}
    if any("fräs" in n or "fraes" in n or "dws" in n for n in names):
        return "CUT"
    if any("susp" in n for n in names):
        return "SCM"
    if "betondruck" in names or "betonmenge" in names:
        # SOB or KELLY (both use concrete)
        if any("seilkraft hauptwinde" in n for n in names):
            return "KELLY"
        return "SOB"
    if any("seilkraft" in n for n in names):
        return "GRAB"
    return ""


def _parse_data_lines(
    sections: list[str],
    field_defs: list[DatFieldDef],
    start_time: datetime | None,
) -> list[SensorSeries]:
    """Best-effort parsing of DAT data lines into sensor series.

    Uses sign characters (+/-) and the direction field as landmarks
    to split fixed-width data lines into fields.
    """
    dat_sections = [s for s in sections if s.startswith("DAT")]
    if not dat_sections or not field_defs:
        return []

    num_fields = len(field_defs)
    data_length = len(dat_sections[0]) - 3  # minus "DAT" prefix

    # Determine field widths using the meta[2] + sign heuristic
    widths = _estimate_field_widths(field_defs, data_length)
    if not widths:
        logger.debug("Could not determine field widths for %d fields, %d chars", num_fields, data_length)
        return []

    # Parse all data lines
    all_values: list[list[float | None]] = [[] for _ in range(num_fields)]

    for dat_section in dat_sections:
        data = dat_section[3:]  # strip "DAT"
        if len(data) != data_length:
            continue

        pos = 0
        for i, (fdef, w) in enumerate(zip(field_defs, widths)):
            if pos + w > len(data):
                break
            raw = data[pos : pos + w]
            pos += w

            try:
                if fdef.is_direction:
                    val = {"R": 1.0, "L": -1.0, " ": 0.0}.get(raw.strip(), 0.0)
                else:
                    val = float(raw)
                    if fdef.divisor and fdef.divisor != 1:
                        val /= fdef.divisor
            except (ValueError, TypeError):
                val = None

            all_values[i].append(val)

    # Convert to SensorSeries
    sensors = []
    for i, fdef in enumerate(field_defs):
        values = all_values[i]
        if not values:
            continue
        sensors.append(SensorSeries(
            sensor_name=fdef.name,
            unit=fdef.unit,
            start_time=start_time,
            interval_ms=1000,
            values=values,
        ))

    return sensors


def _apply_scaling_corrections(
    sensors: list[SensorSeries], machine_slug: str
) -> list[SensorSeries]:
    """Auto-correct missing divisors in DAT-parsed sensor values.

    Uses physical range heuristics to detect when raw integer values
    are in sub-units (centimeters, decibar, etc.) and applies the
    appropriate divisor.
    """
    from macchine.harmonize.calibration import detect_dat_divisor, is_calibrated

    for sensor in sensors:
        # Only correct sensors with known physical ranges that are considered calibrated
        if machine_slug and not is_calibrated(sensor.sensor_name, machine_slug):
            continue

        divisor = detect_dat_divisor(sensor.values, sensor.sensor_name)
        if divisor > 1:
            sensor.values = [
                v / divisor if v is not None else None for v in sensor.values
            ]
            logger.info(
                "Auto-corrected %s: applied ÷%d (raw values in sub-units)",
                sensor.sensor_name,
                divisor,
            )

    return sensors


def _estimate_field_widths(field_defs: list[DatFieldDef], data_length: int) -> list[int] | None:
    """Estimate field widths from header metadata and data line length.

    Strategy: use meta[2] as base width, with adjustments to match data_length.
    Direction fields (meta[0]==2) are always 1 char.
    """
    num_fields = len(field_defs)

    # Start with meta[2] as base width for each field
    base_widths = []
    for f in field_defs:
        if f.is_direction:
            base_widths.append(1)
        elif f.meta and len(f.meta) > 2:
            base_widths.append(f.meta[2])
        else:
            base_widths.append(3)  # fallback

    base_total = sum(base_widths)

    if base_total == data_length:
        return base_widths

    # Try adding 1 to signed fields (for the sign character)
    adjusted = list(base_widths)
    for i, f in enumerate(field_defs):
        if f.signed and not f.is_direction:
            adjusted[i] += 1

    if sum(adjusted) == data_length:
        return adjusted

    # If still not matching, try uniform distribution of the remaining chars
    diff = data_length - sum(adjusted)
    if abs(diff) <= num_fields:
        # Distribute difference across non-direction fields
        non_dir = [i for i, f in enumerate(field_defs) if not f.is_direction]
        if non_dir and diff != 0:
            step = 1 if diff > 0 else -1
            for j in range(abs(diff)):
                adjusted[non_dir[j % len(non_dir)]] += step
        if sum(adjusted) == data_length:
            return adjusted

    logger.debug(
        "Width estimation failed: %d fields, data_length=%d, base=%d, adjusted=%d",
        num_fields, data_length, base_total, sum(adjusted),
    )
    return None
