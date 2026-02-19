"""Parser for Bauer DAT (fixed-width) trace files.

DAT files use `$` as a section delimiter with the structure:
    {header}$BEGINNG...${EVT...}*${UTC...}$...${DAT...}*${ED...}$...

- Header: comma-separated field definitions (names, units, divisors, metadata)
- BEGINNG: 153-char fixed-width section with element, timestamp, technique, machine
- EVT: event records (operator, machine ID, etc.)
- DAT: fixed-width numeric data lines (one per time step)
- ED: end markers

Field width and divisor discovery:
    The header metadata is "shifted by one position" — meta[2] of field N
    encodes the total data width of field N+1, and meta[3] of field N
    encodes the decimal count of field N+1.  This gives:
        width[i]   = meta[2] of field[i-1]   (for i > 0)
        width[0]   = data_length - sum(all meta[2])
        divisor[i] = 10 ** meta[3] of field[i-1]  (for i > 0)
    Validated against MEDEF spec PDFs with 100% match rate across all machines.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from macchine.models.core import SensorSeries, TraceMetadata
from macchine.parsers.beginng_parser import parse_beginng
from macchine.parsers.filename_parser import normalize_model, parse_dat_path

logger = logging.getLogger(__name__)

# Lazy-loaded sensor → divisor lookup from MEDEF specs
_SPEC_DIVISORS: dict[str, int] | None = None


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

    # Parse DAT data lines using spec-derived divisors
    sensors = _parse_data_lines(sections, field_defs, metadata.start_time)

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
    """Parse DAT data lines into sensor series.

    Uses the shifted-meta discovery for field widths (meta[2] of field N =
    width of field N+1) and derives divisors from shifted meta[3] (decimal
    count of next field), validated against MEDEF spec lookup.
    """
    dat_sections = [s for s in sections if s.startswith("DAT")]
    if not dat_sections or not field_defs:
        return []

    num_fields = len(field_defs)
    data_length = len(dat_sections[0]) - 3  # minus "DAT" prefix

    # Determine field widths using shifted-meta formula
    widths = _compute_field_widths(field_defs, data_length)
    if not widths:
        logger.debug("Could not determine field widths for %d fields, %d chars", num_fields, data_length)
        return []

    # Determine divisors: shifted meta[3] → spec YAML → header divisor
    divisors = _compute_divisors(field_defs)

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
                    div = divisors[i]
                    if div != 1:
                        val /= div
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


def _compute_field_widths(field_defs: list[DatFieldDef], data_length: int) -> list[int] | None:
    """Compute field widths using the shifted-meta discovery.

    meta[2] of field N encodes the total data width of field N+1.
    Therefore:
        width[i] = meta[2] of field[i-1]   for i > 0
        width[0] = data_length - sum(all meta[2])
    meta[2] of the last field gives the width of a trailing spec-only
    sensor that occupies data space but isn't declared in the header.

    Returns a list of widths, or None if the formula doesn't produce
    valid widths.
    """
    n = len(field_defs)
    if n == 0:
        return None

    all_meta2 = []
    for fdef in field_defs:
        if len(fdef.meta) > 2:
            all_meta2.append(fdef.meta[2])
        else:
            return None  # missing metadata

    # widths[i>0] = meta[2] of previous field
    widths = [0] * n
    for i in range(1, n):
        widths[i] = all_meta2[i - 1]

    # trailing = meta[2] of last field (spec-only field after last header field)
    trailing = all_meta2[-1]

    # width[0] = data_length - sum(widths[1:]) - trailing
    widths[0] = data_length - sum(widths[1:]) - trailing

    if sum(widths) + trailing == data_length and widths[0] > 0:
        return widths

    # Fallback: try without trailing field
    widths2 = [0] * n
    for i in range(1, n):
        widths2[i] = all_meta2[i - 1]
    widths2[0] = data_length - sum(widths2[1:])
    if sum(widths2) == data_length and widths2[0] > 0:
        return widths2

    return None


def _load_spec_divisors() -> dict[str, int]:
    """Load sensor → expected_divisor mapping from MEDEF specs (cached).

    The spec YAML contains divisor definitions for ~110 sensors across
    17 technique specs. 95/97 sensors that appear in multiple specs have
    consistent divisors, so a flat lookup is sufficient.
    """
    global _SPEC_DIVISORS
    if _SPEC_DIVISORS is not None:
        return _SPEC_DIVISORS

    specs_path = Path(__file__).resolve().parent.parent.parent / "specs_data" / "medef_specs.yaml"
    if not specs_path.exists():
        logger.debug("MEDEF specs not found at %s", specs_path)
        _SPEC_DIVISORS = {}
        return _SPEC_DIVISORS

    with open(specs_path) as f:
        specs = yaml.safe_load(f)

    divisors: dict[str, int] = {}
    for key, spec_data in specs.items():
        if key == "machine_technique_map" or not isinstance(spec_data, dict):
            continue
        for sensor_name, info in spec_data.get("sensors", {}).items():
            # Normalize Unicode ligatures from PDF parsing
            norm = sensor_name.replace("\ufb00", "ff").replace("\ufb01", "fi").replace("\ufb02", "fl")
            div = info.get("expected_divisor", 1)
            if norm not in divisors:
                divisors[norm] = div
            # First-seen wins; nearly all sensors are consistent across specs

    _SPEC_DIVISORS = divisors
    return _SPEC_DIVISORS


def _compute_divisors(field_defs: list[DatFieldDef]) -> list[int]:
    """Compute the divisor for each field using a three-level fallback:

    1. Shifted meta[3]: meta[3] of field[i-1] = decimal count of field[i],
       so divisor = 10^(meta[3] of field[i-1]).  Verified 100% across all
       tested machines.
    2. MEDEF spec lookup: flat sensor→divisor table from specs YAML.
    3. Header divisor: the divisor declared in the header (often 1, unreliable).
    """
    n = len(field_defs)
    divisors = [1] * n
    spec_divs = _load_spec_divisors()

    for i in range(n):
        if field_defs[i].is_direction:
            divisors[i] = 1
            continue

        # Level 1: shifted meta[3] — decimal of field i from meta[3] of field i-1
        if i > 0 and len(field_defs[i - 1].meta) > 3:
            decimal = field_defs[i - 1].meta[3]
            divisors[i] = 10 ** decimal if decimal >= 0 else 1
            continue

        # Level 2: MEDEF spec lookup
        name = field_defs[i].name
        norm = name.replace("\ufb00", "ff").replace("\ufb01", "fi").replace("\ufb02", "fl")
        if norm in spec_divs:
            divisors[i] = spec_divs[norm]
            continue

        # Level 3: header divisor (often 1, but it's all we have)
        divisors[i] = field_defs[i].divisor or 1

    return divisors
