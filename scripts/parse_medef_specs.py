"""Parse Bauer MEDEF format PDFs into structured YAML.

Reads all MEDEF-Format*.pdf files in specs_data/ via `pdftotext -layout`,
extracts the [$DAT] sensor table from each, and writes a structured YAML
file to specs_data/medef_specs.yaml.

Each entry contains: technique, MEDEF version, and per-sensor specs
(characters, decimal places, signed, format, unit, expected_divisor).

Usage:
    uv run python scripts/parse_medef_specs.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import yaml

SPECS_DIR = Path(__file__).resolve().parent.parent / "specs_data"
OUTPUT_PATH = SPECS_DIR / "medef_specs.yaml"


def extract_pdf_text(pdf_path: Path) -> str:
    """Run pdftotext -layout on a PDF and return the text."""
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  WARNING: pdftotext failed for {pdf_path.name}: {result.stderr}")
        return ""
    return result.stdout


def extract_technique_and_version(text: str) -> tuple[str, int]:
    """Extract technique name and MEDEF version from PDF header.

    Looks for a line like "KELLY           MEDEF 8" near the top.
    """
    # Match lines like "KELLY           MEDEF 8" or "Seilgreifer           MEDEF 7"
    # The technique name and MEDEF version are on the same line, separated by whitespace
    for line in text.split("\n")[:30]:
        m = re.search(r"^\s+(\S.*?)\s+MEDEF\s+(\d+)\s", line)
        if m:
            technique = m.group(1).strip()
            version = int(m.group(2))
            return technique, version
    return "", 0


def extract_dat_technique(text: str) -> str:
    """Extract technique name from the [$DAT] section header.

    Looks for a line like "[$DAT] KELLY" or "[$DAT] Seilgreifer".
    """
    m = re.search(r"\[\$DAT\]\s+(.+)", text)
    if m:
        return m.group(1).strip()
    return ""


def parse_sensor_table(text: str) -> list[dict]:
    """Parse the sensor table from the [$DAT] section.

    The table has columns: Name, Type, Signed, Characters, Decimal, Format, Unit, Comment
    separated by variable whitespace.
    """
    # Find the [$DAT] section
    dat_match = re.search(r"\[\$DAT\]", text)
    if not dat_match:
        return []

    dat_text = text[dat_match.start():]

    # Find the table header line — search for the full line containing the header
    header_match = re.search(
        r"Name\s+Type\s+Signed\s+Characters\s+Decimal\s+Format\s+Unit",
        dat_text,
    )
    if not header_match:
        return []

    # Get the FULL line containing the header (including leading whitespace)
    # This is needed so column positions are absolute within the line
    header_abs_start = dat_text.rfind("\n", 0, header_match.start())
    if header_abs_start < 0:
        header_abs_start = 0
    else:
        header_abs_start += 1  # skip the newline itself

    header_end = dat_text.index("\n", header_match.start())
    full_header_line = dat_text[header_abs_start:header_end]

    col_positions = {}
    for col_name in ["Name", "Type", "Signed", "Characters", "Decimal", "Format", "Unit", "Comment"]:
        idx = full_header_line.find(col_name)
        if idx >= 0:
            col_positions[col_name] = idx

    # Parse lines after the header until we hit an empty section or [$...] marker
    table_start = header_end + 1
    lines = dat_text[table_start:].split("\n")

    sensors = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Stop at section markers like [$BEGINN], [$BWE], or end of content
        if re.match(r"\s*\[\$", line):
            break

        # Stop at copyright/footer
        if "© Bauer" in line or "Data protection" in line:
            break

        # Skip empty lines, pattern lines, validation lines
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("$DAT") or stripped.startswith("enter element"):
            i += 1
            continue

        # Try to parse as a sensor row - must have Name column content
        # and Type column should be one of: String, Signed, Unsigned, Numeric
        sensor = _try_parse_sensor_row(line, col_positions)
        if sensor:
            # Check for continuation lines (format string split across lines)
            while i + 1 < len(lines):
                next_line = lines[i + 1]
                next_stripped = next_line.strip()
                if not next_stripped:
                    i += 1
                    continue
                # Continuation line: starts with whitespace and contains format-like text
                # or just additional comment text, but NOT a new sensor row
                if _try_parse_sensor_row(next_line, col_positions):
                    break  # Next sensor row
                if re.match(r"\s*\[\$", next_line):
                    break
                if "© Bauer" in next_line:
                    break
                # It's a continuation line - absorb it
                i += 1
            sensors.append(sensor)
        i += 1

    return sensors


def _try_parse_sensor_row(line: str, col_positions: dict) -> dict | None:
    """Try to parse a line as a sensor table row.

    Returns sensor dict or None if the line doesn't look like a sensor row.
    """
    if len(line) < 20:
        return None

    # The Name column starts at col_positions["Name"]
    name_start = col_positions.get("Name", 0)
    type_start = col_positions.get("Type", 0)
    signed_start = col_positions.get("Signed", 0)
    chars_start = col_positions.get("Characters", 0)
    decimal_start = col_positions.get("Decimal", 0)
    format_start = col_positions.get("Format", 0)
    unit_start = col_positions.get("Unit", 0)
    comment_start = col_positions.get("Comment", len(line))

    # Pad line if needed
    padded = line.ljust(max(col_positions.values()) + 50 if col_positions else len(line) + 50)

    # Extract fields by column position
    name = padded[name_start:type_start].strip()
    type_val = padded[type_start:signed_start].strip()
    signed_val = padded[signed_start:chars_start].strip()
    chars_val = padded[chars_start:decimal_start].strip()
    decimal_val = padded[decimal_start:format_start].strip()

    if comment_start > 0 and comment_start < len(padded):
        format_val = padded[format_start:unit_start].strip()
        unit_val = padded[unit_start:comment_start].strip()
    else:
        format_val = padded[format_start:unit_start].strip()
        unit_val = padded[unit_start:].strip()

    # Validate: must have a name and valid type
    if not name:
        return None
    valid_types = {"String", "Signed", "Unsigned", "Numeric"}
    if type_val not in valid_types:
        return None

    # Parse numeric fields
    try:
        signed = int(signed_val) if signed_val else 0
        characters = int(chars_val) if chars_val else 0
        decimal = int(decimal_val) if decimal_val else 0
    except ValueError:
        return None

    # Clean up format and unit
    format_val = format_val.strip("[] \t")
    unit_val = unit_val.strip()
    # Remove trailing comment text from unit
    if "  " in unit_val:
        unit_val = unit_val.split("  ")[0].strip()

    # Compute expected divisor
    expected_divisor = 10 ** decimal if decimal > 0 else 1

    return {
        "name": name,
        "type": type_val,
        "signed": signed == 1,
        "characters": characters,
        "decimal": decimal,
        "format": format_val if format_val else None,
        "unit": unit_val if unit_val else None,
        "expected_divisor": expected_divisor,
    }


def make_spec_key(technique: str, version: int, filename: str) -> str:
    """Create a unique key for this spec entry.

    Uses technique name and version. Falls back to filename-based key
    for disambiguation.
    """
    # Normalize technique name for use as YAML key
    tech_clean = technique.replace(" ", "_").replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    return f"{tech_clean}_v{version}"


def parse_all_pdfs() -> dict:
    """Parse all MEDEF PDF specs and return structured data."""
    pdf_files = sorted(SPECS_DIR.glob("MEDEF-Format*.pdf"))
    if not pdf_files:
        print("ERROR: No MEDEF-Format*.pdf files found in specs_data/")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF files in {SPECS_DIR}")

    all_specs = {}
    seen_keys = {}  # Track duplicates

    for pdf_path in pdf_files:
        print(f"\nParsing: {pdf_path.name}")

        text = extract_pdf_text(pdf_path)
        if not text:
            continue

        # Extract metadata
        header_technique, version = extract_technique_and_version(text)
        dat_technique = extract_dat_technique(text)
        technique = dat_technique or header_technique

        if not technique or not version:
            print(f"  WARNING: Could not determine technique/version")
            print(f"    header_technique={header_technique!r}, version={version}")
            print(f"    dat_technique={dat_technique!r}")
            continue

        print(f"  Technique: {technique}, MEDEF version: {version}")

        # Parse sensor table
        sensors = parse_sensor_table(text)
        print(f"  Sensors found: {len(sensors)}")

        if not sensors:
            print(f"  WARNING: No sensors parsed from {pdf_path.name}")
            continue

        # Create spec key
        spec_key = make_spec_key(technique, version, pdf_path.name)

        # Handle duplicates (e.g., kelly7.pdf and Keyy.pdf both being KELLY_v8)
        if spec_key in seen_keys:
            existing_file = seen_keys[spec_key]
            existing_count = len(all_specs[spec_key]["sensors"])
            # Keep the one with more sensors, or append filename suffix
            if len(sensors) > existing_count:
                print(f"  Replacing {spec_key} (was from {existing_file} with {existing_count} sensors)")
            elif len(sensors) == existing_count:
                print(f"  Duplicate of {spec_key} (from {existing_file}, same sensor count) — skipping")
                continue
            else:
                print(f"  Skipping duplicate {spec_key} (from {existing_file} has {existing_count} vs {len(sensors)} sensors)")
                continue

        seen_keys[spec_key] = pdf_path.name

        # Build sensors dict
        sensors_dict = {}
        for s in sensors:
            sensor_entry = {
                "characters": s["characters"],
                "decimal": s["decimal"],
                "signed": s["signed"],
                "unit": s["unit"],
                "expected_divisor": s["expected_divisor"],
            }
            if s["format"]:
                sensor_entry["format"] = s["format"]
            sensors_dict[s["name"]] = sensor_entry

        all_specs[spec_key] = {
            "technique": technique,
            "medef_version": version,
            "source_file": pdf_path.name,
            "sensors": sensors_dict,
        }

    return all_specs


# Machine → technique mapping
# Determined from BEGINNG technique fields and sensor signatures in actual DAT files
MACHINE_TECHNIQUE_MAP = {
    "bg28h_6061": ["KELLY_v8"],
    "bg30v_2872": ["Seilgreifer_v7", "KELLY_v8", "SOB_v7"],
    "bg33v_5610": ["KELLY_v8", "SOB_v7", "CSM_v7"],
    "bg42v_5925": ["Seilgreifer_v7", "KELLY_v8"],
    "bg45v_4027": ["KELLY_v8", "SOB_v7", "Seilgreifer_v7"],
    "gb50_601": ["Seilgreifer_v7"],
    "mc86_621": ["CUT_v8", "Fraese_HDS_v7"],
    "cube0_482": ["CUT_v8", "Fraese_HDS_v7"],
}


def write_yaml(specs: dict) -> None:
    """Write the parsed specs to YAML."""
    # Add machine_technique_map
    output = dict(specs)
    output["machine_technique_map"] = MACHINE_TECHNIQUE_MAP

    # Custom YAML formatting for readability
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    class YamlDumper(yaml.SafeDumper):
        pass

    # Preserve dict ordering
    def dict_representer(dumper, data):
        return dumper.represent_mapping("tag:yaml.org,2002:map", data.items())

    YamlDumper.add_representer(dict, dict_representer)

    with open(OUTPUT_PATH, "w") as f:
        f.write("# MEDEF Spec Definitions — parsed from Bauer PDF documentation\n")
        f.write("# Generated by: python scripts/parse_medef_specs.py\n")
        f.write("#\n")
        f.write("# For each technique+version: the official sensor channel definitions.\n")
        f.write("# expected_divisor = 10^decimal — what the spec says the divisor should be.\n")
        f.write("# NOTE: Actual firmware often encodes at much finer resolution.\n")
        f.write("#       See output/diagnostics/spec_validation_report.txt for details.\n\n")
        yaml.dump(output, f, Dumper=YamlDumper, default_flow_style=False,
                  allow_unicode=True, sort_keys=False, width=120)

    print(f"\nWrote {OUTPUT_PATH}")


def main():
    specs = parse_all_pdfs()

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {len(specs)} technique specs parsed")
    print(f"{'=' * 60}")

    total_sensors = 0
    for key, spec in sorted(specs.items()):
        n = len(spec["sensors"])
        total_sensors += n
        print(f"  {key}: {spec['technique']} (MEDEF {spec['medef_version']}) — {n} sensors")

    print(f"\nTotal: {total_sensors} sensor definitions across {len(specs)} specs")

    write_yaml(specs)


if __name__ == "__main__":
    main()
