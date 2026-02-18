"""Paths, constants, and configuration."""

from __future__ import annotations

from pathlib import Path

# Default raw data location
DEFAULT_RAW_DIR = Path("/Users/vigji/Downloads/bauer")

# Output subdirectories
TRACES_DIR = "traces"
METADATA_DIR = "metadata"
TRACE_INDEX_FILE = "trace_index.parquet"
MERGED_TRACE_INDEX_FILE = "merged_trace_index.parquet"
FLEET_REGISTRY_FILE = "fleet_registry.json"
QUALITY_DIR = "quality"

# Known techniques
TECHNIQUES = {
    "SOB": "Continuous Flight Auger",
    "KELLY": "Kelly Drilling",
    "CUT": "Diaphragm Wall Cutter",
    "SCM": "Soil Cement Mixing",
    "GRAB": "Grab",
    "DMS": "Deep Mixing",
    "FREE": "Free Recording",
}

# Known machines (serial → model info)
KNOWN_MACHINES = {
    "01K00044171": ("BG-33-V", 5610),
    "01K00046456": ("BG-42-V", 5925),
    "01K00033511": ("BG-45-V", 4027),
    "01K00027551": ("BG-30-V", 2872),
    "01K00046811": ("MC-86", 621),
    "01K00047564": ("GB-50", 601),
    "01K00032425": ("BG-45-V", 3923),
    "01K00040846": ("CUBE0", 482),
}

# Sampling rate
DEFAULT_INTERVAL_MS = 1000  # 1 Hz
