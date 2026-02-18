"""Core data models for the Bauer fleet analysis platform."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path


class MachineType(Enum):
    BG = "BG"
    GB = "GB"
    MC = "MC"
    CUBE = "CUBE"
    UNKNOWN = "UNKNOWN"


@dataclass
class Machine:
    """A Bauer construction machine."""

    serial_id: str  # e.g. "01K00044171"
    model_name: str  # e.g. "BG-33-V"
    machine_number: int  # e.g. 5610
    machine_type: MachineType = MachineType.UNKNOWN
    filename_code: str = ""  # e.g. "bg33v" — lowercase slug for filenames

    def __post_init__(self):
        if not self.filename_code:
            self.filename_code = self.model_name.lower().replace("-", "").replace(" ", "")
        if self.machine_type == MachineType.UNKNOWN:
            self.machine_type = self._infer_type()

    def _infer_type(self) -> MachineType:
        upper = self.model_name.upper()
        if upper.startswith("BG"):
            return MachineType.BG
        if upper.startswith("GB"):
            return MachineType.GB
        if upper.startswith("MC"):
            return MachineType.MC
        if "CUBE" in upper or "BC" in upper:
            return MachineType.CUBE
        return MachineType.UNKNOWN

    @property
    def slug(self) -> str:
        """Short unique identifier like 'bg33v_5610'."""
        return f"{self.filename_code}_{self.machine_number}"


@dataclass
class Site:
    """A construction site where machines operate."""

    site_id: str  # e.g. "1508"
    name: str  # e.g. "VICENZA"
    raw_dir_name: str  # e.g. "2026-02-16_1508 - VICENZA"
    country: str = ""
    city: str = ""
    date_range: tuple[datetime | None, datetime | None] = (None, None)
    json_count: int = 0
    dat_count: int = 0
    techniques: list[str] = field(default_factory=list)


@dataclass
class Element:
    """A construction element (pile, wall panel, etc.)."""

    name: str  # e.g. "palo 96 dms"
    element_id: int | None = None  # from JSON elementId
    site_id: str = ""
    machine_slug: str = ""
    technique: str = ""


@dataclass
class SensorSeries:
    """A single sensor time series from a trace."""

    sensor_name: str  # original name, e.g. "Drehmoment"
    unit: str  # e.g. "kNm"
    start_time: datetime | None = None
    interval_ms: int = 1000
    values: list[float] = field(default_factory=list)
    canonical_name: str = ""  # English name, filled by harmonization

    @property
    def sample_count(self) -> int:
        return len(self.values)

    @property
    def duration_s(self) -> float:
        return len(self.values) * self.interval_ms / 1000.0


@dataclass
class TraceMetadata:
    """Metadata for a single trace file (one element recording)."""

    source_path: Path
    format: str  # "json" or "dat"
    element_name: str = ""
    element_id: int | None = None
    site_id: str = ""
    machine_serial: str = ""
    machine_model: str = ""
    machine_number: int | None = None
    machine_slug: str = ""
    technique: str = ""
    start_time: datetime | None = None
    upload_time: datetime | None = None
    medef_version: int | None = None
    operator: str = ""
    sensor_count: int = 0
    sample_count: int = 0
    duration_s: float = 0.0

    @property
    def trace_id(self) -> str:
        """Unique identifier derived from source path stem."""
        return self.source_path.stem
