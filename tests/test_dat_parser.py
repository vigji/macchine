"""Tests for DAT parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from macchine.parsers.dat_parser import parse_dat_file
from macchine.parsers.beginng_parser import parse_beginng

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestDatParser:
    def test_parse_metadata(self):
        path = FIXTURES_DIR / "trieste_kelly.dat"
        meta, sensors = parse_dat_file(path)
        assert meta.format == "dat"
        assert meta.element_name == "A40"
        assert meta.machine_number == 6061
        assert meta.start_time is not None

    def test_sensor_count(self):
        path = FIXTURES_DIR / "trieste_kelly.dat"
        meta, sensors = parse_dat_file(path)
        assert meta.sensor_count > 15  # KELLY has many sensors

    def test_operator_extracted(self):
        path = FIXTURES_DIR / "trieste_kelly.dat"
        meta, _ = parse_dat_file(path)
        # Operator should be extracted from EVT 10200
        assert meta.operator != ""


class TestBeginngParser:
    def test_scm_beginng(self):
        section = "BEGINNA000100                   0palo 167 dms   130624140339                    BauerFIRE 1470                     000000                      BG33V_5610"
        result = parse_beginng(section)
        assert result["element_name"] == "palo 167 dms"
        assert result["machine_code"] == "BG33V_5610"
        assert result["machine_number"] == 5610
        assert result["timestamp"].year == 2024
        assert result["timestamp"].month == 6
        assert result["timestamp"].day == 13

    def test_cut_beginng(self):
        section = "BEGINN9000100                   0DPW005         140126141440                    BauerSOB                           000000                        BC5X_482"
        result = parse_beginng(section)
        assert result["element_name"] == "DPW005"
        assert result["machine_code"] == "BC5X_482"
        assert result["machine_number"] == 482
        assert result["technique_raw"] == "SOB"
