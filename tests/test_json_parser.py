"""Tests for JSON parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from macchine.parsers.json_parser import parse_json_file


class TestJsonParserSCM:
    """Test parsing of SCM (Soil Cement Mixing) JSON file from Vicenza."""

    def test_parse_metadata(self, vicenza_scm_path):
        meta, sensors = parse_json_file(vicenza_scm_path)
        assert meta.format == "json"
        assert meta.technique == "SCM"
        assert meta.element_id == 39378552
        assert meta.element_name == "palo 96 dms"
        assert meta.medef_version == 8
        assert meta.operator == "MARIUS"
        assert meta.start_time is not None
        assert meta.start_time.year == 2024
        assert meta.start_time.month == 7

    def test_sensor_count(self, vicenza_scm_path):
        meta, sensors = parse_json_file(vicenza_scm_path)
        assert meta.sensor_count == len(sensors)
        assert meta.sensor_count > 20  # SCM has many sensors

    def test_sensor_names(self, vicenza_scm_path):
        _, sensors = parse_json_file(vicenza_scm_path)
        names = {s.sensor_name for s in sensors}
        assert "Tiefe" in names
        assert "Drehmoment" in names
        assert "Drehzahl" in names

    def test_sensor_values(self, vicenza_scm_path):
        _, sensors = parse_json_file(vicenza_scm_path)
        depth = next(s for s in sensors if s.sensor_name == "Tiefe")
        assert depth.unit == "m"
        assert depth.interval_ms == 1000
        assert len(depth.values) == 25  # small sample
        assert depth.values[0] == pytest.approx(0.01)

    def test_sample_count_and_duration(self, vicenza_scm_path):
        meta, sensors = parse_json_file(vicenza_scm_path)
        assert meta.sample_count == 25
        assert meta.duration_s == pytest.approx(25.0)


class TestJsonParserKELLY:
    """Test parsing of KELLY JSON file from Roma."""

    def test_parse_kelly_metadata(self, roma_kelly_path):
        meta, sensors = parse_json_file(roma_kelly_path)
        assert meta.technique == "KELLY"
        assert meta.element_name == "33"
        assert meta.medef_version == 8

    def test_kelly_has_predefined_sensors(self, roma_kelly_path):
        _, sensors = parse_json_file(roma_kelly_path)
        names = {s.sensor_name for s in sensors}
        # Kelly predefined sensors
        assert "Seilkraft Hauptwinde" in names
        assert "Drehmoment" in names
        assert "Tiefe" in names


class TestJsonParserEdgeCases:
    """Test edge cases."""

    def test_xxxxx_element_name_cleared(self, vicenza_xxxxx_path):
        meta, _ = parse_json_file(vicenza_xxxxx_path)
        # "xxxxx" should be treated as unnamed
        assert meta.element_name != "xxxxx"
