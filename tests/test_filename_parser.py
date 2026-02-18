"""Tests for filename and path parser."""

from __future__ import annotations

from datetime import datetime

import pytest

from macchine.parsers.filename_parser import (
    normalize_model,
    parse_dat_filename,
    parse_json_filename,
    parse_machine_dir,
    parse_site_dir,
)


class TestSiteDir:
    def test_numeric_with_name(self):
        result = parse_site_dir("2026-02-16_1508 - VICENZA")
        assert result["site_id"] == "1508"
        assert result["name"] == "VICENZA"

    def test_numeric_no_separator(self):
        result = parse_site_dir("2026-02-16_1427TriesteFerrier")
        assert result["site_id"] == "1427"
        assert result["name"] == "TriesteFerrier"

    def test_nonnumeric(self):
        result = parse_site_dir("2026-02-16_CS-Antwerpen")
        assert result["site_id"] == "CS-Antwerpen"

    def test_with_spaces(self):
        result = parse_site_dir("2026-02-16_1454 - Paris L18.3 OA20")
        assert result["site_id"] == "1454"
        assert "Paris" in result["name"]

    def test_numeric_no_name(self):
        result = parse_site_dir("2026-02-16_1501")
        assert result["site_id"] == "1501"


class TestMachineDir:
    def test_standard_format(self):
        result = parse_machine_dir("BG-33-V #5610 | 01K00044171")
        assert result["machine_model"] == "BG-33-V"
        assert result["machine_number"] == 5610
        assert result["machine_serial"] == "01K00044171"
        assert result["machine_slug"] == "bg33v_5610"

    def test_null_in_model(self):
        result = parse_machine_dir("MC-86-null 621 | 01K00046811")
        assert result["machine_model"] == "MC-86"
        assert result["machine_number"] == 621
        assert result["machine_serial"] == "01K00046811"

    def test_gb50_null(self):
        result = parse_machine_dir("GB-50-null 601 | 01K00047564")
        assert result["machine_model"] == "GB-50"
        assert result["machine_number"] == 601

    def test_unidentified(self):
        result = parse_machine_dir("Unidentified")
        assert result == {}

    def test_bare_serial(self):
        result = parse_machine_dir("01K00033511")
        assert result["machine_serial"] == "01K00033511"


class TestJsonFilename:
    def test_standard(self):
        result = parse_json_filename("palo 96 dms_202407180938.json")
        assert result["element_name"] == "palo 96 dms"
        assert result["filename_timestamp"] == datetime(2024, 7, 18, 9, 38)

    def test_numeric_element(self):
        result = parse_json_filename("33_202409261903.json")
        assert result["element_name"] == "33"

    def test_xxxxx(self):
        result = parse_json_filename("xxxxx_202409021230.json")
        assert result["element_name"] == "xxxxx"

    def test_pisa_style(self):
        result = parse_json_filename("P-2_202410150905.json")
        assert result["element_name"] == "P-2"
        assert result["filename_timestamp"] == datetime(2024, 10, 15, 9, 5)


class TestDatFilename:
    def test_pattern_a_with_serial(self):
        result = parse_dat_filename("01K00040846_cube0_482_20260212_093428_00001287_DPW008.dat")
        assert result["machine_serial"] == "01K00040846"
        assert result["machine_model"] == "cube0"
        assert result["machine_number"] == 482
        assert result["element_name"] == "DPW008"
        assert result["filename_timestamp"] == datetime(2026, 2, 12, 9, 34, 28)

    def test_pattern_b_without_serial(self):
        result = parse_dat_filename("bg33v_5610_20240613_140339_00002593_palo_167_dms_.dat")
        assert result["machine_model"] == "bg33v"
        assert result["machine_number"] == 5610
        assert result["element_name"] == "palo 167 dms"

    def test_pattern_b_bc5x_alias(self):
        result = parse_dat_filename("bc5x_482_20260126_060642_00001259_dpw001xx_.dat")
        assert result["machine_model"] == "cube0"
        assert result["machine_number"] == 482


class TestModelAliases:
    def test_bc5x(self):
        assert normalize_model("bc5x") == "cube0"

    def test_bg33v(self):
        assert normalize_model("bg33v") == "bg33v"

    def test_case_insensitive(self):
        assert normalize_model("BG-33-V") == "bg33v"
