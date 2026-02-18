"""Shared fixtures for macchine tests."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir():
    return FIXTURES_DIR


@pytest.fixture
def vicenza_scm_path():
    return FIXTURES_DIR / "vicenza_scm.json"


@pytest.fixture
def roma_kelly_path():
    return FIXTURES_DIR / "roma_kelly.json"


@pytest.fixture
def vicenza_xxxxx_path():
    return FIXTURES_DIR / "vicenza_xxxxx.json"
