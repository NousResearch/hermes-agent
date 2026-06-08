from __future__ import annotations

import os
from pathlib import Path

import pytest

from wisdom.config import load_wisdom_config
from wisdom.db import WisdomDB
from wisdom.models import WisdomConfig


@pytest.fixture
def wisdom_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key in (
        "HERMES_WISDOM_ENABLED",
        "HERMES_WISDOM_DB_PATH",
        "HERMES_WISDOM_CAPTURE_MODE",
        "HERMES_WISDOM_MAX_RESULTS",
        "HERMES_WISDOM_INTERPRET_TIMEOUT",
        "HERMES_WISDOM_INTERPRETATION_MODE",
        "HERMES_WISDOM_APPLICATION_MODE",
        "HERMES_WISDOM_APPLY_TIMEOUT",
    ):
        monkeypatch.delenv(key, raising=False)
    return home


@pytest.fixture
def wisdom_config(wisdom_home: Path) -> WisdomConfig:
    return load_wisdom_config(
        {
            "wisdom": {
                "enabled": True,
                "db_path": str(wisdom_home / "wisdom" / "wisdom.db"),
                "capture_mode": "explicit",
                "max_results": 5,
                "interpret_timeout_seconds": 5,
                "interpretation": {"mode": "deterministic"},
                "application": {"mode": "deterministic", "timeout_seconds": 30},
            }
        }
    )


@pytest.fixture
def wisdom_db(wisdom_config: WisdomConfig) -> WisdomDB:
    db = WisdomDB(wisdom_config.db_path)
    db.init()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def isolated_env_db(wisdom_home: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = wisdom_home / "wisdom" / "wisdom.db"
    monkeypatch.setenv("HERMES_WISDOM_DB_PATH", str(db_path))
    return db_path
