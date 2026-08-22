"""Regression tests for tui_gateway._save_cfg honoring Hermes-home override (Issue #91587).

Verifies that _save_cfg respects the context-local get_hermes_home_override()
matching _load_cfg_raw(), so that profile config mutations write to the profile's
config.yaml instead of clobbering the base ~/.hermes/config.yaml.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tui_gateway import server as srv


@pytest.fixture(autouse=True)
def reset_cfg_cache():
    """Reset module-level config cache before and after each test."""
    with srv._cfg_lock:
        srv._cfg_cache = None
        srv._cfg_mtime = None
        srv._cfg_path = None
    yield
    with srv._cfg_lock:
        srv._cfg_cache = None
        srv._cfg_mtime = None
        srv._cfg_path = None


def test_save_cfg_default_home(tmp_path):
    """When no override is active, _save_cfg writes to _hermes_home / config.yaml."""
    base_home = tmp_path / "base_hermes"
    base_home.mkdir()
    base_cfg = base_home / "config.yaml"
    base_cfg.write_text("model:\n  default: gpt-4\n", encoding="utf-8")

    with patch.object(srv, "_hermes_home", base_home):
        srv._save_cfg({"model": {"default": "claude-3-opus"}})

        assert base_cfg.exists()
        loaded = yaml.safe_load(base_cfg.read_text(encoding="utf-8"))
        assert loaded["model"]["default"] == "claude-3-opus"
        assert srv._cfg_path == base_cfg


def test_save_cfg_honors_home_override(tmp_path):
    """When a Hermes-home override is active, _save_cfg writes to the overridden path."""
    base_home = tmp_path / "base_hermes"
    base_home.mkdir()
    base_cfg = base_home / "config.yaml"
    base_cfg.write_text("model:\n  default: base-model\n", encoding="utf-8")

    profile_home = tmp_path / "profile_work"
    profile_home.mkdir()
    profile_cfg = profile_home / "config.yaml"
    profile_cfg.write_text("model:\n  default: profile-model\n", encoding="utf-8")

    with patch.object(srv, "_hermes_home", base_home):
        token = set_hermes_home_override(str(profile_home))
        try:
            # Load raw from profile
            raw_cfg = srv._load_cfg_raw()
            assert raw_cfg["model"]["default"] == "profile-model"

            # Mutate and save
            raw_cfg["model"]["default"] = "profile-updated"
            srv._save_cfg(raw_cfg)

            # Profile config was updated
            profile_loaded = yaml.safe_load(profile_cfg.read_text(encoding="utf-8"))
            assert profile_loaded["model"]["default"] == "profile-updated"
            assert srv._cfg_path == profile_cfg

            # Base config was NOT touched
            base_loaded = yaml.safe_load(base_cfg.read_text(encoding="utf-8"))
            assert base_loaded["model"]["default"] == "base-model"
        finally:
            reset_hermes_home_override(token)
