"""Tests for credential file passthrough registry (tools/credential_files.py)."""

import os
from pathlib import Path

import pytest

from tools.credential_files import (
    clear_credential_files,
    get_credential_file_mounts,
    register_credential_file,
    register_credential_files,
    reset_config_cache,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry between tests."""
    clear_credential_files()
    reset_config_cache()
    yield
    clear_credential_files()
    reset_config_cache()


class TestRegisterCredentialFile:
    def test_registers_existing_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "token.json").write_text('{"token": "abc"}')

        result = register_credential_file("token.json")

        assert result is True
        mounts = get_credential_file_mounts()
        assert len(mounts) == 1
        assert mounts[0]["host_path"] == str(tmp_path / "token.json")
        assert mounts[0]["container_path"] == "/root/.hermes/token.json"

    def test_skips_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        result = register_credential_file("nonexistent.json")

        assert result is False
        assert get_credential_file_mounts() == []

    def test_custom_container_base(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "cred.json").write_text("{}")

        register_credential_file("cred.json", container_base="/home/user/.hermes")

        mounts = get_credential_file_mounts()
        assert mounts[0]["container_path"] == "/home/user/.hermes/cred.json"

    def test_deduplicates_by_container_path(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "token.json").write_text("{}")

        register_credential_file("token.json")
        register_credential_file("token.json")

        mounts = get_credential_file_mounts()
        assert len(mounts) == 1


class TestRegisterCredentialFiles:
    def test_string_entries(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "a.json").write_text("{}")
        (tmp_path / "b.json").write_text("{}")

        missing = register_credential_files(["a.json", "b.json"])

        assert missing == []
        assert len(get_credential_file_mounts()) == 2

    def test_dict_entries(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "token.json").write_text("{}")

        missing = register_credential_files([
            {"path": "token.json", "description": "OAuth token"},
        ])

        assert missing == []
        assert len(get_credential_file_mounts()) == 1

    def test_returns_missing_files(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "exists.json").write_text("{}")

        missing = register_credential_files([
            "exists.json",
            "missing.json",
            {"path": "also_missing.json"},
        ])

        assert missing == ["missing.json", "also_missing.json"]
        assert len(get_credential_file_mounts()) == 1

    def test_empty_list(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert register_credential_files([]) == []


class TestConfigCredentialFiles:
    def test_loads_from_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "oauth.json").write_text("{}")
        (tmp_path / "config.yaml").write_text(
            "terminal:\n  credential_files:\n    - oauth.json\n"
        )

        mounts = get_credential_file_mounts()

        assert len(mounts) == 1
        assert mounts[0]["host_path"] == str(tmp_path / "oauth.json")

    def test_config_skips_missing_files(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(
            "terminal:\n  credential_files:\n    - nonexistent.json\n"
        )

        mounts = get_credential_file_mounts()
        assert mounts == []

    def test_combines_skill_and_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "skill_token.json").write_text("{}")
        (tmp_path / "config_token.json").write_text("{}")
        (tmp_path / "config.yaml").write_text(
            "terminal:\n  credential_files:\n    - config_token.json\n"
        )

        register_credential_file("skill_token.json")
        mounts = get_credential_file_mounts()

        assert len(mounts) == 2
        paths = {m["container_path"] for m in mounts}
        assert "/root/.hermes/skill_token.json" in paths
        assert "/root/.hermes/config_token.json" in paths


class TestGetMountsRechecksExistence:
    def test_removed_file_excluded_from_mounts(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        token = tmp_path / "token.json"
        token.write_text("{}")

        register_credential_file("token.json")
        assert len(get_credential_file_mounts()) == 1

        # Delete the file after registration
        token.unlink()
        assert get_credential_file_mounts() == []
