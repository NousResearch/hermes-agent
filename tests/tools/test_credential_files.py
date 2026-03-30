"""Tests for credential file passthrough and skills directory mounting."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.credential_files import (
    clear_credential_files,
    get_credential_file_mounts,
    get_skills_directory_mount,
    iter_skills_files,
    register_credential_file,
    register_credential_files,
    reset_config_cache,
)


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset module state between tests."""
    clear_credential_files()
    reset_config_cache()
    yield
    clear_credential_files()
    reset_config_cache()


class TestRegisterCredentialFiles:
    def test_dict_with_path_key(self, tmp_path):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "token.json").write_text("{}")

        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            missing = register_credential_files([{"path": "token.json"}])

        assert missing == []
        mounts = get_credential_file_mounts()
        assert len(mounts) == 1
        assert mounts[0]["host_path"] == str(hermes_home / "token.json")
        assert mounts[0]["container_path"] == "/root/.hermes/token.json"

    def test_dict_with_name_key_fallback(self, tmp_path):
        """Skills use 'name' instead of 'path' — both should work."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "google_token.json").write_text("{}")

        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            missing = register_credential_files([
                {"name": "google_token.json", "description": "OAuth token"},
            ])

        assert missing == []
        mounts = get_credential_file_mounts()
        assert len(mounts) == 1
        assert "google_token.json" in mounts[0]["container_path"]

    def test_string_entry(self, tmp_path):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "secret.key").write_text("key")

        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            missing = register_credential_files(["secret.key"])

        assert missing == []
        mounts = get_credential_file_mounts()
        assert len(mounts) == 1

    def test_missing_file_reported(self, tmp_path):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()

        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            missing = register_credential_files([
                {"name": "does_not_exist.json"},
            ])

        assert "does_not_exist.json" in missing
        assert get_credential_file_mounts() == []

    def test_path_takes_precedence_over_name(self, tmp_path):
        """When both path and name are present, path wins."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "real.json").write_text("{}")

        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            missing = register_credential_files([
                {"path": "real.json", "name": "wrong.json"},
            ])

        assert missing == []
        mounts = get_credential_file_mounts()
        assert "real.json" in mounts[0]["container_path"]


class TestSkillsDirectoryMount:
    def test_returns_mount_when_skills_dir_exists(self, tmp_path):
        hermes_home = tmp_path / ".hermes"
        skills_dir = hermes_home / "skills"
        skills_dir.mkdir(parents=True)
        (skills_dir / "test-skill").mkdir()
        (skills_dir / "test-skill" / "SKILL.md").write_text("# test")

        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            mount = get_skills_directory_mount()

        assert mount is not None
        assert mount["host_path"] == str(skills_dir)
        assert mount["container_path"] == "/root/.hermes/skills"

    def test_returns_none_when_no_skills_dir(self, tmp_path):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()

        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            mount = get_skills_directory_mount()

        assert mount is None

    def test_custom_container_base(self, tmp_path):
        hermes_home = tmp_path / ".hermes"
        (hermes_home / "skills").mkdir(parents=True)

        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            mount = get_skills_directory_mount(container_base="/home/user/.hermes")

        assert mount["container_path"] == "/home/user/.hermes/skills"

    def test_symlinks_are_sanitized(self, tmp_path):
        """Symlinks in skills dir should be excluded from the mount."""
        hermes_home = tmp_path / ".hermes"
        skills_dir = hermes_home / "skills"
        skills_dir.mkdir(parents=True)
        (skills_dir / "legit.md").write_text("# real skill")
        # Create a symlink pointing outside the skills tree
        secret = tmp_path / "secret.txt"
        secret.write_text("TOP SECRET")
        (skills_dir / "evil_link").symlink_to(secret)

        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            mount = get_skills_directory_mount()

        assert mount is not None
        # The mount path should be a sanitized copy, not the original
        safe_path = Path(mount["host_path"])
        assert safe_path != skills_dir
        # Legitimate file should be present
        assert (safe_path / "legit.md").exists()
        assert (safe_path / "legit.md").read_text() == "# real skill"
        # Symlink should NOT be present
        assert not (safe_path / "evil_link").exists()

    def test_no_symlinks_returns_original_dir(self, tmp_path):
        """When no symlinks exist, the original dir is returned (no copy)."""
        hermes_home = tmp_path / ".hermes"
        skills_dir = hermes_home / "skills"
        skills_dir.mkdir(parents=True)
        (skills_dir / "skill.md").write_text("ok")

        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            mount = get_skills_directory_mount()

        assert mount["host_path"] == str(skills_dir)


class TestIterSkillsFiles:
    def test_returns_files_skipping_symlinks(self, tmp_path):
        hermes_home = tmp_path / ".hermes"
        skills_dir = hermes_home / "skills"
        (skills_dir / "cat" / "myskill").mkdir(parents=True)
        (skills_dir / "cat" / "myskill" / "SKILL.md").write_text("# skill")
        (skills_dir / "cat" / "myskill" / "scripts").mkdir()
        (skills_dir / "cat" / "myskill" / "scripts" / "run.sh").write_text("#!/bin/bash")
        # Add a symlink that should be filtered
        secret = tmp_path / "secret"
        secret.write_text("nope")
        (skills_dir / "cat" / "myskill" / "evil").symlink_to(secret)

        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            files = iter_skills_files()

        paths = {f["container_path"] for f in files}
        assert "/root/.hermes/skills/cat/myskill/SKILL.md" in paths
        assert "/root/.hermes/skills/cat/myskill/scripts/run.sh" in paths
        # Symlink should be excluded
        assert not any("evil" in f["container_path"] for f in files)

    def test_empty_when_no_skills_dir(self, tmp_path):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()

        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            assert iter_skills_files() == []
