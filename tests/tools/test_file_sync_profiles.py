"""Tests for syncing profile registry files to remote backends."""

from pathlib import Path

import pytest

from tools.credential_files import clear_credential_files
from tools.environments.file_sync import iter_sync_files


@pytest.fixture(autouse=True)
def _isolated_hermes_home(monkeypatch, tmp_path):
    """Isolate HERMES_HOME and wrapper paths for profile sync tests."""
    import tools.credential_files as credential_files

    clear_credential_files()
    credential_files._config_files = None

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))

    yield tmp_path

    clear_credential_files()
    credential_files._config_files = None


def test_iter_sync_files_includes_profile_registry_and_wrappers(tmp_path):
    default_home = tmp_path / ".hermes"
    profile_dir = default_home / "profiles" / "pythonguru"
    profile_dir.mkdir(parents=True)
    (profile_dir / "SOUL.md").write_text("be useful", encoding="utf-8")
    (profile_dir / "config.yaml").write_text("model: test\n", encoding="utf-8")
    (default_home / "active_profile").write_text("pythonguru\n", encoding="utf-8")

    wrapper_dir = tmp_path / ".local" / "bin"
    wrapper_dir.mkdir(parents=True)
    wrapper_path = wrapper_dir / "pythonguru"
    wrapper_path.write_text('#!/bin/sh\nexec hermes -p pythonguru "$@"\n', encoding="utf-8")

    files = iter_sync_files("/home/remote/.hermes")
    remote_map = {remote: host for host, remote in files}

    assert "/home/remote/.hermes/active_profile" in remote_map
    assert remote_map["/home/remote/.hermes/active_profile"] == str(
        default_home / "active_profile"
    )
    assert "/home/remote/.hermes/profiles/pythonguru/SOUL.md" in remote_map
    assert remote_map["/home/remote/.hermes/profiles/pythonguru/SOUL.md"] == str(
        profile_dir / "SOUL.md"
    )
    assert "/home/remote/.hermes/profiles/pythonguru/config.yaml" in remote_map
    assert "/home/remote/.local/bin/pythonguru" in remote_map
    assert remote_map["/home/remote/.local/bin/pythonguru"] == str(wrapper_path)


def test_iter_sync_files_keeps_profile_registry_when_current_home_is_named_profile(
    monkeypatch, tmp_path
):
    default_home = tmp_path / ".hermes"
    profile_dir = default_home / "profiles" / "pythonguru"
    skills_dir = profile_dir / "skills"
    skills_dir.mkdir(parents=True)
    (profile_dir / "SOUL.md").write_text("named profile", encoding="utf-8")
    (skills_dir / "skill.md").write_text("# skill", encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(profile_dir))

    files = iter_sync_files("/srv/hermes/.hermes")
    remote_paths = {remote for _, remote in files}

    assert "/srv/hermes/.hermes/profiles/pythonguru/SOUL.md" in remote_paths
    assert "/srv/hermes/.hermes/skills/skill.md" in remote_paths
