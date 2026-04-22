"""Tests for the NacosCliClient subprocess wrapper."""
from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock

import pytest

from tools.nacos_cli_client import (
    NacosAuthError,
    NacosCliClient,
    NacosCliError,
    NacosCliNotInstalled,
    NacosCliOutputError,
    NacosNotFound,
    NacosSkillEntry,
    NacosTimeout,
    NacosVersionConflict,
)


# ------------------------------------------------------------------ error hierarchy

def test_exception_hierarchy():
    assert issubclass(NacosCliNotInstalled, NacosCliError)
    assert issubclass(NacosAuthError, NacosCliError)
    assert issubclass(NacosNotFound, NacosCliError)
    assert issubclass(NacosVersionConflict, NacosCliError)
    assert issubclass(NacosTimeout, NacosCliError)
    assert issubclass(NacosCliOutputError, NacosCliError)


# ------------------------------------------------------------------ diagnostics

def test_check_installed_absolute_path_present(tmp_path):
    fake_bin = tmp_path / "nacos-cli"
    fake_bin.write_text("#!/bin/sh\necho 1.0.0\n")
    fake_bin.chmod(0o755)
    client = NacosCliClient(bin_path=str(fake_bin))
    assert client.check_installed() is True


def test_check_installed_absolute_path_missing(tmp_path):
    client = NacosCliClient(bin_path=str(tmp_path / "nope"))
    assert client.check_installed() is False


def test_check_installed_relative_uses_which(monkeypatch):
    monkeypatch.setattr("tools.nacos_cli_client.shutil.which", lambda _: "/fake/nacos-cli")
    assert NacosCliClient().check_installed() is True
    monkeypatch.setattr("tools.nacos_cli_client.shutil.which", lambda _: None)
    assert NacosCliClient().check_installed() is False


def test_version_success(monkeypatch):
    def fake(*args, **kwargs):
        r = MagicMock(); r.returncode = 0; r.stdout = "3.2.1\n"; r.stderr = ""
        return r
    monkeypatch.setattr("tools.nacos_cli_client.subprocess.run", fake)
    assert NacosCliClient().version() == "3.2.1"


def test_version_raises_not_installed_on_filenotfound(monkeypatch):
    def fake(*args, **kwargs):
        raise FileNotFoundError()
    monkeypatch.setattr("tools.nacos_cli_client.subprocess.run", fake)
    with pytest.raises(NacosCliNotInstalled):
        NacosCliClient().version()


def test_version_raises_timeout(monkeypatch):
    def fake(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="nacos-cli", timeout=5)
    monkeypatch.setattr("tools.nacos_cli_client.subprocess.run", fake)
    with pytest.raises(NacosTimeout):
        NacosCliClient().version()


def test_version_nonzero_raises_cli_error(monkeypatch):
    def fake(*args, **kwargs):
        r = MagicMock(); r.returncode = 2; r.stdout = ""; r.stderr = "oops"
        return r
    monkeypatch.setattr("tools.nacos_cli_client.subprocess.run", fake)
    with pytest.raises(NacosCliError):
        NacosCliClient().version()


# ------------------------------------------------------------------ _run dispatcher

def _mock_run(monkeypatch, returncode=0, stdout="", stderr=""):
    def fake(*args, **kwargs):
        r = MagicMock(); r.returncode = returncode; r.stdout = stdout; r.stderr = stderr
        return r
    monkeypatch.setattr("tools.nacos_cli_client.subprocess.run", fake)


def test_run_parses_json_success(monkeypatch):
    _mock_run(monkeypatch, stdout='{"ok": true}')
    assert NacosCliClient()._run(["skill-list", "--json"]) == {"ok": True}


def test_run_returns_raw_when_expect_json_false(monkeypatch):
    _mock_run(monkeypatch, stdout="hello")
    assert NacosCliClient()._run(["whatever"], expect_json=False) == "hello"


def test_run_maps_401_to_auth_error(monkeypatch):
    _mock_run(monkeypatch, returncode=1, stderr="HTTP 401 Unauthorized")
    with pytest.raises(NacosAuthError):
        NacosCliClient()._run(["skill-list"])


def test_run_maps_403_to_auth_error(monkeypatch):
    _mock_run(monkeypatch, returncode=1, stderr="forbidden: missing scope")
    with pytest.raises(NacosAuthError):
        NacosCliClient()._run(["skill-list"])


def test_run_maps_404_to_not_found(monkeypatch):
    _mock_run(monkeypatch, returncode=1, stderr="HTTP 404 not found: skill 'foo'")
    with pytest.raises(NacosNotFound):
        NacosCliClient()._run(["skill-get", "--name", "foo"])


def test_run_maps_409_to_version_conflict(monkeypatch):
    _mock_run(monkeypatch, returncode=1, stderr="HTTP 409 Conflict: stale version")
    with pytest.raises(NacosVersionConflict):
        NacosCliClient()._run(["skill-upload"])


def test_run_generic_error(monkeypatch):
    _mock_run(monkeypatch, returncode=2, stderr="something else broke")
    with pytest.raises(NacosCliError):
        NacosCliClient()._run(["skill-list"])


def test_run_timeout(monkeypatch):
    def fake(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="nacos-cli", timeout=30)
    monkeypatch.setattr("tools.nacos_cli_client.subprocess.run", fake)
    with pytest.raises(NacosTimeout):
        NacosCliClient()._run(["skill-list"])


def test_run_raises_output_error_on_bad_json(monkeypatch):
    _mock_run(monkeypatch, stdout="not-json")
    with pytest.raises(NacosCliOutputError):
        NacosCliClient()._run(["skill-list", "--json"])


# ------------------------------------------------------------------ list_skills

def test_list_skills_parses_entries(monkeypatch):
    payload = {
        "skills": [
            {
                "name": "code-review", "namespace": "public",
                "group": "hermes-skills", "version": "1.0.0",
                "description": "Review code", "author": "alice",
                "updatedAt": "2026-04-20T10:00:00Z",
                "checksum": "sha256:abc",
            }
        ]
    }
    _mock_run(monkeypatch, stdout=json.dumps(payload))
    entries = NacosCliClient().list_skills(namespace="public", group="hermes-skills")
    assert len(entries) == 1
    assert entries[0].name == "code-review"
    assert entries[0].version == "1.0.0"
    assert entries[0].checksum == "sha256:abc"


def test_list_skills_empty(monkeypatch):
    _mock_run(monkeypatch, stdout='{"skills": []}')
    assert NacosCliClient().list_skills() == []


def test_list_skills_passes_query(monkeypatch):
    seen = {}
    def fake(cmd, *args, **kwargs):
        seen["cmd"] = cmd
        r = MagicMock(); r.returncode = 0; r.stdout = '{"skills": []}'; r.stderr = ""
        return r
    monkeypatch.setattr("tools.nacos_cli_client.subprocess.run", fake)
    NacosCliClient().list_skills(query="code")
    assert "--query" in seen["cmd"]
    assert "code" in seen["cmd"]


# ------------------------------------------------------------------ get_skill

def test_get_skill_returns_path_and_checksum(monkeypatch, tmp_path):
    zp = tmp_path / "code-review.zip"
    meta = {"file": str(zp), "checksum": "sha256:abc", "name": "code-review"}
    def fake(*args, **kwargs):
        zp.write_bytes(b"PK")
        r = MagicMock(); r.returncode = 0; r.stdout = json.dumps(meta); r.stderr = ""
        return r
    monkeypatch.setattr("tools.nacos_cli_client.subprocess.run", fake)
    path, checksum = NacosCliClient().get_skill("code-review", output_dir=tmp_path)
    assert path == zp
    assert checksum == "sha256:abc"


def test_get_skill_passes_version(monkeypatch, tmp_path):
    seen = {}
    def fake(cmd, *args, **kwargs):
        seen["cmd"] = cmd
        r = MagicMock(); r.returncode = 0
        r.stdout = json.dumps({"file": str(tmp_path / "x.zip"), "checksum": None})
        r.stderr = ""
        (tmp_path / "x.zip").write_bytes(b"PK")
        return r
    monkeypatch.setattr("tools.nacos_cli_client.subprocess.run", fake)
    NacosCliClient().get_skill("x", version="1.2.3", output_dir=tmp_path)
    assert "--version" in seen["cmd"]
    assert "1.2.3" in seen["cmd"]


# ------------------------------------------------------------------ upload_skill

def test_upload_skill_parses_result(monkeypatch, tmp_path):
    zp = tmp_path / "x.zip"; zp.write_bytes(b"PK")
    _mock_run(monkeypatch, stdout='{"uploaded": true, "version": "1.0.1", "checksum": "sha256:xyz"}')
    result = NacosCliClient().upload_skill(zp, namespace="team-a")
    assert result["version"] == "1.0.1"


def test_upload_skill_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        NacosCliClient().upload_skill(tmp_path / "missing.zip")


# ------------------------------------------------------------------ sync_namespace

def test_sync_namespace(monkeypatch, tmp_path):
    _mock_run(monkeypatch, stdout='{"synced": ["a", "b"], "skipped": []}')
    result = NacosCliClient().sync_namespace(output_dir=str(tmp_path))
    assert result["synced"] == ["a", "b"]


# ------------------------------------------------------------------ NacosSkillEntry

def test_skill_entry_from_json_defaults():
    entry = NacosSkillEntry.from_json({"name": "foo"})
    assert entry.name == "foo"
    assert entry.namespace == "public"
    assert entry.group == "hermes-skills"
    assert entry.version == "latest"
    assert entry.author is None
