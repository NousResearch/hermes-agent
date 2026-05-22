import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

from tools import bitwarden_tool as bwtool


def _completed(stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(["bw"], returncode, stdout, stderr)


def test_bitwarden_status_redacts_email_and_reports_locked(monkeypatch):
    def fake_run(cmd, **kwargs):
        if cmd == ["bw", "--version"]:
            return _completed("2026.3.0\n")
        if cmd[:3] == ["bw", "status", "--raw"]:
            return _completed(json.dumps({
                "status": "locked",
                "userEmail": "davie.lam01@gmail.com",
                "lastSync": "2026-05-21T21:45:56.155Z",
            }))
        raise AssertionError(cmd)

    monkeypatch.delenv("BW_SESSION", raising=False)
    monkeypatch.setattr(bwtool.subprocess, "run", fake_run)
    monkeypatch.setattr(bwtool, "_iter_session_files", lambda: [])

    data = json.loads(bwtool.bitwarden_status())

    assert data["success"] is True
    assert data["available"] is True
    assert data["status"] == "locked"
    assert data["unlocked"] is False
    assert data["user_email"] == "da***1@gmail.com"


def test_bitwarden_search_requires_unlocked(monkeypatch):
    monkeypatch.setattr(bwtool, "_find_unlocked_session", lambda: (None, "none", {"status": "locked"}))

    data = json.loads(bwtool.bitwarden_search("example"))

    assert data["success"] is False
    assert data["needs_unlock"] is True
    assert "locked" in data["error"].lower()


def test_bitwarden_search_returns_metadata_without_password(monkeypatch):
    monkeypatch.setattr(bwtool, "_require_unlocked", lambda: ("session", "/tmp/hermes_bw_session_test", {"status": "unlocked"}))
    monkeypatch.setattr(bwtool, "_list_items", lambda query, session, limit: [{
        "name": "Example Login",
        "type": 1,
        "login": {
            "username": "user@example.com",
            "password": "super-secret",
            "uris": [{"uri": "https://example.com"}],
        },
        "notes": "private note",
    }])

    raw = bwtool.bitwarden_search("Example")
    data = json.loads(raw)

    assert data["success"] is True
    assert data["items"][0]["name"] == "Example Login"
    assert data["items"][0]["has_password"] is True
    assert "super-secret" not in raw
    assert "private note" not in raw


def test_bitwarden_get_secret_ref_writes_mode_600_env_file_without_returning_secret(monkeypatch, tmp_path):
    secret = "super-secret-token"
    monkeypatch.setattr(bwtool, "_SECRET_DIR", tmp_path)
    monkeypatch.setattr(bwtool, "_require_unlocked", lambda: ("session", "/tmp/hermes_bw_session_test", {"status": "unlocked"}))
    monkeypatch.setattr(bwtool, "_find_exact_item", lambda item_name, session: {
        "name": item_name,
        "type": 1,
        "login": {"username": "api-token", "password": secret, "uris": []},
    })

    raw = bwtool.bitwarden_get_secret_ref("Example Token", env_var="EXAMPLE_TOKEN")
    data = json.loads(raw)

    assert data["success"] is True
    assert data["secret_returned"] is False
    assert secret not in raw
    env_file = Path(data["env_file"])
    assert env_file.exists()
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
    assert "EXAMPLE_TOKEN=" in env_file.read_text()
    assert secret in env_file.read_text()


def test_bitwarden_get_secret_ref_rejects_bad_env_var(monkeypatch):
    monkeypatch.setattr(bwtool, "_require_unlocked", lambda: ("session", "env", {"status": "unlocked"}))

    data = json.loads(bwtool.bitwarden_get_secret_ref("Example", env_var="BAD-NAME"))

    assert data["success"] is False
    assert "env_var" in data["error"]
