"""Tests for tools.audit_log — structured audit logging."""

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.audit_log import (
    _DEFAULT_MAX_BYTES,
    _rotate_file,
    format_audit_summary,
    get_audit_config,
    get_audit_log_path,
    is_audit_enabled,
    read_audit_entries,
    write_audit_entry,
)


class TestAuditConfig:
    """Test configuration resolution."""

    def test_audit_enabled_by_default(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert is_audit_enabled()

    def test_get_audit_log_path_resolves_hermes_home(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        path = get_audit_log_path()
        assert str(tmp_path) in str(path)
        assert path.name == "audit.log"

    def test_get_audit_config_returns_defaults(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        max_bytes, max_days = get_audit_config()
        assert max_bytes == _DEFAULT_MAX_BYTES
        assert max_days == 7


class TestWriteAuditEntry:
    """Test writing audit entries."""

    def test_write_creates_file_and_entry(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        log_path = get_audit_log_path()

        write_audit_entry(
            command="ls -la",
            session_key="test-session",
            task_id="default",
            exit_code=0,
        )

        assert log_path.exists()
        entries = read_audit_entries()
        assert len(entries) == 1
        entry = entries[0]
        assert entry["command"] == "ls -la"
        assert entry["session_key"] == "test-session"
        assert entry["exit_code"] == 0
        assert "timestamp" in entry

    def test_write_blocked_entry(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        write_audit_entry(
            command="rm -rf /",
            blocked=True,
            block_reason="hardline block",
        )

        entries = read_audit_entries()
        assert len(entries) == 1
        assert entries[0]["blocked"] is True
        assert entries[0]["block_reason"] == "hardline block"

    def test_write_respects_disabled_flag(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        config_path = tmp_path / "config.yaml"
        config_path.write_text("logging:\n  audit_enabled: false\n", encoding="utf-8")

        with patch("tools.audit_log._resolve_config_value", side_effect=lambda k, d: False if k == "audit_enabled" else d):
            write_audit_entry(command="ls")

        log_path = get_audit_log_path()
        assert not log_path.exists()

    def test_write_appends_multiple_entries(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        write_audit_entry(command="ls")
        write_audit_entry(command="cat /etc/hostname")
        write_audit_entry(command="git status")

        entries = read_audit_entries()
        assert len(entries) == 3
        assert entries[0]["command"] == "git status"  # newest first (reverse=True)


class TestReadAuditEntries:
    """Test reading audit entries."""

    def test_read_empty_log(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        entries = read_audit_entries()
        assert entries == []

    def test_read_with_session_filter(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        write_audit_entry(command="ls", session_key="session-a")
        write_audit_entry(command="cat", session_key="session-b")
        write_audit_entry(command="git", session_key="session-a")

        entries_a = read_audit_entries(session_filter="session-a")
        assert len(entries_a) == 2
        assert all(e["session_key"] == "session-a" for e in entries_a)

    def test_read_respects_max_lines(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        for i in range(50):
            write_audit_entry(command=f"cmd-{i}")

        entries = read_audit_entries(max_lines=10)
        assert len(entries) == 10
        # Newest first (reverse=True)
        assert entries[0]["command"] == "cmd-49"


class TestLogRotation:
    """Test log rotation behavior."""

    def test_rotation_creates_backup(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        log_path = get_audit_log_path()
        log_path.write_text("old entry\n", encoding="utf-8")

        _rotate_file(log_path, "test")

        assert log_path.with_suffix(".log.1").exists()
        assert not log_path.exists()  # original moved

    def test_rotation_shifts_existing(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        log_path = get_audit_log_path()

        # Create .1 and .2
        log_path.with_suffix(".log.1").write_text("rotation 1\n", encoding="utf-8")
        log_path.with_suffix(".log.2").write_text("rotation 2\n", encoding="utf-8")
        log_path.write_text("current\n", encoding="utf-8")

        _rotate_file(log_path, "test")

        assert log_path.with_suffix(".log.1").exists()
        assert log_path.with_suffix(".log.2").exists()
        assert log_path.with_suffix(".log.3").exists()
        assert not log_path.exists()

    def test_rotation_drops_oldest(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        log_path = get_audit_log_path()

        # Create .1, .2, .3
        for i in range(1, 4):
            log_path.with_suffix(f".log.{i}").write_text(f"rotation {i}\n", encoding="utf-8")
        log_path.write_text("current\n", encoding="utf-8")

        _rotate_file(log_path, "test")

        # .3 should have been deleted (max_rotations=3)
        assert not log_path.with_suffix(".log.4").exists()
        assert log_path.with_suffix(".log.3").exists()


class TestFormatAuditSummary:
    """Test summary formatting."""

    def test_empty_entries(self) -> None:
        assert "No audit entries" in format_audit_summary([])

    def test_summary_with_mixed_entries(self) -> None:
        entries = [
            {"blocked": False, "user_approved": False},
            {"blocked": True, "user_approved": False},
            {"blocked": False, "user_approved": True},
        ]
        summary = format_audit_summary(entries)
        assert "3 total" in summary
        assert "1 blocked" in summary
        assert "1 user-approved" in summary
