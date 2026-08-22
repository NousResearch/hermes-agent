"""Sanitized task-id path components for sandbox storage (#92271).

Session-scoped container ids look like ``session:<key>`` and are used as
directory names under the sandbox root. Windows forbids ``:`` in path
segments, so the raw id made every persistent Docker/Singularity backend
crash with ``NotADirectoryError (WinError 267)``.
"""

import os

import pytest

from tools.environments.base import sanitize_task_id_for_path


def test_session_scoped_task_id_loses_colon():
    safe = sanitize_task_id_for_path("session:20260822_210946_71d78a")
    assert ":" not in safe
    assert safe == "session-20260822_210946_71d78a"


def test_plain_task_ids_pass_through_unchanged():
    assert sanitize_task_id_for_path("default") == "default"
    assert sanitize_task_id_for_path("rl-bench-42") == "rl-bench-42"


def test_all_windows_forbidden_characters_replaced():
    raw = 'a<b>c:"d/e\\f|g?h*i'
    safe = sanitize_task_id_for_path(raw)
    assert not set('<>:"/\\|?*') & set(safe)


def test_whitespace_only_id_falls_back_to_default():
    assert sanitize_task_id_for_path("   ") == "default"


def test_sanitized_dir_is_creatable(tmp_path):
    target = tmp_path / "docker" / sanitize_task_id_for_path(
        "session:20260822_210946_71d78a"
    )
    target.mkdir(parents=True)
    assert target.is_dir()


@pytest.mark.parametrize("backend_dir", ["docker", "hermes-overlays"])
def test_persistent_layout_uses_sanitized_names(backend_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_SANDBOX_DIR", str(tmp_path))
    from tools.environments.base import get_sandbox_dir

    raw_id = "session:20260822_210946_71d78a"
    target = get_sandbox_dir() / backend_dir / sanitize_task_id_for_path(raw_id)
    target.mkdir(parents=True)
    assert target.is_dir()
    assert ":" not in str(target.relative_to(tmp_path))
