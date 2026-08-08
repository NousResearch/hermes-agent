"""Unit tests for terminal_tool.ensure_task_env — the lazy sandbox bring-up
used by vision_analyze so a session's first action can be an in-sandbox file
read without a prior terminal command (issue #62825)."""

from types import SimpleNamespace
from unittest.mock import patch

import tools.terminal_tool as tt


def _clear(*task_ids):
    for task_id in task_ids:
        tt._active_environments.pop(task_id, None)
        tt._last_activity.pop(task_id, None)


def test_local_backend_is_noop(monkeypatch):
    """Local backend reads images host-side — no sandbox is created."""
    monkeypatch.setenv("TERMINAL_ENV", "local")
    with patch.object(tt, "_create_environment") as create:
        assert tt.ensure_task_env("t-local") is None
    create.assert_not_called()


def test_non_local_creates_and_reuses(monkeypatch):
    """A non-local backend with no active env creates one, caches it, and a
    second call reuses the cache instead of spawning a duplicate."""
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    task_id = "t-ssh"
    eff = tt._resolve_container_task_id(task_id)
    _clear(eff, task_id)
    fake = SimpleNamespace(execute=lambda *a, **k: {"returncode": 0, "output": ""})
    try:
        with patch.object(tt, "_create_environment", return_value=fake) as create:
            assert tt.ensure_task_env(task_id) is fake
            create.assert_called_once()
            assert tt.get_active_env(task_id) is fake

            # Already active -> no second creation.
            assert tt.ensure_task_env(task_id) is fake
            create.assert_called_once()
    finally:
        _clear(eff, task_id)


def test_creation_failure_returns_none_and_caches_nothing(monkeypatch):
    """A failed bring-up is best-effort: return None and leave no env cached so
    the caller keeps its fail-closed path."""
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    task_id = "t-ssh-fail"
    eff = tt._resolve_container_task_id(task_id)
    _clear(eff, task_id)
    try:
        with patch.object(tt, "_create_environment", side_effect=RuntimeError("boom")):
            assert tt.ensure_task_env(task_id) is None
        assert tt.get_active_env(task_id) is None
    finally:
        _clear(eff, task_id)
