"""``rollback.list`` RPC: fall back to the cross-project view (#10505).

Bare ``/rollback`` in the classic CLI already fell back to a cross-project
"all directories" view when the current directory has zero checkpoints
(hermes_cli.cli_commands_mixin, reapply of PR #10633 by @nightq). The TUI's
own ``rollback.list`` RPC — used by the Ink TUI's ``/rollback`` (and, per
#90029, planned for Desktop) — never got the same fix and just returned an
empty list, offering no information.

``rollback.list``'s handler bodies live in tui_gateway/methods_tools.py but
are rebound onto tui_gateway.server's globals at import time (see
method_ctx.py) — the installed handler in ``server._methods`` is what's
under test, invoked exactly as the JSON-RPC dispatcher would.
"""

from types import SimpleNamespace

import pytest

import tools.checkpoint_manager as cpm
import tui_gateway.server as server


def _call(method, params=None):
    handler = server._methods[method]
    return handler(1, params or {})


@pytest.fixture()
def checkpoint_store(tmp_path, monkeypatch):
    monkeypatch.setattr(cpm, "CHECKPOINT_BASE", tmp_path / "checkpoints")
    return cpm.CheckpointManager(enabled=True, max_snapshots=50)


def _fake_session(monkeypatch, *, cwd: str, mgr: cpm.CheckpointManager):
    session = {"cwd": cwd, "agent": SimpleNamespace(_checkpoint_mgr=mgr)}
    monkeypatch.setattr(server, "_sess", lambda params, rid: (session, None))


def test_rollback_list_falls_back_to_all_directories(tmp_path, monkeypatch, checkpoint_store):
    mgr = checkpoint_store
    other_project = tmp_path / "other-project"
    other_project.mkdir()
    (other_project / "main.py").write_text("print('hi')\n", encoding="utf-8")
    assert mgr.ensure_checkpoint(str(other_project), "baseline") is True

    empty_project = tmp_path / "empty-project"
    empty_project.mkdir()
    _fake_session(monkeypatch, cwd=str(empty_project), mgr=mgr)

    resp = _call("rollback.list", {"session_id": "s1"})

    assert "error" not in resp
    result = resp["result"]
    assert result["enabled"] is True
    assert result["all_directories"] is True
    assert len(result["checkpoints"]) == 1
    assert result["checkpoints"][0]["hash"]


def test_rollback_list_no_checkpoints_anywhere(tmp_path, monkeypatch, checkpoint_store):
    mgr = checkpoint_store
    project = tmp_path / "project"
    project.mkdir()
    _fake_session(monkeypatch, cwd=str(project), mgr=mgr)

    resp = _call("rollback.list", {"session_id": "s1"})

    result = resp["result"]
    assert result["enabled"] is True
    assert result["all_directories"] is False
    assert result["checkpoints"] == []


def test_rollback_list_uses_own_directory_when_present(tmp_path, monkeypatch, checkpoint_store):
    """The fallback must not fire when the session's own cwd already has
    checkpoints — same-directory checkpoints always take priority."""
    mgr = checkpoint_store
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text("print('own')\n", encoding="utf-8")
    assert mgr.ensure_checkpoint(str(project), "own baseline") is True

    other_project = tmp_path / "other-project"
    other_project.mkdir()
    (other_project / "main.py").write_text("print('other')\n", encoding="utf-8")
    assert mgr.ensure_checkpoint(str(other_project), "other baseline") is True

    _fake_session(monkeypatch, cwd=str(project), mgr=mgr)

    resp = _call("rollback.list", {"session_id": "s1"})

    result = resp["result"]
    assert result["all_directories"] is False
    assert len(result["checkpoints"]) == 1
