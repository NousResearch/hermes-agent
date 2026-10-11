"""Delegation launch routes run before a child can become a writer."""
from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from hermes_cli import admission_contract
from hermes_cli import admission_runtime
from hermes_cli.admission import Admission
from hermes_cli.admission_contract import AdmissionErrorCode, AdmissionResult
from tools import delegate_tool
from tools.delegate_tool_child_run import _Heartbeat


def _rejected(request_id: str) -> AdmissionResult:
    return AdmissionResult(
        request_id=request_id,
        state="rejected",
        reason=AdmissionErrorCode.UNSUPPORTED_PATH.value,
    )


def test_public_delegate_route_rejects_before_child_construction(monkeypatch):
    parent = SimpleNamespace(_delegate_depth=0, session_id="parent-session")
    built: list[object] = []
    credentials = {
        "model": "test-model", "provider": None, "base_url": None,
        "api_key": None, "api_mode": None, "command": None, "args": None,
    }

    monkeypatch.setattr(delegate_tool, "_get_max_spawn_depth", lambda: 1)
    monkeypatch.setattr(delegate_tool, "_load_config", dict)
    monkeypatch.setattr(delegate_tool, "_get_max_concurrent_children", lambda: 1)
    monkeypatch.setattr(delegate_tool, "_resolve_delegation_credentials", lambda *_args: credentials)
    monkeypatch.setattr(
        delegate_tool,
        "_build_children",
        lambda *_args, **_kwargs: (built.append(object()), ([], None))[1],
    )
    monkeypatch.setattr(
        admission_contract,
        "route_launch_path",
        lambda _path, _caller, *, request_id: _rejected(request_id),
    )

    result = delegate_tool.delegate_task(goal="do not construct", parent_agent=parent)

    assert "admission path rejected: unsupported_path" in result
    assert built == []


def test_direct_child_helper_rejects_before_child_execution(monkeypatch):
    started: list[object] = []
    monkeypatch.setattr(
        admission_contract,
        "route_launch_path",
        lambda _path, _caller, *, request_id: _rejected(request_id),
    )
    monkeypatch.setattr(
        delegate_tool,
        "_fabricated_entry",
        lambda *_args: started.append(object()) or {"status": "error"},
    )
    monkeypatch.setattr(
        delegate_tool,
        "_start_heartbeat",
        lambda *_args: (_ for _ in ()).throw(AssertionError("heartbeat must follow admission")),
    )
    monkeypatch.setattr(
        delegate_tool,
        "_register_child",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("registration must follow admission")),
    )

    result = delegate_tool._run_single_child(0, "do not run", child=SimpleNamespace())

    assert result == {"status": "error"}
    assert len(started) == 1


def test_queued_delegate_admission_is_cancelled_before_child_execution(monkeypatch):
    cancelled: list[str] = []

    class Run:
        worktree_info = {"path": "/isolated/worktree"}

        def __init__(self, *_args):
            pass

        def seed_workspace(self):
            return None

        def elapsed(self):
            return 0.0

        def attach_worktree(self, entry):
            return entry

    monkeypatch.setattr(delegate_tool, "_ChildRun", Run)
    monkeypatch.setattr(
        admission_runtime,
        "admit_writer",
        lambda **kwargs: Admission(
            kwargs["request_id"], "delegate:test", "queued", {"gateway": 1}, True, None, None, "capacity",
        ),
    )
    monkeypatch.setattr(admission_runtime, "cancel_admission_request", lambda request_id: cancelled.append(request_id) or True)
    monkeypatch.setattr(delegate_tool, "_fabricated_entry", lambda *_args: {"status": "error"})
    child = SimpleNamespace(_delegate_admission_required=True, _subagent_id="delegate-queued")

    result = delegate_tool._run_single_child(0, "do not run", child=child)

    assert result == {"status": "error"}
    assert cancelled == ["delegate-queued"]


def test_live_delegate_heartbeat_renews_its_admission_lease(monkeypatch):
    renewed: list[str] = []
    child = SimpleNamespace(
        get_activity_summary=lambda: {"api_call_count": 1, "current_tool": None, "max_iterations": 2},
    )
    parent = SimpleNamespace(_touch_activity=lambda _desc: None)
    heartbeat = _Heartbeat(child, parent, 0)
    heartbeat.admission_lease_id = "lease-live-delegate"
    monkeypatch.setattr(admission_runtime, "renew_admission", lambda lease_id: renewed.append(lease_id) or True)

    heartbeat.tick()

    assert renewed == ["lease-live-delegate"]


def test_unisolated_delegate_rejection_is_schema_safe_before_child_start(tmp_path, monkeypatch):
    """A fail-closed pre-spawn rejection must not need controller tables to clean up."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_ADMISSION_LEDGER", raising=False)

    class Run:
        worktree_info = None

        def __init__(self, *_args):
            pass

        def seed_workspace(self):
            return None

        def elapsed(self):
            return 0.0

        def attach_worktree(self, entry):
            return entry

        def await_child(self):
            pytest.fail("rejected admission must not start a child")

    monkeypatch.setattr(delegate_tool, "_ChildRun", Run)
    monkeypatch.setattr(
        delegate_tool,
        "_fabricated_entry",
        lambda *_args: {"status": "error", "error": "admission rejected: unsupported_path"},
    )

    result = delegate_tool._run_single_child(
        0,
        "do not run",
        child=SimpleNamespace(_delegate_admission_required=True, _subagent_id="delegate-unisolated"),
    )

    assert result == {"status": "error", "error": "admission rejected: unsupported_path"}
    with sqlite3.connect(admission_runtime.ledger_path()) as connection:
        rejection = connection.execute(
            "SELECT caller, reason FROM admission_rejections WHERE request_id = ?",
            ("delegate-unisolated",),
        ).fetchone()
        active_requests = connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'admission_requests'"
        ).fetchone()[0]
    assert rejection == ("delegate", AdmissionErrorCode.UNSUPPORTED_PATH.value)
    assert active_requests == 0


def test_writer_admission_boundary_preserves_direct_read_only_seams():
    """A lease needs a stamp or a resolved checkout, not write tools alone."""
    from run_agent import AIAgent

    direct_writer = object.__new__(AIAgent)
    direct_writer.valid_tool_names = {"write_file"}
    direct_reader = object.__new__(AIAgent)
    direct_reader.valid_tool_names = {"read_file"}

    assert delegate_tool._requires_writer_admission(
        SimpleNamespace(_delegate_admission_required=True)
    )
    # Write tools without a resolved checkout are not a lease. Slack children
    # have terminal and must still start.
    assert not delegate_tool._requires_writer_admission(direct_writer)
    direct_writer._delegate_repo_root = "/repo"
    assert delegate_tool._requires_writer_admission(direct_writer)
    assert not delegate_tool._requires_writer_admission(direct_reader)
    assert not delegate_tool._requires_writer_admission(
        SimpleNamespace(_subagent_id="subagent-test", valid_tool_names={"terminal"})
    )
