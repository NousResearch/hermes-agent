"""Cross-process profile-scoped live roster for delegation.status."""

import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest


def _status_from_fresh_process(repo: Path, home: Path, result_path: Path) -> list[dict]:
    script = """
import json
import os
from pathlib import Path
from tui_gateway import server
reply = server.dispatch({"id": 1, "method": "delegation.status", "params": {}})
Path(os.environ["STATUS_PATH"]).write_text(json.dumps(reply["result"]["active"], sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo,
        env={
            **os.environ,
            "HERMES_HOME": str(home),
            "PYTHONPATH": str(repo),
            "STATUS_PATH": str(result_path),
        },
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result_path.read_text())


def test_delegation_status_reads_live_child_from_another_process(tmp_path):
    """A serving process sees a same-profile gateway child's sanitized live record."""
    repo = Path(__file__).parents[2]
    home = tmp_path / "shared-profile"
    other_home = tmp_path / "other-profile"
    ready = tmp_path / "ready"
    producer = """
import os
import time
from pathlib import Path
from types import SimpleNamespace
from tools.delegate_tool_child_run import _register_child

child = SimpleNamespace(
    _subagent_id="sa-cross-process", _parent_subagent_id=None,
    _delegate_depth=1, _delegation_id="deleg-cross-process",
    _parent_session_id="parent-durable", model="test-model",
)
_register_child(child, None, "inspect API_KEY=super-secret", owner_session_id=None,
                owner_transport=None, owner_session_record=None)
Path(os.environ["READY_PATH"]).write_text("ready")
time.sleep(30)
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", producer],
        cwd=repo,
        env={
            **os.environ,
            "HERMES_HOME": str(home),
            "PYTHONPATH": str(repo),
            "READY_PATH": str(ready),
        },
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ready.exists(), "producer did not register its child"

        # The same profile remains visible across A → B → A import contexts,
        # while the independent profile is empty.
        active = _status_from_fresh_process(repo, home, tmp_path / "first-status.json")
        assert len(active) == 1
        assert active[0] == {
            "delegation_id": "deleg-cross-process",
            "depth": 0,
            "goal": "inspect API_KEY=***",
            "last_tool": None,
            "model": "test-model",
            "owner_agent_session_id": "parent-durable",
            "parent_id": None,
            "started_at": active[0]["started_at"],
            "status": "running",
            "subagent_id": "sa-cross-process",
            "tool_count": 0,
        }
        state_db = home / "state.db"
        with sqlite3.connect(state_db) as db:
            persisted = db.execute(
                "SELECT goal FROM delegation_live_subagents WHERE subagent_id = ?",
                ("sa-cross-process",),
            ).fetchone()
            columns = {
                row[1]
                for row in db.execute("PRAGMA table_info(delegation_live_subagents)")
            }
        assert persisted == ("inspect API_KEY=***",)
        assert "super-secret" not in persisted[0]
        assert not (
            {
                "prompt",
                "context",
                "result",
                "summary",
                "missed_steer",
                "callbacks",
                "routing",
                "auth",
            }
            & columns
        )

        assert (
            _status_from_fresh_process(repo, other_home, tmp_path / "other-status.json")
            == []
        )
        assert (
            _status_from_fresh_process(repo, home, tmp_path / "second-status.json")
            == active
        )

        proc.terminate()
        proc.wait(timeout=10)
        proc = None
        deadline = time.monotonic() + 13
        while time.monotonic() < deadline:
            if not _status_from_fresh_process(repo, home, tmp_path / "after-exit.json"):
                break
            time.sleep(0.1)
        assert (
            _status_from_fresh_process(repo, home, tmp_path / "after-exit.json") == []
        )
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=10)


def test_delegation_status_read_does_not_mutate_state_db(tmp_path):
    """The observer opens the profile roster read-only; it never repairs or archives state.db."""
    from tools.delegation_roster import publish

    repo = Path(__file__).parents[2]
    home = tmp_path / "profile"
    publish({
        "_roster_home": str(home),
        "subagent_id": "sa-readonly",
        "depth": 0,
        "goal": "observe only",
        "started_at": time.time(),
    })
    state_db = home / "state.db"
    before_read = state_db.stat()

    active = _status_from_fresh_process(repo, home, tmp_path / "status.json")
    assert len(active) == 1
    assert active[0] == {
        "subagent_id": "sa-readonly",
        "parent_id": None,
        "owner_agent_session_id": None,
        "delegation_id": None,
        "depth": 0,
        "goal": "observe only",
        "model": None,
        "started_at": active[0]["started_at"],
        "status": "running",
        "tool_count": 0,
        "last_tool": None,
    }
    after_read = state_db.stat()
    assert (after_read.st_mtime_ns, after_read.st_size) == (
        before_read.st_mtime_ns,
        before_read.st_size,
    )


def test_delegation_status_drops_child_after_owner_unregisters(tmp_path):
    """A foreign observer loses a completed child without touching its owner process."""
    repo = Path(__file__).parents[2]
    home = tmp_path / "shared-profile"
    ready, release, done = tmp_path / "ready", tmp_path / "release", tmp_path / "done"
    producer = """
import os
import time
from pathlib import Path
from types import SimpleNamespace
from tools.delegate_tool_child_run import _register_child
from tools.delegate_tool_registry import _unregister_subagent

child = SimpleNamespace(
    _subagent_id="sa-complete", _parent_subagent_id=None, _delegate_depth=1,
    _delegation_id="deleg-complete", _parent_session_id="parent", model="test-model",
)
_register_child(child, None, "finish cleanly", owner_session_id=None,
                owner_transport=None, owner_session_record=None)
Path(os.environ["READY_PATH"]).write_text("ready")
while not Path(os.environ["RELEASE_PATH"]).exists():
    time.sleep(.01)
_unregister_subagent("sa-complete")
Path(os.environ["DONE_PATH"]).write_text("done")
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", producer],
        cwd=repo,
        env={
            **os.environ,
            "HERMES_HOME": str(home),
            "PYTHONPATH": str(repo),
            "READY_PATH": str(ready),
            "RELEASE_PATH": str(release),
            "DONE_PATH": str(done),
        },
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert _status_from_fresh_process(repo, home, tmp_path / "running.json")

        release.write_text("release")
        while not done.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert done.exists(), "producer did not unregister its completed child"
        proc.wait(timeout=10)
        proc = None

        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            if not _status_from_fresh_process(repo, home, tmp_path / "completed.json"):
                break
            time.sleep(0.1)
        assert _status_from_fresh_process(repo, home, tmp_path / "completed.json") == []
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=10)


def test_delegation_status_sanitizes_local_records_with_the_public_allowlist(
    monkeypatch,
):
    """The official RPC applies the same public snapshot to local children as foreign leases."""
    from hermes_constants import get_hermes_home
    from tools import delegate_tool_registry as registry
    from tools.delegation_roster import _ALLOWLIST
    import tools.delegation_roster as roster
    from tui_gateway import server

    monkeypatch.setattr(roster, "list_live", lambda home: [])
    monkeypatch.setattr(
        registry,
        "_active_subagents",
        {
            "sa-local": {
                "subagent_id": "sa-local",
                "_roster_home": str(get_hermes_home()),
                "parent_id": "parent-local",
                "owner_agent_session_id": "session-local",
                "delegation_id": "deleg-local",
                "depth": 1,
                "goal": "inspect API_KEY=local-secret",
                "model": "test-model",
                "started_at": time.time(),
                "status": "running",
                "tool_count": 2,
                "last_tool": "call API_KEY=local-tool-secret",
                "private_callback": "must never cross the RPC boundary",
            }
        },
    )

    reply = server.dispatch({"id": 1, "method": "delegation.status", "params": {}})
    active = reply["result"]["active"]

    assert len(active) == 1
    assert set(active[0]) == set(_ALLOWLIST)
    rendered = json.dumps(active[0])
    assert "local-secret" not in rendered
    assert "local-tool-secret" not in rendered


def test_delegation_status_scopes_a_single_server_across_profiles_a_b_a(
    tmp_path, monkeypatch
):
    """The profile parameter scopes both foreign leases and local observer projection on every call."""
    from tools.delegation_roster import publish
    from tui_gateway import server

    home_a, home_b = tmp_path / "profile-a", tmp_path / "profile-b"
    publish({"_roster_home": str(home_a), "subagent_id": "a", "goal": "A"})
    publish({"_roster_home": str(home_b), "subagent_id": "b", "goal": "B"})
    monkeypatch.setattr(server, "_hermes_home", home_a)
    monkeypatch.setattr(
        server, "_profile_home", lambda name: {"a": home_a, "b": home_b}.get(name)
    )

    status = server._methods["delegation.status"]
    assert [
        row["subagent_id"]
        for row in status("a-1", {"profile": "a"})["result"]["active"]
    ] == ["a"]
    assert [
        row["subagent_id"] for row in status("b", {"profile": "b"})["result"]["active"]
    ] == ["b"]
    assert [
        row["subagent_id"]
        for row in status("a-2", {"profile": "a"})["result"]["active"]
    ] == ["a"]

    # Pause remains the pre-existing local process-wide spawn gate; it is not roster-derived control.
    pause = server._methods["delegation.pause"]
    try:
        assert (
            pause("pause", {"profile": "b", "paused": True})["result"]["paused"] is True
        )
        assert status("a-paused", {"profile": "a"})["result"]["paused"] is True
    finally:
        pause("resume", {"profile": "a", "paused": False})


def test_active_subagent_contract_is_closed_to_the_observer_allowlist():
    """The public RPC cannot acquire future completion/control fields by model permissiveness."""
    from pydantic import ValidationError
    from tools.delegation_roster import _ALLOWLIST
    from tui_gateway.contracts.billing_delegation_pets import ActiveSubagent

    schema = ActiveSubagent.model_json_schema()
    assert set(schema["properties"]) == set(_ALLOWLIST)
    assert schema["additionalProperties"] is False
    with pytest.raises(ValidationError):
        ActiveSubagent.model_validate({"subagent_id": "x", "missed_steer": "private"})
