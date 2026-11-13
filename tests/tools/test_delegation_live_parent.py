"""The live manifest names the spawning session from the first write, so a transcript viewer can place a running
delegation under its parent without matching on time, goal text or the eventual tool result.

Real ``delegate_task`` and real manifest I/O; only child construction and the child run are faked (no model call).
"""

import json
import threading
from unittest.mock import MagicMock

import pytest

from tools import delegate_tool as dt
from tools.delegation_live_log import create_live_transcripts, live_transcript_root, update_manifest_statuses


def _parent(session_id, depth=0):
    parent = MagicMock()
    parent.session_id = session_id
    parent.base_url, parent.api_key, parent.provider = "https://openrouter.ai/api/v1", "parent-key", "openrouter"
    parent.api_mode, parent.model, parent.platform = "chat_completions", "anthropic/claude-sonnet-4", "cli"
    parent.request_overrides = {}
    parent.reasoning_config = None
    parent.providers_allowed = parent.providers_ignored = parent.providers_order = parent.provider_sort = None
    parent._session_db, parent._delegate_depth, parent._active_children = None, depth, []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = parent.tool_progress_callback = parent.thinking_callback = None
    return parent


def _manifest_for_goal(goal):
    for path in live_transcript_root().glob("*/manifest.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if [t["goal"] for t in manifest["tasks"]] == [goal]:
            return manifest
    return None


def _read_manifest(delegation_id):
    return json.loads((live_transcript_root() / delegation_id / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture
def fake_children(monkeypatch):
    """Stub child build/run; record the manifest each child sees when it is built and when it runs."""
    seen = {"built": {}, "ran": {}}
    run_gate = {"barrier": None, "status": "completed"}

    def build(**kw):
        seen["built"][kw["goal"]] = _manifest_for_goal(kw["goal"])
        child = MagicMock()
        child._delegate_role, child._subagent_id, child.model = "leaf", f"sub-{kw['goal']}", "m"
        return child

    def run(task_index, goal, child=None, parent_agent=None, **_kw):
        seen["ran"][goal] = (child._delegation_id, _read_manifest(child._delegation_id))
        if run_gate["barrier"] is not None:
            run_gate["barrier"].wait()  # both delegations are live at once
        status = run_gate["status"]
        return {"task_index": task_index, "status": status, "summary": "done" if status == "completed" else None,
                "exit_reason": {"completed": "completed", "failed": "error", "interrupted": "interrupted"}[status],
                "error": "child blew up" if status == "failed" else None, "api_calls": 0, "duration_seconds": 0.1}

    monkeypatch.setattr(dt, "_load_config", dict)
    monkeypatch.setattr(dt, "_build_child_agent", build)
    monkeypatch.setattr(dt, "_run_single_child", run)
    return seen, run_gate


@pytest.mark.parametrize("terminal", ["completed", "failed", "interrupted"])
def test_manifest_names_the_parent_before_the_child_runs_and_keeps_it_when_settled(fake_children, terminal):
    seen, gate = fake_children
    gate["status"] = terminal
    result = json.loads(dt.delegate_task(goal="parented task", parent_agent=_parent("sess-parent-A")))

    assert seen["built"]["parented task"]["parent_session_id"] == "sess-parent-A"
    deleg_id, running = seen["ran"]["parented task"]
    assert running["parent_session_id"] == "sess-parent-A" and "completed" not in running
    assert result["results"][0]["status"] == terminal
    settled = _read_manifest(deleg_id)
    assert settled["parent_session_id"] == "sess-parent-A"
    assert settled["tasks"][0]["status"] == terminal and "completed" in settled


@pytest.mark.parametrize("session_id", [None, "", "   ", 123, MagicMock()])
def test_a_missing_or_unusable_parent_id_stays_unbound(fake_children, monkeypatch, session_id):
    """No guess from the environment: another room's id must never be borrowed."""
    seen, _gate = fake_children
    monkeypatch.setenv("HERMES_SESSION_ID", "sess-some-other-room")
    dt.delegate_task(goal="orphan task", parent_agent=_parent(session_id))

    deleg_id, running = seen["ran"]["orphan task"]
    assert "parent_session_id" not in seen["built"]["orphan task"]
    assert "parent_session_id" not in running
    assert "parent_session_id" not in _read_manifest(deleg_id)


def test_concurrent_parents_do_not_cross(fake_children):
    seen, gate = fake_children
    gate["barrier"] = threading.Barrier(2, timeout=10)
    threads = [threading.Thread(target=dt.delegate_task,
                                kwargs={"goal": f"task of {sid}", "parent_agent": _parent(sid)})
               for sid in ("sess-room-1", "sess-room-2")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)

    ids = {}
    for sid in ("sess-room-1", "sess-room-2"):
        deleg_id, running = seen["ran"][f"task of {sid}"]
        assert running["parent_session_id"] == sid
        assert _read_manifest(deleg_id)["parent_session_id"] == sid
        ids[sid] = deleg_id
    assert ids["sess-room-1"] != ids["sess-room-2"]


def test_nested_delegation_records_its_direct_parent_not_the_root(fake_children, monkeypatch):
    seen, _gate = fake_children
    monkeypatch.setattr(dt, "_get_max_spawn_depth", lambda: 3)
    orchestrator = _parent("sess-orchestrator-child", depth=1)
    orchestrator._parent_session_id = "sess-telegram-root"
    dt.delegate_task(goal="grandchild task", parent_agent=orchestrator)

    _deleg_id, running = seen["ran"]["grandchild task"]
    assert running["parent_session_id"] == "sess-orchestrator-child"


def test_only_the_parent_identifier_is_added(tmp_path):
    secret = "sk-proj-" + "A" * 40
    goal = f"deploy using {secret}"
    plain_id, _w, _p = create_live_transcripts([{"goal": goal}], context=f"ctx {secret}", home=tmp_path)
    parented_id, _w, _p = create_live_transcripts([{"goal": goal}], context=f"ctx {secret}", home=tmp_path,
                                                  parent_session_id="sess-parent-A")
    root = live_transcript_root(tmp_path)
    plain = json.loads((root / plain_id / "manifest.json").read_text(encoding="utf-8"))
    parented = json.loads((root / parented_id / "manifest.json").read_text(encoding="utf-8"))

    assert set(parented) - set(plain) == {"parent_session_id"} and set(plain) <= set(parented)
    assert parented["parent_session_id"] == "sess-parent-A"
    assert secret not in json.dumps(parented)


def test_a_legacy_manifest_without_a_parent_settles_unchanged(tmp_path):
    deleg_id, _w, _p = create_live_transcripts([{"goal": "legacy caller"}], home=tmp_path)
    update_manifest_statuses(deleg_id, [{"task_index": 0, "status": "completed"}], home=tmp_path)
    manifest = json.loads((live_transcript_root(tmp_path) / deleg_id / "manifest.json").read_text(encoding="utf-8"))
    assert "parent_session_id" not in manifest and manifest["tasks"][0]["status"] == "completed"
