"""Tests for tools/delegation_live_log.py — live subagent transcripts.

Covers:
- writer event rendering + truncation + append/flush semantics
- failure-swallowing when the target dir is unwritable
- the tool_progress_callback observe() demux (assistant/tool events in order)
- dispatch-time creation: paths pre-created with a header, manifest written
- retention pruning of stale live dirs
- delegate_task return-shape: live_transcripts in sync + background dispatch
"""

import json
import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tools import delegation_live_log as dll
from tools.delegation_live_log import (
    LiveTranscriptWriter,
    create_live_transcripts,
    live_transcript_root,
    prune_stale_live_dirs,
    update_manifest_statuses,
    wrap_progress_callback,
)


# ---------------------------------------------------------------------------
# Writer unit tests
# ---------------------------------------------------------------------------


def test_writer_precreates_file_with_header():
    w = LiveTranscriptWriter("deleg_test1", 0, "do the thing", context="some ctx")
    assert w.path is not None and w.path.exists()
    text = w.path.read_text(encoding="utf-8")
    assert "Hermes subagent live transcript" in text
    assert "delegation: deleg_test1" in text
    assert "goal: do the thing" in text
    assert "kickoff" in text
    assert "some ctx" in text
    # Lives under the hermes cache/delegation/live root, named task-<n>.log
    assert w.path.name == "task-0.log"
    assert w.path.parent.name == "deleg_test1"
    assert w.path.parent.parent == live_transcript_root()


def test_writer_event_lines_append_in_order_and_flush_immediately():
    w = LiveTranscriptWriter("deleg_order", 1, "goal")
    w.assistant_text("I'll inspect the repo first.")
    w.tool_start("terminal", "ls -la /tmp")
    w.tool_result("terminal", result="file1\nfile2", duration=1.234, is_error=False)
    w.thinking("hmm, next step")
    # No close() needed: every event is flushed on write.
    lines = w.path.read_text(encoding="utf-8").splitlines()
    body = [ln for ln in lines if "|" in ln and not ln.startswith("=")]
    joined = "\n".join(body)
    assert "assistant" in joined and "I'll inspect the repo first." in joined
    assert "-> terminal(ls -la /tmp)" in joined
    assert "terminal ok 1.2s: file1 file2" in joined
    assert "hmm, next step" in joined
    # Ordering: assistant before tool before result before think
    idx = {k: joined.index(k) for k in ("I'll inspect", "-> terminal", "terminal ok", "hmm,")}
    assert idx["I'll inspect"] < idx["-> terminal"] < idx["terminal ok"] < idx["hmm,"]


def test_writer_truncates_long_text_with_elision_note():
    w = LiveTranscriptWriter("deleg_trunc", 0, "g")
    w.assistant_text("x" * 5000)
    w.tool_result("web_search", result="y" * 5000)
    text = w.path.read_text(encoding="utf-8")
    assert "…(+" in text  # elision marker present
    # No line carries the full 5000 chars
    assert all(len(ln) < 1200 for ln in text.splitlines())


def test_writer_collapses_newlines_to_single_line_events():
    w = LiveTranscriptWriter("deleg_nl", 0, "g")
    before = len(w.path.read_text(encoding="utf-8").splitlines())
    w.assistant_text("line1\nline2\n\nline3")
    after = w.path.read_text(encoding="utf-8").splitlines()
    assert len(after) == before + 1
    assert "line1 line2 line3" in after[-1]


def test_writer_swallows_failures_when_dir_unwritable(tmp_path):
    # Point the writer at a root that is actually a FILE — mkdir will fail.
    bogus_root = tmp_path / "not-a-dir"
    bogus_root.write_text("occupied")
    w = LiveTranscriptWriter("deleg_fail", 0, "g", root=bogus_root)
    assert w.path is None
    # All writes must be silent no-ops.
    w.assistant_text("hello")
    w.tool_start("terminal", "ls")
    w.marker("done")
    w.observe("tool.completed", "terminal", result="x")
    w.finalize({"status": "completed"})


def test_writer_disables_itself_after_write_failure():
    w = LiveTranscriptWriter("deleg_disable", 0, "g")
    # Delete the parent dir out from under it and make writing impossible by
    # replacing the path with a directory.
    p = w.path
    p.unlink()
    p.mkdir()
    w.assistant_text("should not raise")
    assert w._ok is False
    w.assistant_text("still silent")  # no raise on subsequent calls


def test_stream_deltas_buffer_and_flush_as_one_line():
    w = LiveTranscriptWriter("deleg_stream", 0, "g")
    w.add_stream_delta("Hello ")
    w.add_stream_delta("world, ")
    w.add_stream_delta("streaming.")
    # Not yet flushed
    assert "Hello world" not in w.path.read_text(encoding="utf-8")
    w.flush_stream()
    text = w.path.read_text(encoding="utf-8")
    assert "Hello world, streaming." in text
    # tool_start also flushes pending stream text first
    w.add_stream_delta("more text")
    w.tool_start("read_file", "foo.py")
    text = w.path.read_text(encoding="utf-8")
    assert text.index("more text") < text.index("-> read_file")


# ---------------------------------------------------------------------------
# observe() demux — the tool_progress_callback seam
# ---------------------------------------------------------------------------


def test_observe_maps_child_callback_events_to_lines():
    w = LiveTranscriptWriter("deleg_observe", 0, "g")
    w.observe("subagent.start", preview="kick off the goal")
    w.observe("_thinking", "first line of thinking")
    w.observe("reasoning.available", "_thinking", "deep reasoning text", None)
    w.observe("tool.started", "terminal", "ls /tmp", {"command": "ls /tmp"})
    w.observe("tool.completed", "terminal", None, None,
              duration=0.5, is_error=False, result="ok output")
    w.observe("subagent.text", preview="final reply ")
    w.observe("subagent.text", preview="streamed in parts")
    w.observe("subagent.complete", preview="short", status="completed",
              duration_seconds=3.2, summary="did the thing")
    text = w.path.read_text(encoding="utf-8")
    assert "kick off the goal" in text
    assert "first line of thinking" in text
    assert "deep reasoning text" in text
    assert "-> terminal(ls /tmp)" in text
    assert "terminal ok 0.5s: ok output" in text
    assert "final reply streamed in parts" in text
    assert "status=completed" in text
    assert "did the thing" in text


def test_observe_marks_tool_errors():
    w = LiveTranscriptWriter("deleg_err", 0, "g")
    w.observe("tool.completed", "web_search", None, None,
              is_error=True, result="Error: boom")
    assert "web_search ERROR" in w.path.read_text(encoding="utf-8")


def test_finalize_records_budget_exhaustion_and_errors():
    w = LiveTranscriptWriter("deleg_final", 0, "g")
    w.finalize({"status": "failed", "exit_reason": "max_iterations",
                "error": "Subagent did not produce a response."})
    text = w.path.read_text(encoding="utf-8")
    assert "end status=failed" in text
    assert "exit_reason=max_iterations" in text
    assert "iteration budget exhausted" in text
    assert "did not produce a response" in text


def test_wrap_progress_callback_tees_and_preserves_inner():
    w = LiveTranscriptWriter("deleg_wrap", 0, "g")
    seen = []

    def inner(event_type, tool_name=None, preview=None, args=None, **kw):
        seen.append((event_type, tool_name))

    inner_flushed = []
    inner._flush = lambda: inner_flushed.append(True)

    cb = wrap_progress_callback(inner, w)
    cb("tool.started", "terminal", "echo hi", None)
    cb("_thinking", "pondering")
    assert seen == [("tool.started", "terminal"), ("_thinking", "pondering")]
    text = w.path.read_text(encoding="utf-8")
    assert "-> terminal(echo hi)" in text and "pondering" in text
    # _flush contract preserved
    cb._flush()
    assert inner_flushed == [True]


def test_wrap_progress_callback_with_no_inner_still_records():
    w = LiveTranscriptWriter("deleg_noinner", 0, "g")
    cb = wrap_progress_callback(None, w)
    cb("tool.started", "read_file", "a.py", None)
    cb._flush()  # must not raise
    assert "-> read_file(a.py)" in w.path.read_text(encoding="utf-8")


def test_wrap_progress_callback_writer_failure_does_not_block_inner():
    w = LiveTranscriptWriter("deleg_wfail", 0, "g")
    w.observe = MagicMock(side_effect=RuntimeError("disk on fire"))
    seen = []
    cb = wrap_progress_callback(lambda *a, **k: seen.append(a), w)
    cb("tool.started", "terminal", "x", None)  # must not raise
    assert len(seen) == 1


# ---------------------------------------------------------------------------
# Dispatch-time creation + manifest + retention
# ---------------------------------------------------------------------------


def test_create_live_transcripts_precreates_paths_and_manifest():
    tasks = [{"goal": "task A"}, {"goal": "task B", "context": "ctx B"}]
    deleg_id, writers, paths = create_live_transcripts(tasks, context="shared ctx")
    assert deleg_id and deleg_id.startswith("deleg_")
    assert len(writers) == 2 and all(w is not None for w in writers)
    assert len(paths) == 2
    for i, p in enumerate(paths):
        assert os.path.isabs(p)
        assert p.endswith(f"task-{i}.log")
        assert Path(p).exists()  # tail -f works immediately
    manifest = json.loads(
        (live_transcript_root() / deleg_id / "manifest.json").read_text()
    )
    assert manifest["task_count"] == 2
    assert manifest["tasks"][0]["goal"] == "task A"
    assert manifest["tasks"][0]["status"] == "running"
    assert manifest["tasks"][1]["log"] == paths[1]
    # Per-task context beats shared context in the kickoff line.
    assert "ctx B" in Path(paths[1]).read_text(encoding="utf-8")


def test_update_manifest_statuses():
    tasks = [{"goal": "a"}, {"goal": "b"}]
    deleg_id, _writers, _paths = create_live_transcripts(tasks)
    update_manifest_statuses(deleg_id, [
        {"task_index": 0, "status": "completed", "exit_reason": "completed"},
        {"task_index": 1, "status": "error"},
    ])
    manifest = json.loads(
        (live_transcript_root() / deleg_id / "manifest.json").read_text()
    )
    assert manifest["tasks"][0]["status"] == "completed"
    assert manifest["tasks"][1]["status"] == "error"
    assert "completed" in manifest


def test_update_manifest_statuses_none_id_is_noop():
    update_manifest_statuses(None, [{"task_index": 0, "status": "completed"}])


def test_prune_stale_live_dirs():
    root = live_transcript_root()
    old_dir = root / "deleg_old00001"
    new_dir = root / "deleg_new00001"
    old_dir.mkdir(parents=True)
    new_dir.mkdir(parents=True)
    (old_dir / "task-0.log").write_text("old")
    (new_dir / "task-0.log").write_text("new")
    stale = time.time() - 8 * 86400
    os.utime(old_dir, (stale, stale))
    removed = prune_stale_live_dirs(max_age_days=7)
    assert removed == 1
    assert not old_dir.exists()
    assert new_dir.exists()


def test_create_live_transcripts_survives_root_failure(monkeypatch):
    monkeypatch.setattr(
        dll, "live_transcript_root",
        lambda: (_ for _ in ()).throw(RuntimeError("no home")),
    )
    deleg_id, writers, paths = create_live_transcripts([{"goal": "g"}])
    assert deleg_id is None
    assert writers == [None]
    assert paths == []


# ---------------------------------------------------------------------------
# delegate_task return-shape integration
# ---------------------------------------------------------------------------


def _make_parent():
    parent = MagicMock()
    parent._delegate_depth = 0
    parent.session_id = "sess-live"
    parent._interrupt_requested = False
    parent._active_children = []
    parent._active_children_lock = None
    return parent


_CREDS = {
    "model": "m", "provider": None, "base_url": None, "api_key": None,
    "api_mode": None, "command": None, "args": None,
}


def _fake_run(task_index, goal, child=None, parent_agent=None, **kw):
    return {
        "task_index": task_index, "status": "completed",
        "summary": f"done: {goal}", "api_calls": 1,
        "duration_seconds": 0.1, "model": "m", "exit_reason": "completed",
    }


def test_delegate_task_sync_result_includes_live_transcripts(monkeypatch):
    import tools.delegate_tool as dt

    parent = _make_parent()
    fake_child = MagicMock()
    fake_child._delegate_role = "leaf"
    fake_child.tool_progress_callback = None
    monkeypatch.setattr(dt, "_build_child_agent", lambda **kw: fake_child)
    monkeypatch.setattr(dt, "_run_single_child", _fake_run)
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *a, **k: _CREDS)

    out = json.loads(dt.delegate_task(goal="sync goal", parent_agent=parent))
    assert "live_transcripts" in out
    assert len(out["live_transcripts"]) == 1
    p = Path(out["live_transcripts"][0])
    assert p.exists()
    assert "sync goal" in p.read_text(encoding="utf-8")
    # Per-task entries carry their own path + a terminal marker was written.
    assert out["results"][0]["live_transcript"] == str(p)
    assert "end status=completed" in p.read_text(encoding="utf-8")


def test_delegate_task_background_dispatch_includes_live_transcripts(monkeypatch):
    import tools.delegate_tool as dt
    from tools import async_delegation as ad
    from tools.process_registry import process_registry

    parent = _make_parent()
    fake_child = MagicMock()
    fake_child._delegate_role = "leaf"
    fake_child._subagent_id = "s1"
    fake_child.tool_progress_callback = None

    gate = threading.Event()

    def slow_child(task_index, goal, child=None, parent_agent=None, **kw):
        gate.wait(timeout=60)
        return _fake_run(task_index, goal)

    monkeypatch.setattr(dt, "_build_child_agent", lambda **kw: fake_child)
    monkeypatch.setattr(dt, "_run_single_child", slow_child)
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *a, **k: _CREDS)

    out = json.loads(dt.delegate_task(
        goal="bg goal", background=True, parent_agent=parent,
    ))
    try:
        assert out["status"] == "dispatched"
        assert "live_transcripts" in out
        assert len(out["live_transcripts"]) == 1
        live = Path(out["live_transcripts"][0])
        # Pre-created at dispatch time — tail -f attaches immediately,
        # while the child is still running behind the gate.
        assert live.exists()
        assert "bg goal" in live.read_text(encoding="utf-8")
        assert "live_transcripts_hint" in out
        # The dir name matches the returned delegation handle.
        assert live.parent.name == out["delegation_id"]
    finally:
        gate.set()
        # Drain the completion so it can't leak into other tests.
        deadline = time.time() + 30
        evt = None
        while time.time() < deadline:
            try:
                evt = process_registry.completion_queue.get(timeout=0.5)
                break
            except Exception:
                continue
        ad._reset_for_tests()

    assert evt is not None
    # The completion event carries the same paths for the consolidated block.
    assert evt.get("live_transcripts") == out["live_transcripts"]
    assert evt["results"][0]["live_transcript"] == out["live_transcripts"][0]


def test_batch_dispatch_creates_one_log_per_task(monkeypatch):
    import tools.delegate_tool as dt

    parent = _make_parent()

    def make_child(**kw):
        c = MagicMock()
        c._delegate_role = "leaf"
        c.tool_progress_callback = None
        return c

    monkeypatch.setattr(dt, "_build_child_agent", make_child)
    monkeypatch.setattr(dt, "_run_single_child", _fake_run)
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *a, **k: _CREDS)

    out = json.loads(dt.delegate_task(
        tasks=[{"goal": "alpha"}, {"goal": "beta"}], parent_agent=parent,
    ))
    assert len(out["live_transcripts"]) == 2
    names = [Path(p).name for p in out["live_transcripts"]]
    assert names == ["task-0.log", "task-1.log"]
    # Both under the same delegation dir
    parents = {Path(p).parent for p in out["live_transcripts"]}
    assert len(parents) == 1
    for p, goal in zip(out["live_transcripts"], ("alpha", "beta")):
        assert goal in Path(p).read_text(encoding="utf-8")


def test_child_progress_events_land_in_live_log(monkeypatch):
    """Events fired through the child's (wrapped) tool_progress_callback land
    in the transcript file in order — the seam the real agent loop drives."""
    import tools.delegate_tool as dt

    parent = _make_parent()
    built = []

    def make_child(**kw):
        c = MagicMock()
        c._delegate_role = "leaf"
        c.tool_progress_callback = None
        built.append(c)
        return c

    def run_child(task_index, goal, child=None, parent_agent=None, **kw):
        # Simulate what agent/tool_executor.py + conversation_loop.py emit.
        cb = child.tool_progress_callback
        cb("_thinking", "planning the work")
        cb("tool.started", "terminal", "echo hi", {"command": "echo hi"})
        cb("tool.completed", "terminal", None, None,
           duration=0.2, is_error=False, result="hi")
        return _fake_run(task_index, goal)

    monkeypatch.setattr(dt, "_build_child_agent", make_child)
    monkeypatch.setattr(dt, "_run_single_child", run_child)
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *a, **k: _CREDS)

    out = json.loads(dt.delegate_task(goal="observable goal", parent_agent=parent))
    text = Path(out["live_transcripts"][0]).read_text(encoding="utf-8")
    assert "planning the work" in text
    assert "-> terminal(echo hi)" in text
    assert "terminal ok 0.2s: hi" in text
    assert text.index("planning") < text.index("-> terminal") < text.index("terminal ok")


def test_delegate_task_proceeds_when_transcripts_unavailable(monkeypatch):
    """Live-log failure must never break delegation itself."""
    import tools.delegate_tool as dt
    from tools import delegation_live_log as _dll

    parent = _make_parent()
    fake_child = MagicMock()
    fake_child._delegate_role = "leaf"
    fake_child.tool_progress_callback = None
    monkeypatch.setattr(dt, "_build_child_agent", lambda **kw: fake_child)
    monkeypatch.setattr(dt, "_run_single_child", _fake_run)
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *a, **k: _CREDS)
    monkeypatch.setattr(
        _dll, "live_transcript_root",
        lambda: (_ for _ in ()).throw(RuntimeError("nope")),
    )

    out = json.loads(dt.delegate_task(goal="resilient", parent_agent=parent))
    assert out["results"][0]["status"] == "completed"
    assert "live_transcripts" not in out


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
