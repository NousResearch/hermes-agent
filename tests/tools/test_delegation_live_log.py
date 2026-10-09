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


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))


# ---------------------------------------------------------------------------
# Credential redaction
# ---------------------------------------------------------------------------
#
# These transcripts land under ``cache/delegation``, which delegate_tool mounts
# READ-ONLY into remote terminal backends — so a line written here is readable
# from inside the sandbox. The rendered events are exactly the secret-bearing
# surfaces (tool args, tool results, streamed assistant text), and every other
# sink for that data already routes through the canonical redactor.

_BEARER = "sk-ant-api03-" + "R" * 24
_ENV_KEY = "sk-proj-" + "L" * 24
_AWS = "wJalrXUtnFEMIK7MDENG" + "bPxRfiCY"


def test_tool_args_are_redacted_before_hitting_disk():
    w = LiveTranscriptWriter("deleg_redact_args", 0, "g")
    w.observe(
        "tool.started",
        "terminal",
        f'curl -H "Authorization: Bearer {_BEARER}" https://api.internal',
        None,
    )
    body = w.path.read_text(encoding="utf-8")
    assert _BEARER not in body
    assert "terminal" in body, "redaction must not gut the operational detail"


def test_tool_results_are_redacted_before_hitting_disk():
    w = LiveTranscriptWriter("deleg_redact_result", 0, "g")
    w.observe(
        "tool.completed",
        "terminal",
        None,
        None,
        result=f"OPENAI_API_KEY={_ENV_KEY}\nAWS_SECRET_ACCESS_KEY={_AWS}",
        duration=0.4,
    )
    body = w.path.read_text(encoding="utf-8")
    assert _ENV_KEY not in body
    assert _AWS not in body
    assert "OPENAI_API_KEY" in body, "key NAMES stay — only the values are masked"


def test_streamed_assistant_text_is_redacted():
    w = LiveTranscriptWriter("deleg_redact_stream", 0, "g")
    w.observe("subagent.text", None, f"the key is {_ENV_KEY}")
    w.flush_stream()
    assert _ENV_KEY not in w.path.read_text(encoding="utf-8")


def test_goal_header_is_redacted():
    """The header bypasses event(); a pasted key in the goal must not survive."""
    w = LiveTranscriptWriter("deleg_redact_goal", 0, f"deploy using {_BEARER}")
    body = w.path.read_text(encoding="utf-8")
    assert _BEARER not in body
    assert "deploy using" in body


def test_manifest_goal_is_redacted():
    """manifest.json shares the mounted dir with the .log files.

    Redacting the log header while ``_write_manifest`` serialises the same goal
    verbatim would leave the credential exposed one file over — both sinks in
    ``cache/delegation/live/<id>/`` are readable from inside a sandbox.
    """
    delegation_id, _writers, _paths = create_live_transcripts(
        [{"goal": f"deploy using {_BEARER}"}]
    )

    manifest = json.loads(
        (live_transcript_root() / delegation_id / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    goal = manifest["tasks"][0]["goal"]

    assert _BEARER not in goal
    assert "deploy using" in goal, "redaction must not blank the goal entirely"


def test_no_file_in_the_dispatch_directory_carries_the_raw_key():
    """Whole-directory sweep: every artefact dispatch writes is covered."""
    delegation_id, _writers, _paths = create_live_transcripts(
        [{"goal": f"deploy using {_BEARER}"}, {"goal": "second task"}]
    )

    directory = live_transcript_root() / delegation_id
    written = sorted(p.name for p in directory.iterdir())

    assert "manifest.json" in written
    assert any(name.endswith(".log") for name in written)
    for path in directory.iterdir():
        assert _BEARER not in path.read_text(encoding="utf-8"), (
            f"{path.name} leaked the credential"
        )


def test_thinking_text_is_redacted():
    w = LiveTranscriptWriter("deleg_redact_think", 0, "g")
    w.observe("_thinking", f"I should use {_ENV_KEY} here")
    assert _ENV_KEY not in w.path.read_text(encoding="utf-8")


def test_redaction_covers_every_helper_via_the_event_chokepoint():
    """Any helper that reaches disk goes through event(), so all are covered."""
    w = LiveTranscriptWriter("deleg_redact_all", 0, "g")
    w.assistant_text(f"a {_ENV_KEY}")
    w.thinking(f"b {_ENV_KEY}")
    w.tool_start("terminal", f"c {_ENV_KEY}")
    w.tool_result("terminal", result=f"d {_ENV_KEY}")
    w.marker(f"e {_ENV_KEY}")
    w.finalize({"status": "error", "error": f"f {_ENV_KEY}"})
    body = w.path.read_text(encoding="utf-8")
    assert _ENV_KEY not in body, "a write path escaped the redactor"


def test_benign_transcript_content_is_untouched():
    """Redaction must not mangle ordinary transcript text."""
    w = LiveTranscriptWriter("deleg_redact_benign", 0, "refactor the parser")
    w.observe("tool.started", "read_file", "src/parser.py", None)
    w.observe("tool.completed", "read_file", None, None, result="def parse(x): ...", duration=1.5)
    body = w.path.read_text(encoding="utf-8")
    assert "src/parser.py" in body
    assert "def parse(x)" in body
    assert "refactor the parser" in body
