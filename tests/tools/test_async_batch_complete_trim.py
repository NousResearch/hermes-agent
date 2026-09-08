"""Async batch-complete messages must arrive trimmed (summary-first).

Forensics: [ASYNC DELEGATION BATCH COMPLETE] messages injected 20-27k chars each into
parent sessions. The runner-side summary budget (min(24k static, dynamic headroom)) does
not bind on the async path when the parent's context state is unknown, so the injected
message must enforce its own per-summary cap: head-limited summary + pointer to the full
summary spilled to disk (the live transcript's assistant lines are capped at 600 chars and
are NOT a full-fidelity copy). See claude.com/blog/maximizing-the-value-of-your-claude-code-sessions:
subagents should return compact answers — heavy extraction belongs in the child.
"""
import os

from tools.process_registry_notifications import (
    _BATCH_SUMMARY_CAP,
    _trim_batch_summary,
    format_process_notification,
)


def _evt(summary: str, *, status: str = "completed") -> dict:
    return {
        "type": "async_delegation",
        "delegation_id": "deleg_trim_test",
        "is_batch": True,
        "status": "completed",
        "results": [{"task_index": 0, "status": status, "summary": summary,
                     "live_transcript": "/tmp/live/task-0.log"}],
        "goals": ["do the thing"],
        "session_key": "agent:main:cli:dm:local",
    }


def test_short_summary_passes_through_untouched():
    summary = "All good.\nDone."
    assert _trim_batch_summary(summary, {}) == summary


def test_over_cap_summary_is_head_limited_with_spill_pointer(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hh"))
    summary = "\n".join(f"line {i} " + "x" * 40 for i in range(400)) + "\nFINAL OUTCOME: ok"
    assert len(summary) > _BATCH_SUMMARY_CAP
    out = _trim_batch_summary(summary, {})
    assert len(out) < len(summary)
    assert out.startswith("line 0")
    assert "TRUNCATED" in out
    assert out.rstrip().endswith("FINAL OUTCOME: ok")  # trailing outcome line survives
    # A spill file was written and the full text is intact on disk
    assert "more chars" in out
    # The pointer path must exist and contain the full summary
    import re
    m = re.search(r"full summary: (\S+?);", out)
    assert m, "footer must point at the spilled full summary"
    with open(m.group(1).replace("~", os.path.expanduser("~")), encoding="utf-8") as fh:
        assert fh.read() == summary


def test_existing_runner_spill_path_is_reused_no_second_spill(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hh"))
    summary = "head " * 2000
    from tools import process_registry_notifications as prn
    calls = []
    monkeypatch.setattr(prn, "_spill_full_summary", lambda s: calls.append(s) or "/tmp/full.txt")
    entry = {"summary_full_path": "/tmp/runner-spilled.txt"}
    out = _trim_batch_summary(summary, entry)
    assert not calls  # runner-side spill reused; no duplicate file
    assert "/tmp/runner-spilled.txt" in out


def test_injected_batch_message_stays_small_and_keeps_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hh"))
    summary = "\n".join(f"line {i} " + "y" * 50 for i in range(500))
    text = format_process_notification(_evt(summary))
    assert text is not None
    assert text.startswith("[ASYNC DELEGATION BATCH COMPLETE — deleg_trim_test]")
    # Whole message (preamble + 1 task) far below the 20-27k forensics payload
    assert len(text) < _BATCH_SUMMARY_CAP + 2000
    assert "TRUNCATED" in text and "/tmp/live/task-0.log" in text


def test_partial_output_summary_is_also_trimmed(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hh"))
    summary = "partial " * 1000
    evt = _evt(summary, status="error")
    evt["results"][0]["error"] = "boom"
    text = format_process_notification(evt)
    assert text is not None
    assert "Partial output:" in text
    assert len(text) < _BATCH_SUMMARY_CAP + 2000
    assert "TRUNCATED" in text
