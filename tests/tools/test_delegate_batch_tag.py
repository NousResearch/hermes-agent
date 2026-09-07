"""Batch tag on delegation progress lines (#p1-campaign feedback, Sep 2026).

When a parent fans out N subagents and a child fans out its own M, both
batches print ``[n/N]`` completion lines to the same console. Without a
batch tag ``✓ [3/3]`` and ``✓ [3/9]`` are indistinguishable. Every progress
surface carries a human-readable ``set N`` ordinal (not a raw id slice).
"""
import types

import pytest

import tools.delegate_tool as dt
from tools.delegate_tool import _batch_prefix, _build_child_progress_callback, format_batch_tag


@pytest.fixture(autouse=True)
def _fresh_ordinals(monkeypatch):
    monkeypatch.setattr(dt, "_BATCH_ORDINALS", {})


def test_format_batch_tag_assigns_stable_ordinals_per_batch():
    assert format_batch_tag("deleg_6a664903") == "set 1"
    assert format_batch_tag("deleg_b2ac1234") == "set 2"
    assert format_batch_tag("deleg_6a664903") == "set 1"  # same batch, same label
    assert format_batch_tag(None) == ""
    assert format_batch_tag("") == ""


@pytest.mark.parametrize(
    "deleg, idx, count, expected",
    [
        ("deleg_6a664903", 2, 9, "[set 1 · 3/9] "),
        (None, 2, 9, "[3/9] "),
        ("deleg_6a664903", 0, 1, "[set 1] "),
        (None, 0, 1, ""),
    ],
)
def test_batch_prefix_shapes(deleg, idx, count, expected):
    assert _batch_prefix(deleg, idx, count) == expected


class _Spinner:
    def __init__(self):
        self.lines = []

    def print_above(self, line):
        self.lines.append(line)

    def update_text(self, text):
        self.lines.append(f"<spin>{text}")


def test_child_tree_lines_and_relayed_events_carry_batch_tag():
    relayed = []
    parent = types.SimpleNamespace(
        _delegate_spinner=_Spinner(),
        tool_progress_callback=lambda et, name=None, preview=None, args=None, **kw: relayed.append((et, kw)),
    )
    ref = {}
    cb = _build_child_progress_callback(2, "triage cluster", parent, 9, subagent_id="sa-2", session_ref=ref)
    # Stamped by delegate_task AFTER the callback is built — must be picked up lazily.
    ref["delegation_id"] = "deleg_6a664903"
    ref["session_id"] = "child-sess"

    cb("subagent.start")
    cb("tool.started", "terminal", "ls")

    tree = parent._delegate_spinner.lines
    assert tree[0].startswith(" [set 1 · 3/9] ├─ 🔀 triage cluster")
    assert tree[1].startswith(" [set 1 · 3/9] ├─ ")
    assert all(kw.get("delegation_id") == "deleg_6a664903" for _, kw in relayed)
    assert all(kw.get("child_session_id") == "child-sess" for _, kw in relayed)


def test_child_tree_prefix_without_batch_id_is_unchanged():
    parent = types.SimpleNamespace(_delegate_spinner=_Spinner(), tool_progress_callback=None)
    cb = _build_child_progress_callback(0, "solo goal", parent, 3, session_ref={})
    cb("subagent.start")
    assert parent._delegate_spinner.lines[0].startswith(" [1/3] ├─ 🔀 solo goal")


def test_batch_completion_lines_are_attributable_across_two_batches(monkeypatch, tmp_path):
    """Two interleaved batches: every ✓ line names its own ``set N``."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir()
    lines = []
    parent = types.SimpleNamespace(
        session_id="root", model="m", tool_progress_callback=None, _delegate_spinner=None,
        _safe_print=lambda line: lines.append(line),
    )
    monkeypatch.setattr(
        dt, "_run_single_child",
        lambda task_index, goal, child=None, parent_agent=None, **kw: {
            "task_index": task_index, "status": "completed", "summary": "ok",
            "error": None, "api_calls": 1, "duration_seconds": 1,
        },
    )
    monkeypatch.setattr(dt, "_build_child_preserving_parent_tools",
                        lambda **kw: types.SimpleNamespace(tool_progress_callback=None))
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *a, **k: {
        "model": "m", "provider": "openrouter", "base_url": "https://x/v1",
        "api_key": "k", "api_mode": "chat_completions"})

    import re

    for n in (3, 9):
        res = dt.delegate_task(
            tasks=[{"goal": f"batch of {n}: worker task number {i}"} for i in range(n)],
            parent_agent=parent,
        )
        assert "error" not in str(res)[:20], res
    headers = [re.match(r"\s*🔀 \[(set \d+)\] delegating (\d+) tasks", l) for l in lines]
    headers = [m for m in headers if m]
    assert [(m.group(1), int(m.group(2))) for m in headers] == [("set 1", 3), ("set 2", 9)]

    done = [l for l in lines if "✓ [" in l]
    assert len(done) == 12
    assert sum(1 for l in done if "✓ [set 1 · " in l and "/3]" in l) == 3
    assert sum(1 for l in done if "✓ [set 2 · " in l and "/9]" in l) == 9
