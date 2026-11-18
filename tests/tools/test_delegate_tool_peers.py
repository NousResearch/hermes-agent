"""delegation.peer_messages: batch siblings share findings mid-run (Perplexity Computer Multiagent port)."""

import json
from types import SimpleNamespace

import pytest

from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext
from agent.plugin_live_notes import attach_to_tool_message
from agent.prompt_builder import STEER_MARKER_OPEN
from tools import delegate_tool_config, delegate_tool_peers as peers
from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock


def _batch(n):
    return [(i, {"goal": f"g{i}"}, SimpleNamespace(_subagent_id=f"sa-{i}", tools=[{"function": {"name": "read_file"}}],
                                                    valid_tool_names={"read_file"})) for i in range(n)]


@pytest.fixture
def live(monkeypatch):
    added = []

    def register(child, accepting=True):
        with _active_subagents_lock:
            _active_subagents[child._subagent_id] = {"agent": child, "accepting_steer": accepting}
        added.append(child._subagent_id)

    yield register
    with _active_subagents_lock:
        for sid in added:
            _active_subagents.pop(sid, None)


def test_off_by_default_and_on_wires_identical_schema_with_closed_roster(monkeypatch):
    monkeypatch.setattr(delegate_tool_config, "_load_config", dict)
    off = _batch(3)
    peers.wire_batch_peers(off)
    assert all(len(c.tools) == 1 and peers.peer_brief_note(c) == "" for _, _, c in off)

    monkeypatch.setattr(delegate_tool_config, "_load_config", lambda: {"peer_messages": True})
    single = _batch(1)
    peers.wire_batch_peers(single)
    assert len(single[0][2].tools) == 1  # a lone child has no siblings
    on = _batch(3)
    peers.wire_batch_peers(on)
    schemas = [json.dumps(c.tools, sort_keys=True) for _, _, c in on]
    assert len(set(schemas)) == 1, "siblings must keep byte-identical tools[] (shared prompt cache)"
    assert all(peers.MESSAGE_PEER_TOOL_NAME in c.valid_tool_names for _, _, c in on)
    assert on[1][2]._delegate_peers == {"task-1": "sa-0", "task-2": "sa-1", "task-3": "sa-2"}
    assert "You are task-2 of 3" in peers.peer_brief_note(on[1][2])


def test_note_rides_running_siblings_next_tool_result_without_user_authority(monkeypatch, live):
    monkeypatch.setattr(delegate_tool_config, "_load_config", lambda: {"peer_messages": True})
    batch = _batch(3)
    peers.wire_batch_peers(batch)
    sender, running, done = (c for _, _, c in batch)
    live(sender), live(running), live(done, accepting=False)

    call = INLINE_TOOL_EXECUTORS[peers.MESSAGE_PEER_TOOL_NAME]
    out = json.loads(call(sender, {"to": "all", "message": "Q3 revenue is 4.1M, not 3.8M"}, InlineToolContext("t")))
    assert out["queued_for"] == ["task-2"] and out["already_finished"] == ["task-3"]
    assert "error" in json.loads(call(sender, {"to": "task-1", "message": "x"}, InlineToolContext("t")))  # self
    assert "error" in json.loads(call(sender, {"to": "task-9", "message": "x"}, InlineToolContext("t")))

    row = {"role": "tool", "tool_call_id": "c1", "content": "file body"}
    attach_to_tool_message(running, row)
    assert row["content"].startswith("file body") and "Q3 revenue is 4.1M" in row["content"]
    assert "Peer message from task-1" in row["content"] and STEER_MARKER_OPEN not in row["content"]
    again = {"role": "tool", "tool_call_id": "c2", "content": "next"}
    attach_to_tool_message(running, again)
    assert again["content"] == "next"  # delivered once
