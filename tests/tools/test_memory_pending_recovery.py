"""Recover competing memory proposals without bypassing approval (#134625)."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hermes_cli.write_approval_commands import handle_pending_subcommand
from tools import write_approval as wa
from tools.memory_tool import MemoryStore, load_on_disk_store, memory_tool


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(wa, "evaluate_gate", lambda *args, **kwargs: wa.GateDecision(allow=True))
    return tmp_path


@pytest.mark.parametrize("shape", ["single", "batch"])
@pytest.mark.parametrize("state", ["changed", "deleted", "ambiguous", "legacy", "disabled"])
def test_refresh_rechecks_current_entry_and_requires_a_new_approval(home, shape, state):
    original, current, proposed = "Service endpoint: original", "Service endpoint: session A", "Service endpoint: session B"
    store = MemoryStore()
    store.load_from_disk()
    assert store.add("memory", original)["success"]
    assert store.add("memory", "An independent fact")["success"]
    op = {"action": "replace", "old_text": "Service endpoint:", "content": proposed, "matched_entry": original}
    payload = {"action": "batch", "target": "memory", "operations": [op]} if shape == "batch" else {**op, "target": "memory"}
    if state == "legacy":
        op.pop("matched_entry")
        if shape == "single":
            payload.pop("matched_entry")
    record = wa.stage_write(wa.MEMORY, payload, summary="Session B proposal", origin="foreground")
    pinned = deepcopy(record)
    if state == "deleted":
        assert store.remove("memory", "Service endpoint:")["success"]
    else:
        assert store.replace("memory", "Service endpoint:", current)["success"]
    if state == "ambiguous":
        assert store.add("memory", "Service endpoint: a different service")["success"]
    before = load_on_disk_store().memory_entries
    out = handle_pending_subcommand(wa.MEMORY, ["approve", record["id"]], memory_store=load_on_disk_store())
    assert "Approved 0" in out
    reviewer = load_on_disk_store()
    if state == "disabled":
        reviewer.memory_enabled = False
    refreshed = handle_pending_subcommand(wa.MEMORY, ["refresh", record["id"]], memory_store=reviewer)
    assert refreshed is not None, "The shared approval command needs a refresh path"
    assert load_on_disk_store().memory_entries == before
    pending = wa.list_pending(wa.MEMORY)
    if state != "changed":
        assert pending == [pinned]
        return
    assert len(pending) == 1
    fresh = pending[0]
    assert fresh["id"] != record["id"]
    assert all(text in refreshed for text in (original, current, proposed, fresh["id"]))
    fresh_op = fresh["payload"]["operations"][0] if shape == "batch" else fresh["payload"]
    assert fresh_op["matched_entry"] == current
    assert fresh_op["content"] == proposed
    assert "Approved 1" in handle_pending_subcommand(wa.MEMORY, ["approve", fresh["id"]], memory_store=load_on_disk_store())
    assert load_on_disk_store().memory_entries == [proposed, "An independent fact"]
    assert wa.list_pending(wa.MEMORY) == []


@pytest.mark.parametrize("shape", ["single", "batch"])
@pytest.mark.parametrize("target", ["memory", "user"])
def test_consolidation_stop_blocks_further_attempts_until_next_turn(home, shape, target):
    store = MemoryStore()
    store.load_from_disk()
    assert store.add(target, "Existing fact")["success"]
    bad = {"action": "replace", "old_text": "missing anchor", "content": "new fact"}
    kwargs = {"operations": [bad]} if shape == "batch" else bad
    for _ in range(15):
        result = json.loads(memory_tool(target=target, store=store, **kwargs))
        assert result["success"] is False
    assert result["done"] is True
    blocked = json.loads(memory_tool("add", target, "must not write after stop", store=store))
    assert blocked.get("done") is True
    assert blocked["success"] is False
    assert store._consolidation_failures == store._MAX_CONSOLIDATION_FAILURES_PER_TURN
    assert load_on_disk_store()._entries_for(target) == ["Existing fact"]
    store.reset_consolidation_failures()
    assert json.loads(memory_tool("add", target, "next turn", store=store))["success"]
    assert load_on_disk_store()._entries_for(target) == ["Existing fact", "next turn"]
