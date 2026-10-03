"""Capacity diagnostics are actionable hints, never automatic memory eviction."""

import json

import pytest

from tools.memory_tool import MemoryStore
from tools.registry import registry


def _dispatch(store, **args):
    return json.loads(registry.dispatch("memory", args, store=store))


@pytest.mark.parametrize("target", ["memory", "user"])
@pytest.mark.parametrize("action", ["add", "replace"])
def test_capacity_candidates_identify_live_entries_without_mutating_memory(target, action, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    entries = [
        "Long note " + "a" * 140,
        "A short preference",
        "B short preference",
        "Another long note " + "b" * 140,
        "Yes",
        "Other preference",
        "Archive candidate " + "c" * 130,
    ]
    store = MemoryStore()
    for entry in entries:
        assert _dispatch(store, action="add", target=target, content=entry)["success"]
    size = store._char_count(target)
    store = MemoryStore(memory_char_limit=size, user_char_limit=size)
    store.load_from_disk()
    path = store._path_for(target)
    before = path.read_bytes()
    prompt = store.format_for_system_prompt(target)

    args = {"action": action, "target": target, "content": "New preference " + "z" * 200}
    if action == "replace":
        args["old_text"] = entries[4]
    result = _dispatch(store, **args)

    assert result["success"] is False
    assert result["current_entries"] == entries
    candidates = result["candidates_for_eviction"]
    assert [item["index"] for item in candidates] == [4, 5, 1, 2, 6]
    for candidate in candidates:
        assert candidate["text"] == entries[candidate["index"]]
        assert candidate["preview"] == candidate["text"][:120]
        assert candidate["reason"]
    assert len(candidates[-1]["preview"]) == 120
    assert path.read_bytes() == before
    assert store._entries_for(target) == entries
    assert store.format_for_system_prompt(target) == prompt


@pytest.mark.parametrize("target", ["memory", "user"])
def test_capacity_hints_do_not_escape_the_failure_budget_or_other_errors(target, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    store = MemoryStore(memory_char_limit=40, user_char_limit=40)
    empty_overflow = _dispatch(store, action="add", target=target, content="x" * 50)
    assert empty_overflow["candidates_for_eviction"] == []
    assert not store._path_for(target).exists()
    assert _dispatch(store, action="add", target=target, content="Known preference")["success"]
    before = store._path_for(target).read_bytes()

    unmatched = _dispatch(store, action="replace", target=target, old_text="absent", content="Correction")
    assert not unmatched["success"]
    assert "candidates_for_eviction" not in unmatched
    store.reset_consolidation_failures()
    for _ in range(store._MAX_CONSOLIDATION_FAILURES_PER_TURN):
        overflow = _dispatch(store, action="add", target=target, content="z" * 50)
        assert overflow["candidates_for_eviction"][0]["text"] == "Known preference"
    terminal = _dispatch(store, action="add", target=target, content="z" * 50)
    assert terminal["done"]
    assert "candidates_for_eviction" not in terminal
    assert store._path_for(target).read_bytes() == before
