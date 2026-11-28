"""Capacity diagnostics are actionable hints, never automatic memory eviction."""

import json

import pytest

from tools.memory_tool import MemoryStore
from tools.registry import registry


def _dispatch(store, **args):
    return json.loads(registry.dispatch("memory", args, store=store))


def _full_store(target, entries):
    store = MemoryStore()
    for entry in entries:
        assert _dispatch(store, action="add", target=target, content=entry)["success"]
    size = store._char_count(target)
    store = MemoryStore(memory_char_limit=size, user_char_limit=size)
    store.load_from_disk()
    return store


@pytest.mark.parametrize("target", ["memory", "user"])
def test_overflow_names_redundant_entries_without_mutating_memory(target, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    entries = [
        "Project uses pnpm workspaces; run pnpm install at the repo root",
        "User prefers concise answers without preamble",
        "Deploys go through the staging cluster first, never straight to prod",
        "Run pnpm install at the repo root because the project uses pnpm workspaces",
        "Timezone is Europe/Lisbon",
    ]
    store = _full_store(target, entries)
    path = store._path_for(target)
    before, prompt = path.read_bytes(), store.format_for_system_prompt(target)

    result = _dispatch(store, action="add", target=target,
                       content="Always ship to the staging cluster first and never deploy straight to prod")

    assert result["success"] is False and result["current_entries"] == entries
    by_index = {c["index"]: c for c in result["candidates_for_eviction"]}
    assert set(by_index) == {0, 2, 3}  # the pnpm pair repeats each other; entry 2 repeats the new text
    assert "entry 3" in by_index[0]["reason"] and "entry 0" in by_index[3]["reason"]
    assert "being written" in by_index[2]["reason"]
    assert path.read_bytes() == before
    assert store.format_for_system_prompt(target) == prompt


def test_an_entry_with_facts_the_new_text_lacks_is_not_reported_as_repeated(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    store = _full_store("memory", ["Project uses pnpm workspaces; production deploys require manual approval "
                                   "from the release owner, ticket filed in the tracker"])

    result = _dispatch(store, action="add", target="memory", content="Project uses pnpm workspaces")

    assert result["success"] is False and result["candidates_for_eviction"] == []


@pytest.mark.parametrize("target", ["memory", "user"])
def test_capacity_hints_do_not_escape_the_failure_budget_or_other_errors(target, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    store = MemoryStore(memory_char_limit=60, user_char_limit=60)
    empty_overflow = _dispatch(store, action="add", target=target, content="x" * 70)
    assert empty_overflow["candidates_for_eviction"] == []
    assert not store._path_for(target).exists()
    assert _dispatch(store, action="add", target=target, content="Known preference for dark themes")["success"]
    before = store._path_for(target).read_bytes()

    unmatched = _dispatch(store, action="patch", target=target, old_text="absent", content="Correction")
    assert not unmatched["success"]
    assert "candidates_for_eviction" not in unmatched
    store.reset_consolidation_failures()
    repeat = "Known preference for dark themes in every editor and terminal"
    for _ in range(store._MAX_CONSOLIDATION_FAILURES_PER_TURN):
        overflow = _dispatch(store, action="add", target=target, content=repeat)
        assert overflow["candidates_for_eviction"][0]["index"] == 0
    terminal = _dispatch(store, action="add", target=target, content=repeat)
    assert terminal["done"]
    assert "candidates_for_eviction" not in terminal
    assert store._path_for(target).read_bytes() == before
