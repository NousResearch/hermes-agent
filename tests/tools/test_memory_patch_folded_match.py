"""``patch`` edits an entry selected through the typography-folded match (#107270).

An old_text re-typed with straight quotes, ``-`` for a dash, unwrapped lines or a literal
``\\n`` selects the entry, but the splice looked for the exact text, found nothing and
answered "Patch is a no-op" without writing. ``remove`` on the same old_text works.
"""

import json

import pytest

from tools.memory_tool import MemoryStore, _pin_matched_entries, apply_memory_pending
from tools.memory_tool_store import _folded_spans, _normalize_for_match
from tools.registry import registry

FLEET = "User runs a fleet (“Omarchy” — Trinity) at home"
TESTS = "Tests: run ‘make test’ (needs\n  docker up) first"


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
    s = MemoryStore(memory_char_limit=2000, user_char_limit=1000)
    s.load_from_disk()
    s.add("memory", FLEET)
    s.add("memory", TESTS)
    return s


def _tool(store, **args):
    return json.loads(registry.get_entry("memory").handler(args, store=store))


@pytest.mark.parametrize(("old_text", "new_text", "expected"), [
    ('fleet ("Omarchy" - Trinity)', "fleet (Zeus hub)", "User runs a fleet (Zeus hub) at home"),
    ("run 'make test' (needs docker up)", "run make test-fast", "Tests: run make test-fast first"),
    ("(needs\\n docker up)", "(needs docker compose up)",
     "Tests: run ‘make test’ (needs docker compose up) first"),
])
def test_single_patch_edits_the_folded_span(store, old_text, new_text, expected):
    result = _tool(store, action="patch", target="memory", old_text=old_text, new_text=new_text)

    assert result["success"] is True, result
    assert expected in store.memory_entries


def test_batch_patch_edits_the_folded_span(store):
    result = _tool(store, target="memory", operations=[
        {"action": "patch", "old_text": '("Omarchy" - Trinity)', "content": "(Zeus hub)"},
        {"action": "patch", "old_text": "(needs docker up)", "content": "(needs docker compose up)"}])

    assert result["success"] is True, result
    assert store.memory_entries == ["User runs a fleet (Zeus hub) at home",
                                    "Tests: run ‘make test’ (needs docker compose up) first"]


def test_staged_patch_applies_on_approval(store):
    payload = {"action": "patch", "target": "memory", "old_text": '"Omarchy" - Trinity', "content": "Zeus hub"}
    assert _pin_matched_entries(store, payload) is None
    assert payload["matched_entry"] == FLEET

    result = apply_memory_pending(payload, store)

    assert result["success"] is True, result
    assert store.memory_entries[0] == "User runs a fleet (Zeus hub) at home"


def test_edge_whitespace_and_repeated_span(store):
    store.add("memory", "Prefers “tea”  daily, “tea” at night too")
    repeated = _tool(store, action="patch", target="memory", old_text='"tea"', new_text='"coffee"')
    assert repeated["success"] is False and "appears 2 times" in repeated["error"]

    result = _tool(store, action="patch", target="memory", old_text='"tea" daily', new_text="coffee daily")
    assert result["success"] is True, result
    assert "Prefers coffee daily, “tea” at night too" in store.memory_entries

    assert _folded_spans("a “b”  c", '"b" ') == [(2, 7)]  # a trailing space keeps the run
    for text in (FLEET, TESTS, "  x\\n–y\t z  ", "\\n"):
        spans = _folded_spans(text, _normalize_for_match(text))
        assert all(_normalize_for_match(text[s:e]) == _normalize_for_match(text) for s, e in spans)
