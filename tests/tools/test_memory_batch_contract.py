"""Batch shape is validated before staging and during native-store replay."""

import copy
import json

import pytest

from hermes_cli.config import load_config, save_config
from tools import memory_tool as mt
from tools import write_approval as wa
from tools.skill_provenance import reset_current_write_origin, set_current_write_origin


@pytest.fixture
def store():
    result = mt.MemoryStore(memory_char_limit=60, user_char_limit=60)
    result.load_from_disk()
    assert result.add("memory", "Project uses Python")["success"]
    assert result.add("user", "User prefers concise replies")["success"]
    result.load_from_disk()
    return result


@pytest.mark.parametrize("route", ["direct", "foreground", "background", "store", "replay"])
@pytest.mark.parametrize("operations", [
    [], {}, "add", 0, [None], [[]], ["add"], [{}],
    [{"action": []}], [{"action": "delete"}],
    [{"action": "add", "content": "New fact", "target": "memory"}],
    [{"action": "add", "content": "New fact", "target": "user"}],
    [{"action": "add", "content": "New fact", "typo": True}],
    [{"action": "add", "content": 1}],
    [{"action": "add", "new_text": ["New fact"]}],
    [{"action": "remove", "old_text": False}],
    [{"action": "add", "content": " "}],
    [{"action": "replace", "content": "New fact"}],
])
def test_malformed_batch_neither_stages_nor_changes_either_store(store, route, operations):
    before = {target: store._path_for(target).read_bytes() for target in ("memory", "user")}
    original = copy.deepcopy(operations)
    config = load_config()
    config.setdefault("memory", {})["write_approval"] = route in {"foreground", "background"}
    save_config(config)
    token = set_current_write_origin("background_review" if route == "background" else "foreground")
    try:
        if route == "store":
            result = store.apply_batch("memory", operations)
        elif route == "replay":
            result = mt.apply_memory_pending(
                {"action": "batch", "target": "memory", "operations": operations}, store)
        else:
            # An explicit malformed operations value must not fall back to the single action.
            result = json.loads(mt.memory_tool(
                action="add", target="memory", content="Fallback must not be written",
                operations=operations, store=store))
    finally:
        reset_current_write_origin(token)
    assert result["success"] is False, result
    assert result.get("error")
    assert not result.get("staged")
    assert wa.list_pending(wa.MEMORY) == []
    assert operations == original
    assert {target: store._path_for(target).read_bytes() for target in before} == before


@pytest.mark.parametrize("target", ["memory", "user"])
@pytest.mark.parametrize("route", ["direct", "approved"])
def test_valid_single_store_batch_preserves_gate_final_budget_and_prompt(store, target, route):
    config = load_config()
    config.setdefault("memory", {})["write_approval"] = route == "approved"
    save_config(config)
    other = "user" if target == "memory" else "memory"
    other_before = store._path_for(other).read_bytes()
    before = store._path_for(target).read_bytes()
    snapshot = store.format_for_system_prompt(target)
    content = "Keep notes short and use focused changes"
    operations = [
        {"action": "add", "content": None, "new_text": content},
        {"action": "remove", "old_text": store._entries_for(target)[0]},
    ]
    result = json.loads(mt.memory_tool(target=target, operations=operations, store=store))
    assert result["success"], result
    if route == "approved":
        assert result["staged"]
        assert store._path_for(target).read_bytes() == before
        from hermes_cli.write_approval_commands import handle_pending_subcommand
        reply = handle_pending_subcommand(
            wa.MEMORY, ["approve", result["pending_id"]], memory_store=store)
        assert "Approved 1" in reply, reply
    assert wa.list_pending(wa.MEMORY) == []
    fresh = mt.MemoryStore(memory_char_limit=60, user_char_limit=60)
    fresh.load_from_disk()
    assert fresh._entries_for(target) == [content]
    assert store._path_for(other).read_bytes() == other_before
    assert store.format_for_system_prompt(target) == snapshot
