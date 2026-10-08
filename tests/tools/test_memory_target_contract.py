"""Memory calls must name one store before writing or staging a proposal."""

import copy
import json
from types import SimpleNamespace

import pytest

from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS
from hermes_cli.config import load_config, save_config
from tools.memory_tool import MemoryStore, apply_memory_pending
from tools.registry import registry
from tools.skill_provenance import reset_current_write_origin, set_current_write_origin
from tools import write_approval as wa


def _call(route, store, args):
    if route == "store":
        return store.apply_batch(args.get("target"), args["operations"])
    if route == "replay":
        return apply_memory_pending({"action": "batch", **args}, store)
    if route == "inline":
        agent = SimpleNamespace(_memory_store=store, _memory_manager=None)
        return json.loads(INLINE_TOOL_EXECUTORS["memory"](agent, args, None))
    return json.loads(registry.dispatch("memory", args, store=store))


@pytest.mark.parametrize("route", ["registry", "inline", "store", "replay"])
@pytest.mark.parametrize("origin", ["direct", "foreground", "background_review"])
@pytest.mark.parametrize("target_fields,nested_fields", [
    ({}, {}),
    ({"target": None}, {}),
    ({}, {"target": "user"}),
    ({"target": "memory"}, {"target": "user"}),
    ({"target": "user"}, {"target": "user"}),
    ({"target": "user"}, {"target": None}),
])
def test_invalid_targets_leave_files_and_pending_queue_unchanged(
    route, origin, target_fields, nested_fields,
):
    store = MemoryStore()
    store.load_from_disk()
    for target in ("memory", "user"):
        assert store.add(target, f"Existing {target} fact")["success"]
    before = {t: store._path_for(t).read_bytes() for t in ("memory", "user")}
    config = load_config()
    config.setdefault("memory", {})["write_approval"] = origin != "direct"
    save_config(config)
    # A bad target in a later operation must reject the entire batch.
    args = {**target_fields, "operations": [
        {"action": "add", "content": "First fact"},
        {"action": "add", "content": "Second fact", **nested_fields},
    ]}
    original = copy.deepcopy(args)
    token = set_current_write_origin(origin)
    try:
        result = _call(route, store, args)
    finally:
        reset_current_write_origin(token)
    assert result["success"] is False, result
    assert "target" in result["error"] and "top level" in result["error"]
    assert "retry" in result["error"].lower()
    assert not result.get("staged")
    assert wa.list_pending(wa.MEMORY) == []
    assert args == original
    assert {t: store._path_for(t).read_bytes() for t in before} == before


@pytest.mark.parametrize("route", ["registry", "inline"])
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("target", ["memory", "user"])
def test_missing_target_can_be_corrected_without_writing_to_the_other_store(route, batch, target):
    store = MemoryStore()
    store.load_from_disk()
    for name in ("memory", "user"):
        assert store.add(name, f"Existing {name} fact")["success"]
    other = "user" if target == "memory" else "memory"
    other_before = store._path_for(other).read_bytes()
    snapshot = store.format_for_system_prompt(target)
    operation = {"action": "add", "content": "User prefers metric units"}
    args = {"operations": [operation]} if batch else operation
    rejected = _call(route, store, args)
    assert rejected["success"] is False, rejected
    result = _call(route, store, {**args, "target": target})
    assert result["success"] is True, result
    fresh = MemoryStore()
    fresh.load_from_disk()
    assert fresh._entries_for(target) == [f"Existing {target} fact", operation["content"]]
    assert store._path_for(other).read_bytes() == other_before
    assert store.format_for_system_prompt(target) == snapshot
