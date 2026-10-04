"""Malformed native arrays must not replace or partially merge a live plan."""
import json
from types import SimpleNamespace

import pytest

from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext
from tools.registry import registry
from tools.todo_tool import TodoStore


def _dispatch(path, store, args):
    if path == "registry":
        return json.loads(registry.dispatch("todo_list", args, store=store))
    return json.loads(INLINE_TOOL_EXECUTORS["todo_list"](
        SimpleNamespace(_todo_store=store), args, InlineToolContext("native-todo"),
    ))


@pytest.mark.parametrize("path", ["registry", "inline"])
@pytest.mark.parametrize("merge", [False, True])
@pytest.mark.parametrize("invalid", [None, 42, "not an object", ["nested"]])
def test_native_invalid_item_rejects_whole_write(path, merge, invalid):
    store = TodoStore()
    store.write([{"id": "1", "content": "Keep this task", "status": "in_progress"}])
    before = store.snapshot()
    # The first row would mutate the existing object during a merge before
    # a later invalid row raises; assert accumulated state, not just the error.
    result = _dispatch(path, store, {"todos": [
        {"id": "1", "content": "Changed", "status": "completed"}, invalid,
    ], "merge": merge})
    assert result.get("error"), result
    assert "object" in result["error"]
    assert "summary" not in result
    assert store.snapshot() == before


@pytest.mark.parametrize("path", ["registry", "inline"])
def test_native_valid_read_write_merge_and_clear(path):
    store = TodoStore()
    item = {"id": "1", "content": "Actual task", "status": "pending"}
    written = _dispatch(path, store, {"todos": [item]})
    assert written["todos"] == [item]
    assert _dispatch(path, store, {}) == written
    merged = _dispatch(path, store, {"todos": [{"id": "1", "status": "completed"}], "merge": True})
    assert merged["todos"] == [{**item, "status": "completed"}]
    assert merged["revision"] > written["revision"]
    assert _dispatch(path, store, {"todos": [item]})["todos"] == [item]
    assert _dispatch(path, store, {"todos": []})["todos"] == []
    assert _dispatch(path, None, {}).get("error")
