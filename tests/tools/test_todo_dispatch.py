"""Todo dispatch must not turn misspelled writes into reads (regression for #126656)."""

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
        SimpleNamespace(_todo_store=store), args, InlineToolContext("test-todo"),
    ))


@pytest.mark.parametrize("path", ["registry", "inline"])
@pytest.mark.parametrize("populated", [False, True])
@pytest.mark.parametrize("args", [
    {"action": "add", "list": [{"content": "lost plan"}]},
    {"todos": [], "merg": True},
    {"store": None},
])
def test_unknown_arguments_are_errors_without_mutating_state(path, populated, args):
    store = TodoStore()
    if populated:
        store.write([{"id": "1", "content": "Keep me", "status": "pending"}])
    before = store.snapshot()
    result = _dispatch(path, store, args)
    assert result.get("error"), result
    assert "todos" in result["error"] and "merge" in result["error"]
    for name in args.keys() - {"todos", "merge"}:
        assert name in result["error"]
    assert store.snapshot() == before
    assert "summary" not in result


@pytest.mark.parametrize("path", ["registry", "inline"])
def test_valid_calls_preserve_read_replace_merge_clear_and_error_contracts(path):
    store = TodoStore()
    assert _dispatch(path, store, {})["todos"] == []
    item = {"id": "1", "content": "Plan", "status": "pending"}
    # Registry dispatch receives normalized arguments; inline dispatch owns its
    # coercion wrapper. String recovery is covered at that wrapper's entry point.
    written = _dispatch(path, store, {"todos": [item]})
    assert written["todos"] == [item]
    assert _dispatch(path, store, {}) == written
    merged = _dispatch(path, store, {"todos": [{"id": "1", "status": "completed"}], "merge": True})
    assert merged["todos"] == [{**item, "status": "completed"}]
    assert merged["revision"] > written["revision"]
    before = store.snapshot()
    assert _dispatch(path, store, {"todos": 42}).get("error")
    assert store.snapshot() == before
    assert _dispatch(path, store, {"todos": []})["todos"] == []
    assert _dispatch(path, None, {}).get("error")
