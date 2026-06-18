"""Regression tests for TR-F1 — per-task isolation of the resolved-tool-name
scope that ``execute_code`` uses as its sandbox-tool fallback.

Before the fix, ``model_tools._last_resolved_tool_names`` was a single process
global written by every ``get_tool_definitions`` call (including from concurrent
delegation worker threads). When one subagent's worker overwrote it, another
subagent's ``execute_code`` sandbox-scope fallback could read the sibling's tool
list. These tests pin the scope per-thread/context and assert the isolation.
"""

import threading
from unittest import mock

import model_tools


def test_get_prefers_contextvar_over_global():
    """A pinned per-task scope wins over the process global."""
    model_tools._last_resolved_tool_names = ["global_only"]
    token = model_tools.set_resolved_tool_names_scope(["scoped_a", "scoped_b"])
    try:
        assert model_tools.get_last_resolved_tool_names() == ["scoped_a", "scoped_b"]
    finally:
        model_tools.reset_resolved_tool_names_scope(token)


def test_fresh_thread_falls_back_to_global():
    """A thread that never pinned a scope reads the global (back-compat)."""
    model_tools._last_resolved_tool_names = ["g1", "g2"]
    seen = {}

    def worker():
        # Fresh thread => ContextVar default is None => global fallback.
        seen["v"] = model_tools.get_last_resolved_tool_names()

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert seen["v"] == ["g1", "g2"]


def test_concurrent_scopes_isolated_per_thread():
    """Two workers pin different scopes concurrently; neither observes the
    other's. With the old global-only behaviour the last writer would win and
    both reads would return the same list."""
    results = {}
    pinned = threading.Barrier(2)  # both have pinned before either reads

    def worker(name, tools):
        token = model_tools.set_resolved_tool_names_scope(list(tools))
        try:
            pinned.wait(timeout=5)
            results[name] = model_tools.get_last_resolved_tool_names()
        finally:
            model_tools.reset_resolved_tool_names_scope(token)

    a = threading.Thread(target=worker, args=("A", ["a1", "a2", "a3"]))
    b = threading.Thread(target=worker, args=("B", ["b1"]))
    a.start()
    b.start()
    a.join()
    b.join()

    assert results["A"] == ["a1", "a2", "a3"]
    assert results["B"] == ["b1"]


def test_execute_code_fallback_uses_task_scope():
    """handle_function_call('execute_code', enabled_tools=None) must derive the
    sandbox scope from the per-task ContextVar, not the racy global."""
    # Global says one thing; the per-task scope says another. The dispatch must
    # receive the per-task scope.
    model_tools._last_resolved_tool_names = ["global_tool"]
    token = model_tools.set_resolved_tool_names_scope(["task_tool_1", "task_tool_2"])
    captured = {}

    def _fake_dispatch(function_name, function_args, **kwargs):
        captured["enabled_tools"] = kwargs.get("enabled_tools")
        return "{}"

    try:
        with mock.patch.object(model_tools.registry, "dispatch", _fake_dispatch):
            model_tools.handle_function_call(
                "execute_code",
                {"code": "pass"},
                task_id="t-isolation",
                enabled_tools=None,
                skip_pre_tool_call_hook=True,
            )
    finally:
        model_tools.reset_resolved_tool_names_scope(token)

    assert captured["enabled_tools"] == ["task_tool_1", "task_tool_2"]
