"""Regression tests for the owner-aware workspace mutation fence (#95874).

The fence denies known built-in mutating tools for a delegated child whose
workspace domain is quarantined because a prior timed-out / cancelled child
is still live there. Reads, unknown plugin/MCP tools, other domains, and
non-delegated callers are never fenced. Release is driven by real completion.
"""

from __future__ import annotations

import importlib.util
import threading
from pathlib import Path

import pytest

# Importing model_tools triggers discover_builtin_tools(), which registers
# the built-in tools (write_file, read_file, patch, ...) into the registry
# before any test inspects or stubs a handler.
import model_tools  # noqa: F401


def _fence():
    import agent.workspace_mutation_fence as fence
    return fence


def _registry():
    from tools.registry import registry as tool_registry
    return tool_registry


@pytest.fixture(autouse=True)
def _clean_fence(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir()
    fence = _fence()
    fence.reset_for_tests()
    fence.set_grace_seconds_for_tests(0.0)
    yield
    fence.reset_for_tests()
    fence.set_grace_seconds_for_tests(fence.DEFAULT_QUARANTINE_GRACE_SECONDS)


def _stub_handler(name: str, sentinel: str, captured: dict):
    reg = _registry()
    entry = reg._tools.get(name)
    assert entry is not None, f"{name} not registered"

    def _h(args, **kw):
        captured["called"] = True
        captured["args"] = args
        return sentinel

    real = entry.handler
    entry.handler = _h
    return real


def _restore_handler(name: str, real):
    reg = _registry()
    entry = reg._tools.get(name)
    if entry is not None:
        entry.handler = real


def _call(function_name, function_args, *, delegated=False, owner_id=None):
    import model_tools
    from agent.delegation_context import delegated_child_context
    fence = _fence()

    def _do():
        return model_tools.handle_function_call(
            function_name=function_name,
            function_args=function_args,
            task_id="t",
            tool_call_id="tc",
            skip_pre_tool_call_hook=True,
            skip_tool_request_middleware=True,
            skip_tool_execution_middleware=True,
        )

    if delegated:
        with fence.owning_delegated_child(owner_id or "child-2"):
            with delegated_child_context(owner_id or "child-2"):
                return _do()
    return _do()


def test_fence_module_present():
    assert importlib.util.find_spec("agent.workspace_mutation_fence") is not None


def test_mutation_denied_when_stale_live_owner_in_same_domain(tmp_path):
    fence = _fence()
    domain = tmp_path / "workspace-a"
    domain.mkdir()
    target = domain / "f.txt"

    # First child: dispatched, then times out / is cancelled while still live.
    live = threading.Event()
    live.set()  # still live
    fence.bind_owner("child-1", domain, is_live=lambda: live.is_set())
    fence.mark_timeout_or_cancel("child-1")  # grace=0 -> immediate quarantine

    captured = {}
    real = _stub_handler("write_file", '{"ok": true}', captured)
    try:
        result = _call("write_file", {"path": str(target), "content": "x"},
                       delegated=True, owner_id="child-2")
    finally:
        _restore_handler("write_file", real)

    assert "denied" in result.lower(), f"expected deny, got: {result!r}"
    assert not captured.get("called"), "handler should not have been reached"


def test_readonly_in_same_domain_remains_available(tmp_path):
    fence = _fence()
    domain = tmp_path / "workspace-a"
    domain.mkdir()
    target = domain / "f.txt"

    live = threading.Event(); live.set()
    fence.bind_owner("child-1", domain, is_live=lambda: live.is_set())
    fence.mark_timeout_or_cancel("child-1")

    captured = {}
    real = _stub_handler("read_file", '{"ok": true}', captured)
    try:
        result = _call("read_file", {"path": str(target)}, delegated=True, owner_id="child-2")
    finally:
        _restore_handler("read_file", real)

    assert captured.get("called") is True, "read should be admitted"
    assert "denied" not in result.lower()


def test_mutation_in_other_domain_remains_available(tmp_path):
    fence = _fence()
    domain_a = tmp_path / "workspace-a"; domain_a.mkdir()
    domain_b = tmp_path / "workspace-b"; domain_b.mkdir()
    target_b = domain_b / "f.txt"

    live = threading.Event(); live.set()
    fence.bind_owner("child-1", domain_a, is_live=lambda: live.is_set())
    fence.mark_timeout_or_cancel("child-1")

    captured = {}
    real = _stub_handler("write_file", '{"ok": true}', captured)
    try:
        result = _call("write_file", {"path": str(target_b), "content": "x"},
                       delegated=True, owner_id="child-2")
    finally:
        _restore_handler("write_file", real)

    assert captured.get("called") is True, "other-domain mutation should be admitted"
    assert "denied" not in result.lower()


def test_nondelegated_caller_not_fenced(tmp_path):
    fence = _fence()
    domain = tmp_path / "workspace-a"; domain.mkdir()
    target = domain / "f.txt"

    live = threading.Event(); live.set()
    fence.bind_owner("child-1", domain, is_live=lambda: live.is_set())
    fence.mark_timeout_or_cancel("child-1")

    captured = {}
    real = _stub_handler("write_file", '{"ok": true}', captured)
    try:
        result = _call("write_file", {"path": str(target), "content": "x"}, delegated=False)
    finally:
        _restore_handler("write_file", real)

    assert captured.get("called") is True, "non-delegated caller must not be fenced"
    assert "denied" not in result.lower()


def test_release_on_real_completion_re_admits(tmp_path):
    fence = _fence()
    domain = tmp_path / "workspace-a"; domain.mkdir()
    target = domain / "f.txt"

    live = threading.Event(); live.set()
    fence.bind_owner("child-1", domain, is_live=lambda: live.is_set())
    fence.mark_timeout_or_cancel("child-1")

    # While stale-live: denied.
    captured = {}
    real = _stub_handler("write_file", '{"ok": true}', captured)
    try:
        denied = _call("write_file", {"path": str(target), "content": "x"},
                        delegated=True, owner_id="child-2")
        assert "denied" in denied.lower()

        # Stale owner actually finishes -> release.
        live.clear()
        fence.release_owner("child-1")

        result = _call("write_file", {"path": str(target), "content": "x"},
                       delegated=True, owner_id="child-2")
    finally:
        _restore_handler("write_file", real)

    assert captured.get("called") is True, "after release, mutation should be admitted"
    assert "denied" not in result.lower()


def test_stale_owner_own_mutations_denied_via_gate(tmp_path):
    fence = _fence()
    domain = tmp_path / "workspace-a"; domain.mkdir()
    target = domain / "f.txt"

    live = threading.Event(); live.set()
    fence.bind_owner("child-1", domain, is_live=lambda: live.is_set())
    fence.mark_timeout_or_cancel("child-1")

    # The stale owner itself (same owner_id) has its mutation gate closed.
    captured = {}
    real = _stub_handler("write_file", '{"ok": true}', captured)
    try:
        result = _call("write_file", {"path": str(target), "content": "x"},
                       delegated=True, owner_id="child-1")
    finally:
        _restore_handler("write_file", real)

    assert "denied" in result.lower()
    assert not captured.get("called")


def test_unknown_plugin_tool_not_fenced(tmp_path):
    """Unknown plugin/MCP tools are left outside the initial fencing claim."""
    fence = _fence()
    domain = tmp_path / "workspace-a"; domain.mkdir()

    live = threading.Event(); live.set()
    fence.bind_owner("child-1", domain, is_live=lambda: live.is_set())
    fence.mark_timeout_or_cancel("child-1")

    # An unknown tool name is not in KNOWN_BUILTIN_MUTATING_TOOLS -> never fenced.
    assert "some_plugin_tool" not in fence.KNOWN_BUILTIN_MUTATING_TOOLS
    msg = fence.deny_delegated_mutation("some_plugin_tool", {"path": str(domain / "x")})
    assert msg is None


def test_run_single_child_arms_fence_on_timeout(tmp_path, monkeypatch):
    """End-to-end wiring: the real ``_run_single_child`` binds an owner,
    marks it stale on timeout, and releases on real completion — so a later
    delegated mutation in the same domain is denied while the child is live
    and re-admitted after the child actually finishes.
    """
    import tools.delegate_tool as dt
    fence = _fence()
    fence.reset_for_tests()
    fence.set_grace_seconds_for_tests(0.0)

    domain = tmp_path / "workspace-a"
    domain.mkdir()
    target = domain / "f.txt"

    release = threading.Event()

    class _StubChild:
        _subagent_id = "stale-child-1"
        session_id = "stale-child-1"
        _delegate_role = "leaf"
        _delegate_workspace_domain = str(domain)
        tool_progress_callback = None
        _credential_pool = None
        _delegate_saved_tool_names = []
        _delegate_output_schema = None

        def run_conversation(self, user_message, task_id=None, stream_callback=None):
            release.wait(5.0)
            return {"final_response": "done"}

        def get_activity_summary(self):
            return {"api_call_count": 0, "max_iterations": 0,
                    "current_tool": None, "last_activity_ts": None}

    class _StubParent:
        platform = "cli"
        _delegate_depth = 1
        _active_children = []
        _active_children_lock = threading.Lock()
        _print_fn = None
        tool_progress_callback = None
        thinking_callback = None

        def _touch_activity(self, *a, **k):
            pass

    monkeypatch.setattr(dt, "_get_child_timeout", lambda: 0.3)
    monkeypatch.setattr(dt, "_dump_subagent_timeout_diagnostic",
                        lambda **kw: None)
    monkeypatch.setattr(dt, "_register_subagent", lambda *a, **k: None)
    monkeypatch.setattr(dt, "_unregister_subagent", lambda *a, **k: None)

    child = _StubChild()
    parent = _StubParent()

    entry = dt._run_single_child(
        task_index=0,
        goal="work",
        child=child,
        parent_agent=parent,
    )
    assert entry["status"] == "timeout"

    # While the stale child is still live (release not yet set), a later
    # delegated mutation in the same domain must be denied.
    captured = {}
    real = _stub_handler("write_file", '{"ok": true}', captured)
    try:
        denied = _call("write_file", {"path": str(target), "content": "x"},
                       delegated=True, owner_id="child-2")
    finally:
        _restore_handler("write_file", real)
    assert "denied" in denied.lower()
    assert not captured.get("called")

    # Now the stale child actually finishes -> release fires -> re-admitted.
    release.set()
    import time as _time
    _time.sleep(0.3)

    captured2 = {}
    real2 = _stub_handler("write_file", '{"ok": true}', captured2)
    try:
        result = _call("write_file", {"path": str(target), "content": "x"},
                       delegated=True, owner_id="child-2")
    finally:
        _restore_handler("write_file", real2)
    assert captured2.get("called") is True
    assert "denied" not in result.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
