"""A ``check_fn`` that never answers must not hold the tool-schema build (and the turn) open."""

import json
import threading
from contextvars import ContextVar

import pytest

from tests.waiting import wait_until
from hermes_agent.tools import registry as registry_module
from hermes_agent.tools.registry import ToolRegistry, invalidate_check_fn_cache

_RELEASE = threading.Event()
_published: ContextVar = ContextVar("probe_published", default=None)


def _register(reg, name, check_fn):
    reg.register(name=name, toolset="probe", handler=lambda args, **kw: json.dumps({"ok": True}),
                 schema={"name": name, "description": name, "parameters": {"type": "object", "properties": {}}},
                 check_fn=check_fn)


def _build_within(reg, names, seconds):
    """Run the build on a thread the test can abandon, so a regression fails instead of hanging."""
    out = {}
    worker = threading.Thread(target=lambda: out.setdefault("defs", reg.get_definitions(names, quiet=True)),
                              daemon=True)
    worker.start()
    worker.join(seconds)
    return worker.is_alive(), sorted(d["function"]["name"] for d in out.get("defs", []))


@pytest.fixture(autouse=True)
def _short_budget(monkeypatch):
    monkeypatch.setattr(registry_module, "_CHECK_FN_PROBE_TIMEOUT_SECONDS", 0.3, raising=False)
    _RELEASE.clear()
    invalidate_check_fn_cache()
    yield
    _RELEASE.set()
    invalidate_check_fn_cache()


def test_a_wedged_check_fn_drops_only_its_tool_and_is_not_restarted():
    calls = []
    entered = threading.Event()

    def wedged():
        calls.append(1)
        entered.set()
        _RELEASE.wait(30)
        return True

    reg = ToolRegistry()
    _register(reg, "wedged", wedged)
    _register(reg, "fine", lambda: True)
    _register(reg, "uncached_wedged", registry_module.no_cache_check_fn(lambda: _RELEASE.wait(30)))

    # Two cold builds at once: they share one worker for the wedged probe instead of each starting
    # (and abandoning) its own.
    builds = [threading.Thread(target=_build_within, args=(reg, {"wedged", "fine", "uncached_wedged"}, 5.0),
                               daemon=True) for _ in range(2)]
    for build in builds:
        build.start()
    assert entered.wait(5.0)
    stuck, names = _build_within(reg, {"wedged", "fine", "uncached_wedged"}, 5.0)
    for build in builds:
        build.join(5.0)
    assert not stuck, "a wedged check_fn held the tool-schema build open"
    assert names == ["fine"]

    # A later build (TTL expired) answers from the parked probe instead of a fresh thread.
    invalidate_check_fn_cache()
    stuck, names = _build_within(reg, {"wedged", "fine"}, 5.0)
    assert not stuck and names == ["fine"]
    assert len(calls) == 1

    # Once the wedged probe finally answers, its record leaves the table (MCP refreshes mint a new
    # check_fn each time, so nothing else would ever replace it).
    _RELEASE.set()
    assert wait_until(lambda: not registry_module._check_fn_inflight, timeout=5.0)


def test_a_probe_thread_that_cannot_start_does_not_park_the_check(monkeypatch):
    class _NoThreads:
        def __init__(self, *a, **kw):
            pass

        def start(self):
            raise RuntimeError("can't start new thread")

    reg = ToolRegistry()
    _register(reg, "flaky_host", lambda: True)
    monkeypatch.setattr(registry_module, "_ProbeThread", _NoThreads)
    assert reg.get_definitions({"flaky_host"}, quiet=True) == []
    assert not registry_module._check_fn_inflight

    # Thread capacity is back: the very next check runs instead of answering "still not answering".
    monkeypatch.setattr(registry_module, "_ProbeThread", threading.Thread)
    invalidate_check_fn_cache()
    assert [d["function"]["name"] for d in reg.get_definitions({"flaky_host"}, quiet=True)] == ["flaky_host"]


def test_an_uncached_check_answers_for_its_own_caller_while_another_runs(monkeypatch):
    """``no_cache_check_fn`` probes may answer per caller context (Kanban refuses a delegated child),
    so a concurrent caller must never inherit another caller's verdict."""
    from contextlib import nullcontext

    from hermes_agent.agent.delegation_context import delegated_child_context
    from hermes_agent.tools import kanban_tools

    entered, release, results = threading.Event(), threading.Event(), {}

    def configured_profile():
        if not entered.is_set():
            entered.set()
            assert release.wait(5)
        return True

    monkeypatch.setattr(kanban_tools, "_profile_has_kanban_toolset", configured_profile)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    # The parent's probe is held until the child has answered: give it a budget a loaded runner
    # cannot exhaust (the file's 0.3 s default is for the wedged-probe tests).
    monkeypatch.setattr(registry_module, "_CHECK_FN_PROBE_TIMEOUT_SECONDS", 10.0)
    reg = ToolRegistry()
    _register(reg, "kanban_probe", kanban_tools._check_kanban_mode)

    def build(name, child):
        with delegated_child_context() if child else nullcontext():
            results[name] = [d["function"]["name"] for d in reg.get_definitions({"kanban_probe"}, quiet=True)]

    parent = threading.Thread(target=build, args=("parent", False), daemon=True)
    parent.start()
    assert entered.wait(5)
    child = threading.Thread(target=build, args=("child", True), daemon=True)
    child.start()
    child.join(5)
    release.set()
    parent.join(5)
    assert results == {"parent": ["kanban_probe"], "child": []}


def test_a_bounded_probe_keeps_scope_errors_and_contextvar_publication():
    from hermes_agent.agent.secret_scope import get_secret, reset_secret_scope, set_secret_scope

    def publishing_probe():
        _published.set(get_secret("PROBE_ONLY_KEY"))
        return True

    def raising_probe():
        raise RuntimeError("boom")

    reg = ToolRegistry()
    _register(reg, "scoped", publishing_probe)
    _register(reg, "raises", raising_probe)
    token = set_secret_scope({"PROBE_ONLY_KEY": "s3cret"})
    try:
        names = sorted(d["function"]["name"] for d in reg.get_definitions({"scoped", "raises"}, quiet=True))
    finally:
        reset_secret_scope(token)
    assert names == ["scoped"]
    assert _published.get() == "s3cret"
