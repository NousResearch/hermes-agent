"""A ``check_fn`` that never answers must not hold the tool-schema build (and the turn) open."""

import json
import threading
from contextvars import ContextVar

import pytest

from tools import registry as registry_module
from tools.registry import ToolRegistry, invalidate_check_fn_cache

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


def test_a_bounded_probe_keeps_scope_errors_and_contextvar_publication():
    from agent.secret_scope import get_secret, reset_secret_scope, set_secret_scope

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
