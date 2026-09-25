"""Plugin hooks fired from a launch-profile turn see a bound profile scope under multiplexing.

Regression for #118538: routed turns bind ``_profile_runtime_scope(home)`` (HERMES_HOME override +
secrets + terminal policy), but the launch profile's own turns went through
``launch_profile_runtime_scope`` / ``_profile_runtime_scope_tokens(None)`` which bound secrets and
terminal policy only. Under multiplexing an unset override is the fail-closed "unbound context"
signal (``serves_routed_profile``, per-home slots, third-party runtime bindings such as OMH's
``pre_tool_call`` gate), so every plugin hook a launch-profile turn fired looked unscoped and a
fail-closed plugin vetoed every tool call. The dispatcher (``plugins_dispatch``) and the tool
executor already propagate contextvars; the seam is the launch scope itself.

The two hook kinds delivered off-turn by long-lived worker threads (stream observers in
``agent.plugin_stream_hooks``, the plugin event bus in ``plugins_dispatch``) had the sibling gap:
the worker ran callbacks in its own empty context, so even a routed turn's observer saw no scope.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.secret_scope import set_multiplex_active
from hermes_cli import plugins as plugins_mod
from hermes_constants import get_hermes_home, get_hermes_home_override
from tools.daemon_pool import DaemonThreadPoolExecutor
from tools.thread_context import propagate_context_to_thread
import tui_gateway.server as server
from tui_gateway import launch_profile_policy as lpp


@pytest.fixture
def two_homes(tmp_path, monkeypatch):
    launch = tmp_path / "hermes_home"
    routed = launch / "profiles" / "beta"
    for home, tag in ((launch, "alpha"), (routed, "beta")):
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(
            f"plugins:\n  enabled: [stub]\n  entries:\n    stub:\n      settings:\n        x: {tag}\n",
            encoding="utf-8")
        (home / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(server, "_hermes_home", launch)
    monkeypatch.setattr(server, "_served_profile_homes", set())
    monkeypatch.setattr(lpp, "_authority", None)
    plugins_mod._reset_plugin_managers_for_tests()
    seen: list[dict] = []

    def stub_pre_tool_call(**_kw):
        from hermes_cli.config import load_config_readonly
        entry = ((load_config_readonly().get("plugins") or {}).get("entries") or {}).get("stub") or {}
        seen.append({"home": get_hermes_home().name, "x": (entry.get("settings") or {}).get("x"),
                     "bound": get_hermes_home_override() is not None})
        return None

    # Plugin managers are keyed per home: each profile loads its own copy of the plugin.
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    for home in (launch, routed):
        token = set_hermes_home_override(str(home))
        try:
            manager = plugins_mod.get_plugin_manager()
        finally:
            reset_hermes_home_override(token)
        manager._discovered = True  # never scan the real plugin tree
        manager._hooks.setdefault("pre_tool_call", []).append(stub_pre_tool_call)
    yield launch, routed, seen
    set_multiplex_active(False)
    plugins_mod._reset_plugin_managers_for_tests()


def _fire_from_tool_worker():
    """The real hop: a tool worker thread + the bounded hook dispatcher."""
    pool = DaemonThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(propagate_context_to_thread(
            lambda: plugins_mod._dispatch_pre_tool_call_hooks(
                "terminal", {"command": "true"}, tool_call_id="c1", turn_id="t1"))).result(10)
    finally:
        pool.shutdown(wait=False)


def _fire_off_turn_workers(launch_manager_hook_seen):
    """The two worker-thread deliveries: stream observer queue and the plugin event bus."""
    from agent import plugin_stream_hooks
    manager = plugins_mod.get_plugin_manager()
    observed: list[dict] = []

    def observer(**_kw):
        observed.append({"home": get_hermes_home().name, "bound": get_hermes_home_override() is not None})

    manager._hooks.setdefault("on_stream_end", []).append(observer)
    manager._subscribe_event("stub", "stub:tick", observer)
    try:
        assert plugin_stream_hooks.enqueue_plugin_stream_hook("on_stream_end", session_id="s")
        assert manager._dispatch_event("stub:tick", {}) == 1
        assert manager._wait_for_event_dispatch(timeout=5.0)
        plugin_stream_hooks.shutdown_plugin_stream_hook_dispatcher(timeout=5.0)
    finally:
        manager._hooks["on_stream_end"].remove(observer)
        manager._remove_plugin_subscriptions("stub")
    return observed


def test_launch_profile_turn_hooks_see_bound_scope_a_b_a(two_homes):
    launch, routed, seen = two_homes
    set_multiplex_active(True)
    from gateway.run import _profile_runtime_scope

    with lpp.launch_profile_runtime_scope(launch):
        _fire_from_tool_worker()
        off_turn_a = _fire_off_turn_workers(seen)
    with _profile_runtime_scope(routed, {}):
        _fire_from_tool_worker()
        off_turn_b = _fire_off_turn_workers(seen)
    scopes = server._profile_runtime_scope_tokens(None)  # serve backend: launch-profile session
    try:
        _fire_from_tool_worker()
    finally:
        server._release_profile_runtime_scope_tokens(scopes)

    assert [(r["home"], r["x"]) for r in seen] == [
        ("hermes_home", "alpha"), ("beta", "beta"), ("hermes_home", "alpha")]
    assert all(r["bound"] for r in seen), seen
    assert off_turn_a == [{"home": "hermes_home", "bound": True}] * 2, off_turn_a
    assert off_turn_b == [{"home": "beta", "bound": True}] * 2, off_turn_b
    assert get_hermes_home_override() is None  # every scope released


def test_single_profile_launch_scope_binds_no_override(two_homes):
    """Control: a host that never multiplexes keeps the standalone shape (no override, ``os.environ``
    precedence), so single-profile installs are byte-identical."""
    launch, _routed, seen = two_homes
    scopes = server._profile_runtime_scope_tokens(None)
    try:
        assert get_hermes_home_override() is None
        _fire_from_tool_worker()
    finally:
        server._release_profile_runtime_scope_tokens(scopes)
    assert seen == [{"home": "hermes_home", "x": "alpha", "bound": False}]
    assert Path(get_hermes_home()) == launch
