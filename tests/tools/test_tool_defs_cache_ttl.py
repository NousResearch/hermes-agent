"""Regression tests for #84726: the outer tool-definitions memo in
``model_tools.get_tool_definitions()`` must never outlive the registry's
check_fn TTL.

Before the fix, the memo was keyed only on toolsets / generation / config /
profile flags — a hit returned without consulting the registry, so a service
that came up or went down stayed hidden or announced indefinitely (the
registry's 30 s check_fn TTL + 60 s failure grace were unenforceable). The
fix rolls a time-bucket aligned with ``_CHECK_FN_TTL_SECONDS`` into the
cache key and refreshes LRU order on hit.
"""

import pytest

import model_tools
from tools import registry as reg


@pytest.fixture
def ttl_probe(monkeypatch):
    """Register an isolated probe tool whose check_fn reads mutable state,
    with the shared monotonic clock under test control.

    ``model_tools`` and ``tools.registry`` both ``import time`` — the same
    module object — so one patch drives both the outer bucket and the inner
    check_fn TTL cache.
    """
    clock = {"now": 1000.0}
    monkeypatch.setattr(reg.time, "monotonic", lambda: clock["now"])

    # Keep the tool-search progressive-disclosure assembly out of the way:
    # it would defer the probe tool (a plugin toolset) behind the three
    # bridge tools, masking the check_fn TTL behavior under test.
    from tools.tool_search import ToolSearchConfig

    monkeypatch.setattr(
        "tools.tool_search.load_config",
        lambda: ToolSearchConfig.from_raw({"enabled": "off"}),
    )

    state = {"up": True}
    probe_calls = {"n": 0}

    def _check_fn():
        probe_calls["n"] += 1
        return state["up"]

    reg.registry.register(
        name="__ttl_probe_tool__",
        toolset="__ttl_probe_toolset__",
        schema={
            "name": "__ttl_probe_tool__",
            "description": "availability probe for #84726",
            "parameters": {"type": "object", "properties": {}},
        },
        handler=lambda args, **kw: "",
        check_fn=_check_fn,
    )
    model_tools._clear_tool_defs_cache()
    reg.invalidate_check_fn_cache()
    try:
        yield state, probe_calls, clock
    finally:
        reg.registry.deregister("__ttl_probe_tool__")
        model_tools._clear_tool_defs_cache()
        reg.invalidate_check_fn_cache()


def _probe_names():
    return {
        t["function"]["name"]
        for t in model_tools.get_tool_definitions(
            enabled_toolsets=["__ttl_probe_toolset__"],
            quiet_mode=True,
        )
    }


def _advance_past_ttl_and_grace(clock):
    """Roll the clock past the check_fn TTL AND the failure-grace window so
    the registry re-probes and honors a flipped verdict."""
    clock["now"] += (
        reg._CHECK_FN_TTL_SECONDS + reg._CHECK_FN_FAILURE_GRACE_SECONDS + 1
    )


def test_service_down_disappears_within_check_fn_ttl(ttl_probe):
    """#84726: a tool whose check_fn flips to False stops being announced
    within the registry TTL horizon — no manual outer-cache clear needed."""
    state, probe_calls, clock = ttl_probe
    assert "__ttl_probe_tool__" in _probe_names()

    state["up"] = False  # service goes down
    _advance_past_ttl_and_grace(clock)

    assert "__ttl_probe_tool__" not in _probe_names()


def test_service_up_appears_within_check_fn_ttl(ttl_probe):
    """#84726: a tool whose check_fn flips to True stops being hidden within
    the registry TTL horizon (the #35561 cronjob-after-env-set case)."""
    state, probe_calls, clock = ttl_probe
    # First resolution happens while the service is down.
    state["up"] = False
    assert "__ttl_probe_tool__" not in _probe_names()

    state["up"] = True  # service comes up
    _advance_past_ttl_and_grace(clock)

    assert "__ttl_probe_tool__" in _probe_names()


def test_hit_within_same_bucket_uses_outer_cache(ttl_probe):
    """#84726: within the same time-bucket a memo hit must NOT consult the
    registry — the ~570x fast path is preserved."""
    state, probe_calls, clock = ttl_probe
    assert "__ttl_probe_tool__" in _probe_names()
    assert probe_calls["n"] == 1  # probe ran once on the initial miss

    # Same bucket, same everything: the hit must serve from the memo without
    # re-running the check_fn probe.
    assert "__ttl_probe_tool__" in _probe_names()
    assert probe_calls["n"] == 1


def test_hit_refreshes_lru_order(ttl_probe):
    """#84726: a hit moves the entry to the most-recent end so cap eviction
    (_TOOL_DEFS_CACHE_MAX) is a real LRU instead of FIFO."""
    state, probe_calls, clock = ttl_probe
    model_tools._clear_tool_defs_cache()

    key_a = frozenset(["__ttl_probe_toolset__"])
    key_b = frozenset(["__ttl_probe_toolset__", "__other_probe_toolset__"])

    model_tools.get_tool_definitions(
        enabled_toolsets=["__ttl_probe_toolset__"], quiet_mode=True
    )
    model_tools.get_tool_definitions(
        enabled_toolsets=["__ttl_probe_toolset__", "__other_probe_toolset__"],
        quiet_mode=True,
    )

    # A inserted first, then B: A sits at the eviction frontier.
    keys = list(model_tools._tool_defs_cache)
    assert len(keys) == 2
    assert keys[0][0] == key_a
    assert keys[1][0] == key_b

    # Hitting A again must move it to the most-recent end.
    model_tools.get_tool_definitions(
        enabled_toolsets=["__ttl_probe_toolset__"], quiet_mode=True
    )
    keys = list(model_tools._tool_defs_cache)
    assert keys[0][0] == key_b
    assert keys[1][0] == key_a
