"""Auth/reconnect waiting must be bounded per turn.

Each recovery wait was individually reasonable and collectively awful. One
failing MCP tool call could burn, in sequence:

    5s   waiting for a session to reappear
  + 10s  in the OAuth manager's handle_401
  + 15s  waiting on the reconnect that triggers
  + 15s  waiting on the session-expired reconnect when that path declined
  ------
    45s  of pure waiting, before the retry RPC even starts

per call, with nothing capping the total across several failing tools in one
turn. From the model's side that is indistinguishable from a hang.

The waits now share one per-turn budget. These tests pin the arithmetic, the
skip behavior once it is spent, and — importantly — that a spent budget stops
Hermes WAITING without stopping it RECOVERING: the reconnect is still
signalled so the server task rebuilds in the background.
"""
from __future__ import annotations

import json
import threading
import time
from unittest.mock import MagicMock

import pytest


pytest.importorskip("mcp.client.auth.oauth2")


@pytest.fixture
def mcp(monkeypatch, tmp_path):
    """The budget module (arithmetic + the clamp every call site goes through)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool  # noqa: F401 — origin first: the split modules resolve state through it
    from tools import mcp_tool_recovery

    mcp_tool_recovery._reset_recovery_budget_for_tests()
    yield mcp_tool_recovery
    mcp_tool_recovery._reset_recovery_budget_for_tests()


def _bind_turn(turn_id):
    from tools import approval_context

    return approval_context.set_current_observability_context(turn_id=turn_id)


@pytest.fixture
def in_turn():
    """Bind a turn id through the real contextvar, as the tool dispatcher does around every call."""
    from tools import approval_context

    tokens = _bind_turn("turn-1")
    yield "turn-1"
    approval_context.reset_current_observability_context(tokens)


@pytest.fixture
def handlers():
    from tools import mcp_tool_handlers

    return mcp_tool_handlers


@pytest.fixture
def loop_mod():
    from tools import mcp_tool_loop

    return mcp_tool_loop


# ---------------------------------------------------------------------------
# Budget arithmetic
# ---------------------------------------------------------------------------


def test_a_fresh_turn_has_the_full_budget(mcp, in_turn):
    assert mcp._recovery_budget_remaining() == mcp._TURN_RECOVERY_BUDGET_SEC


def test_spending_reduces_the_remaining_budget(mcp, in_turn):
    mcp._charge_recovery_budget(4.0)
    assert mcp._recovery_budget_remaining() == pytest.approx(
        mcp._TURN_RECOVERY_BUDGET_SEC - 4.0
    )
    mcp._charge_recovery_budget(4.0)
    assert mcp._recovery_budget_remaining() == pytest.approx(
        mcp._TURN_RECOVERY_BUDGET_SEC - 8.0
    )


def test_the_budget_floors_at_zero(mcp, in_turn):
    mcp._charge_recovery_budget(1000.0)
    assert mcp._recovery_budget_remaining() == 0.0


def test_the_total_across_many_waits_cannot_exceed_the_budget(mcp, in_turn):
    """The point of the whole change: a bound on the TURN, not per call."""
    granted = []
    for _ in range(20):
        with mcp._recovery_wait("probe", 15.0) as allowed:
            granted.append(allowed)
            mcp._charge_recovery_budget(allowed)
    assert sum(granted) <= mcp._TURN_RECOVERY_BUDGET_SEC + 0.5
    assert granted[0] == mcp._TURN_RECOVERY_BUDGET_SEC
    assert granted[-1] == 0.0


def test_concurrent_waits_cannot_each_take_the_whole_budget(mcp, in_turn):
    """Tools of one turn run concurrently: the grant is reserved before the wait starts, so
    overlapping waits split the budget instead of each reading the same remainder."""
    granted, inside, release = [], threading.Barrier(3), threading.Event()

    def _waiter():
        _bind_turn(in_turn)
        with mcp._recovery_wait("probe", 15.0) as allowed:
            granted.append(allowed)
            inside.wait(timeout=10)
            release.wait(timeout=10)

    threads = [threading.Thread(target=_waiter) for _ in range(2)]
    for t in threads:
        t.start()
    inside.wait(timeout=10)  # both are inside their wait at the same time
    release.set()
    for t in threads:
        t.join()
    assert sorted(granted) == [0.0, mcp._TURN_RECOVERY_BUDGET_SEC]


def test_an_unused_reservation_is_returned_to_the_turn(mcp, in_turn):
    with mcp._recovery_wait("probe", 10.0):
        pass  # the session was already back: nothing was waited
    assert mcp._recovery_budget_remaining() == pytest.approx(mcp._TURN_RECOVERY_BUDGET_SEC, abs=0.5)


def test_a_wait_is_clamped_to_what_is_left(mcp, in_turn):
    mcp._charge_recovery_budget(mcp._TURN_RECOVERY_BUDGET_SEC - 2.0)
    with mcp._recovery_wait("probe", 15.0) as allowed:
        assert allowed == pytest.approx(2.0)


def test_a_wait_never_gets_more_than_it_asked_for(mcp, in_turn):
    with mcp._recovery_wait("probe", 3.0) as allowed:
        assert allowed == 3.0


def test_elapsed_time_inside_the_block_is_charged(mcp, in_turn):
    before = mcp._recovery_budget_remaining()
    with mcp._recovery_wait("probe", 10.0):
        time.sleep(0.2)
    assert before - mcp._recovery_budget_remaining() >= 0.2


def test_a_raising_wait_still_charges_its_budget(mcp, in_turn):
    """A wait that blows up must not leak budget back to the turn."""
    before = mcp._recovery_budget_remaining()
    with pytest.raises(RuntimeError):
        with mcp._recovery_wait("probe", 10.0):
            time.sleep(0.15)
            raise RuntimeError("transport exploded mid-wait")
    assert before - mcp._recovery_budget_remaining() >= 0.15


# ---------------------------------------------------------------------------
# Scoping
# ---------------------------------------------------------------------------


def test_turns_have_independent_budgets(mcp):
    from tools import approval_context

    tokens = _bind_turn("turn-a")
    try:
        mcp._charge_recovery_budget(mcp._TURN_RECOVERY_BUDGET_SEC)
        assert mcp._recovery_budget_remaining() == 0.0
    finally:
        approval_context.reset_current_observability_context(tokens)

    tokens = _bind_turn("turn-b")
    try:
        assert mcp._recovery_budget_remaining() == mcp._TURN_RECOVERY_BUDGET_SEC
    finally:
        approval_context.reset_current_observability_context(tokens)


def test_calls_outside_a_turn_are_never_starved(mcp):
    """Startup discovery and CLI paths have no turn to protect."""
    assert mcp._current_recovery_turn_key() == ""
    mcp._charge_recovery_budget(1000.0)
    assert mcp._recovery_budget_remaining() == mcp._TURN_RECOVERY_BUDGET_SEC


def test_the_registry_is_bounded(mcp):
    from tools import approval_context

    for i in range(mcp._MAX_TRACKED_RECOVERY_TURNS + 30):
        tokens = _bind_turn(f"turn-{i}")
        try:
            mcp._charge_recovery_budget(1.0)
        finally:
            approval_context.reset_current_observability_context(tokens)
    assert len(mcp._turn_recovery_spent) == mcp._MAX_TRACKED_RECOVERY_TURNS


def test_concurrent_charges_are_serialised(mcp, in_turn):
    def _worker():  # a new thread starts with an empty context: bind the same turn in it
        _bind_turn(in_turn)
        for _ in range(100):
            mcp._charge_recovery_budget(0.01)

    threads = [threading.Thread(target=_worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    spent = mcp._TURN_RECOVERY_BUDGET_SEC - mcp._recovery_budget_remaining()
    assert spent == pytest.approx(8.0, abs=0.01)


# ---------------------------------------------------------------------------
# Behavior at the call sites
# ---------------------------------------------------------------------------


def _down_server(name):
    server = MagicMock()
    server.name = name
    server.session = None
    server._is_recycled_stdio.return_value = False
    return server


def _forget_breaker(name):
    from tools import mcp_tool

    mcp_tool._server_error_counts.pop(name, None)
    mcp_tool._server_breaker_opened_at.pop(name, None)


def test_a_spent_budget_skips_the_session_ready_wait(mcp, in_turn, handlers, loop_mod, monkeypatch):
    """The regression: the turn must not sit on a wait it cannot afford — and it must stop
    WAITING without stopping RECOVERING: the reconnect is still signalled."""
    from tools import mcp_tool_discovery

    waited, signalled = {"n": 0}, {"n": 0}

    def _never_ready(*a, **kw):
        waited["n"] += 1
        time.sleep(kw.get("timeout", 0.0))
        return False

    def _signal(server):
        signalled["n"] += 1
        return True

    server = _down_server("srv")
    monkeypatch.setattr(loop_mod, "_wait_for_server_session_ready", _never_ready)
    monkeypatch.setattr(loop_mod, "_signal_reconnect", _signal)
    monkeypatch.setattr(mcp_tool_discovery, "_get_connected_server_for_call", lambda name: server)
    try:
        mcp._charge_recovery_budget(mcp._TURN_RECOVERY_BUDGET_SEC)
        started = time.monotonic()
        acquired, error = handlers._acquire_call_server("srv", 60.0)
        elapsed = time.monotonic() - started

        assert acquired is None
        assert waited["n"] == 0, "waited on a budget that was already spent"
        assert elapsed < 2.0
        assert signalled["n"] == 1
        assert "reconnect" in json.loads(error)["error"].lower()
    finally:
        _forget_breaker("srv")


def test_the_session_ready_wait_is_charged_to_the_turn(mcp, in_turn, handlers, loop_mod, monkeypatch):
    from tools import mcp_tool_discovery

    granted = []

    def _never_ready(srv, *, old_session=None, timeout=15.0):
        granted.append(timeout)
        time.sleep(0.2)
        return False

    server = _down_server("srv1")
    monkeypatch.setattr(loop_mod, "_wait_for_server_session_ready", _never_ready)
    monkeypatch.setattr(loop_mod, "_signal_reconnect", lambda s: True)
    monkeypatch.setattr(mcp_tool_discovery, "_get_connected_server_for_call", lambda name: server)
    try:
        handlers._acquire_call_server("srv1", 60.0)
        assert granted == [5.0]
        assert mcp._TURN_RECOVERY_BUDGET_SEC - mcp._recovery_budget_remaining() >= 0.2
    finally:
        _forget_breaker("srv1")


def test_a_spent_budget_short_circuits_oauth_recovery(mcp, in_turn, handlers, loop_mod, monkeypatch):
    """handle_401 must not be entered with no time to wait on it."""
    monkeypatch.setattr(handlers, "_is_auth_error", lambda exc: True)
    called = {"n": 0}

    def _loop(*a, **kw):
        called["n"] += 1
        return True

    monkeypatch.setattr(loop_mod, "_run_on_mcp_loop", _loop)

    mcp._charge_recovery_budget(mcp._TURN_RECOVERY_BUDGET_SEC)
    try:
        out = json.loads(handlers._handle_auth_error_and_retry(
            "srv3", Exception("401"), lambda: "{}", "tools/call op"))
        assert called["n"] == 0, "entered OAuth recovery with no budget"
        # Recovery was never attempted, so this is NOT a verdict that the user must re-authenticate:
        # a refreshable token must not send them to `hermes mcp login`.
        assert "needs_reauth" not in out
        assert "NOT attempted" in out["error"]
    finally:
        _forget_breaker("srv3")


def test_recovery_still_runs_normally_with_budget_available(mcp, in_turn, handlers, loop_mod, monkeypatch):
    """The bound must not disable recovery on an ordinary first failure."""
    from tools import mcp_tool

    granted = {}

    def _run(factory, timeout=None):
        granted["handle_401"] = timeout
        return True

    def _reconnect(server_name, srv, *, op_description, timeout=15.0):
        granted["reconnect"] = timeout
        return True

    monkeypatch.setattr(handlers, "_is_auth_error", lambda exc: True)
    monkeypatch.setattr(loop_mod, "_run_on_mcp_loop", _run)
    monkeypatch.setattr(loop_mod, "_signal_reconnect_and_wait", _reconnect)
    server = MagicMock()
    server._reconnect_event = MagicMock()
    mcp_tool._servers["srv4"] = server
    try:
        out = handlers._handle_auth_error_and_retry(
            "srv4", Exception("401"), lambda: json.dumps({"result": "ok"}), "tools/call op")
        assert json.loads(out)["result"] == "ok"
        assert granted["handle_401"] == 10.0
        assert 0.0 < granted["reconnect"] <= 15.0
    finally:
        mcp_tool._servers.pop("srv4", None)
        _forget_breaker("srv4")


@pytest.mark.parametrize("side_effects", [False, True])
def test_a_spent_budget_still_signals_the_session_expired_reconnect(
        mcp, in_turn, handlers, loop_mod, monkeypatch, side_effects):
    """Read and write-capable paths alike: the reconnect is requested with a zero wait."""
    granted = []

    def _reconnect(server_name, srv, *, op_description, timeout=15.0):
        granted.append(timeout)
        return False

    monkeypatch.setattr(handlers, "_is_session_expired_error", lambda exc: True)
    monkeypatch.setattr(handlers, "_lookup_reconnectable_server", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(loop_mod, "_signal_reconnect_and_wait", _reconnect)
    mcp._charge_recovery_budget(mcp._TURN_RECOVERY_BUDGET_SEC)
    try:
        handlers._handle_session_expired_and_retry(
            "srv5", Exception("session expired"), lambda: "{}", "tools/call op",
            call_may_have_side_effects=side_effects)
        assert granted == [0.0]
    finally:
        _forget_breaker("srv5")


def test_a_zero_wait_checks_the_session_once_and_never_sleeps(loop_mod, monkeypatch):
    """What the zero-budget reconnect relies on in the real poll helper."""
    monkeypatch.setattr(loop_mod.time, "sleep", lambda s: pytest.fail("slept on a zero wait"))
    ready = MagicMock()
    ready.is_set.return_value = True
    assert loop_mod._wait_for_server_session_ready(MagicMock(session=None, _ready=ready), timeout=0.0) is False
    assert loop_mod._wait_for_server_session_ready(MagicMock(session=object(), _ready=ready), timeout=0.0) is True


def test_worst_case_turn_impact_stays_within_the_budget(mcp, in_turn, monkeypatch):
    """Drive every recovery wait back to back and measure the total.

    Each site sleeps for whatever it is granted, so the elapsed wall clock is
    the turn impact this change exists to bound.
    """
    # Scale the real budget down rather than sleeping through 15s of it: the
    # mechanism under test is the shared cap, not its exact value.
    monkeypatch.setattr(mcp, "_TURN_RECOVERY_BUDGET_SEC", 2.0)

    started = time.monotonic()
    for op, requested in [
        ("session-ready", 5.0),
        ("oauth-recovery", 10.0),
        ("oauth-reconnect", 15.0),
        ("session-expired-reconnect", 15.0),
    ] * 3:
        with mcp._recovery_wait(op, requested) as allowed:
            if allowed > 0:
                time.sleep(min(allowed, 20.0))
    elapsed = time.monotonic() - started

    # Pre-fix this sequence would have slept 45s per round, 135s in total.
    assert elapsed <= mcp._TURN_RECOVERY_BUDGET_SEC + 1.0, (
        f"a turn spent {elapsed:.1f}s waiting on MCP recovery; the budget is "
        f"{mcp._TURN_RECOVERY_BUDGET_SEC:.0f}s"
    )


# ---------------------------------------------------------------------------
# The wiring that makes the budget real
# ---------------------------------------------------------------------------


def test_the_turn_key_reaches_a_tool_handler_for_real(mcp, tmp_path, monkeypatch):
    """Without this, the entire budget is a silent no-op.

    The turn key comes from a contextvar the tool dispatcher binds around
    dispatch. MCP tool handlers run inside that dispatch — but if the
    contextvar were not visible there (a different thread, a lost context),
    every budget lookup would fall into the "no turn to protect" branch, the
    cap would never apply, and nothing would fail: waits would simply go back
    to being unbounded. So exercise the REAL dispatcher, with no
    monkeypatching of the turn id.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from model_tools import handle_function_call
    from tools.registry import registry, tool_result

    seen = {}

    def _handler(args, **kw):
        seen["key"] = mcp._current_recovery_turn_key()
        seen["before"] = mcp._recovery_budget_remaining()
        mcp._charge_recovery_budget(5.0)
        seen["after"] = mcp._recovery_budget_remaining()
        return tool_result(ok=True)

    registry.register(
        name="_budget_wiring_probe",
        toolset="testing",
        schema={
            "name": "_budget_wiring_probe",
            "description": "test only",
            "parameters": {"type": "object", "properties": {}},
        },
        handler=_handler,
    )
    try:
        handle_function_call("_budget_wiring_probe", {}, turn_id="TURN-XYZ")

        assert seen["key"] == "TURN-XYZ", (
            "the turn key is not visible inside a tool handler — the recovery "
            "budget would never apply to any MCP wait"
        )
        assert seen["before"] == mcp._TURN_RECOVERY_BUDGET_SEC
        # And the charge is scoped to that turn, not discarded.
        assert seen["after"] == pytest.approx(
            mcp._TURN_RECOVERY_BUDGET_SEC - 5.0
        )
    finally:
        registry._tools.pop("_budget_wiring_probe", None)
