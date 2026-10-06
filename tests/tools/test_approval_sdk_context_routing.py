"""SDK approvals on a reused session route by the runtime's per-turn snapshot, never by the context
inherited from the turn that created the session.

The SDK reader task, and so every ``can_use_tool`` approval worker, runs in a copy of the context of
the turn that created the session (``run_coroutine_threadsafe`` captures the caller's context;
measured with claude-agent-sdk 0.2.144). A live contextvar read there names that first turn's
session key and cron posture. With a ``context_provider`` the validated snapshot is authoritative
for the request, a malformed snapshot fails closed, and provider-less callers keep the live read.
"""

import contextvars
import json
import threading

import pytest

from gateway.session_context import set_session_vars
from tools import approval as mod
from tools import approval_context as ctx
from tools import approval_sdk_gateway as sdk_gateway
from tools import approval_session_notify as session_notify

WAIT_S = 30.0  # generous bound for every join
CANONICAL = json.dumps({"tool_name": "Bash", "tool_input": {"command": "true"}},
                       sort_keys=True, separators=(",", ":"))
NO_APPROVER = {"choice": "deny", "reason": "no approver available (background context)"}


class _Operators:
    """Recording approvers that answer ``once`` for their own key; ``paged`` lists every key asked."""

    def __init__(self):
        self.paged: list[str] = []
        self.keys: set[str] = set()

    def _notify(self, key):
        def notify(_data):
            self.paged.append(key)
            mod.resolve_gateway_approval(key, "once")
        return notify

    def register(self, key, *, session_scoped=True):
        """The messaging gateway registers both a turn and a session-scoped approver; the TUI only a turn one."""
        self.keys.add(key)
        mod.register_gateway_notify(key, self._notify(key))
        if session_scoped:
            session_notify.register_session_notify(key, self._notify(key))

    def unregister(self, key):
        mod.unregister_gateway_notify(key)
        session_notify.unregister_session_notify(key)


def _run(fn):
    box = {}
    thread = threading.Thread(target=lambda: box.update(value=fn()), daemon=True)
    thread.start()
    thread.join(WAIT_S)
    assert not thread.is_alive()
    return box["value"]


def _turn_context(*, key="", cron=False):
    def bind():
        if cron:
            set_session_vars(cron_session="1")
        if key:
            ctx.set_current_session_key(key)
    context = contextvars.Context()
    context.run(bind)
    return context


class _ReusedSdkSession:
    """One long-lived SDK session as the runtime drives it: the callback is built in the creating turn
    with the runtime's provider shape, every turn refreshes the snapshot on its own thread, and every
    request runs in a copy of the CREATING turn's context."""

    def __init__(self, provider=None, **creating_turn):
        self.snapshot = None
        self.inherited = _turn_context(**creating_turn)

        def create():
            self.snapshot = sdk_gateway.current_approval_turn_context()
            if provider is False:
                return sdk_gateway.build_sdk_gateway_approval_callback()
            return sdk_gateway.build_sdk_gateway_approval_callback(
                context_provider=provider or (lambda: self.snapshot or {}))

        self.callback = _run(lambda: self.inherited.copy().run(create))

    def turn(self, **turn):
        context = _turn_context(**turn)
        self.snapshot = _run(lambda: context.run(sdk_gateway.current_approval_turn_context))

    def request(self):
        return _run(lambda: self.inherited.copy().run(
            lambda: self.callback("true", "true", canonical_tool_input=CANONICAL)))


@pytest.fixture
def operators(monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
    monkeypatch.setattr(ctx, "_get_approval_config", lambda: {"timeout": WAIT_S, "mode": "manual"})
    monkeypatch.setattr(ctx, "_fire_approval_hook", lambda name, **kw: None)
    ops = _Operators()
    try:
        yield ops
    finally:
        for key in ops.keys:
            ops.unregister(key)
            with mod._lock:
                mod._gateway_queues.pop(key, None)


def test_reused_session_routes_to_the_current_turns_key(operators):
    operators.register("sess-k1")
    operators.register("sess-k2")
    session = _ReusedSdkSession(key="sess-k1")
    assert session.request() == "once"
    assert operators.paged == ["sess-k1"]

    operators.paged.clear()
    session.turn(key="sess-k2")
    assert session.request() == "once"
    assert operators.paged == ["sess-k2"]  # the creating turn's approvers receive nothing


def test_interactive_born_session_never_pages_from_a_cron_turn(operators):
    operators.register("sess-interactive")
    session = _ReusedSdkSession(key="sess-interactive")
    session.turn(key="sess-cron-job", cron=True)
    assert session.snapshot["gateway"] is False

    assert session.request() == NO_APPROVER
    assert operators.paged == []


def test_cron_born_session_pages_the_later_interactive_turn(operators):
    operators.register("sess-later")
    session = _ReusedSdkSession(key="sess-cron-job", cron=True)
    assert session.request() == NO_APPROVER
    assert operators.paged == []

    session.turn(key="sess-later")
    assert session.request() == "once"
    assert operators.paged == ["sess-later"]


def test_compression_rotation_routes_to_the_continuation_key(operators):
    """Manual /compress rotates ``agent.session_id`` without retiring the SDK session; the TUI moves its
    approver from the parent key to the continuation (``_sync_session_key_after_compress``)."""
    operators.register("sess-parent", session_scoped=False)
    session = _ReusedSdkSession(key="sess-parent")
    assert session.request() == "once"

    operators.paged.clear()
    operators.unregister("sess-parent")
    operators.register("sess-continuation", session_scoped=False)
    session.turn(key="sess-continuation")
    assert session.request() == "once"
    assert operators.paged == ["sess-continuation"]


def _raise():
    raise RuntimeError("provider failed")


@pytest.mark.parametrize("provided", [
    lambda: {},
    lambda: {"gateway": True},
    lambda: {"gateway": "yes", "session_key": "sess-inherited"},
    lambda: {"gateway": True, "session_key": "sess-inherited", **{f"extra_{i}": i for i in range(7)}},
    _raise,
], ids=["empty", "missing-key", "wrong-type", "oversized", "raises"])
def test_malformed_snapshot_fails_closed_despite_an_inherited_approver(operators, provided):
    operators.register("sess-inherited")
    session = _ReusedSdkSession(provider=provided, key="sess-inherited")
    assert session.request() == NO_APPROVER
    assert operators.paged == []


def test_provider_less_callers_keep_the_live_read(operators):
    operators.register("sess-live")
    session = _ReusedSdkSession(provider=False, key="sess-live")
    assert session.request() == "once"
    assert operators.paged == ["sess-live"]
