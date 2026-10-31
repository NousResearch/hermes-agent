"""Opt-in handback wakes only the most recent screen-driving conversation, once."""

import asyncio
import contextvars
import json
from types import SimpleNamespace

import pytest

from gateway import run as gateway_run, session_context, wake
from hermes_constants import get_hermes_home
from tools.bot_desktop import lease, runtime
from tools.computer_use import tool


@pytest.fixture
def rig(monkeypatch):
    # The behavioral regressions also run on the base before the helper exists.
    try:
        from tools.bot_desktop import handback
    except ImportError:
        handback = None
    if handback is not None:
        monkeypatch.setattr(handback, "_drivers", {})
        monkeypatch.setattr(handback, "_generations", {}, raising=False)
        monkeypatch.setattr(handback, "_subscribed", False)
    monkeypatch.setattr(lease, "_listeners", [])
    monkeypatch.setattr(runtime, "ensure_started_for_tool", lambda: None)
    monkeypatch.setattr(runtime, "published_env", lambda: {"DISPLAY": ":37"})
    monkeypatch.setattr(tool, "_new_backend", lambda mode: tool._NoopBackend())
    tool.reset_backend_for_tests()
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    config = home / "config.yaml"
    config.write_text(json.dumps({"bot_desktop": {"resume_on_handback": True}}))
    clock = SimpleNamespace(now=100.0)
    if handback is not None:
        monkeypatch.setattr(handback, "monotonic", lambda: clock.now)
    runner = SimpleNamespace(_running=True, _gateway_loop=SimpleNamespace(is_running=lambda: True))
    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: runner)
    scheduled, calls, contexts = [], [], []
    ambient = contextvars.ContextVar("handback_test_ambient", default="empty")
    ambient.set("tool")

    def schedule(coro, loop, **kwargs):
        assert loop is runner._gateway_loop
        contexts.append(ambient.get())
        scheduled.append(coro)

    async def observe(runner, route, text, still_pending, *, producer):
        calls.append(SimpleNamespace(route=route, text=text, pending=still_pending(), producer=producer))

    monkeypatch.setattr("agent.async_utils.safe_schedule_threadsafe", schedule)
    monkeypatch.setattr(wake, "wake_route", observe)

    def bind(name="A", **kwargs):
        values = dict(platform="local", chat_id="chat-" + name, chat_type="dm",
            thread_id="thread-" + name, user_id="user-" + name, user_name="name-" + name,
            scope_id="scope-" + name, session_key="route-" + name, session_id="parent-" + name,
            cron_session="")
        session_context.set_session_vars(**(values | kwargs))

    async def drain():
        while scheduled:
            await scheduled.pop(0)

    bind()
    yield SimpleNamespace(handback=handback, config=config, runner=runner, clock=clock,
                          scheduled=scheduled, calls=calls, contexts=contexts, bind=bind,
                          drain=lambda: asyncio.run(drain()))
    for coro in scheduled:
        coro.close()
    session_context.clear_session_vars([])
    tool.reset_backend_for_tests()


def _act():
    assert json.loads(tool.handle_computer_use({"action": "list_apps"}))["apps"] == []


def _hand_back():
    lease.acquire("v")
    lease.release("v")


def test_admitted_computer_action_wakes_once_with_captured_route_and_empty_context(rig):
    _act()
    rig.bind("other")  # A later unrelated context cannot change the screen driver's route.
    lease.acquire("v")
    lease.release("stale-viewer")
    lease.release(unless_human=True)
    assert not rig.scheduled
    lease.release("v")
    lease.release("v")
    _hand_back()  # No intervening screen action: the record was already consumed.
    assert len(rig.scheduled) == 1
    assert rig.contexts == ["empty"]
    rig.drain()
    call, = rig.calls
    assert call.route == {"platform": "local", "chat_id": "chat-A", "chat_type": "dm",
                          "thread_id": "thread-A", "user_id": "user-A", "user_name": "name-A",
                          "scope_id": "scope-A", "session_key": "route-A", "parent_session_id": "parent-A"}
    assert call.producer == "bot-screen-handback" and call.pending
    assert call.text == (
        "[Hermes gateway notice, not typed by the user: the person handed the Bot Screen back to you. "
        "Re-capture the screen before acting. If your earlier request in this conversation is not "
        "finished, continue it without asking them to confirm and do not repeat steps already done; "
        "otherwise say briefly that you have control again.]"
    )


@pytest.mark.parametrize("gate", ["off", "unset", "no-action", "cron", "no-key", "no-runner", "stopped", "loop"])
def test_ineligible_actions_never_wake(rig, monkeypatch, gate):
    changes = {
        "off": lambda: rig.config.write_text(json.dumps({"bot_desktop": {"resume_on_handback": False}})),
        "unset": lambda: rig.config.write_text("{}"),
        "no-action": lambda: None,
        "cron": lambda: session_context.set_session_vars(session_key="route-A", cron_session="1"),
        "no-key": session_context.set_session_vars,
        "no-runner": lambda: monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: None),
        "stopped": lambda: setattr(rig.runner, "_running", False),
        "loop": lambda: setattr(rig.runner, "_gateway_loop", None),
    }
    changes[gate]()
    if gate != "no-action":
        _act()
    _hand_back()
    assert not rig.scheduled


def test_last_driver_wins(rig):
    _act()
    rig.bind("B")
    _act()
    _hand_back()
    assert len(rig.scheduled) == 1
    rig.drain()
    assert [call.route["session_key"] for call in rig.calls] == ["route-B"]


def test_consumed_newer_driver_still_supersedes_pending_wake(rig):
    _act()
    _hand_back()  # A's wake has not checked still_pending yet.
    rig.bind("B")
    _act()
    _hand_back()  # B's record is popped too before either wake runs.
    assert len(rig.scheduled) == 2
    rig.drain()
    assert [(call.route["session_key"], call.pending) for call in rig.calls] == [
        ("route-A", False), ("route-B", True)]


@pytest.mark.parametrize("platform,chat_type,eligible", [
    ("telegram", "group", False), ("discord", "channel", False),
    ("slack", "thread", False), ("telegram", "", False),
    ("telegram", "dm", True), ("discord", "dm", True), ("local", "", True),
])
def test_only_single_person_routes_record_and_wake(rig, platform, chat_type, eligible):
    rig.bind(platform=platform, chat_type=chat_type)
    _act()
    assert bool(rig.handback._drivers) is eligible
    _hand_back()
    assert len(rig.scheduled) == int(eligible)
    rig.drain()
    if eligible:
        assert rig.calls[0].pending


def test_disabling_after_recording_disarms_handback(rig):
    _act()
    rig.config.write_text(json.dumps({"bot_desktop": {"resume_on_handback": False}}))
    _hand_back()
    assert not rig.scheduled


@pytest.mark.parametrize("enabled", [True, False])
def test_listener_reads_record_profile_config_and_restores_ambient_home(rig, tmp_path, enabled):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    rig.config.write_text(json.dumps({"bot_desktop": {"resume_on_handback": enabled}}))
    # Record while enabled, then make its profile differ from the lease writer's context.
    if not enabled:
        rig.config.write_text(json.dumps({"bot_desktop": {"resume_on_handback": True}}))
    _act()
    rig.config.write_text(json.dumps({"bot_desktop": {"resume_on_handback": enabled}}))
    other = tmp_path / "other-profile"
    other.mkdir()
    (other / "config.yaml").write_text(json.dumps({"bot_desktop": {"resume_on_handback": not enabled}}))
    token = set_hermes_home_override(other)
    try:
        lease.acquire("v", profile_key=str(rig.config.parent))
        lease.release("v", profile_key=str(rig.config.parent))
        assert get_hermes_home() == other
    finally:
        reset_hermes_home_override(token)
    assert len(rig.scheduled) == int(enabled)
    rig.drain()


def test_background_side_scope_supersedes_conversation_driver(rig):
    _act()
    # session_tui_side._side_scope binds only task identity and workspace/profile data.
    session_context.set_session_vars(session_key="bg_fixture", session_id="bg_fixture")
    _act()
    assert not rig.handback._drivers
    _hand_back()
    rig.drain()
    assert not rig.calls


@pytest.mark.parametrize("driver", ["group", "cron"])
@pytest.mark.parametrize("pending", [False, True])
def test_ineligible_driver_supersedes_record_and_pending_wake(rig, driver, pending):
    _act()
    if pending:
        _hand_back()
    rig.bind("G", platform="telegram", chat_type="group" if driver == "group" else "dm",
             cron_session="1" if driver == "cron" else "")
    _act()
    _hand_back()
    rig.drain()
    assert [(call.route["session_key"], call.pending) for call in rig.calls] == (
        [("route-A", False)] if pending else [])


@pytest.mark.parametrize("chat_id,eligible", [("G123", False), ("C123", False), ("D123", True)])
def test_slack_dm_requires_a_one_to_one_channel(rig, chat_id, eligible):
    rig.bind(platform="slack", chat_type="dm", chat_id=chat_id)
    _act()
    _hand_back()
    rig.drain()
    assert len(rig.calls) == int(eligible)
    if eligible:
        assert rig.calls[0].pending


@pytest.mark.parametrize("state", ["forced-human", "refused-human", "pending"])
def test_display_stop_disarms_handback(rig, monkeypatch, state):
    from tui_gateway import server

    monkeypatch.setattr(runtime, "stop", lambda: True)
    _act()
    if state == "pending":
        _hand_back()
    else:
        lease.acquire("v")
    response = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "display.stop",
                                      "params": {"force": state == "forced-human"}})
    assert ("error" in response) is (state == "refused-human")
    lease.release("v")
    rig.drain()
    assert [(call.route["session_key"], call.pending) for call in rig.calls] == (
        [("route-A", False)] if state == "pending" else [])


@pytest.mark.parametrize("superseded", ["action", "human"])
def test_pending_wake_is_cancelled_when_control_is_used_again(rig, superseded):
    _act()
    _hand_back()
    if superseded == "action":
        _act()
    else:
        lease.acquire("new-viewer")
    rig.drain()
    assert len(rig.calls) == 1 and not rig.calls[0].pending


def test_old_driver_does_not_wake(rig):
    _act()
    rig.clock.now += 60 * 60 + 1
    _hand_back()
    assert not rig.scheduled


@pytest.mark.parametrize("local", [True, False])
def test_browser_fence_records_only_the_local_bot_desktop_driver(rig, local):
    from tools.browser_session import run_fenced

    info = {"session_name": "fixture", "features": {"local": local}}
    assert run_fenced(info, lambda: {"success": True}) == {"success": True}
    _hand_back()
    assert len(rig.scheduled) == int(local)
    rig.drain()
    if local:
        assert rig.calls[0].route["session_key"] == "route-A"


def test_listener_failure_never_escapes_the_lease_writer(rig, monkeypatch, caplog):
    def fail(*args, **kwargs):
        raise RuntimeError("fixture scheduling unavailable")

    _act()
    monkeypatch.setattr(wake, "wake_route", fail)
    with caplog.at_level("DEBUG", logger="tools.bot_desktop.handback"):
        _hand_back()
    assert lease.get().holder == lease.AGENT
    assert "handback" in caplog.text.lower()
