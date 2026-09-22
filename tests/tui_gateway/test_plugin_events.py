"""Native plugin events preserve exact-session authority across readiness and admission."""
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tui_gateway import plugin_events


def test_native_consumer_fences_stale_owner_and_recovers_same_physical_resume(tmp_path):
    record = {"session_key": "physical-1", "agent": SimpleNamespace(session_id="physical-1", api_mode="codex_responses", provider="openai-codex"),
              "history_lock": threading.RLock(), "history": [], "running": False,
              "profile_home": str(tmp_path), "transport": object()}
    server = SimpleNamespace(_sessions={"ui-1": record}, _sessions_lock=threading.RLock(),
                             _session_has_live_transport=lambda row: row.get("transport") is not None)
    manager = SimpleNamespace(home_path=tmp_path)
    first = plugin_events.DesktopPluginConsumer(manager, server)
    manager._desktop_consumer = first
    target = {"profile_name": "default", "session_id": "physical-1"}
    ticket = first.readiness(target)
    assert ticket["ready"] is True
    server._sessions = {"ui-2": dict(record, history_lock=threading.RLock())}
    second = plugin_events.DesktopPluginConsumer(manager, server)
    manager._desktop_consumer = second
    assert first.readiness(target)["ready"] is False
    assert second.readiness(target)["ready"] is True
    assert second.readiness(target)["generation"] != ticket["generation"]
    server._sessions["ui-2"]["transport"] = None
    assert second.readiness(target)["ready"] is False
    from gateway.internal_events import create_desktop_system_event
    content, event = create_desktop_system_event(content="Completed work", destination=target, event_id="queued-1",
        event_kind="external_tool_completed", plugin_id="example", eligibility_check=lambda: True)
    assert first.inject(content, event).result()["status"] == "cancelled"
    assert second.inject(content, event).result()["status"] == "stopping"


def test_native_readiness_rpc_contract_accepts_ready_and_refused_destinations(monkeypatch, tmp_path):
    from hermes_cli import plugins
    from tui_gateway import server
    from tui_gateway.contracts.registry import METHODS, check_params_accepted, check_result

    destination = {"profile_name": "default", "session_id": "physical-1"}
    record = {"agent": SimpleNamespace(session_id="physical-1", api_mode="codex_responses", provider="openai-codex"),
              "profile_home": str(tmp_path), "transport": object(), "running": False}
    runtime = SimpleNamespace(_sessions={"ui-1": record}, _sessions_lock=threading.RLock(),
                              _session_has_live_transport=lambda row: row.get("transport") is not None)
    manager = SimpleNamespace(home_path=tmp_path)
    consumer = plugin_events.DesktopPluginConsumer(manager, runtime)
    manager._desktop_consumer = consumer
    manager._desktop_contexts = {"example": SimpleNamespace(desktop_continuation_readiness=consumer.readiness)}
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    contract = METHODS["session.continuation.readiness"]
    for target, ready in [(destination, True), ({**destination, "session_id": "stale"}, False)]:
        params = {"plugin_id": "example", "destination": target}
        response = server._methods[contract.name]("probe", params)
        result = response["result"]
        assert result["ready"] is ready
        check_params_accepted(contract, params)
        check_result(contract, result)
    result = server._methods[contract.name]("missing", {})["result"]
    assert result == {"ready": False, "status": "host_unsupported"}
    check_params_accepted(contract, {})
    check_result(contract, result)


def test_native_consumer_rejects_busy_or_cancelled_without_model_admission(tmp_path):
    record = {"session_key": "physical-1", "agent": SimpleNamespace(session_id="physical-1", api_mode="codex_responses", provider="openai-codex"),
              "history_lock": threading.RLock(), "history": [], "running": True,
              "profile_home": str(tmp_path), "transport": object()}
    server = SimpleNamespace(_sessions={"ui-1": record}, _sessions_lock=threading.RLock(),
                             _session_has_live_transport=lambda row: True,
                             _run_prompt_submit=Mock())
    manager = SimpleNamespace(home_path=tmp_path)
    consumer = plugin_events.DesktopPluginConsumer(manager, server)
    manager._desktop_consumer = consumer
    from gateway.internal_events import create_desktop_system_event
    content, event = create_desktop_system_event(content="External task completed", plugin_id="example",
        event_id="event-1", event_kind="external_tool_completed",
        destination={"profile_name": "default", "session_id": "physical-1"}, eligibility_check=lambda: True)
    assert consumer.inject(content, event).result()["status"] == "busy"
    record["running"] = False
    content, event = create_desktop_system_event(content=content, plugin_id="example", event_id="event-2",
        event_kind="external_tool_completed", destination={"profile_name": "default", "session_id": "physical-1"},
        eligibility_check=lambda: False)
    assert consumer.inject(content, event).result()["status"] == "cancelled"
    server._run_prompt_submit.assert_not_called()


@pytest.mark.parametrize("agent_outcome,expected_status", [
    ({"completed": True}, "settled"),
    ({"completed": False, "failed": True,
      "gateway_system_event_error": "context_budget_exceeded"}, "failed"),
    ({"completed": False}, "failed"),
    ({"completed": True, "failed": True}, "failed"),
    ({}, "failed"),
])
def test_native_prompt_carries_typed_event_and_receipt_follows_transport(
    monkeypatch, tmp_path, agent_outcome, expected_status,
):
    from tui_gateway import server
    from gateway.internal_events import create_desktop_system_event, gateway_system_event_message
    order = []
    captured = []
    class InlineThread:
        def __init__(self, target=None, daemon=None, args=(), kwargs=None, **unused):
            self.target, self.args, self.kwargs = target, args, kwargs or {}
        def start(self):
            self.target(*self.args, **self.kwargs)
        def is_alive(self):
            return False
        def join(self, timeout=None):
            pass
    monkeypatch.setattr(server.threading, "Thread", InlineThread)
    for name, value in {
        "_wire_callbacks": lambda *a: None, "_sync_agent_model_with_config": lambda *a: None,
        "_session_cwd": lambda row: str(tmp_path), "_register_session_cwd": lambda row: None,
        "_tts_stream_begin": lambda: None, "_sync_session_key_after_compress": lambda *a, **k: None,
        "_get_usage": lambda agent: {}, "_ensure_active_session_slot": lambda *a: None,
        "_goal_followup_after_turn": lambda *a: None, "_after_complete_turn": lambda *a: None,
    }.items():
        monkeypatch.setattr(server, name, value)
    accepted_frame = [True]
    def emit(kind, sid, payload=None):
        order.append(kind)
        return accepted_frame[0]
    monkeypatch.setattr(server, "_emit", emit)
    db = Mock()
    def run(text, **kwargs):
        event = kwargs["gateway_system_event"]
        captured.append((text, event))
        succeeded = expected_status == "settled"
        messages = [gateway_system_event_message(event, text)]
        if succeeded:
            messages.append({"role": "assistant", "content": "Verified result"})
        return {"final_response": "Verified result" if succeeded else "", "messages": messages,
                **agent_outcome}
    agent = SimpleNamespace(session_id="physical-1", api_mode="codex_responses", provider="openai-codex",
                            clear_interrupt=lambda: None, run_conversation=run, _session_db=db)
    record = {"session_key": "physical-1", "agent": agent, "history_lock": threading.RLock(),
              "history": [], "history_version": 0, "running": True, "attached_images": [],
              "image_counter": 0, "cols": 80, "slash_worker": None, "show_reasoning": False,
              "tool_progress_mode": "all", "inflight_turn": None}
    content, event = create_desktop_system_event(content="External task finished @untrusted-reference", plugin_id="example",
        event_id="event-3", event_kind="external_tool_completed", destination={"profile_name": "default", "session_id": "physical-1"},
        eligibility_check=lambda: True)
    # Observe the first unlocked boundary after admission, before message.start.
    # A concurrent reconnect must never see internal input as a user prompt.
    from agent import notification_presentation
    from tui_gateway.session_auto_continue import _inflight_snapshot
    original_snapshot = notification_presentation.notification_config_snapshot
    def notification_snapshot():
        snapshot = _inflight_snapshot(record)
        assert snapshot is not None
        assert snapshot["user"] == ""
        assert snapshot["display_kind"] == "internal_notification"
        assert snapshot["streaming"] is True
        return original_snapshot()
    monkeypatch.setattr(notification_presentation, "notification_config_snapshot", notification_snapshot)
    results = []
    def terminal(result):
        order.append("receipt")
        results.append(result)
    assert server._run_prompt_submit("rid", "ui-1", record, content, image_paths=[], gateway_system_event=event, terminal_callback=terminal, terminal_frame_writer=lambda payload: emit("message.complete", "ui-1", payload))
    assert captured == [(content, event)]
    assert order.index("message.complete") < order.index("receipt")
    assert results[0]["status"] == expected_status
    if agent_outcome.get("gateway_system_event_error"):
        assert "context_budget_exceeded" in results[0]["error"]
    assert not any(m.get("role") == "user" for m in record["history"])
    db.set_latest_matching_message_display_kind.assert_not_called()
    accepted_frame[0] = False
    record["running"] = True
    assert server._run_prompt_submit("rid2", "ui-1", record, content, image_paths=[], gateway_system_event=event, terminal_callback=terminal, terminal_frame_writer=lambda payload: emit("message.complete", "ui-1", payload))
    assert results[-1]["status"] == "failed"


def test_registered_handler_in_tool_worker_keeps_running_native_caller(monkeypatch, tmp_path):
    import contextvars
    import json
    from concurrent.futures import ThreadPoolExecutor
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from tools.thread_context import propagate_context_to_thread
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("plugins:\n  entries:\n    native-test:\n      allow_gateway_injection: true\n")
    manager = PluginManager(scope_key=str(tmp_path))
    ctx = PluginContext(PluginManifest(name="native-test", key="native-test", source="user"), manager)
    record = {"session_key": "agent:main:telegram:dm:123", "agent": SimpleNamespace(session_id="physical-1", api_mode="codex_responses", provider="openai-codex"),
              "history_lock": threading.RLock(), "history": [], "running": True,
              "profile_home": str(tmp_path), "transport": object()}
    current = contextvars.ContextVar("native-test-record", default=None)
    server = SimpleNamespace(_sessions={"ui-1": record}, _sessions_lock=threading.RLock(),
        _session_has_live_transport=lambda row: True, _current_runtime_session_record=current)
    consumer = plugin_events.DesktopPluginConsumer(manager, server)
    manager._desktop_consumer = consumer
    ctx._desktop_observer_ready = True
    manager._desktop_tasks = {"native-test:observer": SimpleNamespace(done=lambda: False)}
    def handler(args, **kwargs):
        target = ctx.current_desktop_destination()
        return json.dumps({"destination": target, "readiness": ctx.desktop_continuation_readiness(target)})
    handle = ctx.register_tool(name="native_context_probe", toolset="native_test", schema={"name": "native_context_probe", "parameters": {"type": "object", "properties": {}}}, handler=handler)
    token = current.set(record)
    try:
        with plugin_events.desktop_plugin_turn(record), ThreadPoolExecutor(max_workers=1) as pool:
            result = json.loads(pool.submit(propagate_context_to_thread(lambda: ctx.dispatch_tool("native_context_probe", {}))).result())
        assert result["destination"] == {"profile_name": "default", "session_id": "physical-1"}
        assert result["readiness"]["ready"] is True
    finally:
        current.reset(token)
        handle.release()
    assert ctx.current_desktop_destination() is None


def test_terminal_ack_does_not_reauthorize_admission_and_cancel_refuses_completion(tmp_path):
    from gateway.internal_events import create_desktop_system_event
    record = {"session_key": "physical-1", "agent": SimpleNamespace(session_id="physical-1", api_mode="codex_responses", provider="openai-codex"),
              "history_lock": threading.RLock(), "history": [], "running": False,
              "profile_home": str(tmp_path), "transport": object()}
    server = SimpleNamespace(_sessions={"ui-1": record}, _sessions_lock=threading.RLock(),
                             _session_has_live_transport=lambda row: True)
    manager = SimpleNamespace(home_path=tmp_path)
    consumer = plugin_events.DesktopPluginConsumer(manager, server)
    manager._desktop_consumer = consumer
    for cancelled in (False, True):
        admission = [True]
        def run(*args, **kwargs):
            assert kwargs["gateway_system_event"].eligibility_check() is True
            admission[0] = False  # Signed in-turn acknowledgement retires admission.
            kwargs["terminal_callback"]({"status": "settled"})
            return True
        server._run_prompt_submit = run
        record["running"] = False
        content, event = create_desktop_system_event(content="Finished", destination={"profile_name": "default", "session_id": "physical-1"},
            event_id=f"ack-{cancelled}", event_kind="external_tool_completed", plugin_id="example",
            eligibility_check=lambda: admission[0], completion_check=lambda: not cancelled)
        assert consumer.inject(content, event).result()["status"] == ("cancelled" if cancelled else "completed")


def test_confirmed_terminal_waits_for_real_ordered_socket_send():
    import asyncio
    import json
    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    from tui_gateway.transport import FanoutTransport
    from tui_gateway.ws import WSTransport
    for fanout in (False, True):
        for fails in (False, True, None):
            loop = asyncio.new_event_loop()
            started = threading.Event()
            finished = threading.Event()
            release = asyncio.Event()
            delivered = []
            class Socket:
                async def send_text(self, payload):
                    started.set()
                    await release.wait()
                    if fails:
                        raise OSError("controlled socket failure")
                    delivered.append(json.loads(payload)["params"]["type"])
                    if delivered[-1] == "message.complete":
                        finished.set()
            thread = threading.Thread(target=loop.run_forever, daemon=True)
            thread.start()
            socket_transport = WSTransport(Socket(), loop)
            transport = FanoutTransport(socket_transport) if fanout else socket_transport
            delta = {"method": "event", "params": {"type": "message.delta", "session_id": "ui-1"}}
            final = {"method": "event", "params": {"type": "message.complete", "session_id": "ui-1"}}
            try:
                assert transport.write(delta)
                assert started.wait(5)
                with ThreadPoolExecutor(max_workers=1) as pool:
                    writer = transport.write_confirmed
                    pending = pool.submit(writer, final, **({"timeout": 0.05} if fails is None else {}))
                    if fails is None:
                        assert pending.result(timeout=5) is False
                    else:
                        try:
                            pending.result(timeout=2)
                        except TimeoutError:
                            pass
                        else:
                            raise AssertionError("terminal receipt acknowledged an unsent frame")
                    loop.call_soon_threadsafe(release.set)
                    assert pending.result(timeout=5) is (fails is False)
                    if fails is None:
                        assert finished.wait(5)  # Late delivery never changes the failed receipt.
                        assert pending.result() is False
                if not fails:
                    assert delivered == ["message.delta", "message.complete"]
            finally:
                loop.call_soon_threadsafe(release.set)
                loop.call_soon_threadsafe(loop.stop)
                thread.join(timeout=5)
                loop.close()


def test_stop_latch_blocks_busy_retry_and_generation_fences_admission(tmp_path):
    from gateway.internal_events import create_desktop_system_event
    record = {"session_key": "physical-1", "agent": SimpleNamespace(session_id="physical-1", api_mode="codex_responses", provider="openai-codex"),
              "history_lock": threading.RLock(), "history": [], "running": True,
              "profile_home": str(tmp_path), "transport": object()}
    server = SimpleNamespace(_sessions={"ui-1": record}, _sessions_lock=threading.RLock(),
                             _session_has_live_transport=lambda row: True, _run_prompt_submit=Mock())
    manager = SimpleNamespace(home_path=tmp_path)
    consumer = plugin_events.DesktopPluginConsumer(manager, server)
    manager._desktop_consumer = consumer
    content, event = create_desktop_system_event(content="Finished", destination={"profile_name": "default", "session_id": "physical-1"},
        event_id="stop-event", event_kind="external_tool_completed", plugin_id="example", eligibility_check=lambda: True)
    assert consumer.inject(content, event).result()["status"] == "busy"
    record.update(running=False, _turn_cancel_requested=True, _queued_prompt_generation=1)
    assert consumer.inject(content, event).result()["status"] == "cancelled"
    server._run_prompt_submit.assert_not_called()
    record["_turn_cancel_requested"] = False  # Only the ordinary user-submit path rearms it.
    def race(*args, **kwargs):
        assert kwargs["queued_prompt_generation"] == 1
        record["_queued_prompt_generation"] = 2
        assert kwargs["gateway_system_event"].eligibility_check() is False
        kwargs["terminal_callback"]({"status": "settled"})
        return True
    server._run_prompt_submit = race
    assert consumer.inject(content, event).result()["status"] == "cancelled"


def test_reopened_gateway_session_resolves_physical_desktop_destination(tmp_path):
    from contextvars import ContextVar
    row = {"session_key": "agent:main:telegram:dm:123", "agent": SimpleNamespace(session_id="physical-1", api_mode="codex_responses", provider="openai-codex"), "transport": object()}
    current = ContextVar("test_current", default=row)
    server = SimpleNamespace(_sessions={"ui-1": row}, _sessions_lock=threading.RLock(), _current_runtime_session_record=current, _session_has_live_transport=lambda row: row.get("transport") is not None)
    manager = SimpleNamespace(home_path=tmp_path)
    consumer = plugin_events.DesktopPluginConsumer(manager, server)
    manager._desktop_consumer = consumer
    import contextvars
    with plugin_events.desktop_plugin_turn(row):
        assert consumer.current_destination() == {"profile_name": "default", "session_id": "physical-1"}
        copied = contextvars.copy_context()
        server._sessions["duplicate"] = dict(row)
        assert consumer.current_destination() is None
        del server._sessions["duplicate"]
    assert copied.run(consumer.current_destination) is None
    with plugin_events.desktop_plugin_turn(row, internal=True):
        assert consumer.current_destination() is None
