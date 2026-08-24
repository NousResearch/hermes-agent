from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path

import pytest

from tui_gateway.turn_recovery import TurnLimits
from tui_gateway.turn_recovery_runtime import TurnRecoveryRuntime


MOBILE_ID = "11111111-1111-4111-8111-111111111111"
CLIENT_TURN_ID = "22222222-2222-4222-8222-222222222222"


class FakeServer:
    def __init__(self) -> None:
        self._methods = {}
        self._sessions = {}
        self.calls: list[tuple[str, dict]] = []
        self.persisted_sessions: list[str] = []
        self.next_runtime = 1
        self._methods.update(
            {
                "session.create": self._create,
                "session.resume": self._resume,
                "prompt.submit": self._submit,
                "session.interrupt": self._interrupt,
            }
        )

    def _ensure_session_db_row(self, session):
        self.persisted_sessions.append(session["session_key"])

    @staticmethod
    def _ok(rid, result):
        return {"jsonrpc": "2.0", "id": rid, "result": result}

    @staticmethod
    def _err(rid, code, message, *, data=None):
        error = {"code": code, "message": message}
        if data is not None:
            error["data"] = data
        return {"jsonrpc": "2.0", "id": rid, "error": error}

    def _create(self, rid, params):
        self.calls.append(("session.create", dict(params)))
        runtime_id = f"runtime-{self.next_runtime}"
        stored_id = f"stored-{self.next_runtime}"
        self.next_runtime += 1
        self._sessions[runtime_id] = {
            "session_key": stored_id,
            "history_lock": threading.Lock(),
            "transport": object(),
        }
        return self._ok(
            rid,
            {"session_id": runtime_id, "stored_session_id": stored_id},
        )

    def _resume(self, rid, params):
        self.calls.append(("session.resume", dict(params)))
        stored_id = params["session_id"]
        runtime_id = f"runtime-{self.next_runtime}"
        self.next_runtime += 1
        self._sessions[runtime_id] = {
            "session_key": stored_id,
            "history_lock": threading.Lock(),
            "transport": object(),
        }
        return self._ok(
            rid,
            {"session_id": runtime_id, "stored_session_id": stored_id},
        )

    def _submit(self, rid, params):
        self.calls.append(("prompt.submit", dict(params)))
        return self._ok(rid, {"status": "streaming"})

    def _interrupt(self, rid, params):
        self.calls.append(("session.interrupt", dict(params)))
        return self._ok(rid, {"status": "interrupted"})


@pytest.fixture
def installed(tmp_path: Path):
    server = FakeServer()
    runtime = TurnRecoveryRuntime(
        server,
        bindings_path=tmp_path / "bindings.json",
        limits=TurnLimits(),
    )
    runtime.install()
    return server, runtime


def _result(response: dict) -> dict:
    assert "error" not in response, response
    return response["result"]


def _open(server: FakeServer) -> dict:
    return _result(
        server._methods["session.open"](
            "open-1", {"mobile_session_id": MOBILE_ID}
        )
    )


def _submit_v2(server: FakeServer, runtime_session_id: str) -> dict:
    return _result(
        server._methods["prompt.submit"](
            "submit-1",
            {
                "session_id": runtime_session_id,
                "version": 2,
                "client_turn_id": CLIENT_TURN_ID,
                "text": "hello",
            },
        )
    )


class TestSessionOpen:
    def test_creates_binding_with_exact_capability(self, installed):
        server, _runtime = installed

        result = _open(server)

        assert result["turn_recovery"] is True
        assert result["automatic_resubmit"] is False
        assert result["runtime_session_id"] == "runtime-1"
        assert result["stored_session_id"] == "stored-1"
        assert result["mobile_session_id"] == MOBILE_ID
        assert result["binding_version"] == 1
        assert server.persisted_sessions == ["stored-1"]
        assert result["capabilities"]["turn_recovery"]["version"] == 2
        assert result["capabilities"]["turn_recovery"]["methods"] == [
            "session.open",
            "turn.reconcile",
            "turn.interrupt",
        ]
        assert server._sessions["runtime-1"]["_turn_recovery_mobile_session_id"] == MOBILE_ID

    def test_reconnect_resumes_same_stored_session_and_increments_version(self, installed):
        server, _runtime = installed
        first = _open(server)

        second = _result(
            server._methods["session.open"](
                "open-2", {"mobile_session_id": MOBILE_ID}
            )
        )

        assert second["stored_session_id"] == first["stored_session_id"]
        assert second["runtime_session_id"] != first["runtime_session_id"]
        assert second["binding_version"] == 2
        assert server.calls[-1] == (
            "session.resume",
            {"session_id": first["stored_session_id"], "source": "android"},
        )

    def test_binding_survives_runtime_reconstruction(self, tmp_path: Path):
        path = tmp_path / "bindings.json"
        server1 = FakeServer()
        runtime1 = TurnRecoveryRuntime(server1, bindings_path=path)
        runtime1.install()
        first = _open(server1)

        server2 = FakeServer()
        runtime2 = TurnRecoveryRuntime(server2, bindings_path=path)
        runtime2.install()
        second = _open(server2)

        assert second["stored_session_id"] == first["stored_session_id"]
        assert second["binding_version"] == 2
        assert json.loads(path.read_text())[MOBILE_ID] == {
            "stored_session_id": "stored-1",
            "binding_version": 2,
        }

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "11111111-1111-4111-8111-11111111111A",
            "11111111111141118111111111111111",
            "not-a-uuid",
        ],
    )
    def test_rejects_noncanonical_mobile_uuid(self, installed, bad):
        server, _runtime = installed
        response = server._methods["session.open"](
            "bad-open", {"mobile_session_id": bad}
        )
        assert response["error"]["code"] == -32602
        assert server.calls == []

    def test_accepts_canonical_persisted_mobile_uuid_from_older_version(self, installed):
        server, _runtime = installed
        persisted_v1 = "11111111-1111-1111-8111-111111111111"

        result = _result(
            server._methods["session.open"](
                "open-v1", {"mobile_session_id": persisted_v1}
            )
        )

        assert result["mobile_session_id"] == persisted_v1


class TestPromptSubmitV2:
    def test_acknowledges_first_submit_and_calls_legacy_handler_once(self, installed):
        server, _runtime = installed
        opened = _open(server)

        ack = _submit_v2(server, opened["runtime_session_id"])

        assert ack["accepted"] is True
        assert ack["automatic_resubmit"] is False
        assert ack["client_turn_id"] == CLIENT_TURN_ID
        assert ack["status"] == "accepted"
        assert ack["last_seq"] == 0
        assert ack["created"] is True
        uuid.UUID(ack["turn_id"])
        assert [name for name, _ in server.calls].count("prompt.submit") == 1

    def test_duplicate_submit_returns_same_turn_without_second_execution(self, installed):
        server, _runtime = installed
        opened = _open(server)
        first = _submit_v2(server, opened["runtime_session_id"])

        second = _submit_v2(server, opened["runtime_session_id"])

        assert second["turn_id"] == first["turn_id"]
        assert second["created"] is False
        assert [name for name, _ in server.calls].count("prompt.submit") == 1

    def test_second_distinct_v2_turn_is_rejected_while_first_is_active(self, installed):
        server, _runtime = installed
        opened = _open(server)
        _submit_v2(server, opened["runtime_session_id"])
        other_client_turn_id = "33333333-3333-4333-8333-333333333333"

        response = server._methods["prompt.submit"](
            "submit-2",
            {
                "session_id": opened["runtime_session_id"],
                "version": 2,
                "client_turn_id": other_client_turn_id,
                "text": "do not queue over the active recovery turn",
            },
        )

        assert response["error"]["code"] == 4091
        assert response["error"]["data"]["reason"] == "turn_active"
        assert [name for name, _ in server.calls].count("prompt.submit") == 1

    def test_legacy_submit_is_untouched(self, installed):
        server, _runtime = installed
        opened = _open(server)

        response = server._methods["prompt.submit"](
            "legacy", {"session_id": opened["runtime_session_id"], "text": "old"}
        )

        assert _result(response) == {"status": "streaming"}

    def test_rejects_v2_submit_outside_open_binding(self, installed):
        server, _runtime = installed
        response = server._methods["prompt.submit"](
            "submit",
            {
                "session_id": "unknown",
                "version": 2,
                "client_turn_id": CLIENT_TURN_ID,
                "text": "hello",
            },
        )
        assert response["error"]["code"] == -32602


class TestRecoveryEventsAndMethods:
    def test_decorates_live_message_events_with_contiguous_journal_sequence(self, installed):
        server, runtime = installed
        opened = _open(server)
        ack = _submit_v2(server, opened["runtime_session_id"])
        sid = opened["runtime_session_id"]

        start = runtime.decorate_event("message.start", sid, None)
        delta = runtime.decorate_event(
            "message.delta", sid, {"text": "hel", "rendered": "ignored"}
        )
        complete = runtime.decorate_event(
            "message.complete",
            sid,
            {"text": "hello", "status": "complete", "usage": {"total": 1}},
        )

        assert start == {
            "type": "message.start",
            "session_id": sid,
            "turn_id": ack["turn_id"],
            "seq": 1,
            "message_id": start["message_id"],
            "payload": {},
        }
        assert delta["seq"] == 2
        assert delta["payload"] == {"text": "hel"}
        assert complete["seq"] == 3
        assert complete["payload"] == {"text": "hello", "status": "completed"}

    def test_reconcile_accepts_turn_id_and_returns_replay(self, installed):
        server, runtime = installed
        opened = _open(server)
        ack = _submit_v2(server, opened["runtime_session_id"])
        sid = opened["runtime_session_id"]
        runtime.decorate_event("message.start", sid, None)
        runtime.decorate_event("message.delta", sid, {"text": "x"})

        page = _result(
            server._methods["turn.reconcile"](
                "reconcile",
                {"session_id": sid, "turn_id": ack["turn_id"], "after_seq": 0},
            )
        )

        assert page["mode"] == "events"
        assert [event["seq"] for event in page["events"]] == [1, 2]
        assert page["turn_id"] == ack["turn_id"]

    def test_interrupt_stops_legacy_turn_and_closes_recovery_turn(self, installed):
        server, runtime = installed
        opened = _open(server)
        ack = _submit_v2(server, opened["runtime_session_id"])
        sid = opened["runtime_session_id"]
        runtime.decorate_event("message.start", sid, None)

        result = _result(
            server._methods["turn.interrupt"](
                "interrupt", {"session_id": sid, "turn_id": ack["turn_id"]}
            )
        )

        assert result["automatic_resubmit"] is False
        assert result["client_turn_id"] == CLIENT_TURN_ID
        assert result["turn_id"] == ack["turn_id"]
        assert result["status"] == "interrupted"
        assert result["last_seq"] == 2
        assert [name for name, _ in server.calls].count("session.interrupt") == 1

    def test_error_event_becomes_terminal_failed_message(self, installed):
        server, runtime = installed
        opened = _open(server)
        _submit_v2(server, opened["runtime_session_id"])

        event = runtime.decorate_event(
            "error", opened["runtime_session_id"], {"message": "boom"}
        )

        assert event["type"] == "message.complete"
        assert event["payload"] == {"text": "Error: boom", "status": "failed"}


class TestReadyAdvertisement:
    def test_ready_payload_contains_protocol_and_capability(self, installed):
        _server, runtime = installed
        payload = runtime.ready_payload({"skin": {"name": "x"}, "change_events": True})
        assert payload["skin"] == {"name": "x"}
        assert payload["change_events"] is True
        assert payload["protocol"] == {"name": "hermes-jsonrpc", "major": 2}
        assert payload["capabilities"]["turn_recovery"]["version"] == 2


class TestGatewayIntegration:
    def test_server_registers_all_recovery_methods(self):
        from tui_gateway import server

        assert {
            "session.open",
            "prompt.submit",
            "turn.reconcile",
            "turn.interrupt",
        } <= set(server._methods)
        assert server._turn_recovery_runtime.server is server

    def test_server_event_frame_delegates_to_recovery_runtime(self, monkeypatch):
        from tui_gateway import server

        monkeypatch.setattr(
            server._turn_recovery_runtime,
            "decorate_event",
            lambda event, sid, payload: {
                "type": event,
                "session_id": sid,
                "turn_id": "turn-1",
                "seq": 7,
                "message_id": "message-1",
                "payload": payload or {},
            },
        )

        frame = server._event_frame("message.delta", "sid", {"text": "x"})

        assert frame["params"]["turn_id"] == "turn-1"
        assert frame["params"]["seq"] == 7

    def test_stdio_and_websocket_ready_payloads_use_runtime_capability(self):
        from tui_gateway import entry, ws

        for payload in (
            entry.build_gateway_ready_payload({"name": "stdio"}),
            ws.build_gateway_ready_payload({"name": "websocket"}),
        ):
            assert payload["protocol"] == {"name": "hermes-jsonrpc", "major": 2}
            assert payload["capabilities"]["turn_recovery"]["version"] == 2
            assert payload["change_events"] is True
