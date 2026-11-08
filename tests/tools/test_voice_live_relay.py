"""The Azure realtime voice-live relay: delegation translation and open-connection lifecycle.

``/api/audio/voice-live-stream`` is a server-side WebSocket relay: the browser never sees the
Azure URL or resource key. The browser opens the session with ``{"type": "session.start",
"history": [...]}``; the relay opens the vendor socket (preview URL + ``api-key`` header),
waits ``session.created``, sends the Azure session config (which now carries a
``delegate_to_hermes`` function tool), waits ``session.updated`` and only then emits a
synthetic ``session.started`` to the browser.

Model-driven delegation on Azure is function calling: the vendor's
``response.function_call_arguments.done`` is forwarded to the browser (the desktop turns its
``call_id`` into the existing ``onDelegation``/``prompt.submit`` flow), while the browser's
``session.commentary.append`` / ``session.thinking.append`` frames accumulate server-side per
delegation and ``session.commentary.complete`` produces ONE ``conversation.item.create``
``function_call_output`` + ``response.create``. ``think`` text is the fallback result when no
commentary arrived. ``session.close`` closes the vendor socket.

These tests pin the pure translation, the tracker, the open handshake lifecycle and the pump —
with a scripted fake vendor socket, no network.
"""

import json
import queue
import threading

import pytest

from tests.waiting import wait_until
from tools import voice_live
from tools.voice_live_relay import (
    RELAY_CLOSE,
    ClientEventResult,
    DelegationTracker,
    VoiceLiveRelay,
    redact_secrets,
    translate_client_event,
    translate_vendor_event,
)

AZURE_KEY = "azure-relay-test-key-0042"
AZURE_BASE = "https://relay-test.openai.azure.com"
DEPLOYMENT = "gpt-4o-realtime-preview"
WS_URL = (
    f"wss://relay-test.openai.azure.com/openai/realtime"
    f"?api-version={voice_live.AZURE_REALTIME_API_VERSION}&deployment={DEPLOYMENT}"
)


class FakeVendorWS:
    """Scripted stand-in for the send/recv/close wrapper ``_azure_ws_connect`` returns.

    ``recv`` blocks on a queue (event-driven, no sleeps): scripted events first, then
    ``TimeoutError`` polls once the script is exhausted, so the relay's poll loop idles.
    """

    def __init__(self, events=()):
        self._q: queue.Queue = queue.Queue()
        for event in events:
            self._q.put(event)
        self.sent = []
        self.closed = False

    def push(self, event):
        self._q.put(event)

    def send(self, message):
        if self.closed:
            raise ConnectionError("send on closed socket")
        self.sent.append(json.loads(message))

    def recv(self, timeout=None):
        try:
            event = self._q.get(timeout=timeout if timeout else 0.2)
        except queue.Empty:
            raise TimeoutError("poll")
        if isinstance(event, Exception):
            raise event
        return event if isinstance(event, str) else json.dumps(event)

    def close(self):
        self.closed = True

    def sent_types(self):
        return [m.get("type") for m in self.sent]


def _live_section(**overrides):
    section = {"api_key": AZURE_KEY, "base_url": AZURE_BASE, "model": DEPLOYMENT, "voice": "echo"}
    section.update(overrides)
    return section


def _install_fake_connect(monkeypatch, ws):
    captured = {}

    def fake_connect(url, api_key, timeout):
        captured.update(url=url, api_key=api_key, timeout=timeout)
        return ws

    monkeypatch.setattr(voice_live, "_azure_ws_connect", fake_connect)
    monkeypatch.setattr(voice_live, "_live_section", lambda voice=None: _live_section())
    monkeypatch.setattr(voice_live, "_resolve_credentials", lambda live: (AZURE_KEY, AZURE_BASE))
    return captured


class TestDelegateToolInAzureSessionConfig:
    def test_azure_config_carries_the_delegate_tool(self, monkeypatch):
        monkeypatch.setattr(voice_live, "_live_section", lambda voice=None: _live_section())
        config = voice_live.build_azure_session_config()
        tools = config["tools"]
        assert len(tools) == 1
        tool = tools[0]
        assert tool["type"] == "function"
        assert tool["name"] == voice_live.DELEGATE_TO_HERMES_TOOL_NAME == "delegate_to_hermes"
        assert tool["parameters"]["required"] == ["request"]
        assert "backend" in tool["description"].lower()

    def test_openai_session_config_has_no_tools(self, monkeypatch):
        """The OpenAI code path is unchanged: no function tools leak into /live/sessions."""
        monkeypatch.setattr(voice_live, "_live_section", lambda voice=None: {"voice": "willow"})
        config = voice_live.build_session_config()
        assert "tools" not in config
        assert config["delegation"] == {"type": "client"}


class TestOpenAzureRealtimeSession:
    def test_open_returns_live_socket_after_session_updated(self, monkeypatch):
        ws = FakeVendorWS([
            {"type": "session.created", "session": {"id": "sess_1"}},
            {"type": "session.updated", "session": {"id": "sess_1"}},
        ])
        captured = _install_fake_connect(monkeypatch, ws)
        history = [{"type": "message", "role": "user", "content": "hi"}]

        conn = voice_live.open_azure_realtime_session(history=history, timeout=7.0)

        assert conn.session_id == "sess_1"
        assert conn.url == WS_URL
        assert conn.api_key == AZURE_KEY
        assert conn.deployment == DEPLOYMENT
        assert conn.ws is ws
        assert captured["api_key"] == AZURE_KEY  # key travels via the header arg, not the URL
        assert AZURE_KEY not in captured["url"]
        # session.update carries only Realtime session fields; history is seeded as conversation items
        assert ws.sent_types() == ["session.update", "conversation.item.create"]
        session = ws.sent[0]["session"]
        assert session["tools"][0]["name"] == "delegate_to_hermes"
        assert session["input_audio_transcription"] == {"model": "whisper-1"}
        assert "input" not in session
        assert ws.sent[1] == {"type": "conversation.item.create", "item": history[0]}
        # the open helper leaves the socket OPEN for the relay pump
        assert ws.closed is False

    def test_create_azure_realtime_session_still_closes_and_keeps_contract(self, monkeypatch):
        ws = FakeVendorWS([
            {"type": "session.created", "session": {"id": "sess_9"}},
            {"type": "session.updated"},
        ])
        captured = _install_fake_connect(monkeypatch, ws)

        result = voice_live.create_azure_realtime_session("v=0 offer", timeout=3.0)

        assert result["session"] == {"id": "sess_9", "model": DEPLOYMENT}
        assert result["transport"]["type"] == "websocket"
        assert result["transport"]["url"] == captured["url"]
        assert AZURE_KEY not in json.dumps(result)
        assert ws.closed is True

    def test_error_before_created_raises_and_closes(self, monkeypatch):
        ws = FakeVendorWS([{"type": "error", "error": {"message": "invalid api-key"}}])
        _install_fake_connect(monkeypatch, ws)
        with pytest.raises(RuntimeError, match="invalid api-key"):
            voice_live.open_azure_realtime_session()
        assert ws.closed is True
        assert ws.sent == []

    def test_error_on_update_raises_and_closes(self, monkeypatch):
        ws = FakeVendorWS([
            {"type": "session.created", "session": {"id": "s"}},
            {"type": "error", "error": {"message": "bad voice"}},
        ])
        _install_fake_connect(monkeypatch, ws)
        with pytest.raises(RuntimeError, match="bad voice"):
            voice_live.open_azure_realtime_session()
        assert ws.sent_types() == ["session.update"]
        assert ws.closed is True

    def test_socket_failure_raises_and_closes(self, monkeypatch):
        ws = FakeVendorWS([TimeoutError("timed out")])
        _install_fake_connect(monkeypatch, ws)
        with pytest.raises(RuntimeError, match="socket closed"):
            voice_live.open_azure_realtime_session()
        assert ws.closed is True

    def test_missing_key_raises_before_connect(self, monkeypatch):
        monkeypatch.setattr(voice_live, "_live_section", lambda voice=None: _live_section(api_key=""))
        monkeypatch.setattr(voice_live, "_resolve_credentials", lambda live: ("", AZURE_BASE))
        monkeypatch.setattr(voice_live, "_azure_ws_connect", lambda *a, **k: pytest.fail("must not connect"))
        with pytest.raises(ValueError):
            voice_live.open_azure_realtime_session()

    def test_non_json_frames_are_skipped(self, monkeypatch):
        ws = FakeVendorWS([
            "not-json",
            {"type": "session.created", "session": {"id": "s2"}},
            {"type": "session.updated"},
        ])
        _install_fake_connect(monkeypatch, ws)
        conn = voice_live.open_azure_realtime_session()
        assert conn.session_id == "s2"
        assert ws.sent_types() == ["session.update"]

    def test_explicit_credentials_are_used_without_resolving(self, monkeypatch):
        """The relay passes the profile-scoped credentials it already resolved."""
        ws = FakeVendorWS([
            {"type": "session.created", "session": {"id": "s3"}},
            {"type": "session.updated"},
        ])
        _install_fake_connect(monkeypatch, ws)
        monkeypatch.setattr(
            voice_live, "_resolve_credentials",
            lambda live: pytest.fail("route already resolved credentials"))
        conn = voice_live.open_azure_realtime_session(
            api_key="explicit-key", base_url="https://other.openai.azure.com")
        assert conn.api_key == "explicit-key"
        assert conn.url.startswith("wss://other.openai.azure.com/openai/realtime?")


class TestDelegationTracker:
    def test_complete_emits_one_function_call_output_plus_response_create(self):
        tracker = DelegationTracker()
        tracker.note_function_call("call_1", "delegate_to_hermes", '{"request": "hi"}')
        tracker.append("call_1", "commentary", "Hello")
        tracker.append("call_1", "commentary", "world")

        text, events = tracker.complete("call_1")

        assert text == "Hello world"
        assert [e["type"] for e in events] == ["conversation.item.create", "response.create"]
        item = events[0]["item"]
        assert item["type"] == "function_call_output"
        assert item["call_id"] == "call_1"
        assert item["output"] == "Hello world"

    def test_think_is_the_fallback_when_no_commentary(self):
        tracker = DelegationTracker()
        tracker.append("call_2", "think", "checking…")
        tracker.append("call_2", "think", "still checking")
        text, events = tracker.complete("call_2")
        assert text == "checking… still checking"
        assert events[0]["item"]["output"] == text

    def test_commentary_wins_over_think(self):
        tracker = DelegationTracker()
        tracker.append("call_3", "think", "progress")
        tracker.append("call_3", "commentary", "the answer")
        text, _ = tracker.complete("call_3")
        assert text == "the answer"

    def test_complete_without_anything_still_answers_the_call(self):
        tracker = DelegationTracker()
        text, events = tracker.complete("call_never_seen")
        assert text == ""
        assert events[0]["item"]["call_id"] == "call_never_seen"
        assert events[1] == {"type": "response.create"}

    def test_delegations_are_independent_and_popped(self):
        tracker = DelegationTracker()
        tracker.append("a", "commentary", "A")
        tracker.append("b", "commentary", "B")
        assert tracker.complete("a")[0] == "A"
        assert tracker.complete("b")[0] == "B"
        # a completed delegation is gone: a late complete gets the empty fallback
        assert tracker.complete("a")[0] == ""


class TestTranslateClientEvent:
    def test_audio_frames_pass_through_to_the_vendor(self):
        tracker = DelegationTracker()
        frame = {"type": "input_audio_buffer.append", "audio": "AAAA"}
        result = translate_client_event(frame, tracker)
        assert isinstance(result, ClientEventResult)
        assert result.vendor == [frame]
        assert result.to_browser == []
        assert result.close is False

    def test_commentary_append_is_buffered_not_forwarded(self):
        tracker = DelegationTracker()
        tracker.note_function_call("call_1", "delegate_to_hermes", "{}")
        result = translate_client_event(
            {"type": "session.commentary.append", "delegation_id": "call_1", "content": "hi"}, tracker)
        assert result.vendor == [] and result.to_browser == [] and result.close is False
        assert tracker.complete("call_1")[0] == "hi"

    def test_thinking_append_is_buffered_as_fallback(self):
        tracker = DelegationTracker()
        result = translate_client_event(
            {"type": "session.thinking.append", "delegation_id": "call_9", "content": "working"}, tracker)
        assert result.vendor == []
        assert tracker.complete("call_9")[0] == "working"

    def test_commentary_complete_produces_the_single_output_pair(self):
        tracker = DelegationTracker()
        tracker.note_function_call("call_1", "delegate_to_hermes", "{}")
        translate_client_event(
            {"type": "session.commentary.append", "delegation_id": "call_1", "content": "done"}, tracker)
        result = translate_client_event(
            {"type": "session.commentary.complete", "delegation_id": "call_1"}, tracker)
        assert result.close is False and result.to_browser == []
        assert [e["type"] for e in result.vendor] == ["conversation.item.create", "response.create"]
        assert result.vendor[0]["item"]["call_id"] == "call_1"
        assert result.vendor[0]["item"]["output"] == "done"

    def test_session_close_closes_without_vendor_traffic(self):
        result = translate_client_event({"type": "session.close"}, DelegationTracker())
        assert result.close is True
        assert result.vendor == [] and result.to_browser == []

    def test_session_start_is_only_valid_as_the_first_frame(self):
        result = translate_client_event({"type": "session.start"}, DelegationTracker())
        assert result.close is False and result.vendor == []
        assert result.to_browser[0]["type"] == "error"

    def test_unknown_event_type_is_rejected_to_the_browser_only(self):
        result = translate_client_event({"type": "session.mindmeld"}, DelegationTracker())
        assert result.vendor == [] and result.close is False
        assert result.to_browser[0]["type"] == "error"
        assert "session.mindmeld" in result.to_browser[0]["message"]

    def test_instructions_append_is_swallowed_not_forwarded(self):
        """Azure Realtime has no session.instructions.append client event; forwarding it would
        earn a vendor error, so the relay drops it silently."""
        result = translate_client_event(
            {"type": "session.instructions.append", "delegation_id": None, "content": "Be terse."},
            DelegationTracker())
        assert result.vendor == [] and result.to_browser == [] and result.close is False

    def test_append_with_null_delegation_id_is_a_silent_noop(self):
        """think()/speak() legitimately send delegation_id null when no call is active."""
        tracker = DelegationTracker()
        for frame in (
            {"type": "session.thinking.append", "delegation_id": None, "content": "hmm"},
            {"type": "session.commentary.append", "content": "x"},
            {"type": "session.commentary.append", "delegation_id": "", "content": "x"},
        ):
            result = translate_client_event(frame, tracker)
            assert result.vendor == [] and result.to_browser == [] and result.close is False
        assert tracker.complete("anything")[0] == ""  # nothing was buffered

    @pytest.mark.parametrize("frame", [
        "not-a-dict",
        {"type": "session.commentary.append", "delegation_id": "c", "content": 5},
        {"type": "session.commentary.complete"},  # no delegation_id
    ])
    def test_malformed_frames_never_reach_the_vendor(self, frame):
        result = translate_client_event(frame, DelegationTracker())
        assert result.vendor == [] and result.close is False
        assert result.to_browser[0]["type"] == "error"


class TestTranslateVendorEvent:
    def test_session_created_is_consumed_not_forwarded(self):
        out = translate_vendor_event({"type": "session.created", "session": {"id": "s"}}, DelegationTracker())
        assert out is None

    def test_function_call_arguments_done_is_forwarded_and_tracked(self):
        tracker = DelegationTracker()
        event = {
            "type": "response.function_call_arguments.done",
            "call_id": "call_7", "name": "delegate_to_hermes", "arguments": '{"request":"hi"}',
        }
        out = translate_vendor_event(event, tracker)
        assert out == event
        # the tracker learned the call, so a later complete maps onto this call_id
        _text, events = tracker.complete("call_7")
        assert events[0]["item"]["call_id"] == "call_7"

    def test_audio_and_transcript_events_pass_through(self):
        tracker = DelegationTracker()
        for event in (
            {"type": "response.audio.delta", "delta": "AAAA"},
            {"type": "conversation.item.created", "item": {"type": "message"}},
            {"type": "error", "error": {"message": "soft failure"}},
        ):
            assert translate_vendor_event(event, tracker) == event

    def test_secrets_are_redacted_everywhere_in_the_event(self):
        event = {
            "type": "error",
            "error": {"message": f"auth failed for {AZURE_KEY} at {WS_URL}"},
            "detail": [f"url={WS_URL}", {"nested": AZURE_KEY}],
        }
        out = translate_vendor_event(event, DelegationTracker(), secrets=(AZURE_KEY, WS_URL))
        blob = json.dumps(out)
        assert AZURE_KEY not in blob
        assert WS_URL not in blob
        assert "[redacted]" in blob

    def test_non_dict_events_are_dropped(self):
        assert translate_vendor_event("garbage", DelegationTracker()) is None
        assert translate_vendor_event(None, DelegationTracker()) is None


class TestRedactSecrets:
    def test_scalars_lists_dicts_and_non_strings(self):
        assert redact_secrets("key=abc123", ("abc123",)) == "key=[redacted]"
        assert redact_secrets(["abc123", 5, None], ("abc123",)) == ["[redacted]", 5, None]
        assert redact_secrets({"a": {"b": "abc123"}}, ("abc123",)) == {"a": {"b": "[redacted]"}}
        assert redact_secrets(7, ("abc123",)) == 7
        assert redact_secrets("plain", ()) == "plain"
        assert redact_secrets("plain", ("",)) == "plain"  # empty secrets are ignored


class TestVoiceLiveRelayPump:
    def _start(self, ws, secrets=()):
        received = []
        done = threading.Event()

        def send_to_browser(event):
            received.append(event)
            if event.get("type") == "session.closed":
                done.set()

        relay = VoiceLiveRelay(ws, "sess_pump", send_to_browser=send_to_browser, secrets=secrets)
        client_q: queue.Queue = queue.Queue()
        stop = threading.Event()
        thread = threading.Thread(target=relay.run, args=(client_q, stop), daemon=True)
        thread.start()
        return relay, client_q, stop, thread, received, done

    def test_vendor_events_pump_to_browser_and_complete_round_trips(self):
        ws = FakeVendorWS([
            {"type": "response.function_call_arguments.done",
             "call_id": "call_1", "name": "delegate_to_hermes", "arguments": "{}"},
            {"type": "response.audio.delta", "delta": "AAAA"},
        ])
        _relay, client_q, stop, thread, received, done = self._start(ws)
        try:
            wait_until(lambda: any(e.get("call_id") == "call_1" for e in received),
                       what="function_call_arguments.done forwarded")
            wait_until(lambda: any(e.get("type") == "response.audio.delta" for e in received),
                       what="audio delta forwarded")
            assert not any(e.get("type") == "session.created" for e in received)

            client_q.put({"type": "session.commentary.append",
                          "delegation_id": "call_1", "content": "the answer"})
            client_q.put({"type": "session.commentary.complete", "delegation_id": "call_1"})
            wait_until(lambda: ws.sent_types() == ["conversation.item.create", "response.create"],
                       what="function_call_output + response.create sent to vendor")
            item = ws.sent[0]["item"]
            assert item["type"] == "function_call_output"
            assert item["call_id"] == "call_1"
            assert item["output"] == "the answer"
        finally:
            stop.set()
            client_q.put(RELAY_CLOSE)
            thread.join(timeout=10)
        assert done.is_set()
        assert received[-1]["type"] == "session.closed"
        assert ws.closed is True

    def test_client_session_close_closes_vendor_and_reports(self):
        ws = FakeVendorWS()
        _relay, client_q, _stop, thread, received, _done = self._start(ws)
        client_q.put({"type": "session.close"})
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert ws.closed is True
        assert received[-1]["type"] == "session.closed"
        assert received[-1]["reason"] == "client_close"
        assert ws.sent == []

    def test_vendor_socket_death_ends_the_relay_with_session_closed(self):
        ws = FakeVendorWS([ConnectionError("vendor went away")])
        _relay, _client_q, _stop, thread, received, _done = self._start(ws)
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert ws.closed is True
        assert received[-1]["type"] == "session.closed"
        assert received[-1]["reason"]

    def test_secrets_never_reach_the_browser_sink(self, caplog):
        import logging

        ws = FakeVendorWS([
            {"type": "error", "error": {"message": f"leak {AZURE_KEY} {WS_URL}"}},
        ])
        with caplog.at_level(logging.DEBUG, logger="tools.voice_live_relay"):
            _relay, _client_q, stop, thread, received, _done = self._start(ws, secrets=(AZURE_KEY, WS_URL))
            try:
                wait_until(lambda: any(e.get("type") == "error" for e in received),
                           what="error event forwarded")
            finally:
                stop.set()
                thread.join(timeout=10)
        blob = json.dumps(received)
        assert AZURE_KEY not in blob
        assert WS_URL not in blob
        assert AZURE_KEY not in caplog.text

    def test_close_is_idempotent(self):
        ws = FakeVendorWS()
        relay = VoiceLiveRelay(ws, "s", send_to_browser=lambda e: None)
        relay.close()
        relay.close()
        assert ws.closed is True
