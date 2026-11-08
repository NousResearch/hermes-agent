"""One voice-mode state for a shared-owner session (T9 review round 1).

Ink's ``/voice on|off`` runs the owner's per-session ``/voice`` handler and ``voice.record`` gates on
that mode. Every other writer must use the same state: ``/voice off`` stops the session's live
capture (and its late callbacks), a spoken stop phrase turns the owner mode off, and the wake path
can turn it on. Real handlers; only the microphone is fake.
"""

import asyncio
import sys
from types import SimpleNamespace

import pytest

from gateway import run_api
from gateway.config import Platform
from gateway.slash_commands import GatewaySlashCommandsMixin
from tui_gateway import server
from tui_gateway.transport import bind_transport, reset_transport


class _Mic:
    def __init__(self):
        self.active, self.stops, self.callbacks = False, 0, {}

    def start_continuous(self, **kw):
        if not self.active:  # the real wrapper keeps the live capture's callbacks: a no-op True
            self.active, self.callbacks = True, kw
        return True

    def is_continuous_active(self):
        return self.active

    def stop_continuous(self, **kw):
        self.active = False
        self.stops += 1

    def set_voice_busy_probe(self, *a):
        pass


class _Sink:
    def __init__(self):
        self.frames = []

    def write(self, obj):
        self.frames.append(obj)
        return True

    def close(self):
        pass


class _Runner:
    def __init__(self, sessions):
        self._voice_mode = {}
        self.session_authority = SimpleNamespace(sessions=sessions)

    def _voice_key_for_source(self, source):
        return source.chat_id

    def _delivery_adapter_for(self, source):
        return None

    def _save_voice_modes(self):
        pass

    def _get_guild_id(self, event):
        return None


@pytest.fixture
def owner(monkeypatch):
    mic, sink = _Mic(), _Sink()
    monkeypatch.setitem(sys.modules, "hermes_cli.voice", mic)
    monkeypatch.setattr(server, "_load_cfg", dict)
    monkeypatch.setattr(server, "_tts_lease_async", lambda *a: None)
    monkeypatch.setattr(server, "_stdio_transport", sink)
    monkeypatch.setattr(server, "_voice_event_sid", "")
    monkeypatch.setenv("HERMES_VOICE", "0")
    sources = {sid: SimpleNamespace(chat_id=sid, platform=Platform.LOCAL) for sid in ("owner", "other")}
    runner = _Runner({sid: SimpleNamespace(source=src) for sid, src in sources.items()})
    monkeypatch.setattr(run_api, "_ACTIVE_RUNNER", runner)

    def rpc(method, **params):
        token = bind_transport(sink)
        try:
            return server.handle_request({"id": method, "method": method, "params": params})
        finally:
            reset_transport(token)

    def slash(sid, arg):
        event = SimpleNamespace(source=sources[sid], get_command_args=lambda: arg)
        return asyncio.run(GatewaySlashCommandsMixin._handle_voice_command(runner, event))

    return SimpleNamespace(mic=mic, sink=sink, runner=runner, rpc=rpc, slash=slash)


def _transcripts(sink):
    return [f["params"]["payload"] for f in sink.frames if f.get("params", {}).get("type") == "voice.transcript"]


def test_voice_off_stops_the_sessions_capture_and_drops_late_text(owner):
    owner.slash("owner", "on")
    assert owner.rpc("voice.record", action="start", session_id="owner")["result"]["status"] == "recording"
    owner.slash("owner", "off")
    assert owner.runner._voice_mode["owner"] == "off"
    assert owner.mic.active is False and owner.mic.stops == 1
    owner.mic.callbacks["on_transcript"]("AFTER_VOICE_OFF")
    assert _transcripts(owner.sink) == []


def test_another_sessions_voice_off_leaves_this_capture_running(owner):
    owner.slash("owner", "on")
    owner.rpc("voice.record", action="start", session_id="owner")
    owner.slash("other", "off")
    assert owner.mic.active is True and owner.mic.stops == 0
    owner.mic.callbacks["on_transcript"]("still mine")
    assert [p["text"] for p in _transcripts(owner.sink)] == ["still mine"]


def test_spoken_stop_turns_the_owner_mode_off(owner):
    owner.slash("owner", "on")
    owner.rpc("voice.record", action="start", session_id="owner")
    owner.mic.callbacks["on_stop_phrase"]("stop")
    assert owner.runner._voice_mode["owner"] == "off"
    assert server._session_voice_enabled("owner") is False
    assert _transcripts(owner.sink)[-1]["stop_phrase"] is True


def test_no_speech_limit_turns_the_owner_mode_off(owner):
    owner.slash("owner", "on")
    owner.rpc("voice.record", action="start", session_id="owner")
    owner.mic.callbacks["on_silent_limit"]()
    assert owner.runner._voice_mode["owner"] == "off"
    assert _transcripts(owner.sink)[-1]["no_speech_limit"] is True



def test_a_second_sessions_start_is_refused_while_another_captures(owner):
    owner.slash("owner", "on")
    owner.slash("other", "on")
    owner.rpc("voice.record", action="start", session_id="owner")
    refused = owner.rpc("voice.record", action="start", session_id="other")
    assert refused["error"]["code"] == 4016 and "another session" in refused["error"]["message"]
    owner.slash("other", "off")
    assert owner.mic.active is True
    owner.mic.callbacks["on_transcript"]("A_TEXT")
    frames = [f["params"] for f in owner.sink.frames if f.get("params", {}).get("type") == "voice.transcript"]
    assert [f["session_id"] for f in frames] == ["owner"]
    owner.slash("owner", "off")
    assert owner.mic.active is False
    owner.mic.callbacks["on_transcript"]("A_AFTER_OFF")
    assert len(_transcripts(owner.sink)) == 1


def test_another_session_may_start_once_the_capture_ended(owner):
    owner.slash("owner", "on")
    owner.slash("other", "on")
    owner.rpc("voice.record", action="start", session_id="owner")
    owner.mic.active = False
    assert owner.rpc("voice.record", action="start", session_id="other")["result"]["status"] == "recording"
    assert server._voice_capture_sid == "other"


@pytest.fixture
def stop_phrase(monkeypatch):
    import types

    monkeypatch.setitem(sys.modules, "tools.voice_mode_transcript",
                        types.SimpleNamespace(is_voice_stop_phrase=lambda t: t.strip().lower() == "stop"))
    monkeypatch.setattr(server, "_tts_stream_stop", lambda user_barge=False: None)


def test_typed_stop_turns_the_owner_mode_off_not_the_sidecar_flag(owner, stop_phrase, monkeypatch):
    owner.slash("owner", "on")
    owner.rpc("voice.record", action="start", session_id="owner")
    monkeypatch.setenv("HERMES_VOICE", "1")  # a sidecar session's own voice chat
    reply = owner.rpc("prompt.submit", session_id="owner", text="stop")
    assert reply["result"] == {"voice_stopped": True}
    assert owner.runner._voice_mode["owner"] == "off" and owner.mic.active is False
    assert server._voice_mode_enabled() is True


def test_typed_stop_reads_the_owner_mode(owner, stop_phrase):
    owner.slash("owner", "on")  # HERMES_VOICE stays "0": the owner mode is the gate
    assert owner.rpc("prompt.submit", session_id="owner", text="stop")["result"] == {"voice_stopped": True}
    assert owner.runner._voice_mode["owner"] == "off"


def test_full_duplex_stop_turns_the_owner_mode_off_not_the_sidecar_flag(owner, stop_phrase, monkeypatch):
    owner.slash("owner", "on")
    owner.rpc("voice.record", action="start", session_id="owner")
    monkeypatch.setenv("HERMES_VOICE", "1")
    server._deliver_fd_transcript("stop")
    assert owner.runner._voice_mode["owner"] == "off"
    assert server._voice_mode_enabled() is True
    assert _transcripts(owner.sink)[-1]["stop_phrase"] is True
