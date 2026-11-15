"""The one full-duplex mic belongs to the latest voice turn (0b-4, #152308; seam found by @kvnloo).

The listener runs in its owner's context: capture settings, STT and stop phrases are that turn's
profile, its interjection interrupts only that turn and is delivered to that session. A newer
voice turn moves the mic; a take already tripped finishes under the owner it began with.
"""

import os
import sys
import threading
import time
import types

import pytest

from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
from tui_gateway import server


class _Mic:
    """``tools.voice_mode`` stand-in: one blocking listen per listener, tripped by the test."""

    def __init__(self, tmp_path):
        self.tmp_path, self.lock = tmp_path, threading.Lock()
        self.listens, self.trip, self.started = [], threading.Event(), threading.Event()
        self.deaf = threading.Event()  # set: the listen loop stops polling should_stop (mid-block)
        self.transcribe_gate, self.text = None, None
        self.transcripts, self.emitted = [], []

    def full_duplex_listen(self, should_stop, is_playing=None, on_trigger=None, multiplier=None, **_kw):
        record = {"home": get_hermes_home(), "mult": multiplier, "stopped": False}
        with self.lock:
            self.listens.append(record)
        self.started.set()
        while self.deaf.is_set() or not should_stop():
            if self.trip.wait(0.01) and on_trigger is not None:
                self.trip.clear()
                on_trigger("generation")
                wav = self.tmp_path / f"take-{len(self.listens)}.wav"
                wav.write_bytes(b"RIFF")
                return str(wav)
        record["stopped"] = True
        return None

    def transcribe_recording(self, path, model=None):
        if self.transcribe_gate is not None:
            self.transcribe_gate.wait(5)
        home = get_hermes_home()
        self.transcripts.append(home)
        return {"success": True, "transcript": self.text or f"said in {home.name}"}


@pytest.fixture
def mic(tmp_path, monkeypatch):
    fake = _Mic(tmp_path)
    monkeypatch.setitem(sys.modules, "tools.voice_mode", types.SimpleNamespace(
        full_duplex_listen=fake.full_duplex_listen, transcribe_recording=fake.transcribe_recording,
        is_audio_output_active=lambda: False, stop_playback=lambda: None))
    monkeypatch.setenv("HERMES_VOICE", "1")
    monkeypatch.setattr(server, "_fd_listener_active", False)
    monkeypatch.setattr(server, "_fd_owner", None, raising=False)
    monkeypatch.setattr(server, "_sessions", {})
    emitted = fake.emitted
    monkeypatch.setattr(server, "_emit", lambda event, sid, payload=None: emitted.append((event, sid, payload)))
    monkeypatch.setattr(server, "_voice_emit", lambda event, payload=None: emitted.append((event, "<ptt>", payload)))
    yield fake
    monkeypatch.setenv("HERMES_VOICE", "0")  # any live listener ends
    _wait(lambda: not server._fd_listener_active)


def _wait(pred, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


def _session(tmp_path, sid, profile):
    home = tmp_path / profile
    home.mkdir(exist_ok=True)
    interrupted = threading.Event()
    record = {"running": True, "agent": types.SimpleNamespace(interrupt=interrupted.set)}
    server._sessions[sid] = record
    return types.SimpleNamespace(sid=sid, home=home, record=record, interrupted=interrupted)


def _arm_as(turn):
    """Arm from *turn*'s worker the way ``_start_turn_voice`` runs: its record and home bound."""
    record_token = server._current_runtime_session_record.set(turn.record if turn else None)
    home_token = set_hermes_home_override(str(turn.home)) if turn else None
    try:
        server._arm_full_duplex_listener()
    finally:
        if home_token is not None:
            reset_hermes_home_override(home_token)
        server._current_runtime_session_record.reset(record_token)


def test_a_newer_voice_turn_takes_the_mic_and_its_interjection(mic, tmp_path):
    a, b = _session(tmp_path, "sid-a", "a"), _session(tmp_path, "sid-b", "b")
    _arm_as(a)
    assert _wait(lambda: len(mic.listens) == 1)
    _arm_as(b)  # A's turn is still running: the mic moves anyway
    assert _wait(lambda: len(mic.listens) == 2)
    assert mic.listens[0]["home"] == a.home and mic.listens[0]["stopped"]
    assert mic.listens[1]["home"] == b.home
    mic.trip.set()
    assert _wait(lambda: any(e[0] == "voice.transcript" for e in mic.emitted))
    assert mic.transcripts == [b.home]
    assert b.interrupted.is_set() and not a.interrupted.is_set()
    assert ("voice.interrupted", "sid-b", None) in mic.emitted
    assert ("voice.transcript", "sid-b", {"text": "said in b"}) in mic.emitted


def test_a_tripped_take_finishes_under_its_original_owner(mic, tmp_path):
    a, b = _session(tmp_path, "sid-a", "a"), _session(tmp_path, "sid-b", "b")
    mic.transcribe_gate = threading.Event()
    _arm_as(a)
    assert _wait(lambda: len(mic.listens) == 1)
    mic.trip.set()
    assert _wait(lambda: a.interrupted.is_set())
    _arm_as(b)  # B claims while A's take is still being transcribed
    assert len(mic.listens) == 1 and server._fd_owner["session"] is b.record  # one stream at a time
    mic.transcribe_gate.set()
    assert _wait(lambda: len(mic.listens) == 2)
    assert mic.transcripts == [a.home]
    assert ("voice.transcript", "sid-a", {"text": "said in a"}) in mic.emitted
    assert mic.listens[1]["home"] == b.home and not b.interrupted.is_set()


def test_the_same_owner_rearming_keeps_one_listener(mic, tmp_path):
    a = _session(tmp_path, "sid-a", "a")
    _arm_as(a)
    assert _wait(lambda: len(mic.listens) == 1)
    gen = server._fd_gen
    _arm_as(a)
    _arm_as(None)  # a sessionless arm (voice.tts RPC) joins the live owner
    assert server._fd_gen == gen  # no hand-off was requested, so the listener keeps running
    assert len(mic.listens) == 1 and not mic.listens[0]["stopped"]
    assert server._fd_owner["session"] is a.record


def test_the_owner_is_dropped_when_its_turn_ends_while_another_runs(mic, tmp_path):
    a, other = _session(tmp_path, "sid-a", "a"), _session(tmp_path, "sid-c", "c")
    _arm_as(a)
    assert _wait(lambda: len(mic.listens) == 1)
    a.record["running"] = False  # the owner's turn is done, no TTS pending; sid-c still runs
    assert _wait(lambda: not server._fd_listener_active)
    assert mic.listens[0]["stopped"] and server._fd_owner is None
    assert other.record["running"] and not other.interrupted.is_set()


def test_voice_off_cancels_a_pending_hand_off(mic, tmp_path, monkeypatch):
    a, b = _session(tmp_path, "sid-a", "a"), _session(tmp_path, "sid-b", "b")
    mic.transcribe_gate = threading.Event()
    _arm_as(a)
    assert _wait(lambda: len(mic.listens) == 1)
    mic.trip.set()
    assert _wait(lambda: a.interrupted.is_set())
    _arm_as(b)
    monkeypatch.setenv("HERMES_VOICE", "0")
    mic.transcribe_gate.set()
    assert _wait(lambda: not server._fd_listener_active)
    assert len(mic.listens) == 1 and server._fd_owner is None


def test_multiplexing_without_a_profile_scope_never_opens_the_mic(mic, tmp_path):
    from agent.secret_scope import reset_multiplex_context, set_multiplex_context
    a = _session(tmp_path, "sid-a", "a")
    token = set_multiplex_context(True)
    try:
        _arm_as(a)  # no secret scope bound: the take would read the launch env
    finally:
        reset_multiplex_context(token)
    assert _wait(lambda: not server._fd_listener_active)
    assert mic.listens == [] and server._fd_owner is None


def test_a_trip_after_the_mic_moved_is_dropped_not_given_to_the_old_owner(mic, tmp_path):
    a, b = _session(tmp_path, "sid-a", "a"), _session(tmp_path, "sid-b", "b")
    _arm_as(a)
    assert _wait(lambda: len(mic.listens) == 1)
    mic.deaf.set()  # A's listener is mid-block: it has not seen the hand-off yet
    _arm_as(b)
    mic.trip.set()  # speech lands in that block
    assert _wait(lambda: not mic.trip.is_set())
    mic.deaf.clear()
    assert _wait(lambda: len(mic.listens) == 2)
    assert mic.transcripts == [] and not a.interrupted.is_set()
    assert not any(e[0] in ("voice.interrupted", "voice.transcript") for e in mic.emitted)
    assert mic.listens[1]["home"] == b.home


def test_a_generation_phase_interjection_interrupts_its_turn_and_is_emitted(mic, tmp_path):
    """Moved from test_tui_gateway_server.py: speech during LLM generation interrupts the running
    turn through ``agent.interrupt()`` and the capture is emitted as ``voice.transcript``."""
    a = _session(tmp_path, "sid-a", "a")
    mic.text = "wait, try another way"
    _arm_as(a)
    assert _wait(lambda: len(mic.listens) == 1)
    mic.trip.set()
    assert _wait(lambda: any(e[0] == "voice.transcript" for e in mic.emitted))
    assert a.interrupted.is_set()
    assert ("voice.interrupted", "sid-a", None) in mic.emitted
    assert ("voice.transcript", "sid-a", {"text": "wait, try another way"}) in mic.emitted


def test_a_stop_phrase_mid_generation_ends_the_voice_chat(mic, tmp_path, monkeypatch):
    """Moved from test_tui_gateway_server.py: a bare stop phrase interrupts the turn AND ends the
    voice chat, emitted as the explicit stop_phrase signal."""
    a = _session(tmp_path, "sid-a", "a")
    mic.text = "stop"  # the default voice.stop_phrases
    monkeypatch.setitem(sys.modules, "hermes_cli.voice", types.SimpleNamespace(stop_continuous=lambda **_kw: None))
    _arm_as(a)
    assert _wait(lambda: len(mic.listens) == 1)
    mic.trip.set()
    assert _wait(lambda: any(e[0] == "voice.transcript" for e in mic.emitted))
    assert a.interrupted.is_set()
    assert ("voice.transcript", "sid-a", {"stop_phrase": True, "text": "stop"}) in mic.emitted
    assert os.environ.get("HERMES_VOICE") == "0"
