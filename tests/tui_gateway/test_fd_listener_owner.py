"""The one full-duplex mic belongs to the latest voice turn (0b-4, #152308; seam found by @kvnloo).

The listener runs in its owner's context: capture settings, STT and stop phrases are that turn's
profile, its interjection interrupts only that turn and is delivered to that session. A newer
voice turn moves the mic; a take already tripped finishes under the owner it began with.
"""

import os
import sys
import threading
import types

import pytest

from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
from tests.waiting import wait_until
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
        record = {"home": get_hermes_home(), "mult": multiplier, "stopped": False,
                  "is_playing": is_playing() if is_playing is not None else None}
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
    monkeypatch.setattr(server, "_fd_closed", False, raising=False)
    monkeypatch.setattr(server, "_sessions", {})
    emitted = fake.emitted
    monkeypatch.setattr(server, "_emit", lambda event, sid, payload=None: emitted.append((event, sid, payload)))
    monkeypatch.setattr(server, "_voice_emit", lambda event, payload=None: emitted.append((event, "<ptt>", payload)))
    yield fake
    monkeypatch.setenv("HERMES_VOICE", "0")  # any live listener ends
    wait_until(lambda: not server._fd_listener_active, what="the listener to end at teardown")


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
    wait_until(lambda: len(mic.listens) == 1)
    _arm_as(b)  # A's turn is still running: the mic moves anyway
    wait_until(lambda: len(mic.listens) == 2)
    assert mic.listens[0]["home"] == a.home and mic.listens[0]["stopped"]
    assert mic.listens[1]["home"] == b.home
    mic.trip.set()
    wait_until(lambda: any(e[0] == "voice.transcript" for e in mic.emitted))
    assert mic.transcripts == [b.home]
    assert b.interrupted.is_set() and not a.interrupted.is_set()
    assert ("voice.interrupted", "sid-b", None) in mic.emitted
    assert ("voice.transcript", "sid-b", {"text": "said in b"}) in mic.emitted


def test_a_tripped_take_finishes_under_its_original_owner(mic, tmp_path):
    a, b = _session(tmp_path, "sid-a", "a"), _session(tmp_path, "sid-b", "b")
    mic.transcribe_gate = threading.Event()
    _arm_as(a)
    wait_until(lambda: len(mic.listens) == 1)
    mic.trip.set()
    wait_until(lambda: a.interrupted.is_set())
    _arm_as(b)  # B claims while A's take is still being transcribed
    assert len(mic.listens) == 1 and server._fd_owner["session"] is b.record  # one stream at a time
    mic.transcribe_gate.set()
    wait_until(lambda: len(mic.listens) == 2)
    assert mic.transcripts == [a.home]
    assert ("voice.transcript", "sid-a", {"text": "said in a"}) in mic.emitted
    assert mic.listens[1]["home"] == b.home and not b.interrupted.is_set()


def test_the_same_owner_rearming_keeps_one_listener(mic, tmp_path):
    a = _session(tmp_path, "sid-a", "a")
    _arm_as(a)
    wait_until(lambda: len(mic.listens) == 1)
    gen = server._fd_gen
    _arm_as(a)
    _arm_as(None)  # a sessionless arm (voice.tts RPC) joins the live owner
    assert server._fd_gen == gen  # no hand-off was requested, so the listener keeps running
    assert len(mic.listens) == 1 and not mic.listens[0]["stopped"]
    assert server._fd_owner["session"] is a.record


def test_the_owner_is_dropped_when_its_turn_ends_while_another_runs(mic, tmp_path):
    a, other = _session(tmp_path, "sid-a", "a"), _session(tmp_path, "sid-c", "c")
    _arm_as(a)
    wait_until(lambda: len(mic.listens) == 1)
    a.record["running"] = False  # the owner's turn is done, no TTS pending; sid-c still runs
    wait_until(lambda: not server._fd_listener_active)
    assert mic.listens[0]["stopped"] and server._fd_owner is None
    assert other.record["running"] and not other.interrupted.is_set()


def test_voice_off_cancels_a_pending_hand_off(mic, tmp_path, monkeypatch):
    a, b = _session(tmp_path, "sid-a", "a"), _session(tmp_path, "sid-b", "b")
    mic.transcribe_gate = threading.Event()
    _arm_as(a)
    wait_until(lambda: len(mic.listens) == 1)
    mic.trip.set()
    wait_until(lambda: a.interrupted.is_set())
    _arm_as(b)
    monkeypatch.setenv("HERMES_VOICE", "0")
    mic.transcribe_gate.set()
    wait_until(lambda: not server._fd_listener_active)
    assert len(mic.listens) == 1 and server._fd_owner is None


def test_shutdown_cancels_a_pending_hand_off(mic, tmp_path, monkeypatch):
    """Client gone (stdin EOF or a signal reaches ``_shutdown_sessions``) while B waits for A's take:
    A's take still finishes as A, and no listener opens for B."""
    a, b = _session(tmp_path, "sid-a", "a"), _session(tmp_path, "sid-b", "b")
    mic.transcribe_gate = threading.Event()
    _arm_as(a)
    wait_until(lambda: len(mic.listens) == 1)
    mic.trip.set()
    wait_until(lambda: a.interrupted.is_set())
    _arm_as(b)  # the hand-off waits behind A's in-flight take
    _stub_shutdown_steps(monkeypatch)
    server._shutdown_sessions()
    mic.transcribe_gate.set()
    wait_until(lambda: len(mic.listens) == 2 or not server._fd_listener_active)
    assert len(mic.listens) == 1, "a listener opened for B after shutdown"
    assert not server._fd_listener_active and server._fd_owner is None
    assert mic.transcripts == [a.home]
    assert ("voice.transcript", "sid-a", {"text": "said in a"}) in mic.emitted


def _stub_shutdown_steps(monkeypatch, flush=None):
    """``_shutdown_sessions`` runs for real; only its other steps are stubbed (*flush* replaces the flush)."""
    monkeypatch.setattr(server, "_flush_sessions_before_exit", flush or (lambda *_a, **_k: None))
    for step in ("_release_gateway_wake_owner", "_stop_turns_before_exit"):
        monkeypatch.setattr(server, step, lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_close_session_by_id", lambda sid, **kw: None)


@pytest.mark.parametrize("handoff", [False, True], ids=["arm-spawn-gap", "handoff-spawn-gap"])
def test_shutdown_after_owner_selection_does_not_start_a_listener(mic, tmp_path, monkeypatch, handoff):
    """Shutdown lands after an owner was chosen (a first arm, or a listener exit picking the pending
    owner) but before its thread starts: the mic is not opened for it (the real ``full_duplex_listen``
    opens the input stream before its first stop check)."""
    a, b = _session(tmp_path, "sid-a", "a"), _session(tmp_path, "sid-b", "b")
    if handoff:
        _arm_as(a)
        wait_until(lambda: len(mic.listens) == 1)
    prior = len(mic.listens)
    chosen, release_spawn = threading.Event(), threading.Event()
    flushing, release_flush = threading.Event(), threading.Event()
    real_spawn, real_listener = server._fd_spawn_listener, server._full_duplex_listener
    selected_returned = threading.Event()

    def paused_spawn(owner, gen):
        chosen.set()
        assert release_spawn.wait(10)
        return real_spawn(owner, gen)
    monkeypatch.setattr(server, "_fd_spawn_listener", paused_spawn)

    def observed_listener(gen, owner):  # the selected owner's listener thread: completion handshake
        try:
            real_listener(gen, owner)
        finally:
            selected_returned.set()
    monkeypatch.setattr(server, "_full_duplex_listener", observed_listener)

    def flush(*_a, **_k):
        flushing.set()
        release_flush.wait(10)
    _stub_shutdown_steps(monkeypatch, flush)
    arm = threading.Thread(target=_arm_as, args=(b if handoff else a,), daemon=True)
    shutdown = threading.Thread(target=server._shutdown_sessions, daemon=True)
    try:
        arm.start()
        assert chosen.wait(10), "spawn boundary not reached"
        shutdown.start()
        assert flushing.wait(10), "real shutdown did not reach post-release flush"
        assert server._fd_owner is None
        for turn in (a, b):  # so a listener that did open can still finish
            turn.record["running"] = False
        release_spawn.set()
        assert selected_returned.wait(10), "the selected owner's listener thread never finished"
        assert len(mic.listens) == prior, "a listener starts after shutdown released the selected owner"
    finally:
        for turn in (a, b):
            turn.record["running"] = False
        release_spawn.set()
        release_flush.set()
        arm.join(10)
        if shutdown.ident is not None:
            shutdown.join(10)


def test_shutdown_stops_a_live_listener_while_its_turn_runs(mic, tmp_path, monkeypatch):
    """A listener already listening (or whose stream opened just as shutdown began) stops at its next
    block, though its owner's turn still runs."""
    a = _session(tmp_path, "sid-a", "a")
    _arm_as(a)
    wait_until(lambda: len(mic.listens) == 1)
    _stub_shutdown_steps(monkeypatch)
    server._shutdown_sessions()
    wait_until(lambda: mic.listens[0]["stopped"], what="the live listener to stop (it kept the mic open after shutdown)")
    assert a.record["running"]
    wait_until(lambda: not server._fd_listener_active, what="the mic released")


def test_an_arm_after_shutdown_never_claims_the_mic(mic, tmp_path, monkeypatch):
    """A voice turn still starting when the process shuts down cannot republish an owner."""
    a = _session(tmp_path, "sid-a", "a")
    spawned = []
    monkeypatch.setattr(server, "_fd_spawn_listener", lambda owner, gen: spawned.append(owner["sid"]))
    _stub_shutdown_steps(monkeypatch)
    server._shutdown_sessions()
    _arm_as(a)
    assert spawned == [] and server._fd_owner is None, "an arm after shutdown claimed the mic"
    assert not server._fd_listener_active and mic.listens == []


def test_a_record_removed_with_its_saved_sid_keeps_its_owner_target(mic, tmp_path):
    """A turn arming while its session is being closed: the record is out of the registry with its sid
    stamped (``_sid``); the interjection still goes to that sid, not the push-to-talk target."""
    a = _session(tmp_path, "sid-a", "a")
    del server._sessions["sid-a"]  # what session_lifecycle does on close: out of the registry...
    a.record["_sid"] = "sid-a"  # ...with the live sid stamped on the record
    _arm_as(a)
    wait_until(lambda: len(mic.listens) == 1)
    mic.trip.set()
    wait_until(lambda: any(e[0] == "voice.interrupted" for e in mic.emitted))
    assert ("voice.interrupted", "sid-a", None) in mic.emitted, "a removed record fell back to the push-to-talk target"


def test_multiplexing_without_a_profile_scope_never_opens_the_mic(mic, tmp_path):
    from agent.secret_scope import reset_multiplex_context, set_multiplex_context
    a = _session(tmp_path, "sid-a", "a")
    token = set_multiplex_context(True)
    try:
        _arm_as(a)  # no secret scope bound: the take would read the launch env
    finally:
        reset_multiplex_context(token)
    wait_until(lambda: not server._fd_listener_active)
    assert mic.listens == [] and server._fd_owner is None


def test_a_trip_after_the_mic_moved_is_dropped_not_given_to_the_old_owner(mic, tmp_path):
    a, b = _session(tmp_path, "sid-a", "a"), _session(tmp_path, "sid-b", "b")
    _arm_as(a)
    wait_until(lambda: len(mic.listens) == 1)
    mic.deaf.set()  # A's listener is mid-block: it has not seen the hand-off yet
    _arm_as(b)
    mic.trip.set()  # speech lands in that block
    wait_until(lambda: not mic.trip.is_set())
    mic.deaf.clear()
    wait_until(lambda: len(mic.listens) == 2)
    assert mic.transcripts == [] and not a.interrupted.is_set()
    assert not any(e[0] in ("voice.interrupted", "voice.transcript") for e in mic.emitted)
    assert mic.listens[1]["home"] == b.home


def test_a_generation_phase_interjection_interrupts_its_turn_and_is_emitted(mic, tmp_path):
    """Moved from test_tui_gateway_server.py: speech during LLM generation interrupts the running
    turn through ``agent.interrupt()`` and the capture is emitted as ``voice.transcript``."""
    a = _session(tmp_path, "sid-a", "a")
    mic.text = "wait, try another way"
    _arm_as(a)
    wait_until(lambda: len(mic.listens) == 1)
    mic.trip.set()
    wait_until(lambda: any(e[0] == "voice.transcript" for e in mic.emitted))
    assert a.interrupted.is_set()
    assert ("voice.interrupted", "sid-a", None) in mic.emitted
    assert ("voice.transcript", "sid-a", {"text": "wait, try another way"}) in mic.emitted
    assert mic.listens[0]["is_playing"] is False  # generation phase: nothing is playing
    wait_until(lambda: not list(tmp_path.glob("take-*.wav")), what="the capture deleted after delivery")


def test_a_stop_phrase_mid_generation_ends_the_voice_chat(mic, tmp_path, monkeypatch):
    """Moved from test_tui_gateway_server.py: a bare stop phrase interrupts the turn AND ends the
    voice chat, emitted as the explicit stop_phrase signal."""
    a = _session(tmp_path, "sid-a", "a")
    mic.text = "stop"  # the default voice.stop_phrases
    monkeypatch.setitem(sys.modules, "hermes_cli.voice", types.SimpleNamespace(stop_continuous=lambda **_kw: None))
    _arm_as(a)
    wait_until(lambda: len(mic.listens) == 1)
    mic.trip.set()
    wait_until(lambda: any(e[0] == "voice.transcript" for e in mic.emitted))
    assert a.interrupted.is_set()
    assert ("voice.transcript", "sid-a", {"stop_phrase": True, "text": "stop"}) in mic.emitted
    assert os.environ.get("HERMES_VOICE") == "0"
