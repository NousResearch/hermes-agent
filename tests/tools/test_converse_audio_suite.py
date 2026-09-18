"""Server-side proof that the converse voice pipeline hears speech — including QUIET speech.

Everything here drives the REAL VAD, endpointer and WAV-capture path (only STT is mocked),
using audio whose block-RMS envelope mimics real speech across a range of volumes. This is
the regression wall for "I'm talking and the server just reports idle": it fails if the
detector stops hearing a soft voice, and it fails if it starts hearing silence/room noise.
"""
import time

import numpy as np
import pytest

from tests.support.converse_audio import (
    RMS_AT_ROOM_FLOOR, RMS_LOUD_SPEECH, RMS_NORMAL_SPEECH, RMS_QUIET_SPEECH,
    collect_session_events, heard, mocked_stt, room_noise, run_utterances, silence, speech_like,
)


class TestConverseHearsSpeechAcrossVolume:
    """Volume matrix through the real ConverseSession (VAD + endpoint + capture; STT mocked)."""

    @pytest.mark.parametrize("rms", [RMS_QUIET_SPEECH, 700, 1000, RMS_NORMAL_SPEECH, RMS_LOUD_SPEECH])
    def test_speech_is_heard(self, rms):
        # A soft voice (500 RMS) through a loud one must all produce a transcript. This is the
        # core "quiet speech works" guarantee — the 0.6 sustained-window + adaptive floor.
        assert heard(speech_like(rms, seed=rms)), f"speech at {rms} RMS was not heard"

    def test_quiet_speech_across_several_takes(self):
        # Robustness to the random envelope: several independent soft-voice takes are all heard.
        for seed in range(5):
            assert heard(speech_like(RMS_QUIET_SPEECH, seed=100 + seed)), f"soft take {seed} missed"


class TestConverseRejectsNonSpeech:
    """The other half of a real VAD: it must NOT fire on silence or steady room noise."""

    def test_silence_is_not_heard(self):
        assert not heard(silence(), timeout=2.5)

    @pytest.mark.parametrize("rms", [120, 200, 250])
    def test_room_noise_is_not_heard(self, rms):
        assert not heard(room_noise(rms, seed=rms), timeout=2.5), f"room noise {rms} false-tripped"

    def test_signal_at_the_room_floor_is_not_heard(self):
        # ~300 RMS "speech" sits in the room tone — indistinguishable, must not trip (avoids
        # spurious turns). A real client normalizes a soft voice well above this.
        assert not heard(speech_like(RMS_AT_ROOM_FLOOR, seed=3), timeout=2.5)


class TestConverseMultiTurn:
    """The socket stays open the whole time: several utterances on one session each produce a
    turn, with quiet gaps between them left un-triggered."""

    def test_three_utterances_yield_three_transcripts(self):
        segs = [speech_like(RMS_NORMAL_SPEECH, seed=s) for s in (11, 22, 33)]
        got = run_utterances(segs, transcript="ok", timeout=6.0)
        assert got == ["ok", "ok", "ok"], f"expected 3 turns, got {got}"

    def test_gap_between_utterances_is_not_its_own_turn(self):
        # A soft voice, a long quiet gap, then a soft voice → exactly two turns (the gap alone
        # never trips).
        segs = [speech_like(RMS_QUIET_SPEECH, seed=44), speech_like(RMS_QUIET_SPEECH, seed=55)]
        got = run_utterances(segs, timeout=6.0)
        assert len(got) == 2, f"expected 2 turns, got {len(got)}: {got}"


class TestBoundaryIsDocumented:
    """Pin the sensitivity boundary so a future change that regresses it is caught: a soft
    voice at 400 RMS is heard, room tone at 250 is not."""

    def test_400_rms_heard_250_noise_silent(self):
        assert heard(speech_like(400, seed=400)), "400 RMS speech should be heard"
        assert not heard(room_noise(250, seed=8), timeout=2.5), "250 RMS noise should be silent"


class TestNoisyRoomSelfCalibrates:
    """A loud ambient (a TV at the mic, ~1500 RMS) must not chatter: the floor calibrates to the
    TV and the onset trigger rises above it, so the TV alone triggers nothing, but a louder voice
    over it is still heard. Real ConverseSession; STT mocked."""

    def _run(self, segments, *, timeout=4.0):
        import queue as _queue

        from tools.voice_converse_loop import ConverseSession
        with mocked_stt("heard"):
            session = ConverseSession(np, input_rate=16000)
            session.start()
            try:
                for seg in segments:
                    session.stream.feed(seg.tobytes())
                got = []
                try:
                    while True:
                        item = session.transcripts.get(timeout=timeout)
                        if item is None:
                            break
                        if isinstance(item, str) and item:
                            got.append(item)
                            break
                except _queue.Empty:
                    pass
                return got
            finally:
                session.stop()

    def test_steady_tv_background_alone_does_not_chatter(self):
        # ~3 s of steady TV-level noise (~1500 RMS), no voice → no turn. Before the self-
        # calibrating floor, the trigger pinned below the TV and this tripped every block.
        got = self._run([room_noise(1500, seconds=3.0, seed=5)], timeout=3.0)
        assert got == [], f"TV background alone produced a spurious turn: {got}"

    def test_voice_over_tv_is_heard(self):
        # TV background, then a louder voice (~3000) over it → heard (the floor sits at the TV,
        # the trigger just above it, the voice clears it).
        segs = [
            room_noise(1500, seconds=1.5, seed=6),
            speech_like(3000, seconds=1.4, seed=7),
            room_noise(1500, seconds=1.6, seed=8),
        ]
        assert self._run(segs, timeout=6.0) == ["heard"], "a voice over the TV should be heard"


class TestQuietTracksReceivedSilence:
    """The quiet signal must track silence in the RECEIVED stream, not wall-clock — so a client
    that holds the socket open and streams only after a wake word never accrues phantom quiet
    time. Real ConverseSession; STT mocked."""

    def test_streamed_silence_accrues_growing_quiet_then_speech_ends_it(self):
        from tools.voice_converse_loop import QuietTick
        # ~2 s of streamed silence (plus the calibration lead-in) at quiet_interval=0.5 →
        # several quiet ticks with a monotonically growing quiet_seconds, then a real
        # utterance ends the run with a transcript.
        events = collect_session_events(
            [silence(seconds=2.0), speech_like(RMS_NORMAL_SPEECH, seed=5), silence(seconds=1.5)],
            quiet_interval=0.5)
        ticks = [e for e in events if isinstance(e, QuietTick)]
        assert len(ticks) >= 2, f"expected quiet ticks from streamed silence, got {events!r}"
        qs = [t.quiet_seconds for t in ticks]
        assert qs == sorted(qs) and qs[0] > 0, f"quiet_seconds should grow from >0: {qs}"
        assert any(isinstance(e, str) and e for e in events), "speech should end with a transcript"

    def test_quiet_is_suppressed_during_a_turn_and_restarts_from_zero_after(self):
        # The turn-time bug: while a turn runs (agent think 14-50s + TTS), the client streams
        # silence it can't speak into. That silence must NOT accrue quiet, and after turn_done
        # the clock restarts from zero so the first advisory is a full interval away. Exercised
        # deterministically against the accounting (no worker thread / wall-clock).
        from tools.voice_converse_loop import ConverseSession, QuietTick

        session = ConverseSession(np, quiet_interval=0.3)  # worker not started
        session.begin_turn()
        for _ in range(60):  # ~1.8s of streamed silence DURING the turn
            session._account_received_silence()
        assert session.transcripts.empty(), "quiet accrued during a turn (user can't speak then)"

        session.end_turn()  # turn_done → clock restarts from zero
        for _ in range(12):  # ~0.36s of silence after the reply
            session._account_received_silence()
        ticks = []
        while not session.transcripts.empty():
            ticks.append(session.transcripts.get_nowait())
        first = next((t for t in ticks if isinstance(t, QuietTick)), None)
        assert first is not None, "quiet should accrue again once the turn is over"
        assert first.quiet_seconds <= 0.35, \
            f"quiet must restart near 0 after turn_done, not carry turn time: {first.quiet_seconds}"

    def test_no_audio_produces_no_quiet_ticks(self):
        # The bug: a session that receives ZERO audio bytes must not accrue quiet time, however
        # long it waits (a held-open socket before the wake word). Feed nothing past a wait that
        # is several quiet_intervals long, and confirm the queue never gets a QuietTick.
        from tools.voice_converse_loop import ConverseSession, QuietTick

        with mocked_stt("unused"):
            session = ConverseSession(np, quiet_interval=0.3)
            session.start()
            try:
                time.sleep(1.6)  # >5 intervals of pure wall-clock, but no audio is fed
                drained = []
                while not session.transcripts.empty():
                    drained.append(session.transcripts.get_nowait())
                assert not any(isinstance(x, QuietTick) for x in drained), \
                    f"quiet ticks fired without any received audio: {drained!r}"
            finally:
                session.stop()
