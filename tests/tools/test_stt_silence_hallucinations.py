"""Tests for the local faster-whisper silence-hallucination hardening.

One shared kwargs owner (`build_local_transcribe_kwargs`) must apply the
three-layer fix at every local whisper call site:

1. Silero VAD filter on by default (``stt.local.vad: false`` restores raw).
2. ``condition_on_previous_text=False`` always.
3. Segment confidence gate: drop segments only when the model BOTH thinks
   the window is non-speech AND decoded it with low confidence — quiet but
   real speech must survive.
"""

from types import SimpleNamespace

from tools.transcription_tools import (
    _LOGPROB_THRESHOLD_DEFAULT,
    _NO_SPEECH_PROB_THRESHOLD_DEFAULT,
    _is_hallucinated_segment,
    _join_confident_segments,
    build_local_transcribe_kwargs,
)


def _seg(text, no_speech_prob=0.0, avg_logprob=-0.2):
    return SimpleNamespace(text=text, no_speech_prob=no_speech_prob, avg_logprob=avg_logprob)


class TestBuildLocalTranscribeKwargs:
    def test_vad_on_by_default(self):
        kwargs = build_local_transcribe_kwargs({})
        assert kwargs["vad_filter"] is True
        assert kwargs["vad_parameters"] == {"min_silence_duration_ms": 500}

    def test_conditioning_always_off(self):
        assert build_local_transcribe_kwargs({})["condition_on_previous_text"] is False
        assert (
            build_local_transcribe_kwargs({"local": {"vad": False}})[
                "condition_on_previous_text"
            ]
            is False
        )


    def test_confidence_thresholds_default_to_faster_whisper_values(self):
        kwargs = build_local_transcribe_kwargs({})
        assert kwargs["no_speech_threshold"] == _NO_SPEECH_PROB_THRESHOLD_DEFAULT
        assert kwargs["log_prob_threshold"] == _LOGPROB_THRESHOLD_DEFAULT

    def test_confidence_thresholds_configurable_reach_model_gate(self):
        # The same stt.local knobs the post-filter reads must also be threaded
        # into faster-whisper's internal gate, or non-English speech is dropped
        # before it ever reaches our segment filter.
        kwargs = build_local_transcribe_kwargs(
            {"local": {"no_speech_prob_threshold": 0.9, "logprob_threshold": -2.0}}
        )
        assert kwargs["no_speech_threshold"] == 0.9
        assert kwargs["log_prob_threshold"] == -2.0

    def test_confidence_thresholds_garbage_falls_back(self):
        kwargs = build_local_transcribe_kwargs(
            {"local": {"no_speech_prob_threshold": "nope", "logprob_threshold": None}}
        )
        assert kwargs["no_speech_threshold"] == _NO_SPEECH_PROB_THRESHOLD_DEFAULT
        assert kwargs["log_prob_threshold"] == _LOGPROB_THRESHOLD_DEFAULT

    def test_language_and_prompt_resolved(self, monkeypatch):
        monkeypatch.delenv("HERMES_LOCAL_STT_LANGUAGE", raising=False)
        cfg = {"language": "en", "local": {"initial_prompt": "Hermes glossary"}}
        kwargs = build_local_transcribe_kwargs(cfg)
        assert kwargs["language"] == "en"
        assert kwargs["initial_prompt"] == "Hermes glossary"


class TestConfidenceGate:
    def test_high_no_speech_and_low_logprob_dropped(self):
        seg = _seg(" You", no_speech_prob=0.9, avg_logprob=-1.5)
        assert _is_hallucinated_segment(
            seg, _NO_SPEECH_PROB_THRESHOLD_DEFAULT, _LOGPROB_THRESHOLD_DEFAULT
        )

    def test_quiet_but_confident_speech_survives(self):
        # High no_speech_prob alone must NOT drop a segment the model decoded
        # confidently (quiet-but-real speech).
        seg = _seg(" hello there", no_speech_prob=0.8, avg_logprob=-0.3)
        assert not _is_hallucinated_segment(
            seg, _NO_SPEECH_PROB_THRESHOLD_DEFAULT, _LOGPROB_THRESHOLD_DEFAULT
        )


    def test_garbage_thresholds_fall_back_to_defaults(self):
        seg = _seg(" ok", no_speech_prob=0.1, avg_logprob=-0.1)
        cfg = {"no_speech_prob_threshold": "high", "logprob_threshold": None}
        assert _join_confident_segments([seg], cfg) == "ok"


class TestTranscribeLocalWiring:
    """_transcribe_local must pass the shared hardened kwargs to the model."""

    def _run(self, monkeypatch, stt_config, segments=None):
        import tools.transcription_tools as tt

        captured = {}

        class FakeModel:
            def transcribe(self, path, **kwargs):
                captured.update(kwargs)
                info = SimpleNamespace(language="en", duration=1.0)
                return iter(segments or [_seg(" hi")]), info

        monkeypatch.setattr(tt, "_HAS_FASTER_WHISPER", True)
        monkeypatch.setattr(tt, "_local_model", FakeModel())
        monkeypatch.setattr(tt, "_local_model_name", "base")
        monkeypatch.setattr(tt, "_load_stt_config", lambda: stt_config)
        monkeypatch.delenv("HERMES_LOCAL_STT_LANGUAGE", raising=False)
        result = tt._transcribe_local("/tmp/fake.wav", "base")
        return captured, result

    def test_hardened_kwargs_reach_model(self, monkeypatch):
        captured, result = self._run(monkeypatch, {})
        assert result["success"] is True
        assert captured["vad_filter"] is True
        assert captured["vad_parameters"] == {"min_silence_duration_ms": 500}
        assert captured["condition_on_previous_text"] is False
        assert captured["no_speech_threshold"] == _NO_SPEECH_PROB_THRESHOLD_DEFAULT
        assert captured["log_prob_threshold"] == _LOGPROB_THRESHOLD_DEFAULT


    def test_hallucinated_segments_filtered_from_transcript(self, monkeypatch):
        segments = [
            _seg(" real speech"),
            _seg(" Дякую за перегляд!", no_speech_prob=0.97, avg_logprob=-1.6),
        ]
        _, result = self._run(monkeypatch, {}, segments=segments)
        assert result["transcript"] == "real speech"
