"""Default STT language contract.

The global ``stt.language`` defaults to ``""`` (auto-detect). A set value is FORCED on
every provider, and the seeded ``"en"`` (July 2026) turned every non-English voice note
into English nonsense — a 17 s Dutch Telegram voice note came back as
"This is an interview with Fabian Pinkhash for OXP … he will be 20 years old" (85 %
word error rate) while auto-detect on the same ``base`` model heard Dutch.

The reason the forced default existed — Whisper misreads short/accented clips — is
covered by the local provider's confidence gate instead: an auto-detected language
with ``language_probability`` below ``stt.local.language_confidence_threshold`` is
re-transcribed with ``stt.local.fallback_language``.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hermes_cli.config import DEFAULT_CONFIG
from tools.transcription_local import low_confidence_fallback_language
from tools.transcription_tools import _resolve_stt_language


class TestDefaultSttLanguage:
    def test_default_config_auto_detects(self):
        assert DEFAULT_CONFIG["stt"]["language"] == ""
        assert DEFAULT_CONFIG["stt"]["local"]["language"] == ""

    def test_default_config_keeps_an_english_fallback_for_the_local_gate(self):
        local = DEFAULT_CONFIG["stt"]["local"]
        assert local["fallback_language"] == "en"
        assert local["language_confidence_threshold"] == 0.6

    def test_nothing_is_forced_by_default(self, monkeypatch):
        monkeypatch.delenv("HERMES_LOCAL_STT_LANGUAGE", raising=False)
        assert _resolve_stt_language("local", dict(DEFAULT_CONFIG["stt"])) is None
        assert _resolve_stt_language("xai", dict(DEFAULT_CONFIG["stt"])) is None

    def test_per_provider_still_wins_over_global(self, monkeypatch):
        monkeypatch.delenv("HERMES_LOCAL_STT_LANGUAGE", raising=False)
        stt = dict(DEFAULT_CONFIG["stt"])
        stt["language"] = "en"
        stt["groq"] = {"language": "he"}
        assert _resolve_stt_language("groq", stt) == "he"
        assert _resolve_stt_language("local", stt) == "en"


def _info(language, probability):
    return SimpleNamespace(language=language, language_probability=probability, duration=17.2)


class TestLowConfidenceFallback:
    CFG = {"fallback_language": "en", "language_confidence_threshold": 0.6}

    def test_low_confidence_guess_falls_back(self):
        assert low_confidence_fallback_language(_info("nl", 0.54), self.CFG) == "en"

    def test_confident_guess_is_kept(self):
        assert low_confidence_fallback_language(_info("nl", 0.71), self.CFG) is None
        assert low_confidence_fallback_language(_info("nl", 0.6), self.CFG) is None

    def test_guess_equal_to_the_fallback_needs_no_second_pass(self):
        assert low_confidence_fallback_language(_info("en", 0.2), self.CFG) is None

    def test_disabled_fallback_and_unknown_shapes(self):
        assert low_confidence_fallback_language(_info("nl", 0.1), {"fallback_language": ""}) is None
        assert low_confidence_fallback_language(_info("nl", 0.1), {}) is None
        no_prob = SimpleNamespace(language="nl")
        assert low_confidence_fallback_language(no_prob, self.CFG) is None

    def test_threshold_is_configurable(self):
        cfg = {"fallback_language": "de", "language_confidence_threshold": 0.9}
        assert low_confidence_fallback_language(_info("nl", 0.85), cfg) == "de"


def _segment(text):
    return SimpleNamespace(text=text, no_speech_prob=0.0, avg_logprob=-0.2)


def _run_local(model, stt_config):
    from tools import transcription_tools as tt

    with patch.object(tt, "_HAS_FASTER_WHISPER", True), \
         patch.object(tt, "_load_stt_config", return_value=stt_config), \
         patch.object(tt, "_get_or_load_local_model", return_value=model), \
         patch("tools.transcription_local._load_stt_config", return_value=stt_config, create=True):
        return tt._transcribe_local("/tmp/clip.ogg", "base")


class TestLocalTranscribeGate:
    STT = {"language": "", "local": {"language": "", "fallback_language": "en",
                                     "language_confidence_threshold": 0.6}}

    def test_low_confidence_auto_detect_retries_with_the_fallback(self):
        model = MagicMock()
        model.transcribe.side_effect = [
            ([_segment("Dit is een lied")], _info("nl", 0.54)),
            ([_segment("This is a lead")], _info("en", 1.0)),
        ]
        result = _run_local(model, self.STT)
        assert result["success"] and result["transcript"] == "This is a lead"
        assert model.transcribe.call_count == 2
        first, second = model.transcribe.call_args_list
        assert "language" not in first.kwargs
        assert second.kwargs["language"] == "en"

    def test_confident_auto_detect_transcribes_once(self):
        model = MagicMock()
        model.transcribe.return_value = ([_segment("Noteer Fabien als een lead")], _info("nl", 0.71))
        result = _run_local(model, self.STT)
        assert result["transcript"] == "Noteer Fabien als een lead"
        assert model.transcribe.call_count == 1
        assert "language" not in model.transcribe.call_args.kwargs

    def test_forced_language_never_hits_the_gate(self):
        model = MagicMock()
        model.transcribe.return_value = ([_segment("hoi")], _info("nl", 0.1))
        forced = {"language": "nl", "local": dict(self.STT["local"])}
        _run_local(model, forced)
        assert model.transcribe.call_count == 1
        assert model.transcribe.call_args.kwargs["language"] == "nl"
