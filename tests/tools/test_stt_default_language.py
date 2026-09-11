"""Default STT language contract.

Teknium (July 2026): the global ``stt.language`` DEFAULTS to "en" because
Whisper auto-detection frequently misidentifies short/accented clips
("STT transcribed the wrong language" class). Users opt back into
auto-detect with ``stt.language: ""``.
"""

from hermes_cli.config import DEFAULT_CONFIG
from tools.transcription_tools import _resolve_stt_language


class TestDefaultSttLanguage:
    def test_default_config_pins_english(self):
        assert DEFAULT_CONFIG["stt"]["language"] == "en"


    def test_per_provider_still_wins_over_default(self, monkeypatch):
        monkeypatch.delenv("HERMES_LOCAL_STT_LANGUAGE", raising=False)
        stt = dict(DEFAULT_CONFIG["stt"])
        stt["groq"] = {"language": "he"}
        assert _resolve_stt_language("groq", stt) == "he"
