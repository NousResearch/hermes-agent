"""Tests for the Phonon local provider (``tools/tts_tool_local.py``, ``tools/phonon_voices.py``).

The engine is faked (no 64MB download, no ptts load) but every assertion is about the seam the
runtime depends on: what ``ptts.TTS`` is constructed with, what ``save`` is called with, and which
file the tool ends up with.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def clear_phonon_cache():
    """Reset the module-level model cache between tests."""
    from tools import tts_tool_local as _tt
    _tt._phonon_model_cache.clear()
    yield
    _tt._phonon_model_cache.clear()


@pytest.fixture
def fake_ptts():
    """A fake ``ptts`` module whose ``TTS`` writes a real (if bogus) file and reports a duration."""
    model = MagicMock()
    model.sample_rate = 24000

    def _save(path, text, **kwargs):
        Path(path).write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt fake")
        return 1.5

    model.save.side_effect = _save
    module = MagicMock()
    module.TTS = MagicMock(return_value=model)
    module.set_num_threads = MagicMock()
    return module, model


@pytest.fixture
def patched(monkeypatch, fake_ptts):
    """Patch the lazy importer the origin module resolves at call time."""
    from tools import tts_tool
    module, model = fake_ptts
    monkeypatch.setattr(tts_tool, "_import_ptts", lambda: module)
    return module, model


class TestGeneratePhonon:
    def test_writes_wav_and_returns_path(self, tmp_path, patched):
        from tools.tts_tool import _generate_phonon

        output = str(tmp_path / "out.wav")
        result = _generate_phonon("Hello world", output, {})
        assert result == output
        assert (tmp_path / "out.wav").exists()

    def test_default_voice_is_the_checkpoint_voice(self, tmp_path, patched):
        """``default`` must reach ptts as None (its own voice), not as a voice name."""
        from tools.tts_tool import _generate_phonon

        _, model = patched
        _generate_phonon("Hi", str(tmp_path / "a.wav"), {"phonon": {"voice": "default"}})
        assert model.save.call_args.kwargs["voice"] is None

    def test_catalogue_voice_is_passed_by_name(self, tmp_path, patched):
        from tools.tts_tool import _generate_phonon

        _, model = patched
        _generate_phonon("Hi", str(tmp_path / "a.wav"), {"phonon": {"voice": "Easton"}})
        assert model.save.call_args.kwargs["voice"] == "Easton"

    def test_lang_follows_the_checkpoint_not_the_voice(self, tmp_path, patched):
        """A French voice on the English checkpoint must still normalize as English.

        ptts warns ("trained for en") when the two disagree, and the model cannot pronounce the
        French spellings, so the checkpoint has to win.
        """
        from tools.tts_tool import _generate_phonon

        module, _ = patched
        _generate_phonon("Bonjour", str(tmp_path / "a.wav"), {"phonon": {"voice": "Noémie"}})
        assert module.TTS.call_args.kwargs["lang"] == "en"

    def test_lang_follows_the_voice_only_on_the_multilingual_checkpoint(self, tmp_path, patched):
        from tools.tts_tool import _generate_phonon

        module, _ = patched
        _generate_phonon("Bonjour", str(tmp_path / "a.wav"),
                         {"phonon": {"model": "multilingual", "voice": "Noémie"}})
        assert module.TTS.call_args.kwargs["lang"] == "fr"

    def test_explicit_lang_wins_over_everything(self, tmp_path, patched):
        from tools.tts_tool import _generate_phonon

        module, _ = patched
        _generate_phonon("Hi", str(tmp_path / "a.wav"),
                         {"phonon": {"voice": "Noémie", "lang": "en"}})
        assert module.TTS.call_args.kwargs["lang"] == "en"

    def test_short_model_tag_expands_to_hub_repo(self, tmp_path, patched):
        from tools.tts_tool import _generate_phonon

        module, _ = patched
        _generate_phonon("Hi", str(tmp_path / "a.wav"), {"phonon": {"model": "40m-multilingual"}})
        assert module.TTS.call_args.args[0] == "hf://gradium/phonon-40m-multilingual"

    def test_model_is_cached_across_calls(self, tmp_path, patched):
        from tools.tts_tool import _generate_phonon

        module, _ = patched
        for i in range(3):
            _generate_phonon("Hi", str(tmp_path / f"{i}.wav"), {"phonon": {"voice": "Easton"}})
        assert module.TTS.call_count == 1


class TestLoadErrors:
    def test_gated_repo_error_names_the_terms_and_hf_auth_login(self, tmp_path, monkeypatch):
        """The 40M checkpoints are gated: the failure has to say how to get access."""
        from tools import tts_tool

        module = monkeypatch_tts_raising(
            "OSError: cannot fetch `config.json` from hf://gradium/phonon-40m-fr: Forbidden: "
            "Access to model gradium/phonon-40m-fr is restricted (error_code=GatedRepo)")
        monkeypatch.setattr(tts_tool, "_import_ptts", lambda: module)
        with pytest.raises(RuntimeError) as excinfo:
            tts_tool._generate_phonon("Bonjour", str(tmp_path / "a.wav"),
                                      {"phonon": {"model": "40m-fr"}})
        message = str(excinfo.value)
        assert "hf auth login" in message
        assert "huggingface.co/gradium/phonon-40m-fr" in message
        assert "31m-en checkpoint is ungated" in message

    def test_other_load_errors_pass_through(self, tmp_path, monkeypatch):
        from tools import tts_tool

        module = monkeypatch_tts_raising("OSError: disk full")
        monkeypatch.setattr(tts_tool, "_import_ptts", lambda: module)
        with pytest.raises(RuntimeError) as excinfo:
            tts_tool._generate_phonon("Hi", str(tmp_path / "a.wav"), {})
        assert "disk full" in str(excinfo.value)
        assert "hf auth login" not in str(excinfo.value)


def monkeypatch_tts_raising(message: str):
    """A fake ptts whose TTS constructor raises ``message`` (as an OSError)."""
    from unittest.mock import MagicMock

    def _raise(*args, **kwargs):
        raise OSError(message)

    module = MagicMock()
    module.TTS = MagicMock(side_effect=_raise)
    module.set_num_threads = MagicMock()
    return module


class TestPhononModelId:
    @pytest.mark.parametrize("raw,expected", [
        ("", "hf://gradium/phonon-31m-en"),
        (None, "hf://gradium/phonon-31m-en"),
        ("31m-en", "hf://gradium/phonon-31m-en"),
        ("en", "hf://gradium/phonon-31m-en"),
        ("fr", "hf://gradium/phonon-40m-fr"),
        ("multilingual", "hf://gradium/phonon-40m-multilingual"),
        ("hf://gradium/phonon-31m-en@abc123", "hf://gradium/phonon-31m-en@abc123"),
        ("gradium/phonon-40m-en", "gradium/phonon-40m-en"),
        ("C:/models/phonon", "C:/models/phonon"),
    ])
    def test_resolution(self, raw, expected):
        from tools.tts_tool_local import _phonon_model_id
        assert _phonon_model_id(raw) == expected


class TestPhononVoiceArg:
    @pytest.mark.parametrize("raw,expected", [
        ("", None), (None, None), ("default", None), ("DEFAULT", None), ("Easton", "Easton"),
        ("  Miles  ", "Miles"),
    ])
    def test_resolution(self, raw, expected):
        from tools.tts_tool_local import _phonon_voice_arg
        assert _phonon_voice_arg(raw) == expected


class TestVoiceCatalogue:
    def test_catalogue_is_complete_and_unique(self):
        from tools.phonon_voices import PHONON_VOICES, PHONON_VOICE_NAMES
        assert len(PHONON_VOICES) == 102
        assert len(set(PHONON_VOICE_NAMES)) == 102
        assert len({row[1] for row in PHONON_VOICES}) == 102  # voice_ids unique too

    def test_lookup_is_accent_and_case_insensitive(self):
        from tools.phonon_voices import phonon_voice
        assert phonon_voice("noemie")[0] == "Noémie"
        assert phonon_voice("NOÉMIE")[0] == "Noémie"
        assert phonon_voice("easton")[0] == "Easton"

    def test_lookup_by_voice_id(self):
        from tools.phonon_voices import phonon_voice
        assert phonon_voice("LMDIYE0Oa377Qh6Z")[0] == "Easton"

    @pytest.mark.parametrize("name,lang", [
        ("Easton", "en"), ("Maeve", "en"), ("Noémie", "fr"), ("Nele", "de"), ("Noa", "es"),
        ("Manuela-Lu", "pt"), ("Azgorath", "en"),
    ])
    def test_lang_for_voice(self, name, lang):
        from tools.phonon_voices import phonon_lang_for_voice
        assert phonon_lang_for_voice(name) == lang

    def test_unknown_voice_has_no_lang(self):
        from tools.phonon_voices import phonon_lang_for_voice
        assert phonon_lang_for_voice("Not A Voice") is None


class TestDispatcher:
    def test_missing_ptts_returns_actionable_error(self, monkeypatch, tmp_path):
        """provider=phonon without the extra must explain how to install it, not crash."""
        import hermes_yaml as yaml
        from tools import tts_tool

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(tts_tool, "_check_phonon_available", lambda: False)
        monkeypatch.setattr(tts_tool, "_import_ptts", _raise_import_error)
        monkeypatch.setattr(tts_tool, "_ready_after_first_use_install", lambda _p: False)
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({"tts": {"provider": "phonon"}}))

        result = json.loads(tts_tool.text_to_speech_tool(text="Hello"))
        assert result["success"] is False
        assert "phonon" in result["error"].lower()

    def test_provider_is_registered_everywhere(self):
        from agent.tts_registry import _BUILTIN_NAMES
        from tools.tts_command_provider import BUILTIN_TTS_PROVIDERS
        from tools.tts_tool import _BUILTIN_DISPATCH, _BUILTIN_REQUIREMENTS
        for name in ("phonon", "ollama"):
            assert name in BUILTIN_TTS_PROVIDERS
            assert name in _BUILTIN_NAMES
            assert name in _BUILTIN_DISPATCH
            assert name in _BUILTIN_REQUIREMENTS

    def test_text_caps(self):
        from tools.tts_tool_delivery import _resolve_max_text_length
        assert _resolve_max_text_length("phonon") == 2000
        assert _resolve_max_text_length("ollama") == 4096

    def test_opus_conversion_paths(self):
        from tools.tts_tool import _FFMPEG_OPUS_PROVIDERS
        # Both emit WAV, so a voice-bubble platform needs ffmpeg to reach Ogg/Opus.
        assert {"phonon", "ollama"} <= _FFMPEG_OPUS_PROVIDERS


def _raise_import_error():
    raise ImportError("ptts not installed")
