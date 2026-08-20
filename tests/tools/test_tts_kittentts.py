"""Tests for the KittenTTS local provider in tools/tts_tool.py."""

import json
import sys
import types
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("HERMES_SESSION_PLATFORM",):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def clear_kittentts_cache():
    """Reset the module-level model cache between tests."""
    from tools import tts_tool as _tt
    _tt._kittentts_model_cache.clear()
    yield
    _tt._kittentts_model_cache.clear()


@pytest.fixture
def mock_kittentts_module():
    """Inject a fake kittentts + soundfile module that return stub objects."""
    fake_model = MagicMock()
    # 24kHz float32 PCM at ~2s of silence
    fake_model.generate.return_value = [0.0] * 48000
    fake_cls = MagicMock(return_value=fake_model)
    fake_kittentts = MagicMock()
    fake_kittentts.KittenTTS = fake_cls

    # Stub soundfile — the real package isn't installed in CI venv, and
    # _generate_kittentts does `import soundfile as sf` at runtime.
    fake_sf = MagicMock()
    def _fake_write(path, audio, samplerate):
        # Emulate writing a real file so downstream path checks succeed.
        import pathlib
        pathlib.Path(path).write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt fake")
    fake_sf.write = _fake_write

    with patch.dict(
        "sys.modules",
        {"kittentts": fake_kittentts, "soundfile": fake_sf},
    ):
        yield fake_model, fake_cls


class TestGenerateKittenTts:
    def test_successful_wav_generation(self, tmp_path, mock_kittentts_module):
        from tools.tts_tool import _generate_kittentts

        fake_model, fake_cls = mock_kittentts_module
        output_path = str(tmp_path / "test.wav")
        result = _generate_kittentts("Hello world", output_path, {})

        assert result == output_path
        assert (tmp_path / "test.wav").exists()
        fake_cls.assert_called_once()
        fake_model.generate.assert_called_once()

    def test_config_passes_voice_speed_cleantext(self, tmp_path, mock_kittentts_module):
        from tools.tts_tool import _generate_kittentts

        fake_model, _ = mock_kittentts_module
        config = {
            "kittentts": {
                "model": "KittenML/kitten-tts-mini-0.8",
                "voice": "Luna",
                "speed": 1.25,
                "clean_text": False,
            }
        }
        _generate_kittentts("Hi there", str(tmp_path / "out.wav"), config)

        call_kwargs = fake_model.generate.call_args.kwargs
        assert call_kwargs["voice"] == "Luna"
        assert call_kwargs["speed"] == 1.25
        assert call_kwargs["clean_text"] is False


    def test_missing_kittentts_raises_import_error(self, tmp_path, monkeypatch):
        """When kittentts package is not installed, _import_kittentts raises."""
        import sys
        monkeypatch.setitem(sys.modules, "kittentts", None)
        from tools.tts_tool import _generate_kittentts

        with pytest.raises((ImportError, TypeError)):
            _generate_kittentts("Hi", str(tmp_path / "out.wav"), {})


class TestCheckKittenttsAvailable:
    def test_reports_available_when_package_present(self, monkeypatch):
        import importlib.util
        from tools.tts_tool import _check_kittentts_available

        fake_spec = MagicMock()
        monkeypatch.setattr(
            importlib.util, "find_spec",
            lambda name: fake_spec if name == "kittentts" else None,
        )
        assert _check_kittentts_available() is True

    def test_reports_unavailable_when_package_missing(self, monkeypatch):
        import importlib.util
        from tools.tts_tool import _check_kittentts_available

        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
        assert _check_kittentts_available() is False


class TestDispatcherBranch:
    def test_kittentts_not_installed_returns_helpful_error(self, monkeypatch, tmp_path):
        """When provider=kittentts but package missing, return JSON error with setup hint."""
        import sys
        monkeypatch.setitem(sys.modules, "kittentts", None)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from tools.tts_tool import text_to_speech_tool

        # Write a config telling it to use kittentts
        import yaml
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"tts": {"provider": "kittentts"}})
        )

        result = json.loads(text_to_speech_tool(text="Hello"))
        assert result["success"] is False
        assert "kittentts" in result["error"].lower()
        assert "hermes setup tts" in result["error"].lower()


# ---------------------------------------------------------------------------
# Voice selection contract (#79459)
# ---------------------------------------------------------------------------

class TestKittenTtsVoiceSelection:
    def test_top_level_tts_voice_used_when_no_provider_voice(
        self, tmp_path, mock_kittentts_module
    ):
        """``tts.voice`` (the plugin-dispatch key) must reach kittentts when
        ``tts.kittentts.voice`` is unset, instead of silently rendering the
        provider default (#79459)."""
        from tools.tts_tool import _generate_kittentts

        fake_model, _ = mock_kittentts_module
        config = {"voice": "Luna"}
        _generate_kittentts("Hi", str(tmp_path / "out.wav"), config)

        assert fake_model.generate.call_args.kwargs["voice"] == "Luna"

    def test_provider_voice_beats_top_level_voice(self, tmp_path, mock_kittentts_module):
        from tools.tts_tool import _generate_kittentts

        fake_model, _ = mock_kittentts_module
        config = {"voice": "Jasper", "kittentts": {"voice": "Luna"}}
        _generate_kittentts("Hi", str(tmp_path / "out.wav"), config)

        assert fake_model.generate.call_args.kwargs["voice"] == "Luna"

    def test_unknown_voice_raises_with_available_list(self, tmp_path, mock_kittentts_module):
        """A voice absent from the model's voice table must fail loudly with
        the valid choices — never silently render a different voice."""
        from tools.tts_tool import _generate_kittentts

        fake_model, _ = mock_kittentts_module
        inner = MagicMock()
        inner.all_voice_names = ["expr-voice-2-f", "expr-voice-2-m"]
        inner.voice_aliases = {"Jasper": "expr-voice-2-m"}
        fake_model.model = inner

        config = {"kittentts": {"voice": "Zoltan"}}
        with pytest.raises(RuntimeError) as excinfo:
            _generate_kittentts("Hi", str(tmp_path / "out.wav"), config)

        msg = str(excinfo.value)
        assert "Zoltan" in msg
        assert "expr-voice-2-f" in msg
        # alias must be accepted as a valid choice too
        assert "Jasper" in msg
        fake_model.generate.assert_not_called()

    def test_known_alias_voice_passes_validation(self, tmp_path, mock_kittentts_module):
        from tools.tts_tool import _generate_kittentts

        fake_model, _ = mock_kittentts_module
        inner = MagicMock()
        inner.all_voice_names = ["expr-voice-2-m"]
        inner.voice_aliases = {"Jasper": "expr-voice-2-m"}
        fake_model.model = inner

        config = {"kittentts": {"voice": "Jasper"}}
        _generate_kittentts("Hi", str(tmp_path / "out.wav"), config)

        assert fake_model.generate.call_args.kwargs["voice"] == "Jasper"


# ---------------------------------------------------------------------------
# Legacy library compatibility (#79459): PyPI 0.1.x expects local file
# paths and has no clean_text kwarg.
# ---------------------------------------------------------------------------

class _LegacyKittenModel:
    """Mimics kittentts 0.1.3: ctor takes local paths, generate(text, voice, speed)."""

    instances = []

    def __init__(self, model_path=None, voices_path=None):
        if model_path is not None and "/" in str(model_path):
            # ort.InferenceSession on a HF repo id
            raise RuntimeError(
                "[ONNXRuntimeError] : 3 : NO_SUCHFILE : Load model from "
                f"{model_path} failed"
            )
        self.model_path = model_path
        self.voices_path = voices_path
        self.available_voices = ["expr-voice-2-f", "expr-voice-2-m"]
        self.last_generate = None
        _LegacyKittenModel.instances.append(self)

    def generate(self, text, voice="expr-voice-2-m", speed=1.0):
        self.last_generate = {"text": text, "voice": voice, "speed": speed}
        return [0.0] * 24000


@pytest.fixture
def legacy_kittentts(monkeypatch):
    from tools import tts_tool

    _LegacyKittenModel.instances = []
    monkeypatch.setattr(tts_tool, "_import_kittentts", lambda: _LegacyKittenModel)
    # Reset the module-level model cache so every test constructs through
    # its own monkeypatched resolution/constructor stubs.
    monkeypatch.setattr(tts_tool, "_kittentts_model_cache", {})
    # Stub soundfile (not installed in CI venv).
    fake_sf = MagicMock()
    fake_sf.write = lambda path, audio, samplerate: (
        __import__("pathlib").Path(path).write_bytes(b"RIFF fake")
    )
    monkeypatch.setitem(sys.modules, "soundfile", fake_sf)
    yield


class TestKittenTtsLegacyCompatibility:
    def test_repo_id_resolved_to_local_paths_for_legacy_library(
        self, tmp_path, monkeypatch, legacy_kittentts
    ):
        """When the installed kittentts cannot handle HF repo ids, hermes
        resolves the repo via huggingface_hub and passes explicit local
        paths to the constructor."""
        from tools import tts_tool

        local_model = tmp_path / "model.onnx"
        local_voices = tmp_path / "voices.npz"
        local_model.write_bytes(b"m")
        local_voices.write_bytes(b"v")
        monkeypatch.setattr(
            tts_tool,
            "_resolve_kittentts_model_files",
            lambda name: (str(local_model), str(local_voices), {}),
        )

        config = {
            "kittentts": {
                "model": "KittenML/kitten-tts-nano-0.8-int8",
                "voice": "expr-voice-2-f",
            }
        }
        result = tts_tool._generate_kittentts("Hi", str(tmp_path / "out.wav"), config)

        assert result == str(tmp_path / "out.wav")
        inst = _LegacyKittenModel.instances[-1]
        assert inst.model_path == str(local_model)
        assert inst.voices_path == str(local_voices)
        # legacy generate() has no clean_text kwarg — must not receive it
        assert inst.last_generate["voice"] == "expr-voice-2-f"

    def test_legacy_voice_validation_uses_available_voices_attr(
        self, tmp_path, monkeypatch, legacy_kittentts
    ):
        from tools import tts_tool

        # Let construction succeed via resolved local paths (no "/" in the
        # fake names so the legacy ctor accepts them), then the voice check
        # must reject an explicit 'Jasper' when the resolution carried no
        # alias metadata.
        monkeypatch.setattr(
            tts_tool,
            "_resolve_kittentts_model_files",
            lambda name: ("model.onnx", "voices.npz", {}),
        )
        config = {"kittentts": {"voice": "Jasper"}}  # not in legacy voice table
        with pytest.raises(RuntimeError) as excinfo:
            tts_tool._generate_kittentts("Hi", str(tmp_path / "out.wav"), config)
        assert "expr-voice-2-f" in str(excinfo.value)

    def test_legacy_default_voice_normalized_via_repo_aliases(
        self, tmp_path, monkeypatch, legacy_kittentts
    ):
        """The no-config default (Jasper) must keep working on the legacy
        0.1.x stack: the repo config.json voice_aliases mapping carried by
        the HF resolution normalizes it to a raw voice the model accepts,
        instead of the validation rejecting it (review on #79581)."""
        from tools import tts_tool

        aliases = {"Jasper": "expr-voice-2-m", "Luna": "expr-voice-2-f"}
        monkeypatch.setattr(
            tts_tool,
            "_resolve_kittentts_model_files",
            lambda name: ("model.onnx", "voices.npz", aliases),
        )
        # No voice configured anywhere -> DEFAULT_KITTENTTS_VOICE (Jasper)
        result = tts_tool._generate_kittentts("Hi", str(tmp_path / "out.wav"), {})

        assert result == str(tmp_path / "out.wav")
        inst = _LegacyKittenModel.instances[-1]
        assert inst.last_generate["voice"] == "expr-voice-2-m"

    def test_legacy_configured_alias_normalized_before_generate(
        self, tmp_path, monkeypatch, legacy_kittentts
    ):
        """An explicitly configured alias voice is normalized too, so users
        can keep alias names across library versions."""
        from tools import tts_tool

        aliases = {"Luna": "expr-voice-2-f"}
        monkeypatch.setattr(
            tts_tool,
            "_resolve_kittentts_model_files",
            lambda name: ("model.onnx", "voices.npz", aliases),
        )
        config = {"kittentts": {"voice": "Luna"}}
        tts_tool._generate_kittentts("Hi", str(tmp_path / "out.wav"), config)

        assert _LegacyKittenModel.instances[-1].last_generate["voice"] == "expr-voice-2-f"

    def test_legacy_alias_mapping_to_unknown_voice_still_fails(
        self, tmp_path, monkeypatch, legacy_kittentts
    ):
        """Alias normalization must not mask misconfiguration: an alias whose
        target is absent from the loaded model still fails loudly."""
        from tools import tts_tool

        aliases = {"Mystery": "expr-voice-9-z"}  # target not in available_voices
        monkeypatch.setattr(
            tts_tool,
            "_resolve_kittentts_model_files",
            lambda name: ("model.onnx", "voices.npz", aliases),
        )
        config = {"kittentts": {"voice": "Mystery"}}
        with pytest.raises(RuntimeError) as excinfo:
            tts_tool._generate_kittentts("Hi", str(tmp_path / "out.wav"), config)
        assert "expr-voice-9-z" in str(excinfo.value)

    def test_unresolvable_failure_surfaces_upgrade_hint(self, monkeypatch):
        """Constructor failure + impossible resolution = actionable error."""
        from tools.tts_tool import _construct_kittentts_model

        def broken_ctor(model_name=None, model_path=None, voices_path=None):
            raise RuntimeError("NO_SUCHFILE")

        monkeypatch.setattr(
            "tools.tts_tool._resolve_kittentts_model_files", lambda name: None
        )
        with pytest.raises(RuntimeError) as excinfo:
            _construct_kittentts_model(broken_ctor, "KittenML/kitten-tts-nano-0.8-int8")
        assert "kittentts-0.8.1" in str(excinfo.value)


class TestResolveKittenTtsModelFiles:
    def test_non_repo_name_returns_none(self):
        from tools.tts_tool import _resolve_kittentts_model_files

        assert _resolve_kittentts_model_files("local-model") is None
        assert _resolve_kittentts_model_files("") is None

    def test_resolves_repo_layout_via_hf_hub(self, tmp_path, monkeypatch):
        from tools import tts_tool

        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "type": "ONNX2",
                    "model_file": "model.onnx",
                    "voices": "voices.npz",
                    "voice_aliases": {"Jasper": "expr-voice-2-m", "Empty": ""},
                }
            ),
            encoding="utf-8",
        )
        files = {
            "config.json": str(config_path),
            "model.onnx": str(tmp_path / "model.onnx"),
            "voices.npz": str(tmp_path / "voices.npz"),
        }
        fake_hub = types.SimpleNamespace(
            hf_hub_download=lambda repo_id, filename, **kw: files[filename]
        )
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

        resolved = tts_tool._resolve_kittentts_model_files("KittenML/kitten-tts-nano-0.8-int8")
        assert resolved == (
            str(tmp_path / "model.onnx"),
            str(tmp_path / "voices.npz"),
            {"Jasper": "expr-voice-2-m"},  # empty-valued entries dropped
        )

    def test_resolves_repo_without_aliases_field(self, tmp_path, monkeypatch):
        from tools import tts_tool

        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps({"type": "ONNX2", "model_file": "model.onnx"}),
            encoding="utf-8",
        )
        files = {
            "config.json": str(config_path),
            "model.onnx": str(tmp_path / "model.onnx"),
        }
        fake_hub = types.SimpleNamespace(
            hf_hub_download=lambda repo_id, filename, **kw: files[filename]
        )
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

        resolved = tts_tool._resolve_kittentts_model_files("KittenML/legacy-model")
        assert resolved == (str(tmp_path / "model.onnx"), None, {})

    def test_repo_without_model_file_returns_none(self, tmp_path, monkeypatch):
        from tools import tts_tool

        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"type": "unknown"}), encoding="utf-8")
        fake_hub = types.SimpleNamespace(
            hf_hub_download=lambda repo_id, filename, **kw: str(config_path)
        )
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

        assert tts_tool._resolve_kittentts_model_files("KittenML/legacy-model") is None
