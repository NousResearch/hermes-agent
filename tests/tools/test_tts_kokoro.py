"""Tests for the native Kokoro TTS provider.

These tests pin the registration / availability / cache / dispatch / config
paths for Kokoro without requiring the ``mlx-audio`` and ``misaki`` packages
to actually be installed. The synthesis step is monkey-patched: a stub
KokoroPipeline yields fake audio chunks so the test runs on every CI runner,
including non-Apple-Silicon ones where ``mlx_audio`` cannot be installed.

The shape mirrors ``test_tts_piper.py`` deliberately — Kokoro is the
in-process Apple-Silicon-native sibling of Piper.
"""

import json
import sys
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools import tts_tool
from tools.tts_tool import (
    BUILTIN_TTS_PROVIDERS,
    DEFAULT_KOKORO_LANG_CODE,
    DEFAULT_KOKORO_REPO,
    DEFAULT_KOKORO_VOICE,
    PROVIDER_MAX_TEXT_LENGTH,
    _check_kokoro_available,
    check_tts_requirements,
    text_to_speech_tool,
)


# ---------------------------------------------------------------------------
# Registry / constants
# ---------------------------------------------------------------------------

class TestKokoroRegistration:
    def test_kokoro_is_a_builtin_provider(self):
        assert "kokoro" in BUILTIN_TTS_PROVIDERS

    def test_kokoro_has_a_text_length_cap(self):
        assert PROVIDER_MAX_TEXT_LENGTH.get("kokoro", 0) > 0

    def test_kokoro_defaults_resolve(self):
        # The defaults must be populated and well-formed; downstream config
        # resolution falls back to these when ``tts.kokoro.*`` is absent.
        assert DEFAULT_KOKORO_REPO.startswith("mlx-community/")
        assert DEFAULT_KOKORO_VOICE.startswith(("af_", "am_", "bf_", "bm_"))
        assert DEFAULT_KOKORO_LANG_CODE in {"a", "b", "j", "z", "p", "i", "f", "h", "e"}


# ---------------------------------------------------------------------------
# _check_kokoro_available
# ---------------------------------------------------------------------------

class TestCheckKokoroAvailable:
    def test_returns_bool_without_raising(self):
        # We don't care about the current environment's answer — just that
        # the probe never raises on a machine without mlx-audio installed.
        assert isinstance(_check_kokoro_available(), bool)

    def test_returns_false_when_mlx_audio_missing(self, monkeypatch):
        # Patch find_spec to simulate mlx-audio not installed.
        import importlib.util as _util
        original = _util.find_spec

        def fake_find_spec(name, *args, **kwargs):
            if name in ("mlx_audio", "misaki"):
                return None
            return original(name, *args, **kwargs)

        monkeypatch.setattr(_util, "find_spec", fake_find_spec)
        assert _check_kokoro_available() is False

    def test_returns_true_when_both_present(self, monkeypatch):
        # Both modules report present.
        import importlib.util as _util
        sentinel = MagicMock()
        original = _util.find_spec

        def fake_find_spec(name, *args, **kwargs):
            if name in ("mlx_audio", "misaki"):
                return sentinel
            return original(name, *args, **kwargs)

        monkeypatch.setattr(_util, "find_spec", fake_find_spec)
        assert _check_kokoro_available() is True


# ---------------------------------------------------------------------------
# _import_kokoro
# ---------------------------------------------------------------------------

class TestImportKokoro:
    def test_raises_importerror_with_actionable_message(self, monkeypatch):
        # Force the underlying imports to fail to verify the user-facing
        # error message points at the correct extra.
        def kokoro_import_fail(*args, **kwargs):
            raise ImportError("No module named 'mlx_audio'")

        # The simplest way to force ImportError on the inner imports is to
        # remove the modules from sys.modules and then poison the loader
        # path. Using monkeypatch on sys.modules is cleanest.
        monkeypatch.setitem(sys.modules, "mlx_audio", None)
        monkeypatch.setitem(sys.modules, "mlx_audio.tts", None)
        monkeypatch.setitem(sys.modules, "mlx_audio.tts.models", None)
        monkeypatch.setitem(sys.modules, "mlx_audio.tts.models.kokoro", None)

        with pytest.raises(ImportError):
            tts_tool._import_kokoro()


# ---------------------------------------------------------------------------
# _generate_kokoro_tts — stubbed so we don't need mlx-audio installed
# ---------------------------------------------------------------------------

class _FakeMxArray:
    """Stand-in for an mlx.core.array shape (1, N) — pretends to be audio."""

    def __init__(self, samples):
        # samples is a numpy float32 array. We mimic mx.array shape (1, N).
        import numpy as np
        self._np = np.asarray(samples, dtype=np.float32).reshape(1, -1)
        self.shape = self._np.shape
        # Make np.asarray(self) flatten back to 1-D in _generate_kokoro_tts
        self.__array__ = lambda dtype=None: self._np


class _FakeOutput:
    def __init__(self, audio):
        self.audio = audio


class _FakeResult:
    def __init__(self, audio):
        self.output = _FakeOutput(audio)


class _StubKokoroPipeline:
    """Stand-in for KokoroPipeline. Records calls + yields fake audio."""

    instances: list = []
    calls: list = []

    def __init__(self, lang_code, model, repo_id):
        self.lang_code = lang_code
        self.model = model
        self.repo_id = repo_id
        _StubKokoroPipeline.instances.append(self)

    def __call__(self, text, voice, speed=1.0):
        import numpy as np
        _StubKokoroPipeline.calls.append({
            "text": text, "voice": voice, "speed": speed,
            "lang_code": self.lang_code, "repo_id": self.repo_id,
        })
        # Yield two fake chunks of 0.5 s each at 24 kHz so the WAV is real.
        sr = 24_000
        for _ in range(2):
            samples = np.zeros(sr // 2, dtype=np.float32)
            yield _FakeResult(_FakeMxArray(samples))


def _fake_load_model(_path):
    return MagicMock(name="FakeModel")


def _fake_snapshot_download(repo_id):
    # Return a tmp-like path; the test doesn't actually need files there
    # because _fake_load_model ignores the path.
    return "/tmp/fake-kokoro-snapshot"


def _stub_import_kokoro():
    return _StubKokoroPipeline, _fake_load_model, _fake_snapshot_download


@pytest.fixture(autouse=True)
def _reset_kokoro_cache():
    """Clear the module-level pipeline cache between tests."""
    tts_tool._kokoro_pipeline_cache.clear()
    _StubKokoroPipeline.instances = []
    _StubKokoroPipeline.calls = []
    yield
    tts_tool._kokoro_pipeline_cache.clear()


class TestGenerateKokoroTts:
    def test_writes_wav_with_real_audio_bytes(self, tmp_path, monkeypatch):
        # numpy is the one hard runtime dep we need (not stubbed).
        pytest.importorskip("numpy")

        monkeypatch.setattr(tts_tool, "_import_kokoro", _stub_import_kokoro)

        out_path = str(tmp_path / "out.wav")
        config = {"kokoro": {"voice": "af_bella", "speed": 1.0}}

        result = tts_tool._generate_kokoro_tts("hello", out_path, config)

        assert result == out_path
        assert Path(out_path).exists()
        # 2 chunks * 0.5 s * 24000 Hz * 2 bytes = 48,000 bytes of frames.
        # Plus the WAV header (~44 bytes). Allow some flex.
        size = Path(out_path).stat().st_size
        assert size > 40_000, f"WAV too small: {size} bytes"

        # Verify WAV is structurally valid + correct sample rate
        with wave.open(out_path, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 24_000

    def test_pipeline_cache_reused_across_calls(self, tmp_path, monkeypatch):
        pytest.importorskip("numpy")
        monkeypatch.setattr(tts_tool, "_import_kokoro", _stub_import_kokoro)

        config = {"kokoro": {"voice": "af_bella"}}
        tts_tool._generate_kokoro_tts("one", str(tmp_path / "a.wav"), config)
        tts_tool._generate_kokoro_tts("two", str(tmp_path / "b.wav"), config)

        # Pipeline should have been constructed exactly once for the
        # default (repo, lang) cache key.
        assert len(_StubKokoroPipeline.instances) == 1
        # Both synthesize calls went through.
        assert [c["text"] for c in _StubKokoroPipeline.calls] == ["one", "two"]

    def test_lang_code_change_invalidates_cache(self, tmp_path, monkeypatch):
        """A user switching from American to British English should NOT
        reuse the existing pipeline — different lang_code = different
        misaki phonemizer, different cache key."""
        pytest.importorskip("numpy")
        monkeypatch.setattr(tts_tool, "_import_kokoro", _stub_import_kokoro)

        cfg_a = {"kokoro": {"lang_code": "a", "voice": "af_bella"}}
        cfg_b = {"kokoro": {"lang_code": "b", "voice": "bf_alice"}}
        tts_tool._generate_kokoro_tts("one", str(tmp_path / "a.wav"), cfg_a)
        tts_tool._generate_kokoro_tts("two", str(tmp_path / "b.wav"), cfg_b)

        assert len(_StubKokoroPipeline.instances) == 2
        assert {p.lang_code for p in _StubKokoroPipeline.instances} == {"a", "b"}

    def test_empty_audio_raises_runtime(self, tmp_path, monkeypatch):
        pytest.importorskip("numpy")

        class _EmptyKokoroPipeline(_StubKokoroPipeline):
            def __call__(self, text, voice, speed=1.0):
                # Yield nothing — simulates phonemizer producing no chunks
                # (e.g. all-whitespace input or unknown language).
                if False:
                    yield None  # pragma: no cover

        def stub_import():
            return _EmptyKokoroPipeline, _fake_load_model, _fake_snapshot_download

        monkeypatch.setattr(tts_tool, "_import_kokoro", stub_import)

        with pytest.raises(RuntimeError, match="produced no audio"):
            tts_tool._generate_kokoro_tts(
                "  ", str(tmp_path / "out.wav"), {"kokoro": {"voice": "af_bella"}}
            )

    def test_speed_and_voice_threaded_through(self, tmp_path, monkeypatch):
        pytest.importorskip("numpy")
        monkeypatch.setattr(tts_tool, "_import_kokoro", _stub_import_kokoro)

        config = {"kokoro": {"voice": "am_michael", "speed": 1.3}}
        tts_tool._generate_kokoro_tts("hi", str(tmp_path / "out.wav"), config)

        assert _StubKokoroPipeline.calls[0]["voice"] == "am_michael"
        assert _StubKokoroPipeline.calls[0]["speed"] == 1.3

    def test_default_config_when_kokoro_section_absent(self, tmp_path, monkeypatch):
        pytest.importorskip("numpy")
        monkeypatch.setattr(tts_tool, "_import_kokoro", _stub_import_kokoro)

        # No ``kokoro`` block → defaults must apply.
        tts_tool._generate_kokoro_tts("hi", str(tmp_path / "out.wav"), {})

        call = _StubKokoroPipeline.calls[0]
        assert call["voice"] == DEFAULT_KOKORO_VOICE
        assert call["speed"] == 1.0
        assert call["lang_code"] == DEFAULT_KOKORO_LANG_CODE
        assert call["repo_id"] == DEFAULT_KOKORO_REPO


# ---------------------------------------------------------------------------
# text_to_speech_tool end-to-end (provider == "kokoro")
# ---------------------------------------------------------------------------

class TestTextToSpeechToolWithKokoro:
    def test_dispatches_to_kokoro(self, tmp_path, monkeypatch):
        pytest.importorskip("numpy")
        monkeypatch.setattr(tts_tool, "_import_kokoro", _stub_import_kokoro)

        cfg = {"provider": "kokoro", "kokoro": {"voice": "af_bella"}}
        monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: cfg)

        result = text_to_speech_tool(
            text="hi", output_path=str(tmp_path / "clip.wav")
        )
        data = json.loads(result)

        assert data["success"] is True, data
        assert data["provider"] == "kokoro"
        assert Path(data["file_path"]).exists()

    def test_missing_package_surfaces_error(self, tmp_path, monkeypatch):
        def raise_import():
            raise ImportError("No module named 'mlx_audio'")

        monkeypatch.setattr(tts_tool, "_import_kokoro", raise_import)

        cfg = {"provider": "kokoro"}
        monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: cfg)

        result = text_to_speech_tool(
            text="hi", output_path=str(tmp_path / "clip.wav")
        )
        data = json.loads(result)

        assert data["success"] is False
        assert "mlx-audio" in data["error"] or "kokoro" in data["error"].lower()
        assert "hermes-agent[kokoro]" in data["error"]


# ---------------------------------------------------------------------------
# check_tts_requirements
# ---------------------------------------------------------------------------

class TestCheckTtsRequirementsKokoro:
    def test_kokoro_install_satisfies_requirements(self, monkeypatch):
        # Drop every other provider so we can isolate the kokoro signal.
        monkeypatch.setattr(
            tts_tool, "_import_edge_tts",
            lambda: (_ for _ in ()).throw(ImportError()),
        )
        monkeypatch.setattr(
            tts_tool, "_import_elevenlabs",
            lambda: (_ for _ in ()).throw(ImportError()),
        )
        monkeypatch.setattr(
            tts_tool, "_import_openai_client",
            lambda: (_ for _ in ()).throw(ImportError()),
        )
        monkeypatch.setattr(
            tts_tool, "_import_mistral_client",
            lambda: (_ for _ in ()).throw(ImportError()),
        )
        monkeypatch.setattr(tts_tool, "_check_neutts_available", lambda: False)
        monkeypatch.setattr(tts_tool, "_check_kittentts_available", lambda: False)
        monkeypatch.setattr(tts_tool, "_check_piper_available", lambda: False)
        monkeypatch.setattr(
            tts_tool, "_has_any_command_tts_provider", lambda: False
        )
        monkeypatch.setattr(tts_tool, "_has_openai_audio_backend", lambda: False)
        for env in ("MINIMAX_API_KEY", "XAI_API_KEY", "GEMINI_API_KEY",
                    "GOOGLE_API_KEY", "MISTRAL_API_KEY", "ELEVENLABS_API_KEY"):
            monkeypatch.delenv(env, raising=False)

        # Now toggle the kokoro check on and off.
        monkeypatch.setattr(tts_tool, "_check_kokoro_available", lambda: False)
        assert check_tts_requirements() is False

        monkeypatch.setattr(tts_tool, "_check_kokoro_available", lambda: True)
        assert check_tts_requirements() is True
