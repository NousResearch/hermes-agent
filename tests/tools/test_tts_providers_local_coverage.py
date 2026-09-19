"""Tests for the local TTS providers (piper, kittentts, neutts) in tools/tts_tool.py.

This file covers the local-generator cluster of ``tools/tts_tool.py``:
``_check_neutts_available``, ``_check_kittentts_available``,
``_default_neutts_ref_audio`` / ``_default_neutts_ref_text``,
``_generate_neutts``, the LRU cache helper ``_tts_cache_get_or_load``,
``_check_piper_available``, ``_get_piper_voices_dir``,
``_resolve_piper_voice_path``, ``_generate_piper_tts`` (incl. its nested
``_load_piper_voice``) and ``_generate_kittentts`` (incl. ``_load_kittentts_model``).

All SDK imports are faked via ``sys.modules`` injection — no network, no
multi-GB model downloads. Synthesis output is stubbed to write real bytes so
downstream file-path assertions hold.
"""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools import tts_tool


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_local_caches():
    """Reset the per-process local model/voice caches between tests.

    Both generators key their module-level caches on model/voice identity,
    which would otherwise bleed state across tests in the same interpreter.
    """
    tts_tool._piper_voice_cache.clear()
    tts_tool._kittentts_model_cache.clear()
    yield
    tts_tool._piper_voice_cache.clear()
    tts_tool._kittentts_model_cache.clear()


@pytest.fixture
def mock_piper_module():
    """Inject a fake ``piper`` module with ``PiperVoice`` + ``SynthesisConfig``.

    ``PiperVoice.load(...)`` returns a voice stub whose ``synthesize_wav``
    writes a short, valid WAV frame set so ``wave.open`` closes cleanly and the
    produced file is non-empty.
    """
    fake_voice = MagicMock()

    def _synthesize(text, wav_file, syn_config=None):
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22050)
        wav_file.writeframes(b"\x00\x00" * 1024)

    fake_voice.synthesize_wav.side_effect = _synthesize

    fake_voice_cls = MagicMock()  # stand-in for piper.PiperVoice
    fake_voice_cls.load.return_value = fake_voice

    fake_syn_cls = MagicMock(return_value=MagicMock())  # piper.SynthesisConfig

    fake_piper = types.SimpleNamespace(
        PiperVoice=fake_voice_cls,
        SynthesisConfig=fake_syn_cls,
    )
    with patch.dict("sys.modules", {"piper": fake_piper}):
        yield fake_piper


@pytest.fixture
def mock_kittentts_module():
    """Inject a fake ``kittentts`` + ``soundfile`` module.

    ``KittenTTS(model_name)`` returns a model stub whose ``generate`` returns a
    1D sequence; ``soundfile.write`` writes real bytes so the target path
    materializes.
    """
    fake_model = MagicMock()
    fake_model.generate.return_value = [0.0] * 48000  # ~2s @ 24kHz

    fake_model_cls = MagicMock(return_value=fake_model)

    fake_kittentts = MagicMock()
    fake_kittentts.KittenTTS = fake_model_cls

    fake_sf = MagicMock()

    def _fake_write(path, audio, samplerate):
        Path(path).write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt fake")

    fake_sf.write = _fake_write

    with patch.dict("sys.modules", {"kittentts": fake_kittentts, "soundfile": fake_sf}):
        yield fake_model, fake_model_cls


def _prepare_piper_files(tmp_path, voice):
    """Create the (.onnx + .onnx.json) pair so the resolver finds a cached voice."""
    model = tmp_path / f"{voice}.onnx"
    model.write_bytes(b"model")
    (tmp_path / f"{voice}.onnx.json").write_text("{}")
    return model


# ---------------------------------------------------------------------------
# Availability probes
# ---------------------------------------------------------------------------


class TestCheckLocalProvidersAvailable:
    @pytest.mark.parametrize(
        "probe, pkg",
        [
            ("_check_neutts_available", "neutts"),
            ("_check_kittentts_available", "kittentts"),
            ("_check_piper_available", "piper"),
        ],
    )
    def test_true_when_package_present(self, probe, pkg):
        with patch.object(
            importlib.util, "find_spec", lambda name: MagicMock() if name == pkg else None
        ):
            assert getattr(tts_tool, probe)() is True

    @pytest.mark.parametrize(
        "probe",
        ["_check_neutts_available", "_check_kittentts_available", "_check_piper_available"],
    )
    def test_false_when_package_missing(self, probe):
        with patch.object(importlib.util, "find_spec", lambda name: None):
            assert getattr(tts_tool, probe)() is False

    @pytest.mark.parametrize(
        "probe",
        ["_check_neutts_available", "_check_kittentts_available", "_check_piper_available"],
    )
    def test_false_when_probe_raises(self, probe):
        def _boom(name):
            raise RuntimeError("probe exploded")

        with patch.object(importlib.util, "find_spec", side_effect=_boom):
            assert getattr(tts_tool, probe)() is False


# ---------------------------------------------------------------------------
# Default NeuTTS reference assets
# ---------------------------------------------------------------------------


class TestDefaultNeuttsRefs:
    def test_default_ref_audio_path(self):
        ref = tts_tool._default_neutts_ref_audio()
        assert isinstance(ref, str)
        assert Path(ref).is_absolute()
        assert ref.endswith("jo.wav")

    def test_default_ref_text_path(self):
        ref = tts_tool._default_neutts_ref_text()
        assert isinstance(ref, str)
        assert Path(ref).is_absolute()
        assert ref.endswith("jo.txt")


# ---------------------------------------------------------------------------
# _generate_neutts
# ---------------------------------------------------------------------------


class TestGenerateNeutts:
    def test_success_wav_uses_defaults(self, tmp_path, monkeypatch):
        """WAV output path → no conversion; defaults flow straight into the cmd."""
        output_path = str(tmp_path / "out.wav")
        mock_run = MagicMock(return_value=MagicMock(returncode=0, stderr="OK: done"))
        monkeypatch.setattr(tts_tool.subprocess, "run", mock_run)

        result = tts_tool._generate_neutts("Hello", output_path, {})

        assert result == output_path
        cmd = mock_run.call_args.args[0]
        assert cmd[0] == sys.executable
        args = list(zip(cmd[::2], cmd[1::2]))
        opts = dict(args)
        assert opts["--text"] == "Hello"
        assert opts["--out"] == output_path
        assert opts["--ref-audio"] == tts_tool._default_neutts_ref_audio()
        assert opts["--ref-text"] == tts_tool._default_neutts_ref_text()
        assert opts["--model"] == "neuphonic/neutts-air-q4-gguf"
        assert opts["--device"] == "cpu"

    def test_config_overrides_flow_to_cmd(self, tmp_path, monkeypatch):
        output_path = str(tmp_path / "clip.wav")
        mock_run = MagicMock(return_value=MagicMock(returncode=0, stderr="OK: done"))
        monkeypatch.setattr(tts_tool.subprocess, "run", mock_run)

        config = {
            "neutts": {
                "ref_audio": "/samples/me.wav",
                "ref_text": "/samples/me.txt",
                "model": "neuphonic/neutts-air-base",
                "device": "cuda",
            }
        }
        tts_tool._generate_neutts("Hi", output_path, config)

        cmd = mock_run.call_args.args[0]
        opts = dict(zip(cmd[::2], cmd[1::2]))
        assert opts["--ref-audio"] == "/samples/me.wav"
        assert opts["--ref-text"] == "/samples/me.txt"
        assert opts["--model"] == "neuphonic/neutts-air-base"
        assert opts["--device"] == "cuda"

    def test_synth_failure_raises_filtering_ok_lines(self, tmp_path, monkeypatch):
        mock_run = MagicMock(
            return_value=MagicMock(returncode=1, stderr="OK: partial\nsynthesis exploded mid-way")
        )
        monkeypatch.setattr(tts_tool.subprocess, "run", mock_run)

        with pytest.raises(RuntimeError, match="NeuTTS synthesis failed") as exc_info:
            tts_tool._generate_neutts("Hi", str(tmp_path / "out.wav"), {})

        msg = str(exc_info.value)
        assert "synthesis exploded mid-way" in msg
        assert "OK:" not in msg

    def test_non_wav_with_ffmpeg_converts(self, tmp_path, monkeypatch):
        output_path = str(tmp_path / "out.mp3")
        mock_run = MagicMock(return_value=MagicMock(returncode=0, stderr="OK: done"))
        monkeypatch.setattr(tts_tool.subprocess, "run", mock_run)
        monkeypatch.setattr(tts_tool.shutil, "which", lambda name: "/usr/bin/ffmpeg")
        monkeypatch.setattr(tts_tool.os, "remove", MagicMock())

        result = tts_tool._generate_neutts("Hi", output_path, {})

        assert result == output_path
        # first call = synthesis, second = ffmpeg conversion
        assert mock_run.call_count == 2
        conv_cmd = mock_run.call_args_list[1].args[0]
        assert conv_cmd[0] == "/usr/bin/ffmpeg"
        assert conv_cmd[-1] == output_path
        tts_tool.os.remove.assert_called_once()

    def test_non_wav_without_ffmpeg_renames(self, tmp_path, monkeypatch):
        output_path = str(tmp_path / "out.ogg")
        mock_run = MagicMock(return_value=MagicMock(returncode=0, stderr="OK: done"))
        monkeypatch.setattr(tts_tool.subprocess, "run", mock_run)
        monkeypatch.setattr(tts_tool.shutil, "which", lambda name: None)
        monkeypatch.setattr(tts_tool.os, "rename", MagicMock())

        result = tts_tool._generate_neutts("Hi", output_path, {})

        assert result == output_path
        mock_run.assert_called_once()  # synthesis only; no conversion call
        assert tts_tool.os.rename.call_count == 1
        renamed_from = tts_tool.os.rename.call_args.args[0]
        assert renamed_from.endswith(".wav")


# ---------------------------------------------------------------------------
# LRU cache helper
# ---------------------------------------------------------------------------


class TestTtsCacheGetOrLoad:
    def test_miss_loads_and_caches(self):
        cache = {}
        loaded = []

        def load():
            loaded.append(1)
            return "value"

        assert tts_tool._tts_cache_get_or_load(cache, "k", load) == "value"
        assert cache["k"] == "value"
        assert len(loaded) == 1

    def test_hit_returns_cached_without_reload(self):
        cache = {"k": "cached"}
        loaded = []

        def load():
            loaded.append(1)
            return "fresh"

        assert tts_tool._tts_cache_get_or_load(cache, "k", load) == "cached"
        assert loaded == []

    def test_over_cap_evicts_lru(self):
        cache = {}
        calls = []

        def make(value):
            def load():
                calls.append(value)
                return value
            return load

        tts_tool._tts_cache_get_or_load(cache, "a", make("a"))
        tts_tool._tts_cache_get_or_load(cache, "b", make("b"))
        tts_tool._tts_cache_get_or_load(cache, "c", make("c"))
        tts_tool._tts_cache_get_or_load(cache, "d", make("d"))

        # cap is 3 → "a" (oldest) must be evicted
        assert set(cache) == {"b", "c", "d"}

    def test_hit_refreshes_recency(self):
        cache = {}
        calls = []

        def make(value):
            def load():
                calls.append(value)
                return value
            return load

        tts_tool._tts_cache_get_or_load(cache, "a", make("a"))
        tts_tool._tts_cache_get_or_load(cache, "b", make("b"))
        tts_tool._tts_cache_get_or_load(cache, "c", make("c"))

        # touch "a" → order becomes b, c, a
        assert tts_tool._tts_cache_get_or_load(cache, "a", make("a")) == "a"
        # inserting "d" should evict "b", not "a"
        tts_tool._tts_cache_get_or_load(cache, "d", make("d"))
        assert set(cache) == {"a", "c", "d"}
        assert cache["a"] == "a"


# ---------------------------------------------------------------------------
# _get_piper_voices_dir
# ---------------------------------------------------------------------------


class TestGetPiperVoicesDir:
    def test_resolves_and_creates_directory(self, tmp_path, monkeypatch):
        expected = tmp_path / "piper-voices"
        monkeypatch.setattr("hermes_constants.get_hermes_dir", lambda *a: str(expected))

        result = tts_tool._get_piper_voices_dir()

        assert result == expected
        assert expected.is_dir()


# ---------------------------------------------------------------------------
# _resolve_piper_voice_path
# ---------------------------------------------------------------------------


class TestResolvePiperVoicePath:
    def test_direct_onnx_file_returned(self, tmp_path):
        model = tmp_path / "custom.onnx"
        model.write_bytes(b"onnx")
        assert tts_tool._resolve_piper_voice_path(str(model), tmp_path) == str(model)

    def test_cached_voice_name_returned(self, tmp_path):
        model = _prepare_piper_files(tmp_path, "en_US-lessac-medium")
        result = tts_tool._resolve_piper_voice_path("en_US-lessac-medium", tmp_path)
        assert result == str(model)

    def test_empty_voice_uses_default_name(self, tmp_path):
        model = _prepare_piper_files(tmp_path, tts_tool.DEFAULT_PIPER_VOICE)
        result = tts_tool._resolve_piper_voice_path("", tmp_path)
        assert result == str(model)

    def test_download_success(self, tmp_path, monkeypatch):
        # subprocess.run is used as the downloader; simulate it writing the
        # .onnx + .onnx.json so the post-download existence check passes.
        def _fake_download(*a, **k):
            _prepare_piper_files(tmp_path, "en_US-lessac-medium")
            return MagicMock(returncode=0, stderr="")

        monkeypatch.setattr(tts_tool.subprocess, "run", _fake_download)

        result = tts_tool._resolve_piper_voice_path("en_US-lessac-medium", tmp_path)
        assert result.endswith("en_US-lessac-medium.onnx")

    def test_download_failure_raises(self, tmp_path, monkeypatch):
        mock_run = MagicMock(return_value=MagicMock(returncode=1, stderr="boom"))
        monkeypatch.setattr(tts_tool.subprocess, "run", mock_run)

        with pytest.raises(RuntimeError, match="download failed") as exc_info:
            tts_tool._resolve_piper_voice_path("en_US-lessac-medium", tmp_path)
        assert "boom" in str(exc_info.value)

    def test_download_timeout_raises(self, tmp_path, monkeypatch):
        def _timeout(*a, **k):
            raise tts_tool.subprocess.TimeoutExpired(cmd="", timeout=300)

        monkeypatch.setattr(tts_tool.subprocess, "run", _timeout)

        with pytest.raises(RuntimeError, match="timed out after 300s"):
            tts_tool._resolve_piper_voice_path("en_US-lessac-medium", tmp_path)

    def test_download_completed_but_file_missing_raises(self, tmp_path, monkeypatch):
        mock_run = MagicMock(return_value=MagicMock(returncode=0, stderr=""))
        monkeypatch.setattr(tts_tool.subprocess, "run", mock_run)

        with pytest.raises(RuntimeError, match="missing"):
            tts_tool._resolve_piper_voice_path("en_US-lessac-medium", tmp_path)


# ---------------------------------------------------------------------------
# _generate_piper_tts
# ---------------------------------------------------------------------------


class TestGeneratePiperTts:
    def test_loads_voice_and_writes_wav(self, tmp_path, mock_piper_module):
        model = _prepare_piper_files(tmp_path, "amy")
        output_path = str(tmp_path / "out.wav")
        config = {"piper": {"voice": str(model), "voices_dir": str(tmp_path)}}

        result = tts_tool._generate_piper_tts("hello", output_path, config)

        assert result == output_path
        assert Path(output_path).exists()
        assert Path(output_path).stat().st_size > 0
        mock_piper_module.PiperVoice.load.assert_called_once_with(str(model), use_cuda=False)
        fake_voice = mock_piper_module.PiperVoice.load.return_value
        fake_voice.synthesize_wav.assert_called()
        # No advanced knobs → synthesize_wav called WITHOUT syn_config
        assert "syn_config" not in fake_voice.synthesize_wav.call_args.kwargs

    def test_advanced_knobs_build_synconfig(self, tmp_path, mock_piper_module):
        model = _prepare_piper_files(tmp_path, "amy")
        config = {
            "piper": {
                "voice": str(model),
                "voices_dir": str(tmp_path),
                "length_scale": 1.2,
                "noise_scale": 0.5,
                "volume": 1.5,
                "speaker_id": 7,
            }
        }

        tts_tool._generate_piper_tts("hi", str(tmp_path / "out.wav"), config)

        fake_syn = mock_piper_module.SynthesisConfig
        fake_syn.assert_called_once()
        kwargs = fake_syn.call_args.kwargs
        assert kwargs["length_scale"] == 1.2
        assert kwargs["noise_scale"] == 0.5
        assert kwargs["volume"] == 1.5
        assert kwargs["speaker_id"] == 7

        fake_voice = mock_piper_module.PiperVoice.load.return_value
        assert "syn_config" in fake_voice.synthesize_wav.call_args.kwargs

    def test_boolean_speaker_id_coerced_to_zero(self, tmp_path, mock_piper_module):
        model = _prepare_piper_files(tmp_path, "amy")
        config = {
            "piper": {
                "voice": str(model),
                "voices_dir": str(tmp_path),
                "speaker_id": True,  # must be dropped → 0
            }
        }

        tts_tool._generate_piper_tts("hi", str(tmp_path / "out.wav"), config)

        kwargs = mock_piper_module.SynthesisConfig.call_args.kwargs
        assert kwargs["speaker_id"] == 0

    def test_voice_cache_reused_across_calls(self, tmp_path, mock_piper_module):
        model = _prepare_piper_files(tmp_path, "amy")
        config = {"piper": {"voice": str(model), "voices_dir": str(tmp_path)}}

        tts_tool._generate_piper_tts("one", str(tmp_path / "a.wav"), config)
        tts_tool._generate_piper_tts("two", str(tmp_path / "b.wav"), config)

        assert mock_piper_module.PiperVoice.load.call_count == 1

    def test_non_wav_with_ffmpeg_converts(self, tmp_path, mock_piper_module, monkeypatch):
        model = _prepare_piper_files(tmp_path, "amy")
        output_path = str(tmp_path / "out.mp3")
        config = {"piper": {"voice": str(model), "voices_dir": str(tmp_path)}}

        mock_run = MagicMock()
        monkeypatch.setattr(tts_tool.subprocess, "run", mock_run)
        monkeypatch.setattr(tts_tool.shutil, "which", lambda name: "/usr/bin/ffmpeg")
        monkeypatch.setattr(tts_tool.os, "remove", MagicMock())

        result = tts_tool._generate_piper_tts("hi", output_path, config)

        assert result == output_path
        assert mock_run.call_count == 1
        conv_cmd = mock_run.call_args.args[0]
        assert conv_cmd[0] == "/usr/bin/ffmpeg"
        assert conv_cmd[-1] == output_path
        tts_tool.os.remove.assert_called_once()

    def test_non_wav_without_ffmpeg_renames(self, tmp_path, mock_piper_module, monkeypatch):
        model = _prepare_piper_files(tmp_path, "amy")
        output_path = str(tmp_path / "out.ogg")
        config = {"piper": {"voice": str(model), "voices_dir": str(tmp_path)}}

        monkeypatch.setattr(tts_tool.shutil, "which", lambda name: None)

        result = tts_tool._generate_piper_tts("hi", output_path, config)

        assert result == output_path
        assert Path(output_path).exists()  # wav was renamed to the requested path

    def test_non_wav_ffmpeg_remove_oserror_suppressed(self, tmp_path, mock_piper_module, monkeypatch):
        """os.remove() failing on the temp WAV must not abort the generation."""
        model = _prepare_piper_files(tmp_path, "amy")
        output_path = str(tmp_path / "out.mp3")
        config = {"piper": {"voice": str(model), "voices_dir": str(tmp_path)}}

        monkeypatch.setattr(tts_tool.subprocess, "run", MagicMock())
        monkeypatch.setattr(tts_tool.shutil, "which", lambda name: "/usr/bin/ffmpeg")
        monkeypatch.setattr(tts_tool.os, "remove", MagicMock(side_effect=OSError("busy")))

        result = tts_tool._generate_piper_tts("hi", output_path, config)

        assert result == output_path

    def test_resolve_error_propagates(self, tmp_path, mock_piper_module, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("voice not found")

        monkeypatch.setattr(tts_tool, "_resolve_piper_voice_path", _boom)

        with pytest.raises(RuntimeError, match="voice not found"):
            tts_tool._generate_piper_tts(
                "hi", str(tmp_path / "out.wav"),
                {"piper": {"voice": "ghost", "voices_dir": str(tmp_path)}},
            )

    def test_missing_package_raises_import_error(self, tmp_path):
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setitem(sys.modules, "piper", None)
        try:
            with pytest.raises((ImportError, TypeError)):
                tts_tool._generate_piper_tts(
                    "hi", str(tmp_path / "out.wav"),
                    {"piper": {"voices_dir": str(tmp_path)}},
                )
        finally:
            monkeypatch.undo()


# ---------------------------------------------------------------------------
# _generate_kittentts
# ---------------------------------------------------------------------------


class TestGenerateKittenTts:
    def test_successful_wav_generation(self, tmp_path, mock_kittentts_module):
        fake_model, fake_cls = mock_kittentts_module
        output_path = str(tmp_path / "test.wav")

        result = tts_tool._generate_kittentts("Hello world", output_path, {})

        assert result == output_path
        assert (tmp_path / "test.wav").exists()
        fake_cls.assert_called_once()
        fake_model.generate.assert_called_once_with(
            "Hello world", voice=tts_tool.DEFAULT_KITTENTTS_VOICE,
            speed=1.0, clean_text=True,
        )

    def test_config_overrides_voice_speed_cleantext(self, tmp_path, mock_kittentts_module):
        fake_model, _ = mock_kittentts_module
        config = {
            "kittentts": {
                "model": "KittenML/kitten-tts-mini-0.8",
                "voice": "Luna",
                "speed": 1.25,
                "clean_text": False,
            }
        }

        tts_tool._generate_kittentts("Hi", str(tmp_path / "out.wav"), config)

        call_kwargs = fake_model.generate.call_args.kwargs
        assert call_kwargs["voice"] == "Luna"
        assert call_kwargs["speed"] == 1.25
        assert call_kwargs["clean_text"] is False

    def test_model_cached_across_calls(self, tmp_path, mock_kittentts_module):
        fake_model, fake_cls = mock_kittentts_module

        tts_tool._generate_kittentts("one", str(tmp_path / "a.wav"), {})
        tts_tool._generate_kittentts("two", str(tmp_path / "b.wav"), {})

        # same default model key → constructor called only once
        assert fake_cls.call_count == 1

    def test_non_wav_with_ffmpeg_converts(self, tmp_path, mock_kittentts_module, monkeypatch):
        fake_model, _ = mock_kittentts_module
        output_path = str(tmp_path / "out.mp3")

        mock_run = MagicMock()
        monkeypatch.setattr(tts_tool.subprocess, "run", mock_run)
        monkeypatch.setattr(tts_tool.shutil, "which", lambda name: "/usr/bin/ffmpeg")
        monkeypatch.setattr(tts_tool.os, "remove", MagicMock())

        result = tts_tool._generate_kittentts("hi", output_path, {})

        assert result == output_path
        assert mock_run.call_count == 1
        conv_cmd = mock_run.call_args.args[0]
        assert conv_cmd[0] == "/usr/bin/ffmpeg"
        assert conv_cmd[-1] == output_path
        tts_tool.os.remove.assert_called_once()

    def test_non_wav_without_ffmpeg_renames(self, tmp_path, mock_kittentts_module, monkeypatch):
        fake_model, _ = mock_kittentts_module
        output_path = str(tmp_path / "out.ogg")

        monkeypatch.setattr(tts_tool.shutil, "which", lambda name: None)

        result = tts_tool._generate_kittentts("hi", output_path, {})

        assert result == output_path
        assert Path(output_path).exists()

    def test_missing_package_raises_import_error(self, tmp_path):
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setitem(sys.modules, "kittentts", None)
        try:
            with pytest.raises((ImportError, TypeError)):
                tts_tool._generate_kittentts("hi", str(tmp_path / "out.wav"), {})
        finally:
            monkeypatch.undo()
