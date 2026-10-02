"""Tests for the KittenTTS local provider in tools/tts_tool.py."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("HERMES_SESSION_PLATFORM",):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def clear_kittentts_cache():
    """Reset the module-level model cache between tests."""
    from tools import tts_tool_local as _tt
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






class TestDispatcherBranch:
    def test_kittentts_not_installed_returns_helpful_error(self, monkeypatch, tmp_path):
        """When provider=kittentts but package missing, return JSON error with setup hint."""
        import sys
        monkeypatch.setitem(sys.modules, "kittentts", None)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from tools.tts_tool import text_to_speech_tool

        # Write a config telling it to use kittentts
        import hermes_yaml as yaml
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"tts": {"provider": "kittentts"}})
        )

        result = json.loads(text_to_speech_tool(text="Hello"))
        assert result["success"] is False
        assert "kittentts" in result["error"].lower()


class TestShortenEspeakDataPath:
    """espeak-ng 1.52 keeps the data dir in path_home[160]; a longer path is truncated and
    espeak_Initialize exit(1)s the backend (#131776). The fix links it short before the
    kittentts import that fixes phonemizer's data path."""

    @staticmethod
    def _fake_loader(data_path: str):
        from types import SimpleNamespace
        return SimpleNamespace(get_data_path=lambda: data_path)

    def test_short_path_left_alone(self, monkeypatch, tmp_path):
        import sys
        data_path = str(tmp_path / "ng-data")
        loader = self._fake_loader(data_path)
        monkeypatch.setitem(sys.modules, "espeakng_loader", loader)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))

        from tools.tts_tool_local import _shorten_espeak_data_path
        _shorten_espeak_data_path()

        assert not (tmp_path / "home" / "cache").exists()  # nothing linked
        assert loader.get_data_path() == data_path  # nothing patched

    def test_long_path_swapped_for_short_symlink(self, monkeypatch, tmp_path):
        import sys
        real = tmp_path / ("x" * 170) / "espeak-ng-data"
        real.mkdir(parents=True)
        loader = self._fake_loader(str(real))
        monkeypatch.setitem(sys.modules, "espeakng_loader", loader)
        home = tmp_path / "home"
        monkeypatch.setenv("HERMES_HOME", str(home))

        from tools.tts_tool_local import _shorten_espeak_data_path
        _shorten_espeak_data_path()

        link = home / "cache" / "espeak-ng-data"
        assert link.is_symlink()
        assert Path(os.readlink(link)) == Path(str(real))
        assert loader.get_data_path() == str(link)  # patched: misaki imports read this
        assert len(os.fsencode(loader.get_data_path())) < 160

    def test_relink_when_symlink_points_elsewhere(self, monkeypatch, tmp_path):
        import sys
        stale = tmp_path / "stale-data"
        stale.mkdir()
        real = tmp_path / ("y" * 170) / "espeak-ng-data"
        real.mkdir(parents=True)
        loader = self._fake_loader(str(real))
        monkeypatch.setitem(sys.modules, "espeakng_loader", loader)
        home = tmp_path / "home"
        monkeypatch.setenv("HERMES_HOME", str(home))
        link = home / "cache" / "espeak-ng-data"
        link.parent.mkdir(parents=True)
        link.symlink_to(stale)

        from tools.tts_tool_local import _shorten_espeak_data_path
        _shorten_espeak_data_path()

        assert Path(os.readlink(link)) == Path(str(real))

    def test_second_call_is_idempotent(self, monkeypatch, tmp_path):
        import sys
        real = tmp_path / ("z" * 170) / "espeak-ng-data"
        real.mkdir(parents=True)
        loader = self._fake_loader(str(real))
        monkeypatch.setitem(sys.modules, "espeakng_loader", loader)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))

        from tools.tts_tool_local import _shorten_espeak_data_path
        _shorten_espeak_data_path()
        patched = loader.get_data_path()
        _shorten_espeak_data_path()  # patched value is short -> early return

        assert loader.get_data_path() == patched

    def test_real_file_at_link_target_is_not_clobbered(self, monkeypatch, tmp_path):
        import sys
        real = tmp_path / ("w" * 170) / "espeak-ng-data"
        real.mkdir(parents=True)
        loader = self._fake_loader(str(real))
        monkeypatch.setitem(sys.modules, "espeakng_loader", loader)
        home = tmp_path / "home"
        monkeypatch.setenv("HERMES_HOME", str(home))
        link = home / "cache" / "espeak-ng-data"
        link.parent.mkdir(parents=True)
        link.write_text("user data, keep it")

        from tools.tts_tool_local import _shorten_espeak_data_path
        _shorten_espeak_data_path()

        assert link.read_text() == "user data, keep it"
        assert loader.get_data_path() == str(real)  # not patched

    def test_missing_loader_is_a_noop(self, monkeypatch, tmp_path):
        import sys
        monkeypatch.setitem(sys.modules, "espeakng_loader", None)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from tools.tts_tool_local import _shorten_espeak_data_path
        _shorten_espeak_data_path()  # must not raise

    def test_generate_swaps_path_before_model_load(self, monkeypatch, tmp_path, mock_kittentts_module):
        """The swap must happen before KittenTTS is instantiated: misaki fixes phonemizer's
        data path at import time, after which no environment value can shorten it."""
        import sys
        real = tmp_path / ("v" * 170) / "espeak-ng-data"
        real.mkdir(parents=True)
        loader = self._fake_loader(str(real))
        monkeypatch.setitem(sys.modules, "espeakng_loader", loader)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))

        from tools.tts_tool import _generate_kittentts
        _generate_kittentts("Hello", str(tmp_path / "out.wav"), {})

        fake_cls = mock_kittentts_module[1]
        assert fake_cls.call_count == 1
        assert loader.get_data_path() != str(real)  # patched before the model import chain ran
        assert (tmp_path / "home" / "cache" / "espeak-ng-data").is_symlink()
