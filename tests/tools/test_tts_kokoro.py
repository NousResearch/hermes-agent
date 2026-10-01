"""Tests for the native Kokoro local provider (tools/tts_tool_local.py).

The kokoro_onnx / soundfile modules are faked, so no ONNX runtime or network is
needed; the model-download path is pointed at HERMES_HOME fixtures and its
urllib streamer is stubbed. Mirrors tests/tools/test_tts_kittentts.py.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("HERMES_SESSION_PLATFORM",):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def clear_kokoro_cache():
    """Reset the module-level model cache between tests."""
    from tools import tts_tool_local as _tt
    _tt._kokoro_model_cache.clear()
    yield
    _tt._kokoro_model_cache.clear()


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Every test in this module resolves HERMES_HOME (and HOME, so a ``~`` config path can
    never reach the real user home even through a bug) to its own tmp dir — the engine writes
    model assets under <HERMES_HOME>/share/kokoro/, and the conftest home-I/O guard refuses
    any probe against the real home."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))


@pytest.fixture
def seed_model_files(tmp_path):
    """HERMES_HOME sandbox with both model assets pre-placed, so no download runs."""
    from tools.tts_tool_local import _get_kokoro_dir
    kokoro_dir = _get_kokoro_dir()
    (kokoro_dir / "kokoro-v1.0.onnx").write_bytes(b"fake onnx")
    (kokoro_dir / "voices-v1.0.bin").write_bytes(b"fake voices")
    return kokoro_dir


@pytest.fixture
def fake_kokoro_sdk():
    """Inject fake kokoro_onnx + soundfile modules and patch the tts_tool importer seam.
    No model assets are placed — tests that synthesize combine this with seed_model_files."""
    fake_model = MagicMock()
    # (samples, sample_rate) — ~0.2s of silence at 24kHz, kokoro-onnx's create() shape.
    fake_model.create.return_value = ([0.0] * 4800, 24000)
    fake_model.get_voices.return_value = {"af_heart", "af_bella", "am_michael"}
    fake_cls = MagicMock(return_value=fake_model)
    fake_kokoro = MagicMock()
    fake_kokoro.Kokoro = fake_cls

    fake_sf = MagicMock()
    def _fake_write(path, audio, samplerate):
        import pathlib
        pathlib.Path(path).write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt fake")
    fake_sf.write = _fake_write

    with patch.dict("sys.modules", {"kokoro_onnx": fake_kokoro, "soundfile": fake_sf}), \
         patch("tools.tts_tool._import_kokoro_onnx", return_value=fake_cls):
        yield fake_cls, fake_model


@pytest.fixture
def kokoro_env(fake_kokoro_sdk, seed_model_files):
    """Fake SDK plus pre-placed model assets: synthesis without any download I/O."""
    return fake_kokoro_sdk


class TestGenerateKokoro:
    def test_successful_wav_generation(self, tmp_path, kokoro_env):
        from tools.tts_tool_local import _generate_kokoro

        out = str(tmp_path / "out.wav")
        assert _generate_kokoro("Hello world", out, {}) == out
        assert (tmp_path / "out.wav").exists()
        fake_cls, fake_model = kokoro_env
        fake_cls.assert_called_once()
        fake_model.create.assert_called_once()

    def test_default_voice_and_speed(self, tmp_path, kokoro_env):
        from tools.tts_tool_local import _generate_kokoro

        _generate_kokoro("Hi", str(tmp_path / "out.wav"), {})
        kwargs = kokoro_env[1].create.call_args.kwargs
        assert kwargs["voice"] == "af_heart"
        assert kwargs["speed"] == 1.0

    def test_per_call_speed_overrides_subconfig(self, tmp_path, kokoro_env):
        """The tool's per-call speed is staged under ``_call_speed`` and wins over BOTH
        tts.kokoro.speed and the global tts.speed (the documented precedence, and the
        #130693 shadowing shape)."""
        from tools.tts_tool_local import _generate_kokoro

        _generate_kokoro("Hi", str(tmp_path / "out.wav"),
                         {"kokoro": {"speed": 1.5}, "speed": 2.5, "_call_speed": 0.75})
        assert kokoro_env[1].create.call_args.kwargs["speed"] == 0.75

    def test_subconfig_speed_beats_global_speed(self, tmp_path, kokoro_env):
        """tts.kokoro.speed is the provider default; the global tts.speed multiplier only
        applies when the provider block does not set one."""
        from tools.tts_tool_local import _generate_kokoro

        _generate_kokoro("Hi", str(tmp_path / "out.wav"),
                         {"kokoro": {"speed": 1.5}, "speed": 2.5})
        assert kokoro_env[1].create.call_args.kwargs["speed"] == 1.5

    def test_global_speed_used_without_provider_speed(self, tmp_path, kokoro_env):
        from tools.tts_tool_local import _generate_kokoro

        _generate_kokoro("Hi", str(tmp_path / "out.wav"), {"speed": 1.25})
        assert kokoro_env[1].create.call_args.kwargs["speed"] == 1.25

    def test_subconfig_speed_used_without_per_call(self, tmp_path, kokoro_env):
        from tools.tts_tool_local import _generate_kokoro

        _generate_kokoro("Hi", str(tmp_path / "out.wav"), {"kokoro": {"speed": 1.5}})
        assert kokoro_env[1].create.call_args.kwargs["speed"] == 1.5

    def test_speed_clamped_to_kokoro_range(self, tmp_path, kokoro_env):
        from tools.tts_tool_local import _generate_kokoro

        _generate_kokoro("Hi", str(tmp_path / "out.wav"), {"_call_speed": 4.0})
        assert kokoro_env[1].create.call_args.kwargs["speed"] == 2.0

    def test_non_finite_per_call_speed_resets_at_staging(self, tmp_path, kokoro_env):
        """The dispatcher resets a NaN per-call speed to 1.0 before any provider sees it
        (min/max keeps NaN, so the staging clamp alone would propagate it)."""
        from tools.tts_tool import _apply_call_overrides

        staged, _ = _apply_call_overrides({}, float("nan"), None)
        assert staged["_call_speed"] == 1.0
        assert staged["speed"] == 1.0

    def test_non_finite_speed_resets_to_one(self, tmp_path, kokoro_env):
        """A YAML ``.nan`` sails through min/max — it must not reach the engine."""
        from tools.tts_tool_local import _generate_kokoro

        _generate_kokoro("Hi", str(tmp_path / "out.wav"), {"_call_speed": float("nan")})
        assert kokoro_env[1].create.call_args.kwargs["speed"] == 1.0

    def test_non_numeric_speed_resets_to_one(self, tmp_path, kokoro_env):
        """A YAML speed value that won't parse degrades to 1.0, never a raw TypeError."""
        from tools.tts_tool_local import _generate_kokoro

        _generate_kokoro("Hi", str(tmp_path / "out.wav"), {"kokoro": {"speed": "fast"}})
        assert kokoro_env[1].create.call_args.kwargs["speed"] == 1.0

    def test_null_speed_resets_to_one(self, tmp_path, kokoro_env):
        from tools.tts_tool_local import _generate_kokoro

        _generate_kokoro("Hi", str(tmp_path / "out.wav"), {"kokoro": {"speed": None}})
        assert kokoro_env[1].create.call_args.kwargs["speed"] == 1.0

    def test_british_voice_phonemizes_en_gb(self, tmp_path, kokoro_env):
        from tools.tts_tool_local import _generate_kokoro

        _generate_kokoro("Hi", str(tmp_path / "out.wav"), {"kokoro": {"voice": "bf_emma"}})
        assert kokoro_env[1].create.call_args.kwargs["lang"] == "en-gb"

    def test_american_voice_phonemizes_en_us(self, tmp_path, kokoro_env):
        from tools.tts_tool_local import _generate_kokoro

        _generate_kokoro("Hi", str(tmp_path / "out.wav"), {})
        assert kokoro_env[1].create.call_args.kwargs["lang"] == "en-us"

    def test_japanese_voice_phonemizes_ja(self, tmp_path, kokoro_env):
        """Non-English catalog voices phonemize in their own language (j = Japanese)."""
        from tools.tts_tool_local import _generate_kokoro

        _generate_kokoro("こんにちは", str(tmp_path / "out.wav"), {"kokoro": {"voice": "jf_alpha"}})
        assert kokoro_env[1].create.call_args.kwargs["lang"] == "ja"

    def test_voice_prefix_language_table(self):
        """The full catalog prefix map, verified against the espeak backend
        kokoro-onnx bundles (es-es/it-it/... are NOT valid espeak codes)."""
        from tools.tts_tool_local import _kokoro_voice_lang

        assert _kokoro_voice_lang("af_heart") == "en-us"
        assert _kokoro_voice_lang("bf_emma") == "en-gb"
        assert _kokoro_voice_lang("ef_dora") == "es"
        assert _kokoro_voice_lang("ff_siwis") == "fr-fr"
        assert _kokoro_voice_lang("hf_alpha") == "hi"
        assert _kokoro_voice_lang("if_sara") == "it"
        assert _kokoro_voice_lang("jf_alpha") == "ja"
        assert _kokoro_voice_lang("pf_dora") == "pt-br"
        assert _kokoro_voice_lang("zf_xiaobei") == "cmn"
        assert _kokoro_voice_lang("xf_future") == "en-us"  # unknown prefix -> default

    def test_mp3_output_converted_not_wav_renamed(self, tmp_path, kokoro_env):
        """A .mp3 request must produce an mp3 container (ffmpeg), not a renamed WAV."""
        from tools.tts_tool_local import _generate_kokoro

        out = str(tmp_path / "out.mp3")
        with patch("tools.tts_tool_local._finalize_wav_output",
                   side_effect=lambda wav, o: o) as fin:
            _generate_kokoro("Hi", out, {})
        fin.assert_called_once()
        assert fin.call_args[0][1] == out  # second arg is the requested container

    def test_voice_from_config(self, tmp_path, kokoro_env):
        from tools.tts_tool_local import _generate_kokoro

        _generate_kokoro("Hi", str(tmp_path / "out.wav"), {"kokoro": {"voice": "bf_emma"}})
        assert kokoro_env[1].create.call_args.kwargs["voice"] == "bf_emma"

    def test_synthesis_failure_lists_known_voices(self, tmp_path, kokoro_env):
        from tools.tts_tool_local import _generate_kokoro

        fake_cls, fake_model = kokoro_env
        fake_model.create.side_effect = ValueError("unknown voice")
        with pytest.raises(RuntimeError) as excinfo:
            _generate_kokoro("Hi", str(tmp_path / "out.wav"), {})
        assert "af_heart" in str(excinfo.value)
        assert "VOICES.md" in str(excinfo.value)

    def test_model_cached_across_calls(self, tmp_path, kokoro_env):
        from tools.tts_tool_local import _generate_kokoro

        _generate_kokoro("One", str(tmp_path / "a.wav"), {})
        _generate_kokoro("Two", str(tmp_path / "b.wav"), {})
        kokoro_env[0].assert_called_once()  # engine constructed once


class TestKokoroModelResolution:
    def test_missing_model_downloads_then_loads(self, tmp_path, fake_kokoro_sdk):
        """No cached assets: both URLs are fetched, then the engine loads."""
        fetched = []

        def fake_download(url, dest):
            fetched.append(url)
            dest.write_bytes(b"fetched")

        with patch("tools.tts_tool_local._kokoro_download", side_effect=fake_download):
            from tools.tts_tool_local import _load_kokoro_model_for_config
            _load_kokoro_model_for_config({})

        from tools.tts_tool_local import _KOKORO_MODEL_URL, _KOKORO_VOICES_URL
        assert fetched == [_KOKORO_MODEL_URL, _KOKORO_VOICES_URL]
        assert (tmp_path / "share/kokoro/kokoro-v1.0.onnx").exists()

    def test_explicit_paths_skip_share_dir(self, tmp_path, fake_kokoro_sdk):
        model = tmp_path / "custom.onnx"
        model.write_bytes(b"m")
        voices = tmp_path / "custom.bin"
        voices.write_bytes(b"v")
        with patch("tools.tts_tool_local._kokoro_download") as dl:
            from tools.tts_tool_local import _load_kokoro_model_for_config
            _load_kokoro_model_for_config(
                {"kokoro": {"model_path": str(model), "voices_path": str(voices)}})
        dl.assert_not_called()

    def test_download_failure_raises_manual_recipe(self, tmp_path, monkeypatch):
        """Missing SDK raises the install hint before any path/download work (a missing
        kokoro-onnx extra must not trigger a ~350MB download first)."""
        monkeypatch.setitem(sys.modules, "kokoro_onnx", None)
        with patch("tools.tts_tool_local._kokoro_download") as dl, \
             pytest.raises((RuntimeError, ImportError), match="kokoro") as excinfo:
            from tools.tts_tool_local import _load_kokoro_model_for_config
            _load_kokoro_model_for_config({})
        dl.assert_not_called()
        assert not (tmp_path / "share/kokoro/kokoro-v1.0.onnx").exists()

    def test_configured_path_missing_never_downloads(self, tmp_path):
        """An explicit tts.kokoro.model_path that doesn't exist is an installation mistake —
        raise the manual recipe instead of quietly downloading ~350MB into it."""
        from tools.tts_tool_local import _resolve_kokoro_paths
        model = tmp_path / "missing.onnx"
        with patch("tools.tts_tool_local._kokoro_download") as dl, \
             pytest.raises(RuntimeError, match="not found at the configured path"):
            _resolve_kokoro_paths({"model_path": str(model)})
        dl.assert_not_called()

    def test_mixed_config_one_key_set_other_downloads(self, tmp_path):
        """The two path keys are independent: with only model_path configured (valid), the
        unset voices_path still downloads into the shared cache — the pair-level guard of an
        earlier draft wrongly bricked this shape."""
        from tools.tts_tool_local import _resolve_kokoro_paths
        model = tmp_path / "custom.onnx"
        model.write_bytes(b"m")

        def fake_download(url, dest):
            dest.write_bytes(b"v")

        with patch("tools.tts_tool_local._kokoro_download", side_effect=fake_download) as dl:
            model_path, voices_path = _resolve_kokoro_paths({"model_path": str(model)})
        assert model_path == model
        assert voices_path == tmp_path / "share/kokoro/voices-v1.0.bin"
        assert voices_path.read_bytes() == b"v"
        assert dl.call_count == 1

    def test_relative_explicit_path_rejected(self, tmp_path):
        """A relative tts.kokoro.model_path would silently download into the launch dir."""
        with pytest.raises(RuntimeError) as excinfo:
            from tools.tts_tool_local import _resolve_kokoro_paths
            _resolve_kokoro_paths({"model_path": "relative/model.onnx"})
        assert "absolute path" in str(excinfo.value)

    def test_tilde_explicit_path_expanded(self, tmp_path):
        """``~`` in a config path expands against the (test-pinned) HOME — no download runs."""
        import os
        from tools.tts_tool_local import _resolve_kokoro_paths
        (tmp_path / "m.onnx").write_bytes(b"m")
        (tmp_path / "v.bin").write_bytes(b"v")
        with patch("tools.tts_tool_local._kokoro_download") as dl:
            model, voices = _resolve_kokoro_paths(
                {"model_path": str(tmp_path / "m.onnx"), "voices_path": "~/v.bin"})
        dl.assert_not_called()
        assert model == tmp_path / "m.onnx"
        assert not str(voices).startswith("~")
        assert voices == Path(os.environ["HOME"]) / "v.bin"


class TestKokoroDownloadIntegrity:
    @staticmethod
    def _fake_urlopen(chunks, headers):
        """Context-manager response yielding *chunks* then clean EOF."""
        resp = MagicMock()
        resp.headers = headers
        resp.read.side_effect = [*chunks, b""]
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    def test_truncated_download_never_installs(self, tmp_path):
        """A clean early EOF (short read vs Content-Length) must leave no cached file
        and no temp fragment under any name."""
        from tools.tts_tool_local import _kokoro_download
        dest = tmp_path / "kokoro-v1.0.onnx"
        with patch("urllib.request.urlopen", return_value=self._fake_urlopen(
                [b"x" * 10], {"Content-Length": "1000"})), \
             pytest.raises(RuntimeError, match="truncated"):
            _kokoro_download("https://example.invalid/model.onnx", dest)
        assert not dest.exists()
        assert not list(tmp_path.glob("*.part"))

    def test_empty_body_never_installs(self, tmp_path):
        from tools.tts_tool_local import _kokoro_download
        dest = tmp_path / "voices-v1.0.bin"
        with patch("urllib.request.urlopen", return_value=self._fake_urlopen([], {})), \
             pytest.raises(RuntimeError, match="empty body"):
            _kokoro_download("https://example.invalid/voices.bin", dest)
        assert not dest.exists()
        assert not list(tmp_path.glob("*.part"))

    def test_complete_download_installs(self, tmp_path):
        from tools.tts_tool_local import _kokoro_download
        dest = tmp_path / "kokoro-v1.0.onnx"
        payload = b"m" * 2048
        with patch("urllib.request.urlopen", return_value=self._fake_urlopen(
                [payload[:1024], payload[1024:]], {"Content-Length": str(len(payload))})):
            _kokoro_download("https://example.invalid/model.onnx", dest)
        assert dest.read_bytes() == payload

    def test_download_leaves_a_reusable_lock_marker(self, tmp_path):
        """The single-flight lock file stays (unlinking it re-opens the flock race);
        crashed .part fragments are swept on the next download instead."""
        from tools.tts_tool_local import _kokoro_download
        dest = tmp_path / "kokoro-v1.0.onnx"
        (tmp_path / "kokoro-v1.0.onnx.crashed.part").write_bytes(b"junk")
        with patch("urllib.request.urlopen", return_value=self._fake_urlopen(
                [b"payload"], {})):
            _kokoro_download("https://example.invalid/model.onnx", dest)
        assert dest.with_name(dest.name + ".lock").exists()
        assert not list(tmp_path.glob("*.part"))


class TestWarmCache:
    def test_warmer_fills_the_synthesis_slot(self, kokoro_env):
        from tools import tts_tool_lifecycle
        warmer = tts_tool_lifecycle._local_tts_warmers()["kokoro"]
        warmer({})
        from tools import tts_tool_local as _tt
        assert len(_tt._kokoro_model_cache) == 1


class TestDispatcherBranch:
    def test_kokoro_not_installed_returns_helpful_error(self, monkeypatch, tmp_path):
        """When provider=kokoro but kokoro-onnx is missing, return a JSON error with the hint."""
        monkeypatch.setitem(sys.modules, "kokoro_onnx", None)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from tools.tts_tool import text_to_speech_tool

        import hermes_yaml as yaml
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"tts": {"provider": "kokoro"}}))

        result = json.loads(text_to_speech_tool(text="Hello"))
        assert result["success"] is False
        assert "kokoro" in result["error"].lower()

    def test_kokoro_is_a_builtin_provider_name(self):
        from tools.tts_command_provider import BUILTIN_TTS_PROVIDERS
        assert "kokoro" in BUILTIN_TTS_PROVIDERS

    def test_kokoro_has_a_requirement_check(self):
        from tools.tts_tool import _BUILTIN_REQUIREMENTS
        assert "kokoro" in _BUILTIN_REQUIREMENTS
