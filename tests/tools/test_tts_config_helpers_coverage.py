"""Coverage for the config-helper + module-header cluster of tools/tts_tool.py.

Exercises the low-level helpers in the ~60-746 line range: environment
resolution, lazy imports, config coercion, response readers, max-text-length
resolution, audio-delivery packing, the config loader, provider selection,
and MiniMax runtime resolution.  Every test asserts observable behaviour
(return value, written file, exception type + message) on real inputs —
no network, no real API keys; the import layer is mocked where a provider
SDK is required.
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tools import tts_tool


# ---------------------------------------------------------------------------
# Environment resolution
# ---------------------------------------------------------------------------
class TestGetEnvValue:
    def test_returns_env_value(self):
        with patch("hermes_cli.config.get_env_value", return_value="v"):
            assert tts_tool.get_env_value("FOO") == "v"

    def test_returns_default_when_env_none(self):
        with patch("hermes_cli.config.get_env_value", return_value=None):
            assert tts_tool.get_env_value("FOO", "dflt") == "dflt"

    def test_import_error_falls_back_to_os_environ(self, monkeypatch):
        monkeypatch.setenv("TTS_TEST_ENV_LOOKUP", "os-val")
        # Simulate hermes_cli.config being importable but lacking get_env_value,
        # forcing the ImportError fallback to os.getenv.
        monkeypatch.setitem(sys.modules, "hermes_cli.config", SimpleNamespace())
        assert tts_tool.get_env_value("TTS_TEST_ENV_LOOKUP", "dflt") == "os-val"


class TestResolveProviderKey:
    def test_delegates_to_shared_resolver(self):
        with patch(
            "tools.tool_backend_helpers.resolve_provider_secret", return_value="secret"
        ) as rps:
            assert tts_tool._resolve_provider_key("X", "prov") == "secret"
        rps.assert_called_once()

    def test_import_error_falls_back_to_env(self, monkeypatch):
        monkeypatch.setitem(
            sys.modules, "tools.tool_backend_helpers", SimpleNamespace()
        )
        monkeypatch.setattr(tts_tool, "get_env_value", lambda *a, **kw: "env-k")
        assert tts_tool._resolve_provider_key("X", "prov") == "env-k"


# ---------------------------------------------------------------------------
# Lazy imports (SDK import layer mocked; no network / no real keys)
# ---------------------------------------------------------------------------
class TestLazyImportEdgeTts:
    def test_returns_module(self):
        fake = SimpleNamespace()
        with patch.dict(sys.modules, {"edge_tts": fake}), \
             patch("tools.lazy_deps.ensure") as ensure:
            assert tts_tool._import_edge_tts() is fake
        ensure.assert_called_once()

    def test_import_error_from_ensure_passthrough(self):
        fake = SimpleNamespace()
        with patch.dict(sys.modules, {"edge_tts": fake}), \
             patch("tools.lazy_deps.ensure", side_effect=ImportError("no")):
            assert tts_tool._import_edge_tts() is fake

    def test_non_import_exception_from_ensure_passthrough(self):
        fake = SimpleNamespace()
        with patch.dict(sys.modules, {"edge_tts": fake}), \
             patch("tools.lazy_deps.ensure", side_effect=RuntimeError("boom")):
            assert tts_tool._import_edge_tts() is fake


class TestLazyImportElevenLabs:
    def test_returns_class(self):
        fake_mod = SimpleNamespace(ElevenLabs=SimpleNamespace)
        with patch.dict(sys.modules, {"elevenlabs.client": fake_mod}), \
             patch("tools.lazy_deps.ensure") as ensure:
            assert tts_tool._import_elevenlabs() is SimpleNamespace
        ensure.assert_called_once()

    def test_import_error_from_ensure_passthrough(self):
        fake_mod = SimpleNamespace(ElevenLabs=type("EL", (), {}))
        with patch.dict(sys.modules, {"elevenlabs.client": fake_mod}), \
             patch("tools.lazy_deps.ensure", side_effect=ImportError("no")):
            assert tts_tool._import_elevenlabs() is fake_mod.ElevenLabs

    def test_non_import_exception_from_ensure_passthrough(self):
        fake_mod = SimpleNamespace(ElevenLabs=type("EL", (), {}))
        with patch.dict(sys.modules, {"elevenlabs.client": fake_mod}), \
             patch("tools.lazy_deps.ensure", side_effect=RuntimeError("boom")):
            assert tts_tool._import_elevenlabs() is fake_mod.ElevenLabs


class TestElevenLabsEnvironmentKwargs:
    def test_no_base_url_returns_empty(self):
        assert tts_tool._elevenlabs_environment_kwargs({}) == {}

    def test_base_url_without_wss_defaults_scheme(self):
        caps = []

        class FakeEnv:
            def __init__(self, base, wss):
                caps.append((base, wss))

        fake_mod = SimpleNamespace(ElevenLabsEnvironment=FakeEnv)
        with patch.dict(sys.modules, {"elevenlabs.environment": fake_mod}):
            result = tts_tool._elevenlabs_environment_kwargs(
                {"base_url": "http://localhost:8080"}
            )
        assert result["environment"] is not None
        assert caps == [("http://localhost:8080", "ws://localhost:8080")]

    def test_base_url_with_wss_passthrough(self):
        caps = []

        class FakeEnv:
            def __init__(self, base, wss):
                caps.append((base, wss))

        fake_mod = SimpleNamespace(ElevenLabsEnvironment=FakeEnv)
        with patch.dict(sys.modules, {"elevenlabs.environment": fake_mod}):
            result = tts_tool._elevenlabs_environment_kwargs(
                {"base_url": "https://self.host/", "wss_url": "wss://self.host/"}
            )
        assert result["environment"] is not None
        assert caps == [("https://self.host", "wss://self.host")]


class TestLazyImportOpenAi:
    def test_returns_class(self):
        fake_mod = SimpleNamespace(OpenAI=type("OpenAI", (), {}))
        with patch.dict(sys.modules, {"openai": fake_mod}):
            assert tts_tool._import_openai_client() is fake_mod.OpenAI


class TestLazyImportMistral:
    def test_returns_class(self):
        fake_mod = SimpleNamespace(Mistral=type("Mistral", (), {}))
        with patch.dict(sys.modules, {"mistralai.client": fake_mod}), \
             patch("tools.lazy_deps.ensure") as ensure:
            assert tts_tool._import_mistral_client() is fake_mod.Mistral
        ensure.assert_called_once()

    def test_import_error_from_ensure_passthrough(self):
        fake_mod = SimpleNamespace(Mistral=type("Mistral", (), {}))
        with patch.dict(sys.modules, {"mistralai.client": fake_mod}), \
             patch("tools.lazy_deps.ensure", side_effect=ImportError("no")):
            assert tts_tool._import_mistral_client() is fake_mod.Mistral

    def test_non_import_exception_from_ensure_passthrough(self):
        fake_mod = SimpleNamespace(Mistral=type("Mistral", (), {}))
        with patch.dict(sys.modules, {"mistralai.client": fake_mod}), \
             patch("tools.lazy_deps.ensure", side_effect=RuntimeError("boom")):
            assert tts_tool._import_mistral_client() is fake_mod.Mistral


class TestLazyImportSounddevice:
    def test_returns_module(self):
        fake = SimpleNamespace()
        with patch.dict(sys.modules, {"sounddevice": fake}):
            assert tts_tool._import_sounddevice() is fake


class TestLazyImportKittenTts:
    def test_returns_class(self):
        fake_mod = SimpleNamespace(KittenTTS=type("KittenTTS", (), {}))
        with patch.dict(sys.modules, {"kittentts": fake_mod}):
            assert tts_tool._import_kittentts() is fake_mod.KittenTTS


class TestLazyImportPiper:
    def test_returns_class(self):
        fake_mod = SimpleNamespace(PiperVoice=type("PiperVoice", (), {}))
        with patch.dict(sys.modules, {"piper": fake_mod}):
            assert tts_tool._import_piper() is fake_mod.PiperVoice


# ---------------------------------------------------------------------------
# Config coercion
# ---------------------------------------------------------------------------
class TestConfigBool:
    @pytest.mark.parametrize("value,expected", [(True, True), (False, False)])
    def test_bool_passthrough(self, value, expected):
        assert tts_tool._config_bool(value) is expected

    def test_none_returns_default(self):
        assert tts_tool._config_bool(None) is False
        assert tts_tool._config_bool(None, default=True) is True

    @pytest.mark.parametrize(
        "value,expected", [(1, True), (0, False), (3.5, True), (-1, True)]
    )
    def test_numeric_truthiness(self, value, expected):
        assert tts_tool._config_bool(value) is expected

    @pytest.mark.parametrize(
        "value",
        ["1", "true", "TRUE", "yes", "on", "enabled", " Enabled "],
    )
    def test_truthy_strings(self, value):
        assert tts_tool._config_bool(value) is True

    @pytest.mark.parametrize(
        "value",
        ["0", "false", "no", "off", "disabled", "FALSE"],
    )
    def test_falsy_strings(self, value):
        assert tts_tool._config_bool(value) is False

    def test_unknown_string_returns_default(self):
        assert tts_tool._config_bool("pancake") is False
        assert tts_tool._config_bool("pancake", default=True) is True


# ---------------------------------------------------------------------------
# Response readers
# ---------------------------------------------------------------------------
class TestResponseHasExplicitStream:
    def test_non_callable_iter_content_is_false(self):
        class R:
            iter_content = "not callable"

        assert tts_tool._response_has_explicit_stream(R()) is False

    def test_requests_module_iter_content_true(self):
        class R:
            def iter_content(self, chunk_size=1024):
                return iter(())

        R.__module__ = "requests.models"
        assert tts_tool._response_has_explicit_stream(R()) is True

    def test_vars_has_iter_content_true(self):
        class R:
            def iter_content(self, chunk_size=1024):
                return iter(())

        assert tts_tool._response_has_explicit_stream(R()) is True


class TestCloseResponse:
    def test_calls_close(self):
        class R:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        r = R()
        tts_tool._close_response(r)
        assert r.closed is True

    def test_close_error_swallowed(self):
        class R:
            def close(self):
                raise RuntimeError("boom")

        # Must not propagate.
        tts_tool._close_response(R())

    def test_no_close_noop(self):
        tts_tool._close_response(object())  # must not raise


class TestReadTtsResponseBytes:
    def test_streaming_chunks_returned(self):
        class R:
            def iter_content(self, chunk_size=1024):
                yield b"abc"
                yield b"def"

            def close(self):
                pass

        assert (
            tts_tool._read_tts_response_bytes(R(), label="x", limit=1000)
            == b"abcdef"
        )

    def test_str_chunks_encoded(self):
        class R:
            def iter_content(self, chunk_size=1024):
                yield "ab"
                yield "cd"

            def close(self):
                pass

        assert tts_tool._read_tts_response_bytes(R(), label="x", limit=1000) == b"abcd"

    def test_content_str_encoded(self):
        class R:
            def __init__(self):
                self.content = "h\u00e9llo"

        assert (
            tts_tool._read_tts_response_bytes(R(), label="x", limit=1000)
            == "h\u00e9llo".encode()
        )

    def test_content_bytes_returned(self):
        class R:
            def __init__(self):
                self.content = b"raw-bytes"

        assert tts_tool._read_tts_response_bytes(R(), label="x", limit=1000) == b"raw-bytes"

    def test_oversize_raises_and_closes(self):
        class R:
            def iter_content(self, chunk_size=1024):
                yield b"a" * 200

            def close(self):
                self.closed = True

        r = R()
        with pytest.raises(RuntimeError, match="x response exceeds 10 bytes"):
            tts_tool._read_tts_response_bytes(r, label="x", limit=10)
        assert r.closed is True


class TestReadTtsResponseJson:
    def test_json_from_bytes(self):
        class R:
            def __init__(self):
                self.content = b'{"k": "v"}'

        assert tts_tool._read_tts_response_json(R(), label="x", limit=1000) == {
            "k": "v"
        }

    def test_fallback_to_json_method_when_empty(self):
        class R:
            def json(self):
                return {"a": 1}

        assert tts_tool._read_tts_response_json(R(), label="x", limit=1000) == {"a": 1}

    def test_empty_given_explicit_stream(self):
        class R:
            def iter_content(self, chunk_size=1024):
                return iter(())

            def close(self):
                pass

        assert tts_tool._read_tts_response_json(R(), label="x", limit=1000) == {}


class TestWriteTtsResponseToFile:
    def test_writes_bytes(self, tmp_path):
        class R:
            def __init__(self):
                self.content = b"audio-bytes"

        out = str(tmp_path / "a.mp3")
        tts_tool._write_tts_response_to_file(R(), out, label="x", limit=1000)
        assert (tmp_path / "a.mp3").read_bytes() == b"audio-bytes"


# ---------------------------------------------------------------------------
# Max-text-length resolution
# ---------------------------------------------------------------------------
class TestResolveMaxTextLength:
    def test_no_provider_returns_fallback(self):
        assert tts_tool._resolve_max_text_length(None) == tts_tool.FALLBACK_MAX_TEXT_LENGTH

    def test_positive_override_wins(self):
        cfg = {"edge": {"max_text_length": 123}}
        assert tts_tool._resolve_max_text_length("edge", cfg) == 123

    def test_bool_override_ignored(self):
        cfg = {"edge": {"max_text_length": True}}
        assert tts_tool._resolve_max_text_length("edge", cfg) == \
            tts_tool.PROVIDER_MAX_TEXT_LENGTH["edge"]

    def test_non_positive_override_ignored(self):
        cfg = {"edge": {"max_text_length": 0}}
        assert tts_tool._resolve_max_text_length("edge", cfg) == \
            tts_tool.PROVIDER_MAX_TEXT_LENGTH["edge"]

    def test_builtin_default_table(self):
        assert tts_tool._resolve_max_text_length("openai") == \
            tts_tool.PROVIDER_MAX_TEXT_LENGTH["openai"]

    def test_elevenlabs_model_aware_table(self):
        cfg = {"elevenlabs": {"model_id": "eleven_flash_v2_5"}}
        assert tts_tool._resolve_max_text_length("elevenlabs", cfg) == 40000

    def test_command_provider_override(self):
        cfg = {"providers": {"mycmd": {"type": "command", "command": "x", "max_text_length": 777}}}
        assert tts_tool._resolve_max_text_length("mycmd", cfg) == 777

    def test_command_provider_default_cap(self):
        cfg = {"providers": {"mycmd": {"type": "command", "command": "x"}}}
        assert tts_tool._resolve_max_text_length("mycmd", cfg) == \
            tts_tool.DEFAULT_COMMAND_TTS_MAX_TEXT_LENGTH

    def test_command_provider_bool_override_ignored(self):
        cfg = {"providers": {"mycmd": {"type": "command", "command": "x", "max_text_length": True}}}
        assert tts_tool._resolve_max_text_length("mycmd", cfg) == \
            tts_tool.DEFAULT_COMMAND_TTS_MAX_TEXT_LENGTH

    def test_unknown_provider_returns_fallback(self):
        assert tts_tool._resolve_max_text_length("mystery") == \
            tts_tool.FALLBACK_MAX_TEXT_LENGTH


# ---------------------------------------------------------------------------
# Audio delivery profile
# ---------------------------------------------------------------------------
class TestResolveAudioDeliveryProfile:
    def test_default_platform(self):
        p = tts_tool._resolve_audio_delivery_profile(None)
        assert p.platform == "default"
        assert p.max_file_bytes == 10 * 1024 * 1024
        assert p.safety_ratio == 0.85

    def test_named_platform(self):
        p = tts_tool._resolve_audio_delivery_profile("Telegram")
        assert p.platform == "telegram"
        assert p.max_file_bytes == 50 * 1024 * 1024

    def test_delivery_profile_overrides(self):
        cfg = {"delivery_profiles": {"telegram": {"max_file_bytes": 1024, "safety_ratio": 0.9}}}
        p = tts_tool._resolve_audio_delivery_profile("telegram", cfg)
        assert p.max_file_bytes == 1024
        assert p.safety_ratio == 0.9

    def test_invalid_max_file_bytes_falls_back(self):
        for bad in (0, -5, "big", True):
            cfg = {"delivery_profiles": {"default": {"max_file_bytes": bad}}}
            p = tts_tool._resolve_audio_delivery_profile("default", cfg)
            assert p.max_file_bytes == 10 * 1024 * 1024

    def test_invalid_safety_ratio_falls_back(self):
        for bad in (0, 1.5, 2, -1, "x"):
            cfg = {"delivery_profiles": {"default": {"safety_ratio": bad}}}
            p = tts_tool._resolve_audio_delivery_profile("default", cfg)
            assert p.safety_ratio == 0.85


# ---------------------------------------------------------------------------
# Long-form chunking
# ---------------------------------------------------------------------------
class TestSplitOversizedSentence:
    def test_no_split_for_fits(self):
        assert tts_tool._split_oversized_sentence("hello world", 20) == ["hello world"]

    def test_word_longer_than_max_chunked(self):
        assert tts_tool._split_oversized_sentence("abcdefghijk", 5) == [
            "abcde",
            "fghij",
            "k",
        ]

    def test_flushes_current_before_oversize_word(self):
        assert tts_tool._split_oversized_sentence("hello abcdef", 5) == [
            "hello",
            "abcde",
            "f",
        ]

    def test_mixed_words(self):
        assert tts_tool._split_oversized_sentence("ab cd efg", 4) == [
            "ab",
            "cd",
            "efg",
        ]


class TestSplitTextForTts:
    def test_non_positive_max_uses_fallback(self):
        chunks = tts_tool._split_text_for_tts("a" * 6000, 0)
        assert all(len(c) <= tts_tool.FALLBACK_MAX_TEXT_LENGTH for c in chunks)
        assert sum(len(c) for c in chunks) == 6000

    def test_empty_text_returns_empty(self):
        assert tts_tool._split_text_for_tts("   ", 100) == []

    def test_short_text_returns_single_chunk(self):
        assert tts_tool._split_text_for_tts("Hello world", 100) == ["Hello world"]

    def test_long_text_chunked(self):
        text = "First sentence here. Second sentence is longer than the cap! Short."
        chunks = tts_tool._split_text_for_tts(text, 20)
        assert all(0 < len(c) <= 20 for c in chunks)
        assert "".join(chunks)


# ---------------------------------------------------------------------------
# Audio delivery packing
# ---------------------------------------------------------------------------
class TestPackAudioFilesForDelivery:
    def _write(self, tmp_path, name, size):
        p = tmp_path / name
        p.write_bytes(b"x" * size)
        return str(p)

    def test_single_group(self, tmp_path):
        profile = tts_tool.AudioDeliveryProfile("default", 100, 0.8)
        a = self._write(tmp_path, "a.mp3", 30)
        assert tts_tool._pack_audio_files_for_delivery([a], profile) == [[a]]

    def test_flushes_on_overflow(self, tmp_path):
        profile = tts_tool.AudioDeliveryProfile("default", 100, 0.8)  # target 80
        a = self._write(tmp_path, "a.mp3", 60)
        b = self._write(tmp_path, "b.mp3", 60)
        groups = tts_tool._pack_audio_files_for_delivery([a, b], profile)
        assert groups == [[a], [b]]

    def test_flushes_on_suffix_change(self, tmp_path):
        profile = tts_tool.AudioDeliveryProfile("default", 100, 0.8)  # target 80
        a = self._write(tmp_path, "a.mp3", 40)
        b = self._write(tmp_path, "b.wav", 40)
        groups = tts_tool._pack_audio_files_for_delivery([a, b], profile)
        assert groups == [[a], [b]]


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
class TestLoadTtsConfig:
    def test_returns_tts_section(self):
        with patch(
            "hermes_cli.config.load_config",
            return_value={"tts": {"provider": "openai"}},
        ):
            assert tts_tool._load_tts_config() == {"provider": "openai"}

    def test_missing_key_returns_default(self):
        with patch("hermes_cli.config.load_config", return_value={"other": 1}):
            assert tts_tool._load_tts_config() == {}

    def test_import_error_returns_default(self, monkeypatch, caplog):
        monkeypatch.setitem(sys.modules, "hermes_cli.config", SimpleNamespace())
        with caplog.at_level("DEBUG"):
            assert tts_tool._load_tts_config() == {}
        assert any("default TTS config" in r.message for r in caplog.records)

    def test_unexpected_error_returns_default(self, caplog):
        with patch(
            "hermes_cli.config.load_config", side_effect=RuntimeError("boom")
        ):
            with caplog.at_level("WARNING"):
                assert tts_tool._load_tts_config() == {}
        assert any("Failed to load TTS config" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------
class TestGetProvider:
    def test_nous_maps_to_openai(self):
        assert tts_tool._get_provider({"provider": "nous"}) == "openai"

    def test_explicit_provider(self):
        assert tts_tool._get_provider({"provider": "MISTRAL"}) == "mistral"

    def test_default_when_missing(self):
        assert tts_tool._get_provider({}) == tts_tool.DEFAULT_PROVIDER


# ---------------------------------------------------------------------------
# MiniMax runtime resolution
# ---------------------------------------------------------------------------
class TestResolveMiniMaxTtsRuntime:
    def _patch_key(self, global_key="", cn_key=""):
        return patch.object(
            tts_tool,
            "_resolve_provider_key",
            side_effect=lambda env_var, provider_id: {
                "MINIMAX_API_KEY": global_key,
                "MINIMAX_CN_API_KEY": cn_key,
            }.get(env_var, ""),
        )

    def test_configured_region_wins(self):
        with self._patch_key(global_key="gk", cn_key="ck"):
            rt = tts_tool._resolve_minimax_tts_runtime(
                {"minimax": {"region": "cn"}}
            )
        assert rt.region == "cn"
        assert rt.credential_source == "MINIMAX_CN_API_KEY"
        assert rt.api_key == "ck"
        assert rt.endpoint == tts_tool.DEFAULT_MINIMAX_CN_BASE_URL
        # credential excluded from repr by field(repr=False)
        assert "ck" not in repr(rt)

    def test_global_region_when_global_key_present(self):
        with self._patch_key(global_key="gk"):
            rt = tts_tool._resolve_minimax_tts_runtime({})
        assert rt.region == "global"
        assert rt.credential_source == "MINIMAX_API_KEY"
        assert rt.api_key == "gk"
        assert rt.endpoint == tts_tool.DEFAULT_MINIMAX_BASE_URL

    def test_cn_when_only_cn_key(self):
        with self._patch_key(cn_key="ck"):
            rt = tts_tool._resolve_minimax_tts_runtime({})
        assert rt.region == "cn"
        assert rt.credential_source == "MINIMAX_CN_API_KEY"

    def test_no_keys_raises_value_error(self):
        with self._patch_key():
            with pytest.raises(ValueError, match="MINIMAX_API_KEY not set"):
                tts_tool._resolve_minimax_tts_runtime({})

    def test_invalid_region_raises(self):
        with self._patch_key(global_key="gk"):
            with pytest.raises(ValueError, match="must be 'global' or 'cn'"):
                tts_tool._resolve_minimax_tts_runtime({"minimax": {"region": "eu"}})

    def test_non_dict_minimax_config_treated_as_empty(self):
        with self._patch_key(global_key="gk"):
            rt = tts_tool._resolve_minimax_tts_runtime({"minimax": "bogus"})
        assert rt.region == "global"
        assert rt.api_key == "gk"

    def test_endpoint_host_mismatch_raises(self):
        with self._patch_key(global_key="gk"):
            with pytest.raises(ValueError, match="points to the 'cn' MiniMax endpoint"):
                tts_tool._resolve_minimax_tts_runtime(
                    {
                        "minimax": {
                            "region": "global",
                            "base_url": "https://api.minimaxi.com/v1/t2a_v2",
                        }
                    }
                )
