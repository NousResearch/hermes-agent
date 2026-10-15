"""Coverage for the main entry points of ``tools.tts_tool.py``.

Focus: ``text_to_speech_tool`` (framing + real provider-dispatch resolution),
``check_tts_requirements``, ``_resolve_openai_audio_client_config`` and
``_has_openai_audio_backend``.

Provider SDKs/network are never touched: the single-provider generators are
stubbed with functions that write a real (non-empty) file, so the dispatch bar
(framing, provider resolution, output bookkeeping, voice-compatibility and
delivery packing) is exercised against real code paths.

Missing-line clusters targeted (from the coverage report for
``tools/tts_tool.py``):

  * ``text_to_speech_tool`` : 3530, 3537-3538, 3553, 3561, 3563, 3580,
    3595-3612, 3624, 3638-3639, 3650, 3677, 3695-3702
  * ``_text_to_speech_single`` dispatch/framing exercised via the
    ``text_to_speech_tool`` entry point: 3456-3459, 3460-3463, 3469,
    3479-3493 (plus the openai dispatch branch framing)
  * ``check_tts_requirements`` : 3728, 3733, 3735, 3737-3741, 3743-3745,
    3747-3749, 3751-3755, 3757-3762, 3764-3767, 3770-3773, 3774-3779,
    3781-3789
  * ``_resolve_openai_audio_client_config`` and ``_has_openai_audio_backend``
"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tools import tts_tool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _success_result(path: str, provider: str, voice_compatible: bool = False) -> str:
    """Build the JSON string ``_text_to_speech_single`` returns on success."""
    return json.dumps(
        {
            "success": True,
            "file_path": path,
            "file_paths": [path],
            "provider": provider,
            "voice_compatible": voice_compatible,
        },
        ensure_ascii=False,
    )


def _write_audio(path: str, size: int = 64) -> str:
    """Write a real, non-empty audio file at *path* and return it."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x00" * size)
    return str(p)


def _raising(exc_type, *args):
    """Return a callable that raises ``exc_type(*args)`` — for monkeypatch
    which, unlike ``unittest.mock.patch``, has no ``side_effect`` kwarg."""
    def _callable(*_a, **_k):
        raise exc_type(*args)
    return _callable


@pytest.fixture(autouse=True)
def _allow_audio_writes(monkeypatch):
    """Deterministic write-safety + text normalization for entry-point tests."""
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.setattr(
        "agent.file_safety.is_write_denied", lambda path: False, raising=False
    )
    monkeypatch.setattr(
        "agent.file_safety.is_write_approval_required", lambda path: False,
        raising=False,
    )
    # Keep TTS text normalization deterministic for this test module.
    monkeypatch.setattr(
        "tools.tts_text_normalize.prepare_spoken_text",
        lambda text, max_chars=None: text.strip(),
        raising=False,
    )


# ---------------------------------------------------------------------------
# text_to_speech_tool — framing branches (single-provider generator stubbed)
# ---------------------------------------------------------------------------
class TestTextToSpeechToolFraming:
    def test_missing_text_returns_error(self):
        result = json.loads(tts_tool.text_to_speech_tool(text=""))
        assert result["success"] is False
        assert "Text is required" in result["error"]

    def test_whitespace_only_text_returns_error(self):
        result = json.loads(tts_tool.text_to_speech_tool(text="   \n  "))
        assert result["success"] is False
        assert "Text is required" in result["error"]

    def test_text_empty_after_cleanup_returns_error(self):
        # prepare_spoken_text returned empty -> the guard at 3539-3540 fires.
        with patch(
            "tools.tts_text_normalize.prepare_spoken_text",
            lambda text, max_chars=None: "",
        ):
            result = json.loads(tts_tool.text_to_speech_tool(text="hello"))
        assert result["success"] is False
        assert "Text is empty after TTS cleanup" in result["error"]

    def test_prepare_spoken_text_exception_falls_back_to_strip(self, tmp_path):
        # 3537-3538: a failing normalizer falls back to plain strip.
        def boom(text, max_chars=None):
            raise RuntimeError("normalizer down")

        out = tmp_path / "out.mp3"
        out.write_bytes(b"\x00" * 16)
        with patch("tools.tts_text_normalize.prepare_spoken_text", boom), \
             patch.object(tts_tool, "_text_to_speech_single") as single:
            single.return_value = _success_result(str(out), "edge")
            tts_tool.text_to_speech_tool(text="  hello  ", output_path=str(out))
            # text passed to the single generator is stripped, not raw.
            assert single.call_args[1]["text"] == "hello"

    def test_resolves_configured_provider(self, tmp_path):
        # provider picked from config when none passed -> dispatched to it.
        out = tmp_path / "out.mp3"
        out.write_bytes(b"\x00" * 16)
        with patch.object(tts_tool, "_load_tts_config",
                          return_value={"provider": "openai"}), \
             patch.object(tts_tool, "_text_to_speech_single") as single:
            single.return_value = _success_result(str(out), "openai")
            tts_tool.text_to_speech_tool(text="hi", output_path=str(out))
        assert single.call_args[1]["provider"] == "openai"

    def test_provider_override_is_dispatched(self, tmp_path):
        out = tmp_path / "out.mp3"
        out.write_bytes(b"\x00" * 16)
        with patch.object(tts_tool, "_text_to_speech_single") as single:
            single.return_value = _success_result(str(out), "minimax")
            tts_tool.text_to_speech_tool(
                text="hi", output_path=str(out), provider="minimax"
            )
        assert single.call_args[1]["provider"] == "minimax"

    def test_provider_case_and_whitespace_normalized(self, tmp_path):
        # 3553: provider is lowercased + stripped before dispatch.
        out = tmp_path / "out.mp3"
        out.write_bytes(b"\x00" * 16)
        with patch.object(tts_tool, "_text_to_speech_single") as single:
            single.return_value = _success_result(str(out), "openai")
            tts_tool.text_to_speech_tool(
                text="hi", output_path=str(out), provider="  OPENAI  "
            )
        assert single.call_args[1]["provider"] == "openai"

    def test_speed_passed_through(self, tmp_path):
        out = tmp_path / "out.mp3"
        out.write_bytes(b"\x00" * 16)
        with patch.object(tts_tool, "_text_to_speech_single") as single:
            single.return_value = _success_result(str(out), "edge")
            tts_tool.text_to_speech_tool(
                text="hi", output_path=str(out), speed=2.0
            )
        assert single.call_args[1]["speed"] == 2.0

    def test_returns_success_shape_and_writes_file(self, tmp_path):
        out = str(tmp_path / "voice.mp3")
        with patch.object(tts_tool, "_text_to_speech_single") as single:
            single.side_effect = (
                lambda text, output_path, **kw: _success_result(
                    _write_audio(output_path), "edge"
                )
            )
            result = json.loads(tts_tool.text_to_speech_tool(
                text="hello", output_path=out
            ))
        # Contract shape: success + single delivery file + tag + provider.
        assert result["success"] is True
        assert result["file_path"] == out
        assert result["file_paths"] == [out]
        assert result["provider"] == "edge"
        assert result["chunk_count"] == 1
        assert result["delivery_file_count"] == 1
        assert result["combined_chunks"] is False
        assert result["delivery_profile"]["platform"] == "default"
        assert result["media_tag"].startswith("MEDIA:")
        assert result["voice_compatible"] is False
        assert Path(out).stat().st_size > 0

    def test_chunks_empty_returns_error(self):
        # 3561: chunking produced nothing -> "Text is required".
        with patch.object(tts_tool, "_split_text_for_tts", return_value=[]):
            result = json.loads(tts_tool.text_to_speech_tool(text="hello"))
        assert result["success"] is False
        assert "Text is required" in result["error"]

    def test_multi_chunk_paths_and_logging(self, tmp_path):
        # 3563 (split log) + 3624 (per-chunk suffixed path) + combine path.
        base = tmp_path / "voice.mp3"
        with patch.object(tts_tool, "_split_text_for_tts",
                          return_value=["first chunk", "second chunk"]), \
             patch.object(tts_tool, "_text_to_speech_single") as single, \
             patch.object(tts_tool, "_concat_audio_files") as concat:
            def _stub(text, output_path, **kw):
                return _success_result(_write_audio(output_path), "edge")

            single.side_effect = _stub
            concat.side_effect = (
                lambda paths, out, **kw: _write_audio(out, 256)
            )
            result = json.loads(tts_tool.text_to_speech_tool(
                text="hello world", output_path=str(base)
            ))
        assert result["success"] is True
        assert result["chunk_count"] == 2
        assert result["combined_chunks"] is True
        # Both per-chunk paths used the .chunkNNN suffix.
        for call in single.call_args_list:
            chunk_path = call[1]["output_path"]
            assert ".chunk" in chunk_path
            assert chunk_path.endswith(".mp3")
        # The chunk paths were distinct per chunk index.
        chunk_paths = [str(call[1]["output_path"]) for call in single.call_args_list]
        assert len(set(chunk_paths)) == 2
        # Delivery derived a combined output file.
        assert Path(result["file_path"]).stat().st_size > 0

    def test_output_path_traversal_returns_error(self):
        result = json.loads(tts_tool.text_to_speech_tool(
            text="hello", output_path="audio/../../etc/cron.d/x"
        ))
        assert result["success"] is False
        assert "traversal" in result["error"]

    def test_output_path_protected_returns_error(self, monkeypatch):
        # 3595-3601: path denied by the file-safety guard.
        monkeypatch.setattr("agent.file_safety.is_write_denied",
                            lambda path: True, raising=False)
        result = json.loads(tts_tool.text_to_speech_tool(
                text="hello", output_path="/etc/out.mp3"
            ))
        assert result["success"] is False
        assert "protected credential or system path" in result["error"]

    def test_no_output_path_uses_default_dir_and_mp3(self, tmp_path, monkeypatch):
        # 3603-3612: no output_path -> timestamped path under DEFAULT_OUTPUT_DIR.
        monkeypatch.setattr(tts_tool, "DEFAULT_OUTPUT_DIR", str(tmp_path))
        with patch.object(tts_tool, "_text_to_speech_single") as single:
            single.side_effect = (
                lambda text, output_path, **kw: _success_result(
                    _write_audio(output_path), "edge"
                )
            )
            result = json.loads(tts_tool.text_to_speech_tool(text="hello"))
        assert result["success"] is True
        assert str(tmp_path) in result["file_path"]
        assert result["file_path"].endswith(".mp3")
        single.assert_called_once()

    def test_chunk_invalid_json_raises_runtime_error(self, tmp_path):
        # 3638-3639 + 3699-3702: non-JSON chunk result -> delivery error.
        with (
            patch.object(tts_tool, "_text_to_speech_single",
                         return_value="not-json"),
            patch.object(tts_tool, "_split_text_for_tts",
                         return_value=["chunk"]),
        ):
            result = json.loads(tts_tool.text_to_speech_tool(
                text="hello", output_path=str(tmp_path / "out.mp3")
            ))
        assert result["success"] is False
        assert "TTS long-form generation failed" in result["error"]

    def test_chunk_no_output_audio_raises(self, tmp_path):
        # 3650: success claimed but no file on disk -> delivery error.
        with patch.object(tts_tool, "_text_to_speech_single",
                          return_value=_success_result(str(tmp_path / "missing.mp3"), "edge")), \
             patch.object(tts_tool, "_split_text_for_tts",
                          return_value=["chunk"]):
            Path(tmp_path / "missing.mp3").unlink(missing_ok=True)
            result = json.loads(tts_tool.text_to_speech_tool(
                text="hello", output_path=str(tmp_path / "out.mp3")
            ))
        assert result["success"] is False
        assert "produced no final audio" in result["error"]

    def test_voice_compatible_media_tag_prefix(self, tmp_path):
        # 3677: voice_compatible True -> [[audio_as_voice]] prefixes the tag.
        out = _write_audio(str(tmp_path / "voice.ogg"))
        with patch.object(tts_tool, "_text_to_speech_single") as single:
            single.return_value = _success_result(out, "edge",
                                                  voice_compatible=True)
            result = json.loads(tts_tool.text_to_speech_tool(
                text="hello", output_path=out
            ))
        assert result["success"] is True
        assert result["voice_compatible"] is True
        assert result["media_tag"].startswith("[[audio_as_voice]]\nMEDIA:")

    def test_delivery_value_error_is_wrapped(self, tmp_path):
        # 3695-3698: _build_audio_delivery_files raising ValueError.
        out = _write_audio(str(tmp_path / "voice.mp3"))
        with patch.object(tts_tool, "_text_to_speech_single",
                          return_value=_success_result(out, "edge")), \
             patch.object(tts_tool, "_build_audio_delivery_files",
                          side_effect=ValueError("limit exceeded")):
            result = json.loads(tts_tool.text_to_speech_tool(
                text="hello", output_path=out
            ))
        assert result["success"] is False
        assert "TTS delivery error" in result["error"]
        assert "limit exceeded" in result["error"]

    def test_delivery_unexpected_error_is_wrapped(self, tmp_path):
        # 3699-3702: unexpected exception during delivery packing.
        out = _write_audio(str(tmp_path / "voice.mp3"))
        with patch.object(tts_tool, "_text_to_speech_single",
                          return_value=_success_result(out, "edge")), \
             patch.object(tts_tool, "_build_audio_delivery_files",
                          side_effect=RuntimeError("boom")):
            result = json.loads(tts_tool.text_to_speech_tool(
                text="hello", output_path=out
            ))
        assert result["success"] is False
        assert "TTS long-form generation failed" in result["error"]
        assert "boom" in result["error"]


# ---------------------------------------------------------------------------
# text_to_speech_tool — real provider-dispatch resolution
# ---------------------------------------------------------------------------
class TestTextToSpeechToolRealDispatch:
    """Drive the real ``_text_to_speech_single`` dispatch via the entry point,
    stubbing only the provider SDK generator so no network/SDK is touched."""

    def _openai_generator(self, text, output_path, tts_config, *,
                          api_key=None, base_url=None, model=None, voice=None,
                          speed=None, instructions=None):
        return _write_audio(output_path)

    def test_openai_dispatch_resolves_provider_and_writes_file(self, tmp_path):
        out = str(tmp_path / "voice.mp3")
        with patch.object(tts_tool, "_import_openai_client",
                          return_value=MagicMock()), \
             patch.object(tts_tool, "_generate_openai_tts",
                          side_effect=self._openai_generator), \
             patch.object(tts_tool, "_load_tts_config", return_value={}):
            result = json.loads(tts_tool.text_to_speech_tool(
                text="hello", output_path=out, provider="openai"
            ))
        assert result["success"] is True
        assert result["provider"] == "openai"
        assert result["file_path"] == out
        assert result["voice_compatible"] is False
        assert Path(out).stat().st_size > 0

    def test_openai_dispatch_missing_sdk_package(self, tmp_path):
        # 3301-3305: openai import failure -> honest error, no dispatch.
        out = str(tmp_path / "voice.mp3")
        with patch.object(tts_tool, "_import_openai_client",
                          side_effect=ImportError("no openai")), \
             patch.object(tts_tool, "_load_tts_config", return_value={}):
            result = json.loads(tts_tool.text_to_speech_tool(
                text="hello", output_path=out, provider="openai"
            ))
        assert result["success"] is False
        assert "openai" in result["error"].lower()
        assert not Path(out).exists()

    def test_openai_dispatch_telegram_voice_compatible(self, tmp_path, monkeypatch):
        # 3469: [[audio_as_voice]] prefix when the platform wants Opus and the
        # generated file is actually Ogg.
        out = str(tmp_path / "voice.ogg")
        monkeypatch.setattr(
            "gateway.session_context.get_session_env",
            lambda name, default="": "telegram",
            raising=False,
        )
        with patch.object(tts_tool, "_import_openai_client",
                          return_value=MagicMock()), \
             patch.object(tts_tool, "_generate_openai_tts",
                          side_effect=self._openai_generator), \
             patch.object(tts_tool, "_repair_ogg_container", lambda f: f), \
             patch.object(tts_tool, "_load_tts_config", return_value={}):
            result = json.loads(tts_tool.text_to_speech_tool(
                text="hello", output_path=out, provider="openai"
            ))
        assert result["success"] is True
        assert result["voice_compatible"] is True
        assert result["media_tag"].startswith("[[audio_as_voice]]\nMEDIA:")

    def test_openai_dispatch_value_error_returns_chunk_error(self, tmp_path):
        # 3479-3483: ValueError from the generator surfaces as a config error.
        out = str(tmp_path / "voice.mp3")
        with patch.object(tts_tool, "_import_openai_client",
                          return_value=MagicMock()), \
             patch.object(tts_tool, "_generate_openai_tts",
                          side_effect=ValueError("missing api key")), \
             patch.object(tts_tool, "_load_tts_config", return_value={}):
            result = json.loads(tts_tool.text_to_speech_tool(
                text="hello", output_path=out, provider="openai"
            ))
        assert result["success"] is False
        assert "TTS configuration error" in result["error"]
        assert "missing api key" in result["error"]

    def test_openai_dispatch_file_not_found_returns_dependency_error(self, tmp_path):
        # 3484-3488: FileNotFoundError reported as a dependency problem.
        out = str(tmp_path / "voice.mp3")
        with patch.object(tts_tool, "_import_openai_client",
                          return_value=MagicMock()), \
             patch.object(tts_tool, "_generate_openai_tts",
                          side_effect=FileNotFoundError("ffmpeg missing")), \
             patch.object(tts_tool, "_load_tts_config", return_value={}):
            result = json.loads(tts_tool.text_to_speech_tool(
                text="hello", output_path=out, provider="openai"
            ))
        assert result["success"] is False
        assert "TTS dependency missing" in result["error"]

    def test_openai_dispatch_unexpected_exception_returns_generic_error(self, tmp_path):
        # 3489-3493: any other exception is wrapped as a generation failure.
        out = str(tmp_path / "voice.mp3")
        with patch.object(tts_tool, "_import_openai_client",
                          return_value=MagicMock()), \
             patch.object(tts_tool, "_generate_openai_tts",
                          side_effect=RuntimeError("backend exploded")), \
             patch.object(tts_tool, "_load_tts_config", return_value={}):
            result = json.loads(tts_tool.text_to_speech_tool(
                text="hello", output_path=out, provider="openai"
            ))
        assert result["success"] is False
        assert "TTS generation failed" in result["error"]
        assert "backend exploded" in result["error"]

    def test_unknown_provider_falls_to_default_and_errors(self, tmp_path):
        # An unrecognized provider name is not rejected up-front: dispatch
        # falls through to the default Edge path, which reports availability.
        out = str(tmp_path / "voice.mp3")
        with patch.object(tts_tool, "_import_edge_tts",
                          side_effect=ImportError("no edge")), \
             patch.object(tts_tool, "_check_neutts_available",
                          return_value=False), \
             patch.object(tts_tool, "_dispatch_to_plugin_provider",
                          return_value=None), \
             patch.object(tts_tool, "_load_tts_config", return_value={}):
            result = json.loads(tts_tool.text_to_speech_tool(
                text="hello", output_path=out, provider="bogus-provider"
            ))
        assert result["success"] is False
        assert "No TTS provider available" in result["error"]


# ---------------------------------------------------------------------------
# check_tts_requirements
# ---------------------------------------------------------------------------
class TestCheckTtsRequirements:
    def _setup(self, monkeypatch, provider, config=None):
        """Force ``check_tts_requirements`` to evaluate *provider*."""
        monkeypatch.setattr(tts_tool, "_load_tts_config",
                            lambda: config or {"provider": provider})
        monkeypatch.setattr(tts_tool, "_get_provider",
                            lambda tts_config: provider)
        monkeypatch.setattr(tts_tool, "_resolve_command_provider_config",
                            lambda p, c: None)

    def test_command_provider_returns_true(self, monkeypatch):
        self._setup(monkeypatch, "my-cli")
        monkeypatch.setattr(
            tts_tool, "_resolve_command_provider_config",
            lambda p, c: {"type": "command", "command": "echo hi"},
        )
        assert tts_tool.check_tts_requirements() is True

    def test_edge_importable_true(self, monkeypatch):
        self._setup(monkeypatch, "edge")
        monkeypatch.setattr(tts_tool, "_import_edge_tts",
                            lambda: MagicMock())
        assert tts_tool.check_tts_requirements() is True

    def test_edge_unavailable_falls_to_neutts(self, monkeypatch):
        self._setup(monkeypatch, "edge")
        monkeypatch.setattr(tts_tool, "_import_edge_tts",
                            _raising(ImportError))
        monkeypatch.setattr(tts_tool, "_check_neutts_available",
                            lambda: True)
        assert tts_tool.check_tts_requirements() is True

    def test_edge_unavailable_neutts_missing(self, monkeypatch):
        self._setup(monkeypatch, "edge")
        monkeypatch.setattr(tts_tool, "_import_edge_tts",
                            _raising(ImportError))
        monkeypatch.setattr(tts_tool, "_check_neutts_available",
                            lambda: False)
        assert tts_tool.check_tts_requirements() is False

    def test_elevenlabs_import_fail_returns_false(self, monkeypatch):
        self._setup(monkeypatch, "elevenlabs")
        monkeypatch.setattr(tts_tool, "_import_elevenlabs",
                            _raising(ImportError))
        assert tts_tool.check_tts_requirements() is False

    def test_elevenlabs_imported_but_no_key_false(self, monkeypatch):
        self._setup(monkeypatch, "elevenlabs")
        monkeypatch.setattr(tts_tool, "_import_elevenlabs",
                            lambda: MagicMock())
        monkeypatch.setattr(tts_tool, "_resolve_provider_key", lambda *a, **k: "")
        assert tts_tool.check_tts_requirements() is False

    def test_elevenlabs_imported_with_key_true(self, monkeypatch):
        self._setup(monkeypatch, "elevenlabs")
        monkeypatch.setattr(tts_tool, "_import_elevenlabs",
                            lambda: MagicMock())
        monkeypatch.setattr(tts_tool, "_resolve_provider_key",
                            lambda *a, **k: "el-key")
        assert tts_tool.check_tts_requirements() is True

    def _fake_find_spec(self, monkeypatch, present):
        real = importlib.util.find_spec

        def fake(name, *a, **k):
            if name == "openai":
                return SimpleNamespace(name="openai") if present else None
            return real(name, *a, **k)

        monkeypatch.setattr(importlib.util, "find_spec", fake)

    def test_openai_missing_sdk_false(self, monkeypatch):
        self._setup(monkeypatch, "openai")
        self._fake_find_spec(monkeypatch, present=False)
        assert tts_tool.check_tts_requirements() is False

    def test_openai_backend_present_true(self, monkeypatch):
        self._setup(monkeypatch, "openai")
        self._fake_find_spec(monkeypatch, present=True)
        monkeypatch.setattr(tts_tool, "_has_openai_audio_backend",
                            lambda: True)
        assert tts_tool.check_tts_requirements() is True

    def test_openai_backend_missing_false(self, monkeypatch):
        self._setup(monkeypatch, "openai")
        self._fake_find_spec(monkeypatch, present=True)
        monkeypatch.setattr(tts_tool, "_has_openai_audio_backend",
                            lambda: False)
        assert tts_tool.check_tts_requirements() is False

    def test_deepinfra_missing_sdk_false(self, monkeypatch):
        self._setup(monkeypatch, "deepinfra")
        self._fake_find_spec(monkeypatch, present=False)
        assert tts_tool.check_tts_requirements() is False

    def test_deepinfra_imported_with_key_true(self, monkeypatch):
        self._setup(monkeypatch, "deepinfra")
        self._fake_find_spec(monkeypatch, present=True)
        monkeypatch.setattr(tts_tool, "_resolve_provider_key",
                            lambda *a, **k: "di-key")
        assert tts_tool.check_tts_requirements() is True

    def test_minimax_runtime_resolvable_true(self, monkeypatch):
        self._setup(monkeypatch, "minimax")
        monkeypatch.setattr(tts_tool, "_resolve_minimax_tts_runtime",
                            lambda c: SimpleNamespace(region="global"))
        assert tts_tool.check_tts_requirements() is True

    def test_minimax_runtime_failure_false(self, monkeypatch):
        self._setup(monkeypatch, "minimax")
        monkeypatch.setattr(tts_tool, "_resolve_minimax_tts_runtime",
                            _raising(ValueError, "no key"))
        assert tts_tool.check_tts_requirements() is False

    def test_xai_credentials_present_true(self, monkeypatch):
        self._setup(monkeypatch, "xai")
        monkeypatch.setattr(
            "tools.xai_http.resolve_xai_http_credentials",
            lambda: {"api_key": "xai-key"},
        )
        assert tts_tool.check_tts_requirements() is True

    def test_xai_credentials_missing_false(self, monkeypatch):
        self._setup(monkeypatch, "xai")
        monkeypatch.setattr(
            "tools.xai_http.resolve_xai_http_credentials",
            lambda: {"api_key": ""},
        )
        assert tts_tool.check_tts_requirements() is False

    def test_xai_credentials_raises_false(self, monkeypatch):
        self._setup(monkeypatch, "xai")
        monkeypatch.setattr(
            "tools.xai_http.resolve_xai_http_credentials",
            _raising(RuntimeError, "backend down"),
        )
        assert tts_tool.check_tts_requirements() is False

    def test_gemini_key_present_true(self, monkeypatch):
        self._setup(monkeypatch, "gemini")
        monkeypatch.setattr(tts_tool, "_resolve_provider_key",
                            lambda *a, **k: "gem-key")
        assert tts_tool.check_tts_requirements() is True

    def test_gemini_key_missing_false(self, monkeypatch):
        self._setup(monkeypatch, "gemini")
        monkeypatch.setattr(tts_tool, "_resolve_provider_key",
                            lambda *a, **k: "")
        assert tts_tool.check_tts_requirements() is False

    def test_mistral_import_fail_false(self, monkeypatch):
        self._setup(monkeypatch, "mistral")
        monkeypatch.setattr(tts_tool, "_import_mistral_client",
                            _raising(ImportError))
        assert tts_tool.check_tts_requirements() is False

    def test_mistral_import_ok_with_key_true(self, monkeypatch):
        self._setup(monkeypatch, "mistral")
        monkeypatch.setattr(tts_tool, "_import_mistral_client",
                            lambda: MagicMock())
        monkeypatch.setattr(tts_tool, "_resolve_provider_key",
                            lambda *a, **k: "m-key")
        assert tts_tool.check_tts_requirements() is True

    def test_neutts_available_true(self, monkeypatch):
        self._setup(monkeypatch, "neutts")
        monkeypatch.setattr(tts_tool, "_check_neutts_available",
                            lambda: True)
        assert tts_tool.check_tts_requirements() is True

    def test_neutts_unavailable_false(self, monkeypatch):
        self._setup(monkeypatch, "neutts")
        monkeypatch.setattr(tts_tool, "_check_neutts_available",
                            lambda: False)
        assert tts_tool.check_tts_requirements() is False

    def test_kittentts_available_true(self, monkeypatch):
        self._setup(monkeypatch, "kittentts")
        monkeypatch.setattr(tts_tool, "_check_kittentts_available",
                            lambda: True)
        assert tts_tool.check_tts_requirements() is True

    def test_kittentts_unavailable_false(self, monkeypatch):
        self._setup(monkeypatch, "kittentts")
        monkeypatch.setattr(tts_tool, "_check_kittentts_available",
                            lambda: False)
        assert tts_tool.check_tts_requirements() is False

    def test_piper_available_true(self, monkeypatch):
        self._setup(monkeypatch, "piper")
        monkeypatch.setattr(tts_tool, "_check_piper_available", lambda: True)
        assert tts_tool.check_tts_requirements() is True

    def test_piper_unavailable_false(self, monkeypatch):
        self._setup(monkeypatch, "piper")
        monkeypatch.setattr(tts_tool, "_check_piper_available", lambda: False)
        assert tts_tool.check_tts_requirements() is False

    def test_unknown_provider_plugin_available_true(self, monkeypatch):
        self._setup(monkeypatch, "my-plugin")
        monkeypatch.setattr("hermes_cli.plugins._ensure_plugins_discovered",
                            MagicMock(), raising=False)
        plugin = MagicMock()
        plugin.is_available.return_value = True
        monkeypatch.setattr("agent.tts_registry.get_provider",
                            lambda name: plugin, raising=False)
        assert tts_tool.check_tts_requirements() is True

    def test_unknown_provider_plugin_exception_false(self, monkeypatch):
        self._setup(monkeypatch, "my-plugin")
        monkeypatch.setattr("hermes_cli.plugins._ensure_plugins_discovered",
                            MagicMock(), raising=False)
        monkeypatch.setattr("agent.tts_registry.get_provider",
                            _raising(RuntimeError, "boom"), raising=False)
        assert tts_tool.check_tts_requirements() is False


# ---------------------------------------------------------------------------
# _resolve_openai_audio_client_config
# ---------------------------------------------------------------------------
class TestResolveOpenaiAudioClientConfig:
    def test_vendor_selection_env_key_fallback(self):
        # 3836-3838: stored vendor selection, no config key -> env key.
        config = {"openai": {}}
        with patch.object(tts_tool, "_load_tts_config", return_value=config), \
             patch.object(tts_tool, "read_selection", return_value="openai"), \
             patch.object(tts_tool, "resolve_openai_audio_api_key",
                          return_value="env-key"), \
             patch.object(tts_tool, "resolve_managed_tool_gateway") as gw:
            result = tts_tool._resolve_openai_audio_client_config()
        gw.assert_not_called()
        assert result == ("env-key", tts_tool.DEFAULT_OPENAI_BASE_URL, False)

    def test_never_configured_env_key_fallback(self):
        # 3850-3852: no stored selection, no config key -> env key.
        with patch.object(tts_tool, "_load_tts_config", return_value={}), \
             patch.object(tts_tool, "read_selection", return_value=None), \
             patch.object(tts_tool, "resolve_openai_audio_api_key",
                          return_value="env-key"):
            assert tts_tool._resolve_openai_audio_client_config() == (
                "env-key", tts_tool.DEFAULT_OPENAI_BASE_URL, False,
            )

    def test_never_configured_managed_gateway_fallback(self):
        # 3854-3856 + 3869-3873: legacy ladder ends at the managed gateway.
        managed = SimpleNamespace(
            nous_user_token="managed-token",
            gateway_origin="https://gateway.example.com",
        )
        with patch.object(tts_tool, "_load_tts_config", return_value={}), \
             patch.object(tts_tool, "read_selection", return_value=None), \
             patch.object(tts_tool, "resolve_openai_audio_api_key",
                          return_value=""), \
             patch.object(tts_tool, "resolve_managed_tool_gateway",
                          return_value=managed):
            assert tts_tool._resolve_openai_audio_client_config() == (
                "managed-token",
                "https://gateway.example.com/v1",
                True,
            )

    def test_never_configured_no_keys_managed_enabled_raises_extended(self):
        # 3856-3867: managed tools enabled -> error gains the gateway hint.
        with patch.object(tts_tool, "_load_tts_config", return_value={}), \
             patch.object(tts_tool, "read_selection", return_value=None), \
             patch.object(tts_tool, "resolve_openai_audio_api_key",
                          return_value=""), \
             patch.object(tts_tool, "resolve_managed_tool_gateway",
                          return_value=None), \
             patch.object(tts_tool, "managed_nous_tools_enabled",
                          return_value=True):
            with pytest.raises(ValueError) as exc:
                tts_tool._resolve_openai_audio_client_config()
        assert "VOICE_TOOLS_OPENAI_KEY" in str(exc.value)
        assert "managed OpenAI audio" in str(exc.value)

    def test_never_configured_no_keys_managed_disabled_raises_plain(self):
        # 3856-3858: managed disabled -> concise error, no gateway hint.
        with patch.object(tts_tool, "_load_tts_config", return_value={}), \
             patch.object(tts_tool, "read_selection", return_value=None), \
             patch.object(tts_tool, "resolve_openai_audio_api_key",
                          return_value=""), \
             patch.object(tts_tool, "resolve_managed_tool_gateway",
                          return_value=None), \
             patch.object(tts_tool, "managed_nous_tools_enabled",
                          return_value=False):
            with pytest.raises(ValueError) as exc:
                tts_tool._resolve_openai_audio_client_config()
        assert (
            str(exc.value)
            == "Neither tts.openai.api_key in config nor "
            "VOICE_TOOLS_OPENAI_KEY/OPENAI_API_KEY is set"
        )


# ---------------------------------------------------------------------------
# _has_openai_audio_backend
# ---------------------------------------------------------------------------
class TestHasOpenaiAudioBackend:
    def test_resolvable_returns_true(self):
        with patch.object(tts_tool, "_resolve_openai_audio_client_config",
                          return_value=("k", "https://x/v1", False)):
            assert tts_tool._has_openai_audio_backend() is True

    def test_unresolvable_value_error_returns_false(self):
        with patch.object(tts_tool, "_resolve_openai_audio_client_config",
                          side_effect=ValueError("no creds")):
            assert tts_tool._has_openai_audio_backend() is False


# ---------------------------------------------------------------------------
# _text_to_speech_single — per-provider dispatch (SDK generators stubbed)
# ---------------------------------------------------------------------------
class TestTextToSpeechSingleDispatch:
    """Drive ``_text_to_speech_single`` directly, stubbing only the provider
    generators so the dispatch resolution + output bookkeeping is exercised
    against real code paths (no SDK / network touch)."""

    def _gen_openai(self, text, output_path, tts_config, **kw):
        return _write_audio(output_path)

    def _gen(self, text, output_path, tts_config):
        return _write_audio(output_path)

    def _gen_command(self, text, output_path, provider_name, config, tts_config):
        return _write_audio(output_path)

    # -- framing guards ----------------------------------------------------
    def test_blank_text_returns_error(self):
        result = json.loads(tts_tool._text_to_speech_single(text=""))
        assert result["success"] is False
        assert "Text is required" in result["error"]

    def test_speed_injected_clamped_from_caller(self, tmp_path):
        out = str(tmp_path / "voice.mp3")
        with patch.object(tts_tool, "_import_openai_client",
                          return_value=MagicMock()), \
             patch.object(tts_tool, "_generate_openai_tts") as gen:
            gen.side_effect = self._gen_openai
            tts_tool._text_to_speech_single(
                text="hi", output_path=out, speed=9.9,
                provider="openai", tts_config_override={"provider": "openai"},
            )
            # 3170-3172: caller speed is clamped and injected into the config.
            assert gen.call_args[0][2]["speed"] == 4.0

    def test_speed_lower_bound_clamped(self, tmp_path):
        out = str(tmp_path / "voice.mp3")
        with patch.object(tts_tool, "_import_openai_client",
                          return_value=MagicMock()), \
             patch.object(tts_tool, "_generate_openai_tts") as gen:
            gen.side_effect = self._gen_openai
            tts_tool._text_to_speech_single(
                text="hi", output_path=out, speed=-1.0,
                provider="openai", tts_config_override={"provider": "openai"},
            )
            assert gen.call_args[0][2]["speed"] == 0.25

    def test_provider_resolved_from_config_when_not_passed(self, tmp_path):
        # 3178: no provider kwarg -> _get_provider(tts_config) decides dispatch.
        out = str(tmp_path / "voice.mp3")
        with patch.object(tts_tool, "_get_provider",
                          return_value="openai") as gp, \
             patch.object(tts_tool, "_import_openai_client",
                          return_value=MagicMock()), \
             patch.object(tts_tool, "_generate_openai_tts") as gen:
            gen.side_effect = self._gen_openai
            result = json.loads(tts_tool._text_to_speech_single(
                text="hi", output_path=out,
                tts_config_override={"provider": "edge"},
            ))
        assert gp.call_count == 1
        assert result["provider"] == "openai"
        assert gen.call_count == 1

    def test_text_over_provider_cap_logs_warning(self, tmp_path):
        out = str(tmp_path / "voice.mp3")
        with patch.object(tts_tool, "_resolve_max_text_length",
                          return_value=3), \
             patch.object(tts_tool, "_import_openai_client",
                          return_value=MagicMock()), \
             patch.object(tts_tool, "_generate_openai_tts") as gen, \
             patch.object(tts_tool.logger, "warning") as warn:
            gen.side_effect = self._gen_openai
            result = json.loads(tts_tool._text_to_speech_single(
                text="hello world", output_path=out,
                provider="openai", tts_config_override={"provider": "openai"},
            ))
        assert result["success"] is True
        assert warn.call_count == 1
        assert "cap" in str(warn.call_args[0][0])

    # -- output path determination -----------------------------------------
    def test_traversal_output_path_returns_error(self):
        result = json.loads(tts_tool._text_to_speech_single(
            text="hello", output_path="audio/../../etc/cron.d/x",
            provider="openai", tts_config_override={"provider": "openai"},
        ))
        assert result["success"] is False
        assert "traversal" in result["error"]

    def test_output_path_denied_returns_error(self, tmp_path, monkeypatch):
        out = str(tmp_path / "voice.mp3")
        monkeypatch.setattr("agent.file_safety.is_write_denied",
                            lambda path: True, raising=False)
        result = json.loads(tts_tool._text_to_speech_single(
            text="hello", output_path=out,
            provider="openai", tts_config_override={"provider": "openai"},
        ))
        assert result["success"] is False
        assert "protected credential or system path" in result["error"]

    def test_default_output_dir_mp3_fallback(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tts_tool, "DEFAULT_OUTPUT_DIR", str(tmp_path))
        with patch.object(tts_tool, "_import_openai_client",
                          return_value=MagicMock()), \
             patch.object(tts_tool, "_generate_openai_tts") as gen:
            gen.side_effect = self._gen_openai
            result = json.loads(tts_tool._text_to_speech_single(
                text="hello", provider="openai",
                tts_config_override={"provider": "openai"},
            ))
        assert result["success"] is True
        assert result["file_path"].endswith(".mp3")
        assert str(tmp_path) in result["file_path"]

    def test_default_output_dir_ogg_when_platform_wants_opus(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(tts_tool, "DEFAULT_OUTPUT_DIR", str(tmp_path))
        monkeypatch.setattr("gateway.session_context.get_session_env",
                            lambda name, default="": "telegram", raising=False)
        with patch.object(tts_tool, "_import_openai_client",
                          return_value=MagicMock()), \
             patch.object(tts_tool, "_generate_openai_tts") as gen, \
             patch.object(tts_tool, "_repair_ogg_container", lambda f: f):
            gen.side_effect = self._gen_openai
            result = json.loads(tts_tool._text_to_speech_single(
                text="hello", provider="openai",
                tts_config_override={"provider": "openai"},
            ))
        assert result["success"] is True
        assert result["file_path"].endswith(".ogg")
        assert result["voice_compatible"] is True

    def test_default_output_dir_command_format(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tts_tool, "DEFAULT_OUTPUT_DIR", str(tmp_path))
        cfg = {"providers": {"my-cli": {
            "type": "command", "command": "echo hi", "format": "ogg",
        }}}
        with patch.object(tts_tool, "_generate_command_tts") as gen:
            gen.side_effect = self._gen_command
            result = json.loads(tts_tool._text_to_speech_single(
                text="hello", provider="my-cli", tts_config_override=cfg,
            ))
        assert result["success"] is True
        assert result["file_path"].endswith(".ogg")

    # -- command-provider branch ------------------------------------------
    def test_command_provider_branch_writes_and_resolves(self, tmp_path):
        out = str(tmp_path / "voice.mp3")
        cfg = {"providers": {"my-cli": {
            "type": "command", "command": "echo hi", "format": "mp3",
        }}}
        with patch.object(tts_tool, "_generate_command_tts") as gen:
            gen.side_effect = self._gen_command
            result = json.loads(tts_tool._text_to_speech_single(
                text="hello", output_path=out, provider="my-cli",
                tts_config_override=cfg,
            ))
        assert result["success"] is True
        assert result["provider"] == "my-cli"
        assert Path(out).stat().st_size > 0
        # The resolved provider name is forwarded to the command generator.
        assert gen.call_args[0][2] == "my-cli"

    def test_command_provider_output_path_extension_aligned(self, tmp_path):
        out = str(tmp_path / "voice.wav")
        cfg = {"providers": {"my-cli": {
            "type": "command", "command": "echo hi", "format": "flac",
        }}}
        with patch.object(tts_tool, "_generate_command_tts") as gen:
            gen.side_effect = self._gen_command
            result = json.loads(tts_tool._text_to_speech_single(
                text="hello", output_path=out, provider="my-cli",
                tts_config_override=cfg,
            ))
        assert result["success"] is True
        expected = str(tmp_path / "voice.flac")
        assert result["file_path"] == expected
        assert Path(expected).stat().st_size > 0
        # 3231: the configured output format replaced the caller's extension.
        assert gen.call_args[0][1] == expected

    def test_command_provider_voice_compatible(self, tmp_path):
        mp3 = str(tmp_path / "voice.mp3")
        ogg = _write_audio(str(tmp_path / "voice.ogg"))
        cfg = {"providers": {"my-cli": {
            "type": "command", "command": "echo hi",
            "format": "mp3", "voice_compatible": True,
        }}}
        with patch.object(tts_tool, "_generate_command_tts") as gen, \
             patch.object(tts_tool, "_convert_to_opus",
                          lambda f: _write_audio(str(tmp_path / "voice.ogg"))):
            gen.side_effect = self._gen_command
            result = json.loads(tts_tool._text_to_speech_single(
                text="hello", output_path=mp3, provider="my-cli",
                tts_config_override=cfg,
            ))
        assert result["success"] is True
        assert result["file_path"] == ogg
        assert result["voice_compatible"] is True

    # -- plugin-registered provider branch --------------------------------
    def test_plugin_provider_dispatch_writes_file(self, tmp_path):
        plugin_path = _write_audio(str(tmp_path / "plugin.mp3"))
        with patch.object(tts_tool, "_dispatch_to_plugin_provider",
                          return_value=plugin_path), \
             patch.object(tts_tool, "_plugin_provider_is_voice_compatible",
                          return_value=False):
            result = json.loads(tts_tool._text_to_speech_single(
                text="hello", output_path=str(tmp_path / "x.mp3"),
                provider="my-plugin", tts_config_override={},
            ))
        assert result["success"] is True
        assert result["file_path"] == plugin_path
        assert result["provider"] == "my-plugin"

    def test_plugin_provider_voice_compatible_converts_to_opus(self, tmp_path):
        mp3 = _write_audio(str(tmp_path / "voice.mp3"))
        ogg = _write_audio(str(tmp_path / "voice.ogg"))
        with patch.object(tts_tool, "_dispatch_to_plugin_provider",
                          return_value=mp3), \
             patch.object(tts_tool, "_plugin_provider_is_voice_compatible",
                          return_value=True), \
             patch.object(tts_tool, "_convert_to_opus",
                          lambda f: _write_audio(str(tmp_path / "voice.ogg"))):
            result = json.loads(tts_tool._text_to_speech_single(
                text="hello", output_path=str(tmp_path / "x.mp3"),
                provider="my-plugin", tts_config_override={},
            ))
        assert result["success"] is True
        assert result["file_path"] == ogg
        assert result["voice_compatible"] is True

    # -- ElevenLabs -------------------------------------------------------
    def test_elevenlabs_branch(self, tmp_path):
        out = str(tmp_path / "voice.mp3")
        with patch.object(tts_tool, "_import_elevenlabs",
                          return_value=MagicMock()), \
             patch.object(tts_tool, "_generate_elevenlabs") as gen:
            gen.side_effect = self._gen
            result = json.loads(tts_tool._text_to_speech_single(
                text="hello", output_path=out, provider="elevenlabs",
                tts_config_override={"provider": "elevenlabs"},
            ))
        assert result["success"] is True
        assert result["provider"] == "elevenlabs"
        assert Path(out).stat().st_size > 0

    def test_elevenlabs_missing_package_error(self, tmp_path):
        with patch.object(tts_tool, "_import_elevenlabs",
                          side_effect=ImportError("no eleven")):
            result = json.loads(tts_tool._text_to_speech_single(
                text="hi", output_path=str(tmp_path / "v.mp3"),
                provider="elevenlabs",
                tts_config_override={"provider": "elevenlabs"},
            ))
        assert result["success"] is False
        assert "elevenlabs" in result["error"].lower()

    # -- DeepInfra --------------------------------------------------------
    def test_deepinfra_branch(self, tmp_path):
        out = str(tmp_path / "voice.mp3")
        with patch.object(tts_tool, "_import_openai_client",
                          return_value=MagicMock()), \
             patch.object(tts_tool, "_generate_deepinfra_tts") as gen:
            gen.side_effect = self._gen
            result = json.loads(tts_tool._text_to_speech_single(
                text="hello", output_path=out, provider="deepinfra",
                tts_config_override={"provider": "deepinfra"},
            ))
        assert result["success"] is True
        assert result["provider"] == "deepinfra"
        assert Path(out).stat().st_size > 0

    def test_deepinfra_missing_openai_sdk_error(self, tmp_path):
        with patch.object(tts_tool, "_import_openai_client",
                          side_effect=ImportError("no openai")):
            result = json.loads(tts_tool._text_to_speech_single(
                text="hi", output_path=str(tmp_path / "v.mp3"),
                provider="deepinfra",
                tts_config_override={"provider": "deepinfra"},
            ))
        assert result["success"] is False
        assert "deepinfra" in result["error"].lower()

    # -- MiniMax ----------------------------------------------------------
    def test_minimax_branch(self, tmp_path):
        out = str(tmp_path / "voice.mp3")
        with patch.object(tts_tool, "_generate_minimax_tts") as gen:
            gen.side_effect = self._gen
            result = json.loads(tts_tool._text_to_speech_single(
                text="hello", output_path=out, provider="minimax",
                tts_config_override={"provider": "minimax"},
            ))
        assert result["success"] is True
        assert result["provider"] == "minimax"
        assert Path(out).stat().st_size > 0

    # -- xAI --------------------------------------------------------------
    def test_xai_branch(self, tmp_path):
        out = str(tmp_path / "voice.mp3")
        with patch.object(tts_tool, "_generate_xai_tts") as gen:
            gen.side_effect = self._gen
            result = json.loads(tts_tool._text_to_speech_single(
                text="hello", output_path=out, provider="xai",
                tts_config_override={"provider": "xai"},
            ))
        assert result["success"] is True
        assert result["provider"] == "xai"
        assert Path(out).stat().st_size > 0

    # -- Mistral ----------------------------------------------------------
    def test_mistral_branch(self, tmp_path):
        out = str(tmp_path / "voice.mp3")
        with patch.object(tts_tool, "_import_mistral_client",
                          return_value=MagicMock()), \
             patch.object(tts_tool, "_generate_mistral_tts") as gen:
            gen.side_effect = self._gen
            result = json.loads(tts_tool._text_to_speech_single(
                text="hello", output_path=out, provider="mistral",
                tts_config_override={"provider": "mistral"},
            ))
        assert result["success"] is True
        assert result["provider"] == "mistral"
        assert Path(out).stat().st_size > 0

    def test_mistral_missing_package_error(self, tmp_path):
        with patch.object(tts_tool, "_import_mistral_client",
                          side_effect=ImportError("no mistral")):
            result = json.loads(tts_tool._text_to_speech_single(
                text="hi", output_path=str(tmp_path / "v.mp3"),
                provider="mistral",
                tts_config_override={"provider": "mistral"},
            ))
        assert result["success"] is False
        assert "mistral" in result["error"].lower()

    # -- Gemini -----------------------------------------------------------
    def test_gemini_branch(self, tmp_path):
        out = str(tmp_path / "voice.mp3")
        with patch.object(tts_tool, "_generate_gemini_tts") as gen:
            gen.side_effect = self._gen
            result = json.loads(tts_tool._text_to_speech_single(
                text="hello", output_path=out, provider="gemini",
                tts_config_override={"provider": "gemini"},
            ))
        assert result["success"] is True
        assert result["provider"] == "gemini"
        assert Path(out).stat().st_size > 0

    # -- NeuTTS -----------------------------------------------------------
    def test_neutts_branch(self, tmp_path):
        out = str(tmp_path / "voice.mp3")
        with patch.object(tts_tool, "_check_neutts_available",
                          return_value=True), \
             patch.object(tts_tool, "_generate_neutts") as gen:
            gen.side_effect = self._gen
            result = json.loads(tts_tool._text_to_speech_single(
                text="hello", output_path=out, provider="neutts",
                tts_config_override={"provider": "neutts"},
            ))
        assert result["success"] is True
        assert result["provider"] == "neutts"
        assert Path(out).stat().st_size > 0

    def test_neutts_unavailable_error(self, tmp_path):
        with patch.object(tts_tool, "_check_neutts_available",
                          return_value=False):
            result = json.loads(tts_tool._text_to_speech_single(
                text="hi", output_path=str(tmp_path / "v.mp3"),
                provider="neutts",
                tts_config_override={"provider": "neutts"},
            ))
        assert result["success"] is False
        assert "neutts" in result["error"].lower()

    # -- KittenTTS --------------------------------------------------------
    def test_kittentts_branch(self, tmp_path):
        out = str(tmp_path / "voice.mp3")
        with patch.object(tts_tool, "_import_kittentts",
                          return_value=MagicMock()), \
             patch.object(tts_tool, "_generate_kittentts") as gen:
            gen.side_effect = self._gen
            result = json.loads(tts_tool._text_to_speech_single(
                text="hello", output_path=out, provider="kittentts",
                tts_config_override={"provider": "kittentts"},
            ))
        assert result["success"] is True
        assert result["provider"] == "kittentts"
        assert Path(out).stat().st_size > 0

    def test_kittentts_missing_package_error(self, tmp_path):
        with patch.object(tts_tool, "_import_kittentts",
                          side_effect=ImportError("no kittentts")):
            result = json.loads(tts_tool._text_to_speech_single(
                text="hi", output_path=str(tmp_path / "v.mp3"),
                provider="kittentts",
                tts_config_override={"provider": "kittentts"},
            ))
        assert result["success"] is False
        assert "kittentts" in result["error"].lower()

    # -- Piper ------------------------------------------------------------
    def test_piper_branch(self, tmp_path):
        out = str(tmp_path / "voice.mp3")
        with patch.object(tts_tool, "_import_piper",
                          return_value=MagicMock()), \
             patch.object(tts_tool, "_generate_piper_tts") as gen:
            gen.side_effect = self._gen
            result = json.loads(tts_tool._text_to_speech_single(
                text="hello", output_path=out, provider="piper",
                tts_config_override={"provider": "piper"},
            ))
        assert result["success"] is True
        assert result["provider"] == "piper"
        assert Path(out).stat().st_size > 0

    def test_piper_missing_package_error(self, tmp_path):
        with patch.object(tts_tool, "_import_piper",
                          side_effect=ImportError("no piper")):
            result = json.loads(tts_tool._text_to_speech_single(
                text="hi", output_path=str(tmp_path / "v.mp3"),
                provider="piper",
                tts_config_override={"provider": "piper"},
            ))
        assert result["success"] is False
        assert "piper" in result["error"].lower()

    # -- Edge default (async) ---------------------------------------------
    def test_edge_default_branch_threadpool(self, tmp_path):
        out = str(tmp_path / "voice.mp3")

        async def _edge(text, output_path, tts_config):
            return _write_audio(output_path)

        with patch.object(tts_tool, "_import_edge_tts",
                          return_value=MagicMock()), \
             patch.object(tts_tool, "_generate_edge_tts",
                          side_effect=_edge):
            result = json.loads(tts_tool._text_to_speech_single(
                text="hello", output_path=out, provider="edge",
                tts_config_override={"provider": "edge"},
            ))
        assert result["success"] is True
        assert result["provider"] == "edge"
        assert Path(out).stat().st_size > 0

    def test_edge_unavailable_falls_back_to_neutts(self, tmp_path):
        out = str(tmp_path / "voice.mp3")
        with patch.object(tts_tool, "_import_edge_tts",
                          side_effect=ImportError("no edge")), \
             patch.object(tts_tool, "_check_neutts_available",
                          return_value=True), \
             patch.object(tts_tool, "_generate_neutts") as gen:
            gen.side_effect = self._gen
            result = json.loads(tts_tool._text_to_speech_single(
                text="hello", output_path=out, provider="edge",
                tts_config_override={"provider": "edge"},
            ))
        assert result["success"] is True
        assert result["provider"] == "neutts"
        assert Path(out).stat().st_size > 0

    # -- Post-generation checks -------------------------------------------
    def test_no_audio_written_returns_error(self, tmp_path):
        out = str(tmp_path / "voice.mp3")
        with patch.object(tts_tool, "_import_openai_client",
                          return_value=MagicMock()), \
             patch.object(tts_tool, "_generate_openai_tts",
                          lambda *a, **k: None):
            result = json.loads(tts_tool._text_to_speech_single(
                text="hello", output_path=out, provider="openai",
                tts_config_override={"provider": "openai"},
            ))
        assert result["success"] is False
        assert "produced no output" in result["error"]

    def test_local_provider_opus_conversion_when_platform_wants_opus(
        self, tmp_path, monkeypatch
    ):
        mp3 = str(tmp_path / "voice.mp3")
        ogg = _write_audio(str(tmp_path / "voice.ogg"))
        monkeypatch.setattr("gateway.session_context.get_session_env",
                            lambda name, default="": "telegram", raising=False)
        with patch.object(tts_tool, "_import_kittentts",
                          return_value=MagicMock()), \
             patch.object(tts_tool, "_generate_kittentts") as gen, \
             patch.object(tts_tool, "_convert_to_opus",
                          lambda f: _write_audio(str(tmp_path / "voice.ogg"))):
            gen.side_effect = self._gen
            result = json.loads(tts_tool._text_to_speech_single(
                text="hello", output_path=mp3, provider="kittentts",
                tts_config_override={"provider": "kittentts"},
            ))
        assert result["success"] is True
        assert result["file_path"] == ogg
        assert result["voice_compatible"] is True


# ---------------------------------------------------------------------------
# _resolve_openai_audio_client_config — residual branches
# ---------------------------------------------------------------------------
class TestResolveOpenaiAudioClientConfigResidual:
    def test_selected_nous_managed_gateway_available(self):
        managed = SimpleNamespace(
            nous_user_token="nous-token",
            gateway_origin="https://nous-gateway.example.com",
        )
        with patch.object(tts_tool, "_load_tts_config", return_value={}), \
             patch.object(tts_tool, "read_selection",
                          return_value=tts_tool.NOUS_MANAGED_PROVIDER), \
             patch.object(tts_tool, "resolve_managed_tool_gateway",
                          return_value=managed):
            assert tts_tool._resolve_openai_audio_client_config() == (
                "nous-token",
                "https://nous-gateway.example.com/v1",
                True,
            )

    def test_selected_nous_managed_gateway_unavailable_raises(self):
        with patch.object(tts_tool, "_load_tts_config", return_value={}), \
             patch.object(tts_tool, "read_selection",
                          return_value=tts_tool.NOUS_MANAGED_PROVIDER), \
             patch.object(tts_tool, "resolve_managed_tool_gateway",
                          return_value=None):
            with pytest.raises(ValueError) as exc:
                tts_tool._resolve_openai_audio_client_config()
        assert "not available" in str(exc.value)

    def test_selected_vendor_cfg_api_key_wins(self):
        config = {"openai": {"api_key": "cfg-key", "base_url": "https://x/v1"}}
        with patch.object(tts_tool, "_load_tts_config", return_value=config), \
             patch.object(tts_tool, "read_selection", return_value="openai"):
            assert tts_tool._resolve_openai_audio_client_config() == (
                "cfg-key", "https://x/v1", False,
            )

    def test_selected_vendor_no_credentials_raises(self):
        with patch.object(tts_tool, "_load_tts_config", return_value={}), \
             patch.object(tts_tool, "read_selection", return_value="openai"), \
             patch.object(tts_tool, "resolve_openai_audio_api_key",
                          return_value=""):
            with pytest.raises(ValueError) as exc:
                tts_tool._resolve_openai_audio_client_config()
        assert "neither tts.openai.api_key" in str(exc.value)

    def test_never_configured_cfg_api_key_wins(self):
        config = {"openai": {"api_key": "cfg-key"}}
        with patch.object(tts_tool, "_load_tts_config", return_value=config), \
             patch.object(tts_tool, "read_selection", return_value=None):
            assert tts_tool._resolve_openai_audio_client_config() == (
                "cfg-key", tts_tool.DEFAULT_OPENAI_BASE_URL, False,
            )


# ---------------------------------------------------------------------------
# text_to_speech_tool — residual output-path resolution in the wrapper
# ---------------------------------------------------------------------------
class TestTextToSpeechToolOutputPathResidual:
    def test_command_provider_output_path_extension_aligned(self, tmp_path):
        cfg = {"providers": {"my-cli": {
            "type": "command", "command": "echo hi", "format": "flac",
        }}}
        with patch.object(tts_tool, "_load_tts_config", return_value=cfg), \
             patch.object(tts_tool, "_text_to_speech_single") as single:
            single.side_effect = lambda text, output_path, **kw: _success_result(
                _write_audio(output_path), "my-cli",
            )
            result = json.loads(tts_tool.text_to_speech_tool(
                text="hi", output_path=str(tmp_path / "voice.wav"),
                provider="my-cli",
            ))
        assert result["success"] is True
        expected = str(tmp_path / "voice.flac")
        # The wrapper (3590) re-extensioned the caller's path to the config fmt.
        assert single.call_args[1]["output_path"] == expected

    def test_command_provider_default_output_dir_format(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tts_tool, "DEFAULT_OUTPUT_DIR", str(tmp_path))
        cfg = {"providers": {"my-cli": {
            "type": "command", "command": "echo hi", "format": "ogg",
        }}}
        with patch.object(tts_tool, "_load_tts_config", return_value=cfg), \
             patch.object(tts_tool, "_text_to_speech_single") as single:
            single.side_effect = lambda text, output_path, **kw: _success_result(
                _write_audio(output_path), "my-cli",
            )
            result = json.loads(tts_tool.text_to_speech_tool(
                text="hi", provider="my-cli",
            ))
        assert result["success"] is True
        assert single.call_args[1]["output_path"].endswith(".ogg")

    def test_default_output_dir_ogg_when_platform_wants_opus(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(tts_tool, "DEFAULT_OUTPUT_DIR", str(tmp_path))
        monkeypatch.setattr("gateway.session_context.get_session_env",
                            lambda name, default="": "telegram", raising=False)
        with patch.object(tts_tool, "_load_tts_config", return_value={}), \
             patch.object(tts_tool, "_text_to_speech_single") as single:
            single.side_effect = lambda text, output_path, **kw: _success_result(
                _write_audio(output_path), "openai",
            )
            result = json.loads(tts_tool.text_to_speech_tool(
                text="hi", provider="openai",
            ))
        assert result["success"] is True
        # The wrapper selected .ogg because the platform wants Opus.
        assert single.call_args[1]["output_path"].endswith(".ogg")


# ---------------------------------------------------------------------------
# _text_to_speech_single — edge RuntimeError re-run path
# ---------------------------------------------------------------------------
class TestTextToSpeechSingleEdgeRetry:
    def test_edge_threadpool_runtime_error_reruns_synchronously(self, tmp_path):
        out = str(tmp_path / "voice.mp3")

        async def _edge(text, output_path, tts_config):
            raise RuntimeError("boom")

        with patch.object(tts_tool, "_import_edge_tts",
                          return_value=MagicMock()), \
             patch.object(tts_tool, "_generate_edge_tts", side_effect=_edge):
            result = json.loads(tts_tool._text_to_speech_single(
                text="hello", output_path=out, provider="edge",
                tts_config_override={"provider": "edge"},
            ))
        # The synchronous re-run also raises, so the caller sees a generic error.
        assert result["success"] is False
        assert "TTS generation failed" in result["error"]
        assert "boom" in result["error"]
