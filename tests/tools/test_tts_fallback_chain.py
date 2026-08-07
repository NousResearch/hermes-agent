"""Ordered TTS provider fallback chain — ``tts.fallback`` (#65752).

The chain is ``[provider] + fallback``. Each entry is a full attempt through
the real dispatcher (``_attempt_tts_provider``), so per-provider concerns —
the text-length cap, the output path and extension, Opus voice routing, and
Ogg container repair — stay per provider instead of leaking across entries.

Hermetic: provider backends are monkeypatched; no network, no ffmpeg, no
model downloads.
"""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from gateway.session_context import _UNSET, _VAR_MAP
from tools import tts_tool


def _reset_session_context() -> None:
    for var in _VAR_MAP.values():
        var.set(_UNSET)


@pytest.fixture(autouse=True)
def _clean_session_platform(monkeypatch):
    _reset_session_context()
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    yield
    _reset_session_context()


class TestChainResolution:
    def test_no_fallback_key_yields_single_entry(self):
        assert tts_tool._resolve_provider_chain("edge", {}) == ["edge"]

    def test_fallback_appended_in_order(self):
        chain = tts_tool._resolve_provider_chain(
            "openai", {"fallback": ["neutts", "edge"]},
        )
        assert chain == ["openai", "neutts", "edge"]

    def test_entries_normalized_and_deduped(self):
        chain = tts_tool._resolve_provider_chain(
            "edge", {"fallback": ["  EDGE ", "NeuTTS", "neutts"]},
        )
        assert chain == ["edge", "neutts"]

    def test_scalar_fallback_accepted(self):
        assert tts_tool._resolve_provider_chain("openai", {"fallback": "edge"}) == [
            "openai", "edge",
        ]

    def test_unknown_entry_is_a_config_error(self):
        """#65752: "Unknown names are a config error".

        Skipping looks friendlier but hides the failure the user needs to see:
        a typo would leave the chain shorter than written, so a primary outage
        falls through to the Edge default and the user never learns their
        fallback was never wired.
        """
        with pytest.raises(tts_tool.TTSFallbackConfigError) as exc:
            tts_tool._resolve_provider_chain(
                "edge", {"fallback": ["definitely-not-a-provider", "neutts"]},
            )
        msg = str(exc.value)
        # Only the bad entry is reported as unknown; `neutts` still appears
        # later in the message as one of the valid built-ins to choose from.
        unknown_part = msg.split("unknown provider(s):")[1].split(".")[0]
        assert "definitely-not-a-provider" in unknown_part
        assert "neutts" not in unknown_part
        assert "tts.commands" in msg                   # says how to declare one

    def test_every_unknown_entry_is_named_at_once(self):
        """One pass, not one error per run — fix the whole config in one go."""
        with pytest.raises(tts_tool.TTSFallbackConfigError) as exc:
            tts_tool._resolve_provider_chain("edge", {"fallback": ["nope-one", "nope-two"]})
        assert "nope-one" in str(exc.value) and "nope-two" in str(exc.value)

    def test_the_tool_reports_it_instead_of_synthesising(self, monkeypatch):
        """text_to_speech_tool must surface the config error, not fall through.

        Falling through to a working primary is exactly how the broken chain
        would stay invisible.
        """
        import json as _json

        monkeypatch.setattr(
            tts_tool, "_load_tts_config",
            lambda: {"provider": "edge", "fallback": ["definitely-not-a-provider"]},
        )
        called = []
        monkeypatch.setattr(
            tts_tool, "_attempt_tts_provider",
            lambda *a, **k: called.append(a) or "{}",
        )

        out = _json.loads(tts_tool.text_to_speech_tool("hello"))

        assert out.get("success") is False
        assert "definitely-not-a-provider" in out.get("error", "")
        assert called == [], "synthesis ran despite a misconfigured chain"

    def test_the_registry_probe_does_not_raise_on_a_bad_chain(self, monkeypatch):
        """check_tts_requirements runs during schema assembly for every tool."""
        monkeypatch.setattr(
            tts_tool, "_load_tts_config",
            lambda: {"provider": "edge", "fallback": ["definitely-not-a-provider"]},
        )
        monkeypatch.setattr(tts_tool, "_provider_is_available", lambda n, c: n == "edge")

        assert tts_tool.check_tts_requirements() is True, (
            "a typo'd fallback entry took the whole tool schema down"
        )

    def test_unknown_primary_is_kept_for_the_dispatcher_to_report(self):
        chain = tts_tool._resolve_provider_chain("definitely-not-a-provider", {})
        assert chain == ["definitely-not-a-provider"]

    def test_malformed_fallback_values_ignored(self):
        for bad in ({"fallback": 5}, {"fallback": None}, {"fallback": [1, None, ""]}):
            assert tts_tool._resolve_provider_chain("edge", bad) == ["edge"]


class TestChainFallThrough:
    def test_second_provider_used_when_first_fails(self, tmp_path, monkeypatch):
        out = tmp_path / "speech.mp3"

        def failing_openai(*_a, **_k):
            raise RuntimeError("quota exhausted")

        async def edge_writes(_text, output_path, _cfg):
            Path(output_path).write_bytes(b"mp3")
            return output_path

        monkeypatch.setattr(tts_tool, "_load_tts_config",
                            lambda: {"provider": "openai", "fallback": ["edge"]})
        monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: object())
        monkeypatch.setattr(tts_tool, "_generate_openai_tts", failing_openai)
        monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: object())
        monkeypatch.setattr(tts_tool, "_generate_edge_tts", edge_writes)

        result = json.loads(tts_tool.text_to_speech_tool("hello", output_path=str(out)))

        assert result["success"] is True
        assert result["provider"] == "edge"

    def test_all_failing_returns_one_aggregated_error(self, tmp_path, monkeypatch):
        def boom_openai(*_a, **_k):
            raise RuntimeError("quota exhausted")

        async def boom_edge(*_a, **_k):
            raise RuntimeError("network down")

        monkeypatch.setattr(tts_tool, "_load_tts_config",
                            lambda: {"provider": "openai", "fallback": ["edge"]})
        monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: object())
        monkeypatch.setattr(tts_tool, "_generate_openai_tts", boom_openai)
        monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: object())
        monkeypatch.setattr(tts_tool, "_generate_edge_tts", boom_edge)
        monkeypatch.setattr(tts_tool, "_check_neutts_available", lambda: False)

        result = json.loads(tts_tool.text_to_speech_tool(
            "hello", output_path=str(tmp_path / "s.mp3"),
        ))

        assert result["success"] is False
        assert "quota exhausted" in result["error"]
        assert "network down" in result["error"]

    def test_single_provider_error_shape_is_unchanged(self, tmp_path, monkeypatch):
        """No fallback configured ⇒ the dispatcher's own error, untouched."""
        def boom(*_a, **_k):
            raise RuntimeError("quota exhausted")

        monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "openai"})
        monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: object())
        monkeypatch.setattr(tts_tool, "_generate_openai_tts", boom)

        result = json.loads(tts_tool.text_to_speech_tool(
            "hello", output_path=str(tmp_path / "s.mp3"),
        ))

        assert result["success"] is False
        assert result["error"].startswith("TTS generation failed (openai)")
        assert "providers in the chain" not in result["error"]


class TestPerProviderIsolation:
    def test_failed_provider_truncation_does_not_shrink_next_input(self, tmp_path, monkeypatch):
        """A tight cap on the failing provider must not shrink the retry text."""
        seen = {}

        def strict_openai(text, *_a, **_k):
            seen["openai"] = len(text)
            raise RuntimeError("nope")

        async def edge_writes(text, output_path, _cfg):
            seen["edge"] = len(text)
            Path(output_path).write_bytes(b"mp3")
            return output_path

        monkeypatch.setattr(tts_tool, "_load_tts_config",
                            lambda: {"provider": "openai", "fallback": ["edge"]})
        monkeypatch.setattr(
            tts_tool, "_resolve_max_text_length",
            lambda provider, _cfg: 10 if provider == "openai" else 10_000,
        )
        monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: object())
        monkeypatch.setattr(tts_tool, "_generate_openai_tts", strict_openai)
        monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: object())
        monkeypatch.setattr(tts_tool, "_generate_edge_tts", edge_writes)

        long_text = "x" * 500
        tts_tool.text_to_speech_tool(long_text, output_path=str(tmp_path / "s.mp3"))

        assert seen["openai"] == 10
        assert seen["edge"] == 500, "the fallback must receive the untruncated text"

    def test_extension_resolved_per_provider_on_opus_platform(self, tmp_path, monkeypatch):
        """openai writes .ogg natively; the edge fallback must not inherit it."""
        seen = {}

        def failing_openai(_text, output_path, *_a, **_k):
            seen["openai"] = output_path
            raise RuntimeError("nope")

        async def edge_writes(_text, output_path, _cfg):
            seen["edge"] = output_path
            Path(output_path).write_bytes(b"mp3")
            return output_path

        monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
        monkeypatch.setattr(tts_tool, "_load_tts_config",
                            lambda: {"provider": "openai", "fallback": ["edge"]})
        monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: object())
        monkeypatch.setattr(tts_tool, "_generate_openai_tts", failing_openai)
        monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: object())
        monkeypatch.setattr(tts_tool, "_generate_edge_tts", edge_writes)
        monkeypatch.setattr(tts_tool, "_convert_to_opus", lambda p: None)

        tts_tool.text_to_speech_tool("hello")

        assert seen["openai"].endswith(".ogg")
        assert seen["edge"].endswith(".mp3"), (
            "edge must resolve its own extension, not reuse the failed "
            "provider's native-Opus path"
        )


class TestChainPreservesCurrentContracts:
    """The chain must not regress contracts main gained after this branch."""

    def test_opus_routing_covers_non_telegram_platforms(self, tmp_path, monkeypatch):
        """Matrix is in OPUS_VOICE_PLATFORMS — the fallback must honor it."""
        opus = tmp_path / "speech.ogg"

        def failing_openai(*_a, **_k):
            raise RuntimeError("nope")

        async def edge_writes(_text, output_path, _cfg):
            Path(output_path).write_bytes(b"mp3")
            return output_path

        def fake_convert(path):
            opus.write_bytes(b"OggS")
            return str(opus)

        monkeypatch.setenv("HERMES_SESSION_PLATFORM", "matrix")
        monkeypatch.setattr(tts_tool, "_load_tts_config",
                            lambda: {"provider": "openai", "fallback": ["edge"]})
        monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: object())
        monkeypatch.setattr(tts_tool, "_generate_openai_tts", failing_openai)
        monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: object())
        monkeypatch.setattr(tts_tool, "_generate_edge_tts", edge_writes)
        monkeypatch.setattr(tts_tool, "_convert_to_opus", Mock(side_effect=fake_convert))

        result = json.loads(tts_tool.text_to_speech_tool("hello"))

        assert result["success"] is True
        assert result["provider"] == "edge"
        assert result["voice_compatible"] is True, (
            "Matrix must get a voice bubble, not an MP3 attachment"
        )
        assert result["media_tag"].startswith("[[audio_as_voice]]")

    def test_container_repair_runs_on_the_fallback_attempt(self, tmp_path, monkeypatch):
        """A fallback writing MP3 bytes to a .ogg path must still be repaired."""
        repair = Mock(side_effect=lambda p: p)

        def failing_openai(*_a, **_k):
            raise RuntimeError("nope")

        async def edge_writes(_text, output_path, _cfg):
            Path(output_path).write_bytes(b"ID3mp3-bytes")
            return output_path

        monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
        monkeypatch.setattr(tts_tool, "_load_tts_config",
                            lambda: {"provider": "openai", "fallback": ["edge"]})
        monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: object())
        monkeypatch.setattr(tts_tool, "_generate_openai_tts", failing_openai)
        monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: object())
        monkeypatch.setattr(tts_tool, "_generate_edge_tts", edge_writes)
        monkeypatch.setattr(tts_tool, "_repair_ogg_container", repair)
        monkeypatch.setattr(tts_tool, "_convert_to_opus", lambda p: None)

        tts_tool.text_to_speech_tool("hello", output_path=str(tmp_path / "s.mp3"))

        repair.assert_called_once()

    def test_instructions_and_speed_reach_the_fallback_provider(self, tmp_path, monkeypatch):
        """speed folds into tts_config; instructions forward as a kwarg."""
        seen = {}

        async def failing_edge(*_a, **_k):
            raise RuntimeError("nope")

        def openai_writes(_text, output_path, cfg, instructions=None):
            seen["instructions"] = instructions
            seen["speed"] = cfg.get("speed")
            Path(output_path).write_bytes(b"mp3")
            return output_path

        monkeypatch.setattr(tts_tool, "_load_tts_config",
                            lambda: {"provider": "edge", "fallback": ["openai"]})
        monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: object())
        monkeypatch.setattr(tts_tool, "_generate_edge_tts", failing_edge)
        monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: object())
        monkeypatch.setattr(tts_tool, "_generate_openai_tts", openai_writes)

        tts_tool.text_to_speech_tool(
            "hello", output_path=str(tmp_path / "s.mp3"),
            speed=1.5, instructions="speak warmly",
        )

        assert seen["instructions"] == "speak warmly"
        assert seen["speed"] == 1.5


class TestRequirementsCheck:
    def test_chain_available_when_only_the_fallback_is(self, monkeypatch):
        """Primary dead, fallback alive ⇒ the tool schema stays exposed."""
        monkeypatch.setattr(tts_tool, "_load_tts_config",
                            lambda: {"provider": "openai", "fallback": ["neutts"]})
        monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: object())
        monkeypatch.setattr(tts_tool, "_has_openai_audio_backend", lambda: False)
        monkeypatch.setattr(tts_tool, "_check_neutts_available", lambda: True)

        assert tts_tool.check_tts_requirements() is True

    def test_chain_unavailable_when_every_entry_is(self, monkeypatch):
        monkeypatch.setattr(tts_tool, "_load_tts_config",
                            lambda: {"provider": "openai", "fallback": ["neutts"]})
        monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: object())
        monkeypatch.setattr(tts_tool, "_has_openai_audio_backend", lambda: False)
        monkeypatch.setattr(tts_tool, "_check_neutts_available", lambda: False)

        assert tts_tool.check_tts_requirements() is False

    def test_availability_uses_the_real_credential_resolvers(self, monkeypatch):
        """Not raw env-key sniffing — the resolver decides (#65752 review)."""
        calls = []

        def fake_resolve(env_key, provider_name):
            calls.append((env_key, provider_name))
            return "sk-present"

        monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "elevenlabs"})
        monkeypatch.setattr(tts_tool, "_import_elevenlabs", lambda: object())
        monkeypatch.setattr(tts_tool, "_resolve_provider_key", fake_resolve)

        assert tts_tool.check_tts_requirements() is True
        assert ("ELEVENLABS_API_KEY", "elevenlabs") in calls
