"""Regression tests for the TTS opt-in fallback chain (tools/tts_tool.py).

Covers the chain-loop bugs found in review of the fallback-chain PR:
- per-provider text truncation must not leak into the next provider (data loss),
- a failed provider's output path must not pollute the next provider's path,
- the output extension is resolved per-provider (.ogg for native-Opus on
  Telegram, a command provider's configured format, else .mp3) — not a
  hardcoded .mp3 for the whole chain,
- and the happy-path chain fallback (primary fails → next succeeds).

All hermetic: the synthesis boundary (`_synthesize_with_provider`) is faked,
so no real TTS engine, network, or ffmpeg is touched.
"""

from __future__ import annotations

import json
import os

import pytest

import tools.tts_tool as tts


@pytest.fixture(autouse=True)
def _no_platform(monkeypatch):
    # Default: non-Telegram, so want_opus is False unless a test opts in.
    monkeypatch.setattr(
        "gateway.session_context.get_session_env",
        lambda key, default="": default,
        raising=False,
    )
    yield


def _config(provider: str, fallback=None) -> dict:
    cfg = {"provider": provider}
    if fallback is not None:
        cfg["fallback"] = fallback
    return cfg


def _install(monkeypatch, cfg, synth):
    monkeypatch.setattr(tts, "_load_tts_config", lambda: cfg)
    monkeypatch.setattr(tts, "_synthesize_with_provider", synth)


def test_chain_falls_through_to_second_provider(monkeypatch, tmp_path):
    calls = []

    def fake_synth(provider, text, file_str, tts_config, want_opus):
        calls.append(provider)
        if provider == "openai":
            raise RuntimeError("primary down")
        # edge succeeds: actually create the file so the size check passes.
        with open(file_str, "wb") as fh:
            fh.write(b"audio-bytes")
        return file_str

    _install(monkeypatch, _config("openai", ["edge"]), fake_synth)

    result = json.loads(tts.text_to_speech_tool("hello", output_path=str(tmp_path / "out.mp3")))

    assert result["success"] is True
    assert result["provider"] == "edge"
    assert calls == ["openai", "edge"]


def test_truncation_does_not_leak_into_next_provider(monkeypatch, tmp_path):
    seen_lengths = {}

    def fake_max_len(provider, tts_config):
        return 5 if provider == "openai" else 10_000

    def fake_synth(provider, text, file_str, tts_config, want_opus):
        seen_lengths[provider] = len(text)
        if provider == "openai":
            raise RuntimeError("primary down after truncating")
        with open(file_str, "wb") as fh:
            fh.write(b"x")
        return file_str

    monkeypatch.setattr(tts, "_resolve_max_text_length", fake_max_len)
    _install(monkeypatch, _config("openai", ["edge"]), fake_synth)

    long_text = "abcdefghijklmnop"  # 16 chars
    result = json.loads(tts.text_to_speech_tool(long_text, output_path=str(tmp_path / "o.mp3")))

    assert result["success"] is True
    # openai truncated to its 5-char cap; edge must still see the FULL text.
    assert seen_lengths["openai"] == 5
    assert seen_lengths["edge"] == 16


def test_failed_provider_path_does_not_pollute_next(monkeypatch, tmp_path):
    seen_paths = {}

    def fake_synth(provider, text, file_str, tts_config, want_opus):
        seen_paths[provider] = file_str
        if provider == "openai":
            # Simulate a plugin-style provider that returns a custom path but
            # never actually writes it → the size check raises.
            return str(tmp_path / "primary_custom_path.dat")
        with open(file_str, "wb") as fh:
            fh.write(b"x")
        return file_str

    _install(monkeypatch, _config("openai", ["edge"]), fake_synth)

    base = str(tmp_path / "shared.mp3")
    result = json.loads(tts.text_to_speech_tool("hi", output_path=base))

    assert result["success"] is True
    # edge must receive the ORIGINAL base path, not openai's failed custom path.
    assert seen_paths["openai"] == base
    assert seen_paths["edge"] == base


def test_native_opus_provider_gets_ogg_extension_on_telegram(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "gateway.session_context.get_session_env",
        lambda key, default="": "telegram" if key == "HERMES_SESSION_PLATFORM" else default,
        raising=False,
    )
    seen = {}

    def fake_synth(provider, text, file_str, tts_config, want_opus):
        seen[provider] = file_str
        with open(file_str, "wb") as fh:
            fh.write(b"x")
        return file_str

    # No output_path → default dir; extension is resolved per-provider.
    monkeypatch.setattr(tts, "DEFAULT_OUTPUT_DIR", str(tmp_path))
    _install(monkeypatch, _config("openai"), fake_synth)

    result = json.loads(tts.text_to_speech_tool("hi"))

    assert result["success"] is True
    # openai is a native-Opus provider: on Telegram it must write .ogg, not .mp3.
    assert seen["openai"].endswith(".ogg")
    assert result["voice_compatible"] is True


def test_default_provider_gets_mp3_extension_off_telegram(monkeypatch, tmp_path):
    seen = {}

    def fake_synth(provider, text, file_str, tts_config, want_opus):
        seen[provider] = file_str
        with open(file_str, "wb") as fh:
            fh.write(b"x")
        return file_str

    monkeypatch.setattr(tts, "DEFAULT_OUTPUT_DIR", str(tmp_path))
    _install(monkeypatch, _config("edge"), fake_synth)

    result = json.loads(tts.text_to_speech_tool("hi"))

    assert result["success"] is True
    assert seen["edge"].endswith(".mp3")


def test_all_providers_failing_aggregates_errors(monkeypatch, tmp_path):
    def fake_synth(provider, text, file_str, tts_config, want_opus):
        raise RuntimeError(f"{provider} boom")

    _install(monkeypatch, _config("openai", ["edge"]), fake_synth)

    result = json.loads(tts.text_to_speech_tool("hi", output_path=str(tmp_path / "o.mp3")))

    assert result["success"] is False
    # Multi-provider chain → aggregated error naming each attempt.
    assert "openai" in result["error"] and "edge" in result["error"]


def test_single_provider_chain_preserves_legacy_error_shape(monkeypatch, tmp_path):
    def fake_synth(provider, text, file_str, tts_config, want_opus):
        raise RuntimeError("engine offline")

    _install(monkeypatch, _config("edge"), fake_synth)

    result = json.loads(tts.text_to_speech_tool("hi", output_path=str(tmp_path / "o.mp3")))

    assert result["success"] is False
    assert "TTS generation failed" in result["error"]
