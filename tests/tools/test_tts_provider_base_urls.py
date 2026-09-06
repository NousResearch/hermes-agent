"""Class-level base_url parity: every cloud TTS provider honors config base_url.

xAI, MiniMax, Gemini, OpenAI and DeepInfra already read
``tts.<provider>.base_url`` from config.yaml. This locks in the same
contract for the ElevenLabs and Mistral sections (the two that used to
hardcode the SDK default endpoint).
"""
from __future__ import annotations

import sys
import types
from unittest.mock import patch

import tools.tts_tool as tts


# ── ElevenLabs: base_url/wss_url → ElevenLabsEnvironment ──────────────────


def _fake_elevenlabs_environment_module(captured: dict):
    mod = types.ModuleType("elevenlabs.environment")

    class ElevenLabsEnvironment:
        def __init__(self, base, wss):
            captured["base"] = base
            captured["wss"] = wss

    mod.ElevenLabsEnvironment = ElevenLabsEnvironment
    pkg = types.ModuleType("elevenlabs")
    pkg.environment = mod
    return pkg, mod


def test_elevenlabs_no_base_url_uses_sdk_default_environment():
    from tools.tts_tool_providers import _elevenlabs_environment_kwargs

    assert _elevenlabs_environment_kwargs({}) == {}
    assert _elevenlabs_environment_kwargs({"base_url": ""}) == {}


# ── Mistral: tts.mistral.base_url → SDK server_url ────────────────────────


def test_mistral_no_base_url_omits_server_url(tmp_path):
    captured: dict = {}

    class _FakeMistral:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        class audio:  # noqa: N801
            class speech:  # noqa: N801
                @staticmethod
                def complete(**kwargs):
                    return types.SimpleNamespace(audio_data="aGVsbG8=")

    out = tmp_path / "out.mp3"
    with patch.object(tts, "_import_mistral_client", return_value=_FakeMistral), \
         patch.object(tts, "get_env_value", lambda k, *a: "key" if k == "MISTRAL_API_KEY" else None):
        tts._generate_mistral_tts("hi", str(out), {"mistral": {}})

    assert "server_url" not in captured
