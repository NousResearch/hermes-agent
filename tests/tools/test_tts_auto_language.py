"""Behavior contracts for Edge TTS automatic language voice routing."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from tools import tts_tool, tts_tool_language


def _configured(**edge_overrides):
    edge = {
        "voice": "fallback-voice",
        "auto_language": True,
        "voice_by_language": {
            "en": "english-voice",
            "fr": "french-voice",
            "pt": "portuguese-voice",
            "pt-br": "brazilian-voice",
            "ZH_CN": "simplified-chinese-voice",
        },
    }
    edge.update(edge_overrides)
    return {"provider": "edge", "edge": edge}


@pytest.mark.parametrize(
    ("detected", "mapping", "expected_voice"),
    [
        ("pt-br", {"pt": "base", "pt-br": "exact"}, "exact"),
        ("pt-pt", {"pt": "base"}, "base"),
        ("zh-cn", {"ZH_CN": "normalized"}, "normalized"),
        ("de", {"fr": "french"}, "fallback-voice"),
        ("de", {"de": "", "fr": 123}, "fallback-voice"),
    ],
)
def test_edge_voice_resolution_prefers_exact_then_base_without_mutation(
    monkeypatch, detected, mapping, expected_voice
):
    config = _configured(voice_by_language=mapping)
    original = copy.deepcopy(config)
    monkeypatch.setattr(
        tts_tool_language,
        "_detect_dominant_language",
        lambda _text: (detected, 0.99),
    )

    updated = tts_tool_language.with_edge_auto_language_voice(
        config, "This response contains enough alphabetic characters"
    )

    assert updated["edge"]["voice"] == expected_voice
    assert config == original


@pytest.mark.parametrize(
    (
        "auto_language",
        "text",
        "ensure_error",
        "probability",
        "expected_voice",
        "detect_calls",
    ),
    [
        (True, "Ceci est une réponse française complète. " * 4, None, 0.99, "french-voice", 1),
        (True, "This mixed response remains ambiguous. " * 4, None, 0.79, "fallback-voice", 1),
        (True, "Bonjour", None, 0.99, "fallback-voice", 0),
        (
            True,
            "Ceci est une réponse française complète. " * 4,
            RuntimeError("disabled"),
            0.99,
            "fallback-voice",
            0,
        ),
        (
            False,
            "Ceci est une réponse française complète. " * 4,
            None,
            0.99,
            "fallback-voice",
            0,
        ),
    ],
)
def test_edge_detection_is_lazy_deterministic_and_applied_once_to_all_chunks(
    tmp_path,
    monkeypatch,
    auto_language,
    text,
    ensure_error,
    probability,
    expected_voice,
    detect_calls,
):
    config = _configured(auto_language=auto_language, max_text_length=32)
    original = copy.deepcopy(config)
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: config)

    ensure = MagicMock(side_effect=ensure_error)
    monkeypatch.setattr("tools.lazy_deps.ensure", ensure)
    detector_factory = SimpleNamespace(seed=None)
    detect_langs = MagicMock(
        return_value=[SimpleNamespace(lang="FR", prob=probability)]
    )
    monkeypatch.setitem(
        sys.modules,
        "langdetect",
        SimpleNamespace(DetectorFactory=detector_factory, detect_langs=detect_langs),
    )

    mock_communicate = MagicMock()

    def communicate(_text, **_kwargs):
        result = MagicMock()

        async def save_audio(path):
            Path(path).write_bytes(b"ID3audio")

        result.save = AsyncMock(side_effect=save_audio)
        return result

    mock_communicate.side_effect = communicate
    monkeypatch.setattr(
        tts_tool,
        "_import_edge_tts",
        lambda: SimpleNamespace(Communicate=mock_communicate),
    )

    result = json.loads(
        tts_tool.text_to_speech_tool(
            text, output_path=str(tmp_path / "reply.mp3")
        )
    )

    assert result["success"] is True
    assert {call.kwargs["voice"] for call in mock_communicate.call_args_list} == {
        expected_voice
    }
    if len(text.strip()) > 32:
        assert result["chunk_count"] > 1
    assert config == original
    assert detect_langs.call_count == detect_calls
    if detect_calls:
        assert detector_factory.seed == 0
        detect_langs.assert_called_once_with(text.strip())
    if not auto_language or len(text) < tts_tool_language.MIN_LANGUAGE_LETTERS:
        ensure.assert_not_called()
