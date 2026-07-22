"""Behavior contracts for single-provider TTS dispatch.

``text_to_speech_tool()`` delegates one provider's synthesis to
``_synthesize_with_provider()``. These tests pin the observable tool
behavior across that seam: the effective-provider contract (the Edge
default falls back to NeuTTS and the response must say so), actionable
missing-SDK errors, exception messages labeled with the provider that
actually ran, empty-output detection, and the success response shape.
All assertions go through the public tool — no internals are inspected.
"""

from __future__ import annotations

import json
import os

import pytest


@pytest.fixture(autouse=True)
def _default_config(monkeypatch):
    """Pin an empty tts config so the host machine's config.yaml is inert."""
    import tools.tts_tool as tts_tool

    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {})
    yield


def _run_tool(tmp_path, filename="out.mp3"):
    from tools.tts_tool import text_to_speech_tool

    out = tmp_path / filename
    return json.loads(text_to_speech_tool("hello world", output_path=str(out)))


def _write_audio(path: str) -> str:
    with open(path, "wb") as fh:
        fh.write(b"fake-audio-bytes")
    return path


def test_edge_default_falls_back_to_neutts_and_reports_it(monkeypatch, tmp_path):
    """Edge unavailable + NeuTTS available ⇒ success, provider says 'neutts'.

    Downstream format/delivery decisions key off the provider that actually
    ran, so the response must reflect the fallback, not the configured name.
    """
    import tools.tts_tool as tts_tool

    monkeypatch.setattr(
        tts_tool, "_import_edge_tts",
        lambda: (_ for _ in ()).throw(ImportError("no edge-tts")),
    )
    monkeypatch.setattr(tts_tool, "_check_neutts_available", lambda: True)
    monkeypatch.setattr(
        tts_tool, "_generate_neutts",
        lambda text, file_str, cfg: _write_audio(file_str),
    )

    data = _run_tool(tmp_path)

    assert data["success"] is True
    assert data["provider"] == "neutts"


def test_missing_sdk_returns_actionable_error(monkeypatch, tmp_path):
    """Configured provider whose SDK isn't importable ⇒ install hint, no crash."""
    import tools.tts_tool as tts_tool

    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "elevenlabs"})
    monkeypatch.setattr(
        tts_tool, "_import_elevenlabs",
        lambda: (_ for _ in ()).throw(ImportError("not installed")),
    )

    data = _run_tool(tmp_path)

    assert data["success"] is False
    assert "pip install elevenlabs" in data["error"]


def test_generator_exception_is_labeled_with_provider(monkeypatch, tmp_path):
    """Unexpected generator failure ⇒ error names the provider that ran."""
    import tools.tts_tool as tts_tool

    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "minimax"})
    monkeypatch.setattr(
        tts_tool, "_generate_minimax_tts",
        lambda text, file_str, cfg: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    data = _run_tool(tmp_path)

    assert data["success"] is False
    assert "TTS generation failed (minimax)" in data["error"]
    assert "boom" in data["error"]


def test_config_error_is_labeled_as_configuration(monkeypatch, tmp_path):
    """ValueError from a provider ⇒ configuration-error message, not generic."""
    import tools.tts_tool as tts_tool

    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "minimax"})
    monkeypatch.setattr(
        tts_tool, "_generate_minimax_tts",
        lambda text, file_str, cfg: (_ for _ in ()).throw(ValueError("MINIMAX_API_KEY missing")),
    )

    data = _run_tool(tmp_path)

    assert data["success"] is False
    assert "TTS configuration error (minimax)" in data["error"]


def test_empty_output_is_a_failure(monkeypatch, tmp_path):
    """Generator that produces no file ⇒ explicit no-output failure."""
    import tools.tts_tool as tts_tool

    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "minimax"})
    monkeypatch.setattr(
        tts_tool, "_generate_minimax_tts",
        lambda text, file_str, cfg: None,
    )

    data = _run_tool(tmp_path)

    assert data["success"] is False
    assert "produced no output" in data["error"]
    assert "minimax" in data["error"]


def test_success_response_contract(monkeypatch, tmp_path):
    """Happy path ⇒ file on disk, MEDIA tag, provider echoed, no voice flag off-platform."""
    import tools.tts_tool as tts_tool

    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "minimax"})
    monkeypatch.setattr(
        tts_tool, "_generate_minimax_tts",
        lambda text, file_str, cfg: _write_audio(file_str),
    )

    data = _run_tool(tmp_path)

    assert data["success"] is True
    assert data["provider"] == "minimax"
    assert os.path.getsize(data["file_path"]) > 0
    assert data["media_tag"].startswith("MEDIA:")
    assert data["voice_compatible"] is False


def test_nothing_available_reports_setup_guidance(monkeypatch, tmp_path):
    """Edge and NeuTTS both unavailable ⇒ actionable setup error."""
    import tools.tts_tool as tts_tool

    monkeypatch.setattr(
        tts_tool, "_import_edge_tts",
        lambda: (_ for _ in ()).throw(ImportError("no edge-tts")),
    )
    monkeypatch.setattr(tts_tool, "_check_neutts_available", lambda: False)

    data = _run_tool(tmp_path)

    assert data["success"] is False
    assert "No TTS provider available" in data["error"]
