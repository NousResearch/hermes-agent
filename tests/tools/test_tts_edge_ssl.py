"""Regression tests for custom CA handling in Edge TTS."""

import asyncio
import ssl
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _fake_edge_tts():
    communicate_context = MagicMock()
    voices_context = MagicMock()
    communicate = SimpleNamespace(_SSL_CTX=communicate_context)
    voices = SimpleNamespace(_SSL_CTX=voices_context)
    return SimpleNamespace(
        communicate=communicate,
        voices=voices,
        Communicate=MagicMock(),
    ), communicate_context, voices_context


def test_edge_ssl_adds_custom_ca_to_both_dependency_contexts(monkeypatch, tmp_path):
    from agent import ssl_verify
    from tools import tts_tool_providers

    ca_bundle = tmp_path / "corporate-ca.pem"
    ca_bundle.write_text("test certificate bundle", encoding="utf-8")
    edge_tts, communicate_context, voices_context = _fake_edge_tts()
    monkeypatch.setattr(ssl_verify, "resolve_ca_bundle_path", lambda: str(ca_bundle))

    tts_tool_providers._configure_edge_tts_ssl(edge_tts)

    communicate_context.load_verify_locations.assert_called_once_with(cafile=str(ca_bundle))
    voices_context.load_verify_locations.assert_called_once_with(cafile=str(ca_bundle))


def test_edge_ssl_preserves_dependency_defaults_without_custom_ca(monkeypatch):
    from agent import ssl_verify
    from tools import tts_tool_providers

    edge_tts, communicate_context, voices_context = _fake_edge_tts()
    monkeypatch.setattr(ssl_verify, "resolve_ca_bundle_path", lambda: None)

    tts_tool_providers._configure_edge_tts_ssl(edge_tts)

    communicate_context.load_verify_locations.assert_not_called()
    voices_context.load_verify_locations.assert_not_called()


def test_edge_ssl_fails_before_mutation_when_dependency_context_is_missing(
    monkeypatch, tmp_path
):
    from agent import ssl_verify
    from tools import tts_tool_providers

    ca_bundle = tmp_path / "corporate-ca.pem"
    ca_bundle.write_text("test certificate bundle", encoding="utf-8")
    edge_tts, communicate_context, _ = _fake_edge_tts()
    edge_tts.voices = SimpleNamespace()
    monkeypatch.setattr(ssl_verify, "resolve_ca_bundle_path", lambda: str(ca_bundle))

    with pytest.raises(RuntimeError, match="voices SSL context"):
        tts_tool_providers._configure_edge_tts_ssl(edge_tts)

    communicate_context.load_verify_locations.assert_not_called()


def test_pinned_edge_tts_exposes_verified_module_contexts():
    import edge_tts

    assert isinstance(edge_tts.communicate._SSL_CTX, ssl.SSLContext)
    assert isinstance(edge_tts.voices._SSL_CTX, ssl.SSLContext)


def test_generate_edge_tts_configures_ssl_before_synthesis(tmp_path, monkeypatch):
    from tools import tts_tool, tts_tool_providers

    communicate = MagicMock()
    communicate.save = AsyncMock()
    edge_tts = MagicMock()
    edge_tts.Communicate.return_value = communicate
    configure = MagicMock()
    monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: edge_tts)
    monkeypatch.setattr(tts_tool_providers, "_configure_edge_tts_ssl", configure)

    output_path = tmp_path / "out.mp3"
    asyncio.run(
        tts_tool_providers._generate_edge_tts("Hello", str(output_path), {})
    )

    configure.assert_called_once_with(edge_tts)
    edge_tts.Communicate.assert_called_once()
    communicate.save.assert_awaited_once_with(str(output_path))
