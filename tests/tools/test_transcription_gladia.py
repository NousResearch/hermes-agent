"""Tests for the Gladia STT provider (pre-recorded via gladiaio-sdk)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def test_get_provider_gating_keys_on_gladia_api_key(monkeypatch):
    """Explicit-provider gate: GLADIA_API_KEY presence flips ``gladia`` on/off."""
    monkeypatch.delenv("GLADIA_API_KEY", raising=False)
    from tools.transcription_tools import _get_provider

    assert _get_provider({"provider": "gladia"}) == "none"
    monkeypatch.setenv("GLADIA_API_KEY", "test-key")
    assert _get_provider({"provider": "gladia"}) == "gladia"


def test_build_gladia_options_maps_single_language(monkeypatch):
    from tools.transcription_tools import _build_gladia_transcribe_options

    options = _build_gladia_transcribe_options(
        {"language": "en", "gladia": {"model": "solaria-1"}},
        "solaria-1",
    )
    assert options["model"] == "solaria-1"
    assert options["language_config"]["languages"] == ["en"]
    assert "code_switching" not in options.get("language_config", {})


def test_build_gladia_options_prefers_languages_list_and_diarization():
    from tools.transcription_tools import _build_gladia_transcribe_options

    options = _build_gladia_transcribe_options(
        {
            "language": "en",
            "gladia": {
                "languages": ["en", "fr", "es"],
                "code_switching": True,
                "diarization": True,
            },
        },
        "solaria-3",
    )
    assert options["model"] == "solaria-3"
    assert options["language_config"]["languages"] == ["en", "fr", "es"]
    assert options["language_config"]["code_switching"] is True
    assert options["diarization"] is True


def test_build_gladia_options_ignores_code_switching_without_languages(caplog):
    from tools.transcription_tools import _build_gladia_transcribe_options

    with caplog.at_level("WARNING"):
        options = _build_gladia_transcribe_options(
            {"language": "", "gladia": {"language": "", "code_switching": True, "languages": []}},
            "solaria-1",
        )
    assert "language_config" not in options or "code_switching" not in options.get(
        "language_config", {}
    )
    assert "code_switching" in caplog.text


def test_gladia_client_http_headers_include_hermes_version():
    from hermes_cli import __version__
    from tools.transcription_tools import _gladia_client_http_headers

    headers = _gladia_client_http_headers()
    assert headers == {"x-gladia-version": f"hermes-agent/{__version__}"}


def test_transcribe_gladia_happy_path(monkeypatch, tmp_path):
    monkeypatch.setenv("GLADIA_API_KEY", "test-key")
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"\x00" * 16)

    captured: dict = {}

    class _FakePrerecorded:
        def transcribe(self, path, options):
            captured["path"] = path
            captured["options"] = options
            return SimpleNamespace(
                result=SimpleNamespace(
                    transcription=SimpleNamespace(full_transcript="hello from gladia")
                )
            )

    class _FakeClient:
        def __init__(self, api_key=None, http_headers=None, **_kwargs):
            captured["api_key"] = api_key
            captured["http_headers"] = http_headers

        def prerecorded(self):
            return _FakePrerecorded()

    fake_sdk = MagicMock()
    fake_sdk.GladiaClient = _FakeClient

    stt_config = {
        "language": "en",
        "gladia": {"model": "solaria-1", "diarization": False},
    }

    with patch.dict("sys.modules", {"gladiaio_sdk": fake_sdk}), patch(
        "tools.transcription_tools._load_stt_config", return_value=stt_config
    ), patch("tools.lazy_deps.ensure", return_value=True):
        from hermes_cli import __version__
        from tools.transcription_tools import _transcribe_gladia

        result = _transcribe_gladia(str(audio), "solaria-1")

    assert result["success"] is True
    assert result["provider"] == "gladia"
    assert result["transcript"] == "hello from gladia"
    assert captured["api_key"] == "test-key"
    assert captured["http_headers"] == {
        "x-gladia-version": f"hermes-agent/{__version__}"
    }
    assert captured["options"]["language_config"]["languages"] == ["en"]


def test_transcribe_gladia_missing_key(monkeypatch, tmp_path):
    monkeypatch.delenv("GLADIA_API_KEY", raising=False)
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"\x00" * 16)

    from tools.transcription_tools import _transcribe_gladia

    result = _transcribe_gladia(str(audio), "solaria-1")
    assert result["success"] is False
    assert "GLADIA_API_KEY" in result["error"]


def test_transcribe_gladia_sdk_error(monkeypatch, tmp_path):
    monkeypatch.setenv("GLADIA_API_KEY", "test-key")
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"\x00" * 16)

    class _BoomClient:
        def __init__(self, **_kwargs):
            pass

        def prerecorded(self):
            raise RuntimeError("boom")

    fake_sdk = MagicMock()
    fake_sdk.GladiaClient = _BoomClient

    with patch.dict("sys.modules", {"gladiaio_sdk": fake_sdk}), patch(
        "tools.transcription_tools._load_stt_config", return_value={"gladia": {}}
    ), patch("tools.lazy_deps.ensure", return_value=True):
        from tools.transcription_tools import _transcribe_gladia

        result = _transcribe_gladia(str(audio), "solaria-1")

    assert result["success"] is False
    assert "Gladia STT transcription failed" in result["error"]


def test_dispatch_routes_gladia_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("GLADIA_API_KEY", "test-key")
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"\x00" * 16)

    with patch(
        "tools.transcription_tools._transcribe_gladia",
        return_value={"success": True, "transcript": "ok", "provider": "gladia"},
    ) as mocked, patch(
        "tools.transcription_tools._load_stt_config",
        return_value={"enabled": True, "provider": "gladia", "gladia": {"model": "solaria-1"}},
    ):
        from tools.transcription_tools import _transcribe_prepared_audio

        result = _transcribe_prepared_audio(str(audio))

    assert result["provider"] == "gladia"
    mocked.assert_called_once()
