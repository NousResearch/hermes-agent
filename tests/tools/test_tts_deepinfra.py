"""Tests for the DeepInfra TTS provider.

``_generate_deepinfra_tts`` is a thin shim that resolves credentials/model
then delegates to ``_generate_openai_tts``. These two tests pin the
delegation happy path and the no-hardcoded-fallback contract; shared
infrastructure (catalog fetch + tag filter) is covered in
``tests/hermes_cli/test_api_key_providers.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _isolation(monkeypatch):
    import hermes_cli.models as _models_mod
    monkeypatch.setattr(_models_mod, "_deepinfra_catalog_cache", {})
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test-key")
    yield


def test_raises_when_no_model_resolvable(monkeypatch, tmp_path):
    """No-fallback contract: empty config + unreachable catalog → ValueError."""
    import urllib.request
    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda *a, **kw: (_ for _ in ()).throw(Exception("offline")),
    )
    from tools.tts_tool import _generate_deepinfra_tts
    with pytest.raises(ValueError, match="No DeepInfra TTS model available"):
        _generate_deepinfra_tts("hi", str(tmp_path / "out.mp3"), {})


def test_delegates_to_openai_handler_with_deepinfra_creds(monkeypatch, tmp_path):
    """Happy path: pinned model → openai SDK invoked with DeepInfra base_url + key."""
    captured: dict = {}

    class _FakeClient:
        def __init__(self, api_key=None, base_url=None):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            speech = MagicMock()
            speech.create = MagicMock(return_value=MagicMock(stream_to_file=lambda p: None))
            self.audio = MagicMock(speech=speech)
        def close(self):
            pass

    with patch("tools.tts_tool._import_openai_client", return_value=_FakeClient):
        from tools.tts_tool import _generate_deepinfra_tts
        _generate_deepinfra_tts(
            "hello", str(tmp_path / "out.mp3"),
            {"deepinfra": {"model": "vendor/test-tts"}},
        )

    assert "deepinfra" in captured["base_url"]
    assert captured["api_key"] == "test-key"


def test_requirements_follow_explicit_deepinfra_provider(monkeypatch):
    from tools import tts_tool

    monkeypatch.setattr(
        tts_tool,
        "_load_tts_config",
        lambda: {"provider": "deepinfra", "deepinfra": {}},
    )
    monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: object)

    assert tts_tool.check_tts_requirements() is True


def test_unselected_cloud_credentials_do_not_expose_edge_tool(monkeypatch):
    from tools import tts_tool

    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {})
    monkeypatch.setattr(tts_tool, "_import_edge_tts", MagicMock(side_effect=ImportError))
    monkeypatch.setenv("OPENAI_API_KEY", "unselected-key")

    assert tts_tool.check_tts_requirements() is False
