"""Regression tests for #53259.

When TTS/STT packages (edge-tts, elevenlabs, mistralai) installed outside
the venv but importable on sys.path (e.g. via PYTHONPATH, Docker layered
filesystems), the lazy-import helpers must fall through to the raw import
instead of re-raising dependency-provisioning failures as ImportError.

The PM branch provisions through pm.ensure_import (replacing the old
pm.ensure_import provisioning); the helper must swallow its failure the
same way. Uses sys.modules fixtures so builtins.__import__ stays intact.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from pm.extras import ensure_import as _pm_ensure_import


@pytest.fixture(autouse=True)
def _clean_tts_modules():
    """Remove TTS packages from sys.modules so each test starts fresh."""
    removed = {}
    for name in ("edge_tts", "elevenlabs", "elevenlabs.client",
                 "mistralai", "mistralai.client"):
        if name in sys.modules:
            removed[name] = sys.modules.pop(name)
    yield
    for name in ("edge_tts", "elevenlabs", "elevenlabs.client",
                 "mistralai", "mistralai.client"):
        sys.modules.pop(name, None)
    sys.modules.update(removed)


class TestEdgeTtsPythonpathFallback:
    def test_falls_through_on_provisioning_failure(self):
        """A provisioning failure must not prevent raw import."""
        mock_edge_tts = MagicMock()
        with patch.dict(sys.modules, {"edge_tts": mock_edge_tts}), \
             patch("pm.extras.ensure_import", side_effect=RuntimeError("provisioning failed")):
            from tools.tts_tool import _import_edge_tts
            result = _import_edge_tts()
        assert result is mock_edge_tts

    def test_raises_when_package_truly_missing(self):
        """When the package is truly absent, ImportError must propagate."""
        with patch("pm.extras.ensure_import"), \
             patch.dict(sys.modules, {"edge_tts": None}):
            from tools.tts_tool import _import_edge_tts
            with pytest.raises(ImportError):
                _import_edge_tts()


class TestElevenLabsPythonpathFallback:
    def test_falls_through_on_provisioning_failure(self):
        """A provisioning failure must not prevent raw import."""
        mock_cls = MagicMock()
        mock_client_pkg = MagicMock()
        mock_client_pkg.ElevenLabs = mock_cls
        with patch.dict(sys.modules, {
            "elevenlabs": mock_client_pkg,
            "elevenlabs.client": mock_client_pkg,
        }), patch("pm.extras.ensure_import", side_effect=RuntimeError("provisioning failed")):
            from tools.tts_tool import _import_elevenlabs
            result = _import_elevenlabs()
        assert result is mock_cls

    def test_raises_when_package_truly_missing(self):
        """When the package is truly absent, ImportError must propagate."""
        with patch("pm.extras.ensure_import"), \
             patch.dict(sys.modules, {"elevenlabs": None,
                                      "elevenlabs.client": None}):
            from tools.tts_tool import _import_elevenlabs
            with pytest.raises(ImportError):
                _import_elevenlabs()


class TestMistralPythonpathFallback:
    def test_falls_through_on_provisioning_failure(self):
        """A provisioning failure must not prevent raw import."""
        mock_cls = MagicMock()
        mock_mistralai = MagicMock()
        mock_mistralai.Mistral = mock_cls
        with patch.dict(sys.modules, {
            "mistralai": mock_mistralai,
            "mistralai.client": mock_mistralai,
        }), patch("pm.extras.ensure_import", side_effect=RuntimeError("provisioning failed")):
            from tools.tts_tool import _import_mistral_client
            result = _import_mistral_client()
        assert result is mock_cls

    def test_raises_when_package_truly_missing(self):
        """When the package is truly absent, ImportError must propagate."""
        with patch("pm.extras.ensure_import"), \
             patch.dict(sys.modules, {"mistralai": None,
                                      "mistralai.client": None}):
            from tools.tts_tool import _import_mistral_client
            with pytest.raises(ImportError):
                _import_mistral_client()


# ── STT: _transcribe_mistral fallthrough ───────────────────────────────────


class TestMistralSttPythonpathFallback:
    def test_transcribe_mistral_falls_through_on_provisioning_failure(
        self, tmp_path,
    ):
        """A provisioning failure for stt.mistral must not block
        transcription when mistralai is importable via PYTHONPATH."""
        from tools.transcription_tools import _transcribe_mistral

        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"fake-audio")

        mock_client_cls = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "hello world"
        mock_client_cls.return_value.__enter__ = MagicMock(
            return_value=MagicMock(
                audio=MagicMock(
                    transcriptions=MagicMock(
                        complete=MagicMock(return_value=mock_result),
                    ),
                ),
            ),
        )
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        mock_mistralai = MagicMock()
        mock_mistralai.Mistral = mock_client_cls

        with patch.dict(sys.modules, {
            "mistralai": mock_mistralai,
            "mistralai.client": mock_mistralai,
        }), patch("pm.extras.ensure_import", side_effect=RuntimeError("provisioning failed")), \
             patch("hermes_cli.config.get_env_value",
                   return_value="test-key"):
            result = _transcribe_mistral(str(audio_file), "mistral-large-latest")

        assert result["success"] is True
        assert result["transcript"] == "hello world"
