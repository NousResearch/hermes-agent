"""Regression tests for #53259.

When TTS/STT packages (edge-tts, elevenlabs, mistralai) installed outside
the venv but importable on sys.path (e.g. via PYTHONPATH, Docker layered
filesystems), the lazy-import helpers must fall through to the raw import
instead of re-raising lazy_deps.ensure() failures as ImportError.

Uses sys.modules fixtures so builtins.__import__ stays intact — patching
__import__ replaces the helper's own ``from tools.lazy_deps import ...``
and defeats the purpose of the test.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from tools.lazy_deps import FeatureUnavailable


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
    def test_falls_through_on_lazy_deps_failure(self):
        """FeatureUnavailable from ensure() must not prevent raw import."""
        mock_edge_tts = MagicMock()
        with patch.dict(sys.modules, {"edge_tts": mock_edge_tts}), \
             patch("tools.lazy_deps.ensure",
                   side_effect=FeatureUnavailable("tts.edge", (), "test")):
            from tools.tts_tool import _import_edge_tts
            result = _import_edge_tts()
        assert result is mock_edge_tts

    def test_raises_when_package_truly_missing(self):
        """When the package is truly absent, ImportError must propagate."""
        with patch("tools.lazy_deps.ensure"), \
             patch.dict(sys.modules, {"edge_tts": None}):
            from tools.tts_tool import _import_edge_tts
            with pytest.raises(ImportError):
                _import_edge_tts()


class TestElevenLabsPythonpathFallback:
    def test_falls_through_on_lazy_deps_failure(self):
        """FeatureUnavailable from ensure() must not prevent raw import."""
        mock_cls = MagicMock()
        mock_client_pkg = MagicMock()
        mock_client_pkg.ElevenLabs = mock_cls
        with patch.dict(sys.modules, {
            "elevenlabs": mock_client_pkg,
            "elevenlabs.client": mock_client_pkg,
        }), patch("tools.lazy_deps.ensure",
                  side_effect=FeatureUnavailable("tts.elevenlabs", (), "test")):
            from tools.tts_tool import _import_elevenlabs
            result = _import_elevenlabs()
        assert result is mock_cls

    def test_raises_when_package_truly_missing(self):
        """When the package is truly absent, ImportError must propagate."""
        with patch("tools.lazy_deps.ensure"), \
             patch.dict(sys.modules, {"elevenlabs": None,
                                      "elevenlabs.client": None}):
            from tools.tts_tool import _import_elevenlabs
            with pytest.raises(ImportError):
                _import_elevenlabs()


class TestMistralPythonpathFallback:
    def test_falls_through_on_lazy_deps_failure(self):
        """FeatureUnavailable from ensure() must not prevent raw import."""
        mock_cls = MagicMock()
        mock_mistralai = MagicMock()
        mock_mistralai.Mistral = mock_cls
        with patch.dict(sys.modules, {
            "mistralai": mock_mistralai,
            "mistralai.client": mock_mistralai,
        }), patch("tools.lazy_deps.ensure",
                  side_effect=FeatureUnavailable("tts.mistral", (), "test")):
            from tools.tts_tool import _import_mistral_client
            result = _import_mistral_client()
        assert result is mock_cls

    def test_raises_when_package_truly_missing(self):
        """When the package is truly absent, ImportError must propagate."""
        with patch("tools.lazy_deps.ensure"), \
             patch.dict(sys.modules, {"mistralai": None,
                                      "mistralai.client": None}):
            from tools.tts_tool import _import_mistral_client
            with pytest.raises(ImportError):
                _import_mistral_client()


# ── STT: _transcribe_mistral fallthrough ───────────────────────────────────


class TestMistralSttPythonpathFallback:
    def test_transcribe_mistral_falls_through_on_lazy_deps_failure(
        self, tmp_path,
    ):
        """FeatureUnavailable from ensure('stt.mistral') must not block
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
        }), patch("tools.lazy_deps.ensure",
                  side_effect=FeatureUnavailable("stt.mistral", (), "test")), \
             patch("tools.transcription_tools.get_env_value",
                   return_value="test-key"):
            result = _transcribe_mistral(str(audio_file), "mistral-large-latest")

        assert result["success"] is True
        assert result["transcript"] == "hello world"
