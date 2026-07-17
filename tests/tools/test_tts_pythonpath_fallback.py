"""Regression tests for #53259.

TTS lazy-import helpers (``_import_edge_tts``, ``_import_elevenlabs``,
``_import_mistral_client``) must fall through to the raw ``import`` when
``lazy_deps.ensure()`` fails — e.g. because the venv is read-only but the
package is importable on ``sys.path`` via ``PYTHONPATH``.

Before the fix, ``except Exception as e: raise ImportError(str(e))`` killed
execution before the raw import could run.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestLazyDepsFallback:
    """When ``lazy_deps.ensure()`` raises any exception (typically
    ``FeatureUnavailable``), the import helpers must NOT re-raise.
    They should fall through to the raw ``import`` below.
    """

    def test_edge_tts_falls_through_on_lazy_deps_failure(self):
        from tools.lazy_deps import FeatureUnavailable
        from tools import tts_tool

        with patch(
            "tools.lazy_deps.ensure",
            side_effect=FeatureUnavailable("tts.edge", ("edge-tts",), "lazy install disabled"),
        ), patch.dict("sys.modules", {"edge_tts": MagicMock()}):
            result = tts_tool._import_edge_tts()

        assert result is not None

    def test_elevenlabs_falls_through_on_lazy_deps_failure(self):
        from tools.lazy_deps import FeatureUnavailable
        from tools import tts_tool

        mock_client = MagicMock()
        with patch(
            "tools.lazy_deps.ensure",
            side_effect=FeatureUnavailable("tts.elevenlabs", ("elevenlabs",), "lazy install disabled"),
        ), patch.dict(
            "sys.modules",
            {"elevenlabs": MagicMock(), "elevenlabs.client": MagicMock()},
        ):
            with patch("builtins.__import__") as mock_import:
                def _side_effect(name, *args, **kwargs):
                    if name == "elevenlabs.client":
                        mod = MagicMock()
                        mod.ElevenLabs = mock_client
                        return mod
                    return MagicMock()

                mock_import.side_effect = _side_effect
                result = tts_tool._import_elevenlabs()

        assert result is not None

    def test_mistral_falls_through_on_lazy_deps_failure(self):
        from tools.lazy_deps import FeatureUnavailable
        from tools import tts_tool

        mock_mistral = MagicMock()
        with patch(
            "tools.lazy_deps.ensure",
            side_effect=FeatureUnavailable("tts.mistral", ("mistralai",), "lazy install disabled"),
        ), patch.dict(
            "sys.modules",
            {"mistralai": MagicMock(), "mistralai.client": MagicMock()},
        ):
            with patch("builtins.__import__") as mock_import:
                def _side_effect(name, *args, **kwargs):
                    if name == "mistralai.client":
                        mod = MagicMock()
                        mod.Mistral = mock_mistral
                        return mod
                    return MagicMock()

                mock_import.side_effect = _side_effect
                result = tts_tool._import_mistral_client()

        assert result is not None

    def test_edge_tts_raises_when_package_truly_missing(self):
        """When the package is NOT on sys.path, the raw import must still
        raise ImportError — we should not swallow real import failures.
        """
        from tools import tts_tool

        with patch.dict("sys.modules", {"edge_tts": None}):
            with pytest.raises(ImportError):
                tts_tool._import_edge_tts()

    def test_elevenlabs_raises_when_package_truly_missing(self):
        from tools import tts_tool

        with patch.dict("sys.modules", {"elevenlabs": None}):
            with pytest.raises(ImportError):
                tts_tool._import_elevenlabs()

    def test_mistral_raises_when_package_truly_missing(self):
        from tools import tts_tool

        with patch.dict("sys.modules", {"mistralai": None}):
            with pytest.raises(ImportError):
                tts_tool._import_mistral_client()
