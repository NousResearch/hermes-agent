"""Tests for piper TTS lazy-install integration (issue #44001).

Verifies that:
  1. ``tts.piper`` is registered in ``tools/lazy_deps.py`` LAZY_DEPS.
  2. ``_import_piper()`` calls ``ensure("tts.piper")`` before importing.
  3. The ``piper`` entry in ``_check_piper_available`` still works when
     the package IS importable (regression guard).
"""

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# 1. Feature registration
# ---------------------------------------------------------------------------

class TestPiperLazyDepsRegistration:
    """``tts.piper`` must exist in the LAZY_DEPS map."""

    def test_tts_piper_feature_exists(self):
        from tools.lazy_deps import LAZY_DEPS
        assert "tts.piper" in LAZY_DEPS

    def test_tts_piper_installs_piper_tts(self):
        from tools.lazy_deps import LAZY_DEPS
        pkgs = LAZY_DEPS["tts.piper"]
        assert any("piper-tts" in p for p in pkgs), (
            f"Expected 'piper-tts' in tts.piper packages, got {pkgs}"
        )


# ---------------------------------------------------------------------------
# 2. _import_piper calls ensure()
# ---------------------------------------------------------------------------

class TestImportPiperLazyInstall:
    """``_import_piper()`` must call ``ensure("tts.piper")`` before importing."""

    def test_import_piper_calls_ensure(self):
        """ensure() is invoked with feature='tts.piper' and prompt=False."""
        mock_ensure = MagicMock()
        mock_piper_cls = MagicMock()

        with patch("tools.lazy_deps.ensure", mock_ensure), \
             patch.dict("sys.modules", {"piper": MagicMock(PiperVoice=mock_piper_cls)}):
            from tools.tts_tool import _import_piper
            result = _import_piper()

        mock_ensure.assert_called_once_with("tts.piper", prompt=False)
        assert result is not None

    def test_import_piper_ensure_failure_raises_import_error(self):
        """If ensure() raises FeatureUnavailable, _import_piper raises ImportError."""
        from tools.lazy_deps import FeatureUnavailable

        with patch("tools.lazy_deps.ensure", side_effect=FeatureUnavailable("tts.piper", ("piper-tts",), "blocked")):
            from tools.tts_tool import _import_piper
            with pytest.raises(ImportError):
                _import_piper()

    def test_import_piper_ensure_importerror_falls_through(self):
        """If lazy_deps itself is missing (ImportError), fall through to raw import."""
        import sys
        real_module = sys.modules.get("tools.lazy_deps")
        try:
            sys.modules["tools.lazy_deps"] = None  # type: ignore[assignment]
            from tools.tts_tool import _import_piper
            with pytest.raises((ImportError, AttributeError)):
                _import_piper()
        finally:
            if real_module is not None:
                sys.modules["tools.lazy_deps"] = real_module
            else:
                sys.modules.pop("tools.lazy_deps", None)


# ---------------------------------------------------------------------------
# 3. _check_piper_available regression guard
# ---------------------------------------------------------------------------

class TestCheckPiperAvailable:
    """_check_piper_available must still detect importable piper."""

    def test_returns_false_when_piper_not_installed(self):
        from tools.tts_tool import _check_piper_available
        # In the test environment piper-tts is not installed,
        # so this should return False.
        assert _check_piper_available() is False

    def test_returns_true_when_piper_mocked(self):
        mock_spec = MagicMock()
        with patch("importlib.util.find_spec", return_value=mock_spec):
            from tools.tts_tool import _check_piper_available
            assert _check_piper_available() is True
