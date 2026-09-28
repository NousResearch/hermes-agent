"""Tests for tools.transcription_tools.is_local_model_loaded.

The CLI voice status banner shows "Preparing local STT model..." only on a
genuine cold load. ``is_local_model_loaded`` is the warm/cold probe it calls;
these tests pin its contract:

1. No model cached -> False (cold).
2. A cached model matching the requested name -> True (warm).
3. A cached model with a DIFFERENT name -> False (the next transcription
   would cold-reload because ``_get_or_load_local_model`` swaps models when
   the name differs).
4. A cached model and no name given -> True (any warm model counts).
5. After ``_unload_local_model`` -> False (the idle-unload path resets it).
"""

from unittest.mock import MagicMock, patch

from tools.transcription_tools import _unload_local_model, is_local_model_loaded

import tools.transcription_tools as tt


class TestIsLocalModelLoaded:
    def test_no_model_cached_is_cold(self):
        with patch.object(tt, "_local_model", None), patch.object(tt, "_local_model_name", None):
            assert is_local_model_loaded() is False
            assert is_local_model_loaded("base") is False

    def test_matching_model_name_is_warm(self):
        model = MagicMock(name="whisper-model")
        with patch.object(tt, "_local_model", model), patch.object(tt, "_local_model_name", "base"):
            assert is_local_model_loaded("base") is True

    def test_different_model_name_is_cold(self):
        model = MagicMock(name="whisper-model")
        with patch.object(tt, "_local_model", model), patch.object(tt, "_local_model_name", "base"):
            assert is_local_model_loaded("small") is False

    def test_no_name_reports_any_loaded_model_as_warm(self):
        model = MagicMock(name="whisper-model")
        with patch.object(tt, "_local_model", model), patch.object(tt, "_local_model_name", "base"):
            assert is_local_model_loaded() is True

    def test_cached_model_with_no_name_requested_is_cold(self):
        # A model is warm, but there's no recorded name to compare against, so a
        # name-requesting call must be treated as cold (the names can't match).
        model = MagicMock(name="whisper-model")
        with patch.object(tt, "_local_model", model), patch.object(tt, "_local_model_name", None):
            assert is_local_model_loaded("base") is False

    def test_after_unload_is_cold(self):
        model = MagicMock(name="whisper-model")
        with patch.object(tt, "_local_model", model), patch.object(tt, "_local_model_name", "base"):
            _unload_local_model()
            assert tt._local_model is None
            assert tt._local_model_name is None
            assert is_local_model_loaded("base") is False
            assert is_local_model_loaded() is False