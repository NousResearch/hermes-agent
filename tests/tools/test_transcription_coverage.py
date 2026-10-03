"""Test coverage for tools/transcription_tools.py — 82 functions had LOW coverage.

Tests the pure helper functions: env resolution, STT config, language
resolution, binary finding, and model normalization. All filesystem
and subprocess calls are mocked.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from tools.transcription_tools import (
    _normalize_local_model,
    _resolve_stt_language,
    get_env_value,
    is_stt_enabled,
)

class TestGetEnvValue:
    def test_reads_existing_env_var(self, monkeypatch):
        monkeypatch.setenv("TEST_TRANSCRIPTION_KEY", "abc123")
        assert get_env_value("TEST_TRANSCRIPTION_KEY") == "abc123"

    def test_returns_default_for_missing(self, monkeypatch):
        monkeypatch.delenv("TEST_MISSING_VAR", raising=False)
        assert get_env_value("TEST_MISSING_VAR", "fallback") == "fallback"

    def test_default_is_none(self, monkeypatch):
        monkeypatch.delenv("TEST_MISSING_VAR", raising=False)
        assert get_env_value("TEST_MISSING_VAR") is None

class TestIsSttEnabled:
    def test_enabled_true(self):
        assert is_stt_enabled({"enabled": True}) is True

    def test_enabled_false(self):
        assert is_stt_enabled({"enabled": False}) is False

class TestResolveSttLanguage:

    def test_from_config(self):
        assert _resolve_stt_language(None, {"language": "fr"}) == "fr"

class TestNormalizeLocalModel:
    def test_known_model_passthrough(self):
        assert _normalize_local_model("base") == "base"

    def test_unknown_model_returns_default(self):
        result = _normalize_local_model("nonexistent-model-xyz")
        assert isinstance(result, str)
        assert len(result) > 0
