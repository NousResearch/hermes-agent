"""Tests for config.yaml-vs-env-var model resolution in browser_tool.py.

Covers _get_vision_model() and _get_extraction_model():
- config.yaml model beats env var (returns None for centralized routing)
- env var used when config is empty
- neither set returns None
- config-read failure falls back to env var
"""

import os
from unittest.mock import patch

import pytest

from tools.browser_tool import _get_vision_model, _get_extraction_model


# ---------------------------------------------------------------------------
# _get_vision_model
# ---------------------------------------------------------------------------


class TestGetVisionModel:
    def test_config_model_returns_none(self):
        """When config.yaml has auxiliary.vision.model, return None (let aux client route)."""
        cfg = {"auxiliary": {"vision": {"model": "google/gemini-2-flash"}}}
        with (
            patch("hermes_cli.config.read_raw_config", return_value=cfg),
            patch.dict(os.environ, {"AUXILIARY_VISION_MODEL": "env/model"}),
        ):
            assert _get_vision_model() is None

    def test_env_var_used_when_config_empty(self):
        """When config has no auxiliary.vision.model, fall back to env var."""
        cfg = {"auxiliary": {"vision": {}}}
        with (
            patch("hermes_cli.config.read_raw_config", return_value=cfg),
            patch.dict(os.environ, {"AUXILIARY_VISION_MODEL": "env/vision-model"}),
        ):
            assert _get_vision_model() == "env/vision-model"

    def test_env_var_used_when_config_has_no_auxiliary(self):
        """Completely empty config falls back to env var."""
        with (
            patch("hermes_cli.config.read_raw_config", return_value={}),
            patch.dict(os.environ, {"AUXILIARY_VISION_MODEL": "env/model-x"}),
        ):
            assert _get_vision_model() == "env/model-x"

    def test_neither_set_returns_none(self):
        """When neither config nor env var is set, return None."""
        with (
            patch("hermes_cli.config.read_raw_config", return_value={}),
            patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("AUXILIARY_VISION_MODEL", None)
            assert _get_vision_model() is None

    def test_config_read_failure_falls_back_to_env(self):
        """If read_raw_config raises, fall back to env var gracefully."""
        with (
            patch("hermes_cli.config.read_raw_config", side_effect=OSError("disk error")),
            patch.dict(os.environ, {"AUXILIARY_VISION_MODEL": "fallback/model"}),
        ):
            assert _get_vision_model() == "fallback/model"

    def test_config_read_failure_no_env_returns_none(self):
        """Config read failure and no env var: return None."""
        with (
            patch("hermes_cli.config.read_raw_config", side_effect=OSError("disk error")),
            patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("AUXILIARY_VISION_MODEL", None)
            assert _get_vision_model() is None

    def test_env_var_whitespace_stripped(self):
        """Env var with whitespace only is treated as not set."""
        with (
            patch("hermes_cli.config.read_raw_config", return_value={}),
            patch.dict(os.environ, {"AUXILIARY_VISION_MODEL": "  "}),
        ):
            assert _get_vision_model() is None


# ---------------------------------------------------------------------------
# _get_extraction_model
# ---------------------------------------------------------------------------


class TestGetExtractionModel:
    def test_config_model_returns_none(self):
        """When config.yaml has auxiliary.web_extract.model, return None."""
        cfg = {"auxiliary": {"web_extract": {"model": "openai/gpt-4o-mini"}}}
        with (
            patch("hermes_cli.config.read_raw_config", return_value=cfg),
            patch.dict(os.environ, {"AUXILIARY_WEB_EXTRACT_MODEL": "env/extract-model"}),
        ):
            assert _get_extraction_model() is None

    def test_env_var_used_when_config_empty(self):
        """When config has no auxiliary.web_extract.model, use env var."""
        cfg = {"auxiliary": {"web_extract": {}}}
        with (
            patch("hermes_cli.config.read_raw_config", return_value=cfg),
            patch.dict(os.environ, {"AUXILIARY_WEB_EXTRACT_MODEL": "env/extract-v2"}),
        ):
            assert _get_extraction_model() == "env/extract-v2"

    def test_neither_set_returns_none(self):
        """When neither config nor env var is set, return None."""
        with (
            patch("hermes_cli.config.read_raw_config", return_value={}),
            patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("AUXILIARY_WEB_EXTRACT_MODEL", None)
            assert _get_extraction_model() is None

    def test_config_read_failure_falls_back_to_env(self):
        """If read_raw_config raises, fall back to env var gracefully."""
        with (
            patch("hermes_cli.config.read_raw_config", side_effect=FileNotFoundError("no file")),
            patch.dict(os.environ, {"AUXILIARY_WEB_EXTRACT_MODEL": "fallback/extract"}),
        ):
            assert _get_extraction_model() == "fallback/extract"

    def test_config_read_failure_no_env_returns_none(self):
        """Config read failure and no env var: return None."""
        with (
            patch("hermes_cli.config.read_raw_config", side_effect=FileNotFoundError("no file")),
            patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("AUXILIARY_WEB_EXTRACT_MODEL", None)
            assert _get_extraction_model() is None

    def test_env_var_whitespace_stripped(self):
        """Env var with whitespace only is treated as not set."""
        with (
            patch("hermes_cli.config.read_raw_config", return_value={}),
            patch.dict(os.environ, {"AUXILIARY_WEB_EXTRACT_MODEL": "   "}),
        ):
            assert _get_extraction_model() is None
