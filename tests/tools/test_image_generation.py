"""Tests for image_generation_tool — availability probe and fail-fast handler."""

import json
import os
from unittest.mock import patch


# ---------------------------------------------------------------------------
# detect_image_generation_environment
# ---------------------------------------------------------------------------


class TestDetectImageGenerationEnvironment:

    def test_available_when_key_and_client_present(self):
        with patch("tools.image_generation_tool._HAS_FAL_CLIENT", True), \
             patch.dict(os.environ, {"FAL_KEY": "test-key"}):
            from tools.image_generation_tool import detect_image_generation_environment
            result = detect_image_generation_environment()
        assert result["available"] is True
        assert result["reasons"] == []
        assert result["setup"] == []

    def test_unavailable_without_fal_key(self):
        with patch("tools.image_generation_tool._HAS_FAL_CLIENT", True), \
             patch.dict(os.environ, {}, clear=True):
            env = {k: v for k, v in os.environ.items() if k != "FAL_KEY"}
            with patch.dict(os.environ, env, clear=True):
                from tools.image_generation_tool import detect_image_generation_environment
                result = detect_image_generation_environment()
        assert result["available"] is False
        assert any("FAL_KEY" in r for r in result["reasons"])
        assert any("fal.ai" in s for s in result["setup"])

    def test_unavailable_without_fal_client(self):
        with patch("tools.image_generation_tool._HAS_FAL_CLIENT", False), \
             patch.dict(os.environ, {"FAL_KEY": "test-key"}):
            from tools.image_generation_tool import detect_image_generation_environment
            result = detect_image_generation_environment()
        assert result["available"] is False
        assert any("fal-client" in r for r in result["reasons"])
        assert any("pip install fal-client" in s for s in result["setup"])

    def test_both_missing_returns_two_reasons(self):
        with patch("tools.image_generation_tool._HAS_FAL_CLIENT", False):
            env = {k: v for k, v in os.environ.items() if k != "FAL_KEY"}
            with patch.dict(os.environ, env, clear=True):
                from tools.image_generation_tool import detect_image_generation_environment
                result = detect_image_generation_environment()
        assert result["available"] is False
        assert len(result["reasons"]) == 2
        assert len(result["setup"]) == 2


# ---------------------------------------------------------------------------
# _handle_image_generate — fail-fast guard
# ---------------------------------------------------------------------------


class TestHandleImageGenerateFailFast:

    def test_returns_structured_error_when_unavailable(self):
        unavailable_env = {
            "available": False,
            "reasons": ["FAL_KEY environment variable is not set"],
            "setup": ["Get a free API key at https://fal.ai and set FAL_KEY=<your-key>"],
        }
        with patch("tools.image_generation_tool.detect_image_generation_environment",
                   return_value=unavailable_env):
            from tools.image_generation_tool import _handle_image_generate
            result = json.loads(_handle_image_generate({"prompt": "a cat"}))
        assert "error" in result
        assert "FAL_KEY" in result["error"]
        assert "fal.ai" in result["error"]

    def test_error_message_contains_setup_instructions(self):
        unavailable_env = {
            "available": False,
            "reasons": ["fal-client library is not installed", "FAL_KEY environment variable is not set"],
            "setup": ["pip install fal-client", "Get a free API key at https://fal.ai and set FAL_KEY=<your-key>"],
        }
        with patch("tools.image_generation_tool.detect_image_generation_environment",
                   return_value=unavailable_env):
            from tools.image_generation_tool import _handle_image_generate
            result = json.loads(_handle_image_generate({"prompt": "a dog"}))
        assert "pip install fal-client" in result["error"]
        assert "fal.ai" in result["error"]

    def test_no_error_when_available_and_prompt_missing(self):
        available_env = {"available": True, "reasons": [], "setup": []}
        with patch("tools.image_generation_tool.detect_image_generation_environment",
                   return_value=available_env):
            from tools.image_generation_tool import _handle_image_generate
            result = json.loads(_handle_image_generate({}))
        # Should reach the prompt validation, not the availability guard
        assert "error" in result
        assert "prompt" in result["error"].lower()


# ---------------------------------------------------------------------------
# check_image_generation_requirements delegates to detect_*
# ---------------------------------------------------------------------------


class TestCheckImageGenerationRequirements:

    def test_true_when_env_available(self):
        with patch("tools.image_generation_tool.detect_image_generation_environment",
                   return_value={"available": True, "reasons": [], "setup": []}):
            from tools.image_generation_tool import check_image_generation_requirements
            assert check_image_generation_requirements() is True

    def test_false_when_env_unavailable(self):
        with patch("tools.image_generation_tool.detect_image_generation_environment",
                   return_value={"available": False, "reasons": ["x"], "setup": ["y"]}):
            from tools.image_generation_tool import check_image_generation_requirements
            assert check_image_generation_requirements() is False
