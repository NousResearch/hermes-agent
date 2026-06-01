"""Regression tests for PR #27015 — vision tool routing via active thread model.

Verifies that explicit provider/model kwargs propagate through the full
stack (agent → model_tools → registry → vision_tools) and that task-level
per-vision config is still respected.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from tools.vision_tools import (
    _handle_vision_analyze,
    vision_analyze_tool,
)


# ---------------------------------------------------------------------------
# 1. handle_function_call → registry.dispatch propagation
# ---------------------------------------------------------------------------

def test_handle_function_call_passes_provider_model():
    """handle_function_call must forward provider/model kwargs to registry.dispatch."""
    from model_tools import handle_function_call

    with patch("model_tools.registry.dispatch") as mock_dispatch:
        mock_dispatch.return_value = '{"success": true}'
        handle_function_call(
            "browser_get_images",
            {},
            provider="openrouter",
            model="gpt-4o",
        )
        args, kwargs = mock_dispatch.call_args
        assert kwargs["provider"] == "openrouter"
        assert kwargs["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# 2. _handle_vision_analyze extracts provider/model from kwargs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_vision_analyze_extracts_kwargs():
    """_handle_vision_analyze must pluck provider/model from **kw and forward them."""
    with patch("tools.vision_tools._read_main_provider", return_value="openrouter"), \
         patch("tools.vision_tools._read_main_model", return_value="invalid-model"), \
         patch("tools.vision_tools._supports_media_in_tool_results", return_value=False), \
         patch("tools.vision_tools.vision_analyze_tool") as mock_tool:

        await _handle_vision_analyze(
            {"image_url": "https://example.com/img.png", "question": "What is this?"},
            provider="anthropic",
            model="claude-3-opus-20240229",
        )
        args, kwargs = mock_tool.call_args
        assert kwargs.get("provider") == "anthropic"
        assert kwargs.get("model") == "claude-3-opus-20240229"


# ---------------------------------------------------------------------------
# 3. vision_analyze_tool includes task="vision" in call to async_call_llm
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_vision_analyze_tool_includes_task_vision():
    """The call_kwargs passed to async_call_llm must contain task='vision'."""
    with patch("tools.vision_tools.async_call_llm") as mock_async, \
         patch("tools.vision_tools._image_to_base64_data_url", return_value="data:image/png;base64,dummy"), \
         patch("tools.vision_tools._detect_image_mime_type", return_value="image/png"), \
         patch("pathlib.Path.exists", return_value=True):

        mock_async.return_value = MagicMock(content="Looks like a cat.", reasoning_content=None)
        await vision_analyze_tool(
            image_url="https://example.com/img.png",
            user_prompt="Describe this",
            model="gpt-4o",
            provider="openrouter",
        )
        args, kwargs = mock_async.call_args
        assert kwargs["task"] == "vision"


# ---------------------------------------------------------------------------
# 4. Fallback to AUXILIARY_VISION_MODEL env var
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_vision_fallback_to_aux_vision_model_env():
    """When no explicit model is passed, _handle_vision_analyze must pick up
    AUXILIARY_VISION_MODEL from the environment."""
    os.environ["AUXILIARY_VISION_MODEL"] = "google/gemini-pro-vision"
    try:
        with patch("tools.vision_tools._read_main_provider", return_value="openrouter"), \
             patch("tools.vision_tools._read_main_model", return_value="invalid-model"), \
             patch("tools.vision_tools._supports_media_in_tool_results", return_value=False), \
             patch("tools.vision_tools.vision_analyze_tool") as mock_tool:

            await _handle_vision_analyze(
                {"image_url": "https://example.com/img.png", "question": "What?"},
                # No model kwarg, no provider kwarg.
            )
            args, kwargs = mock_tool.call_args
            assert kwargs.get("model") == "google/gemini-pro-vision"
    finally:
        os.environ.pop("AUXILIARY_VISION_MODEL", None)


# ---------------------------------------------------------------------------
# 5. _resolve_task_provider_model prefers explicit args over task config
# ---------------------------------------------------------------------------

def test_resolve_task_provider_model_prefers_explicit_over_config():
    """When explicit provider/model are passed alongside a task, the explicit
    values must win over any task-level config."""
    from agent.auxiliary_client import _resolve_task_provider_model

    # mock _get_auxiliary_task_config to return a task config
    with patch("agent.auxiliary_client._get_auxiliary_task_config") as mock_cfg:
        mock_cfg.return_value = {
            "provider": "openrouter",
            "model": "anthropic/claude-3",
        }
        provider, model, *_ = _resolve_task_provider_model(
            task="vision",
            provider="some-explicit",
            model="explicit-model",
        )
        assert provider == "some-explicit"
        assert model == "explicit-model"
