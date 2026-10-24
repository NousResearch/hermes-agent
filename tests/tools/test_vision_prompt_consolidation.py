"""The image pre-analysis prompt has a single source of truth.

Every entry point that pre-analyzes a user-attached image reads
``tools.vision_tools.get_vision_analysis_prompt`` instead of inlining a private
copy, so the wording cannot drift between the gateway, the CLI and the agent
loop.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from tools.vision_tools import (
    VISION_ANALYSIS_PROMPT_CONCISE,
    VISION_ANALYSIS_PROMPT_DETAILED,
    get_vision_analysis_prompt,
)


def _no_config():
    return patch("hermes_cli.config.load_config", side_effect=Exception("no config"))


def test_concise_default_is_returned_by_default():
    with _no_config():
        prompt = get_vision_analysis_prompt()

    assert prompt == VISION_ANALYSIS_PROMPT_CONCISE
    assert "Concisely describe this image in 2-4 sentences" in prompt


def test_detailed_style_is_opt_in():
    with _no_config():
        prompt = get_vision_analysis_prompt("detailed")

    assert prompt == VISION_ANALYSIS_PROMPT_DETAILED
    assert "thorough detail" in prompt


def test_config_override_wins_for_both_styles():
    cfg = {"auxiliary": {"vision": {"analysis_prompt": "Custom prompt"}}}
    with patch("hermes_cli.config.load_config", return_value=cfg):
        assert get_vision_analysis_prompt("concise") == "Custom prompt"
        assert get_vision_analysis_prompt("detailed") == "Custom prompt"


def test_blank_override_falls_back_to_the_builtin():
    cfg = {"auxiliary": {"vision": {"analysis_prompt": "   "}}}
    with patch("hermes_cli.config.load_config", return_value=cfg):
        assert get_vision_analysis_prompt() == VISION_ANALYSIS_PROMPT_CONCISE


def test_missing_key_falls_back_to_the_builtin():
    with patch("hermes_cli.config.load_config", return_value={}):
        assert get_vision_analysis_prompt() == VISION_ANALYSIS_PROMPT_CONCISE


@pytest.mark.asyncio
async def test_gateway_uses_the_shared_concise_prompt():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    analysis = json.dumps({"success": True, "analysis": "A cat on a chair."})

    with patch(
        "tools.vision_tools.vision_analyze_tool",
        new_callable=AsyncMock,
        return_value=analysis,
    ) as mock_vision:
        await runner._enrich_message_with_vision("what is this?", ["/tmp/cat.png"])

    assert mock_vision.await_args.kwargs["user_prompt"] == get_vision_analysis_prompt("concise")


def test_no_entry_point_inlines_a_private_prompt_copy():
    """The two former inline copies are gone; only the module constants hold them."""
    import inspect

    from agent import vision_message_prep
    from gateway import run_inbound
    from hermes_cli import cli_stream_mixin

    for module in (vision_message_prep, run_inbound, cli_stream_mixin):
        source = inspect.getsource(module)
        assert "Concisely describe this image" not in source, module.__name__
        assert "Describe everything visible in this image" not in source, module.__name__
        assert "get_vision_analysis_prompt" in source, module.__name__
