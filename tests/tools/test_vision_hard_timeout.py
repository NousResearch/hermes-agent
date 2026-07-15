"""Hermetic test for the vision wall-clock hard timeout.

vision_analyze_tool wraps the auxiliary LLM call in asyncio.wait_for so a hung
or stalled provider cannot block the agent turn indefinitely. This exercises
that bound end-to-end with a stalled coroutine and a tiny hard_timeout, with no
network and no real provider.
"""

import asyncio
import json

from unittest.mock import patch

import hermes_cli.config as hermes_config
import tools.vision_tools as vision_tools


# Minimal PNG header — enough for _image_to_base64_data_url to encode.
_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8

# Config with a tiny hard_timeout so the stalled call is bounded fast.
_VISION_CFG = {"auxiliary": {"vision": {"timeout": 120, "hard_timeout": 0.05}}}


def test_stalled_vision_call_is_bounded_by_hard_timeout(tmp_path):
    img = tmp_path / "stall.png"
    img.write_bytes(_PNG_BYTES)

    async def _stalled_call(*args, **kwargs):
        # Never returns — simulates a hung/stalled provider socket. wait_for
        # cancels this at hard_timeout, so the test finishes near-instantly.
        await asyncio.sleep(30)

    # load_config/cfg_get are imported inside the function from hermes_cli.config;
    # async_call_llm is a module-level import on vision_tools.
    with patch.object(hermes_config, "load_config", return_value=_VISION_CFG), \
            patch.object(vision_tools, "async_call_llm", side_effect=_stalled_call):
        result_json = asyncio.run(
            vision_tools.vision_analyze_tool(str(img), "describe this image")
        )

    result = json.loads(result_json)
    assert result["success"] is False
    # The user-facing message on a wall-clock timeout, not a raw traceback.
    assert "taking too long" in result["analysis"].lower()
