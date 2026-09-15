"""vision.max_calls_per_image: default off; cap=1 refuses the second call."""
import json
from unittest.mock import AsyncMock, patch

import pytest

from tools.vision_tools import _handle_vision_analyze


ARGS = {"image_url": "https://example.com/photo.png", "question": "what is this?"}


async def _call(config, session_id="sess-1"):
    backend = AsyncMock(return_value=json.dumps({"result": "ok"}))
    with patch("tools.vision_tools.vision_analyze_tool", backend), patch(
        "tools.vision_tools._should_use_native_vision_fast_path", return_value=False
    ), patch("hermes_cli.config.load_config", return_value=config):
        first = await _handle_vision_analyze(ARGS, session_id=session_id)
        second = await _handle_vision_analyze(ARGS, session_id=session_id)
    return backend, first, second


@pytest.mark.asyncio
async def test_default_off_allows_repeat_calls():
    backend, first, second = await _call({})
    assert backend.await_count == 2
    assert json.loads(first)["result"] == "ok"
    assert json.loads(second)["result"] == "ok"


@pytest.mark.asyncio
async def test_cap_one_refuses_second_call_without_backend():
    backend, first, second = await _call({"vision": {"max_calls_per_image": 1}})
    assert backend.await_count == 1
    assert json.loads(first)["result"] == "ok"
    payload = json.loads(second)
    assert payload.get("success") is False
    assert "BLOCKED" in (payload.get("error") or payload.get("message") or str(payload))
