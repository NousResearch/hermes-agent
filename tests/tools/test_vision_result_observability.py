"""Behavioral coverage for auxiliary vision route observability."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.vision_tools import vision_analyze_tool


VALID_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00"
    b"\x00\x04\x00\x01\xf6\x178U\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _response(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.mark.asyncio
async def test_vision_result_reports_final_route_without_credentials(tmp_path: Path) -> None:
    image = tmp_path / "fixture.png"
    image.write_bytes(VALID_PNG)

    async def resolved_call(**kwargs):
        kwargs["route_info"].update(
            provider="fallback-provider",
            model="fallback/model",
            fallback_reason="payment error",
            api_key="must-not-leak",
            base_url="https://must-not-leak.example",
        )
        return _response("A blue pixel.")

    with (
        patch("tools.vision_tools._image_to_base64_data_url", return_value="data:image/png;base64,AA=="),
        patch("tools.vision_tools.async_call_llm", new=resolved_call),
    ):
        result = json.loads(await vision_analyze_tool(str(image), "describe", "requested/model"))

    assert result == {
        "success": True,
        "analysis": "A blue pixel.",
        "provider": "fallback-provider",
        "model": "fallback/model",
        "fallbackReason": "payment error",
    }
