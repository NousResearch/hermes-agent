"""Regression tests for secret redaction in vision tool logging."""

import importlib
import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _load_vision_tools(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("VISION_TOOLS_DEBUG", "true")

    import tools.vision_tools as vision_tools

    return importlib.reload(vision_tools)


@pytest.mark.asyncio
async def test_remote_image_url_is_redacted_in_logs_and_debug_json(
    tmp_path, monkeypatch, caplog
):
    vision_tools = _load_vision_tools(monkeypatch, tmp_path)
    image_url = (
        "https://cdn.example.com/cat.png"
        "?access_token=super-secret-token"
        "&size=small"
    )

    async def fake_download(_url, destination, max_retries=3):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
        return destination

    with (
        patch.object(vision_tools, "_validate_image_url", return_value=True),
        patch.object(vision_tools, "_download_image", side_effect=fake_download),
        patch.object(
            vision_tools,
            "_image_to_base64_data_url",
            return_value="data:image/png;base64,abc",
        ),
        patch.object(
            vision_tools,
            "async_call_llm",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ),
        patch.object(
            vision_tools,
            "extract_content_or_reasoning",
            return_value="A small cat",
        ),
        caplog.at_level(logging.INFO, logger="tools.vision_tools"),
    ):
        result = json.loads(await vision_tools.vision_analyze_tool(image_url, "describe"))

    assert result["success"] is True
    assert "super-secret-token" not in caplog.text
    assert "access_token=***" in caplog.text

    debug_path = Path(vision_tools.get_debug_session_info()["log_path"])
    debug_payload = json.loads(debug_path.read_text(encoding="utf-8"))
    debug_json = json.dumps(debug_payload)
    assert "super-secret-token" not in debug_json
    assert "access_token=***" in debug_json


@pytest.mark.asyncio
async def test_error_path_redacts_signed_url_in_logs_and_debug_json(
    tmp_path, monkeypatch, caplog
):
    vision_tools = _load_vision_tools(monkeypatch, tmp_path)
    image_url = (
        "https://cdn.example.com/cat.png"
        "?token=super-secret-token"
        "&size=small"
    )
    download_error = RuntimeError(f"download failed for {image_url}")

    with (
        patch.object(vision_tools, "_validate_image_url", return_value=True),
        patch.object(
            vision_tools,
            "_download_image",
            new_callable=AsyncMock,
            side_effect=download_error,
        ),
        caplog.at_level(logging.ERROR, logger="tools.vision_tools"),
    ):
        result = json.loads(await vision_tools.vision_analyze_tool(image_url, "describe"))

    assert result["success"] is False
    assert "super-secret-token" not in caplog.text
    assert "token=***" in caplog.text
    assert "super-secret-token" not in result["error"]
    assert "token=***" in result["error"]

    debug_path = Path(vision_tools.get_debug_session_info()["log_path"])
    debug_payload = json.loads(debug_path.read_text(encoding="utf-8"))
    debug_json = json.dumps(debug_payload)
    assert "super-secret-token" not in debug_json
    assert "token=***" in debug_json
