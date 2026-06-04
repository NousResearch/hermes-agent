"""Tests for the coding_plan/vlm fast-path in vision_analyze_tool.

The fast-path lets users opt in to MiniMax's private /v1/coding_plan/vlm
endpoint (the same one the official minimax-coding-plan-mcp package's
understand_image tool wraps) as a primary or fallback vision backend.

We mock the HTTP layer so tests are fast, deterministic, and don't require
a real API key.
"""
import json
import base64
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from tools.vision_tools import _try_coding_plan_vlm


# ---------- helpers ----------------------------------------------------------

def _fake_data_url(color="red", size=(1, 1)):
    """Build a tiny valid PNG data URL without Pillow dependency.

    A minimal 1x1 solid PNG is 67 bytes — the contents are opaque to
    _try_coding_plan_vlm which never inspects the bytes, only forwards
    the data URL as-is to the API.
    """
    # 1x1 transparent PNG (valid PNG signature + IHDR + IDAT + IEND)
    import zlib, struct
    width, height = size
    raw = b""
    for _ in range(height):
        raw += b"\x00" + b"\x00" * (width * 3)  # filter byte + RGB
    compressed = zlib.compress(raw)
    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff))
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", compressed)
           + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


def _ok_response(content: str) -> MagicMock:
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {
        "content": content,
        "base_resp": {"status_code": 0, "status_msg": "success"},
    }
    return r


def _err_response(status: int, body: str) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.text = body
    r.json.side_effect = json.JSONDecodeError("err", body, 0)
    return r


# ---------- _try_coding_plan_vlm: happy path --------------------------------

@pytest.mark.asyncio
async def test_coding_plan_vlm_success():
    """Successful call returns the content string."""
    cfg = {
        "enabled": True,
        "api_host": "https://api.example.com",
        "api_key": "test-key",
        "timeout": 30,
        "max_tokens": 500,
    }
    with patch("tools.vision_tools.httpx.AsyncClient") as MockClient:
        client = AsyncMock()
        client.post.return_value = _ok_response("A solid red square.")
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        MockClient.return_value = client

        result = await _try_coding_plan_vlm(
            image_data_url=_fake_data_url(),
            user_prompt="describe",
            cfg=cfg,
        )
    assert result == "A solid red square."


# ---------- _try_coding_plan_vlm: failure modes -----------------------------

@pytest.mark.asyncio
async def test_coding_plan_vlm_no_api_key():
    """No key configured → return None, don't raise."""
    cfg = {"enabled": True, "api_host": "https://api.example.com", "api_key": ""}
    with patch.dict("os.environ", {}, clear=True):
        # MINIMAX_CN_API_KEY not in env either
        result = await _try_coding_plan_vlm(
            image_data_url=_fake_data_url(), user_prompt="x", cfg=cfg
        )
    assert result is None


@pytest.mark.asyncio
async def test_coding_plan_vlm_env_var_resolution():
    """${ENV_VAR} placeholder gets resolved from os.environ."""
    cfg = {
        "enabled": True,
        "api_host": "https://api.example.com",
        "api_key": "${MY_TEST_KEY}",
    }
    with patch.dict("os.environ", {"MY_TEST_KEY": "resolved-secret"}, clear=False):
        with patch("tools.vision_tools.httpx.AsyncClient") as MockClient:
            client = AsyncMock()
            client.post.return_value = _ok_response("ok")
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = client

            await _try_coding_plan_vlm(
                image_data_url=_fake_data_url(), user_prompt="x", cfg=cfg
            )
            call_kwargs = client.post.call_args.kwargs
            assert call_kwargs["headers"]["Authorization"] == "Bearer resolved-secret"


@pytest.mark.asyncio
async def test_coding_plan_vlm_http_500_returns_none():
    """Non-200 status → return None (caller falls through)."""
    cfg = {"enabled": True, "api_host": "https://api.example.com", "api_key": "k"}
    with patch("tools.vision_tools.httpx.AsyncClient") as MockClient:
        client = AsyncMock()
        client.post.return_value = _err_response(500, "server boom")
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        MockClient.return_value = client

        result = await _try_coding_plan_vlm(
            image_data_url=_fake_data_url(), user_prompt="x", cfg=cfg
        )
    assert result is None


@pytest.mark.asyncio
async def test_coding_plan_vlm_base_resp_error_returns_none():
    """base_resp.status_code != 0 → return None."""
    cfg = {"enabled": True, "api_host": "https://api.example.com", "api_key": "k"}
    bad = MagicMock()
    bad.status_code = 200
    bad.json.return_value = {
        "content": "should be ignored",
        "base_resp": {"status_code": 1026, "status_msg": "sensitive"},
    }
    with patch("tools.vision_tools.httpx.AsyncClient") as MockClient:
        client = AsyncMock()
        client.post.return_value = bad
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        MockClient.return_value = client

        result = await _try_coding_plan_vlm(
            image_data_url=_fake_data_url(), user_prompt="x", cfg=cfg
        )
    assert result is None


@pytest.mark.asyncio
async def test_coding_plan_vlm_empty_content_returns_none():
    """Empty content string → return None."""
    cfg = {"enabled": True, "api_host": "https://api.example.com", "api_key": "k"}
    with patch("tools.vision_tools.httpx.AsyncClient") as MockClient:
        client = AsyncMock()
        client.post.return_value = _ok_response("   ")
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        MockClient.return_value = client

        result = await _try_coding_plan_vlm(
            image_data_url=_fake_data_url(), user_prompt="x", cfg=cfg
        )
    assert result is None


@pytest.mark.asyncio
async def test_coding_plan_vlm_timeout_returns_none():
    """Timeout → return None, not raise."""
    cfg = {"enabled": True, "api_host": "https://api.example.com", "api_key": "k", "timeout": 5}
    with patch("tools.vision_tools.httpx.AsyncClient") as MockClient:
        client = AsyncMock()
        client.post.side_effect = httpx.TimeoutException("too slow")
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        MockClient.return_value = client

        result = await _try_coding_plan_vlm(
            image_data_url=_fake_data_url(), user_prompt="x", cfg=cfg
        )
    assert result is None


@pytest.mark.asyncio
async def test_coding_plan_vlm_connection_error_returns_none():
    """Network failure → return None, not raise."""
    cfg = {"enabled": True, "api_host": "https://api.example.com", "api_key": "k"}
    with patch("tools.vision_tools.httpx.AsyncClient") as MockClient:
        client = AsyncMock()
        client.post.side_effect = httpx.ConnectError("dns fail")
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        MockClient.return_value = client

        result = await _try_coding_plan_vlm(
            image_data_url=_fake_data_url(), user_prompt="x", cfg=cfg
        )
    assert result is None


# ---------- request shape validation ----------------------------------------

@pytest.mark.asyncio
async def test_coding_plan_vlm_request_shape():
    """Verify the wire format matches MiniMax's documented schema."""
    cfg = {
        "enabled": True,
        "api_host": "https://api.example.com",
        "api_key": "test-key",
        "max_tokens": 1000,
    }
    captured = {}

    async def fake_post(url, json=None, headers=None, **kw):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return _ok_response("ok")

    with patch("tools.vision_tools.httpx.AsyncClient") as MockClient:
        client = AsyncMock()
        client.post.side_effect = fake_post
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        MockClient.return_value = client

        data_url = _fake_data_url()
        await _try_coding_plan_vlm(
            image_data_url=data_url, user_prompt="describe", cfg=cfg
        )

    # URL must be the documented private endpoint
    assert captured["url"] == "https://api.example.com/v1/coding_plan/vlm"
    # Headers must include the MCP source identifier
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["headers"]["MM-API-Source"] == "Hermes-Agent"
    assert captured["headers"]["Content-Type"] == "application/json"
    # Payload schema must match minimax-coding-plan-mcp's client.py
    assert captured["json"]["prompt"] == "describe"
    assert captured["json"]["image_url"] == data_url
    # No nested "messages" — this is not chat/completions
    assert "messages" not in captured["json"]
    assert "model" not in captured["json"]


@pytest.mark.asyncio
async def test_coding_plan_vlm_default_api_host():
    """Empty api_host falls back to the documented MiniMax CN endpoint."""
    cfg = {"enabled": True, "api_key": "k"}  # no api_host
    captured = {}

    async def fake_post(url, **kw):
        captured["url"] = url
        return _ok_response("ok")

    with patch("tools.vision_tools.httpx.AsyncClient") as MockClient:
        client = AsyncMock()
        client.post.side_effect = fake_post
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        MockClient.return_value = client

        await _try_coding_plan_vlm(
            image_data_url=_fake_data_url(), user_prompt="x", cfg=cfg
        )
    assert captured["url"] == "https://api.minimaxi.com/v1/coding_plan/vlm"


@pytest.mark.asyncio
async def test_coding_plan_vlm_trailing_slash_in_host():
    """api_host with trailing slash should not produce double slashes."""
    cfg = {
        "enabled": True,
        "api_host": "https://api.example.com/",
        "api_key": "k",
    }
    captured = {}

    async def fake_post(url, **kw):
        captured["url"] = url
        return _ok_response("ok")

    with patch("tools.vision_tools.httpx.AsyncClient") as MockClient:
        client = AsyncMock()
        client.post.side_effect = fake_post
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        MockClient.return_value = client

        await _try_coding_plan_vlm(
            image_data_url=_fake_data_url(), user_prompt="x", cfg=cfg
        )
    assert captured["url"] == "https://api.example.com/v1/coding_plan/vlm"
