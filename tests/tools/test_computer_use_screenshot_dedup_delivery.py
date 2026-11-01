"""Regression for #147082: only delivered screenshots may suppress later captures."""

import base64
import contextlib
import json
import struct
import zlib
from unittest.mock import Mock

import pytest

import tools.computer_use.tool as cu_tool
from tools.bot_desktop import lease, runtime
from tools.computer_use.backend import CaptureResult


def _png_b64(seed: int = 1) -> str:
    width = height = 64
    raw = b"".join(
        b"\x00" + bytes(((x + y + seed) % 256) for x in range(width * 3))
        for y in range(height)
    )

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode("ascii")


def _is_multimodal(response) -> bool:
    return isinstance(response, dict) and response.get("_multimodal") is True


def _make_agent():
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent.session_id = "s1"
    agent.provider = "test-provider"
    agent.model = "test-model"
    agent._no_list_tool_content_models = set()
    return agent


@pytest.fixture
def capture(monkeypatch):
    cu_tool._reset_screenshot_dedup()
    lease._reset_for_tests()
    frame = CaptureResult(
        mode="som", width=64, height=64, png_b64=_png_b64(),
        elements=[], app="kate", window_title="notes",
    )
    backend = cu_tool._NoopBackend()
    backend.capture = lambda **kw: frame
    monkeypatch.setattr(cu_tool, "_backend_for_call", lambda session_id="": contextlib.nullcontext(backend))
    monkeypatch.setattr(runtime, "ensure_started_for_tool", lambda: None)
    monkeypatch.setattr(cu_tool, "_should_route_through_aux_vision", lambda: False)
    monkeypatch.setattr(cu_tool, "_persist_capture_image", lambda cap: None)
    yield lambda: cu_tool.handle_computer_use({"action": "capture", "mode": "som"}, session_id="s1")
    cu_tool._reset_screenshot_dedup()
    lease._reset_for_tests()


def test_aux_failure_retries_identical_capture(capture, monkeypatch):
    router = Mock(return_value=None)
    monkeypatch.setattr(cu_tool, "_should_route_through_aux_vision", lambda: True)
    monkeypatch.setattr(cu_tool, "_route_capture_through_aux_vision", router)

    first = json.loads(capture())
    second = json.loads(capture())

    assert first["vision_unavailable"] is True
    assert "screen_unchanged" not in second
    assert second["vision_unavailable"] is True
    assert router.call_count == 2


def test_aux_delivery_dedups_identical_capture(capture, monkeypatch):
    delivered = json.dumps({"vision_analysis": "A blank editor window."})
    router = Mock(return_value=delivered)
    monkeypatch.setattr(cu_tool, "_should_route_through_aux_vision", lambda: True)
    monkeypatch.setattr(cu_tool, "_route_capture_through_aux_vision", router)

    assert capture() == delivered
    assert json.loads(capture())["screen_unchanged"] is True
    assert router.call_count == 1


def test_multimodal_delivery_dedups_identical_capture(capture):
    assert _is_multimodal(capture())
    assert json.loads(capture())["screen_unchanged"] is True


def test_post_dispatch_takeover_redelivers_identical_capture(capture, monkeypatch):
    dispatch = cu_tool._dispatch

    def dispatch_then_takeover(*args, **kwargs):
        result = dispatch(*args, **kwargs)
        assert _is_multimodal(result)
        lease.acquire("human")
        lease.release("human")
        return result

    monkeypatch.setattr(cu_tool, "_dispatch", dispatch_then_takeover)
    refused = json.loads(capture())
    assert refused["code"] == "human_has_control"
    monkeypatch.setattr(cu_tool, "_dispatch", dispatch)

    assert _is_multimodal(capture())


@pytest.mark.parametrize("remember_model", [True, False])
def test_stripped_tool_image_redelivers_identical_capture(capture, remember_model):
    agent = _make_agent()
    result = capture()
    assert _is_multimodal(result)
    messages = [{
        "role": "tool", "name": "computer_use", "tool_call_id": "capture",
        "content": result["content"],
    }]

    assert agent._try_strip_image_parts_from_tool_messages(messages, remember_model=remember_model) is True
    assert isinstance(messages[0]["content"], str)
    assert _is_multimodal(capture())


@pytest.mark.parametrize("downgrade", ["non_vision", "provider", "cached"])
def test_downgraded_tool_image_redelivers_identical_capture(capture, monkeypatch, downgrade):
    agent = _make_agent()
    monkeypatch.setattr(agent, "_model_supports_vision", lambda: downgrade != "non_vision")
    monkeypatch.setattr(agent, "_provider_supports_vision_tool_messages", lambda: downgrade != "provider")
    if downgrade == "cached":
        agent._no_list_tool_content_models.add((agent.provider, agent.model))
    result = capture()
    assert _is_multimodal(result)

    content = agent._tool_result_content_for_active_model("computer_use", result)

    assert isinstance(content, str)
    if downgrade == "non_vision":
        assert "error" in json.loads(content)
    else:
        assert content == result["text_summary"]
    assert _is_multimodal(capture())
