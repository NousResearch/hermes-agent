"""Tests for XiaoXing-to-Codex request tools."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from gateway.config import Platform
from gateway.session import SessionSource
from tools.codex_media_request_tool import (
    CODEX_TASK_REQUEST_SCHEMA,
    CODEX_VIDEO_REQUEST_SCHEMA,
    _handle_codex_image_request,
    _handle_codex_task_request,
)


def test_codex_task_request_schema_is_for_non_media_codex_handoff():
    assert CODEX_TASK_REQUEST_SCHEMA["name"] == "codex_task_request"
    desc = CODEX_TASK_REQUEST_SCHEMA["description"]

    assert "non-media" in desc
    assert "must use this tool" in desc
    assert "roleplaying a handoff" in desc
    assert "taking a horn" in desc
    assert "codex_image_request" in desc
    assert "codex_video_request" in desc
    assert "video script" in desc
    assert "media generation brief" in desc


def test_codex_video_request_schema_covers_script_handoffs():
    assert CODEX_VIDEO_REQUEST_SCHEMA["name"] == "codex_video_request"
    desc = CODEX_VIDEO_REQUEST_SCHEMA["description"]

    assert "birthday videos" in desc
    assert "lip-sync clips" in desc
    assert "Bilibili-ready clips" in desc
    assert "video script" in desc
    assert "brief" in desc
    assert "storyboard" in desc
    assert "channel test" in desc


def test_codex_task_request_starts_background_task(monkeypatch):
    source = SessionSource(
        platform=Platform("napcat"),
        chat_id="group:610066383",
        chat_type="group",
        user_id="490008192",
        user_name="Dad",
    )
    runner = SimpleNamespace(
        _codex_media_request_allowed=lambda _source: True,
        _build_codex_task_background_prompt=lambda event, request_type, deliver_back: (
            f"request_type={request_type}\ndeliver_back={deliver_back}\n{event.text}"
        ),
        _start_background_task=AsyncMock(return_value="Background task started"),
    )

    monkeypatch.setattr("tools.codex_media_request_tool._current_session_source", lambda: source)
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: runner)

    result = json.loads(
        asyncio.run(
            _handle_codex_task_request(
                {
                    "request": "Codex，帮我检查刚才为什么没回爸爸",
                    "request_type": "debug",
                    "deliver_back": True,
                }
            )
        )
    )

    assert result["success"] is True
    assert result["status"] == "background_task_started"
    runner._start_background_task.assert_awaited_once()
    call = runner._start_background_task.call_args
    assert "Codex，帮我检查刚才为什么没回爸爸" in call.args[0]
    assert call.kwargs["extra_enabled_toolsets"] == ("terminal", "file")
    assert call.kwargs["excluded_toolsets"] == ("xiaoxing_codex_media", "xiaoxing_codex_task")


def test_codex_task_request_uses_mentor_channel_when_configured(monkeypatch):
    source = SessionSource(
        platform=Platform("napcat"),
        chat_id="group:610066383",
        chat_type="group",
        user_id="490008192",
        user_name="Dad",
    )
    runner = SimpleNamespace(
        _codex_media_request_allowed=lambda _source: True,
        _codex_task_route_for_source=lambda _source: "mentor_channel",
        _build_codex_task_background_prompt=lambda event, request_type, deliver_back: (
            f"request_type={request_type}\ndeliver_back={deliver_back}\n{event.text}"
        ),
        _enqueue_codex_mentor_task=AsyncMock(return_value="queued to Codex mentor channel"),
        _start_background_task=AsyncMock(return_value="Background task started"),
    )

    monkeypatch.setattr("tools.codex_media_request_tool._current_session_source", lambda: source)
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: runner)

    result = json.loads(
        asyncio.run(
            _handle_codex_task_request(
                {
                    "request": "Codex，帮我看一下小星为什么没有回群消息",
                    "request_type": "debug",
                    "deliver_back": True,
                }
            )
        )
    )

    assert result["success"] is True
    assert result["status"] == "mentor_channel_queued"
    assert result["message"] == "queued to Codex mentor channel"
    runner._enqueue_codex_mentor_task.assert_awaited_once()
    runner._start_background_task.assert_not_awaited()


def test_codex_image_request_uses_mentor_channel_when_configured(monkeypatch):
    source = SessionSource(
        platform=Platform("napcat"),
        chat_id="group:610066383",
        chat_type="group",
        user_id="490008192",
        user_name="Dad",
    )
    runner = SimpleNamespace(
        _codex_media_request_allowed=lambda _source: True,
        _codex_media_route_for_source=lambda _source: "mentor_channel",
        _build_codex_media_background_prompt=lambda event, media_kind: (
            f"media_kind={media_kind}\n{event.text}"
        ),
        _enqueue_codex_mentor_media_task=AsyncMock(return_value="queued to Codex mentor channel"),
        _start_background_task=AsyncMock(return_value="Background task started"),
    )

    monkeypatch.setattr("tools.codex_media_request_tool._current_session_source", lambda: source)
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: runner)

    result = json.loads(
        asyncio.run(
            _handle_codex_image_request(
                {"request": "Codex 生成一张小星试麦克风的图，做好发回来"}
            )
        )
    )

    assert result["success"] is True
    assert result["status"] == "mentor_channel_queued"
    assert result["message"] == "queued to Codex mentor channel"
    runner._enqueue_codex_mentor_media_task.assert_awaited_once()
    runner._start_background_task.assert_not_awaited()
