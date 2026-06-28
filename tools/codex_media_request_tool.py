"""XiaoXing-to-Codex media request tools.

These tools do not generate media in the foreground chat turn.  They enqueue a
background Codex task with the origin chat preserved.  Codex owns the Updream
API/script execution, polling, local artifact download, and final
``MEDIA:/absolute/path`` delivery back to the same QQ private chat or group.
"""
from __future__ import annotations

from typing import Any, Optional

from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource
from tools.registry import registry, tool_error, tool_result


def _current_session_source() -> Optional[SessionSource]:
    from gateway.config import Platform
    from gateway.session_context import get_session_env

    platform_name = get_session_env("HERMES_SESSION_PLATFORM", "")
    chat_id = get_session_env("HERMES_SESSION_CHAT_ID", "")
    user_id = get_session_env("HERMES_SESSION_USER_ID", "")
    if not platform_name or not chat_id or not user_id:
        return None

    try:
        platform = Platform(platform_name)
    except Exception:
        return None

    chat_type = "group" if str(chat_id).startswith("group:") else "dm"
    return SessionSource(
        platform=platform,
        chat_id=chat_id,
        chat_name=get_session_env("HERMES_SESSION_CHAT_NAME", ""),
        chat_type=chat_type,
        thread_id=get_session_env("HERMES_SESSION_THREAD_ID", "") or None,
        user_id=user_id,
        user_name=get_session_env("HERMES_SESSION_USER_NAME", ""),
    )


async def _request_codex_media(args: dict[str, Any], media_kind: str) -> str:
    request = str(args.get("request") or args.get("prompt") or "").strip()
    if not request:
        return tool_error("request is required")

    source = _current_session_source()
    if source is None:
        return tool_error(
            "No active gateway session source. Use this tool only from a XiaoXing QQ chat turn."
        )

    try:
        from gateway.run import _gateway_runner_ref
        runner = _gateway_runner_ref()
    except Exception:
        runner = None
    if runner is None:
        return tool_error("Gateway runner is not available in this process.")

    if not runner._codex_media_request_allowed(source):
        return tool_error("This chat or sender is not allowed to start Codex media generation.")

    event = MessageEvent(
        text=f"Codex {media_kind} request from XiaoXing tool: {request}",
        source=source,
    )
    prompt = runner._build_codex_media_background_prompt(event, media_kind)
    route = "background"
    if hasattr(runner, "_codex_media_route_for_source"):
        route = runner._codex_media_route_for_source(source)
    if route in {"mentor", "mentor_channel", "codex_mentor"}:
        confirmation = await runner._enqueue_codex_mentor_media_task(event, media_kind, prompt)
        return tool_result(
            success=True,
            media_kind=media_kind,
            status="mentor_channel_queued",
            message=confirmation,
            delivery=(
                "Codex will run the Updream workflow through the mentor channel "
                "and return the result to the same QQ chat when it completes."
            ),
        )

    toolsets = ("terminal", "file")
    confirmation = await runner._start_background_task(
        prompt,
        event,
        extra_enabled_toolsets=toolsets,
        excluded_toolsets=("xiaoxing_codex_media", "image_gen"),
    )
    return tool_result(
        success=True,
        media_kind=media_kind,
        status="background_task_started",
        message=confirmation,
        delivery=(
            "Codex will run the Updream workflow in the background and return the "
            "result to the same QQ chat when it completes."
        ),
    )


async def _request_codex_task(args: dict[str, Any]) -> str:
    request = str(args.get("request") or args.get("message") or "").strip()
    if not request:
        return tool_error("request is required")

    source = _current_session_source()
    if source is None:
        return tool_error(
            "No active gateway session source. Use this tool only from a XiaoXing QQ chat turn."
        )

    try:
        from gateway.run import _gateway_runner_ref
        runner = _gateway_runner_ref()
    except Exception:
        runner = None
    if runner is None:
        return tool_error("Gateway runner is not available in this process.")

    if not runner._codex_media_request_allowed(source):
        return tool_error("This chat or sender is not allowed to start a Codex task.")

    request_type = str(args.get("request_type") or "task").strip() or "task"
    deliver_back = bool(args.get("deliver_back", True))
    event = MessageEvent(
        text=f"Codex task request from XiaoXing tool: {request}",
        source=source,
    )
    prompt = runner._build_codex_task_background_prompt(
        event,
        request_type=request_type,
        deliver_back=deliver_back,
    )
    route = "background"
    if hasattr(runner, "_codex_task_route_for_source"):
        route = runner._codex_task_route_for_source(source)
    if route in {"mentor", "mentor_channel", "codex_mentor"}:
        confirmation = await runner._enqueue_codex_mentor_task(
            event,
            request_type=request_type,
            deliver_back=deliver_back,
            prompt=prompt,
        )
        return tool_result(
            success=True,
            request_type=request_type,
            status="mentor_channel_queued",
            message=confirmation,
            delivery=(
                "Codex will handle the task through the mentor channel and return "
                "the result to the same QQ chat unless the task is explicitly internal-only."
            ),
        )

    confirmation = await runner._start_background_task(
        prompt,
        event,
        extra_enabled_toolsets=("terminal", "file"),
        excluded_toolsets=("xiaoxing_codex_media", "xiaoxing_codex_task"),
    )
    return tool_result(
        success=True,
        request_type=request_type,
        status="background_task_started",
        message=confirmation,
        delivery=(
            "Codex will handle the task in the background and return the result "
            "to the same QQ chat unless the task is explicitly internal-only."
        ),
    )


async def _handle_codex_image_request(args: dict[str, Any], **kwargs) -> str:
    return await _request_codex_media(args, "image")


async def _handle_codex_video_request(args: dict[str, Any], **kwargs) -> str:
    return await _request_codex_media(args, "video")


async def _handle_codex_task_request(args: dict[str, Any], **kwargs) -> str:
    return await _request_codex_task(args)


CODEX_IMAGE_REQUEST_SCHEMA = {
    "name": "codex_image_request",
    "description": (
        "Ask Codex, through XiaoXing's channel/background bridge, to generate an image "
        "via Codex-owned Updream direct API/script execution for the current QQ private "
        "chat or group and send the generated file back here. "
        "Use when a trusted user asks XiaoXing to let Codex make a picture, cover, poster, "
        "or other still image. XiaoXing should not run scripts itself. Do not use for "
        "public posting, login, comments, follows, DMs, or public account profile changes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "request": {
                "type": "string",
                "description": "The user's exact image-generation request, including subject, style, and constraints.",
            },
        },
        "required": ["request"],
    },
}


CODEX_VIDEO_REQUEST_SCHEMA = {
    "name": "codex_video_request",
    "description": (
        "Ask Codex, through XiaoXing's channel/background bridge, to generate a video "
        "via Codex-owned Updream direct API/script execution for the current QQ private "
        "chat or group and send the generated file back here. "
        "Use this whenever a trusted user asks XiaoXing to have Codex make, edit, cut, "
        "prepare, render, or continue a video/animation, including birthday videos, "
        "lip-sync clips, voice-over videos, video drafts, or Bilibili-ready clips. "
        "Also use this when the user phrases the task as sending a video script, "
        "brief, storyboard, shot list, or channel test to Codex so Codex can make "
        "or edit the video. "
        "XiaoXing should not run scripts itself. "
        "Never report a static-frame fallback as a completed video; if native audio or "
        "lip-sync is required and the backend cannot support it, Codex must report the blocker."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "request": {
                "type": "string",
                "description": "The user's exact video-generation request, including motion, audio/lip-sync needs, duration, and constraints.",
            },
        },
        "required": ["request"],
    },
}


CODEX_TASK_REQUEST_SCHEMA = {
    "name": "codex_task_request",
    "description": (
        "Ask Codex, through XiaoXing's channel/background bridge, to handle a "
        "non-media task or message from the current QQ private chat or group. "
        "Use this for debugging, checking files, explaining status, organizing notes, "
        "retrying the Codex mentor channel after Dad says it is fixed, or other "
        "Codex-owned work that is not image/video generation. Do not use this tool "
        "for a picture, cover, video, animation, video script, storyboard, or media "
        "generation brief; use the image/video Codex tools for those. XiaoXing must use "
        "this tool instead of merely promising to tell Codex or roleplaying a "
        "handoff in chat. Do not write stage directions such as taking a horn, "
        "calling Codex gege, or shouting into the channel in the QQ reply. For "
        "pictures use codex_image_request; for videos or animations use "
        "codex_video_request."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "request": {
                "type": "string",
                "description": "The user's exact Codex task or message, including relevant context and requested return behavior.",
            },
            "request_type": {
                "type": "string",
                "enum": ["task", "debug", "status", "file", "note"],
                "description": "Small routing label for Codex. Use task when unsure.",
            },
            "deliver_back": {
                "type": "boolean",
                "description": "Whether Codex should return the result to the same QQ chat. Defaults to true.",
            },
        },
        "required": ["request"],
    },
}


registry.register(
    name="codex_image_request",
    toolset="xiaoxing_codex_media",
    schema=CODEX_IMAGE_REQUEST_SCHEMA,
    handler=_handle_codex_image_request,
    is_async=True,
    description=CODEX_IMAGE_REQUEST_SCHEMA["description"],
    emoji="",
)

registry.register(
    name="codex_task_request",
    toolset="xiaoxing_codex_task",
    schema=CODEX_TASK_REQUEST_SCHEMA,
    handler=_handle_codex_task_request,
    is_async=True,
    description=CODEX_TASK_REQUEST_SCHEMA["description"],
    emoji="",
)

registry.register(
    name="codex_video_request",
    toolset="xiaoxing_codex_media",
    schema=CODEX_VIDEO_REQUEST_SCHEMA,
    handler=_handle_codex_video_request,
    is_async=True,
    description=CODEX_VIDEO_REQUEST_SCHEMA["description"],
    emoji="",
)
