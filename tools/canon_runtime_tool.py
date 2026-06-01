"""Generic Canon runtime card/input plumbing for Hermes tools.

This module intentionally contains no domain-specific behavior. Priority and
other packs can use the helper functions to ask Canon for rich-card actions or
structured runtime input without copying Canon credential and polling logic.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Optional

from tools.registry import registry, tool_error, tool_result


DEFAULT_TIMEOUT_SECONDS = 300
POLL_SECONDS = 1.0


def _safe_timeout_seconds(value: Any, default: int = DEFAULT_TIMEOUT_SECONDS) -> int:
    try:
        seconds = int(value)
    except (TypeError, ValueError):
        return default
    return min(30 * 60, max(5, seconds))


def resolve_canon_conversation_id(target: Any = None) -> str:
    """Resolve a Canon conversation id from a tool target or current session."""
    raw = str(target or "").strip()
    if raw:
        if raw.startswith("canon:"):
            raw = raw.split(":", 1)[1].strip()
        if raw:
            return raw
        raise ValueError("Canon target must include a conversation id")

    try:
        from gateway.session_context import get_session_env

        platform = get_session_env("HERMES_SESSION_PLATFORM", "").strip().lower()
        chat_id = get_session_env("HERMES_SESSION_CHAT_ID", "").strip()
        if platform == "canon" and chat_id:
            return chat_id
    except Exception:
        pass

    home = os.getenv("CANON_HOME_CHANNEL", "").strip()
    if home:
        return home
    raise ValueError(
        "No Canon conversation id found. Run inside a Canon conversation or pass target='canon:<conversation_id>'."
    )


def _card_has_actions(card: dict[str, Any]) -> bool:
    blocks = card.get("blocks")
    if not isinstance(blocks, list):
        return False
    return any(
        isinstance(block, dict)
        and block.get("kind") == "actions"
        and isinstance(block.get("actions"), list)
        and bool(block.get("actions"))
        for block in blocks
    )


def _new_canon_client():
    from plugins.platforms.canon.adapter import (
        DEFAULT_BASE_URL,
        DEFAULT_STREAM_URL,
        CanonHttpClient,
        _resolve_canon_agent,
    )

    resolved = _resolve_canon_agent(None)
    if not resolved.api_key:
        raise RuntimeError("Canon agent credentials are not configured")
    return CanonHttpClient(
        resolved.api_key,
        base_url=os.getenv("CANON_BASE_URL", "").strip()
        or resolved.base_url
        or DEFAULT_BASE_URL,
        stream_url=os.getenv("CANON_STREAM_URL", "").strip()
        or resolved.stream_url
        or DEFAULT_STREAM_URL,
    )


async def request_canon_runtime_card(
    *,
    target: Any = None,
    card: dict[str, Any],
    card_id: Optional[str] = None,
    response_user_id: Optional[str] = None,
    runtime_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    wait: bool = True,
    native: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    conversation_id = resolve_canon_conversation_id(target)
    timeout_seconds = _safe_timeout_seconds(timeout_seconds)
    expires_at = int((time.time() + timeout_seconds) * 1000)
    effective_card_id = card_id or str(card.get("cardId") or "")

    client = _new_canon_client()
    try:
        request = await client.create_runtime_card_request(
            conversation_id,
            card=card,
            card_id=effective_card_id or None,
            expires_at=expires_at,
            response_user_id=response_user_id,
            runtime_id=runtime_id,
            turn_id=turn_id,
            native=native,
        )
        effective_card_id = (
            str(request.get("cardId") or effective_card_id or card.get("cardId") or "").strip()
        )
        interactive = request.get("interactive")
        if interactive is None:
            interactive = _card_has_actions(card)
        if not wait or interactive is False:
            return {
                "status": "displayed",
                "conversationId": conversation_id,
                "cardId": effective_card_id,
                "request": request,
            }
        if not effective_card_id:
            raise RuntimeError("Canon runtime-card request did not return a card id")

        while time.time() * 1000 <= expires_at:
            response = await client.consume_runtime_card_response(
                conversation_id,
                effective_card_id,
            )
            status = response.get("status")
            if status and status != "pending":
                response.setdefault("conversationId", conversation_id)
                return response
            await asyncio.sleep(POLL_SECONDS)

        try:
            await client.consume_runtime_card_response(
                conversation_id,
                effective_card_id,
                cancel=True,
            )
        finally:
            return {
                "status": "timeout",
                "conversationId": conversation_id,
                "cardId": effective_card_id,
            }
    finally:
        await client.close()


async def request_canon_runtime_input(
    *,
    target: Any = None,
    input_id: str,
    kind: str = "clarify",
    title: Optional[str] = None,
    prompt: Optional[str] = None,
    choices: Optional[list[dict[str, Any]]] = None,
    questions: Optional[list[dict[str, Any]]] = None,
    response_user_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    sensitive: Optional[bool] = None,
    native: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    conversation_id = resolve_canon_conversation_id(target)
    timeout_seconds = _safe_timeout_seconds(timeout_seconds)
    expires_at = int((time.time() + timeout_seconds) * 1000)

    client = _new_canon_client()
    try:
        await client.create_runtime_input_request(
            conversation_id,
            input_id=input_id,
            kind=kind,
            expires_at=expires_at,
            title=title,
            prompt=prompt,
            choices=choices,
            questions=questions,
            response_user_id=response_user_id,
            turn_id=turn_id,
            sensitive=sensitive,
            native=native,
        )

        while time.time() * 1000 <= expires_at:
            response = await client.consume_runtime_input_response(
                conversation_id,
                input_id,
            )
            status = response.get("status")
            if status and status != "pending":
                response.setdefault("conversationId", conversation_id)
                return response
            await asyncio.sleep(POLL_SECONDS)

        try:
            await client.consume_runtime_input_response(
                conversation_id,
                input_id,
                cancel=True,
            )
        finally:
            return {
                "status": "timeout",
                "conversationId": conversation_id,
                "inputId": input_id,
            }
    finally:
        await client.close()


async def canon_runtime_control(args: dict[str, Any], **_kwargs) -> str:
    action = str(args.get("action") or "").strip()
    try:
        if action in {"request_card", "send_card"}:
            card = args.get("card")
            if not isinstance(card, dict):
                return tool_error("card must be a canon.card.v1 object", success=False)
            result = await request_canon_runtime_card(
                target=args.get("target"),
                card=card,
                card_id=args.get("cardId"),
                response_user_id=args.get("responseUserId"),
                runtime_id=args.get("runtimeId"),
                turn_id=args.get("turnId"),
                timeout_seconds=_safe_timeout_seconds(args.get("timeoutSeconds")),
                wait=action == "request_card",
                native=args.get("native") if isinstance(args.get("native"), dict) else None,
            )
            return tool_result(success=True, **result)
        if action == "request_input":
            input_id = str(args.get("inputId") or "").strip()
            if not input_id:
                return tool_error("inputId is required for request_input", success=False)
            questions = args.get("questions")
            choices = args.get("choices")
            result = await request_canon_runtime_input(
                target=args.get("target"),
                input_id=input_id,
                kind=str(args.get("kind") or "clarify"),
                title=args.get("title"),
                prompt=args.get("prompt"),
                choices=choices if isinstance(choices, list) else None,
                questions=questions if isinstance(questions, list) else None,
                response_user_id=args.get("responseUserId"),
                turn_id=args.get("turnId"),
                timeout_seconds=_safe_timeout_seconds(args.get("timeoutSeconds")),
                sensitive=args.get("sensitive") if isinstance(args.get("sensitive"), bool) else None,
                native=args.get("native") if isinstance(args.get("native"), dict) else None,
            )
            return tool_result(success=True, **result)
    except Exception as exc:
        return tool_error(str(exc), success=False)

    return tool_error("action must be one of request_card, send_card, or request_input", success=False)


CANON_RUNTIME_CONTROL_SCHEMA = {
    "name": "canon_runtime_control",
    "description": (
        "Generic Canon runtime card/input request plumbing. Use only for Canon "
        "runtime-control surfaces; domain-specific cards should be built by the "
        "owning domain tool first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["request_card", "send_card", "request_input"],
            },
            "target": {
                "type": "string",
                "description": "Optional Canon target, formatted as canon:<conversation_id>. Defaults to the active Canon chat.",
            },
            "card": {"type": "object"},
            "cardId": {"type": "string"},
            "inputId": {"type": "string"},
            "kind": {"type": "string"},
            "title": {"type": "string"},
            "prompt": {"type": "string"},
            "choices": {"type": "array", "items": {"type": "object"}},
            "questions": {"type": "array", "items": {"type": "object"}},
            "responseUserId": {"type": "string"},
            "runtimeId": {"type": "string"},
            "turnId": {"type": "string"},
            "timeoutSeconds": {"type": "integer"},
            "sensitive": {"type": "boolean"},
            "native": {"type": "object"},
        },
        "required": ["action"],
    },
}


registry.register(
    name="canon_runtime_control",
    toolset="canon",
    schema=CANON_RUNTIME_CONTROL_SCHEMA,
    handler=canon_runtime_control,
    check_fn=lambda: bool(os.getenv("CANON_API_KEY") or os.getenv("CANON_AGENT")),
    is_async=True,
    description=CANON_RUNTIME_CONTROL_SCHEMA["description"],
)
registry.register_toolset_alias("canon_runtime", "canon")
