"""LinkedIn social workflow helper tools for Discord draft cards.

This module registers the ``linkedin_draft_card`` tool used by Hermes messaging
sessions to create Discord review cards with persistent approval buttons.

Notes
-----
The tool only creates the first-stage review card. It stores the exact draft
body in Hermes workflow state so a later Discord button interaction can copy the
unchanged text to ``#approved-posts``. It never publishes to LinkedIn directly.

Examples
--------
The tool is normally invoked through Hermes tool calling::

    linkedin_draft_card({"target": "discord:#linkedin-drafts", ...})
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Coroutine
from typing import Any

from tools.registry import registry
from tools.send_message_tool import _parse_target_ref, _sanitize_error_text


LINKEDIN_DRAFT_CARD_SCHEMA = {
    "name": "linkedin_draft_card",
    "description": (
        "Create a Discord LinkedIn draft review card with approval buttons. "
        "The exact draft text is stored and copied unchanged to the approved-posts "
        "channel when the approval button is clicked. This does not publish to LinkedIn."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "Discord target channel, e.g. discord:#linkedin-drafts or discord:1500764401446289549.",
            },
            "approved_channel_id": {
                "type": "string",
                "description": "Discord channel ID to receive exact approved drafts, e.g. #approved-posts.",
            },
            "title": {"type": "string", "description": "Short title for the draft card."},
            "draft_body": {"type": "string", "description": "Exact full draft body to preserve on approval."},
            "include_publish_button": {
                "type": "boolean",
                "description": "Whether to show the locked publish button. Defaults false.",
            },
        },
        "required": ["target", "approved_channel_id", "draft_body"],
    },
}


def _check_requirements() -> bool:
    """Return whether Discord is configured well enough to create cards.

    Returns
    -------
    bool
        ``True`` when the Discord gateway platform is enabled and has a bot
        token; otherwise ``False``.

    Notes
    -----
    Tool registration uses this as a capability gate. Failing closed prevents
    the LLM from seeing a card-creation tool when Discord cannot send the card.

    Examples
    --------
    Check availability before exposing the tool schema::

        available = _check_requirements()
    """
    try:
        from gateway.config import Platform, load_gateway_config

        cfg = load_gateway_config()
        pconfig = cfg.platforms.get(Platform.DISCORD)
        return bool(pconfig and pconfig.enabled and pconfig.token)
    except Exception:
        return False


def _run(coro: Coroutine[Any, Any, str]) -> str:
    """Run an async tool coroutine from a synchronous registry handler.

    Parameters
    ----------
    coro
        Coroutine that sends the Discord card and returns a JSON string.

    Returns
    -------
    str
        JSON result from the coroutine.

    Notes
    -----
    Most Hermes tool handlers are synchronous. This bridge handles the common
    no-running-loop case used by gateway workers while preserving the older
    behavior for environments that already own an event loop.

    Examples
    --------
    Execute the card-sending coroutine from the registered handler::

        return _run(_send_card(args))
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return loop.run_until_complete(coro)


def _resolve_discord_target(target: str) -> str:
    """Resolve a user-facing Discord target string to a channel ID.

    Parameters
    ----------
    target
        Tool target such as ``discord:#linkedin-drafts`` or
        ``discord:1500764401446289549``.

    Returns
    -------
    str
        Discord channel ID that can receive the REST message.

    Raises
    ------
    ValueError
        Raised when the target is missing, not Discord, or cannot be resolved.

    Notes
    -----
    The tool accepts the same target syntax as ``send_message`` but ultimately
    needs a channel ID for the Discord REST components endpoint.

    Examples
    --------
    Resolve a named Discord channel::

        channel_id = _resolve_discord_target("discord:#linkedin-drafts")
    """
    parts: list[str] = (target or "").split(":", 1)
    if not parts or parts[0].strip().lower() != "discord":
        raise ValueError("target must be a Discord target like discord:#linkedin-drafts")
    target_ref = parts[1].strip() if len(parts) > 1 else ""
    if not target_ref:
        from gateway.config import Platform, load_gateway_config

        home = load_gateway_config().get_home_channel(Platform.DISCORD)
        if not home:
            raise ValueError("No Discord target or home channel was provided")
        return str(home.chat_id)

    chat_id, _thread_id, is_explicit = _parse_target_ref("discord", target_ref)
    if is_explicit and chat_id:
        return str(chat_id)

    from gateway.channel_directory import resolve_channel_name

    resolved = resolve_channel_name("discord", target_ref)
    if not resolved:
        raise ValueError(f"Could not resolve Discord target {target_ref!r}")
    chat_id, _thread_id, is_explicit = _parse_target_ref("discord", resolved)
    if not is_explicit or not chat_id:
        raise ValueError(f"Resolved Discord target {target_ref!r} to unusable value {resolved!r}")
    return str(chat_id)


async def _send_card(args: dict[str, Any]) -> str:
    """Create a Discord LinkedIn draft review card with persistent buttons.

    Parameters
    ----------
    args
        Tool arguments containing target channel, approved-channel ID, title,
        draft body, and optional first-stage publish-button flag.

    Returns
    -------
    str
        JSON result with the Discord message ID and persisted draft ID.

    Raises
    ------
    ValueError
        Raised when required inputs or Discord configuration are missing.
    RuntimeError
        Raised when Discord rejects the REST message request.

    Notes
    -----
    The exact draft body is saved before the Discord request and then updated
    with the source message ID after the request succeeds. This two-step write
    makes button handling resilient even if the process restarts after sending.

    Examples
    --------
    Send a draft card from the registered tool handler::

        result_json = await _send_card(args)
    """
    import aiohttp

    from gateway.config import Platform, load_gateway_config
    from gateway.platforms.base import resolve_proxy_url, proxy_kwargs_for_aiohttp
    from gateway.platforms.discord import (
        _build_linkedin_draft_button_components,
        save_linkedin_draft_record,
    )

    target_channel_id = _resolve_discord_target(str(args.get("target") or ""))
    approved_channel_id = str(args.get("approved_channel_id") or "").strip()
    draft_body = str(args.get("draft_body") or "")
    if not approved_channel_id or not approved_channel_id.isdigit():
        raise ValueError("approved_channel_id must be a Discord channel ID")
    if not draft_body.strip():
        raise ValueError("draft_body is required")

    title = str(args.get("title") or "LinkedIn draft ready for review").strip()
    draft_id = f"ld_{int(time.time())}_{os.getpid()}"
    cfg = load_gateway_config()
    pconfig = cfg.platforms.get(Platform.DISCORD)
    if not pconfig or not pconfig.token:
        raise ValueError("Discord is not configured")

    content = (
        f"**{title}**\n\n"
        "Workflow status: draft review only — not published to LinkedIn.\n"
        f"Draft ID: `{draft_id}`\n\n"
        "Click **Approve for next stage** to copy the exact stored draft unchanged to approved posts."
    )
    record = {
        "draft_id": draft_id,
        "source_channel_id": target_channel_id,
        "source_message_id": "pending",
        "approved_channel_id": approved_channel_id,
        "title": title,
        "draft_body": draft_body,
        "status": "draft_review",
        "created_at": int(time.time()),
    }
    save_linkedin_draft_record(record)

    payload = {
        "content": content,
        "components": _build_linkedin_draft_button_components(
            draft_id,
            include_publish=bool(args.get("include_publish_button", False)),
        ),
        "allowed_mentions": {"parse": []},
    }
    _proxy = resolve_proxy_url(platform_env_var="DISCORD_PROXY")
    _sess_kw, _req_kw = proxy_kwargs_for_aiohttp(_proxy)
    url = f"https://discord.com/api/v10/channels/{target_channel_id}/messages"
    headers = {"Authorization": f"Bot {pconfig.token}", "Content-Type": "application/json"}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30), **_sess_kw) as session:
        async with session.post(url, headers=headers, json=payload, **_req_kw) as resp:
            body = await resp.text()
            if resp.status not in (200, 201):
                raise RuntimeError(f"Discord API error ({resp.status}): {body}")
            data = json.loads(body)

    record["source_message_id"] = str(data.get("id") or "")
    save_linkedin_draft_record(record)
    return json.dumps(
        {
            "success": True,
            "platform": "discord",
            "chat_id": target_channel_id,
            "message_id": record["source_message_id"],
            "draft_id": draft_id,
            "approved_channel_id": approved_channel_id,
        }
    )


def linkedin_draft_card_tool(args: dict[str, Any], **_kw: Any) -> str:
    """Registry handler for the ``linkedin_draft_card`` tool.

    Parameters
    ----------
    args
        Validated or model-supplied tool arguments.
    **_kw
        Extra registry keyword arguments ignored by this handler.

    Returns
    -------
    str
        JSON result. Errors are sanitized before they are returned to the model.

    Notes
    -----
    Sanitizing errors is important because Discord/API failures may include
    request context. The helper must never echo bot tokens or other credentials.

    Examples
    --------
    Invoke from the tool registry::

        result = linkedin_draft_card_tool(args)
    """
    try:
        return _run(_send_card(args))
    except Exception as exc:
        return json.dumps({"error": _sanitize_error_text(str(exc))})


registry.register(
    name="linkedin_draft_card",
    toolset="messaging",
    schema=LINKEDIN_DRAFT_CARD_SCHEMA,
    handler=linkedin_draft_card_tool,
    check_fn=_check_requirements,
    requires_env=[],
    description="Create a Discord LinkedIn draft card with approval buttons.",
    emoji="🔘",
)
