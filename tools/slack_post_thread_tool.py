"""slack_post_thread — post one text message to a Slack channel or thread.

A headless-safe sibling of ``slack_react``: it takes an EXPLICIT chat_id (and
optional thread_ts — omit it for a new root message) with no session
contextvars, so it works from cron/worker runs, and posts via Slack
``chat.postMessage``. It returns the posted message's ts and a best-effort
permalink so a worker can record a receipt (e.g. link it from a Jira comment). It reuses ``slack_react``'s token self-heal. It
lives in its own NON-FLOOR ``slack_post`` toolset (deliberately NOT the floor
``slack`` toolset, which would hand every valid-role user arbitrary
thread-posting). This is the worker's poster because cron hard-disables the
``messaging`` toolset that ``send_message`` lives in.
"""
from __future__ import annotations

import logging

from tools.registry import registry, tool_error, tool_result
from tools.slack_react_tool import _resolve_slack_token

logger = logging.getLogger(__name__)


SLACK_POST_THREAD_SCHEMA = {
    "name": "slack_post_thread",
    "description": (
        "Post a text message to a Slack channel from a background/cron run "
        "(cron cannot use send_message). Give chat_id and message (Slack "
        "mrkdwn); add thread_ts to reply in that thread, omit it to post a new "
        "root message. Returns message_ts (use it as the thread_ts of later "
        "replies) and permalink."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "chat_id": {
                "type": "string",
                "description": "Slack channel id, e.g. 'C0123ABCD'.",
            },
            "thread_ts": {
                "type": "string",
                "description": "Optional root message ts of the thread to reply in, e.g. '1700000000.000100'. Omit to post a new root message.",
            },
            "message": {
                "type": "string",
                "description": "Message text (Slack mrkdwn).",
            },
        },
        "required": ["chat_id", "message"],
    },
}


async def _post_message(token: str, channel: str, thread_ts: str, text: str) -> dict:
    """POST to Slack chat.postMessage (in a thread when thread_ts is set).
    Returns parsed JSON.

    Mirrors slack_react._post_reaction: a fresh aiohttp session with proxy
    support, safe from any event loop.
    """
    import aiohttp
    from gateway.platforms.base import resolve_proxy_url, proxy_kwargs_for_aiohttp

    url = "https://slack.com/api/chat.postMessage"
    _proxy = resolve_proxy_url()
    _sess_kw, _req_kw = proxy_kwargs_for_aiohttp(_proxy)
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"channel": channel, "text": text}
    if thread_ts:
        payload["thread_ts"] = thread_ts
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15), **_sess_kw) as session:
        async with session.post(url, headers=headers, json=payload, **_req_kw) as resp:
            return await resp.json(content_type=None)


async def _get_permalink(token: str, channel: str, message_ts: str) -> str:
    """Resolve a posted message's permalink via chat.getPermalink ("" on any
    failure — the post already landed, so a missing link must not fail it)."""
    import aiohttp
    from gateway.platforms.base import resolve_proxy_url, proxy_kwargs_for_aiohttp

    try:
        _proxy = resolve_proxy_url()
        _sess_kw, _req_kw = proxy_kwargs_for_aiohttp(_proxy)
        headers = {"Authorization": f"Bearer {token}"}
        params = {"channel": channel, "message_ts": message_ts}
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15), **_sess_kw) as session:
            async with session.get("https://slack.com/api/chat.getPermalink",
                                   headers=headers, params=params, **_req_kw) as resp:
                data = await resp.json(content_type=None)
        return data.get("permalink", "") if data.get("ok") else ""
    except Exception as e:
        logger.debug("[slack_post_thread] permalink lookup failed: %s", e)
        return ""


async def _slack_post_thread_handler(args: dict, **_kw) -> str:
    channel = (args.get("chat_id") or "").strip()
    thread_ts = (args.get("thread_ts") or "").strip()
    message = (args.get("message") or "").strip()
    if not channel or not message:
        return tool_error("'chat_id' and 'message' are required.")

    token = _resolve_slack_token()
    if not token:
        logger.warning("[slack_post_thread] no Slack bot token resolved (channel=%s)", channel)
        return tool_error("Slack bot token not configured (SLACK_BOT_TOKEN).")

    try:
        data = await _post_message(token, channel, thread_ts, message)
    except Exception as e:
        logger.warning("[slack_post_thread] request failed (channel=%s ts=%s): %s",
                       channel, thread_ts, e)
        return tool_error(f"Slack post request failed: {e}")

    if data.get("ok"):
        message_ts = data.get("ts", "")
        permalink = await _get_permalink(token, channel, message_ts) if message_ts else ""
        return tool_result(success=True, channel=channel, message_ts=message_ts,
                           thread_ts=thread_ts or message_ts, permalink=permalink)
    err = data.get("error", "unknown")
    logger.warning("[slack_post_thread] Slack API error '%s' (channel=%s ts=%s)",
                   err, channel, thread_ts)
    return tool_error(f"Slack API error: {err}")


def _check_slack_post_thread() -> bool:
    """Available whenever a Slack token is resolvable."""
    return bool(_resolve_slack_token())


registry.register(
    name="slack_post_thread",
    toolset="slack_post",
    schema=SLACK_POST_THREAD_SCHEMA,
    handler=lambda args, **kw: _slack_post_thread_handler(args, **kw),
    check_fn=_check_slack_post_thread,
    requires_env=[],
    is_async=True,
    emoji="💬",
    max_result_size_chars=2000,
)
