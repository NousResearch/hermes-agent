"""Slack message maintenance tools for Hermes gateway/local runtime.

Uses Hermes' SLACK_BOT_TOKEN. The delete helper is intended for deleting
messages authored by the bot, usually from Slack cleanup requests where the
caller has a message ts.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import Any, Dict

from tools.registry import registry

SLACK_API_BASE = "https://slack.com/api"


def check_slack_message_requirements() -> bool:
    return bool(os.getenv("SLACK_BOT_TOKEN", "").strip())


def _token() -> str:
    token = os.getenv("SLACK_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("SLACK_BOT_TOKEN is not set")
    return token


def _default_channel() -> str:
    return (
        os.getenv("SLACK_CHANNEL", "").strip()
        or os.getenv("SLACK_HOME_CHANNEL", "").strip()
    )


def _slack_api(method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    body = urllib.parse.urlencode(
        {key: value for key, value in params.items() if value is not None}
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{SLACK_API_BASE}/{method}",
        data=body,
        headers={
            "Authorization": f"Bearer {_token()}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def slack_delete_message(ts: str, channel: str | None = None) -> str:
    """Delete a Slack message by timestamp from the given/default channel."""
    ts = str(ts or "").strip()
    resolved_channel = str(channel or "").strip() or _default_channel()
    if not ts:
        return json.dumps({"ok": False, "error": "missing_ts"}, ensure_ascii=False)
    if not resolved_channel:
        return json.dumps(
            {
                "ok": False,
                "error": "missing_channel",
                "hint": "Pass channel or set SLACK_CHANNEL/SLACK_HOME_CHANNEL.",
            },
            ensure_ascii=False,
        )

    data = _slack_api("chat.delete", {"channel": resolved_channel, "ts": ts})
    return json.dumps(
        {
            "ok": bool(data.get("ok")),
            "channel": data.get("channel", resolved_channel),
            "ts": data.get("ts", ts),
            "error": data.get("error"),
        },
        ensure_ascii=False,
    )


SLACK_DELETE_MESSAGE_SCHEMA = {
    "name": "slack_delete_message",
    "description": "Delete a Slack message by ts using Hermes' SLACK_BOT_TOKEN. Defaults to SLACK_CHANNEL/SLACK_HOME_CHANNEL when channel is omitted. Usually only bot-authored messages can be deleted.",
    "parameters": {
        "type": "object",
        "properties": {
            "ts": {
                "type": "string",
                "description": "Slack message timestamp, e.g. 1781830158.966569 or the p-link timestamp converted to dotted form.",
            },
            "channel": {
                "type": "string",
                "description": "Optional Slack channel ID, e.g. C07NTPS63QE. If omitted, uses SLACK_CHANNEL then SLACK_HOME_CHANNEL.",
            },
        },
        "required": ["ts"],
    },
}


registry.register(
    name="slack_delete_message",
    toolset="slack",
    schema=SLACK_DELETE_MESSAGE_SCHEMA,
    handler=lambda args, **kw: slack_delete_message(
        args.get("ts", ""), args.get("channel")
    ),
    check_fn=check_slack_message_requirements,
    requires_env=["SLACK_BOT_TOKEN"],
    emoji="🧹",
)
