"""Hermes runtime overview tool.

Read-only tool that exposes the same secret-safe runtime summary as the
``hermesfetch`` terminal command and ``/fetch`` slash command to agent sessions.
"""
from __future__ import annotations

import json
from typing import Any

from hermes_cli.fetch import collect_fetch_info, render_fetch_info
from tools.registry import registry


HERMES_FETCH_SCHEMA = {
    "name": "hermes_fetch",
    "description": (
        "Show a secret-safe fastfetch-style overview of this Hermes Agent runtime: "
        "version, profile, persona, model, gateway/platform status, tools, skills, "
        "cron, memory, MCP, host, runtime, and update state. Read-only."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "format": {
                "type": "string",
                "enum": ["text", "plain", "compact", "json"],
                "description": (
                    "Output format. text is the default fastfetch-style view; plain is neutral "
                    "line-oriented text; compact is shorter; json returns structured data."
                ),
                "default": "text",
            },
            "no_persona": {
                "type": "boolean",
                "description": "Use neutral Hermes styling instead of persona-aware styling for text output.",
                "default": False,
            },
        },
        "additionalProperties": False,
    },
}


def _handle_hermes_fetch(args: dict[str, Any], **_: Any) -> str:
    fmt = str(args.get("format") or "text").strip().lower()
    no_persona = bool(args.get("no_persona", False))
    if fmt not in {"text", "plain", "compact", "json"}:
        return json.dumps(
            {"success": False, "error": "format must be one of: text, plain, compact, json"},
            ensure_ascii=False,
        )

    info = collect_fetch_info()
    if fmt == "json":
        return json.dumps({"success": True, "format": fmt, "info": info}, ensure_ascii=False)

    text = render_fetch_info(
        info,
        plain=(fmt == "plain"),
        compact=(fmt == "compact"),
        no_persona=no_persona,
    )
    return json.dumps({"success": True, "format": fmt, "text": text}, ensure_ascii=False)


registry.register(
    name="hermes_fetch",
    toolset="hermes",
    schema=HERMES_FETCH_SCHEMA,
    handler=_handle_hermes_fetch,
    check_fn=lambda: True,
    emoji="🪽",
    max_result_size_chars=50_000,
)
