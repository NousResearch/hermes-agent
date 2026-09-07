#!/usr/bin/env python3
"""Propose one or more connectors to the user as an inline card in desktop chat.

The card (a switch per connector, plus whatever each one actually needs —
nothing, an API key, or a browser sign-in) lives in the desktop renderer, so
this tool round-trips through the gateway's blocking-prompt bridge — the same
one ``clarify`` uses: tui_gateway emits ``mcp.setup.request``, the renderer
walks the user through the flow via the existing REST endpoints (catalog
install, registry lookup, enable, OAuth), and answers ``mcp.setup.respond``
once the flow settles. This module is just schema + a thin dispatcher over the
platform-injected callback.

Two properties the schema deliberately encodes:

**Connectors, not mechanisms.** The default action is ``connect`` — "make this
usable" — because the model cannot know whether a server is missing,
configured-but-disabled, or configured-but-signed-out, and shouldn't have to
guess. The renderer resolves the connector's real state and does the right
thing. ``install`` / ``enable`` / ``authorize`` remain for the rare case the
agent genuinely means one specific step.

**One card, several connectors.** A task that needs Jira *and* Figma should
ask once. ``servers`` takes a list; the card renders a row per connector with
its own switch, and the tool returns a per-connector outcome so the agent
knows exactly which ones came back.

Lives in the ``desktop_ui`` toolset, which the GUI gateway enables only for
desktop-sourced sessions — on every other surface the agent falls back to
``hermes mcp install <name>`` in the terminal.
"""

import json
from typing import Any, Callable, Dict, List, Optional

from tools.registry import registry, tool_error

_ACTIONS = ("connect", "install", "enable", "authorize")

# One card should be a decision, not a survey. Beyond a handful the user is
# being asked to audit a list rather than consent to it.
_MAX_CONNECTORS = 5

# Prerequisites are a short checklist, not a manual. Past a handful the card
# stops being something the user reads before clicking Connect.
_MAX_STEPS = 6
_MAX_STEP_CHARS = 200


def _connector_names(server: str, servers: Any) -> List[str]:
    """Merge the singular and list forms into deduplicated, ordered names."""
    raw: List[Any] = []

    if isinstance(servers, str):
        # Models occasionally send a JSON array or a comma list as a string.
        try:
            decoded = json.loads(servers)
            raw.extend(decoded if isinstance(decoded, list) else [servers])
        except (TypeError, ValueError):
            raw.extend(servers.split(","))
    elif isinstance(servers, (list, tuple)):
        raw.extend(servers)

    if server:
        raw.insert(0, server)

    names: List[str] = []
    for item in raw:
        name = str(item or "").strip()
        if name and name not in names:
            names.append(name)

    return names


def _clean_steps(steps: Any) -> List[str]:
    """Normalize model-supplied prerequisites into short, ordered lines."""
    if isinstance(steps, str):
        try:
            decoded = json.loads(steps)
        except (TypeError, ValueError):
            decoded = [steps]
        steps = decoded if isinstance(decoded, list) else [steps]

    if not isinstance(steps, (list, tuple)):
        return []

    cleaned = []
    for item in steps:
        text = " ".join(str(item or "").split())
        if text:
            cleaned.append(text[:_MAX_STEP_CHARS])

    return cleaned[:_MAX_STEPS]


def _invoke_setup_callback(
    callback: Callable, names: List[str], action: str, reason: str, steps: List[str]
):
    """Call the platform callback, passing steps only if it takes them.

    Signature inspection rather than a TypeError retry: a retry would re-run a
    callback that raised TypeError internally, and this one puts a card in
    front of the user — showing it twice is worse than dropping the steps.
    """
    if not steps:
        return callback(names, action, reason)

    import inspect

    try:
        params = inspect.signature(callback).parameters
        takes_steps = "steps" in params or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
    except (TypeError, ValueError):
        takes_steps = False

    if not takes_steps:
        return callback(names, action, reason)

    return callback(names, action, reason, steps=steps)


def setup_mcp_tool(
    server: str = "",
    servers: Any = None,
    action: str = "connect",
    reason: str = "",
    steps: Any = None,
    callback: Optional[Callable] = None,
) -> str:
    """Ask the desktop GUI to run a connector setup flow; return its JSON outcome."""
    if callback is None:
        return tool_error(
            "setup_mcp is only available in the Hermes desktop app. Use the "
            "terminal instead: `hermes mcp install <name>` for catalog entries, "
            "`hermes mcp login <name>` for sign-in."
        )

    names = _connector_names((server or "").strip(), servers)
    if not names:
        return tool_error("server (or servers) is required — the connector name(s) to set up.")

    if len(names) > _MAX_CONNECTORS:
        return tool_error(
            f"Too many connectors in one card ({len(names)}); ask for at most "
            f"{_MAX_CONNECTORS} at a time so the user can actually read it."
        )

    action = (action or "connect").strip().lower()
    if action not in _ACTIONS:
        return tool_error(f"action must be one of {', '.join(_ACTIONS)}.")

    try:
        raw = _invoke_setup_callback(callback, names, action, (reason or "").strip(), _clean_steps(steps))
    except Exception as exc:
        return tool_error(f"Connector setup flow failed: {exc}")

    if not raw:
        # The renderer never answered (timeout / closed window). Distinct from
        # an explicit decline, which arrives as {"status": "declined"}.
        return json.dumps(
            {
                "status": "unanswered",
                "server": names[0],
                "servers": names,
                "note": (
                    "The user did not respond to the setup card. Do not retry "
                    "immediately; continue without the connector or ask in chat."
                ),
            },
            ensure_ascii=False,
        )

    # Desktop answers with a JSON object; pass it through, else wrap the raw text.
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return json.dumps({"status": "error", "detail": str(raw)}, ensure_ascii=False)

    return json.dumps(_normalize_outcome(parsed, names), ensure_ascii=False)


def _normalize_outcome(parsed: Any, names: List[str]) -> Dict[str, Any]:
    """Guarantee the agent sees a per-connector breakdown.

    A single-connector card may answer in the pre-existing flat shape
    (``{status, server, tools}``); lift it into ``connectors`` so the model
    reads one contract regardless of how many it asked for.
    """
    if not isinstance(parsed, dict):
        return {"status": "error", "detail": str(parsed)}

    connectors = parsed.get("connectors")
    if not isinstance(connectors, list):
        connectors = [
            {
                "server": parsed.get("server") or names[0],
                "status": parsed.get("status", "error"),
                **({"tools": parsed["tools"]} if isinstance(parsed.get("tools"), list) else {}),
                **({"detail": parsed["detail"]} if parsed.get("detail") else {}),
            }
        ]

    parsed["connectors"] = connectors
    parsed.setdefault("server", connectors[0].get("server") if connectors else names[0])

    return parsed


SETUP_MCP_SCHEMA = {
    "name": "setup_mcp",
    "description": (
        "Propose one or more connectors (MCP servers) to the user as an inline "
        "consent card in the Hermes desktop chat. The card shows a switch per "
        "connector and handles whatever each one needs — nothing at all, an "
        "API key, or a browser sign-in — right there, without opening the "
        "Capabilities tab, and blocks until they act or decline. Use when the "
        "user asks to add/set up a connector (e.g. \"add the linear mcp\"), or "
        "when a task clearly needs one that is missing or signed out. Pass "
        "several names in `servers` to ask once instead of one card at a time. "
        "This is the only way to set a connector up: do not ask the user for an "
        "API key or token, do not ask which authentication method they want, "
        "and do not offer to call the vendor's REST API by hand instead. "
        "A decline applies to the request in hand, not forever: drop the "
        "connector and continue without it rather than re-offering it to "
        "finish the same task, but a later task that needs it — or a retry "
        "after an attempt that failed — is a fresh ask. "
        "Returns JSON {status: connected|partial|declined|unanswered|error, "
        "connectors: [{server, status, tools?, detail?}]}. On declined or "
        "unanswered, continue without the connector. Connectors can come from "
        "the reviewed catalog or the public MCP registry — you do not need to "
        "know which; pass the name the user would recognize (\"notion\")."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "server": {
                "type": "string",
                "description": "A single connector name. Use `servers` for more than one.",
            },
            "servers": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Connector names to offer on one card (max 5), e.g. "
                    "[\"atlassian\", \"figma\"]. Preferred over several "
                    "sequential single-connector calls."
                ),
            },
            "action": {
                "type": "string",
                "enum": ["connect", "install", "enable", "authorize"],
                "description": (
                    "connect (default): make the connector usable, whatever "
                    "that takes — the app resolves whether it needs adding, "
                    "enabling, or signing in. Use install/enable/authorize "
                    "only when you specifically mean that one step."
                ),
            },
            "reason": {
                "type": "string",
                "description": (
                    "One short sentence shown on the card: why these help "
                    "right now (e.g. \"To read the JIRA ticket you linked\")."
                ),
            },
            "steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Prerequisites the user must do elsewhere before the "
                    "connector can work — creating a cloud project, enabling "
                    "an API, turning on a plugin. Max 6 short lines, in order, "
                    "with markdown links to the exact page: \"Create a token "
                    "at [Settings > API](https://example.com/settings/api)\". "
                    "Reviewed connectors already carry their own steps and "
                    "will ignore these; supply them for a connector that has "
                    "none, instead of explaining the setup in chat where the "
                    "user has to hold it in their head while filling the card."
                ),
            },
        },
        "required": [],
    },
}


registry.register(
    name="setup_mcp",
    toolset="desktop_ui",
    schema=SETUP_MCP_SCHEMA,
    handler=lambda args, **kw: setup_mcp_tool(
        server=args.get("server", ""),
        servers=args.get("servers"),
        action=args.get("action", "connect"),
        reason=args.get("reason", ""),
        steps=args.get("steps"),
        callback=kw.get("callback"),
    ),
    emoji="🔌",
)
