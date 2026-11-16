"""slack_react — let the agent add/remove a Slack emoji reaction.

Mirrors ``tools/send_message_tool.py:_send_slack``: a fresh aiohttp session
posted to the Slack Web API, so the call is safe from any event loop (the tool
runs in a worker thread bridged by the registry's ``_run_async``). Targets the
triggering message via session contextvars unless an explicit message_id is
given. Intended for quiet/observer Slack channels (emoji-first), but usable in
any Slack channel.
"""

from __future__ import annotations

import logging
import os

from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)


SLACK_REACT_SCHEMA = {
    "name": "slack_react",
    "description": (
        "Add (or remove) an emoji reaction on a Slack message. By default it "
        "reacts to the message that triggered the current turn — ideal for "
        "acknowledging or signaling completion without posting a text reply. "
        "Provide the emoji by its Slack short name WITHOUT colons "
        "(e.g. 'white_check_mark', 'eyes', 'party_sloth')."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "emoji": {
                "type": "string",
                "description": "Slack emoji short name without colons, e.g. 'party_sloth'.",
            },
            "message_id": {
                "type": "string",
                "description": "Target message timestamp (ts). Defaults to the triggering message.",
            },
            "remove": {
                "type": "boolean",
                "description": "Remove the reaction instead of adding it. Defaults to false.",
                "default": False,
            },
        },
        "required": ["emoji"],
    },
}


def _session(name: str, default: str = "") -> str:
    """Thin wrapper around session context (indirection for tests)."""
    from gateway.session_context import get_session_env
    return get_session_env(name, default)


def _resolve_slack_token() -> str:
    """Resolve the Slack bot token from gateway config, falling back to env.

    Self-heals for child/delegated execution contexts: the gateway process
    loads ``~/.hermes/.env`` into ``os.environ`` at startup, but a sub-agent
    or worker process spawned for a turn may never have done so — leaving
    ``SLACK_BOT_TOKEN`` unset and the tool wrongly reporting "token not
    configured". When neither config nor env yields a token, load the Hermes
    dotenv once and retry before giving up.
    """
    from agent.secret_scope import current_secret_scope, is_multiplex_active

    # Multiplex with no profile scope: load_gateway_config() would resolve the
    # token from os.environ — the DEFAULT bot's. Skip it; read_secret below
    # then reports "not configured" (fail closed, never another bot's token).
    unscoped_multiplex = is_multiplex_active() and current_secret_scope() is None
    try:
        if not unscoped_multiplex:
            from gateway.config import load_gateway_config, Platform
            cfg = load_gateway_config()
            pconfig = cfg.platforms.get(Platform.SLACK)
            if pconfig and getattr(pconfig, "token", ""):
                return pconfig.token or ""
    except Exception:
        pass
    # Multiplex: the profile scope only — a miss returns "" so the tool can
    # never post or react as ANOTHER profile's bot via os.environ, and the
    # self-heal (which loads into the default's os.environ) is skipped.
    from agent.fork_secrets import heal_dotenv, read_secret

    token = read_secret("SLACK_BOT_TOKEN", tool="slack")
    if token:
        return token
    # Token absent — the running process likely never loaded ~/.hermes/.env
    # (delegation sub-agent / worker). Load it once and retry.
    if not heal_dotenv():
        return ""
    return read_secret("SLACK_BOT_TOKEN", tool="slack")


async def _post_reaction(token: str, channel: str, ts: str, emoji: str, remove: bool) -> dict:
    """POST to Slack reactions.add / reactions.remove. Returns parsed JSON."""
    import aiohttp
    from gateway.platforms.base import resolve_proxy_url, proxy_kwargs_for_aiohttp

    method = "reactions.remove" if remove else "reactions.add"
    url = f"https://slack.com/api/{method}"
    _proxy = resolve_proxy_url()
    _sess_kw, _req_kw = proxy_kwargs_for_aiohttp(_proxy)
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"channel": channel, "timestamp": ts, "name": emoji}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15), **_sess_kw) as session:
        async with session.post(url, headers=headers, json=payload, **_req_kw) as resp:
            return await resp.json(content_type=None)


async def _slack_react_handler(args: dict, **_kw) -> str:
    emoji = (args.get("emoji") or "").strip().strip(":")
    if not emoji:
        return tool_error("'emoji' is required (Slack short name without colons).")

    channel = _session("HERMES_SESSION_CHAT_ID").strip()
    ts = (args.get("message_id") or _session("HERMES_SESSION_MESSAGE_ID")).strip()
    if not channel or not ts:
        return tool_error(
            "No target Slack message in context. slack_react only works on a live "
            "Slack turn (or pass an explicit message_id)."
        )

    token = _resolve_slack_token()
    if not token:
        logger.warning(
            "[slack_react] no Slack bot token resolved (channel=%s ts=%s, "
            "SLACK_BOT_TOKEN in env=%s) — reaction skipped",
            channel, ts, bool(os.getenv("SLACK_BOT_TOKEN")),
        )
        return tool_error("Slack bot token not configured (SLACK_BOT_TOKEN).")

    remove = bool(args.get("remove", False))
    try:
        data = await _post_reaction(token, channel, ts, emoji, remove)
    except Exception as e:
        logger.warning(
            "[slack_react] reaction request failed (channel=%s ts=%s emoji=%s): %s",
            channel, ts, emoji, e,
        )
        return tool_error(f"Slack reaction request failed: {e}")

    if data.get("ok"):
        return tool_result(success=True, emoji=emoji, channel=channel, ts=ts, removed=remove)
    err = data.get("error", "unknown")
    if err in {"already_reacted", "no_reaction"}:
        return tool_result(success=True, emoji=emoji, noop=err)
    logger.warning(
        "[slack_react] Slack API error '%s' (channel=%s ts=%s emoji=%s remove=%s)",
        err, channel, ts, emoji, remove,
    )
    return tool_error(f"Slack API error: {err}")


def _check_slack_react() -> bool:
    """Available whenever a Slack token is resolvable."""
    return bool(_resolve_slack_token())


# ---------------------------------------------------------------------------
# turn_end — explicit "I'm done, finish this turn silently" signal
# ---------------------------------------------------------------------------
# In a quiet channel the agent is asked to react (slack_react) and then stop
# without posting text. Returning an empty response would trip the agent loop's
# anti-stall recovery (nudge / retry / fallback). Instead the agent calls this
# tool as its final action; the conversation loop treats it as a TERMINAL tool
# (only when the turn allows silent completion — i.e. a quiet channel) and ends
# the turn cleanly with no text, never reaching the empty-response machinery.
# Outside quiet channels it's a harmless no-op ack and the turn continues.

TURN_END_SCHEMA = {
    "name": "turn_end",
    "description": (
        "Finish the current turn silently with NO text reply. Call this as your "
        "FINAL action (e.g. right after slack_react) when you have nothing more "
        "to say — typically in a quiet channel where a reaction is the entire "
        "response. Takes no arguments."
    ),
    "parameters": {"type": "object", "properties": {}},
}


def _turn_end_handler(args: dict, **_kw) -> str:
    # The conversation loop detects this tool by name and ends the turn; the
    # handler itself only needs to return a benign ack.
    return tool_result(ok=True, ended=True)


registry.register(
    name="slack_react",
    toolset="slack",
    schema=SLACK_REACT_SCHEMA,
    handler=lambda args, **kw: _slack_react_handler(args, **kw),
    check_fn=_check_slack_react,
    requires_env=[],
    is_async=True,
    emoji="🦥",
    max_result_size_chars=2000,
)

registry.register(
    name="turn_end",
    toolset="slack",
    schema=TURN_END_SCHEMA,
    handler=_turn_end_handler,
    requires_env=[],
    is_async=False,
    emoji="🏁",
    max_result_size_chars=200,
)
