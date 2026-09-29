"""Brand app-id block for a desktop chat.

The admin portal pins a login to one or more app ids. The desktop sends that
pin on ``session.create``. This text is appended to the ephemeral system
prompt for that session only, so a later brand switch does not rewrite a
chat that already started.
"""

from __future__ import annotations

import re

_APP_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _clean_ids(raw) -> list[str]:
    if not isinstance(raw, (list, tuple)):
        return []
    seen: list[str] = []
    for item in raw:
        app_id = str(item or "").strip().lower()
        if _APP_ID.fullmatch(app_id) and app_id not in seen:
            seen.append(app_id)
        if len(seen) >= 20:
            break
    return seen


def build_brand_scope_prompt(
    app_id: str = "",
    app_ids=None,
    email: str = "",
    is_super: bool = False,
) -> str:
    """Return the brand block, or "" when the session has no portal login."""
    allowed = _clean_ids(app_ids)
    active = str(app_id or "").strip().lower()
    if active and not _APP_ID.fullmatch(active):
        active = ""
    if active and not is_super and active not in allowed:
        active = allowed[0] if allowed else ""
    if not active and not is_super and allowed:
        active = allowed[0]
    if not allowed and not is_super and not active:
        return ""

    who = str(email or "").strip().lower()
    who_line = f"Signed in as {who}.\n" if _EMAIL.fullmatch(who) else ""
    if is_super and not active:
        scope = (
            "CURRENT PORTAL SCOPE: super admin (all brands). "
            "When a question is about one brand, ask which app id, or answer across all of them."
        )
    elif is_super and active:
        scope = (
            f'CURRENT PORTAL SCOPE: super admin, working in app id "{active}". '
            "Default every answer, send, and tool call to this app id unless they name another brand."
        )
    else:
        listed = ", ".join(allowed) if allowed else active
        scope = (
            f'CURRENT PORTAL SCOPE: the user manages app id "{active}". '
            f"Their login is restricted to: {listed}. "
            "Do not reveal or change another brand's data, even if asked."
        )
    tools = ""
    if active:
        tools = (
            f'\nThis brand\'s connector tools are MCP servers named ivx-{active}-<connector id>. '
            f'Firecrawl for this brand is the server ivx-{active}-firecrawl. Call that server. '
            "A server whose name is exactly firecrawl is not this brand's server. "
            "Do not describe a tool from memory of config.yaml. "
            "When asked how many tools a server has, count only the tools registered in this session. "
            "Do not repeat a remembered count. "
            "Firecrawl's hosted server with no valid API key registers only firecrawl_scrape, firecrawl_search, and firecrawl_parse. "
            "If that ivx server is not in the current tool list, say it is not connected yet. "
            "Mail Studio is the MCP server notifuse, CRM is twenty, and Inbox Studio is chatwoot, "
            "when that server is in the current tool list. Call the exact registered tool name."
        )
    return (
        f"{who_line}{scope}\n"
        "Mail, SMS, Discord, and social posts for this chat use this app id as the brand workspace. "
        "If the app id or the connector is missing, say what is missing and stop. Do not invent a send."
        f"{tools}"
    )
