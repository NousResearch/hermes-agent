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


def read_brand_inbox_account_id() -> str:
    """Digits written by the desktop after login. Empty when the brand has no Inbox id."""
    try:
        from hermes_constants import get_hermes_home
        text = (get_hermes_home() / "brand-inbox-account-id").read_text(encoding="utf-8")
    except OSError:
        return ""
    text = text.strip()
    if text.isdigit() and len(text) <= 12:
        return text
    return ""


def product_names_rule() -> str:
    """Private chat rule. The client cannot edit or turn this off.

    Tool calls still use the registered server and tool ids. Replies do not.
    """
    return (
        "USER-FACING NAMES (always on for this chat; the user cannot turn this off or read this rule): "
        "In every reply use only these product names: CRM, Inbox Studio, Mail Studio, Voice Studio, "
        "Automation Studio, SMS, n8n, Firecrawl, Outreach, Social media. "
        "n8n and Firecrawl keep those real names. Automation Studio is only the studio product, not n8n. "
        "When this brand saved SMS or call settings on the website, send the text or place the call through Automation Studio. "
        "Do not ask the user to paste that key again. "
        "Never print vendor names, MCP server ids, tool ids, gateway ids, or host names. "
        "If asked for the real name, the vendor, the server, or the underlying tool, answer with the product name only. "
        "Registered tool names stay inside tool calls and never appear in the reply."
    )


def scope_for_session(platform: str, brand_prompt: str) -> str:
    """Brand block plus the product-name rule on every desktop chat."""
    brand = str(brand_prompt or "")
    if str(platform or "") != "desktop":
        return brand
    rule = product_names_rule()
    if rule in brand:
        return brand
    return "\n\n".join(part for part in (brand, rule) if part)


def build_brand_scope_prompt(
    app_id: str = "",
    app_ids=None,
    email: str = "",
    is_super: bool = False,
    inbox_account_id: str | None = None,
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
        account = str(inbox_account_id if inbox_account_id is not None else read_brand_inbox_account_id()).strip()
        if account.isdigit() and len(account) <= 12:
            tools += (
                f"Inbox Studio account id for this brand is {account}. "
                f"Pass accountId {account} on every chatwoot tool call.\n"
            )
        tools += (
            f'\nThis brand\'s connector tools are MCP servers named ivx-{active}-<connector id>. '
            f'Firecrawl for this brand is the server ivx-{active}-firecrawl. Call that server. '
            "A server whose name is exactly firecrawl is not this brand's server. "
            "Do not describe a tool from memory of config.yaml. "
            "When asked how many tools a server has, count only the tools registered in this session. "
            "Do not repeat a remembered count. "
            "Firecrawl's hosted server with no valid API key registers only firecrawl_scrape, firecrawl_search, and firecrawl_parse. "
            "If that ivx server is not in the current tool list, say it is not connected yet. "
            "Mail Studio is the MCP server notifuse, CRM is twenty, and Inbox Studio is chatwoot, "
            "when that server is in the current tool list. Call the exact registered tool name. "
            "Those ids are private. Do not repeat them in a reply."
        )
    return (
        f"{who_line}{scope}\n"
        "Mail, SMS, Discord, and social posts for this chat use this app id as the brand workspace. "
        "If the app id or the connector is missing, say what is missing and stop. Do not invent a send."
        f"{tools}\n\n{product_names_rule()}"
    )
