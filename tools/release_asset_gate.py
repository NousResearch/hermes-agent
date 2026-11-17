"""The human signals that autonomy levels (``agent/autonomy.py``) act on.

- **Sign-off / Hold** live on the Notion *Release Assets* row: a product-team
  member sets ``Status = Approved`` (sign-off) or a hold status (veto).
  ``fetch_notion_page`` + ``row_state`` read it fresh on every gated call, and
  ``row_state`` refuses a page whose parent isn't the configured Release
  Assets data source — so an Approved page somewhere else can't stand in.
- **Veto** is a ❌ reaction (``x`` / ``no_entry`` / ``no_entry_sign``) on the
  announcement the tool posted (``post_announcement``); ``veto_reactions``
  lists who reacted, minus the bot itself.

Every failure raises ``GateError`` so callers fail closed: no signal readable
→ no action. Credentials come only from ``agent.secret_scope.get_secret``
(``NOTION_API_KEY`` / ``NOTION_API_TOKEN``, ``SLACK_BOT_TOKEN``).

Not a tool module (registers nothing); shared by gated write tools —
``zendesk_create_draft`` today.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

NOTION_VERSION = "2025-09-03"
_TIMEOUT_SECONDS = 20
_VETO_EMOJI = frozenset({"x", "no_entry", "no_entry_sign"})


class GateError(RuntimeError):
    """A human signal could not be read or did not check out — fail closed."""


class SlackRejected(GateError):
    """Slack answered ``ok: false`` — the call definitely did nothing (unlike
    a timeout or connection error, after which a post may have landed)."""


class RowStopped(GateError):
    """The row is Published or on hold. For an action waiting out its veto
    window this IS the veto; for a fresh call it is a plain refusal."""


@dataclass(frozen=True)
class RowState:
    status: str
    asset_type: str
    destination: str
    url: str
    title: str = ""


def _secret(*names: str) -> str:
    from agent.secret_scope import UnscopedSecretError, get_secret
    try:
        for name in names:
            val = (get_secret(name) or "").strip()
            if val:
                return val
    except UnscopedSecretError:
        return ""
    return ""


async def _http_json(method: str, url: str, headers: dict, **kw) -> tuple[int, dict]:
    """One HTTP call returning ``(status, parsed_json)``; indirection for tests."""
    import aiohttp
    from gateway.platforms.base import resolve_proxy_url, proxy_kwargs_for_aiohttp

    _sess_kw, _req_kw = proxy_kwargs_for_aiohttp(resolve_proxy_url())
    timeout = aiohttp.ClientTimeout(total=_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout, **_sess_kw) as session:
        async with session.request(method, url, headers=headers, **kw, **_req_kw) as resp:
            data = await resp.json(content_type=None)
            return resp.status, data if isinstance(data, dict) else {}


def _hex_id(value: str) -> str:
    return (value or "").replace("-", "").strip().lower()


# ── Notion ───────────────────────────────────────────────────────────────

async def fetch_notion_page(page_id: str) -> dict:
    hexid = _hex_id(page_id)
    if not re.fullmatch(r"[0-9a-f]{32}", hexid):
        raise GateError(f"release_asset_row_id {page_id!r} is not a Notion page id.")
    token = _secret("NOTION_API_KEY", "NOTION_API_TOKEN")
    if not token:
        raise GateError("Notion not configured (NOTION_API_KEY) — cannot read the sign-off row.")
    headers = {"Authorization": f"Bearer {token}", "Notion-Version": NOTION_VERSION}
    try:
        status, data = await _http_json("GET", f"https://api.notion.com/v1/pages/{page_id.strip()}", headers)
    except Exception as e:
        raise GateError(f"Notion request failed: {e}") from e
    if status != 200:
        raise GateError(f"Notion API error (HTTP {status}): {str(data.get('message', ''))[:200]}")
    return data


def _select_name(prop: dict) -> str:
    if not isinstance(prop, dict):
        return ""
    val = prop.get(prop.get("type") or "select")
    if isinstance(val, dict):
        return str(val.get("name") or "").strip()
    return ""


def row_state(page: dict, expected_parent: str, *, status_property: str = "Status",
              asset_type_property: str = "Asset type",
              config_key: str = "zendesk.release_assets_data_source_id") -> RowState:
    """Read Status / Asset type / Destination from a Release Assets row.

    Raises ``GateError`` unless the page's parent data source (or database)
    is ``expected_parent`` — an unset expectation fails closed.
    """
    want = _hex_id(expected_parent)
    if not want:
        raise GateError(
            f"{config_key} is not configured — cannot verify "
            "the sign-off row belongs to Release Assets.")
    parent = page.get("parent") if isinstance(page.get("parent"), dict) else {}
    have = {_hex_id(str(parent.get(k) or "")) for k in ("data_source_id", "database_id")}
    if want not in have:
        raise GateError("That Notion page is not a Release Assets row.")
    if page.get("archived") or page.get("in_trash") or page.get("is_archived"):
        raise RowStopped("That Release Assets row is archived or in the trash.")
    props = page.get("properties") if isinstance(page.get("properties"), dict) else {}
    title = ""
    for prop in props.values():
        if isinstance(prop, dict) and prop.get("type") == "title":
            title = "".join(str(t.get("plain_text") or "") for t in prop.get("title") or []
                            if isinstance(t, dict)).strip()
            break
    return RowState(
        status=_select_name(props.get(status_property)),
        asset_type=_select_name(props.get(asset_type_property)),
        destination=_select_name(props.get("Destination")),
        url=str(page.get("url") or ""),
        title=title,
    )


async def update_page_properties(page_id: str, properties: dict) -> dict:
    """PATCH a page's properties directly (not through ``notion_api``).

    The ONE Notion write that bypasses the write guard (5a's exception): only
    ``tools/release_update_post.py`` calls it, with a fixed body (``Published
    URL`` + ``Status=Published``) after it has posted a Slack-update row.
    """
    hexid = _hex_id(page_id)
    if not re.fullmatch(r"[0-9a-f]{32}", hexid):
        raise GateError(f"{page_id!r} is not a Notion page id.")
    token = _secret("NOTION_API_KEY", "NOTION_API_TOKEN")
    if not token:
        raise GateError("Notion not configured (NOTION_API_KEY) — cannot update the row.")
    headers = {"Authorization": f"Bearer {token}", "Notion-Version": NOTION_VERSION}
    try:
        status, data = await _http_json("PATCH", f"https://api.notion.com/v1/pages/{hexid}",
                                        headers, json={"properties": properties})
    except Exception as e:
        raise GateError(f"Notion request failed: {e}") from e
    if status != 200:
        raise GateError(f"Notion API error (HTTP {status}): {str(data.get('message', ''))[:200]}")
    return data


_MAX_BLOCKS = 2000
_MAX_BLOCK_DEPTH = 8


async def fetch_block_tree(page_id: str) -> list[dict]:
    """The page's content as a block list, each block with its ``children``
    fetched (paginated, depth- and count-capped). Raises ``GateError``."""
    hexid = _hex_id(page_id)
    if not re.fullmatch(r"[0-9a-f]{32}", hexid):
        raise GateError(f"{page_id!r} is not a Notion page id.")
    token = _secret("NOTION_API_KEY", "NOTION_API_TOKEN")
    if not token:
        raise GateError("Notion not configured (NOTION_API_KEY) — cannot read the row's content.")
    headers = {"Authorization": f"Bearer {token}", "Notion-Version": NOTION_VERSION}
    count = 0

    async def children(block_id: str, depth: int) -> list[dict]:
        nonlocal count
        out: list[dict] = []
        cursor = None
        while True:
            params = {"page_size": "100"}
            if cursor:
                params["start_cursor"] = cursor
            try:
                status, data = await _http_json(
                    "GET", f"https://api.notion.com/v1/blocks/{block_id}/children",
                    headers, params=params)
            except Exception as e:
                raise GateError(f"Notion request failed: {e}") from e
            if status != 200:
                raise GateError(f"Notion API error (HTTP {status}): {str(data.get('message', ''))[:200]}")
            for block in data.get("results") or []:
                count += 1
                if count > _MAX_BLOCKS:
                    raise GateError(f"The row's content has more than {_MAX_BLOCKS} blocks.")
                if block.get("has_children") and block.get("type") not in ("child_page", "child_database"):
                    if depth >= _MAX_BLOCK_DEPTH:
                        raise GateError("The row's content is nested too deeply.")
                    block["children"] = await children(str(block.get("id")), depth + 1)
                out.append(block)
            if not data.get("has_more"):
                return out
            cursor = data.get("next_cursor")
            if not cursor:
                return out

    return await children(hexid, 0)


# ── Slack ────────────────────────────────────────────────────────────────

def _slack_headers() -> dict:
    token = _secret("SLACK_BOT_TOKEN")
    if not token:
        raise GateError("Slack bot token not configured (SLACK_BOT_TOKEN).")
    return {"Authorization": f"Bearer {token}"}


async def _slack(method: str, api: str, **kw) -> dict:
    headers = _slack_headers()
    try:
        _status, data = await _http_json(method, f"https://slack.com/api/{api}", headers, **kw)
    except Exception as e:
        raise GateError(f"Slack {api} failed: {e}") from e
    if not data.get("ok"):
        raise SlackRejected(f"Slack {api} error: {data.get('error', 'unknown')}")
    return data


async def post_announcement(channel: str, text: str, thread_ts: str = "") -> str:
    """Post to ``channel`` (in ``thread_ts`` when set); returns the message ts."""
    payload = {"channel": channel, "text": text}
    if thread_ts:
        payload["thread_ts"] = thread_ts
    data = await _slack("POST", "chat.postMessage", json=payload)
    return str(data.get("ts") or "")


async def permalink(channel: str, ts: str) -> str:
    """The message's permalink; falls back to the canonical archive URL."""
    try:
        data = await _slack("GET", "chat.getPermalink", params={"channel": channel, "message_ts": ts})
        link = str(data.get("permalink") or "")
        if link.startswith("https://"):
            return link
    except GateError as e:
        logger.debug("permalink lookup failed: %s", e)
    return f"https://slack.com/archives/{channel}/p{ts.replace('.', '')}"


async def veto_reactions(channel: str, ts: str) -> list[str]:
    """User ids who reacted ❌ to ``channel``/``ts``, excluding the bot."""
    me = str((await _slack("POST", "auth.test")).get("user_id") or "")
    data = await _slack("GET", "reactions.get",
                        params={"channel": channel, "timestamp": ts, "full": "true"})
    reactions = (data.get("message") or {}).get("reactions") or []
    users: set[str] = set()
    for r in reactions:
        if str(r.get("name") or "").split("::")[0] in _VETO_EMOJI:
            users.update(u for u in (r.get("users") or []) if u and u != me)
    return sorted(users)
