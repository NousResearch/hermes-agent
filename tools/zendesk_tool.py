"""zendesk — draft-only access to the Zendesk Help Center (Guide).

Two tools, two toolsets, so RBAC can grant reads without writes:

- ``zendesk_api`` (toolset ``zendesk``) — GET-only, and only on Help Center
  paths (categories, sections, articles, translations, attachments, search,
  user segments, permission groups, locales). Tickets, users and every other
  Support API are outside the allowlist: ticket reads, when wanted, become a
  SEPARATE ``zendesk_tickets`` toolset.
- ``zendesk_create_draft`` (toolset ``zendesk_write``) — creates ONE
  unpublished article. It is a structured tool: the model supplies a section,
  title and HTML body, never a method or a path. Every write goes through
  ``_zendesk_write``, which re-checks the curated ``_WRITE_ALLOWLIST`` and
  refuses any body that isn't ``draft: true`` with ``notify_subscribers:
  false``. Zendesk API tokens carry no scopes (the shared service account can
  publish), so that chokepoint IS the boundary — it never publishes, archives,
  deletes, changes permissions or touches tickets. Whether it may act at all
  is the profile's autonomy level for ``zendesk_create_draft``
  (``agent/autonomy.py``).

**Edits to a published article** always become a separate draft. The classic
API can't stage one (``draft`` is create-only; a PUT on a published
translation goes live, and ``draft: true`` on it unpublishes it), and Guide
Team Publishing's work-in-progress state has no public API (Zendesk staff,
community post 6482703860506, checked 2026-11-16). The result says
``reconcile_required`` and names the live article for a human.

Credentials come ONLY from ``agent.secret_scope.get_secret`` (the multiplex
gateway's per-profile scope): ``ZENDESK_SUBDOMAIN``, ``ZENDESK_EMAIL``,
``ZENDESK_API_TOKEN``. An ``UnscopedSecretError`` means "not configured".

Design: docs/superpowers/specs/2026-11-16-product-education-teammate-design.md
("Dependencies → 2. Zendesk tool"). Plan:
docs/superpowers/plans/2026-11-16-zendesk-toolset.md.
"""
from __future__ import annotations

import base64
import json
import logging
import re
from urllib.parse import urlsplit

from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 30
_SUBDOMAIN = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")

# ── allowlists ───────────────────────────────────────────────────────────
_LOC = r"[a-z]{2}(?:-[a-z0-9]{2,4})?"
_HC = rf"api/v2/help_center/(?:{_LOC}/)?"
_ID = r"\d+"
_J = r"(?:\.json)?"

_READ_ALLOWLIST: tuple["re.Pattern[str]", ...] = tuple(re.compile(p) for p in (
    rf"^{_HC}categories(?:/{_ID})?{_J}$",
    rf"^{_HC}categories/{_ID}/(?:sections|articles){_J}$",
    rf"^{_HC}sections(?:/{_ID})?{_J}$",
    rf"^{_HC}sections/{_ID}/articles{_J}$",
    rf"^{_HC}articles(?:/{_ID})?{_J}$",
    rf"^{_HC}articles/{_ID}/translations(?:/{_LOC})?{_J}$",
    rf"^{_HC}articles/{_ID}/attachments{_J}$",
    rf"^{_HC}articles/search{_J}$",
    rf"^api/v2/help_center/user_segments(?:/{_ID})?{_J}$",
    rf"^api/v2/help_center/locales{_J}$",
    rf"^api/v2/guide/permission_groups(?:/{_ID})?{_J}$",
))

# The ONLY write: create an article in a section. ``_zendesk_write`` also
# requires the body to be an unpublished draft that notifies nobody.
_WRITE_ALLOWLIST: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    ("POST", re.compile(rf"^api/v2/help_center/(?:{_LOC}/)?sections/{_ID}/articles\.json$")),
)


class ZendeskWriteRefused(RuntimeError):
    """A write outside the allowlist or not a forced draft — never sent."""


def _read_allowed(path: str) -> bool:
    return any(p.match(path) for p in _READ_ALLOWLIST)


def _write_allowed(method: str, path: str) -> bool:
    return any(m == method and p.match(path) for m, p in _WRITE_ALLOWLIST)


def _normalize_read_path(raw: str, base: str) -> str | None:
    """Return the request path (``api/v2/...`` plus any query) or None.

    Accepts a bare path or a full URL on exactly ``base``'s host (so a
    ``links.next`` pagination URL can be passed back). Everything else —
    another host, traversal, a path outside the Help Center — is refused.
    """
    raw = (raw or "").strip()
    if not raw or any(ch.isspace() for ch in raw) or "#" in raw or "\\" in raw:
        return None
    if "://" in raw:
        parts, own = urlsplit(raw), urlsplit(base)
        if (parts.scheme, parts.netloc) != (own.scheme, own.netloc) or not own.netloc:
            return None
        raw = parts.path + (f"?{parts.query}" if parts.query else "")
    raw = raw.lstrip("/")
    path, _, _query = raw.partition("?")
    if ".." in path or "//" in path or "%" in path:
        return None
    return raw if _read_allowed(path) else None


# ── credentials + HTTP ───────────────────────────────────────────────────

def _zendesk_creds() -> tuple[str, str, str]:
    """``(base_url, email, token)`` from the active profile's secret scope.

    ``("", "", "")`` when any is missing, the subdomain isn't a bare
    subdomain, or the read is unscoped under multiplex.
    """
    from agent.secret_scope import UnscopedSecretError, get_secret
    try:
        sub = (get_secret("ZENDESK_SUBDOMAIN") or "").strip().lower()
        email = (get_secret("ZENDESK_EMAIL") or "").strip()
        token = (get_secret("ZENDESK_API_TOKEN") or "").strip()
    except UnscopedSecretError:
        return "", "", ""
    if not (sub and email and token) or not _SUBDOMAIN.match(sub) or sub.endswith("-"):
        return "", "", ""
    return f"https://{sub}.zendesk.com", email, token


def _check_zendesk() -> bool:
    return all(_zendesk_creds())


_NOT_CONFIGURED = "Zendesk not configured (set ZENDESK_SUBDOMAIN, ZENDESK_EMAIL, ZENDESK_API_TOKEN)."


async def _zendesk_request(
    base: str, email: str, token: str, path: str,
    method: str = "GET", json_body: dict | None = None,
) -> dict:
    """Call ``base/path`` with Zendesk API-token basic auth. Returns {status, text}."""
    import aiohttp
    from gateway.platforms.base import resolve_proxy_url, proxy_kwargs_for_aiohttp

    url = f"{base}/{path.lstrip('/')}"
    cred = base64.b64encode(f"{email}/token:{token}".encode()).decode()
    headers = {"Authorization": f"Basic {cred}", "Accept": "application/json"}
    _sess_kw, _req_kw = proxy_kwargs_for_aiohttp(resolve_proxy_url())
    timeout = aiohttp.ClientTimeout(total=_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout, **_sess_kw) as session:
        async with session.request(method, url, headers=headers, json=json_body,
                                   allow_redirects=False, **_req_kw) as resp:
            return {"status": resp.status, "text": await resp.text()}


def _parse(resp: dict) -> tuple[bool, object]:
    """``(ok, data_or_error)`` for a {status, text} response."""
    status, text = resp["status"], (resp.get("text") or "").strip()
    if status >= 300:
        return False, f"Zendesk API error (HTTP {status}): {text[:300]}"
    if not text:
        return True, None
    try:
        return True, json.loads(text)
    except ValueError:
        return True, text


async def _zendesk_get(path: str) -> tuple[bool, object]:
    base, email, token = _zendesk_creds()
    if not (base and email and token):
        return False, _NOT_CONFIGURED
    try:
        resp = await _zendesk_request(base, email, token, path)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("[zendesk] GET %s failed: %s", path, e)
        return False, f"Zendesk request failed: {e}"
    return _parse(resp)


async def _zendesk_write(method: str, path: str, body: dict) -> dict:
    """The single write path. Raises ``ZendeskWriteRefused`` before any request
    unless (method, path) is allowlisted AND the body is an unpublished draft
    that notifies no subscribers."""
    if not _write_allowed(method, path):
        raise ZendeskWriteRefused(f"{method} {path} is not an allowed Zendesk write.")
    article = body.get("article") if isinstance(body, dict) else None
    if not isinstance(article, dict) or article.get("draft") is not True:
        raise ZendeskWriteRefused("Zendesk writes must create an unpublished draft (draft: true).")
    if body.get("notify_subscribers") is not False:
        raise ZendeskWriteRefused("Zendesk draft creation must set notify_subscribers: false.")
    base, email, token = _zendesk_creds()
    if not (base and email and token):
        raise ZendeskWriteRefused(_NOT_CONFIGURED)
    return await _zendesk_request(base, email, token, path, method=method, json_body=body)


# ── zendesk_api (read) ───────────────────────────────────────────────────

ZENDESK_API_SCHEMA = {
    "name": "zendesk_api",
    "description": (
        "Read the Zendesk Help Center (READ-ONLY, GET) and get parsed JSON back. "
        "Pass a path WITHOUT a leading slash, e.g. 'api/v2/help_center/en-us/categories.json', "
        "'api/v2/help_center/sections/<id>/articles.json?page[size]=100', "
        "'api/v2/help_center/articles/<id>/translations/en-us.json', "
        "'api/v2/help_center/articles/search.json?query=points' (search sees published only; "
        "the section/category lists include drafts), 'api/v2/guide/permission_groups.json', "
        "'api/v2/help_center/user_segments.json'. A pagination 'links.next' URL can be passed as-is. "
        "Help Center only — tickets and users are not reachable. Use this for ALL Zendesk reads, "
        "never curl or a browser."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string",
                     "description": "Help Center API path (api/v2/help_center/... or api/v2/guide/...) or a links.next URL."},
            "method": {"type": "string", "enum": ["GET"], "default": "GET",
                       "description": "Only GET is supported."},
        },
        "required": ["path"],
    },
}


async def _zendesk_api_handler(args: dict, **_kw) -> str:
    method = (args.get("method") or "GET").strip().upper()
    if method != "GET":
        return tool_error(f"Unsupported method '{method}'. zendesk_api is read-only (GET).")
    base, _email, _token = _zendesk_creds()
    if not base:
        return tool_error(_NOT_CONFIGURED)
    path = _normalize_read_path(args.get("path") or "", base)
    if path is None:
        return tool_error(
            "path is not an allowed Zendesk read. Allowed: Help Center categories, sections, "
            "articles, translations, attachments, search, user_segments, locales, and "
            "api/v2/guide/permission_groups — no tickets, users or other APIs."
        )
    ok, data = await _zendesk_get(path)
    if not ok:
        return tool_error(str(data))
    if isinstance(data, str):
        return tool_result({"ok": True, "text": data})
    return tool_result({"ok": True, "data": data})


registry.register(
    name="zendesk_api",
    toolset="zendesk",
    schema=ZENDESK_API_SCHEMA,
    handler=lambda args, **kw: _zendesk_api_handler(args, **kw),
    check_fn=_check_zendesk,
    requires_env=[],
    is_async=True,
    emoji="📚",
    max_result_size_chars=24000,
)


# ── zendesk_create_draft (write, autonomy-gated) ─────────────────────────

_ACTION = "zendesk_create_draft"
_LOCALE_RE = re.compile(rf"^{_LOC}$")
_DIGITS = re.compile(r"^\d{1,20}$")


def _config() -> dict:
    """The active profile's raw config.yaml (mtime-cached); indirection for tests."""
    from hermes_cli.config import read_raw_config
    return read_raw_config()


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)


def _audit(action: str, target: str) -> None:
    from agent.data_access_audit import record_access
    record_access(tool=_ACTION, action=action, target=target)


def _zendesk_settings(cfg: dict) -> dict:
    z = cfg.get("zendesk") if isinstance(cfg.get("zendesk"), dict) else {}

    def _names(key: str, default: list[str]) -> frozenset[str]:
        raw = z.get(key, default)
        if isinstance(raw, str):
            raw = [raw]
        return frozenset(str(x).strip() for x in (raw or []) if str(x).strip())

    return {
        "data_source": str(z.get("release_assets_data_source_id") or "").strip(),
        # `Published` can never count as sign-off: it means already live.
        "approved": _names("approved_statuses", ["Approved"]) - {"Published"},
        "hold": _names("hold_statuses", ["Hold"]),
    }


def _int_arg(value, name: str, *, nullable: bool = False):
    if value is None and nullable:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer id.")
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str) and _DIGITS.match(value.strip()):
        return int(value.strip())
    raise ValueError(f"{name} must be an integer id.")


def _build_request(args: dict) -> dict:
    """Validate the model's arguments into the stored request (ValueError on bad input)."""
    title = (args.get("title") or "").strip() if isinstance(args.get("title"), str) else ""
    body = args.get("body") if isinstance(args.get("body"), str) else ""
    if not title or len(title) > 255:
        raise ValueError("title is required (max 255 characters).")
    if not body.strip():
        raise ValueError("body (article HTML) is required.")
    locale = str(args.get("locale") or "en-us").strip().lower()
    if not _LOCALE_RE.match(locale):
        raise ValueError(f"locale {locale!r} is not a Zendesk locale like 'en-us'.")
    section = args.get("section_id")
    section_id = None if section in (None, "") else _int_arg(section, "section_id")
    revision = args.get("revision_of_article_id")
    revision_id = None if revision in (None, "") else _int_arg(revision, "revision_of_article_id")
    if section_id is None and revision_id is None:
        raise ValueError("section_id is required (or revision_of_article_id to use its section).")
    return {
        "section_id": section_id,
        "title": title,
        "body": body,
        "locale": locale,
        "permission_group_id": _int_arg(args.get("permission_group_id"), "permission_group_id"),
        "user_segment_id": _int_arg(args.get("user_segment_id"), "user_segment_id", nullable=True),
        "revision_of_article_id": revision_id,
        "release_asset_row_id": str(args.get("release_asset_row_id") or "").strip(),
        "reconcile_with": None,
    }


async def _resolve_revision(req: dict) -> None:
    """For an edit to an existing article: confirm it exists, record it for the
    human to reconcile, and default the section to its section. Never writes it."""
    if req["revision_of_article_id"] is None:
        return
    ok, data = await _zendesk_get(f"api/v2/help_center/articles/{req['revision_of_article_id']}.json")
    if not ok:
        raise ValueError(f"revision_of_article_id: {data}")
    art = (data or {}).get("article") if isinstance(data, dict) else None
    if not isinstance(art, dict):
        raise ValueError("revision_of_article_id: Zendesk returned no article.")
    req["reconcile_with"] = {"id": art.get("id"), "title": art.get("title"),
                             "html_url": art.get("html_url")}
    if req["section_id"] is None:
        req["section_id"] = _int_arg(art.get("section_id"), "section_id")


async def _check_row(req: dict, cfg: dict, *, require_approved: bool) -> "object":
    """Fetch the Release Assets row fresh and check it may drive this draft.

    Raises ``gate.GateError`` with the reason (``gate.RowStopped`` for a
    Published or held row). Returns the ``RowState``.
    """
    from tools import release_asset_gate as gate
    settings = _zendesk_settings(cfg)
    base, _e, _t = _zendesk_creds()
    host = urlsplit(base).netloc
    page = await gate.fetch_notion_page(req["release_asset_row_id"])
    row = gate.row_state(page, settings["data_source"])
    if row.asset_type != "Help article":
        raise gate.GateError(f"The row's Asset type is {row.asset_type!r}, not 'Help article'.")
    if row.destination != host:
        raise gate.GateError(f"The row's Destination is {row.destination!r}, not {host!r}.")
    if row.status == "Published":
        raise gate.RowStopped("The row is already Published — never re-push a live article.")
    if row.status in settings["hold"]:
        raise gate.RowStopped(f"The row is on {row.status!r} — a product-team member held it.")
    if require_approved and row.status not in settings["approved"]:
        raise gate.GateError(
            f"Sign-off required: the row's Status is {row.status or 'empty'!r}; it must be "
            f"{' / '.join(sorted(settings['approved']))} (set by a product-team member).")
    return row


async def _create(req: dict) -> dict:
    """POST the draft. Returns the tool's success payload (raises on failure)."""
    path = f"api/v2/help_center/{req['locale']}/sections/{req['section_id']}/articles.json"
    body = {
        "article": {
            "title": req["title"],
            "locale": req["locale"],
            "body": req["body"],
            "draft": True,
            "permission_group_id": req["permission_group_id"],
            "user_segment_id": req["user_segment_id"],
        },
        "notify_subscribers": False,
    }
    ok, data = _parse(await _zendesk_write("POST", path, body))
    if not ok:
        raise ZendeskWriteRefused(str(data))
    art = (data or {}).get("article") if isinstance(data, dict) else None
    if not isinstance(art, dict):
        raise ZendeskWriteRefused("Zendesk returned no article for the create.")
    base, _e, _t = _zendesk_creds()
    return {
        "ok": True,
        "article_id": art.get("id"),
        "html_url": art.get("html_url"),
        "edit_url": f"{base}/knowledge/articles/{art.get('id')}/{req['locale']}",
        "draft": art.get("draft", True),
        "section_id": req["section_id"],
        "reconcile_required": req["reconcile_with"] is not None,
        "reconcile_with": req["reconcile_with"],
        "note": (
            "Unpublished draft created; a human reviews and publishes it in Zendesk."
            + (" This is a SEPARATE draft revising a live article (Team Publishing staging "
               "has no API): ask the team how to reconcile it with the live one."
               if req["reconcile_with"] else "")),
    }


def _describe(req: dict, row_url: str = "") -> str:
    what = f"*{req['title']}* (section {req['section_id']}, {req['locale']})"
    if req["reconcile_with"]:
        live = req["reconcile_with"]
        what += f", a separate draft revising <{live.get('html_url')}|{live.get('title')}>"
    if row_url:
        what += f", for <{row_url}|this Release Assets row>"
    return what


async def _act(req: dict, level: str, *, pending_id: str = "", row_url: str = "") -> dict:
    result = await _create(req)
    result["level"] = level
    _audit(f"autonomy_{level}",
           f"{result.get('html_url')} row={req['release_asset_row_id'] or '-'}"
           + (f" pending={pending_id}" if pending_id else ""))
    return result


async def _queue_veto(req: dict, cfg: dict, row_url: str) -> str:
    from zoneinfo import ZoneInfo
    from agent import autonomy
    from tools import release_asset_gate as gate
    channel, wait, wh = autonomy.veto_settings(cfg)
    if not channel:
        return tool_error("autonomy.announce_channel is not configured — a veto-level action "
                          "must be announced before it can wait out its veto window.")
    not_before = autonomy.add_working_hours(_now(), wait, wh)
    pending_id = autonomy.save_pending(_ACTION, {
        "status": "announcing", "request": req, "created_at": _now().isoformat(),
        "not_before": not_before.isoformat(), "announce_channel": channel,
    })
    text = (f":hourglass_flowing_sand: I plan to create an *unpublished* Zendesk draft "
            f"{_describe(req, row_url)}. It goes ahead after "
            f"{not_before.astimezone(ZoneInfo(wh.timezone)):%a %d %b %H:%M} "
            f"({wh.timezone}) unless someone reacts :x: here or sets the row to Hold. "
            f"(autonomy: veto · ref {pending_id})")
    try:
        ts = await gate.post_announcement(channel, text)
    except gate.GateError as e:
        autonomy.update_pending(_ACTION, pending_id, status="failed", error=str(e))
        return tool_error(f"Could not announce the veto window, so nothing was queued: {e}")
    autonomy.update_pending(_ACTION, pending_id, status="pending", announcement_ts=ts)
    _audit("autonomy_veto_announced", f"pending={pending_id} row={req['release_asset_row_id']}")
    return tool_result({
        "status": "pending", "pending_id": pending_id, "not_before": not_before.isoformat(),
        "announcement_ts": ts,
        "message": ("Announced. Call zendesk_create_draft again with this pending_id after "
                    "not_before; it re-checks for a veto (❌ or Hold) and then creates the draft."),
    })


async def _execute_pending(pending_id: str, level: str, cfg: dict) -> str:
    from datetime import datetime
    from agent import autonomy
    from tools import release_asset_gate as gate
    rec = autonomy.load_pending(_ACTION, pending_id)
    if rec is None:
        return tool_error(f"No pending zendesk_create_draft action {pending_id!r}.")
    if rec.get("status") != "pending":
        return tool_error(f"Pending action {pending_id} is {rec.get('status')!r}, not pending.")
    req = rec["request"]
    channel, ts = rec.get("announce_channel", ""), rec.get("announcement_ts", "")

    if level == "veto" and _now() < datetime.fromisoformat(rec["not_before"]):
        return tool_error(f"The veto window is open until {rec['not_before']}; try again after.")

    async def _veto(reason: str) -> str:
        autonomy.update_pending(_ACTION, pending_id, status="vetoed", reason=reason)
        _audit("autonomy_vetoed", f"pending={pending_id} {reason}")
        try:
            await gate.post_announcement(channel, f":no_entry_sign: Not creating it: {reason}.", ts)
        except gate.GateError:
            pass
        return tool_result({"status": "vetoed", "pending_id": pending_id, "reason": reason})

    try:
        vetoers = await gate.veto_reactions(channel, ts)
    except gate.GateError as e:
        return tool_error(f"Could not check the announcement for a veto, so nothing was created: {e}")
    if vetoers:
        return await _veto("vetoed with :x: by " + ", ".join(f"<@{u}>" for u in vetoers))
    try:
        row = await _check_row(req, cfg, require_approved=(level == "signoff"))
    except gate.RowStopped as e:
        return await _veto(str(e).rstrip("."))
    except gate.GateError as e:
        return tool_error(str(e))
    try:
        result = await _act(req, level, pending_id=pending_id, row_url=row.url)
    except (ZendeskWriteRefused, ValueError) as e:
        return tool_error(str(e))
    autonomy.update_pending(_ACTION, pending_id, status="done", article_id=result.get("article_id"),
                            html_url=result.get("html_url"))
    try:
        await gate.post_announcement(
            channel, f":white_check_mark: Created the unpublished draft: {result.get('html_url')}", ts)
    except gate.GateError as e:
        result["announce_error"] = str(e)
    return tool_result(result)


ZENDESK_CREATE_DRAFT_SCHEMA = {
    "name": "zendesk_create_draft",
    "description": (
        "Create ONE UNPUBLISHED draft article in the Zendesk Help Center. It never publishes, "
        "edits a live article, archives or deletes — a human publishes in Zendesk. Look up "
        "section_id, permission_group_id and user_segment_id with zendesk_api first. Pass the "
        "Release Assets row id (release_asset_row_id) of the Approved help-article row. For a "
        "change to an existing published article pass revision_of_article_id: a SEPARATE draft "
        "is created and you must ask the team how to reconcile it. Whether it may act is the "
        "profile's autonomy level: 'signoff' needs the row Approved; 'veto' announces in the home "
        "channel and returns a pending_id to call back with after the veto window; 'auto' acts "
        "and reports. list_pending=true lists waiting veto actions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "section_id": {"type": "string", "description": "Target Help Center section id."},
            "title": {"type": "string", "description": "Article title."},
            "body": {"type": "string", "description": "Article body as Zendesk HTML."},
            "permission_group_id": {"type": "integer",
                                    "description": "Guide permission group id (api/v2/guide/permission_groups.json)."},
            "user_segment_id": {"type": "integer",
                                "description": "User segment id; omit for visible-to-everyone (null)."},
            "locale": {"type": "string", "default": "en-us", "description": "Article locale."},
            "release_asset_row_id": {"type": "string",
                                     "description": "Notion Release Assets row (page) id the article comes from."},
            "revision_of_article_id": {"type": "string",
                                       "description": "Existing article this draft revises (creates a separate draft)."},
            "pending_id": {"type": "string",
                           "description": "Run a veto-level action announced earlier (other arguments are ignored)."},
            "list_pending": {"type": "boolean", "description": "List veto actions still waiting."},
        },
        "required": [],
    },
}


async def _zendesk_create_draft_handler(args: dict, **_kw) -> str:
    from agent import autonomy
    from tools import release_asset_gate as gate

    if args.get("list_pending") is True:
        waiting = [{"pending_id": r.get("pending_id"), "title": (r.get("request") or {}).get("title"),
                    "not_before": r.get("not_before"),
                    "release_asset_row_id": (r.get("request") or {}).get("release_asset_row_id")}
                   for r in autonomy.list_pending(_ACTION) if r.get("status") == "pending"]
        return tool_result({"ok": True, "pending": waiting})

    cfg = _config()
    try:
        level = autonomy.resolve_level(_ACTION, cfg)
    except autonomy.AutonomyConfigError as e:
        return tool_error(str(e))
    if level == "off":
        return tool_error("zendesk_create_draft is turned off for this profile (autonomy level 'off').")
    if not _check_zendesk():
        return tool_error(_NOT_CONFIGURED)

    pending_id = str(args.get("pending_id") or "").strip()
    if pending_id:
        return await _execute_pending(pending_id, level, cfg)

    try:
        req = _build_request(args)
        await _resolve_revision(req)
    except ValueError as e:
        return tool_error(str(e))

    row_url = ""
    try:
        if level in ("signoff", "veto") and not req["release_asset_row_id"]:
            return tool_error(f"release_asset_row_id is required at autonomy level {level!r}: "
                              "pass the Release Assets row this article comes from.")
        if req["release_asset_row_id"]:
            row = await _check_row(req, cfg, require_approved=(level == "signoff"))
            row_url = row.url
            if level == "veto" and row.status not in _zendesk_settings(cfg)["approved"]:
                return await _queue_veto(req, cfg, row_url)
    except gate.GateError as e:
        return tool_error(str(e))
    except autonomy.AutonomyConfigError as e:
        return tool_error(str(e))

    try:
        result = await _act(req, level, row_url=row_url)
    except (ZendeskWriteRefused, ValueError) as e:
        return tool_error(str(e))
    if level == "auto":
        channel, _w, _wh = autonomy.veto_settings(cfg)
        if channel:
            try:
                await gate.post_announcement(
                    channel, f":memo: Created an unpublished Zendesk draft {_describe(req, row_url)}: "
                             f"{result.get('html_url')} (autonomy: auto)")
            except gate.GateError as e:
                result["announce_error"] = str(e)
        else:
            result["announce_error"] = "autonomy.announce_channel is not configured"
    return tool_result(result)


registry.register(
    name="zendesk_create_draft",
    toolset="zendesk_write",
    schema=ZENDESK_CREATE_DRAFT_SCHEMA,
    handler=lambda args, **kw: _zendesk_create_draft_handler(args, **kw),
    check_fn=_check_zendesk,
    requires_env=[],
    is_async=True,
    emoji="📝",
    max_result_size_chars=4000,
)
