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
  false``. ``hc:write`` covers publish and delete too (and the service user
  can publish), so that chokepoint IS the boundary — it never publishes, archives,
  deletes, changes permissions or touches tickets. Whether it may act at all
  is the profile's autonomy level for ``zendesk_create_draft``
  (``agent/autonomy.py``).

**Edits to a published article** always become a separate draft. The classic
API can't stage one (``draft`` is create-only; a PUT on a published
translation goes live, and ``draft: true`` on it unpublishes it), and Guide
Team Publishing's work-in-progress state has no public API (Zendesk staff,
community post 6482703860506, checked 2026-11-16). The result says
``reconcile_required`` and names the live article for a human.

Auth is OAuth client credentials ONLY (Zendesk is retiring API tokens):
``ZENDESK_SUBDOMAIN``, ``ZENDESK_OAUTH_CLIENT_ID``, ``ZENDESK_OAUTH_CLIENT_SECRET``,
read ONLY from ``agent.secret_scope.get_secret`` (the multiplex gateway's
per-profile scope); an ``UnscopedSecretError`` means "not configured". Reads
use an ``hc:read`` token, the draft create an ``hc:read hc:write`` token,
cached per (profile, subdomain, client, scope).

Design: docs/superpowers/specs/2026-11-16-product-education-teammate-design.md
("Dependencies → 2. Zendesk tool"). Plan:
docs/superpowers/plans/2026-11-16-zendesk-toolset.md.
"""
from __future__ import annotations

import json
import logging
import re
from urllib.parse import parse_qsl, urlsplit

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


# Pagination, search and filter keys only. `include` (side-loading users and
# other records) and everything else is refused.
_READ_QUERY_KEYS = frozenset({
    "page[size]", "page[after]", "page[before]", "per_page", "page", "query",
    "sort_by", "sort_order", "locale", "label_names", "category", "section",
})


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
    path, _, query = raw.partition("?")
    if ".." in path or "//" in path or "%" in path:
        return None
    if query:
        if ";" in query:  # Rack-style servers also split on ';' — no smuggled keys
            return None
        try:
            keys = {k for k, _v in parse_qsl(query, keep_blank_values=True, strict_parsing=True)}
        except ValueError:
            return None
        if not keys <= _READ_QUERY_KEYS:
            return None
    return raw if _read_allowed(path) else None


# ── credentials, OAuth token, HTTP ───────────────────────────────────────
#
# Zendesk is retiring API tokens (Phase 2, 2026-10-27: none can be created;
# Phase 3, 2027-04-30: all deactivated), so this uses OAuth client
# credentials ONLY — a confidential client created in Admin Center. There is
# no API-token fallback: ZENDESK_EMAIL / ZENDESK_API_TOKEN are never read.
# The token acts as the user who created the client, so that user's Guide
# permissions are the real ceiling (see CLAUDE.md for the setup).
#
# Least privilege per call: reads get an `hc:read` token; only the draft
# create gets `hc:read hc:write`. No global `read`/`write`, no ticket scopes.

_READ_SCOPE = "hc:read"
_WRITE_SCOPE = "hc:read hc:write"
_TOKEN_TTL_SECONDS = 1800       # requested; Zendesk allows 300..172800
_TOKEN_SKEW_SECONDS = 120       # refresh this long before it expires
# (profile home, base url, client id, scope) -> (access token, monotonic expiry)
_TOKEN_CACHE: dict[tuple[str, str, str, str], tuple[str, float]] = {}


class ZendeskAuthError(RuntimeError):
    """The OAuth token request failed (message carries no secret or token)."""


def _monotonic() -> float:
    import time
    return time.monotonic()


def _zendesk_creds() -> tuple[str, str, str]:
    """``(base_url, client_id, client_secret)`` from the active profile's secret scope.

    ``("", "", "")`` when any is missing, the subdomain isn't a bare
    subdomain, or the read is unscoped under multiplex.
    """
    from agent.secret_scope import UnscopedSecretError, get_secret
    try:
        sub = (get_secret("ZENDESK_SUBDOMAIN") or "").strip().lower()
        client_id = (get_secret("ZENDESK_OAUTH_CLIENT_ID") or "").strip()
        secret = (get_secret("ZENDESK_OAUTH_CLIENT_SECRET") or "").strip()
    except UnscopedSecretError:
        return "", "", ""
    if not (sub and client_id and secret) or not _SUBDOMAIN.match(sub) or sub.endswith("-"):
        return "", "", ""
    return f"https://{sub}.zendesk.com", client_id, secret


def _check_zendesk() -> bool:
    return all(_zendesk_creds())


_NOT_CONFIGURED = ("Zendesk not configured (set ZENDESK_SUBDOMAIN, ZENDESK_OAUTH_CLIENT_ID, "
                   "ZENDESK_OAUTH_CLIENT_SECRET).")


def _scrub(text: str, *secrets_: str) -> str:
    """Remove any credential material from a message before it leaves."""
    out = str(text)
    for value in secrets_:
        if value:
            out = out.replace(value, "[redacted]")
    for _key, (token, _exp) in list(_TOKEN_CACHE.items()):
        out = out.replace(token, "[redacted]")
    return out


def _cache_key(base: str, client_id: str, scope: str) -> tuple[str, str, str, str]:
    from hermes_constants import get_hermes_home
    return (str(get_hermes_home()), base, client_id, scope)


async def _token_request(base: str, payload: dict) -> dict:
    """POST the client-credentials grant to the fixed ``base/oauth/tokens``."""
    import aiohttp
    from gateway.platforms.base import resolve_proxy_url, proxy_kwargs_for_aiohttp

    _sess_kw, _req_kw = proxy_kwargs_for_aiohttp(resolve_proxy_url())
    timeout = aiohttp.ClientTimeout(total=_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout, **_sess_kw) as session:
        async with session.request("POST", f"{base}/oauth/tokens", json=payload,
                                   headers={"Accept": "application/json"},
                                   allow_redirects=False, **_req_kw) as resp:
            return {"status": resp.status, "text": await resp.text()}


async def _access_token(scope: str, *, force: bool = False) -> str:
    """A cached OAuth access token for ``scope``. Raises ZendeskAuthError."""
    base, client_id, secret = _zendesk_creds()
    if not (base and client_id and secret):
        raise ZendeskAuthError(_NOT_CONFIGURED)
    key = _cache_key(base, client_id, scope)
    cached = _TOKEN_CACHE.get(key)
    if cached and not force and _monotonic() < cached[1]:
        return cached[0]
    _TOKEN_CACHE.pop(key, None)
    payload = {"grant_type": "client_credentials", "client_id": client_id,
               "client_secret": secret, "scope": scope, "expires_in": _TOKEN_TTL_SECONDS}
    try:
        resp = await _token_request(base, payload)
    except Exception as e:
        raise ZendeskAuthError(_scrub(f"Zendesk OAuth token request failed: {type(e).__name__}",
                                      secret, client_id)) from None
    try:
        data = json.loads(resp.get("text") or "{}")
    except ValueError:
        data = {}
    data = data if isinstance(data, dict) else {}
    token = data.get("access_token")
    if resp["status"] >= 300 or not isinstance(token, str) or not token:
        code = str(data.get("error") or "no access_token")[:60]
        raise ZendeskAuthError(_scrub(
            f"Zendesk OAuth token request failed (HTTP {resp['status']}: {code}). Check the "
            "OAuth client (confidential, client_credentials allowed, scopes hc:read hc:write).",
            secret, client_id))
    ttl = data.get("expires_in")
    ttl = ttl if isinstance(ttl, int) and ttl > 0 else _TOKEN_TTL_SECONDS
    _TOKEN_CACHE[key] = (token, _monotonic() + max(ttl - _TOKEN_SKEW_SECONDS, 0))
    return token


def _drop_token(scope: str) -> None:
    base, client_id, _secret = _zendesk_creds()
    _TOKEN_CACHE.pop(_cache_key(base, client_id, scope), None)


async def _zendesk_request(
    base: str, token: str, path: str,
    method: str = "GET", json_body: dict | None = None,
) -> dict:
    """Call ``base/path`` with a Bearer token. Returns {status, text}."""
    import aiohttp
    from gateway.platforms.base import resolve_proxy_url, proxy_kwargs_for_aiohttp

    url = f"{base}/{path.lstrip('/')}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    _sess_kw, _req_kw = proxy_kwargs_for_aiohttp(resolve_proxy_url())
    timeout = aiohttp.ClientTimeout(total=_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout, **_sess_kw) as session:
        async with session.request(method, url, headers=headers, json=json_body,
                                   allow_redirects=False, **_req_kw) as resp:
            return {"status": resp.status, "text": await resp.text()}


async def _authed_request(path: str, *, scope: str, method: str = "GET",
                          json_body: dict | None = None) -> dict:
    """One API call with a cached token; on 401 drop it and retry ONCE.

    Raises ZendeskAuthError (no token obtainable) or RuntimeError for a
    transport failure — both with credential material scrubbed.
    """
    base, client_id, secret = _zendesk_creds()
    token = await _access_token(scope)
    try:
        resp = await _zendesk_request(base, token, path, method=method, json_body=json_body)
        if resp["status"] == 401:
            _drop_token(scope)
            token = await _access_token(scope, force=True)
            resp = await _zendesk_request(base, token, path, method=method, json_body=json_body)
    except ZendeskAuthError:
        raise
    except Exception as e:
        logger.warning("[zendesk] %s %s failed: %s", method, path, type(e).__name__)
        raise RuntimeError(f"Zendesk request failed ({type(e).__name__}).") from None
    resp["text"] = _scrub(resp.get("text") or "", token, secret, client_id)
    return resp


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
    if not _check_zendesk():
        return False, _NOT_CONFIGURED
    try:
        resp = await _authed_request(path, scope=_READ_SCOPE)
    except (ZendeskAuthError, RuntimeError) as e:
        return False, str(e)
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
    if not _check_zendesk():
        raise ZendeskWriteRefused(_NOT_CONFIGURED)
    try:
        return await _authed_request(path, scope=_WRITE_SCOPE, method=method, json_body=body)
    except (ZendeskAuthError, RuntimeError) as e:
        raise ZendeskWriteRefused(str(e)) from None


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
    base, _client_id, _secret = _zendesk_creds()
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
#
# The content comes FROM THE ROW, never from the model: the title is the
# Release Assets row's title property and the body is the row's own page
# content rendered by tools/notion_blocks_html.py (safe by construction).
# The model only chooses placement (section, locale, the article it revises).
# A row gets at most one draft: an O_EXCL lock per row plus a ledger under
# $HERMES_HOME/autonomy/ (a protected path) record row → article, and a second
# call returns the existing draft instead of creating another.

_ACTION = "zendesk_create_draft"
_LOCALE_RE = re.compile(rf"^{_LOC}$")
_DIGITS = re.compile(r"^\d{1,20}$")

_WRITE_GUARD_MISSING = (
    "zendesk_create_draft is refused: this profile has no enforcing Notion write guard "
    "(notion.write_guard.enabled: true, with agent_write_guard on). Without it the agent could set a Release Assets "
    "row to Approved itself through notion_api, so the row is not a human signal. An admin "
    "must deploy and enable the guard first.")


class _ContentChanged(RuntimeError):
    """The row's content differs from what the veto announcement showed."""


def _title_of(result: dict) -> str:
    return str(result.get("title") or "the draft")


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


def _write_guard_enabled(cfg: dict) -> bool:
    """True iff the Notion write guard is enforcing for this profile (the
    guard's own predicate, so "enabled" can never mean "off" — an enabled but
    invalid block enforces by denying every write) and agent_write_guard
    keeps the agent out of its own config.yaml."""
    from tools.notion_write_guard import autonomy_precondition
    return autonomy_precondition(cfg) is None


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
        "default_permission_group_id": z.get("default_permission_group_id"),
        "default_user_segment_id": z.get("default_user_segment_id"),
    }


def _row_key(row_id) -> str:
    hexid = str(row_id or "").replace("-", "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{32}", hexid):
        raise ValueError("release_asset_row_id is required: the Notion id of the Release "
                         "Assets row (help article) to draft from.")
    return hexid


def _build_spec(args: dict) -> dict:
    """Validate the model's PLACEMENT arguments. Content is never taken from args."""
    locale = str(args.get("locale") or "en-us").strip().lower()
    if not _LOCALE_RE.match(locale):
        raise ValueError(f"locale {locale!r} is not a Zendesk locale like 'en-us'.")
    section = args.get("section_id")
    revision = args.get("revision_of_article_id")
    spec = {
        "row_id": _row_key(args.get("release_asset_row_id")),
        "section_id": None if section in (None, "") else _int_arg(section, "section_id"),
        "locale": locale,
        "revision_of_article_id": None if revision in (None, "") else _int_arg(revision, "revision_of_article_id"),
        "omit_images": args.get("omit_images") is True,
    }
    if spec["section_id"] is None and spec["revision_of_article_id"] is None:
        raise ValueError("section_id is required (or revision_of_article_id to use its section).")
    return spec


async def _resolve_placement(spec: dict, cfg: dict) -> None:
    """Fill permission group, user segment and the article to reconcile with.

    Visibility is never the model's choice: it is copied from the article
    being revised, else taken from zendesk.default_permission_group_id /
    default_user_segment_id (unset segment = null = everyone).
    """
    base, _e, _t = _zendesk_creds()
    rid = spec["revision_of_article_id"]
    if rid is not None:
        ok, data = await _zendesk_get(f"api/v2/help_center/articles/{rid}.json")
        if not ok:
            raise ValueError(f"revision_of_article_id: {data}")
        art = (data or {}).get("article") if isinstance(data, dict) else None
        if not isinstance(art, dict):
            raise ValueError("revision_of_article_id: Zendesk returned no article.")
        art_locale = str(art.get("locale") or spec["locale"]).lower()
        if not _LOCALE_RE.match(art_locale):
            art_locale = spec["locale"]
        spec["reconcile_with"] = {"id": rid, "title": str(art.get("title") or ""),
                                  "html_url": f"{base}/hc/{art_locale}/articles/{rid}"}
        if spec["section_id"] is None:
            spec["section_id"] = _int_arg(art.get("section_id"), "section_id")
        spec["permission_group_id"] = _int_arg(art.get("permission_group_id"), "permission_group_id")
        spec["user_segment_id"] = _int_arg(art.get("user_segment_id"), "user_segment_id", nullable=True)
        return
    settings = _zendesk_settings(cfg)
    if settings["default_permission_group_id"] in (None, ""):
        raise ValueError("zendesk.default_permission_group_id is not configured — an admin must "
                         "set the permission group new drafts get.")
    spec["reconcile_with"] = None
    spec["permission_group_id"] = _int_arg(settings["default_permission_group_id"],
                                           "zendesk.default_permission_group_id")
    spec["user_segment_id"] = _int_arg(settings["default_user_segment_id"],
                                       "zendesk.default_user_segment_id", nullable=True)


def _check_row_fields(row, cfg: dict, *, require_approved: bool) -> None:
    from tools import release_asset_gate as gate
    settings = _zendesk_settings(cfg)
    host = urlsplit(_zendesk_creds()[0]).netloc
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


async def _materialize(spec: dict, cfg: dict, *, require_approved: bool) -> dict:
    """Read the row fresh and render the article it holds. Raises GateError."""
    import hashlib
    from tools import release_asset_gate as gate
    from tools.notion_blocks_html import render_blocks
    page = await gate.fetch_notion_page(spec["row_id"])
    row = gate.row_state(page, _zendesk_settings(cfg)["data_source"])
    _check_row_fields(row, cfg, require_approved=require_approved)
    rendered = render_blocks(await gate.fetch_block_tree(spec["row_id"]))
    if rendered.unsupported:
        raise gate.GateError(
            "The row's content has blocks a Zendesk article can't carry: "
            f"{', '.join(sorted(set(rendered.unsupported)))}. Ask the team to adjust the row.")
    if rendered.images and not spec["omit_images"]:
        raise gate.GateError(
            f"The row has {len(rendered.images)} image(s) "
            f"({'; '.join(rendered.images)}). Uploading screenshots to Zendesk isn't supported, "
            "and Notion image links expire. Either a human adds them in Zendesk after the draft "
            "exists — call again with omit_images=true and report the captions — or the draft "
            "is made by hand.")
    title = row.title
    if not title or len(title) > 255:
        raise gate.GateError("The row's title is empty or longer than 255 characters.")
    if not rendered.html.strip():
        raise gate.GateError("The row has no article content to draft.")
    sha = hashlib.sha256(f"{title}\0{rendered.html}".encode("utf-8")).hexdigest()
    return {"row": row, "title": title, "html": rendered.html, "sha": sha,
            "images_omitted": list(rendered.images)}


def _edit_url(article_id, locale: str) -> str:
    return f"{_zendesk_creds()[0]}/knowledge/articles/{int(article_id)}/{locale}"


def _existing_result(entry: dict) -> dict:
    return {"ok": True, "already_exists": True, **entry,
            "note": "A draft for this Release Assets row already exists; not creating another."}


async def _create_once(spec: dict, cfg: dict, level: str, *, require_approved: bool,
                       expected_sha: str | None = None, pending_id: str = "") -> dict:
    """Create the row's draft unless it already exists. Raises GateError,
    _ContentChanged, ZendeskWriteRefused or autonomy.AutonomyBusy."""
    from agent import autonomy
    key = spec["row_id"]
    with autonomy.exclusive(_ACTION, key):
        existing = autonomy.ledger_get(_ACTION, key)
        if existing:
            return _existing_result(existing)
        m = await _materialize(spec, cfg, require_approved=require_approved)
        if expected_sha and m["sha"] != expected_sha:
            raise _ContentChanged("The row's content changed after the veto announcement.")
        if pending_id:
            autonomy.update_pending(_ACTION, pending_id, status="executing")
        path = f"api/v2/help_center/{spec['locale']}/sections/{spec['section_id']}/articles.json"
        body = {
            "article": {
                "title": m["title"],
                "locale": spec["locale"],
                "body": m["html"],
                "draft": True,
                "permission_group_id": spec["permission_group_id"],
                "user_segment_id": spec["user_segment_id"],
            },
            "notify_subscribers": False,
        }
        resp = await _zendesk_write("POST", path, body)
        ok, data = _parse(resp)
        if not ok:
            raise ZendeskWriteRefused(str(data))
        # Record FIRST, before anything else can fail: the draft exists now.
        art = (data or {}).get("article") if isinstance(data, dict) else None
        art = art if isinstance(art, dict) else {}
        article_id = art.get("id")
        entry = {
            "article_id": article_id,
            "html_url": art.get("html_url"),
            "edit_url": _edit_url(article_id, spec["locale"]) if isinstance(article_id, int) else None,
            "title": m["title"],
            "section_id": spec["section_id"],
            "created_at": _now().isoformat(),
            "level": level,
            "pending_id": pending_id or None,
            "images_omitted": m["images_omitted"],
        }
        autonomy.ledger_put(_ACTION, key, entry)
    _audit(f"autonomy_{level}", f"article={article_id} row={key}"
           + (f" pending={pending_id}" if pending_id else ""))
    reconcile = spec.get("reconcile_with")
    return {
        "ok": True, **entry, "draft": True,
        "row_url": m["row"].url,
        "reconcile_required": reconcile is not None,
        "reconcile_with": reconcile,
        "note": (
            "Unpublished draft created from the row's own content; a human reviews and "
            "publishes it in Zendesk."
            + (f" {len(m['images_omitted'])} image(s) were left out — a human adds them in "
               "Zendesk (captions in images_omitted)." if m["images_omitted"] else "")
            + (" This is a SEPARATE draft revising a live article (Team Publishing staging "
               "has no API): ask the team how to reconcile it with the live one."
               if reconcile else "")),
    }


# Slack text: escape every interpolated value, link only to our own hosts.

def _slack_escape(text) -> str:
    return str(text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _slack_link(url, label) -> str:
    hosts = {urlsplit(_zendesk_creds()[0]).hostname, "www.notion.so", "notion.so"}
    try:
        parts = urlsplit(str(url or ""))
    except ValueError:
        return _slack_escape(label)
    if (parts.scheme == "https" and parts.hostname in hosts
            and not any(c in str(url) for c in "<>| \t\r\n")):
        return f"<{url}|{_slack_escape(label)}>"
    return _slack_escape(label)


def _describe(spec: dict, title: str, row_url: str = "", images_omitted=()) -> str:
    what = (f"*{_slack_escape(title)}* (section {int(spec['section_id'])}, "
            f"{_slack_escape(spec['locale'])})")
    live = spec.get("reconcile_with")
    if live:
        what += f", a separate draft revising {_slack_link(live.get('html_url'), live.get('title') or 'the live article')}"
    if row_url:
        what += f", from {_slack_link(row_url, 'this Release Assets row')}"
    if images_omitted:
        what += (f". {len(images_omitted)} screenshot(s) left out for a human to add: "
                 + "; ".join(_slack_escape(c) for c in images_omitted))
    return what


async def _queue_veto(spec: dict, cfg: dict, m: dict) -> dict:
    from zoneinfo import ZoneInfo
    from agent import autonomy
    from tools import release_asset_gate as gate
    now = _now()
    try:
        plan = autonomy.plan_veto(cfg, now)
    except autonomy.AutonomyConfigError as e:
        return {"error": str(e)}
    pending_id = autonomy.new_pending_id()
    text = (f":hourglass_flowing_sand: I plan to create an *unpublished* Zendesk draft "
            f"{_describe(spec, m['title'], m['row'].url, m['images_omitted'])}. It goes ahead after "
            f"{plan.not_before.astimezone(ZoneInfo(plan.timezone)):%a %d %b %H:%M} ({_slack_escape(plan.timezone)}) "
            f"unless someone reacts :x: here or sets the row to Hold. "
            f"(autonomy: veto · ref {pending_id})")
    try:
        ts = await gate.post_announcement(plan.channel, text)
    except gate.GateError as e:
        return {"error": f"Could not announce the veto window, so nothing was queued: {e}"}
    autonomy.save_pending(_ACTION, pending_id, {
        "spec": spec, "content_sha": m["sha"], "title": m["title"],
        "created_at": now.isoformat(), "not_before": plan.not_before.isoformat(),
        "expires_at": plan.expires_at.isoformat(), "announce_channel": plan.channel,
        "announcement_ts": ts,
    })
    _audit("autonomy_veto_announced", f"pending={pending_id} row={spec['row_id']}")
    return {
        "status": "pending", "pending_id": pending_id, "not_before": plan.not_before.isoformat(),
        "expires_at": plan.expires_at.isoformat(), "announcement_ts": ts,
        "message": ("Announced. It runs when zendesk_create_draft is called with run_due=true "
                    "(release-watch does this) or with this pending_id after not_before, unless "
                    "vetoed (❌ or Hold) or the row's content changes. It expires at expires_at."),
    }


async def _say(channel: str, text: str, thread_ts: str = "") -> str:
    """Best-effort Slack post; returns an error string or ''."""
    from tools import release_asset_gate as gate
    try:
        await gate.post_announcement(channel, text, thread_ts)
        return ""
    except gate.GateError as e:
        return str(e)


async def _execute_pending(pending_id: str, level: str, cfg: dict) -> dict:
    from agent import autonomy
    from tools import release_asset_gate as gate

    async def execute(signed: dict) -> dict:
        try:
            return await _create_once(
                signed["spec"], cfg, level, require_approved=(level == "signoff"),
                expected_sha=None if level == "signoff" else signed["content_sha"],
                pending_id=pending_id)
        except gate.RowStopped as e:
            raise autonomy.PendingStop("vetoed", str(e).rstrip("."))
        except _ContentChanged as e:
            raise autonomy.PendingStop(
                "stale", f"{str(e).rstrip('.')} — re-queue it so the team sees the new version")
        except (gate.GateError, ZendeskWriteRefused, ValueError) as e:
            raise autonomy.PendingRetry(str(e))

    return await autonomy.run_pending(
        _ACTION, pending_id, level, now=_now(), veto_reactions=gate.veto_reactions,
        say=_say, audit=_audit, execute=execute,
        done_fields=lambda r: {"article_id": r.get("article_id")},
        done_text=lambda r: (":white_check_mark: Created the unpublished draft: "
                             + _slack_link(r.get("edit_url"), _title_of(r))),
        verb="creating it", outcome="created")


async def run_due_pending(cfg: dict | None = None) -> list[dict]:
    """Execute every waiting veto action that is due (or expire it).

    For drivers that must not depend on the model remembering a pending_id:
    the teammate's release-watch calls ``zendesk_create_draft(run_due=true)``,
    or a cron pre-run script runs ``python -m tools.zendesk_tool run-due``.
    Applies the same level, write-guard and veto checks as a direct call.
    """
    from agent import autonomy
    cfg = _config() if cfg is None else cfg
    try:
        level = autonomy.resolve_level(_ACTION, cfg)
    except autonomy.AutonomyConfigError as e:
        return [{"error": str(e)}]
    if level == "off":
        return [{"error": "zendesk_create_draft is turned off for this profile (autonomy level 'off')."}]
    if not _write_guard_enabled(cfg):
        return [{"error": _WRITE_GUARD_MISSING}]
    return [{"pending_id": pid, **(await _execute_pending(pid, level, cfg))}
            for pid in autonomy.due_pending_ids(_ACTION, _now())]


ZENDESK_CREATE_DRAFT_SCHEMA = {
    "name": "zendesk_create_draft",
    "description": (
        "Create ONE UNPUBLISHED Zendesk Help Center draft from a Release Assets row. The title and "
        "body are taken from the row itself (its title and page content), never from you, and a "
        "row gets at most one draft (a repeat call returns the existing one). It never publishes, "
        "edits a live article, archives or deletes — a human publishes in Zendesk. You choose only "
        "placement: section_id (find it with zendesk_api) and locale; for a change to an existing "
        "article pass revision_of_article_id — a SEPARATE draft is created in its section and you "
        "must ask the team how to reconcile it. Rows with images are refused unless "
        "omit_images=true, which leaves them out and returns their captions for a human. Whether "
        "it may act is the profile's autonomy level: 'signoff' needs the row Approved; 'veto' "
        "announces in the home channel and runs later (run_due=true executes due ones); 'auto' "
        "acts and reports. list_pending=true lists waiting veto actions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "release_asset_row_id": {"type": "string",
                                     "description": "Notion id of the Release Assets row (Help article) to draft from."},
            "section_id": {"type": "string", "description": "Target Help Center section id."},
            "locale": {"type": "string", "default": "en-us", "description": "Article locale."},
            "revision_of_article_id": {"type": "string",
                                       "description": "Existing article this draft revises (creates a separate draft in its section)."},
            "omit_images": {"type": "boolean",
                            "description": "Leave the row's images out; their captions are returned for a human to add."},
            "pending_id": {"type": "string",
                           "description": "Run one veto-level action announced earlier (other arguments are ignored)."},
            "run_due": {"type": "boolean", "description": "Run every veto action whose window has passed."},
            "list_pending": {"type": "boolean", "description": "List veto actions still waiting."},
        },
        "required": [],
    },
}


def _respond(payload: dict) -> str:
    if "error" in payload and len(payload) == 1:
        return tool_error(payload["error"])
    return tool_result(payload)


async def _zendesk_create_draft_handler(args: dict, **_kw) -> str:
    from agent import autonomy
    from tools import release_asset_gate as gate

    if args.get("list_pending") is True:
        waiting = [{"pending_id": r.get("pending_id"),
                    "title": (r.get("signed") or {}).get("title"),
                    "not_before": (r.get("signed") or {}).get("not_before"),
                    "expires_at": (r.get("signed") or {}).get("expires_at")}
                   for r in autonomy.list_pending(_ACTION) if r.get("status") == "pending"]
        return tool_result({"ok": True, "pending": waiting})

    cfg = _config()
    try:
        level = autonomy.resolve_level(_ACTION, cfg)
    except autonomy.AutonomyConfigError as e:
        return tool_error(str(e))
    if level == "off":
        return tool_error("zendesk_create_draft is turned off for this profile (autonomy level 'off').")
    if not _write_guard_enabled(cfg):
        return tool_error(_WRITE_GUARD_MISSING)
    if not _check_zendesk():
        return tool_error(_NOT_CONFIGURED)

    if args.get("run_due") is True:
        return tool_result({"ok": True, "results": await run_due_pending(cfg)})
    pending_id = str(args.get("pending_id") or "").strip()
    if pending_id:
        return _respond(await _execute_pending(pending_id, level, cfg))

    try:
        spec = _build_spec(args)
        existing = autonomy.ledger_get(_ACTION, spec["row_id"])
        if existing:
            return tool_result(_existing_result(existing))
        await _resolve_placement(spec, cfg)
        if level == "veto":
            m = await _materialize(spec, cfg, require_approved=False)
            if m["row"].status not in _zendesk_settings(cfg)["approved"]:
                return _respond(await _queue_veto(spec, cfg, m))
        result = await _create_once(spec, cfg, level,
                                    require_approved=(level in ("signoff", "veto")))
    except (ValueError, gate.GateError, ZendeskWriteRefused,
            autonomy.AutonomyBusy, autonomy.AutonomyConfigError) as e:
        return tool_error(str(e))

    if level == "auto" and not result.get("already_exists"):
        channel, _w, _wh = autonomy.veto_settings(cfg)
        if channel:
            err = await _say(channel, ":memo: Created an unpublished Zendesk draft "
                             f"{_describe(spec, _title_of(result), result.get('row_url', ''), result.get('images_omitted') or ())}: "
                             f"{_slack_link(result.get('edit_url'), 'open the draft')} (autonomy: auto)")
            if err:
                result["announce_error"] = err
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


if __name__ == "__main__":  # cron pre-run script: python -m tools.zendesk_tool run-due
    import asyncio
    import sys
    if sys.argv[1:] != ["run-due"]:
        sys.exit("usage: python -m tools.zendesk_tool run-due")
    print(json.dumps(asyncio.run(run_due_pending()), ensure_ascii=False, default=str))
