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
