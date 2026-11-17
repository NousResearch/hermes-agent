"""jira_api — a thin, structured proxy over the JIRA Cloud REST API (read-only).

The model calls ``jira_api(path, method)`` and gets parsed JSON back. Mirrors
``notion_api``'s ergonomics, but talks HTTP directly (no CLI dependency) using
Atlassian Cloud basic auth (email:API_TOKEN, base64) against JIRA_BASE_URL. MVP
is GET-only — the reconciliation worker only reads. Credentials live in
~/.hermes/.env (JIRA_BASE_URL / JIRA_EMAIL / JIRA_API_TOKEN) and self-heal for
headless cron/worker runs that never loaded the dotenv (mirrors slack_react).

``jira_api_write`` is the mutating twin, registered under its OWN toolset
(``jira_write``) so granting reads never implies writes. Jira API tokens carry
no scopes, so the curated ``_WRITE_ALLOWLIST`` below IS the boundary: issue
transitions, comments, field edits and issue creation — never DELETE, never
project/workflow/user admin, never bulk. Reads (list transitions, editmeta,
existing comments) stay on ``jira_api``; the two compose through RBAC.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re

from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

_ALLOWED_METHODS = frozenset({"GET"})
_JIRA_TIMEOUT_SECONDS = 30


JIRA_API_SCHEMA = {
    "name": "jira_api",
    "description": (
        "Call the JIRA Cloud REST API (READ-ONLY) and get parsed JSON back. Pass "
        "an API path WITHOUT a leading slash, e.g. "
        "'rest/api/3/issue/EA-123?fields=status' or "
        "'rest/api/3/search/jql?jql=key IN (EA-1,EA-2)&fields=status'. Only GET "
        "is supported. Use this for ALL JIRA reads — never shell out or use curl."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "JIRA REST path beginning with 'rest/', no leading slash.",
            },
            "method": {
                "type": "string",
                "enum": ["GET"],
                "description": "HTTP method. Only GET is supported.",
                "default": "GET",
            },
        },
        "required": ["path"],
    },
}


# Project keys are UPPERCASE[A-Z0-9_]-digits. Numeric issue ids are deliberately
# NOT accepted (a key is what a human reads in Slack), and there is no room for a
# query string or a trailing segment, so `.../comment/../../project` cannot match.
_ISSUE_KEY = r"[A-Z][A-Z0-9_]+-\d+"
_WRITE_ALLOWLIST: tuple[tuple[str, "re.Pattern[str]"], ...] = tuple(
    (method, re.compile(pattern))
    for method, pattern in (
        ("POST", rf"^rest/api/3/issue/{_ISSUE_KEY}/transitions$"),  # move status
        ("POST", rf"^rest/api/3/issue/{_ISSUE_KEY}/comment$"),      # add a comment
        ("PUT",  rf"^rest/api/3/issue/{_ISSUE_KEY}$"),              # edit fields
        ("POST", r"^rest/api/3/issue$"),                             # create an issue
    )
)
_WRITE_ALLOWLIST_HELP = (
    "POST rest/api/3/issue/<KEY>/transitions, POST rest/api/3/issue/<KEY>/comment, "
    "PUT rest/api/3/issue/<KEY>, POST rest/api/3/issue"
)

JIRA_API_WRITE_SCHEMA = {
    "name": "jira_api_write",
    "description": (
        "Make a WRITE call to the JIRA Cloud REST API. Only these are allowed: "
        f"{_WRITE_ALLOWLIST_HELP}. Anything else (DELETE, project/workflow/user "
        "admin, bulk endpoints) is refused. Read first with jira_api (GET): list "
        "valid transitions at rest/api/3/issue/<KEY>/transitions and pass the "
        "chosen transition id here; check editable fields at "
        "rest/api/3/issue/<KEY>/editmeta before a PUT. Body is the JSON object "
        "the endpoint expects, e.g. {\"transition\": {\"id\": \"31\"}} or "
        "{\"body\": {\"type\": \"doc\", ...}} for a comment (ADF)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "One of the allowed JIRA REST paths, no leading slash, no query string.",
            },
            "method": {
                "type": "string",
                "enum": ["POST", "PUT"],
                "description": "HTTP method; must match the allowed (method, path) pair.",
            },
            "body": {
                "type": "object",
                "description": "JSON request body for the endpoint.",
            },
        },
        "required": ["path", "method", "body"],
    },
}


def _write_allowed(method: str, path: str) -> bool:
    return any(m == method and pat.match(path) for m, pat in _WRITE_ALLOWLIST)


def _jira_creds() -> tuple[str, str, str]:
    """Resolve (base_url, email, token), self-healing from ~/.hermes/.env once.

    A headless cron/worker process may never have loaded the dotenv, so load it
    if any var is missing (mirrors notion_api/_ntn_env and slack_react).

    Under ``gateway.multiplex_profiles`` the values come from the owning
    profile's secret scope only and the self-heal is skipped (it would load
    into the default profile's ``os.environ``) — see ``agent.fork_secrets``.
    """
    from agent.fork_secrets import heal_dotenv, read_secret

    def _read() -> tuple[str, str, str]:
        return (
            read_secret("JIRA_BASE_URL", tool="jira_api").rstrip("/"),
            read_secret("JIRA_EMAIL", tool="jira_api"),
            read_secret("JIRA_API_TOKEN", tool="jira_api"),
        )

    base, email, token = _read()
    if not (base and email and token) and heal_dotenv():
        base, email, token = _read()
    return base, email, token


async def _jira_request(
    base_url: str,
    email: str,
    token: str,
    path: str,
    method: str = "GET",
    json_body: dict | None = None,
) -> dict:
    """Call base_url/path with Atlassian basic auth. Returns {status, text}.

    The raw path (incl. an un-encoded ``?jql=key IN (...)`` query) is handed to
    aiohttp/yarl, which percent-encodes it. ``json_body`` is sent as JSON for
    POST/PUT; the default is a bare GET.
    """
    import aiohttp
    from gateway.platforms.base import resolve_proxy_url, proxy_kwargs_for_aiohttp

    url = f"{base_url}/{path.lstrip('/')}"
    cred = base64.b64encode(f"{email}:{token}".encode()).decode()
    headers = {"Authorization": f"Basic {cred}", "Accept": "application/json"}
    _proxy = resolve_proxy_url()
    _sess_kw, _req_kw = proxy_kwargs_for_aiohttp(_proxy)
    timeout = aiohttp.ClientTimeout(total=_JIRA_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout, **_sess_kw) as session:
        async with session.request(method, url, headers=headers, json=json_body, **_req_kw) as resp:
            return {"status": resp.status, "text": await resp.text()}


async def _jira_api_handler(args: dict, **_kw) -> str:
    path = (args.get("path") or "").strip()
    method = (args.get("method") or "GET").strip().upper()

    if method not in _ALLOWED_METHODS:
        return tool_error(f"Unsupported method '{method}'. Only GET is allowed.")
    if not path.startswith("rest/"):
        return tool_error("path must be a JIRA REST path beginning with 'rest/'.")

    base, email, token = _jira_creds()
    if not (base and email and token):
        return tool_error(
            "JIRA not configured (set JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN)."
        )

    try:
        resp = await _jira_request(base, email, token, path)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("[jira_api] request failed (GET %s): %s", path, e)
        return tool_error(f"JIRA request failed: {e}")

    return _shape_response(resp, f"GET {path}")


def _shape_response(resp: dict, what: str) -> str:
    status = resp["status"]
    body = (resp["text"] or "").strip()
    if status >= 400:
        logger.warning("[jira_api] HTTP %s (%s): %s", status, what, body[:300])
        return tool_error(f"JIRA API error (HTTP {status}): {body[:300]}")
    if not body:
        return tool_result({"ok": True, "data": None})
    try:
        data = json.loads(body)
    except ValueError:
        return tool_result({"ok": True, "text": body})
    return tool_result({"ok": True, "data": data})


async def _jira_api_write_handler(args: dict, **_kw) -> str:
    path = (args.get("path") or "").strip()
    method = (args.get("method") or "").strip().upper()
    body = args.get("body")

    if not _write_allowed(method, path):
        return tool_error(
            f"{method} {path} is not an allowed JIRA write. Allowed: "
            f"{_WRITE_ALLOWLIST_HELP}. Use jira_api (GET) for reads."
        )
    if not isinstance(body, dict):
        return tool_error("body must be a JSON object.")

    base, email, token = _jira_creds()
    if not (base and email and token):
        return tool_error(
            "JIRA not configured (set JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN)."
        )

    try:
        resp = await _jira_request(base, email, token, path, method=method, json_body=body)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("[jira_api_write] request failed (%s %s): %s", method, path, e)
        return tool_error(f"JIRA request failed: {e}")
    return _shape_response(resp, f"{method} {path}")


def _check_jira_api() -> bool:
    """Available whenever all three JIRA credentials are resolvable."""
    base, email, token = _jira_creds()
    return bool(base and email and token)


registry.register(
    name="jira_api",
    toolset="jira",
    schema=JIRA_API_SCHEMA,
    handler=lambda args, **kw: _jira_api_handler(args, **kw),
    check_fn=_check_jira_api,
    requires_env=[],
    is_async=True,
    emoji="🎫",
    max_result_size_chars=16000,
)

registry.register(
    name="jira_api_write",
    toolset="jira_write",
    schema=JIRA_API_WRITE_SCHEMA,
    handler=lambda args, **kw: _jira_api_write_handler(args, **kw),
    check_fn=_check_jira_api,
    requires_env=[],
    is_async=True,
    emoji="🎫",
    max_result_size_chars=16000,
)
