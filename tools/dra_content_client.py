"""HTTP client for the DRA Content API.

Every dra_content tool goes through here rather than calling httpx directly,
so the service-token header, the delegated-identity header, timeouts and
error shaping are handled in exactly one place.

Two identities are in play on every call:

  * the SERVICE TOKEN authenticates this Hermes process to content-api. It
    is a transport credential, not a user.
  * X-DRA-On-Behalf-Of names the human the conversation is actually with,
    resolved from the vault the same way every other GWS tool in this
    codebase resolves identity -- never taken from a tool argument, always
    from session context. content-api evaluates every ACL check as that
    person, not as the bare Hermes service identity (see DRA Content
    Stage 9: the service identity is deliberately never a global admin and
    owns only what it itself created, so without this header Hermes could
    not act on a specific person's behalf at all).

No token, secret or credential of any kind is ever part of a tool's return
value -- only content-api's JSON response, which contains none.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

BASE_URL = os.environ.get("DRA_CONTENT_API_URL", "http://content-api:8650").rstrip("/")
SERVICE_TOKEN = os.environ.get("DRA_CONTENT_SERVICE_TOKEN", "")
TIMEOUT = float(os.environ.get("DRA_CONTENT_HTTP_TIMEOUT", "20"))


class DRAContentError(RuntimeError):
    """Raised for any non-2xx response. Carries the status and detail so
    handlers can surface a useful message without re-parsing JSON."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


def _current_actor_email() -> str | None:
    """Resolve the acting human's email from session context.

    Mirrors tools/gws_ops_tools.py's _default_service_name(): same session
    resolution path (_current_telegram_id -> canonical_uid -> vault
    identity), so a user who is authorized for GWS operations resolves to
    the same person here. Returns None for contexts with no session user
    (e.g. a cron job), in which case the call proceeds as the bare service
    identity -- content-api's default_acl still applies, so this never
    results in an over-broad grant, only a narrower one.
    """
    try:
        from tools.gws_auth import _current_telegram_id, canonical_uid
        from tools import gws_vault_client as vault

        tid = _current_telegram_id()
        uid = canonical_uid(tid)
        identity = vault.get_identity(uid, session_uid=uid) if uid else None
        if not identity:
            return None
        emails = identity.get("identities", {}).get("email") or []
        if not emails:
            return None
        # Prefer an internal-domain address (draas.com) if the person has
        # one on file; a Telegram user with only a personal address (an
        # external contact who somehow has a session) gets that instead.
        internal_domains = {
            d.strip().lower()
            for d in os.environ.get("DRA_CONTENT_INTERNAL_DOMAINS", "draas.com").split(",")
            if d.strip()
        }
        for e in emails:
            if str(e).split("@")[-1].strip().lower() in internal_domains:
                return str(e).strip().lower()
        return str(emails[0]).strip().lower()
    except Exception:
        logger.debug("dra_content_client: could not resolve session identity", exc_info=True)
        return None


def _headers() -> dict:
    if not SERVICE_TOKEN:
        raise DRAContentError(500, "DRA_CONTENT_SERVICE_TOKEN is not configured")
    headers = {"Authorization": f"Bearer {SERVICE_TOKEN}"}
    actor = _current_actor_email()
    if actor:
        headers["X-DRA-On-Behalf-Of"] = actor
    return headers


async def _request(method: str, path: str, **kwargs: Any) -> dict:
    url = f"{BASE_URL}{path}"
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.request(method, url, headers=_headers(), **kwargs)
    except httpx.RequestError as exc:
        logger.warning("dra_content_client: %s %s failed: %s", method, path, exc)
        raise DRAContentError(503, f"could not reach DRA Content ({type(exc).__name__})") from exc

    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise DRAContentError(resp.status_code, str(detail))

    if resp.status_code == 204 or not resp.content:
        return {}
    return resp.json()


async def get(path: str, **kwargs: Any) -> dict:
    return await _request("GET", path, **kwargs)


async def post(path: str, json_body: dict | None = None, **kwargs: Any) -> dict:
    return await _request("POST", path, json=json_body, **kwargs)


async def patch(path: str, json_body: dict | None = None, **kwargs: Any) -> dict:
    return await _request("PATCH", path, json=json_body, **kwargs)


async def delete(path: str, **kwargs: Any) -> dict:
    return await _request("DELETE", path, **kwargs)
