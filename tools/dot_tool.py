"""Dot payment tool -- fetch a member's payout status and history from Dot.

Dot is CRWD's payments partner. This tool talks ONLY to the Dot HTTP API -- it
holds no MongoDB/CRWD logic. Gig, membership, and approval lookups stay in the
``crwd_db`` tool; the ``crwd-payment-status`` skill is what combines the two.

Gated on ``DOT_API_KEY`` + ``DOT_API_BASE_URL``. Every failure (network, HTTP,
bad JSON) is returned as ``{"error": ...}`` -- the tool never raises -- so the
coach can fall back to ``crwd_db`` + an honest handoff.

Auth: sends ``Authorization: Bearer <DOT_API_KEY>`` by default. If the Dot API
expects the raw key under a custom header instead, set ``DOT_API_KEY_HEADER``
(e.g. ``x-api-key``) and the key is sent under that header.

NOTE (pending Dot API spec): the exact endpoint paths, query-param names, and
response field names are isolated in the ``_DOT_*`` constants and the
``_dot_get`` / ``_as_items`` seam below, so they can be filled in from the spec
without touching the handler, schema, or skill. Defaults are best-effort.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

_TIMEOUT_S = 8
_HARD_LIMIT = 20

# --- Dot API surface (fill in from the Dot API spec) ---
_DOT_HISTORY_PATH = "/payouts"   # GET: payouts for a user
_DOT_STATUS_PATH = "/payouts"    # GET: payouts, optionally filtered to one gig
_DOT_USER_PARAM = "user_id"      # query param carrying the CRWD user id
_DOT_GIG_PARAM = "campaign_id"   # query param carrying the gig/campaign id


# --- Availability ---

def check_dot_requirements() -> bool:
    """Available only when the Dot API key and base URL are both configured."""
    return bool(
        os.getenv("DOT_API_KEY", "").strip()
        and os.getenv("DOT_API_BASE_URL", "").strip()
    )


# --- HTTP seam ---

def _auth_headers() -> Dict[str, str]:
    """Auth header for Dot. Bearer by default; custom header if configured."""
    key = os.getenv("DOT_API_KEY", "").strip()
    custom = os.getenv("DOT_API_KEY_HEADER", "").strip()
    if custom:
        return {custom: key}
    return {"Authorization": f"Bearer {key}"}


def _dot_get(path: str, params: Dict[str, Any]) -> Tuple[Optional[Any], Optional[str]]:
    """GET ``{base}{path}?params`` from Dot. Returns ``(parsed_json, error)``.

    Never raises: transport/HTTP/JSON problems come back as the error string so
    callers can degrade gracefully.
    """
    base = os.getenv("DOT_API_BASE_URL", "").strip().rstrip("/")
    if not base:
        return None, "DOT_API_BASE_URL is not set"
    clean = {k: v for k, v in params.items() if v not in (None, "")}
    query = urllib.parse.urlencode(clean)
    url = f"{base}{path}" + (f"?{query}" if query else "")
    headers = {"Accept": "application/json", **_auth_headers()}
    req = urllib.request.Request(url, method="GET", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            if not (200 <= resp.status < 300):
                return None, f"HTTP {resp.status}"
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return None, f"HTTP {exc.code}"
    except Exception as exc:  # network / URL / timeout
        return None, str(exc)
    try:
        return json.loads(raw), None
    except Exception:
        return None, "invalid JSON from Dot"


def _as_items(data: Any) -> List[Any]:
    """Best-effort extract a list of payout records from Dot's response.

    Dot may return a bare list or an object wrapping one; this keeps the tool
    resilient to the exact envelope until the spec pins it down.
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "payouts", "results", "items"):
            val = data.get(key)
            if isinstance(val, list):
                return val
        return [data]
    return []


# --- Actions ---

def _get_payment_history(user_id: str, limit: int = 10) -> str:
    user_id = (user_id or "").strip()
    if not user_id:
        return tool_error("user_id is required for get_payment_history")
    row_limit = max(1, min(int(limit or 10), _HARD_LIMIT))
    data, err = _dot_get(_DOT_HISTORY_PATH, {_DOT_USER_PARAM: user_id, "limit": row_limit})
    if err:
        return tool_error(f"Dot lookup failed: {err}")
    items = _as_items(data)[:row_limit]
    return json.dumps(
        {"_type": "dot_payment_history", "items": items, "error": None},
        ensure_ascii=False,
    )


def _get_payment_status(user_id: str, gig_id: str = "", limit: int = 10) -> str:
    user_id = (user_id or "").strip()
    if not user_id:
        return tool_error("user_id is required for get_payment_status")
    row_limit = max(1, min(int(limit or 10), _HARD_LIMIT))
    params: Dict[str, Any] = {_DOT_USER_PARAM: user_id, "limit": row_limit}
    gig_id = (gig_id or "").strip()
    if gig_id:
        params[_DOT_GIG_PARAM] = gig_id
    data, err = _dot_get(_DOT_STATUS_PATH, params)
    if err:
        return tool_error(f"Dot lookup failed: {err}")
    items = _as_items(data)[:row_limit]
    return json.dumps(
        {"_type": "dot_payment_status", "gig_id": gig_id or None, "items": items, "error": None},
        ensure_ascii=False,
    )


# --- Handler ---

def dot_tool(args: Dict[str, Any], **_kw: Any) -> str:
    if not check_dot_requirements():
        return tool_error("Dot is not configured (set DOT_API_KEY and DOT_API_BASE_URL).")
    action = str(args.get("action", "")).strip()
    try:
        if action == "get_payment_status":
            return _get_payment_status(
                user_id=str(args.get("user_id", "")),
                gig_id=str(args.get("gig_id", "") or args.get("campaign_id", "")),
                limit=args.get("limit", 10),
            )
        if action == "get_payment_history":
            return _get_payment_history(
                user_id=str(args.get("user_id", "")),
                limit=args.get("limit", 10),
            )
        return tool_error("Unknown action. Use get_payment_status or get_payment_history.")
    except Exception:
        logger.exception("dot action %r failed", action)
        return tool_error("Dot query failed")


# --- Schema ---

DOT_SCHEMA = {
    "name": "dot",
    "description": (
        "Look up a CRWD member's Dot payout status and history (Dot is CRWD's "
        "payments partner). Read-only. Use for 'did I get paid?', 'where's my "
        "money?', 'when will I be paid?', or 'show my payment history'. Two "
        "actions: get_payment_status (optionally scoped to one gig via gig_id) "
        "and get_payment_history. Returns Dot's payout records only — pair it "
        "with crwd_db for gig/approval context (the crwd-payment-status skill "
        "does this). Escalate genuine money disputes to a human."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["get_payment_status", "get_payment_history"],
                "description": "get_payment_status = payout state (optionally one gig); get_payment_history = all payouts for the member.",
            },
            "user_id": {
                "type": "string",
                "description": "The member's CRWD user_id (from the [CRWD member] context line).",
            },
            "gig_id": {
                "type": "string",
                "description": "Optional gig/campaign id to scope get_payment_status to a single gig.",
            },
            "limit": {
                "type": "integer",
                "description": "Max records to return (default 10, capped at 20).",
                "default": 10,
            },
        },
        "required": ["action", "user_id"],
    },
}


# --- Registration ---

registry.register(
    name="dot",
    toolset="dot",
    schema=DOT_SCHEMA,
    handler=dot_tool,
    check_fn=check_dot_requirements,
    requires_env=["DOT_API_KEY", "DOT_API_BASE_URL"],
    emoji="💰",
)
