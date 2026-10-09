"""Guest mode token store for Telegram guest delivery (Bot API 10.0).

mint_token() / resolve_token() are called by the adapter to track staged
media across the two-hop deliver_<token> flow.  No LLM involvement.
"""

from __future__ import annotations

import time
import uuid
from typing import Optional

_TOKEN_STORE: dict[str, dict] = {}
_TOKEN_TTL = 600  # 10 minutes


def mint_token(file_id: str, media_kind: str) -> str:
    """Mint a delivery token. Stored for _TOKEN_TTL seconds."""
    token = uuid.uuid4().hex[:16]
    _TOKEN_STORE[token] = {
        "file_id": file_id,
        "media_kind": media_kind,
        "expires_at": time.monotonic() + _TOKEN_TTL,
    }
    return token


def resolve_token(token: str) -> Optional[dict]:
    """Resolve a delivery token without consuming it.

    Returns None if the token is unknown or expired.  Tokens are kept in the
    store until natural TTL expiry so repeat taps on the deliver button
    re-deliver rather than falling through to the invalid-token path.
    """
    entry = _TOKEN_STORE.get(token)
    if entry is None:
        return None
    if time.monotonic() > entry["expires_at"]:
        _TOKEN_STORE.pop(token, None)
        return None
    return entry
