"""Explicit test helpers for Wisdom's refresh-free local entitlement gate."""

from __future__ import annotations

import time
from collections.abc import Callable

import jwt


def bind_authorized_wisdom_token(
    monkeypatch,
    org_id: str = "org-1",
    *,
    scopes: tuple[str, ...] = ("wisdom:read", "wisdom:install", "wisdom:publish"),
    expires_in: float = 600,
) -> str:
    """Bind a synthetic, locally decoded token for one test's active profile."""
    token = jwt.encode(
        {
            "org_id": org_id,
            "exp": time.time() + expires_in,
            "wisdom_scopes": list(scopes),
        },
        "test-only-signing-key-for-local-entitlement-fixtures",
        algorithm="HS256",
    )
    status = {"logged_in": True, "access_token": token}
    monkeypatch.setattr(
        "hermes_cli.auth_nous.get_nous_auth_status_local", lambda: status
    )
    return token


def authorized_wisdom_token_fixture(monkeypatch) -> Callable[..., str]:
    """Return a binder and start the test authorized for the common fixture org."""
    bind_authorized_wisdom_token(monkeypatch)
    return lambda org_id="org-1", **kwargs: bind_authorized_wisdom_token(
        monkeypatch, org_id, **kwargs
    )
