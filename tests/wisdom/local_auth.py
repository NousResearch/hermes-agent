"""Explicit synthetic local OAuth snapshots for authorized Wisdom unit fixtures."""
import time

import jwt


def authorize_local(monkeypatch, org="org-1", *, scopes=None, expires_in=3600):
    """Install an expiring JWT snapshot; callable org models account/team switches."""
    def snapshot():
        value = org() if callable(org) else org
        return {
            "logged_in": bool(value),
            "access_token": jwt.encode(
                {"sub": "fixture-user", "org_id": value,
                 "exp": time.time() + expires_in,
                 "wisdom_scopes": scopes if scopes is not None else [
                     "wisdom:read", "wisdom:install", "wisdom:publish"
                 ]},
                "test-only-local-entitlement-key-32-bytes", algorithm="HS256",
            ),
        }
    monkeypatch.setattr("hermes_cli.auth_nous.get_nous_auth_status_local", snapshot)
