"""C1/C2/C3: xai_http consumers under canonical shared xAI OAuth mode."""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import auth


def _jwt(exp: int) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').decode().rstrip("=")
    payload = (
        base64.urlsafe_b64encode(json.dumps({"exp": exp}).encode()).decode().rstrip("=")
    )
    return f"{header}.{payload}.sig"


@pytest.fixture
def shared_env(tmp_path, monkeypatch):
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir()
    hermes_home = tmp_path / "profile"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(shared_dir))
    monkeypatch.setenv("HERMES_XAI_SHARED_AUTH", "1")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    return {
        "shared_dir": shared_dir,
        "hermes_home": hermes_home,
        "store": shared_dir / "xai_oauth.json",
        "profile_auth": hermes_home / "auth.json",
    }


def _write_shared(env, *, access="at-1", refresh="rt-1", generation=1):
    payload = {
        "_schema": 1,
        "generation": generation,
        "access_token": access,
        "refresh_token": refresh,
        "token_type": "Bearer",
        "auth_mode": "oauth_device_code",
        "last_refresh": "2026-07-01T00:00:00Z",
        "discovery": {"token_endpoint": "https://auth.x.ai/oauth/token"},
    }
    env["store"].write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_has_xai_credentials_true_under_clean_shared(shared_env):
    """C2: clean shared mode (no profile tokens) still reports available."""
    from tools.xai_http import has_xai_credentials

    _write_shared(shared_env, access="shared-at", refresh="shared-rt")
    # No profile auth.json at all.
    assert not shared_env["profile_auth"].exists()
    assert has_xai_credentials() is True


def test_has_xai_credentials_false_when_profile_disabled(shared_env):
    from tools.xai_http import has_xai_credentials

    _write_shared(shared_env)
    auth.disable_profile_xai_shared_auth()
    assert has_xai_credentials() is False


def test_has_xai_credentials_fail_closed_on_shared_read_error(shared_env, monkeypatch):
    """R7: shared-mode canonical READ error → False (no legacy fallthrough)."""
    from tools.xai_http import has_xai_credentials

    _write_shared(shared_env, access="shared-at", refresh="shared-rt")
    # Plant a legacy pool row that would have been "available" under gate-off.
    shared_env["profile_auth"].write_text(
        json.dumps(
            {
                "version": 1,
                "providers": {
                    "xai-oauth": {
                        "tokens": {
                            "access_token": "legacy-at",
                            "refresh_token": "legacy-rt",
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    def boom(**_k):
        raise auth.AuthError(
            "unreadable",
            provider="xai-oauth",
            code="xai_shared_store_unreadable",
        )

    monkeypatch.setattr(auth, "_read_shared_xai_state", boom)
    assert has_xai_credentials() is False


def test_has_xai_credentials_recognizes_root_only_promotable(shared_env, monkeypatch):
    """R7: empty shared + root-only sole live grant → True (matches promoter)."""
    from tools.xai_http import has_xai_credentials

    # No shared store, no active-profile grant.
    assert not shared_env["store"].exists()
    root_auth = shared_env["hermes_home"].parent / "root_auth.json"
    root_auth.write_text(
        json.dumps(
            {
                "version": 1,
                "providers": {
                    "xai-oauth": {
                        "tokens": {
                            "access_token": "root-at",
                            "refresh_token": "root-rt",
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(auth, "_global_auth_file_path", lambda: root_auth)
    monkeypatch.setattr(
        auth,
        "_load_global_auth_store",
        lambda: json.loads(root_auth.read_text(encoding="utf-8")),
    )
    assert has_xai_credentials() is True


def test_resolve_canonical_first_over_legacy_pool(shared_env):
    """C1/A6: legacy pool RT must not win over the shared grant."""
    from tools.xai_http import resolve_xai_http_credentials

    _write_shared(shared_env, access="shared-at", refresh="shared-rt", generation=5)
    # Poison profile pool with a different access token that would win pool-first.
    shared_env["profile_auth"].write_text(
        json.dumps(
            {
                "version": 1,
                "credential_pool": {
                    "xai-oauth": [
                        {
                            "id": "legacy",
                            "source": "device_code",
                            "auth_type": "oauth",
                            "access_token": "legacy-pool-at",
                            "refresh_token": "legacy-pool-rt",
                            "priority": 0,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    creds = resolve_xai_http_credentials()
    assert creds["api_key"] == "shared-at"
    assert creds.get("source") == auth.XAI_SHARED_SOURCE


def test_force_refresh_passes_rejected_bearer(shared_env, monkeypatch):
    """C3 helper adopts winner when generation already moved."""
    from tools.xai_http import force_refresh_xai_http_credentials

    old = _jwt(int(time.time()) + 30)
    _write_shared(shared_env, access=old, refresh="rt-old", generation=1)
    posts = []

    def fake_pure(access, refresh, **kwargs):
        posts.append(refresh)
        return {
            "access_token": "new-at",
            "refresh_token": "new-rt",
            "token_type": "Bearer",
            "last_refresh": "2026-07-18T00:00:00Z",
        }

    monkeypatch.setattr(auth, "refresh_xai_oauth_pure", fake_pure)
    # Concurrent winner already rotated.
    _write_shared(shared_env, access="winner-at", refresh="winner-rt", generation=2)

    creds = force_refresh_xai_http_credentials(old)
    assert posts == []
    assert creds["api_key"] == "winner-at"


def test_x_search_retries_after_401(shared_env, monkeypatch):
    """C3: x_search refreshes once on 401 and retries."""
    import requests
    from tools import x_search_tool

    _write_shared(
        shared_env,
        access=_jwt(int(time.time()) + 3600),
        refresh="rt-1",
        generation=1,
    )

    calls = {"n": 0}
    refresh_calls = {"n": 0}

    class FakeResp:
        def __init__(self, status, payload=None):
            self.status_code = status
            self._payload = payload or {}
            self.text = json.dumps(self._payload)

        def raise_for_status(self):
            if self.status_code >= 400:
                err = requests.HTTPError(f"{self.status_code}")
                err.response = self
                raise err

        def json(self):
            return self._payload

    def fake_post(url, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeResp(401, {"error": "bad token"})
        return FakeResp(
            200,
            {
                "output_text": "ok",
                "citations": [],
                "output": [],
            },
        )

    def fake_refresh(rejected=None):
        refresh_calls["n"] += 1
        return {
            "provider": "xai-oauth",
            "api_key": "fresh-at",
            "base_url": "https://api.x.ai/v1",
        }

    monkeypatch.setattr(x_search_tool.requests, "post", fake_post)
    monkeypatch.setattr(
        x_search_tool,
        "resolve_xai_http_credentials",
        lambda **kw: {
            "provider": "xai-oauth",
            "api_key": "stale-at",
            "base_url": "https://api.x.ai/v1",
        },
    )
    monkeypatch.setattr(
        "tools.xai_http.force_refresh_xai_http_credentials",
        fake_refresh,
    )
    monkeypatch.setattr(x_search_tool, "_get_x_search_retries", lambda: 0)
    monkeypatch.setattr(x_search_tool, "_get_x_search_timeout_seconds", lambda: 5)
    monkeypatch.setattr(x_search_tool, "_get_x_search_model", lambda: "grok-4")

    result = x_search_tool.x_search_tool("hello")
    data = json.loads(result)
    assert data.get("answer") == "ok" or "ok" in result
    assert calls["n"] == 2
    assert refresh_calls["n"] == 1
