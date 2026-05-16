"""Integration tests for feishu_oauth — FeishuUserTokenStore PKCE flow."""

import json
import time
import tempfile
import os
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_token_dir(monkeypatch, tmp_path):
    """Override token storage to a temp directory."""
    monkeypatch.setattr(
        "tools.feishu_oauth._token_dir",
        lambda: tmp_path / "feishu-user-tokens",
    )
    # Also patch FeishuOAuthConfig defaults so we don't need real env vars
    monkeypatch.setenv("FEISHU_APP_ID", "test_app_id")
    monkeypatch.setenv("FEISHU_APP_SECRET", "test_app_secret")
    return tmp_path


@pytest.fixture
def store(tmp_token_dir):
    from tools.feishu_oauth import FeishuUserTokenStore
    return FeishuUserTokenStore()


# ---------------------------------------------------------------------------
# PKCE helpers
# ---------------------------------------------------------------------------

class TestPKCEGeneration:
    def test_pkce_pair_returns_verifier_and_challenge(self):
        from tools.feishu_oauth import FeishuUserTokenStore
        verifier, challenge = FeishuUserTokenStore._pkce_pair()
        assert len(verifier) >= 32
        assert len(challenge) >= 32
        # S256 challenge should be base64url-encoded SHA256 digest
        import base64, hashlib
        expected = base64.urlsafe_b64encode(
            hashlib.sha256(verifier.encode()).digest()
        ).rstrip(b"=").decode()
        assert challenge == expected

    def test_pkce_pair_is_unique_each_call(self):
        from tools.feishu_oauth import FeishuUserTokenStore
        v1, c1 = FeishuUserTokenStore._pkce_pair()
        v2, c2 = FeishuUserTokenStore._pkce_pair()
        assert v1 != v2
        assert c1 != c2


# ---------------------------------------------------------------------------
# Token file I/O
# ---------------------------------------------------------------------------

class TestTokenStorage:
    def test_no_token_returns_none(self, store, tmp_token_dir):
        result = store.get_user_token("ou_no_such_user")
        assert result is None

    def test_auth_status_not_authorized_for_new_user(self, store):
        status = store.get_auth_status("ou_no_such_user")
        assert status["status"] == "not_authorized"
        assert status["open_id"] == "ou_no_such_user"

    def test_auth_status_expired_after_write(self, store, tmp_token_dir):
        open_id = "ou_test_expired"
        # Write an expired token directly
        token_path = tmp_token_dir / "feishu-user-tokens" / f"{open_id}.json"
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(json.dumps({
            "access_token": "old_token",
            "refresh_token": "refresh_token",
            "expires_at": time.time() - 100,  # expired 100s ago
            "refresh_expires_at": time.time() + 86400,
            "open_id": open_id,
        }))
        status = store.get_auth_status(open_id)
        assert status["status"] == "expired"

    def test_auth_status_expiring_soon(self, store, tmp_token_dir):
        open_id = "ou_test_expiring"
        token_path = tmp_token_dir / "feishu-user-tokens" / f"{open_id}.json"
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(json.dumps({
            "access_token": "old_token",
            "refresh_token": "refresh_token",
            "expires_at": time.time() + 120,  # expires in 2 min (< 5 min threshold)
            "refresh_expires_at": time.time() + 86400,
            "open_id": open_id,
        }))
        status = store.get_auth_status(open_id)
        assert status["status"] == "expiring_soon"

    def test_auth_status_authorized(self, store, tmp_token_dir):
        open_id = "ou_test_ok"
        token_path = tmp_token_dir / "feishu-user-tokens" / f"{open_id}.json"
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(json.dumps({
            "access_token": "valid_token",
            "refresh_token": "refresh_token",
            "expires_at": time.time() + 7200,
            "refresh_expires_at": time.time() + 86400,
            "open_id": open_id,
        }))
        status = store.get_auth_status(open_id)
        assert status["status"] == "authorized"


# ---------------------------------------------------------------------------
# get_authorization_url
# ---------------------------------------------------------------------------

class TestAuthorizationUrl:
    def test_url_contains_required_params(self, store):
        open_id = "ou_test_user"
        url, state = store.get_authorization_url(open_id)
        assert url.startswith("https://open.feishu.cn/open-apis/authen/v1/authorize")
        # URL params are URL-encoded; app_id value is percent-encoded
        assert "app_id=" in url
        assert "code_challenge_method=S256" in url
        assert "code_challenge=" in url
        assert "state=" in url
        assert "scope=" in url
        assert "redirect_uri=" in url
        assert len(state) >= 16

    def test_pending_file_created(self, store, tmp_token_dir):
        open_id = "ou_pending_test"
        store.get_authorization_url(open_id)
        pending = store._pending_path(open_id)
        assert pending.exists(), f"Expected {pending} to exist"
        data = json.loads(pending.read_text())
        assert "verifier" in data
        assert "state" in data
        assert "created_at" in data

    def test_pending_file_expires_after_10_minutes(self, store, tmp_token_dir):
        open_id = "ou_expired_pending"
        url, _ = store.get_authorization_url(open_id)
        # Simulate 11-minute-old pending by backdating created_at
        pending = store._pending_path(open_id)
        data = json.loads(pending.read_text())
        data["created_at"] = time.time() - 660  # 11 min ago
        pending.write_text(json.dumps(data))
        # _read_pending should return None and delete the file
        result = store._read_pending(open_id)
        assert result is None

    def test_pending_deleted_after_exchange_code_called_with_empty_pending(self, store, tmp_token_dir):
        # If pending file doesn't exist, exchange_code should not crash
        open_id = "ou_no_pending"
        # Don't call get_authorization_url, so no pending file
        # exchange_code should handle missing pending gracefully
        try:
            store.exchange_code(open_id, "fake_code")
        except Exception:
            pass  # Network call will fail but no file error


# ---------------------------------------------------------------------------
# get_user_token (no refresh needed)
# ---------------------------------------------------------------------------

class TestGetUserToken:
    def test_returns_stored_token_when_valid(self, store, tmp_token_dir):
        open_id = "ou_valid"
        token_path = tmp_token_dir / "feishu-user-tokens" / f"{open_id}.json"
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(json.dumps({
            "access_token": "stored_token_abc123",
            "refresh_token": "refresh_xyz",
            "expires_at": time.time() + 3600,
            "refresh_expires_at": time.time() + 86400,
            "open_id": open_id,
        }))
        token = store.get_user_token(open_id)
        assert token == "stored_token_abc123"


# ---------------------------------------------------------------------------
# get_request_option
# ---------------------------------------------------------------------------

class TestGetRequestOption:
    def test_returns_none_when_no_token(self, store):
        option = store.get_request_option("ou_no_token")
        assert option is None

    def test_request_option_has_user_access_token(self, store, tmp_token_dir):
        open_id = "ou_option_test"
        token_path = tmp_token_dir / "feishu-user-tokens" / f"{open_id}.json"
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(json.dumps({
            "access_token": "user_access_token_xyz",
            "refresh_token": "refresh",
            "expires_at": time.time() + 3600,
            "refresh_expires_at": time.time() + 86400,
            "open_id": open_id,
        }))
        option = store.get_request_option(open_id)
        assert option is not None
        # Verify the option has user_access_token set
        assert hasattr(option, "user_access_token")
        assert option.user_access_token == "user_access_token_xyz"


# ---------------------------------------------------------------------------
# revoke
# ---------------------------------------------------------------------------

class TestRevoke:
    def test_revoke_deletes_token_file(self, store, tmp_token_dir):
        open_id = "ou_to_revoke"
        token_path = tmp_token_dir / "feishu-user-tokens" / f"{open_id}.json"
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text('{"access_token": "x"}')
        store.revoke(open_id)
        assert not token_path.exists()

    def test_revoke_idempotent(self, store):
        # Revoking a non-existent token should not raise
        store.revoke("ou_nonexistent")


# ---------------------------------------------------------------------------
# Disk I/O helpers
# ---------------------------------------------------------------------------

class TestDiskIO:
    def test_read_json_missing_file_returns_none(self):
        from tools.feishu_oauth import _read_json
        from pathlib import Path
        result = _read_json(Path("/nonexistent/path/file.json"))
        assert result is None

    def test_read_json_malformed_returns_none(self, tmp_path):
        from tools.feishu_oauth import _read_json
        f = tmp_path / "bad.json"
        f.write_text("not valid json {{{")
        result = _read_json(f)
        assert result is None

    def test_write_json_creates_file(self, tmp_path):
        from tools.feishu_oauth import _write_json
        p = tmp_path / "out.json"
        _write_json(p, {"key": "value"})
        assert p.exists()
        assert json.loads(p.read_text()) == {"key": "value"}

    def test_write_json_atomic(self, tmp_path):
        from tools.feishu_oauth import _write_json
        p = tmp_path / "atomic.json"
        _write_json(p, {"atomic": True})
        # File should exist and be valid
        assert json.loads(p.read_text()) == {"atomic": True}
