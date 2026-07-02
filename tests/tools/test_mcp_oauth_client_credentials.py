"""Tests for the headless client_credentials (M2M) MCP OAuth grant.

Covers the ``oauth.grant: client_credentials`` path in ``tools/mcp_oauth.py``
and ``tools/mcp_oauth_manager.py``: it wires the MCP SDK's headless
``ClientCredentialsOAuthProvider`` so a daemon can authenticate to an
OAuth-fronted MCP gateway with no browser, no callback, and no interactive
re-auth.
"""

from __future__ import annotations

import json

import pytest
from mcp.client.auth.extensions.client_credentials import ClientCredentialsOAuthProvider

from tools import mcp_oauth as mo
from tools import mcp_oauth_manager as mgr


def _cfg(**over):
    cfg = {
        "grant": "client_credentials",
        "client_id": "mcp-gateway",
        "client_secret": "s3cr3t",
        "scope": "profile",
    }
    cfg.update(over)
    return cfg


# ---------------------------------------------------------------------------
# Grant classification
# ---------------------------------------------------------------------------


class TestIsM2MGrant:
    def test_client_credentials_selected(self):
        assert mo.is_m2m_grant({"grant": "client_credentials"}) is True

    def test_case_and_whitespace_insensitive(self):
        assert mo.is_m2m_grant({"grant": " Client_Credentials "}) is True

    def test_non_m2m_grants(self):
        assert mo.is_m2m_grant({"grant": "authorization_code"}) is False
        assert mo.is_m2m_grant({"grant": "private_key_jwt"}) is False  # dropped
        assert mo.is_m2m_grant({"client_id": "x"}) is False
        assert mo.is_m2m_grant(None) is False
        assert mo.is_m2m_grant({}) is False


# ---------------------------------------------------------------------------
# Provider construction
# ---------------------------------------------------------------------------


class TestClientCredentials:
    def test_builds_headless_provider(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        p = mo.build_client_credentials_provider("gw", "https://gw/mcp", _cfg())

        assert isinstance(p, ClientCredentialsOAuthProvider)
        assert p.context.client_metadata.grant_types == ["client_credentials"]
        assert p.context.client_metadata.scope == "profile"
        assert p._fixed_client_info.client_id == "mcp-gateway"
        assert p._fixed_client_info.client_secret == "s3cr3t"
        # Headless: base __init__ constructed with no redirect/callback handlers.
        assert p.context.redirect_handler is None
        assert p.context.callback_handler is None

    def test_default_auth_method_is_basic(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        p = mo.build_client_credentials_provider("gw", "https://gw/mcp", _cfg())
        assert p.context.client_metadata.token_endpoint_auth_method == "client_secret_basic"

    def test_auth_method_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        p = mo.build_client_credentials_provider(
            "gw", "https://gw/mcp", _cfg(token_endpoint_auth_method="client_secret_post")
        )
        assert p.context.client_metadata.token_endpoint_auth_method == "client_secret_post"

    def test_missing_client_secret_raises(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg = _cfg()
        del cfg["client_secret"]
        with pytest.raises(ValueError, match="client_secret"):
            mo.build_client_credentials_provider("gw", "https://gw/mcp", cfg)

    def test_missing_client_id_raises(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg = _cfg()
        del cfg["client_id"]
        with pytest.raises(ValueError, match="client_id"):
            mo.build_client_credentials_provider("gw", "https://gw/mcp", cfg)

    def test_bad_auth_method_raises(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        with pytest.raises(ValueError, match="token_endpoint_auth_method"):
            mo.build_client_credentials_provider(
                "gw", "https://gw/mcp", _cfg(token_endpoint_auth_method="mtls")
            )


class TestDispatch:
    def test_dispatch_client_credentials(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        p = mo.build_m2m_provider("gw", "https://gw/mcp", _cfg())
        assert type(p) is ClientCredentialsOAuthProvider

    def test_dispatch_unknown_grant_raises(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        with pytest.raises(ValueError, match="Unknown M2M oauth grant"):
            mo.build_m2m_provider("gw", "https://gw/mcp", {"grant": "authorization_code"})


class TestBuildOAuthAuthHeadless:
    def test_m2m_bypasses_non_interactive_guard(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        # Force a non-interactive env: the browser path would raise
        # OAuthNonInteractiveError; the M2M path must not.
        monkeypatch.setattr(mo, "_is_interactive", lambda: False)
        p = mo.build_oauth_auth("gw", "https://gw/mcp", _cfg())
        assert isinstance(p, ClientCredentialsOAuthProvider)


# ---------------------------------------------------------------------------
# Manager wiring
# ---------------------------------------------------------------------------


class TestManagerWiring:
    def test_build_provider_uses_sdk_provider(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(mo, "_is_interactive", lambda: False)
        entry = mgr._ProviderEntry(server_url="https://gw/mcp", oauth_config=_cfg())
        provider = mgr.get_manager()._build_provider("gw", entry)
        assert type(provider) is ClientCredentialsOAuthProvider

    def test_is_m2m(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        mgr.reset_manager_for_tests()
        manager = mgr.get_manager()
        # is_m2m reads the entry's oauth_config only — no provider build needed.
        manager._entries["gw"] = mgr._ProviderEntry(
            server_url="https://gw/mcp", oauth_config=_cfg()
        )
        manager._entries["browser"] = mgr._ProviderEntry(
            server_url="https://b/mcp", oauth_config={"client_id": "x"}
        )
        assert manager.is_m2m("gw") is True
        assert manager.is_m2m("browser") is False
        assert manager.is_m2m("never-seen") is False


# ---------------------------------------------------------------------------
# Auth-failure message: M2M vs interactive
# ---------------------------------------------------------------------------


class TestReauthMessage:
    def test_m2m_message_does_not_prompt_login(self):
        from tools.mcp_tool import _needs_reauth_error

        payload = json.loads(_needs_reauth_error("gw", m2m=True))
        assert payload["m2m"] is True
        assert payload["needs_reauth"] is True
        assert "Do NOT run" in payload["error"]
        assert "credential/config" in payload["error"]

    def test_interactive_message_prompts_login(self):
        from tools.mcp_tool import _needs_reauth_error

        payload = json.loads(_needs_reauth_error("gh", m2m=False))
        assert "m2m" not in payload
        assert "hermes mcp login gh" in payload["error"]
