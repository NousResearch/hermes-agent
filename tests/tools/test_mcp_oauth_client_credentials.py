"""Tests for the headless client_credentials (M2M) MCP OAuth grant.

Covers the ``oauth.grant: client_credentials`` path in ``tools/mcp_oauth.py``
and ``tools/mcp_oauth_manager.py``: it wires the MCP SDK's headless
``ClientCredentialsOAuthProvider`` so a daemon can authenticate to an
OAuth-fronted MCP gateway with no browser, no callback, and no interactive
re-auth.
"""

from __future__ import annotations

import json
import logging

import pytest
from mcp.client.auth.extensions.client_credentials import ClientCredentialsOAuthProvider

from tools import mcp_oauth as mo
from tools import mcp_oauth_manager as mgr


@pytest.fixture
def reset_manager():
    """Isolate the process-global MCPOAuthManager singleton.

    Reset on both sides: entries registered here would otherwise outlive the
    test and leak into the rest of the session.
    """
    mgr.reset_manager_for_tests()
    yield
    mgr.reset_manager_for_tests()


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
        # Spelling out the interactive grant is allowed: it means "the default".
        assert mo.is_m2m_grant({"grant": "authorization_code"}) is False
        assert mo.is_m2m_grant({"client_id": "x"}) is False
        assert mo.is_m2m_grant(None) is False
        assert mo.is_m2m_grant({}) is False

    @pytest.mark.parametrize("bad", ["client-credentials", "clientcredentials",
                                     "private_key_jwt", "device_code"])
    def test_unknown_grant_raises_instead_of_falling_back(self, bad):
        """A typo must not silently become the interactive flow.

        Falling through would answer a headless deployment with "run
        `hermes mcp login` interactively first" — advice impossible to follow
        on the very machine the operator is configuring.
        """
        with pytest.raises(ValueError, match="Unknown MCP OAuth grant"):
            mo.is_m2m_grant({"grant": bad})


# ---------------------------------------------------------------------------
# SDK generation compatibility
# ---------------------------------------------------------------------------


class TestScopeKwargCompat:
    """The scope keyword changed name between the two SDK generations.

    mcp 1.x took ``scopes=``; mcp 2.0 renamed it to ``scope=``. Passing the
    wrong spelling is a TypeError at construction, so on a headless
    deployment every M2M server would fail to authenticate at startup. These
    pins are what makes an SDK bump a visible test failure instead of a live
    outage.
    """

    def test_kwarg_follows_the_installed_sdk_signature(self):
        import inspect

        params = inspect.signature(
            ClientCredentialsOAuthProvider.__init__
        ).parameters
        expected = "scopes" if "scopes" in params else "scope"

        kwargs = mo._client_credentials_scope_kwarg("profile")

        assert kwargs == {expected: "profile"}
        # Whatever the spelling, it must be one the SDK actually accepts.
        assert expected in params

    def test_none_scope_is_still_passed_through(self):
        kwargs = mo._client_credentials_scope_kwarg(None)
        assert list(kwargs.values()) == [None]

    def test_configured_scope_reaches_the_client_metadata(self, monkeypatch, tmp_path):
        """End-to-end pin: the config scope must survive the kwarg translation."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg = _cfg()
        cfg["scope"] = "gateway.read gateway.write"

        p = mo.build_client_credentials_provider("gw", "https://gw/mcp", cfg)

        assert p.context.client_metadata.scope == "gateway.read gateway.write"
        assert p._fixed_client_info.scope == "gateway.read gateway.write"
        assert p._configured_scope == "gateway.read gateway.write"


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

    @pytest.mark.parametrize("bad", ["mtls", "private_key_jwt", "none"])
    def test_bad_auth_method_raises(self, monkeypatch, tmp_path, bad):
        """Only the two supported methods are accepted.

        Matched on Hermes' own wording: pydantic also rejects some values, and
        its ValidationError subclasses ValueError with the field name in the
        message — so a loose match would pass even with our check deleted.
        ``private_key_jwt`` and ``none`` are accepted by pydantic, so only our
        check can reject them.
        """
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        with pytest.raises(
            ValueError, match=r"must be 'client_secret_basic' or 'client_secret_post'"
        ):
            mo.build_client_credentials_provider(
                "gw", "https://gw/mcp", _cfg(token_endpoint_auth_method=bad)
            )

    def test_unresolved_env_placeholder_is_rejected(self, monkeypatch, tmp_path):
        """An unset ${VAR} must never be sent to the authorization server.

        Interpolation leaves an unset reference as its own literal, which is
        truthy — so the emptiness check above does not catch it and the
        placeholder would be base64'd into the Basic header, disclosing the
        variable name off-host for an opaque invalid_client.
        """
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg = _cfg(client_secret="${MCP_GW_CLIENT_SECRET}")
        with pytest.raises(ValueError, match="unresolved placeholder"):
            mo.build_client_credentials_provider("gw", "https://gw/mcp", cfg)

    def test_raises_when_sdk_extension_missing(self, monkeypatch, tmp_path):
        """On an SDK without the extension: fail loudly, never return None.

        Returning None would leave the caller connecting with no Authorization
        header at all — which fails later and further away, or silently
        "succeeds" against a server that lists tools unauthenticated.

        ``_HermesClientCredentialsProvider`` only exists when the extension
        imported, so the guard must also come *before* the constructor is
        touched; deleting the attribute makes a mis-ordered guard raise
        NameError instead of our error.
        """
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(mo, "_M2M_AVAILABLE", False)
        monkeypatch.delattr(mo, "_HermesClientCredentialsProvider", raising=False)

        with pytest.raises(mo.OAuthGrantUnavailableError, match=r"mcp>=1\.26\.0"):
            mo.build_client_credentials_provider("gw", "https://gw/mcp", _cfg())


class TestDispatch:
    def test_dispatch_client_credentials(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        p = mo.build_m2m_provider("gw", "https://gw/mcp", _cfg())
        # Our thin subclass — still the SDK provider, only the
        # client_secret_post body is corrected. See _HermesClientCredentialsProvider.
        assert type(p) is mo._HermesClientCredentialsProvider
        assert isinstance(p, ClientCredentialsOAuthProvider)

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
        assert type(provider) is mo._HermesClientCredentialsProvider
        assert isinstance(provider, ClientCredentialsOAuthProvider)

    def test_is_m2m(self, monkeypatch, tmp_path, reset_manager):
        """Register through the public path, then read back.

        Deliberately NOT seeding ``_entries`` directly: entries are keyed by
        (hermes_home, server_name) via ``_key``, and a test that both writes
        and reads through ``_key`` only proves ``is_m2m`` agrees with itself.
        Going through ``get_or_build_provider`` is what makes it catch a
        divergence between the code that registers entries and the code that
        looks them up — the exact shape of the bug this method already had.
        """
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        # Interactive so the browser server registers too; constructing its
        # provider opens nothing (a browser would only open on a request).
        monkeypatch.setattr(mo, "_is_interactive", lambda: True)
        manager = mgr.get_manager()

        manager.get_or_build_provider("gw", "https://gw/mcp", _cfg())
        manager.get_or_build_provider(
            "browser", "https://b/mcp", {"client_id": "x"}
        )

        assert manager.is_m2m("gw") is True
        assert manager.is_m2m("browser") is False
        assert manager.is_m2m("never-seen") is False

    def test_auth_failure_message_is_wired_to_the_registered_grant(
        self, monkeypatch, tmp_path, reset_manager
    ):
        """The manager lookup must actually drive the message the model sees.

        ``_needs_reauth_error`` is easy to test with a hand-passed literal,
        which proves nothing about production: what matters is that
        ``_handle_auth_error_and_retry`` asks the manager and gets the right
        answer for a server registered the normal way.
        """
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(mo, "_is_interactive", lambda: True)
        from tools.mcp_tool import _needs_reauth_error

        manager = mgr.get_manager()
        manager.get_or_build_provider("gw", "https://gw/mcp", _cfg())
        manager.get_or_build_provider(
            "browser", "https://b/mcp", {"client_id": "x"}
        )

        m2m = json.loads(
            _needs_reauth_error("gw", m2m=manager.is_m2m("gw"))
        )
        interactive = json.loads(
            _needs_reauth_error("browser", m2m=manager.is_m2m("browser"))
        )
        assert m2m["m2m"] is True
        assert "hermes mcp login" not in m2m["error"]
        assert "m2m" not in interactive
        assert "hermes mcp login browser" in interactive["error"]


# ---------------------------------------------------------------------------
# Auth-failure message: M2M vs interactive
# ---------------------------------------------------------------------------


class TestReauthMessage:
    def test_m2m_message_does_not_prompt_login(self):
        from tools.mcp_tool import _needs_reauth_error

        payload = json.loads(_needs_reauth_error("gw", m2m=True))
        assert payload["m2m"] is True
        assert payload["needs_reauth"] is True
        assert "credential/config" in payload["error"]
        # The M2M branch must never send the model down the re-auth path: a
        # fresh mint produces the same rejected token.
        assert "Re-authenticating will NOT help" in payload["error"]
        assert "hermes mcp login" not in payload["error"]

    def test_interactive_message_prompts_login(self):
        from tools.mcp_tool import _needs_reauth_error

        payload = json.loads(_needs_reauth_error("gh", m2m=False))
        assert "m2m" not in payload
        assert "hermes mcp login gh" in payload["error"]
