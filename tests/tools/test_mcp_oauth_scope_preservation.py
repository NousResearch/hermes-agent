"""Regression tests for #93719 — explicit MCP OAuth scope overwritten by server
metadata during the SDK challenge loop.

Mechanism (traced on mcp 2.0.0, ``mcp/client/auth/oauth2.py`` Step 3 +
``mcp/client/auth/utils.py::get_client_metadata_scopes``):

* ``_build_client_metadata()`` correctly puts the user's configured
  ``oauth.scope`` into ``client_metadata.scope``.
* On the first 401 challenge, the SDK's ``async_auth_flow`` Step 3
  **unconditionally overwrites** ``client_metadata.scope`` with
  ``get_client_metadata_scopes(www_authenticate_scope, prm, as_meta, grants)``
  — a function that never sees the configured value. The overwrite then feeds
  the ``/authorize`` redirect.

Consequences (each pinned by a test below):
- A configured scope requesting scopes outside the advertised list is dropped.
- With no metadata at all, even a fully-explicit configuration requests nothing.
- The only surviving case is coincidence: configured scope exactly equal to
  what server metadata would produce.

Fix contract: when an explicit ``oauth.scope`` is configured for the server,
the effective requested scope must be built FROM that configuration — server
metadata may add to it (e.g. a WWW-Authenticate-required scope), but may not
replace it.

NOTE on the "pre-fix state" assertions below: they pin the *SDK's current*
behavior via the public ``get_client_metadata_scopes`` helper. If the mcp SDK
fixes scope preservation upstream, these tests will start failing WITHOUT any
Hermes change — that failure means "upgrade and re-evaluate this fix," not a
regression here. They are marked with the ``sdk_behavior_pin`` marker so they
can be deselected or re-baselined against a new SDK version in one place.
"""

from __future__ import annotations

import pytest

from mcp.client.auth.utils import get_client_metadata_scopes

CONFIGURED = "mcp:ea offline_access"

# Marker: tests asserting the installed SDK's (currently buggy) scope
# selection. Register in pyproject if the repo grows more of these.
sdk_behavior_pin = pytest.mark.sdk_behavior_pin


class _AS:
    """Authorization-server metadata advertising an overlapping-but-different set."""

    scopes_supported = ["mcp:ea", "openid", "profile", "email", "offline_access"]


class _ASForeign:
    """Server advertises entirely different scopes than the user configured."""

    scopes_supported = ["basic", "openid"]


class _ASNone:
    """Server publishes no scope metadata at all."""

    scopes_supported = None


@sdk_behavior_pin
def test_configured_scope_dropped_when_server_advertises_different_set():
    """A 401 with no WWW-Authenticate header replaces the configured scope with
    the advertised list — the user's explicit request never reaches /authorize."""
    selected = get_client_metadata_scopes(None, None, _AS(), ["authorization_code", "refresh_token"])

    assert "mcp:ea" in selected and "offline_access" in selected, (
        "sanity: advertised scopes flow through"
    )
    # The bug: this equals the advertised set, NOT the configured one. The fix
    # must make the configured scope the baseline; this assertion pins the
    # pre-fix behavior via the public helper so the regression is visible.
    assert set(selected.split()) != set(CONFIGURED.split()), (
        "pre-fix state: configured scope ignored, server list used verbatim"
    )


@sdk_behavior_pin
def test_configured_scope_dropped_when_server_metadata_has_no_scopes():
    """With no advertised scopes anywhere, the helper returns None — the
    explicitly configured scope is silently not requested at all."""
    selected = get_client_metadata_scopes(None, None, _ASNone(), ["authorization_code", "refresh_token"])

    assert selected is None or selected != CONFIGURED, (
        "pre-fix state: fully-explicit config dropped when server advertises nothing"
    )


@sdk_behavior_pin
def test_wa_challenge_narrows_configured_scope_without_offline_access():
    """When the challenge demands only 'mcp:ea' and AS metadata lacks
    offline_access, the SEP-2207 refresh-token augmentation cannot fire and the
    configured 'offline_access' request is lost -> no refresh token issued."""
    class _ASNoOffline:
        scopes_supported = ["mcp:ea", "openid"]

    selected = get_client_metadata_scopes("mcp:ea", None, _ASNoOffline(), ["authorization_code", "refresh_token"])

    assert "offline_access" not in selected.split(), (
        "pre-fix state: offline_access lost — server will not issue a refresh token"
    )
