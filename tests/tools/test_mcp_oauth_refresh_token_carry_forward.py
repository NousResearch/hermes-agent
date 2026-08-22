"""Tests for refresh_token preservation in ``HermesTokenStorage.set_tokens``.

RFC 6749 §6 makes a new refresh token optional on the
``grant_type=refresh_token`` exchange, and requires the client to replace its
stored token only "if the authorization server issued a new refresh token". A
response that omits ``refresh_token`` therefore obliges the client to keep the
one it already has.

``set_tokens`` used to violate that. It serialized with
``model_dump(mode="json", exclude_none=True)`` and wrote the result over the
token file unconditionally, so a refresh response without a ``refresh_token``
erased the stored one — not even leaving an explicit null behind.

Observed consequence, on a provider whose AS does not rotate (Asana returns
``refresh_token`` only on the ``authorization_code`` grant, and its access
tokens live one hour):

    T+0h    login, connector healthy
    T+1h    access token expires -> refresh succeeds -> stored refresh_token
            is destroyed
    T+2h    access token expires -> nothing to refresh with -> 401 -> the SDK
            falls back to a full authorization_code grant

Interactively that shows up as the OAuth browser dance repeating about one TTL
after every successful login. On a headless gateway there is no browser, so
the server is marked "failed initial OAuth authentication, not retrying
automatically" and its tools vanish for the life of the process.

The contract pinned here:

- A refresh response that omits ``refresh_token`` must not reduce the stored
  credential set.
- A refresh response that carries a new ``refresh_token`` must still replace
  the stored one (rotation keeps working).
- The carry-forward must land on the ``OAuthToken`` the caller passed in, not
  only on the serialized payload. The SDK's
  ``OAuthClientProvider._handle_refresh_response`` assigns
  ``context.current_tokens = token_response`` and then hands that same object
  to ``storage.set_tokens``, so repairing only the payload leaves the live
  provider with ``refresh_token=None``; ``context.can_refresh_token()`` is
  then False at the next expiry and the connector still dies inside the
  running process, recovering only on restart when ``_initialize`` re-reads
  the file.
"""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK 1.26.0+ required")


def _stored(tmp_path, server: str = "srv") -> dict:
    return json.loads((tmp_path / "mcp-tokens" / f"{server}.json").read_text())


class TestRefreshTokenCarryForward:
    def test_omitted_refresh_token_preserves_the_stored_one(self, tmp_path, monkeypatch):
        """A refresh response without refresh_token must not erase the stored one."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from mcp.shared.auth import OAuthToken
        from tools.mcp_oauth import HermesTokenStorage

        storage = HermesTokenStorage("srv")

        # authorization_code grant: carries a refresh token.
        asyncio.run(
            storage.set_tokens(
                OAuthToken(
                    access_token="access-1",
                    token_type="Bearer",
                    expires_in=3600,
                    refresh_token="refresh-1",
                )
            )
        )
        assert _stored(tmp_path)["refresh_token"] == "refresh-1"

        # refresh_token grant, non-rotating AS: no refresh_token in the response.
        asyncio.run(
            storage.set_tokens(
                OAuthToken(
                    access_token="access-2",
                    token_type="Bearer",
                    expires_in=3600,
                )
            )
        )

        on_disk = _stored(tmp_path)
        assert on_disk["refresh_token"] == "refresh-1", (
            "A refresh response that omits refresh_token must leave the stored "
            "one intact (RFC 6749 §6); otherwise the next expiry has nothing to "
            "refresh with and forces a full browser re-authorization."
        )
        assert on_disk["access_token"] == "access-2", (
            "The new access token must still be persisted."
        )

    def test_carry_forward_also_repairs_the_passed_token_object(self, tmp_path, monkeypatch):
        """The live OAuthToken must be repaired, not just the serialized payload.

        The SDK hands ``set_tokens`` the very object it has just assigned to
        ``context.current_tokens``. Repairing only the JSON payload fixes the
        file while leaving the running provider unable to refresh, so the
        connector still dies at the next expiry and only recovers on restart.
        """
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from mcp.shared.auth import OAuthToken
        from tools.mcp_oauth import HermesTokenStorage

        storage = HermesTokenStorage("srv")
        asyncio.run(
            storage.set_tokens(
                OAuthToken(
                    access_token="access-1",
                    token_type="Bearer",
                    expires_in=3600,
                    refresh_token="refresh-1",
                )
            )
        )

        refreshed = OAuthToken(
            access_token="access-2", token_type="Bearer", expires_in=3600
        )
        asyncio.run(storage.set_tokens(refreshed))

        assert refreshed.refresh_token == "refresh-1", (
            "set_tokens must restore refresh_token on the OAuthToken it was "
            "given, because that object is the SDK's context.current_tokens. "
            "Without this, context.can_refresh_token() is False at the next "
            "expiry even though the file on disk is correct."
        )

    def test_rotated_refresh_token_replaces_the_stored_one(self, tmp_path, monkeypatch):
        """A rotating AS must still be able to replace the stored refresh token."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from mcp.shared.auth import OAuthToken
        from tools.mcp_oauth import HermesTokenStorage

        storage = HermesTokenStorage("srv")
        asyncio.run(
            storage.set_tokens(
                OAuthToken(
                    access_token="access-1",
                    token_type="Bearer",
                    expires_in=3600,
                    refresh_token="refresh-1",
                )
            )
        )

        rotated = OAuthToken(
            access_token="access-2",
            token_type="Bearer",
            expires_in=3600,
            refresh_token="refresh-2",
        )
        asyncio.run(storage.set_tokens(rotated))

        assert _stored(tmp_path)["refresh_token"] == "refresh-2", (
            "An explicitly returned refresh_token must replace the stored one, "
            "or rotation-based providers would be pinned to a consumed token."
        )
        assert rotated.refresh_token == "refresh-2", (
            "Carry-forward must not overwrite a genuinely rotated token."
        )

    def test_consecutive_omitting_refreshes_keep_the_refresh_token(self, tmp_path, monkeypatch):
        """The token must survive repeated refreshes, not just the first.

        The original bug was only visible on the *second* expiry: the first
        refresh succeeded and looked healthy while silently destroying the
        credential it would need an hour later.
        """
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from mcp.shared.auth import OAuthToken
        from tools.mcp_oauth import HermesTokenStorage

        storage = HermesTokenStorage("srv")
        asyncio.run(
            storage.set_tokens(
                OAuthToken(
                    access_token="access-0",
                    token_type="Bearer",
                    expires_in=3600,
                    refresh_token="refresh-1",
                )
            )
        )

        for i in range(1, 6):
            token = OAuthToken(
                access_token=f"access-{i}", token_type="Bearer", expires_in=3600
            )
            asyncio.run(storage.set_tokens(token))
            assert token.refresh_token == "refresh-1", f"lost in-memory at refresh {i}"
            assert _stored(tmp_path)["refresh_token"] == "refresh-1", (
                f"lost on disk at refresh {i}"
            )

    def test_first_write_without_any_stored_refresh_token_is_clean(self, tmp_path, monkeypatch):
        """A provider that issues no refresh token at all must still persist.

        There is nothing to carry forward here; the write must succeed and must
        not fabricate the field.
        """
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from mcp.shared.auth import OAuthToken
        from tools.mcp_oauth import HermesTokenStorage

        storage = HermesTokenStorage("srv")
        asyncio.run(
            storage.set_tokens(
                OAuthToken(access_token="access-1", token_type="Bearer", expires_in=3600)
            )
        )

        on_disk = _stored(tmp_path)
        assert on_disk["access_token"] == "access-1"
        assert "refresh_token" not in on_disk

    def test_carry_forward_is_scoped_per_server(self, tmp_path, monkeypatch):
        """One server's refresh token must never leak into another's file."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from mcp.shared.auth import OAuthToken
        from tools.mcp_oauth import HermesTokenStorage

        asyncio.run(
            HermesTokenStorage("alpha").set_tokens(
                OAuthToken(
                    access_token="a",
                    token_type="Bearer",
                    expires_in=3600,
                    refresh_token="alpha-refresh",
                )
            )
        )

        beta_token = OAuthToken(
            access_token="b", token_type="Bearer", expires_in=3600
        )
        asyncio.run(HermesTokenStorage("beta").set_tokens(beta_token))

        assert beta_token.refresh_token is None
        assert "refresh_token" not in _stored(tmp_path, "beta")
        assert _stored(tmp_path, "alpha")["refresh_token"] == "alpha-refresh"


class TestCarryForwardDegradesGracefully:
    """A damaged previous token file must not block persisting a good new one.

    The carry-forward reads the stored token before writing. ``get_tokens``
    tolerates a missing file and a schema-invalid payload, but not every shape:
    ``_read_json`` returns whatever ``json.loads`` produced, so a file holding a
    JSON list, or a string ``expires_at``, raises out of ``get_tokens`` before
    the ``model_validate`` guard. Refresh must still land in that case —
    degrading to the previous overwrite behaviour is strictly better than
    failing the write and leaving the SDK believing it persisted a token.
    """

    def _seed(self, tmp_path, payload: str, server: str = "srv") -> None:
        token_dir = tmp_path / "mcp-tokens"
        token_dir.mkdir(parents=True, exist_ok=True)
        (token_dir / f"{server}.json").write_text(payload)

    @pytest.mark.parametrize(
        "payload,label",
        [
            ('[1, 2, 3]', "json list instead of an object"),
            ('"just-a-string"', "json string instead of an object"),
            ('{"access_token": "a", "expires_at": "not-a-number"}', "non-numeric expires_at"),
        ],
    )
    def test_damaged_previous_file_still_persists_the_new_token(
        self, tmp_path, monkeypatch, payload, label
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from mcp.shared.auth import OAuthToken
        from tools.mcp_oauth import HermesTokenStorage

        self._seed(tmp_path, payload)
        storage = HermesTokenStorage("srv")

        asyncio.run(
            storage.set_tokens(
                OAuthToken(access_token="fresh", token_type="Bearer", expires_in=3600)
            )
        )

        on_disk = _stored(tmp_path)
        assert on_disk["access_token"] == "fresh", (
            f"a {label} in the stored file must not prevent the refreshed access "
            "token from being written"
        )
