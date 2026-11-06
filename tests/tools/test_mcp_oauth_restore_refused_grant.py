"""A re-login that fails or is cancelled restores the pre-attempt token snapshot. When the
authorization server has already refused that snapshot's refresh grant (``invalid_grant``), putting it
back left the server looking authenticated while it could never refresh again — measured three times
in two days on one profile. The rollback keeps the access token and drops the dead grant.
"""

import asyncio
import json

import pytest


def _tokens(access: str, refresh: str | None):
    from mcp.shared.auth import OAuthToken

    return OAuthToken(access_token=access, token_type="Bearer", expires_in=3600, refresh_token=refresh)


def _restored_tokens(storage) -> dict:
    return json.loads(storage._tokens_path().read_text(encoding="utf-8-sig"))


def test_rollback_drops_a_refresh_grant_the_authorization_server_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_oauth import HermesTokenStorage

    storage = HermesTokenStorage("cloud")
    asyncio.run(storage.set_tokens(_tokens("A1", "R-dead")))
    snapshot = storage.snapshot()
    storage.remove()  # what a re-login does before the browser step
    storage.mark_refresh_refused("R-dead")  # ... and the attempt never completed
    storage.restore(snapshot)

    restored = _restored_tokens(storage)
    assert restored["access_token"] == "A1", "the access token may still be inside its TTL"
    assert "refresh_token" not in restored, (
        "a refresh grant the authorization server refused must not be restored")


def test_rollback_keeps_a_refresh_grant_that_was_not_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_oauth import HermesTokenStorage

    storage = HermesTokenStorage("cloud")
    asyncio.run(storage.set_tokens(_tokens("A1", "R-live")))
    snapshot = storage.snapshot()
    storage.remove()
    storage.mark_refresh_refused("R-other")  # another grant, e.g. a sibling attempt's
    storage.restore(snapshot)

    assert _restored_tokens(storage)["refresh_token"] == "R-live"


def test_a_new_grant_clears_the_refusal_record(tmp_path, monkeypatch):
    """The record is about ONE grant: a successful authorization must not poison the next rollback."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_oauth import HermesTokenStorage

    storage = HermesTokenStorage("cloud")
    asyncio.run(storage.set_tokens(_tokens("A1", "R-dead")))
    storage.mark_refresh_refused("R-dead")
    assert storage.refused_refresh_token("R-dead")

    asyncio.run(storage.set_tokens(_tokens("A2", "R-new")))
    assert not storage.refused_refresh_token("R-dead")


@pytest.mark.parametrize("token", [None, ""])
def test_no_refusal_record_for_a_missing_refresh_token(tmp_path, monkeypatch, token):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_oauth import HermesTokenStorage

    storage = HermesTokenStorage("cloud")
    storage.mark_refresh_refused(token)
    assert not storage.refused_refresh_token("R-anything")
