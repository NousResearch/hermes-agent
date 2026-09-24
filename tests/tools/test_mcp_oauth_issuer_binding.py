"""Refresh tokens are bound to the authorization-server issuer that granted them.

The discovered authorization server for an MCP server can change (server migration, protected-resource
metadata edit, DNS takeover). A stored refresh token must never be sent to a different issuer.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("mcp")

from mcp.shared.auth import OAuthToken  # noqa: E402

from tools.mcp_oauth import HermesTokenStorage  # noqa: E402
from tools.mcp_oauth_provider import bind_issuer_from_context, enforce_refresh_token_issuer  # noqa: E402


def _token_file(tmp_path):
    return tmp_path / "mcp-tokens" / "srv.json"


def _stored(tmp_path, issuer):
    payload = {"access_token": "a", "token_type": "Bearer", "expires_in": 3600, "refresh_token": "r"}
    if issuer is not None:
        payload["hermes_issuer"] = issuer
    _token_file(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    _token_file(tmp_path).write_text(json.dumps(payload))
    storage = HermesTokenStorage("srv", hermes_home=tmp_path)
    tokens = asyncio.run(storage.get_tokens())
    assert tokens is not None
    return storage, tokens


def _context(storage, issuer, tokens=None):
    meta = SimpleNamespace(issuer=issuer) if issuer is not None else None
    return SimpleNamespace(storage=storage, oauth_metadata=meta, current_tokens=tokens)


def test_issuer_mismatch_strips_refresh_token_in_memory_and_on_disk(tmp_path):
    storage, tokens = _stored(tmp_path, "https://old.example.com")
    ctx = _context(storage, "https://evil.example.com", tokens)
    enforce_refresh_token_issuer(ctx)
    assert ctx.current_tokens.refresh_token is None
    assert ctx.current_tokens.access_token == "a"  # unexpired access token stays usable
    on_disk = json.loads(_token_file(tmp_path).read_text())
    assert "refresh_token" not in on_disk and on_disk["access_token"] == "a"


def test_matching_issuer_keeps_refresh_token_even_with_trailing_slash(tmp_path):
    storage, tokens = _stored(tmp_path, "https://as.example.com/")
    ctx = _context(storage, "https://as.example.com", tokens)
    enforce_refresh_token_issuer(ctx)
    assert ctx.current_tokens.refresh_token == "r"
    assert json.loads(_token_file(tmp_path).read_text())["refresh_token"] == "r"


def test_legacy_file_without_issuer_adopts_current_issuer_once(tmp_path):
    storage, tokens = _stored(tmp_path, None)
    enforce_refresh_token_issuer(_context(storage, "https://as.example.com", tokens))
    assert tokens.refresh_token == "r"
    assert json.loads(_token_file(tmp_path).read_text())["hermes_issuer"] == "https://as.example.com"
    # Now bound: a later issuer change is rejected.
    storage2 = HermesTokenStorage("srv", hermes_home=tmp_path)
    tokens2 = asyncio.run(storage2.get_tokens())
    assert tokens2 is not None
    enforce_refresh_token_issuer(_context(storage2, "https://evil.example.com", tokens2))
    assert tokens2.refresh_token is None


def test_set_tokens_stamps_bound_issuer_and_get_tokens_keeps_it_out_of_the_sdk_model(tmp_path):
    storage = HermesTokenStorage("srv", hermes_home=tmp_path)
    bind_issuer_from_context(_context(storage, "https://as.example.com"))
    asyncio.run(storage.set_tokens(OAuthToken(access_token="a", token_type="Bearer", expires_in=3600, refresh_token="r")))
    assert json.loads(_token_file(tmp_path).read_text())["hermes_issuer"] == "https://as.example.com"
    fresh = HermesTokenStorage("srv", hermes_home=tmp_path)
    tokens = asyncio.run(fresh.get_tokens())
    assert fresh.loaded_issuer == "https://as.example.com"
    assert not hasattr(tokens, "hermes_issuer")  # never leaks into the wire model
