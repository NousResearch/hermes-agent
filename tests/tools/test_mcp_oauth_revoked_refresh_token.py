"""A rejected refresh must not leave the dead refresh token on disk (#150367).

The SDK's ``OAuthContext.clear_tokens()`` only forgets the pair in memory, so after a definite
``invalid_grant`` the same revoked refresh token stayed in ``mcp-tokens/<server>.json`` and every
new provider — the next gateway request, a cron run, a new agent session — loaded it and sent it
again. The reported install issued 1,411 rejected refreshes in 25 hours that way. The provider now
drops the token on disk while it still holds the refresh fence, and only when the response actually
says the grant is dead: a WAF's HTML block page or a 5xx is not that.
"""

import asyncio
import json

import pytest

pytest.importorskip("mcp")

import httpx2

from tools.mcp_oauth import HermesTokenStorage
from tools.mcp_oauth_provider import HermesProviderMixin

_TOKEN_URL = "https://as.example.com/token"


class SimpleContext:
    """The slice of the SDK's auth context this handler touches."""

    def __init__(self, storage, tokens):
        self.storage = storage
        self.current_tokens = tokens

    def update_token_expiry(self, tokens):
        self.current_tokens = tokens

    def clear_tokens(self):
        self.current_tokens = None

    def is_token_valid(self):
        return False


class _Provider(HermesProviderMixin):
    """A mixin instance with the context and refresh-recovery seam a real provider would own."""

    context: SimpleContext


def _token_file(tmp_path):
    return tmp_path / "mcp-tokens" / "srv.json"


def _context_with_stored_tokens(tmp_path):
    _token_file(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    _token_file(tmp_path).write_text(json.dumps({
        "access_token": "a",
        "token_type": "Bearer",
        "expires_in": 3600,
        "refresh_token": "r",
        "hermes_issuer": "https://as.example.com",
    }))
    storage = HermesTokenStorage("srv", hermes_home=tmp_path)
    tokens = asyncio.run(storage.get_tokens())
    assert tokens is not None
    return SimpleContext(storage, tokens)


def _provider(tmp_path, *, peer_recovered):
    provider = _Provider.__new__(_Provider)
    provider.context = _context_with_stored_tokens(tmp_path)

    async def _reload_tokens_after_refresh_failure():
        return peer_recovered

    provider._hermes_reload_tokens_after_refresh_failure = _reload_tokens_after_refresh_failure
    return provider


def _response(status, **body):
    return httpx2.Response(status, request=httpx2.Request("POST", _TOKEN_URL), **body)


def _on_disk(tmp_path):
    return json.loads(_token_file(tmp_path).read_text())


def test_a_revoked_grant_drops_the_refresh_token_from_disk(tmp_path):
    provider = _provider(tmp_path, peer_recovered=False)

    accepted = asyncio.run(provider._hermes_handle_refresh_response(
        _response(400, json={"error": "invalid_grant", "error_description": "grant revoked"})))

    assert accepted is False
    assert provider.context.current_tokens is None  # the session still ends, as before
    on_disk = _on_disk(tmp_path)
    assert "refresh_token" not in on_disk  # the dead token cannot be re-sent by the next provider
    assert "hermes_issuer" not in on_disk
    assert on_disk["access_token"] == "a"  # an unexpired access token stays usable
    # What the reported loop actually did: every new provider loaded the pair again. After the strip
    # the load yields no refresh token, so the SDK answers ``can_refresh_token() is False``.
    reloaded = asyncio.run(HermesTokenStorage("srv", hermes_home=tmp_path).get_tokens())
    assert reloaded is not None and reloaded.refresh_token is None


@pytest.mark.parametrize("status,body", [
    (400, {"json": {"error": "invalid_client"}}),
    (503, {"json": {"error": "temporarily_unavailable"}}),
    (403, {"text": "<html><body>Request blocked</body></html>"}),
])
def test_an_unclear_refresh_failure_leaves_the_stored_token_alone(tmp_path, status, body):
    provider = _provider(tmp_path, peer_recovered=False)

    accepted = asyncio.run(provider._hermes_handle_refresh_response(_response(status, **body)))

    assert accepted is False
    assert _on_disk(tmp_path)["refresh_token"] == "r"


def test_a_peer_rotated_pair_is_recovered_before_any_strip(tmp_path):
    provider = _provider(tmp_path, peer_recovered=True)

    accepted = asyncio.run(provider._hermes_handle_refresh_response(
        _response(400, json={"error": "invalid_grant"})))

    assert accepted is True
    assert _on_disk(tmp_path)["refresh_token"] == "r"  # the live, peer-written pair survives
