"""Unit tests for guest_mode_tool — mint_token, resolve_token."""

import time

import tools.guest_mode_tool as gmt


def _clear_store():
    gmt._TOKEN_STORE.clear()


class TestTokenStore:
    def setup_method(self):
        _clear_store()

    def test_mint_and_resolve(self):
        token = gmt.mint_token("fid123", "video")
        entry = gmt.resolve_token(token)
        assert entry is not None
        assert entry["file_id"] == "fid123"
        assert entry["media_kind"] == "video"

    def test_resolve_does_not_consume_token(self):
        token = gmt.mint_token("fid123", "audio")
        gmt.resolve_token(token)
        assert gmt.resolve_token(token) is not None  # still resolvable

    def test_resolve_unknown_token_returns_none(self):
        assert gmt.resolve_token("nonexistent") is None

    def test_token_expiry(self):
        token = gmt.mint_token("fid", "video")
        gmt._TOKEN_STORE[token]["expires_at"] = time.monotonic() - 1
        assert gmt.resolve_token(token) is None

    def test_expired_token_removed_from_store(self):
        token = gmt.mint_token("fid", "video")
        gmt._TOKEN_STORE[token]["expires_at"] = time.monotonic() - 1
        gmt.resolve_token(token)
        assert token not in gmt._TOKEN_STORE

    def test_multiple_tokens_are_independent(self):
        t1 = gmt.mint_token("fid1", "video")
        t2 = gmt.mint_token("fid2", "audio")
        assert gmt.resolve_token(t1)["file_id"] == "fid1"
        assert gmt.resolve_token(t2)["file_id"] == "fid2"
        assert len(gmt._TOKEN_STORE) == 2  # both still present

    def test_ttl_is_ten_minutes(self):
        token = gmt.mint_token("fid", "document")
        entry = gmt._TOKEN_STORE[token]
        assert entry["expires_at"] - time.monotonic() > 590  # ~10 min
