"""Attempt isolation, partial-clear recovery and durable-quarantine recovery contracts."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
from pathlib import Path

import pytest

from tools.mcp_oauth import HermesTokenStorage


def seed(home):
    storage = HermesTokenStorage("team", hermes_home=home)
    storage._token_dir.mkdir(parents=True)
    storage._tokens_path().write_text(json.dumps({"access_token": "OLD", "token_type": "Bearer"}))
    storage._client_info_path().write_text(json.dumps({"client_id": "ORIGINAL"}))
    return storage


def test_failed_clear_restores_token_without_retrying_persistently_failing_unlink(tmp_path, monkeypatch):
    storage = seed(tmp_path)
    original = storage.snapshot()
    unlink = Path.unlink

    def cannot_remove_client(path, **kwargs):
        if path == storage._client_info_path():
            raise PermissionError("client cannot be unlinked")
        return unlink(path, **kwargs)

    monkeypatch.setattr(Path, "unlink", cannot_remove_client)
    with pytest.raises(PermissionError):
        storage.begin_reauthorization()
    assert storage.snapshot() == original


def test_unreadable_snapshot_aborts_before_deleting_any_state(tmp_path, monkeypatch):
    storage = seed(tmp_path)
    original = storage.snapshot()
    read = Path.read_bytes

    def unreadable(path):
        if path == storage._client_info_path():
            raise PermissionError("client unreadable")
        return read(path)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", unreadable)
        with pytest.raises(PermissionError):
            storage.begin_reauthorization()
    assert storage.snapshot() == original


def test_explicit_login_finishes_quarantine_after_revocation_fence_timeout(tmp_path, monkeypatch):
    from tools import mcp_oauth
    from tools.mcp_oauth_manager import MCPOAuthManager
    storage = seed(tmp_path)
    manager = MCPOAuthManager()
    hold_fence = mcp_oauth.hold_refresh_fence

    @contextmanager
    def unavailable(path, **kwargs):
        if path == storage._tokens_path():
            raise mcp_oauth.RefreshFenceTimeout("held by peer")
        with hold_fence(path, **kwargs):
            yield

    with monkeypatch.context() as patch:
        patch.setattr(mcp_oauth, "hold_refresh_fence", unavailable)
        with pytest.raises(mcp_oauth.RefreshFenceTimeout):
            manager.remove("team", hermes_home=tmp_path, detach_participants=True)
    assert storage._tokens_path().exists()
    assert storage.rebuild_blocked()
    manager.unblock("team", hermes_home=tmp_path)
    assert not storage.rebuild_blocked()
    assert not storage._tokens_path().exists()
    assert not storage._client_info_path().exists()


def test_staged_provider_never_enters_the_live_profile_cache(tmp_path, monkeypatch):
    from tools.mcp_oauth import force_interactive_oauth
    from tools.mcp_oauth_manager import MCPOAuthManager
    from tools.mcp_oauth_transaction import OAuthAuthorization
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    storage = seed(tmp_path)
    cfg = {"url": "https://example.com/mcp", "auth": "oauth"}
    manager = MCPOAuthManager()
    original = manager.get_or_build_provider("team", cfg["url"], None)
    with OAuthAuthorization(storage, str(tmp_path), "team", cfg), force_interactive_oauth():
        staged = manager.get_or_build_provider("team", cfg["url"], None)
        assert staged is not original
        assert staged.context.storage.pool_path == storage.pool_path
        assert staged.context.storage._tokens_path() != storage._tokens_path()
        with ThreadPoolExecutor() as executor:
            assert executor.submit(manager.get_or_build_provider, "team", cfg["url"], None).result() is original
    assert manager.get_or_build_provider("team", cfg["url"], None) is original
    assert asyncio.run(original.context.storage.get_tokens()).access_token == "OLD"


def test_overlapping_quarantines_survive_one_writer_aborting(tmp_path):
    storage = seed(tmp_path)
    first, second = storage.quarantine(), storage.quarantine()
    storage.cancel_quarantine(second)
    assert storage.rebuild_blocked()
    storage.cancel_quarantine(first)
    assert not storage.rebuild_blocked()


def test_explicit_login_cannot_recover_a_provisional_config_quarantine(tmp_path):
    from tools.mcp_oauth import OAuthAuthorityUnavailableError
    storage = seed(tmp_path)
    original = storage.snapshot()
    owner = storage.quarantine()
    with pytest.raises(OAuthAuthorityUnavailableError):
        storage.unblock_rebuild()
    storage.cancel_quarantine(owner)
    assert storage.snapshot() == original


@pytest.mark.parametrize("contents", ["broken", "{}", "null", '"text"'])
def test_invalid_quarantine_cannot_be_overwritten_and_cleared(tmp_path, contents):
    from tools.mcp_oauth import OAuthAuthorityUnavailableError
    storage = seed(tmp_path)
    storage._quarantine_path().write_text(contents)
    with pytest.raises(OAuthAuthorityUnavailableError):
        storage.quarantine()
    assert storage._quarantine_path().read_text() == contents
    assert storage.rebuild_blocked()


def test_failure_after_config_commit_keeps_new_bytes_quarantined_instead_of_restoring_old(tmp_path, monkeypatch):
    from tools import mcp_oauth
    from tools.mcp_oauth_transaction import OAuthAuthorization
    storage = seed(tmp_path)
    cfg = {"url": "https://new.example.com", "auth": "oauth"}

    def cannot_advance(pool):
        raise OSError("epoch write failed")

    with OAuthAuthorization(storage, str(tmp_path), "team", cfg) as attempt:
        attempt.storage._tokens_path().write_text(json.dumps({"access_token": "NEW", "token_type": "Bearer"}))
        monkeypatch.setattr(mcp_oauth, "advance_pool_epoch", cannot_advance)
        with pytest.raises(OSError, match="epoch write failed"), attempt.publish():
            pass  # the caller's config save succeeded
    assert json.loads(storage._tokens_path().read_text())["access_token"] == "NEW"
    assert not storage.has_cached_tokens()
    assert storage.rebuild_blocked()
