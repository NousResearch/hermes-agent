"""Native filesystem durability and server-scoped OAuth publication recovery."""

import os
from pathlib import Path
import stat

import pytest

from tools import mcp_oauth_login as login
from tools.mcp_oauth import HermesTokenStorage


def test_recovery_marker_does_not_block_a_prefix_related_server(tmp_path):
    storage = HermesTokenStorage("service", hermes_home=tmp_path)
    token_dir = storage._tokens_path().parent
    token_dir.mkdir()
    (token_dir / ".oauth-login-service-other.pending").mkdir()
    login.require_settled_publication(storage)
    (token_dir / ".oauth-login-service.pending").mkdir()
    with pytest.raises(RuntimeError, match="recovery required"):
        login.require_settled_publication(storage)


@pytest.mark.platforms("posix")
def test_underlying_directory_fsync_failure_retains_exact_original_recovery(tmp_path, monkeypatch):
    storage = HermesTokenStorage("service", hermes_home=tmp_path)
    token_dir = storage._tokens_path().parent
    token_dir.mkdir()
    original = {"service.json": b"original-token", "service.client.json": b"original-client"}
    candidate = {"service.json": b"replacement-token", "service.client.json": b"replacement-client"}
    for name, raw in original.items():
        path = token_dir / name
        path.write_bytes(raw)
        path.chmod(0o600)
    published = False
    replace, fsync = os.replace, os.fsync

    def publish(source, target):
        nonlocal published
        result = replace(source, target)
        if Path(source).name == "service.json.next":
            published = True
        return result

    def durable(fd):
        if published and stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("Synthetic persistent directory-fsync failure")
        return fsync(fd)

    monkeypatch.setattr(os, "replace", publish)
    monkeypatch.setattr(os, "fsync", durable)
    with pytest.raises(RuntimeError, match="private recovery files retained"):
        login._publish(storage, candidate, original)
    assert login._read_state(storage) == original
    directories = list(token_dir.glob(".oauth-login-service.*"))
    assert len(directories) == 1
    assert stat.S_IMODE(directories[0].stat().st_mode) == 0o700
    assert {name: (directories[0] / (name + ".previous")).read_bytes() for name in original} == original
    with pytest.raises(RuntimeError, match="recovery required"):
        login.require_settled_publication(storage)
