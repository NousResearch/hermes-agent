"""Real registry cleanup contracts for rejected claims and failed creation."""

import sqlite3
import subprocess

import pytest

from tests.tools.test_docker_environment import _make_dummy_env
from tests.tools.test_docker_environment_reuse_candidates import _stateful_docker
from tools.environments import docker as docker_env
from tools.environments.docker_reuse_owners import ReuseOwners


def test_rejected_claims_remove_token_locks_but_keep_container_guard(tmp_path):
    store = ReuseOwners(tmp_path)
    lease = store.reserve("hermes-box", {"hermes-profile": "profile"}, "fp")
    store.finish("hermes-box", "cid")
    store.release(lease)
    starts = []
    with store._container_guard("hermes-box") as guard:
        inode = guard.path.stat().st_ino
        for _ in range(3):
            assert store.claim("cid", lambda: starts.append(True) or True) is None
        assert starts == []
        with store.transaction() as db:
            assert db.execute("SELECT token FROM owners").fetchall() == []
        assert list(store.path.parent.glob("*.lock")) == [guard.path]
        assert guard.path.stat().st_ino == inode
    claimed = store.claim("cid", lambda: True)
    assert claimed is not None
    try:
        assert claimed.lock.path.is_file()
        with store.transaction() as db:
            assert db.execute("SELECT token FROM owners").fetchall() == [(claimed.token,)]
        assert store.claim("missing", lambda: True) is None
    finally:
        store.release(claimed)


@pytest.mark.parametrize("removal_fault", ["timeout", "nonzero"])
def test_failed_finish_and_removal_remain_reapable(monkeypatch, removal_fault):
    containers, calls = _stateful_docker(monkeypatch)
    env = _make_dummy_env(persistent_filesystem=True)
    store = env._reuse_owners
    env.cleanup()
    assert env.wait_for_cleanup()
    original_containers = set(containers)
    run = docker_env.subprocess.run

    with store.transaction() as db:
        db.execute("CREATE TRIGGER reject_finish BEFORE UPDATE OF container_id ON containers "
                   "BEGIN SELECT RAISE(ABORT, 'finish failed'); END")

    def docker(cmd, **kwargs):
        if cmd[1:3] == ["rm", "-f"]:
            calls.append(list(cmd))
            if removal_fault == "timeout":
                raise subprocess.TimeoutExpired(cmd, 10)
            return subprocess.CompletedProcess(cmd, 1, "", "removal failed")
        return run(cmd, **kwargs)

    monkeypatch.setattr(docker_env.subprocess, "run", docker)
    with pytest.raises(sqlite3.IntegrityError, match="finish failed"):
        env._docker_run(env.cwd)
    surviving = set(containers) - original_containers
    assert len(surviving) == 1
    cid = surviving.pop()
    name = containers[cid]["name"]
    assert containers[cid]["state"] == "running"
    with store.transaction() as db:
        assert db.execute("SELECT container_id, retired FROM containers WHERE name=?", (name,)).fetchone() == (None, 0)
        assert db.execute("SELECT token FROM owners").fetchall() == []
    assert not [path for path in store.path.parent.glob("*.lock") if not path.name.startswith("container-")]
    with store.transaction() as db:
        db.execute("DROP TRIGGER reject_finish")

    def recovered_docker(cmd, **kwargs):
        # The shared fixture resolves IDs only; real Docker also resolves names.
        if cmd[-1] == name:
            cmd = [*cmd[:-1], cid]
        return run(cmd, **kwargs)

    monkeypatch.setattr(docker_env.subprocess, "run", recovered_docker)
    assert store.reap(env._docker_exe, 0, None) == 1
    assert set(containers) == original_containers
    with store.transaction() as db:
        assert db.execute("SELECT retired FROM containers WHERE name=?", (name,)).fetchone() == (1,)
        assert db.execute("SELECT token FROM owners").fetchall() == []
    assert store.reap(env._docker_exe, 0, None) == 0


def test_failed_creation_release_rollback_preserves_owner_evidence(tmp_path):
    store = ReuseOwners(tmp_path)
    lease = store.reserve("hermes-box", {"hermes-profile": "profile"}, "fp")
    with store.transaction() as db:
        db.execute("CREATE TRIGGER reject_release BEFORE DELETE ON owners "
                   "BEGIN SELECT RAISE(ABORT, 'release failed'); END")
    with pytest.raises(sqlite3.IntegrityError, match="release failed"):
        store.release(lease, failed_creation=True)
    assert lease.lock.path.is_file()
    with store.transaction() as db:
        assert db.execute("SELECT token FROM owners").fetchall() == [(lease.token,)]
        assert db.execute("SELECT superseded FROM containers").fetchone() == (None,)
        db.execute("DROP TRIGGER reject_release")
    store.release(lease, failed_creation=True)
    assert not lease.lock.path.exists()
    with store.transaction() as db:
        assert db.execute("SELECT token FROM owners").fetchall() == []
        assert db.execute("SELECT superseded FROM containers").fetchone()[0] is not None
