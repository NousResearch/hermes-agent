"""Docker cleanup filter and creation ownership regressions for #85342."""

import sqlite3
import subprocess

import pytest

from hermes_cli.profiles import validate_profile_name
from tests.tools.test_docker_environment import _make_dummy_env
from tests.tools.test_docker_environment_reuse_candidates import _stateful_docker
from tools.environments import docker as docker_env


@pytest.mark.parametrize("length", [63, 64])
def test_valid_profile_filter_reaps_its_superseded_container(monkeypatch, length):
    name = "p" * length
    validate_profile_name(name)
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: name)
    containers, _ = _stateful_docker(monkeypatch)
    first = _make_dummy_env(persistent_filesystem=True, memory=1024)
    stale = first._container_id
    first.cleanup()
    assert first.wait_for_cleanup()
    successor = _make_dummy_env(persistent_filesystem=True, memory=2048)
    try:
        assert docker_env.reap_orphan_containers(max_age_seconds=0, profile_filter=name) == 1
        assert stale not in containers
        assert successor._container_id in containers
    finally:
        successor.cleanup()
        assert successor.wait_for_cleanup()


def test_failed_label_staging_does_not_keep_an_owner_receipt(monkeypatch, tmp_path):
    containers, _ = _stateful_docker(monkeypatch)
    env = _make_dummy_env(persistent_filesystem=True)
    store = env._reuse_owners
    original_id = env._container_id
    original_lease = env._reuse_lease
    invalid_home = tmp_path / "not-a-directory"
    invalid_home.write_text("fixture")
    env._snap_compat = True
    monkeypatch.setattr(docker_env.Path, "home", lambda: invalid_home)
    try:
        assert env._recreate_container() is False
        with store.transaction() as db:
            owners = db.execute("SELECT token FROM owners").fetchall()
        assert owners == [(original_lease.token,)], "only the original live owner may remain"
        assert env._reuse_lease is original_lease
        assert containers[original_id]["state"] == "running"
    finally:
        env.cleanup()
        assert env.wait_for_cleanup()


@pytest.mark.parametrize("removal_fails", [False, True])
def test_failed_registry_finish_releases_unassigned_owner(monkeypatch, removal_fails):
    containers, calls = _stateful_docker(monkeypatch)
    env = _make_dummy_env(persistent_filesystem=True)
    store = env._reuse_owners
    env.cleanup()
    assert env.wait_for_cleanup()
    original_containers = set(containers)
    failure = sqlite3.OperationalError("finish failed")

    def fail_finish(name, container_id):
        raise failure

    run = docker_env.subprocess.run

    def remove(cmd, **kwargs):
        if cmd[1:3] == ["rm", "-f"]:
            if removal_fails:
                calls.append(list(cmd))
                raise subprocess.TimeoutExpired(cmd, 10)
            # The shared fake resolves IDs only; Docker also accepts the run name.
            container_id = next(cid for cid, rec in containers.items() if rec["name"] == cmd[-1])
            return run([*cmd[:-1], container_id], **kwargs)
        return run(cmd, **kwargs)

    monkeypatch.setattr(store, "finish", fail_finish)
    monkeypatch.setattr(docker_env.subprocess, "run", remove)
    with pytest.raises(sqlite3.OperationalError) as caught:
        env._docker_run(env.cwd)
    assert caught.value is failure
    with store.transaction() as db:
        assert db.execute("SELECT token FROM owners").fetchall() == []
    assert env._reuse_lease is None
    assert any(cmd[1:3] == ["rm", "-f"] for cmd in calls)
    if not removal_fails:
        assert set(containers) == original_containers
