"""A replaced sandbox is reclaimable only after every kernel-held owner lease ends (#84969)."""

import os
import shutil
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import pytest

from hermes_constants import get_hermes_home
from tests.tools.test_docker_environment import _make_dummy_env
from tests.tools.test_docker_environment_reuse_candidates import _stateful_docker
from tests.waiting import wait_until
from tools.environments import docker as docker_env
from tools.environments.docker_reuse_owners import OWNER_LABEL, ReuseOwners


def registered_runtime_fingerprint(environment: docker_env.DockerEnvironment, container_id: str) -> str:
    """Seed the real ownership receipt for the existing-container boundary fixtures."""
    owners = environment._reuse_owners
    if owners is not None:
        with owners.transaction() as db:
            recorded = db.execute("SELECT 1 FROM containers WHERE container_id=?", (container_id,)).fetchone()
        if recorded is None:
            name = "fixture-" + uuid.uuid4().hex
            lease = owners.reserve(name, environment._labels, environment._runtime_fp)
            owners.finish(name, container_id)
            owners.release(lease)
    return environment._runtime_fp


@pytest.mark.parametrize("ending", ["release", "kill"])
def test_real_process_owner_protects_container_until_release_or_death(monkeypatch, tmp_path, ending):
    containers, _calls = _stateful_docker(monkeypatch)
    first = _make_dummy_env(persistent_filesystem=True, memory=1024)
    stale_id = first._container_id
    first.cleanup()
    assert first.wait_for_cleanup()
    ready = tmp_path / "ready"
    child_code = """
import sys
from pathlib import Path
from tools.environments.docker_reuse_owners import ReuseOwners
store = ReuseOwners(Path(sys.argv[1]))
lease = store.claim(sys.argv[2], lambda: True)
assert lease is not None
Path(sys.argv[3]).touch()
sys.stdin.buffer.read(1)
store.release(lease)
"""
    child = subprocess.Popen(
        [sys.executable, "-c", child_code, str(get_hermes_home()), stale_id, str(ready)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    try:
        wait_until(lambda: ready.exists() or child.poll() is not None, what="child acquired sandbox lease")
        assert ready.exists(), child.communicate(timeout=10)
        replacement = _make_dummy_env(persistent_filesystem=True, memory=2048)
        assert docker_env.reap_orphan_containers(max_age_seconds=0) == 0
        assert containers[stale_id]["state"] == "running"
        if ending == "kill":
            child.kill()
            child.communicate(timeout=10)
        else:
            child.communicate(input=b"x", timeout=10)
            assert child.returncode == 0
        assert docker_env.reap_orphan_containers(max_age_seconds=0) == 1
        assert stale_id not in containers
        assert containers[replacement._container_id]["state"] == "running"
        replacement.cleanup()
        assert replacement.wait_for_cleanup()
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=10)


@pytest.mark.parametrize("guard", ["compatible", "young", "foreign-profile", "foreign-label", "missing-lock", "copied-store"])
def test_unproven_or_current_container_is_never_reaped(monkeypatch, tmp_path, guard):
    containers, _calls = _stateful_docker(monkeypatch)
    first = _make_dummy_env(persistent_filesystem=True, memory=1024)
    stale_id = first._container_id
    store = first._reuse_owners
    assert store is not None
    lease = first._reuse_lease
    assert lease is not None
    replacement = None
    if guard != "compatible":
        replacement = _make_dummy_env(persistent_filesystem=True, memory=2048)
    if guard == "missing-lock":
        lease.lock.__exit__(None, None, None)
        lease.lock.path.unlink()
        first._reuse_lease = None  # corrupted record remains, without an owner lock
    else:
        first.cleanup()
        assert first.wait_for_cleanup()
    if guard == "foreign-label":
        containers[stale_id]["labels"][OWNER_LABEL] = "0" * 32
    if guard == "copied-store":
        copied = tmp_path / "copied-home"
        destination = copied / "runtime" / "docker-reuse-owners"
        destination.mkdir(parents=True)
        shutil.copy2(store.path, destination / store.path.name)
        assert ReuseOwners(copied).namespace != store.namespace
        assert ReuseOwners(copied).reap("/usr/bin/docker", 0, None) == 0
    else:
        age = 600 if guard == "young" else 0
        profile = "unrelated-profile" if guard == "foreign-profile" else None
        assert docker_env.reap_orphan_containers(max_age_seconds=age, profile_filter=profile) == 0
    assert containers[stale_id]["state"] == "running"
    first.cleanup()
    assert first.wait_for_cleanup()
    if replacement is not None:
        replacement.cleanup()
        assert replacement.wait_for_cleanup()


def test_claim_cancels_retirement_and_removed_receipt_cannot_be_revived(monkeypatch):
    containers, _calls = _stateful_docker(monkeypatch)
    first = _make_dummy_env(persistent_filesystem=True, memory=1024)
    stale_id = first._container_id
    store = first._reuse_owners
    assert store is not None
    first.cleanup()
    assert first.wait_for_cleanup()
    replacement = _make_dummy_env(persistent_filesystem=True, memory=2048)

    revived = _make_dummy_env(persistent_filesystem=True, memory=1024)
    assert revived._container_id == stale_id
    revived.cleanup()
    assert revived.wait_for_cleanup()
    assert store.reap("/usr/bin/docker", 0, None) == 0  # claim cancelled old retirement

    successor = _make_dummy_env(persistent_filesystem=True, memory=2048)
    assert store.reap("/usr/bin/docker", 0, None) == 1
    assert stale_id not in containers
    starts = []
    assert store.claim(stale_id, lambda: starts.append(stale_id) or True) is None
    assert starts == []  # stale inventory cannot restart a retired container
    for env in (replacement, successor):
        env.cleanup()
        assert env.wait_for_cleanup()


def test_protocol_upgrade_preserves_old_container_and_rejects_downgrade_match(monkeypatch):
    containers, _calls = _stateful_docker(monkeypatch)
    old = _make_dummy_env(persistent_filesystem=True)
    stale_id = old._container_id
    legacy_fp = docker_env._runtime_reuse_fingerprint(
        old._image, old._all_run_args, workdir=old._create_cwd,
        client_env=old._run_client_env, init=not (old._image_uses_s6_init or old._snap_compat))
    containers[stale_id]["labels"]["hermes-runtime-fingerprint"] = legacy_fp
    del containers[stale_id]["labels"][OWNER_LABEL]
    old.cleanup()
    assert old.wait_for_cleanup()

    upgraded = _make_dummy_env(persistent_filesystem=True)
    assert upgraded._container_id != stale_id
    assert upgraded._runtime_fp != legacy_fp  # old client cannot match the ownership protocol
    changed = _make_dummy_env(persistent_filesystem=True, memory=2048)
    upgraded.cleanup()
    assert upgraded.wait_for_cleanup()
    assert docker_env.reap_orphan_containers(max_age_seconds=0) == 1
    assert containers[stale_id]["state"] == "running"  # unknown legacy state is never reclaimed
    changed.cleanup()
    assert changed.wait_for_cleanup()


def test_delayed_cleanup_cannot_release_a_later_recoverys_owner(monkeypatch):
    containers, _calls = _stateful_docker(monkeypatch)
    env = _make_dummy_env(persistent_filesystem=True, memory=1024)
    stale_id = env._container_id
    entered, finish = threading.Event(), threading.Event()

    def strip(_docker, _container):
        entered.set()
        assert finish.wait(5)

    monkeypatch.setattr(docker_env, "strip_sandbox_privileges", strip)
    env.cleanup()
    try:
        assert entered.wait(5)
        assert env._recreate_container()
        current_lease = env._reuse_lease
        assert current_lease is not None
    finally:
        finish.set()
        assert env.wait_for_cleanup()
    replacement = _make_dummy_env(persistent_filesystem=True, memory=2048)
    assert env._reuse_lease is current_lease
    assert docker_env.reap_orphan_containers(max_age_seconds=0) == 0
    assert containers[stale_id]["state"] == "running"
    env.cleanup()
    assert env.wait_for_cleanup()
    assert docker_env.reap_orphan_containers(max_age_seconds=0) == 1
    replacement.cleanup()
    assert replacement.wait_for_cleanup()
