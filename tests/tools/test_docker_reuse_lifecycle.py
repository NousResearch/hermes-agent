"""Active ownership and immutable launch inputs survive the #85342 review cases."""

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from tests.tools.test_docker_environment import _make_dummy_env
from tests.tools.test_docker_environment_reuse_candidates import _stateful_docker
from tests.waiting import wait_until
from tools.environments import docker as docker_env
from tools.environments import docker_reuse_owners as ownership


@pytest.fixture(autouse=True)
def isolated_environments(monkeypatch, tmp_path):
    (tmp_path / "home" / "skills").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TERMINAL_SANDBOX_DIR", str(tmp_path / "sandboxes"))
    environments = []
    initialize = docker_env.DockerEnvironment.__init__

    def initialize_tracked(self, *args, **kwargs):
        environments.append(self)
        initialize(self, *args, **kwargs)

    monkeypatch.setattr(docker_env.DockerEnvironment, "__init__", initialize_tracked)
    yield
    for env in environments:
        if getattr(env, "_container_id", None):
            env.cleanup()
    assert docker_env.DockerEnvironment.wait_for_all_teardowns(timeout=10)


@pytest.mark.parametrize("failed_creation", [False, True])
def test_release_rollback_keeps_kernel_lease_until_retry(tmp_path, failed_creation):
    store = ownership.ReuseOwners(tmp_path)
    lease = store.reserve("hermes-live", {"hermes-profile": "profile"}, "fp")
    contender = """
import sys
from pathlib import Path
from hermes_cli.active_sessions import _FileLock
try:
    with _FileLock(Path(sys.argv[1]), timeout=0):
        print('available')
except RuntimeError:
    print('held')
"""

    def lock_state():
        result = subprocess.run(
            [sys.executable, "-c", contender, str(lease.lock.path)],
            capture_output=True, text=True, timeout=20, check=True,
        )
        return result.stdout.strip()

    assert lock_state() == "held"
    with store.transaction() as db:
        db.execute("CREATE TRIGGER reject_live_release BEFORE DELETE ON owners "
                   "BEGIN SELECT RAISE(ABORT, 'release failed'); END")
    try:
        with pytest.raises(sqlite3.IntegrityError, match="release failed"):
            store.release(lease, failed_creation=failed_creation)
        assert lease.lock.path.is_file()
        assert lock_state() == "held"
    finally:
        with store.transaction() as db:
            db.execute("DROP TRIGGER reject_live_release")
        store.release(lease, failed_creation=failed_creation)
    assert lock_state() == "available"


def test_recovery_keeps_healthy_owner_through_reaper_seam(monkeypatch):
    containers, _ = _stateful_docker(monkeypatch)
    env = _make_dummy_env(persistent_filesystem=True, memory=1024)
    old_id = env._container_id
    _make_dummy_env(persistent_filesystem=True, memory=2048)
    store = env._reuse_owners
    claim = store.claim
    reaps = []

    def claim_after_reap(container_id, start, **kwargs):
        reaps.append(store.reap(env._docker_exe, 0, None))
        return claim(container_id, start, **kwargs)

    monkeypatch.setattr(store, "claim", claim_after_reap)
    popen = docker_env._popen_bash
    monkeypatch.setattr(docker_env, "_popen_bash", lambda _cmd, stdin_data=None, **kwargs:
                        popen(["bash", "-c", "printf 'user service is not running'; exit 1"],
                              stdin_data, **kwargs))
    result = env.execute("check-service")
    assert result["returncode"] == 1
    assert "user service is not running" in result["output"]
    assert reaps == [0]
    assert env._container_id == old_id
    assert containers[old_id]["state"] == "running"


def test_failed_recovery_handoff_retains_both_leases_for_cleanup(monkeypatch):
    _stateful_docker(monkeypatch)
    env = _make_dummy_env(persistent_filesystem=True)
    store = env._reuse_owners
    previous = env._reuse_lease
    with store.transaction() as db:
        db.execute("CREATE TRIGGER reject_previous BEFORE DELETE ON owners "
                   f"WHEN OLD.token='{previous.token}' "
                   "BEGIN SELECT RAISE(ABORT, 'handoff failed'); END")
    try:
        assert env._recreate_container()
        assert env._reuse_lease.previous is previous
        with store.transaction() as db:
            assert len(db.execute("SELECT token FROM owners").fetchall()) == 2
        with pytest.raises(RuntimeError, match="file lock unavailable"):
            with ownership._FileLock(previous.lock.path, timeout=0):
                pytest.fail("failed handoff exposed the previous lease")
    finally:
        with store.transaction() as db:
            db.execute("DROP TRIGGER reject_previous")
        env.cleanup()
        assert env.wait_for_cleanup()
    with store.transaction() as db:
        assert db.execute("SELECT token FROM owners").fetchall() == []


@pytest.mark.parametrize("managed", [True, False])
def test_lockdown_removal_preserves_managed_stopped_owner(monkeypatch, managed):
    containers, _ = _stateful_docker(monkeypatch)
    options = {"persistent_filesystem": True,
               "shared_container_key": "" if managed else "legacy-shared"}
    first = _make_dummy_env(network=True, **options)
    old_id = first._container_id
    old_name = containers[old_id]["name"]
    containers[old_id]["state"] = "exited"
    second = _make_dummy_env(network=False, **options)
    assert containers[second._container_id]["network"] == "none"
    if managed:
        assert containers[old_id]["name"] == old_name
        assert containers[old_id]["state"] == "exited"
        assert first._reuse_lease is not None
    else:
        assert all(record["name"] != old_name for record in containers.values())


def test_last_successful_configuration_is_not_reaped(tmp_path, monkeypatch):
    store = ownership.ReuseOwners(tmp_path)
    labels = {"hermes-profile": "profile", "hermes-task-id": "task"}
    first = store.reserve("hermes-a", labels, "a")
    second = store.reserve("hermes-b", labels, "b")
    store.finish("hermes-a", "cid-a")
    store.supersede(labels, "a")
    store.finish("hermes-b", "cid-b")
    store.supersede(labels, "b")
    store.release(first)
    store.release(second)
    removed = []

    def docker(cmd, **kwargs):
        if cmd[1] == "inspect":
            record = {"Id": "cid-a", "Name": "/hermes-a", "State": {"Running": False},
                      "Config": {"Labels": {ownership.OWNER_LABEL: store.namespace,
                                            "hermes-runtime-fingerprint": "a"}}}
            return subprocess.CompletedProcess(cmd, 0, json.dumps(record), "")
        assert cmd[1:] == ["rm", "cid-a"]
        removed.append(cmd[-1])
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(ownership, "run_capture", docker)
    monkeypatch.setattr(ownership, "strip_sandbox_privileges", lambda *_: None)
    assert store.reap("docker", 0, None) == 1
    assert removed == ["cid-a"]
    with store.transaction() as db:
        assert db.execute("SELECT superseded, retired FROM containers WHERE name='hermes-b'").fetchone() == (None, 0)


@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("snap", [False, True])
def test_env_file_launch_uses_the_fingerprinted_snapshot(monkeypatch, tmp_path, inline, snap):
    containers, _ = _stateful_docker(monkeypatch)
    path = tmp_path / "launch.env"
    path.write_text("FOMA_WITNESS=A\n", encoding="utf-8")
    home = tmp_path / "snap-home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    args = [f"--env-file={path}"] if inline else ["--env-file", str(path)]
    run = docker_env.subprocess.run
    consumed = []
    staged = []

    def launch_with_mutation(cmd, **kwargs):
        if cmd[1] == "run":
            path.write_text("FOMA_WITNESS=B\n", encoding="utf-8")
            if inline:
                operand = next(arg.split("=", 1)[1] for arg in cmd if arg.startswith("--env-file="))
            else:
                operand = cmd[cmd.index("--env-file") + 1]
            staged.append(Path(operand))
            consumed.append(Path(operand).read_text(encoding="utf-8"))
        result = run(cmd, **kwargs)
        path.write_text("FOMA_WITNESS=A\n", encoding="utf-8")
        return result

    monkeypatch.setattr(docker_env.subprocess, "run", launch_with_mutation)
    first = _make_dummy_env(persistent_filesystem=True, extra_args=args, snap_compat=snap)
    second = _make_dummy_env(persistent_filesystem=True, extra_args=args, snap_compat=snap)
    assert first._container_id == second._container_id
    assert len(containers) == 1
    assert consumed == ["FOMA_WITNESS=A\n"]
    assert all(not file.exists() for file in staged)
    if snap:
        assert all(file.is_relative_to(home) for file in staged)


@pytest.mark.parametrize("creator_alive", [False, True])
def test_unfinished_creation_is_recovered_only_after_creator_dies(monkeypatch, tmp_path, creator_alive):
    containers, calls = _stateful_docker(monkeypatch)
    seed = _make_dummy_env(persistent_filesystem=True)
    record = containers[seed._container_id]
    store = seed._reuse_owners
    seed.cleanup()
    assert seed.wait_for_cleanup()
    containers.clear()
    record["name"] = "hermes-unfinished"
    containers["orphan-id"] = record
    ready = tmp_path / "creator-ready"
    script = """
import json
import sys
from pathlib import Path
from tools.environments.docker_reuse_owners import ReuseOwners
store = ReuseOwners(Path(sys.argv[1]))
lease = store.reserve('hermes-unfinished', json.loads(sys.argv[2]), sys.argv[3])
Path(sys.argv[4]).write_text(lease.token, encoding='utf-8')
sys.stdin.read()
"""
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path / "home"), json.dumps(record["labels"]),
         record["labels"]["hermes-runtime-fingerprint"], str(ready)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=Path(__file__).resolve().parents[2],
    )
    try:
        wait_until(ready.is_file, timeout=20, what="creator reservation")
        if not creator_alive:
            child.kill()
            child.communicate(timeout=20)
        before_runs = len([cmd for cmd in calls if cmd[1] == "run"])
        recovered = _make_dummy_env(persistent_filesystem=True)
        if creator_alive:
            assert recovered._container_id != "orphan-id"
            assert "orphan-id" in containers
            assert (store.path.parent / (ready.read_text(encoding="utf-8") + ".lock")).is_file()
        else:
            assert recovered._container_id == "orphan-id"
            assert len([cmd for cmd in calls if cmd[1] == "run"]) == before_runs
            assert not (store.path.parent / (ready.read_text(encoding="utf-8") + ".lock")).exists()
            with store.transaction() as db:
                assert db.execute("SELECT container_id FROM containers WHERE name='hermes-unfinished'").fetchone() == ("orphan-id",)
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=20)
