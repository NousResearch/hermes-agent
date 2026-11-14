"""Docker ownership registry failures preserve safe construction and recovery (#85342).

Faults use real SQLite corruption and triggers against an isolated registry;
Docker commands stay behind the existing stateful fake boundary.
"""

import gc
import sqlite3

import pytest

from tests.tools.test_docker_environment import _make_dummy_env
from tests.tools.test_docker_environment_reuse_candidates import _stateful_docker
from tools.environments import docker as docker_env
from tools.environments.base import EnvironmentConnectionError


@pytest.fixture(autouse=True)
def isolated_registry_and_cleanup(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TERMINAL_SANDBOX_DIR", str(tmp_path / "sandboxes"))
    allocated = []
    original_init = docker_env.DockerEnvironment.__init__

    def tracked_init(self, *args, **kwargs):
        allocated.append(self)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(docker_env.DockerEnvironment, "__init__", tracked_init)
    yield
    for env in allocated:
        if getattr(env, "_container_id", None):
            env.cleanup()
    allocated.clear()
    gc.collect()
    assert docker_env.DockerEnvironment.wait_for_all_teardowns(timeout=10)


def _registry_sql(store, sql):
    with store.transaction() as db:
        db.execute(sql)


def _resolve_removal_names(monkeypatch, containers):
    """Docker rm accepts names too; the stateful fake normally resolves IDs."""
    fake_run = docker_env.subprocess.run

    def resolve_name(cmd, **kwargs):
        if cmd[1:3] == ["rm", "-f"]:
            cid = next((cid for cid, rec in containers.items() if rec["name"] == cmd[-1]), cmd[-1])
            cmd = [*cmd[:-1], cid]
        return fake_run(cmd, **kwargs)

    monkeypatch.setattr(docker_env.subprocess, "run", resolve_name)


def test_recovery_reserve_sqlite_failure_returns_false(monkeypatch, caplog):
    containers, _ = _stateful_docker(monkeypatch)
    env = _make_dummy_env(persistent_filesystem=True)
    store = env._reuse_owners
    containers.clear()
    _registry_sql(store, """
        CREATE TRIGGER fail_reserve BEFORE INSERT ON containers
        BEGIN SELECT RAISE(ABORT, 'diagnostic reserve'); END
    """)
    try:
        assert env._recreate_container() is False
        assert "diagnostic reserve" in caplog.text
        with store.transaction() as db:
            assert db.execute("SELECT token FROM owners").fetchall() == []
    finally:
        _registry_sql(store, "DROP TRIGGER fail_reserve")
        env.cleanup()
        assert env.wait_for_cleanup()


@pytest.mark.parametrize("failure", ["sqlite", "filesystem"])
def test_registry_initialization_is_actionable_and_fail_closed(monkeypatch, tmp_path, failure):
    containers, calls = _stateful_docker(monkeypatch)
    runtime = tmp_path / "home" / "runtime"
    runtime.mkdir(parents=True)
    registry = runtime / "docker-reuse-owners"
    if failure == "sqlite":
        registry.mkdir()
        (registry / "docker-reuse-owners.sqlite3").write_bytes(b"not a sqlite database")
        cause_type = sqlite3.DatabaseError
    else:
        registry.write_text("not a directory")
        cause_type = OSError
    try:
        with pytest.raises(EnvironmentConnectionError, match="ownership registry") as caught:
            _make_dummy_env(persistent_filesystem=True)
        assert isinstance(caught.value.__cause__, cause_type)
        assert "runtime/docker-reuse-owners" in caught.value.retry_hint
        assert "repair" in caught.value.retry_hint and "retry" in caught.value.retry_hint
    finally:
        assert not containers
        assert not any(cmd[1] == "run" for cmd in calls)


@pytest.mark.parametrize("claim_existing", [False, True])
def test_supersede_failure_preserves_healthy_environment(monkeypatch, caplog, claim_existing):
    containers, _ = _stateful_docker(monkeypatch)
    first = _make_dummy_env(persistent_filesystem=True, memory=1024)
    store = first._reuse_owners
    first.cleanup()
    assert first.wait_for_cleanup()
    if claim_existing:
        ready = _make_dummy_env(persistent_filesystem=True, memory=2048)
        ready_id = ready._container_id
        ready.cleanup()
        assert ready.wait_for_cleanup()
    _registry_sql(store, """
        CREATE TRIGGER fail_supersede BEFORE UPDATE OF superseded ON containers
        WHEN NEW.superseded IS NOT NULL
        BEGIN SELECT RAISE(ABORT, 'diagnostic supersede'); END
    """)
    try:
        successor = _make_dummy_env(persistent_filesystem=True, memory=2048)
        cid = successor._container_id
        if claim_existing:
            assert cid == ready_id
        assert containers[cid]["state"] == "running"
        assert successor._reuse_lease is not None
        assert "diagnostic supersede" in caplog.text
        with store.transaction() as db:
            assert db.execute(
                "SELECT container_id FROM containers JOIN owners USING(name) WHERE token=?",
                (successor._reuse_lease.token,),
            ).fetchone() == (cid,)
        # Run a local child at the Docker exec boundary, following the existing
        # test_docker_environment fake-exec pattern. Never spawn the Docker CLI.
        original_popen = docker_env._popen_bash

        def exec_healthy(cmd, stdin_data=None, **kwargs):
            assert cid in cmd
            return original_popen(["bash", "-c", "printf healthy"], stdin_data, **kwargs)

        monkeypatch.setattr(docker_env, "_popen_bash", exec_healthy)
        result = successor.execute("printf healthy")
        assert result["returncode"] == 0
        assert "healthy" in result["output"]
    finally:
        _registry_sql(store, "DROP TRIGGER fail_supersede")


@pytest.mark.parametrize("failure_stage", ["reserve", "finish"])
def test_recovery_retries_next_command_after_registry_is_repaired(monkeypatch, failure_stage):
    containers, _ = _stateful_docker(monkeypatch)
    env = _make_dummy_env(persistent_filesystem=True)
    store = env._reuse_owners
    containers.clear()
    event = "INSERT ON containers" if failure_stage == "reserve" else "UPDATE OF container_id ON containers"
    _registry_sql(store, f"""
        CREATE TRIGGER fail_recovery BEFORE {event}
        BEGIN SELECT RAISE(ABORT, 'retryable registry failure'); END
    """)
    _resolve_removal_names(monkeypatch, containers)
    original_popen = docker_env._popen_bash

    def execute_at_fake_container(cmd, stdin_data=None, **kwargs):
        if env._container_id in containers:
            script = "printf retry-ready"
        else:
            script = "printf 'No such container: retry'; exit 1"
        return original_popen(["bash", "-c", script], stdin_data, **kwargs)

    monkeypatch.setattr(docker_env, "_popen_bash", execute_at_fake_container)
    try:
        failed = env.execute("printf retry-ready")
        assert failed["returncode"] == 1
        assert "No such container: retry" in failed["output"]
        _registry_sql(store, "DROP TRIGGER fail_recovery")
        recovered = env.execute("printf retry-ready")
        assert recovered["returncode"] == 0
        assert recovered["output"].strip() == "retry-ready"
        assert env._container_id in containers
        assert env._reuse_lease is not None
    finally:
        _registry_sql(store, "DROP TRIGGER IF EXISTS fail_recovery")
        env.cleanup()
        assert env.wait_for_cleanup()


@pytest.mark.parametrize("release_fails", [False, True])
@pytest.mark.parametrize("public_execute", [False, True])
def test_recovery_finish_or_release_sqlite_failure_is_degraded(monkeypatch, caplog, release_fails, public_execute):
    containers, _ = _stateful_docker(monkeypatch)
    env = _make_dummy_env(persistent_filesystem=True)
    store = env._reuse_owners
    containers.clear()
    _registry_sql(store, """
        CREATE TRIGGER fail_finish BEFORE UPDATE OF container_id ON containers
        BEGIN SELECT RAISE(ABORT, 'diagnostic finish'); END
    """)
    if release_fails:
        initial_token = env._reuse_lease.token
        _registry_sql(store, f"""
            CREATE TRIGGER fail_release BEFORE DELETE ON owners
            WHEN OLD.token <> '{initial_token}'
            BEGIN SELECT RAISE(ABORT, 'diagnostic release'); END
        """)
    _resolve_removal_names(monkeypatch, containers)
    original_popen = docker_env._popen_bash

    def missing_container(cmd, stdin_data=None, **kwargs):
        return original_popen(
            ["bash", "-c", "printf 'No such container: diagnostic'; exit 1"], stdin_data, **kwargs,
        )

    monkeypatch.setattr(docker_env, "_popen_bash", missing_container)
    try:
        result = env.execute("true") if public_execute else env._recreate_container()
        if public_execute:
            assert result["returncode"] == 1
            assert "No such container: diagnostic" in result["output"]
        else:
            assert result is False
        expected_error = "diagnostic release" if release_fails else "diagnostic finish"
        assert expected_error in caplog.text
        assert not containers, "failed run must remove its newly created container"
        with store.transaction() as db:
            receipts = db.execute("SELECT token FROM owners").fetchall()
        assert len(receipts) == int(release_fails)
        for (token,) in receipts:
            assert (store.path.parent / (token + ".lock")).exists()
    finally:
        _registry_sql(store, "DROP TRIGGER fail_finish")
        if release_fails:
            _registry_sql(store, "DROP TRIGGER fail_release")
        # A release failure intentionally retains its receipt; simulate retry
        # after the registry becomes writable, and close every evidence lease.
        with store.transaction() as db:
            tokens = [row[0] for row in db.execute("SELECT token FROM owners")]
            db.execute("DELETE FROM owners")
        for token in tokens:
            (store.path.parent / (token + ".lock")).unlink(missing_ok=True)
        env.cleanup()
        assert env.wait_for_cleanup()
