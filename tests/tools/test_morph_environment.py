"""Unit tests for the Morph Cloud execution environment backend."""

import importlib
import logging
import sys
import threading
from types import ModuleType, SimpleNamespace
from pathlib import Path

import pytest

_repo_root = Path(__file__).resolve().parent.parent.parent
tools_pkg = ModuleType("tools")
tools_pkg.__path__ = [str(_repo_root / "tools")]
sys.modules["tools"] = tools_pkg
environments_pkg = ModuleType("tools.environments")
environments_pkg.__path__ = [str(_repo_root / "tools" / "environments")]
sys.modules["tools.environments"] = environments_pkg
tools_pkg.environments = environments_pkg


def _exec_response(stdout="", stderr="", exit_code=0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, exit_code=exit_code)


def _resource_spec(vcpus=1, memory=5120, disk_size=51200):
    return SimpleNamespace(vcpus=vcpus, memory=memory, disk_size=disk_size)


class FakeMorphApiError(Exception):
    def __init__(self, status_code, message="Morph API error"):
        self.status_code = status_code
        super().__init__(message)


class FakeSnapshot:
    def __init__(
        self,
        snapshot_id,
        *,
        image_id="morphvm-minimal",
        vcpus=1,
        memory=5120,
        disk_size=51200,
        metadata=None,
        digest=None,
        status="ready",
    ):
        self.id = snapshot_id
        self.refs = SimpleNamespace(image_id=image_id)
        self.spec = _resource_spec(vcpus=vcpus, memory=memory, disk_size=disk_size)
        self.metadata = dict(metadata or {})
        self.digest = digest
        self.status = status
        self.wait_calls = []
        self.set_metadata_calls = []

    def wait_until_ready(self, timeout=None):
        self.wait_calls.append(timeout)
        self.status = "ready"

    def set_metadata(self, metadata):
        self.set_metadata_calls.append(metadata)
        self.metadata = dict(metadata)


class FakeInstance:
    def __init__(
        self,
        instance_id,
        *,
        status="ready",
        created=1,
        metadata=None,
        vcpus=1,
        memory=5120,
        disk_size=51200,
    ):
        self.id = instance_id
        self.status = status
        self.created = created
        self.metadata = dict(metadata or {})
        self.spec = _resource_spec(vcpus=vcpus, memory=memory, disk_size=disk_size)
        self.wake_on = SimpleNamespace(wake_on_ssh=False)
        self.ttl = SimpleNamespace(ttl_seconds=None, ttl_action=None)
        self.resume_calls = 0
        self.stop_calls = 0
        self.wait_calls = []
        self.set_metadata_calls = []
        self.set_ttl_calls = []
        self.set_wake_on_calls = []
        self.exec_calls = []
        self.exec_responses = []
        self.exec_side_effect = None

    def resume(self):
        self.resume_calls += 1
        self.status = "ready"

    def wait_until_ready(self, timeout=None):
        self.wait_calls.append(timeout)
        if self.status in ("pending", "paused", "saving"):
            self.status = "ready"
        if self.status == "error":
            raise RuntimeError("instance error")

    def set_metadata(self, metadata):
        self.set_metadata_calls.append(metadata)
        self.metadata = dict(metadata)

    def set_ttl(self, ttl_seconds, ttl_action=None):
        self.set_ttl_calls.append((ttl_seconds, ttl_action))
        self.ttl.ttl_seconds = ttl_seconds
        self.ttl.ttl_action = ttl_action

    def set_wake_on(self, wake_on_ssh=None, wake_on_http=None):
        self.set_wake_on_calls.append((wake_on_ssh, wake_on_http))
        if wake_on_ssh is not None:
            self.wake_on.wake_on_ssh = wake_on_ssh

    def stop(self):
        self.stop_calls += 1

    def exec(self, command, timeout=None, on_stdout=None, on_stderr=None):
        self.exec_calls.append({"command": command, "timeout": timeout})
        if callable(self.exec_side_effect):
            return self.exec_side_effect(
                command, timeout=timeout, on_stdout=on_stdout, on_stderr=on_stderr
            )
        if self.exec_responses:
            return self.exec_responses.pop(0)
        return _exec_response()


class FakeInstancesAPI:
    def __init__(self, *, listed_instances=None, start_instance=None):
        self.listed_instances = list(listed_instances or [])
        self.start_instance = start_instance or FakeInstance("inst-start", created=999)
        self.by_id = {instance.id: instance for instance in self.listed_instances}
        self.by_id[self.start_instance.id] = self.start_instance
        self.list_calls = []
        self.get_calls = []
        self.get_side_effect = None
        self.start_calls = []

    def list(self, metadata=None):
        self.list_calls.append(metadata)
        return list(self.listed_instances)

    def get(self, instance_id):
        self.get_calls.append(instance_id)
        if self.get_side_effect is not None:
            raise self.get_side_effect
        if instance_id not in self.by_id:
            raise RuntimeError("not found")
        return self.by_id[instance_id]

    def start(
        self,
        snapshot_id,
        metadata=None,
        ttl_seconds=None,
        ttl_action=None,
        timeout=None,
    ):
        self.start_calls.append(
            {
                "snapshot_id": snapshot_id,
                "metadata": metadata,
                "ttl_seconds": ttl_seconds,
                "ttl_action": ttl_action,
                "timeout": timeout,
            }
        )
        self.by_id[self.start_instance.id] = self.start_instance
        self.start_instance.wait_until_ready(timeout=timeout)
        return self.start_instance


class FakeSnapshotsAPI:
    def __init__(self, *, create_snapshot=None):
        self.create_snapshot = create_snapshot or FakeSnapshot("snapshot-created")
        self.create_calls = []

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        snapshot = self.create_snapshot
        snapshot.refs.image_id = kwargs.get("image_id") or snapshot.refs.image_id
        snapshot.spec = _resource_spec(
            vcpus=kwargs.get("vcpus") or snapshot.spec.vcpus,
            memory=kwargs.get("memory") or snapshot.spec.memory,
            disk_size=kwargs.get("disk_size") or snapshot.spec.disk_size,
        )
        snapshot.digest = kwargs.get("digest")
        snapshot.metadata = dict(kwargs.get("metadata") or {})
        return snapshot


class FakeMorphClient:
    def __init__(self, instances_api, snapshots_api):
        self.instances = instances_api
        self.snapshots = snapshots_api


@pytest.fixture()
def morph_factory(monkeypatch):
    holder = {"client": None}
    module = ModuleType("morphcloud")
    module.MorphCloudClient = lambda *args, **kwargs: holder["client"]
    monkeypatch.setitem(sys.modules, "morphcloud", module)
    monkeypatch.delenv("TERMINAL_MORPH_GENERATION", raising=False)
    morph_module = importlib.import_module("tools.environments.morph")

    monkeypatch.setattr(morph_module, "is_interrupted", lambda: False)

    def _make_env(
        *,
        listed_instances=None,
        start_instance=None,
        create_snapshot=None,
        **kwargs,
    ):
        image_id = kwargs.pop("image_id", "morphvm-minimal")
        api = FakeInstancesAPI(
            listed_instances=listed_instances,
            start_instance=start_instance,
        )
        snapshots_api = FakeSnapshotsAPI(
            create_snapshot=create_snapshot,
        )
        holder["client"] = FakeMorphClient(api, snapshots_api)
        env = morph_module.MorphEnvironment(
            image_id=image_id,
            task_id="task-123",
            **kwargs,
        )
        env._morph_module = morph_module
        env._fake_api = api
        env._fake_snapshots_api = snapshots_api
        return env, api

    return _make_env


def test_starts_new_instance_when_workspace_is_missing(morph_factory):
    start_instance = FakeInstance("inst-start", status="pending", created=50)
    env, api = morph_factory(
        listed_instances=[],
        start_instance=start_instance,
        cpu=2,
        memory=6144,
        disk=20480,
        lifetime_seconds=900,
    )

    assert env._instance is start_instance
    assert env._fake_snapshots_api.create_calls[0]["image_id"] == "morphvm-minimal"
    assert api.start_calls == [
        {
            "snapshot_id": "snapshot-created",
            "metadata": {
                "kind": "hermes-agent",
                "hermes-agent:task_id": "task-123",
                "hermes-agent:resource": "workspace",
                "hermes-agent:backend": "morph",
                "hermes-agent:generation": "hermes-morph-v1",
                "hermes-agent:image_id": "morphvm-minimal",
                "hermes-agent:base_snapshot_digest": env._base_snapshot_digest,
                "webUIName": "Hermes Agent task-123",
            },
            "ttl_seconds": 900,
            "ttl_action": "pause",
            "timeout": 180,
        }
    ]
    assert start_instance.wait_calls
    assert start_instance.set_ttl_calls[-1] == (900, "pause")
    assert start_instance.set_wake_on_calls[-1] == (True, None)


def test_materializes_image_base_snapshot_with_digest(morph_factory):
    start_instance = FakeInstance("inst-start", created=50)
    create_snapshot = FakeSnapshot("snapshot-created")
    env, api = morph_factory(
        listed_instances=[],
        start_instance=start_instance,
        create_snapshot=create_snapshot,
        image_id="morphvm-minimal",
        cpu=2,
        memory=6144,
        disk=20480,
        lifetime_seconds=900,
    )

    assert env._instance is start_instance
    assert len(env._fake_snapshots_api.create_calls) == 1
    create_call = env._fake_snapshots_api.create_calls[0]
    assert create_call["image_id"] == "morphvm-minimal"
    assert create_call["vcpus"] == 2
    assert create_call["memory"] == 6144
    assert create_call["disk_size"] == 20480
    assert create_call["digest"].startswith("hermes-morph-base:")
    assert create_call["metadata"]["kind"] == "hermes-agent"
    assert create_call["metadata"]["hermes-agent:resource"] == "base-snapshot"
    assert create_call["metadata"]["hermes-agent:backend"] == "morph"
    assert create_call["metadata"]["hermes-agent:image_id"] == "morphvm-minimal"
    assert create_call["metadata"]["webUIName"] == (
        "Hermes Agent Base morphvm-minimal (2 CPU, 6144 MB, 20480 MB)"
    )
    assert api.start_calls[0]["snapshot_id"] == "snapshot-created"
    assert api.start_calls[0]["metadata"]["webUIName"] == "Hermes Agent task-123"


def test_reuses_ready_instance_without_boot(morph_factory):
    ready = FakeInstance("inst-ready", status="ready", created=10)
    env, api = morph_factory(listed_instances=[ready])

    assert env._instance is ready
    assert api.start_calls == []
    assert api.list_calls[0]["hermes-agent:task_id"] == "task-123"
    assert ready.set_ttl_calls[-1] == (300, "pause")
    assert ready.set_wake_on_calls[-1] == (True, None)


def test_resumes_paused_instance(morph_factory):
    paused = FakeInstance("inst-paused", status="paused", created=11)
    env, api = morph_factory(listed_instances=[paused])

    assert env._instance is paused
    assert paused.resume_calls == 1
    assert paused.wait_calls
    assert api.start_calls == []


def test_attached_instance_backfills_metadata_without_overwriting_name(morph_factory):
    ready = FakeInstance(
        "inst-ready",
        status="ready",
        created=12,
        metadata={
            "kind": "hermes-agent",
            "hermes-agent:task_id": "task-123",
            "hermes-agent:resource": "workspace",
            "hermes-agent:backend": "morph",
            "hermes-agent:generation": "hermes-morph-v1",
            "webUIName": "Custom Workspace",
        },
    )

    env, api = morph_factory(
        listed_instances=[ready],
        image_id="morphvm-minimal",
        cpu=2,
        memory=6144,
        disk=20480,
    )

    assert env._instance is ready
    assert api.start_calls == []
    assert ready.set_metadata_calls[-1]["webUIName"] == "Custom Workspace"
    assert ready.set_metadata_calls[-1]["kind"] == "hermes-agent"
    assert ready.set_metadata_calls[-1]["hermes-agent:image_id"] == "morphvm-minimal"
    assert (
        ready.set_metadata_calls[-1]["hermes-agent:base_snapshot_digest"]
        == env._base_snapshot_digest
    )


def test_missing_instance_refresh_recreates_workspace(morph_factory):
    ready = FakeInstance("inst-ready", status="ready", created=10)
    replacement = FakeInstance("inst-fresh", status="ready", created=11)
    env, api = morph_factory(listed_instances=[ready], start_instance=replacement)
    api.listed_instances = []
    api.by_id.pop("inst-ready", None)
    api.get_side_effect = FakeMorphApiError(404, "missing")

    env._ensure_instance_ready()

    assert api.get_calls[-1] == "inst-ready"
    assert len(api.start_calls) == 1
    assert env._instance is replacement


def test_transient_refresh_error_is_raised_without_recreating_workspace(
    morph_factory, caplog
):
    ready = FakeInstance("inst-ready", status="ready", created=10)
    env, api = morph_factory(listed_instances=[ready])
    api.get_side_effect = FakeMorphApiError(503, "temporary failure")

    with caplog.at_level(logging.WARNING):
        with pytest.raises(FakeMorphApiError, match="temporary failure"):
            env._ensure_instance_ready()

    assert api.get_calls[-1] == "inst-ready"
    assert api.start_calls == []
    assert env._instance is ready
    assert "failed to refresh instance inst-ready for task task-123" in caplog.text


def test_exec_script_uses_runtime_state_dir_not_literal_shell_vars(morph_factory):
    env, _ = morph_factory()

    script = env._build_exec_script("echo hi", "/root", 30, "abc123")

    assert "state_dir=/tmp/hermes-morph" in script
    assert "mkdir -p \"$state_dir\"" in script
    assert "pid_file=\"$state_dir/abc123.pid\"" in script
    assert "pgid_file=\"$state_dir/abc123.pgid\"" in script
    assert "\\$state_dir" not in script
    assert "\\$pid_file" not in script
    assert "\\$pgid_file" not in script


def test_duplicate_matches_quarantine_extras_deterministically(morph_factory):
    older = FakeInstance("inst-old", status="ready", created=10)
    newer = FakeInstance("inst-new", status="ready", created=20)

    env, api = morph_factory(listed_instances=[older, newer])

    assert env._instance is newer
    assert api.start_calls == []
    assert older.set_metadata_calls
    quarantined = older.set_metadata_calls[-1]
    assert quarantined["kind"] == "hermes-agent"
    assert quarantined["hermes-agent:resource"] == "workspace-quarantine"
    assert quarantined["hermes-agent:quarantine_reason"] == "duplicate_task_id"
    assert quarantined["hermes-agent:quarantine_winner"] == "inst-new"
    assert quarantined["hermes-agent:task_id"].startswith("quarantined:task-123:")


def test_all_error_matches_are_quarantined_and_replaced(morph_factory):
    err_a = FakeInstance("inst-a", status="error", created=1)
    err_b = FakeInstance("inst-b", status="error", created=2)
    start_instance = FakeInstance("inst-fresh", status="ready", created=3)

    env, api = morph_factory(listed_instances=[err_a, err_b], start_instance=start_instance)

    assert env._instance is start_instance
    assert len(api.start_calls) == 1
    assert err_a.set_metadata_calls
    assert err_b.set_metadata_calls
    assert (
        err_a.set_metadata_calls[-1]["hermes-agent:quarantine_reason"]
        == "all_matches_error"
    )


def test_execute_wraps_timeout_cwd_and_stdin(morph_factory):
    ready = FakeInstance("inst-ready", status="ready", created=10)
    ready.exec_responses = [_exec_response(stdout="ok", exit_code=0)]
    env, api = morph_factory(listed_instances=[ready])

    result = env.execute("python3", cwd="/workspace", timeout=7, stdin_data="print('hi')")

    assert result == {"output": "ok", "returncode": 0}
    call = ready.exec_calls[-1]
    assert call["command"][:2] == ["bash", "-lc"]
    script = call["command"][2]
    assert "cd /workspace" in script
    assert "timeout 7 bash -lc" in script
    assert "HERMES_EOF_" in script
    assert "print" in script
    assert call["timeout"] == 27
    assert api.get_calls[-1] == "inst-ready"


def test_execute_wraps_sudo_password_pipe(morph_factory, monkeypatch):
    ready = FakeInstance("inst-ready", status="ready", created=10)
    ready.exec_responses = [_exec_response(stdout="done", exit_code=0)]
    env, _ = morph_factory(listed_instances=[ready])
    monkeypatch.setattr(
        env,
        "_prepare_command",
        lambda command: ("sudo -S echo hi", "topsecret\n"),
    )

    env.execute("echo hi")

    script = ready.exec_calls[-1]["command"][2]
    assert "printf" in script
    assert "topsecret" in script
    assert "sudo -S echo hi" in script


def test_persistent_cleanup_detaches_without_stopping_instance(morph_factory):
    ready = FakeInstance("inst-ready", status="ready", created=10)
    env, api = morph_factory(listed_instances=[ready], lifetime_seconds=1200)
    previous_ttl_count = len(ready.set_ttl_calls)

    env.cleanup()

    assert ready.stop_calls == 0
    assert env._instance is ready
    assert len(ready.set_ttl_calls) >= previous_ttl_count
    assert api.get_calls[-1] == "inst-ready"


def test_nonpersistent_cleanup_stops_instance(morph_factory):
    ready = FakeInstance("inst-ready", status="ready", created=10)
    env, _ = morph_factory(listed_instances=[ready], persistent_filesystem=False)

    env.cleanup()

    assert ready.stop_calls == 1
    assert env._instance is None


def test_interrupt_returns_130_and_issues_remote_kill(morph_factory, monkeypatch):
    ready = FakeInstance("inst-ready", status="ready", created=10)
    gate = threading.Event()
    calls = {"count": 0}

    def _exec_side_effect(command, timeout=None, on_stdout=None, on_stderr=None):
        calls["count"] += 1
        if calls["count"] == 1:
            gate.wait(timeout=2)
            return _exec_response(stdout="late", exit_code=0)
        return _exec_response(stdout="", exit_code=0)

    ready.exec_side_effect = _exec_side_effect
    env, _ = morph_factory(listed_instances=[ready])
    monkeypatch.setattr(env._morph_module, "is_interrupted", lambda: True)

    try:
        result = env.execute("sleep 10")
        assert result["returncode"] == 130
        assert calls["count"] >= 2
    finally:
        gate.set()
