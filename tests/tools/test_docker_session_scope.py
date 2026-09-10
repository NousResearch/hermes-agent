"""docker_container_scope: session — persistent per-session containers (#46041).

Companion to test_docker_session_isolation.py, which pins the EPHEMERAL
per-session mode (docker + container_persistent: false). This file pins the
PERSISTENT per-session mode: ``docker_container_scope: session`` keeps the
persistence contract (fs state survives across turns; a stopped container is
restarted on resume) while giving each chat session its own container, keyed
``session:<key>`` with a retention lifecycle:

* ``stop_on_session_end`` (default) — cleanup() stops WITHOUT rm; the session's
  next use reattaches via the label probe (docker start). The orphan reaper
  must NOT reap such stopped containers (label ``hermes-session-retention``).
* ``keep_running`` — cleanup() is the persist no-op.
* ``remove_on_session_end`` — stop+rm (existing ephemeral contract).
* ``idle_ttl`` — the idle reaper force-removes after the session's own TTL.
* ``cleanup_vm`` finds envs cached under ``session:<key>`` from close paths
  holding a different durable id (AIAgent.close() passes session_id), via the
  close-key registry recorded at env creation. Delegate child ids are never
  registered, so their close() still resolves to the parent's env or no-ops.
"""

import datetime as _dt
import subprocess
import time

import pytest

from tools import terminal_tool, terminal_tool_backends
from tools.terminal_tool_lifecycle import (
    _cleanup_inactive_envs,
    cleanup_vm,
    get_active_env,
    is_persistent_env,
)
from tools.environments import docker as docker_env


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """Isolate override/alias/cwd/close-key state and pin terminal env vars."""
    before_overrides = dict(terminal_tool._task_env_overrides)
    terminal_tool._task_env_overrides.clear()
    with terminal_tool._container_alias_lock:
        before_aliases = dict(terminal_tool._container_aliases)
        terminal_tool._container_aliases.clear()
    with terminal_tool._session_cwd_lock:
        before_cwd = dict(terminal_tool._session_cwd)
        terminal_tool._session_cwd.clear()
    with terminal_tool._session_close_keys_lock:
        before_close_keys = dict(terminal_tool._session_close_keys)
        terminal_tool._session_close_keys.clear()
    # The config→env bridge is one-shot; mark it done so tests control env vars.
    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", True)
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "true")
    monkeypatch.setenv("TERMINAL_DOCKER_CONTAINER_SCOPE", "shared")
    monkeypatch.setenv("TERMINAL_DOCKER_SHARED_CONTAINER_KEY", "")
    monkeypatch.setenv("HERMES_SESSION_PROFILE", "")
    yield
    terminal_tool._task_env_overrides.clear()
    terminal_tool._task_env_overrides.update(before_overrides)
    with terminal_tool._container_alias_lock:
        terminal_tool._container_aliases.clear()
        terminal_tool._container_aliases.update(before_aliases)
    with terminal_tool._session_cwd_lock:
        terminal_tool._session_cwd.clear()
        terminal_tool._session_cwd.update(before_cwd)
    with terminal_tool._session_close_keys_lock:
        terminal_tool._session_close_keys.clear()
        terminal_tool._session_close_keys.update(before_close_keys)


def _pin_session(monkeypatch, scope="session", persistent="true", env_type="docker",
                 session_key=None, profile=""):
    """Pin the env vars + session-key reader the way a gateway session would."""
    monkeypatch.setenv("TERMINAL_ENV", env_type)
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", persistent)
    monkeypatch.setenv("TERMINAL_DOCKER_CONTAINER_SCOPE", scope)
    monkeypatch.setenv("HERMES_SESSION_PROFILE", profile)
    monkeypatch.setattr(terminal_tool, "_current_session_key", lambda: session_key or "")


# --- Routing: _resolve_container_task_id ---------------------------------------


def test_session_scope_keys_by_session(monkeypatch):
    """docker + persistent + scope=session keys by session:<key>."""
    _pin_session(monkeypatch, scope="session", session_key="telegram:123:456")
    assert terminal_tool._resolve_container_task_id(None) == "session:telegram:123:456"


def test_no_session_key_stays_default(monkeypatch):
    """CLI (no session key) unchanged under session scope."""
    _pin_session(monkeypatch, scope="session", session_key=None)
    assert terminal_tool._resolve_container_task_id(None) == "default"


def test_shared_container_key_wins_over_session_scope(monkeypatch):
    """An explicit docker_shared_container_key beats session scope (#46041)."""
    _pin_session(monkeypatch, scope="session", session_key="abc")
    monkeypatch.setenv("TERMINAL_DOCKER_SHARED_CONTAINER_KEY", "team")
    assert terminal_tool._resolve_container_task_id(None) == "shared:team"


def test_scope_session_keys_session_for_non_persistent_too(monkeypatch):
    """Non-persistent + scope=session still keys session:<key> (both branches agree)."""
    _pin_session(monkeypatch, scope="session", persistent="false", session_key="abc")
    assert terminal_tool._resolve_container_task_id(None) == "session:abc"


def test_scope_shared_keeps_profile_scoping(monkeypatch):
    """Default scope: docker+persistent still collapses to the profile branch."""
    _pin_session(monkeypatch, scope="shared", session_key="abc", profile="work")
    assert terminal_tool._resolve_container_task_id(None) == "profile:work"


def test_scope_shared_default_profile_stays_literal_default(monkeypatch):
    """Default profile + default scope keeps the shared "default" container (CLI+gateway)."""
    _pin_session(monkeypatch, scope="shared", session_key="abc", profile="")
    assert terminal_tool._resolve_container_task_id(None) == "default"


def test_non_docker_backend_unchanged(monkeypatch):
    """SSH + session scope keys session:<key> via the pre-existing branch."""
    _pin_session(monkeypatch, scope="session", env_type="ssh", session_key="abc")
    assert terminal_tool._resolve_container_task_id(None) == "session:abc"


def test_docker_session_isolation_enabled_for_session_scope(monkeypatch):
    """The isolation predicate must fire for session scope even with persistence on."""
    _pin_session(monkeypatch, scope="session")
    assert terminal_tool._docker_session_isolation_enabled() is True


def test_ephemeral_isolation_still_enabled(monkeypatch):
    """container_persistent: false keeps the existing isolation predicate."""
    _pin_session(monkeypatch, scope="shared", persistent="false")
    assert terminal_tool._docker_session_isolation_enabled() is True


def test_shared_persistent_not_isolated(monkeypatch):
    """The default mode stays non-isolated."""
    _pin_session(monkeypatch, scope="shared", persistent="true")
    assert terminal_tool._docker_session_isolation_enabled() is False


def test_isolation_refuses_process_tagged_override(monkeypatch, tmp_path):
    """Under session scope the process-global cwd must not become the mount."""
    _pin_session(monkeypatch, scope="session", session_key="abc")
    monkeypatch.setenv("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", "true")
    terminal_tool.register_task_env_overrides(
        "session:abc", {"cwd": str(tmp_path), "cwd_source": "process"})
    config = {"env_type": "docker", "docker_mount_cwd_to_workspace": True, "host_cwd": "/tmp/launch"}
    assert terminal_tool._resolve_task_host_cwd(config, "session:abc") is None


def test_session_attached_override_mounts(monkeypatch, tmp_path):
    """A session-tagged cwd override still mounts (unchanged from isolation mode)."""
    _pin_session(monkeypatch, scope="session", session_key="abc")
    monkeypatch.setenv("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", "true")
    terminal_tool.register_task_env_overrides(
        "session:abc", {"cwd": str(tmp_path), "cwd_source": "session"})
    config = {"env_type": "docker", "docker_mount_cwd_to_workspace": True, "host_cwd": None}
    assert terminal_tool._resolve_task_host_cwd(config, "session:abc") == str(tmp_path)


# --- Builder: _build_docker_env -------------------------------------------------


class _RecordingDockerEnv:
    """DockerEnvironment double: records kwargs, no docker daemon needed."""

    instances: list = []

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        # Mirror the real ctor's param → attr mapping for the attrs tests assert on.
        self._scope = kwargs.get("scope", "shared")
        self._session_retention = kwargs.get("session_retention", "stop_on_session_end")
        self._session_ttl_seconds = kwargs.get("session_ttl_seconds", 3600)
        self._session_scoped = kwargs.get("scope") == "session"
        self._persist_across_processes = kwargs.get("persist_across_processes", True)
        _RecordingDockerEnv.instances.append(self)


@pytest.fixture
def recording_env(monkeypatch):
    _RecordingDockerEnv.instances = []
    monkeypatch.setattr(terminal_tool_backends, "_DockerEnvironment", _RecordingDockerEnv)
    monkeypatch.setattr(terminal_tool, "_maybe_reap_docker_orphans", lambda cc: None)
    return _RecordingDockerEnv


def _builder_cc(**overrides):
    config = {
        "env_type": "docker",
        "container_persistent": True,
        "docker_container_scope": "session",
        "docker_session_container_retention": "stop_on_session_end",
        "docker_session_container_ttl_seconds": 3600,
        "docker_persist_across_processes": True,
    }
    config.update(overrides)
    return config


def _build(cc, task_id):
    return terminal_tool_backends._build_docker_env(
        env_type="docker", image="i", cwd="/root", timeout=60, cc=cc,
        task_id=task_id, ssh_config=None, host_cwd=None)


def test_builder_session_scope_passes_retention(monkeypatch, recording_env):
    _pin_session(monkeypatch, scope="session", persistent="true", session_key="abc")
    env = _build(_builder_cc(), "session:abc")
    assert env._scope == "session"
    assert env._session_retention == "stop_on_session_end"
    assert env._session_ttl_seconds == 3600
    assert env._session_scoped is True
    assert env._persist_across_processes is True  # reattachable across processes


def test_builder_remove_on_session_end_drops_persist(monkeypatch, recording_env):
    _pin_session(monkeypatch, scope="session", persistent="true")
    env = _build(_builder_cc(docker_session_container_retention="remove_on_session_end"), "session:abc")
    assert env._session_retention == "remove_on_session_end"
    assert env._persist_across_processes is False


def test_builder_ephemeral_isolation_keeps_old_contract(monkeypatch, recording_env):
    """container_persistent: false keeps the legacy kwargs (no scope=session leak)."""
    _pin_session(monkeypatch, scope="shared", persistent="false")
    env = _build(_builder_cc(container_persistent=False), "session:abc")
    assert env._session_scoped is True  # legacy builder marker
    assert env._scope != "session"
    assert env._persist_across_processes is False


def test_builder_session_scope_config_with_ephemeral_env_forces_shared(monkeypatch, recording_env):
    """scope=session in config but container_persistent=false → ephemeral contract wins;
    the ctor must NOT see scope=session or cleanup() would stop-only an ephemeral env."""
    _pin_session(monkeypatch, scope="session", persistent="false")
    env = _build(_builder_cc(container_persistent=False), "session:abc")
    assert env._scope == "shared"
    assert env._persist_across_processes is False


def test_builder_scope_never_leaks_to_default_container(monkeypatch, recording_env):
    """BLOCKER regression (#46041 review): task_id=default (CLI/shared container) must get
    scope=shared even with docker_container_scope: session in the config — the passthrough
    table must not hand scope=session to the ctor."""
    _pin_session(monkeypatch, scope="session", persistent="true", session_key=None)
    env = _build(_builder_cc(), "default")
    assert env._scope == "shared"
    assert env._session_scoped is False


def test_builder_scope_never_leaks_to_rl_override_sandboxes(monkeypatch, recording_env):
    """BLOCKER regression: RL/benchmark override sandboxes keep the shared contract."""
    _pin_session(monkeypatch, scope="session", persistent="true", session_key="abc")
    terminal_tool.register_task_env_overrides("rl-task-1", {"docker_image": "rl-img"})
    env = _build(_builder_cc(), "rl-task-1")
    assert env._scope == "shared"
    assert env._session_scoped is False


def test_builder_shared_default_untouched(recording_env):
    env = _build(_builder_cc(docker_container_scope="shared"), "default")
    assert env._session_scoped is False
    assert env._scope == "shared"


# --- cleanup(): retention semantics ---------------------------------------------


def _make_env(retention, *, session_scope=True):
    """DockerEnvironment skeleton with real cleanup() logic and a fake docker exe."""
    env = docker_env.DockerEnvironment.__new__(docker_env.DockerEnvironment)
    env._scope = "session" if session_scope else "shared"
    env._session_retention = retention
    env._session_ttl_seconds = 3600
    env._session_scoped = session_scope
    env._persistent = True
    env._persist_across_processes = True
    env._container_id = "deadbeef1234567890"
    env._docker_exe = "/usr/bin/docker"
    env._workspace_dir = None
    env._home_dir = None
    env._cleanup_thread = None
    return env


def _verbs(calls):
    return [c[1] for c in calls]


def test_cleanup_stop_on_session_end_stops_without_rm(monkeypatch):
    calls = []

    def _run(cmd, **kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    env = _make_env("stop_on_session_end")
    env.cleanup()
    env.wait_for_cleanup(timeout=5)
    assert _verbs(calls) == ["stop"], f"stop only, no rm; got {_verbs(calls)}"
    assert env._container_id is None  # a fresh ctor re-probes labels and restarts


def test_cleanup_keep_running_leaves_container(monkeypatch):
    calls = []

    def _run(cmd, **kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    env = _make_env("keep_running")
    env.cleanup()
    env.wait_for_cleanup(timeout=5)
    assert calls == [], "keep_running must not touch the container"
    assert env._container_id is None  # in-process handle dropped, container running


def test_cleanup_remove_on_session_end_stops_and_removes(monkeypatch):
    calls = []

    def _run(cmd, **kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    env = _make_env("remove_on_session_end")
    env._persist_across_processes = False  # the builder sets this for remove retention
    env.cleanup()
    env.wait_for_cleanup(timeout=5)
    assert _verbs(calls) == ["stop", "rm"]


def test_cleanup_ephemeral_isolation_still_removes(monkeypatch):
    """Ephemeral per-session envs (scope=shared) keep the stop+rm contract."""
    calls = []

    def _run(cmd, **kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    env = _make_env("stop_on_session_end", session_scope=False)
    env._persist_across_processes = False
    env.cleanup()
    env.wait_for_cleanup(timeout=5)
    assert _verbs(calls) == ["stop", "rm"]


def test_cleanup_force_remove_beats_retention(monkeypatch):
    """User-initiated teardown (force_remove=True) removes even stop-retention containers."""
    calls = []

    def _run(cmd, **kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    env = _make_env("stop_on_session_end")
    env.cleanup(force_remove=True)
    env.wait_for_cleanup(timeout=5)
    assert _verbs(calls) == ["stop", "rm"]


# --- Orphan reaper: spare stopped stop-retention containers ---------------------


def _reaper_mock(monkeypatch, ids_and_retention, finished_age=900):
    """Fake docker: ps returns the ids; inspect answers FinishedAt then the label."""
    old = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(seconds=finished_age)).isoformat()

    def _run(cmd, **kwargs):
        sub = cmd[1]
        if sub == "ps":
            return subprocess.CompletedProcess(cmd, 0, stdout="\n".join(ids_and_retention) + "\n", stderr="")
        if sub == "inspect":
            fmt = cmd[cmd.index("--format") + 1]
            if "FinishedAt" in fmt:
                return subprocess.CompletedProcess(cmd, 0, stdout=old + "\n", stderr="")
            cid = cmd[-1]
            return subprocess.CompletedProcess(
                cmd, 0, stdout=(ids_and_retention.get(cid) or "") + "\n", stderr="")
        if sub == "rm":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)


def test_reaper_skips_stop_on_session_end_containers(monkeypatch):
    _reaper_mock(monkeypatch, {"cid-stop": "stop_on_session_end", "cid-plain": ""})
    removed = docker_env.reap_orphan_containers(
        max_age_seconds=600, profile_filter="default", docker_exe="/usr/bin/docker")
    assert removed == 1, "only the container WITHOUT the stop-retention label may be reaped"


def test_reaper_reaps_remove_and_idle_ttl_containers(monkeypatch):
    _reaper_mock(monkeypatch, {"cid-rm": "remove_on_session_end", "cid-ttl": "idle_ttl"})
    removed = docker_env.reap_orphan_containers(
        max_age_seconds=600, profile_filter="default", docker_exe="/usr/bin/docker")
    assert removed == 2


def test_reaper_ignores_containers_without_label(monkeypatch):
    """Pre-feature containers (no retention label) reap exactly as before."""
    _reaper_mock(monkeypatch, {"cid-old": ""})
    removed = docker_env.reap_orphan_containers(
        max_age_seconds=600, profile_filter="default", docker_exe="/usr/bin/docker")
    assert removed == 1


# --- Idle reaper: idle_ttl envs expire on their own TTL -------------------------


class _FakeSessionEnv:
    def __init__(self, retention, ttl, scoped=True):
        self._scope = "session" if scoped else "shared"
        self._session_scoped = scoped
        self._session_retention = retention
        self._session_ttl_seconds = ttl


def _seed_env(task_id, env, age_seconds):
    with terminal_tool._env_lock:
        terminal_tool._active_environments[task_id] = env
        terminal_tool._last_activity[task_id] = time.time() - age_seconds


def test_idle_reaper_ttl_cutoff_for_idle_ttl(monkeypatch):
    """idle_ttl env younger than its ttl survives; an older one is force-removed."""
    torn_down = []

    def _fake_teardown(env_, task_id, *, force_remove=False, done_msg=""):
        torn_down.append((task_id, force_remove))

    monkeypatch.setattr("tools.terminal_tool_lifecycle._teardown_env", _fake_teardown)
    _seed_env("session:young", _FakeSessionEnv("idle_ttl", 3600), age_seconds=400)
    _seed_env("session:old", _FakeSessionEnv("idle_ttl", 60), age_seconds=400)
    _cleanup_inactive_envs(lifetime_seconds=300)
    assert ("session:old", True) in torn_down
    assert all(tid != "session:young" for tid, _ in torn_down)
    assert get_active_env("session:young") is not None  # survived
    assert get_active_env("session:old") is None  # reaped


def test_idle_reaper_stops_stop_retention_envs_after_lifetime(monkeypatch):
    """stop-retention envs idle out after lifetime_seconds but are torn down NON-forced:
    cleanup() applies stop-only (resumable), the registry stays bounded."""
    torn_down = []

    def _fake_teardown(env_, task_id, *, force_remove=False, done_msg=""):
        torn_down.append((task_id, force_remove))

    monkeypatch.setattr("tools.terminal_tool_lifecycle._teardown_env", _fake_teardown)
    _seed_env("session:stop", _FakeSessionEnv("stop_on_session_end", 3600), age_seconds=400)
    _cleanup_inactive_envs(lifetime_seconds=300)
    assert torn_down == [("session:stop", False)], "stale stop-retention env is torn down (non-forced)"
    assert get_active_env("session:stop") is None


def test_idle_reaper_never_idles_stop_retention_envs(monkeypatch):
    """stop/keep envs inside the idle window are untouched (bg processes survive)."""
    torn_down = []

    def _fake_teardown(env_, task_id, *, force_remove=False, done_msg=""):
        torn_down.append(task_id)

    monkeypatch.setattr("tools.terminal_tool_lifecycle._teardown_env", _fake_teardown)
    for retention in ("stop_on_session_end", "keep_running"):
        _seed_env(f"session:{retention}", _FakeSessionEnv(retention, 3600), age_seconds=100)
    _cleanup_inactive_envs(lifetime_seconds=300)
    assert torn_down == [], "idle-but-open session envs must survive the sweep"
    assert get_active_env("session:stop_on_session_end") is not None
    assert get_active_env("session:keep_running") is not None


def test_idle_reaper_drops_session_close_key_entry(monkeypatch):
    """Review fix (#46041): the reaper pops _active_environments/_last_activity
    directly, so it must ALSO drop the matching _session_close_keys entry —
    otherwise the registry retains one small dict per reaped session in a
    long-lived gateway process. A surviving (non-stale) env keeps its entry."""
    def _fake_teardown(env_, task_id, *, force_remove=False, done_msg=""):
        pass
    monkeypatch.setattr("tools.terminal_tool_lifecycle._teardown_env", _fake_teardown)
    _seed_env("session:reaped", _FakeSessionEnv("stop_on_session_end", 3600), age_seconds=400)
    _seed_env("session:alive", _FakeSessionEnv("stop_on_session_end", 3600), age_seconds=100)
    with terminal_tool._session_close_keys_lock:
        terminal_tool._session_close_keys["session:reaped"] = {
            "session_key": "reaped", "session_id": "20260909_a"}
        terminal_tool._session_close_keys["session:alive"] = {
            "session_key": "alive", "session_id": "20260909_b"}
    _cleanup_inactive_envs(lifetime_seconds=300)
    assert get_active_env("session:reaped") is None
    assert get_active_env("session:alive") is not None
    with terminal_tool._session_close_keys_lock:
        assert "session:reaped" not in terminal_tool._session_close_keys
        assert terminal_tool._session_close_keys["session:alive"] == {
            "session_key": "alive", "session_id": "20260909_b"}


def test_idle_reaper_ephemeral_envs_still_reaped(monkeypatch):
    """BLOCKER regression (#46041 review): ephemeral per-session envs (_session_scoped=True,
    _scope=shared) must keep the LEGACY idle contract — reaped after lifetime_seconds."""
    torn_down = []

    def _fake_teardown(env_, task_id, *, force_remove=False, done_msg=""):
        torn_down.append((task_id, force_remove))

    monkeypatch.setattr("tools.terminal_tool_lifecycle._teardown_env", _fake_teardown)
    _seed_env("session:ephemeral", _FakeSessionEnv("stop_on_session_end", 3600, scoped=False),
              age_seconds=400)
    _cleanup_inactive_envs(lifetime_seconds=300)
    assert torn_down == [("session:ephemeral", False)], "ephemeral envs are not exempt from idling"


def test_idle_reaper_non_session_envs_unchanged(monkeypatch):
    torn_down = []

    def _fake_teardown(env_, task_id, *, force_remove=False, done_msg=""):
        torn_down.append(task_id)

    monkeypatch.setattr("tools.terminal_tool_lifecycle._teardown_env", _fake_teardown)
    _seed_env("default", _FakeSessionEnv("stop_on_session_end", 3600, scoped=False), age_seconds=10_000)
    _cleanup_inactive_envs(lifetime_seconds=300)
    assert torn_down == ["default"]


# --- Close path: cleanup_vm finds envs via the close-key registry ---------------


def test_cleanup_vm_finds_session_keyed_env_from_session_id(monkeypatch):
    """AIAgent.close() passes the durable session_id; the env lives under session:<key>."""
    env = _FakeSessionEnv("stop_on_session_end", 3600)
    with terminal_tool._env_lock:
        terminal_tool._active_environments["session:telegram:123"] = env
    with terminal_tool._session_close_keys_lock:
        terminal_tool._session_close_keys["session:telegram:123"] = {
            "session_key": "telegram:123", "session_id": "20260908_abc123"}
    torn_down = []

    def _fake_teardown(env_, task_id, *, force_remove=False, done_msg=""):
        torn_down.append(task_id)

    monkeypatch.setattr("tools.terminal_tool_lifecycle._teardown_env", _fake_teardown)
    cleanup_vm("20260908_abc123")  # the close path's id — NOT the cache key
    assert torn_down == ["session:telegram:123"]
    assert get_active_env("session:telegram:123") is None


def test_cleanup_vm_raw_key_still_works(monkeypatch):
    """Plain task ids keep the exact legacy behavior."""
    env = _FakeSessionEnv("stop_on_session_end", 3600)
    with terminal_tool._env_lock:
        terminal_tool._active_environments["default"] = env
    torn_down = []

    def _fake_teardown(env_, task_id, *, force_remove=False, done_msg=""):
        torn_down.append(task_id)

    monkeypatch.setattr("tools.terminal_tool_lifecycle._teardown_env", _fake_teardown)
    cleanup_vm("default")
    assert torn_down == ["default"]


def test_child_close_never_tears_down_parent_env(monkeypatch):
    """Delegate child ids are not in the close-key registry: their close() must not
    reach the parent's session-scoped env."""
    env = _FakeSessionEnv("stop_on_session_end", 3600)
    with terminal_tool._env_lock:
        terminal_tool._active_environments["session:parent-key"] = env
    with terminal_tool._session_close_keys_lock:
        terminal_tool._session_close_keys["session:parent-key"] = {
            "session_key": "parent-key", "session_id": "parent-session-id"}
    torn_down = []

    def _fake_teardown(env_, task_id, *, force_remove=False, done_msg=""):
        torn_down.append(task_id)

    monkeypatch.setattr("tools.terminal_tool_lifecycle._teardown_env", _fake_teardown)
    cleanup_vm("child-session-id-xyz")
    assert torn_down == [], "an unregistered child id must not reach the parent's env"
    assert get_active_env("session:parent-key") is env


def test_record_session_close_key_ignores_non_session_keys():
    terminal_tool._record_session_close_key("default")
    terminal_tool._record_session_close_key("profile:work")
    terminal_tool._record_session_close_key("shared:team")
    with terminal_tool._session_close_keys_lock:
        assert terminal_tool._session_close_keys == {}


def test_is_persistent_env_true_for_session_scope_env():
    """Session-scoped containers skip per-turn teardown (their lifetime is the session)."""
    env = _FakeSessionEnv("stop_on_session_end", 3600)
    with terminal_tool._env_lock:
        terminal_tool._active_environments["session:abc"] = env
    assert is_persistent_env("session:abc") is True
