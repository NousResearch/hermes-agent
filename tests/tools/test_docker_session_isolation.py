"""Per-session docker container isolation (docker + container_persistent: false).

Two user-reported bugs on the docker terminal backend (desktop app, sandboxed
cybersecurity profile):

1. **Stale workspace mount leak** — a NEW chat's container carried the
   PREVIOUS session's workspace bind-mounted rw at /workspace, because the
   mount source was the process-global TERMINAL_CWD env var (written by the
   workspace picker, outliving the session that set it) and because every
   session collapsed onto one shared "default" container.

2. **Broken startup cd (exit 126)** — every command tried to
   ``cd /Users/<user>/...`` (a host path recorded as the session cwd by the
   desktop/TUI gateway) inside the container where it doesn't exist.

These tests pin the fix:

* ``container_persistent: false`` + docker ⇒ each session task_id is its own
  container key (fresh container per session); subagents share the parent's
  container via ``register_container_alias``.
* ``container_persistent: true`` (or any other backend) ⇒ legacy shared
  "default" container, unchanged.
* Mount resolution (``_resolve_task_host_cwd``) refuses process-global cwd
  sources under isolation; only the session's own attached workspace mounts.
* ``_resolve_command_cwd`` discards recorded host-path cwds on container
  backends instead of prefixing commands with an un-cd-able host path.
"""

import os

import pytest

from tools import terminal_tool, terminal_tool_backends
from tools.terminal_tool_lifecycle import is_persistent_env


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """Isolate override/alias/cwd-record state and pin docker isolation env."""
    before_overrides = dict(terminal_tool._task_env_overrides)
    terminal_tool._task_env_overrides.clear()
    with terminal_tool._container_alias_lock:
        before_aliases = dict(terminal_tool._container_aliases)
        terminal_tool._container_aliases.clear()
    with terminal_tool._session_cwd_lock:
        before_cwd = dict(terminal_tool._session_cwd)
        terminal_tool._session_cwd.clear()
    # The config→env bridge is one-shot; mark it done so tests control env vars.
    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", True)
    yield
    terminal_tool._task_env_overrides.clear()
    terminal_tool._task_env_overrides.update(before_overrides)
    with terminal_tool._container_alias_lock:
        terminal_tool._container_aliases.clear()
        terminal_tool._container_aliases.update(before_aliases)
    with terminal_tool._session_cwd_lock:
        terminal_tool._session_cwd.clear()
        terminal_tool._session_cwd.update(before_cwd)


def _enable_isolation(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "false")


def _disable_isolation(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "true")


class TestSessionIsolationKeying:
    def test_persistent_true_keeps_shared_default(self, monkeypatch):
        _disable_isolation(monkeypatch)
        assert terminal_tool._resolve_container_task_id("tui:sess-1") == "default"

    def test_persistent_false_keys_by_session(self, monkeypatch):
        _enable_isolation(monkeypatch)
        assert terminal_tool._resolve_container_task_id("tui:sess-1") == "tui:sess-1"

    def test_two_sessions_get_distinct_keys(self, monkeypatch):
        """The reported bug: session B must not land in session A's container."""
        _enable_isolation(monkeypatch)
        a = terminal_tool._resolve_container_task_id("tui:sess-a")
        b = terminal_tool._resolve_container_task_id("tui:sess-b")
        assert a != b

    def test_none_task_id_still_default_under_isolation(self, monkeypatch):
        _enable_isolation(monkeypatch)
        assert terminal_tool._resolve_container_task_id(None) == "default"

    def test_local_backend_unaffected(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "local")
        monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "false")
        assert terminal_tool._resolve_container_task_id("tui:sess-1") == "default"

    def test_rl_override_isolation_still_wins(self, monkeypatch):
        """Image/env_type overrides keep their own key in BOTH modes."""
        _enable_isolation(monkeypatch)
        terminal_tool.register_task_env_overrides(
            "bench-env", {"docker_image": "custom:latest"}
        )
        try:
            assert terminal_tool._resolve_container_task_id("bench-env") == "bench-env"
        finally:
            terminal_tool.clear_task_env_overrides("bench-env")

    def test_subagent_alias_resolves_to_parent(self, monkeypatch):
        """delegate_task children share the PARENT session's container."""
        _enable_isolation(monkeypatch)
        terminal_tool.register_container_alias("subagent-1", "tui:sess-a")
        assert terminal_tool._resolve_container_task_id("subagent-1") == "tui:sess-a"

    def test_subagent_alias_without_parent_falls_to_default(self, monkeypatch):
        _enable_isolation(monkeypatch)
        terminal_tool.register_container_alias("subagent-2", None)
        assert terminal_tool._resolve_container_task_id("subagent-2") == "default"

    def test_nested_alias_chain_resolves(self, monkeypatch):
        """Orchestrator child spawning its own worker: chain to the root session."""
        _enable_isolation(monkeypatch)
        terminal_tool.register_container_alias("child", "tui:sess-a")
        terminal_tool.register_container_alias("grandchild", "child")
        assert terminal_tool._resolve_container_task_id("grandchild") == "tui:sess-a"

    def test_alias_cycle_does_not_hang(self, monkeypatch):
        _enable_isolation(monkeypatch)
        terminal_tool.register_container_alias("x", "y")
        terminal_tool.register_container_alias("y", "x")
        # Any terminating answer is fine; the invariant is no infinite loop.
        assert terminal_tool._resolve_container_task_id("x") in {"x", "y"}


class TestRoutedScopeQualification:
    """A routed profile must qualify session-derived keys (#123989): one multiplexed process,
    two profiles, one colliding session id (header-less API fingerprint, shared DM chat id)
    → distinct sandboxes and cwd records. No routed home → historical raw key."""

    def test_colliding_session_id_is_isolated_per_routed_profile(self, monkeypatch, tmp_path):
        from agent import secret_scope
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override

        _enable_isolation(monkeypatch)
        raw = "api-9a5f7809eec0aac1"
        assert terminal_tool._resolve_container_task_id(raw) == raw  # CLI / standalone: unchanged
        homes = {}
        for name in ("research", "default"):
            homes[name] = tmp_path / "profiles" / name
            homes[name].mkdir(parents=True)
        secret_scope.set_multiplex_active(True)
        try:
            keys, cwds = {}, {}
            for name in ("research", "default", "research"):  # A → B → A
                token = set_hermes_home_override(str(homes[name]))
                try:
                    keys[name] = terminal_tool._resolve_container_task_id(raw)
                    terminal_tool.register_container_alias(f"child-{name}", raw)
                    assert terminal_tool._resolve_container_task_id(f"child-{name}") == keys[name]
                    if name not in cwds:
                        assert terminal_tool.get_session_cwd(raw) is None
                        terminal_tool.record_session_cwd(raw, f"/workspace/{name}")
                    cwds[name] = terminal_tool.get_session_cwd(raw)
                finally:
                    reset_hermes_home_override(token)
            assert keys["research"] == f"profile:research:{raw}"
            assert keys["default"] == f"default:{raw}"
            assert cwds == {"research": "/workspace/research", "default": "/workspace/default"}
            token = set_hermes_home_override(str(homes["default"]))
            try:
                terminal_tool.clear_session_cwd(raw)
            finally:
                reset_hermes_home_override(token)
            token = set_hermes_home_override(str(homes["research"]))
            try:
                assert terminal_tool.get_session_cwd(raw) == "/workspace/research"
            finally:
                reset_hermes_home_override(token)
        finally:
            secret_scope.set_multiplex_active(False)


class TestSessionScopedMountResolution:
    """_resolve_task_host_cwd: the single owner of the cwd→/workspace mount policy."""

    def _config(self, host_cwd="/Users/prev/dev/oldrepo", mount=True):
        return {
            "env_type": "docker",
            "docker_mount_cwd_to_workspace": mount,
            "host_cwd": host_cwd,
        }

    def test_shared_mode_keeps_legacy_host_cwd(self, monkeypatch):
        _disable_isolation(monkeypatch)
        cfg = self._config()
        assert (
            terminal_tool._resolve_task_host_cwd(cfg, "tui:sess-1")
            == "/Users/prev/dev/oldrepo"
        )

    def test_isolation_refuses_process_global_mount(self, monkeypatch, tmp_path):
        """The reported leak: a fresh session with NO attached workspace must
        not inherit the process-global TERMINAL_CWD-derived mount."""
        _enable_isolation(monkeypatch)
        cfg = self._config(host_cwd=str(tmp_path))
        assert terminal_tool._resolve_task_host_cwd(cfg, "tui:sess-new") is None

    def test_isolation_refuses_process_tagged_override(self, monkeypatch, tmp_path):
        """A cwd override tagged cwd_source='process' (gateway env-var fallback)
        is a launch artifact, not a session workspace — never a mount source."""
        _enable_isolation(monkeypatch)
        terminal_tool.register_task_env_overrides(
            "tui:sess-new", {"cwd": str(tmp_path), "cwd_source": "process"}
        )
        cfg = self._config(host_cwd=str(tmp_path))
        assert terminal_tool._resolve_task_host_cwd(cfg, "tui:sess-new") is None

    def test_isolation_mounts_session_attached_workspace(self, monkeypatch, tmp_path):
        """A workspace the user attached to THIS session does mount."""
        _enable_isolation(monkeypatch)
        ws = tmp_path / "attached"
        ws.mkdir()
        terminal_tool.register_task_env_overrides(
            "tui:sess-new", {"cwd": str(ws), "cwd_source": "session"}
        )
        cfg = self._config(host_cwd="/Users/prev/dev/oldrepo")
        assert terminal_tool._resolve_task_host_cwd(cfg, "tui:sess-new") == str(ws)

    def test_isolation_rejects_nonexistent_session_dir(self, monkeypatch, tmp_path):
        _enable_isolation(monkeypatch)
        terminal_tool.register_task_env_overrides(
            "tui:sess-new",
            {"cwd": str(tmp_path / "gone"), "cwd_source": "session"},
        )
        cfg = self._config()
        assert terminal_tool._resolve_task_host_cwd(cfg, "tui:sess-new") is None

    def test_isolation_rejects_in_container_path_as_mount(self, monkeypatch):
        _enable_isolation(monkeypatch)
        terminal_tool.register_task_env_overrides(
            "tui:sess-new", {"cwd": "/workspace", "cwd_source": "session"}
        )
        cfg = self._config()
        assert terminal_tool._resolve_task_host_cwd(cfg, "tui:sess-new") is None

    def test_mount_flag_off_means_no_mount(self, monkeypatch, tmp_path):
        _enable_isolation(monkeypatch)
        ws = tmp_path / "attached"
        ws.mkdir()
        terminal_tool.register_task_env_overrides(
            "tui:sess-new", {"cwd": str(ws), "cwd_source": "session"}
        )
        cfg = self._config(mount=False)
        assert terminal_tool._resolve_task_host_cwd(cfg, "tui:sess-new") is None

    def test_default_task_keeps_legacy_behavior_under_isolation(self, monkeypatch):
        """The single-session CLI parent ("default") keeps the legacy mount."""
        _enable_isolation(monkeypatch)
        cfg = self._config()
        assert (
            terminal_tool._resolve_task_host_cwd(cfg, None)
            == "/Users/prev/dev/oldrepo"
        )

    def test_non_docker_backend_never_mounts(self, monkeypatch):
        _disable_isolation(monkeypatch)
        cfg = self._config()
        cfg["env_type"] = "modal"
        assert terminal_tool._resolve_task_host_cwd(cfg, "t") is None


class TestRecordedHostCwdDiscardedOnContainers:
    """_resolve_command_cwd must not cd to a recorded HOST path in a sandbox.

    The reported exit-126 bug: the desktop gateway records the host launch
    dir as the session cwd; every subsequent command then ran
    ``cd /Users/<user>/dev/<repo> && <cmd>`` inside the container.
    """

    def test_host_record_discarded_for_docker(self):
        terminal_tool.record_session_cwd("sess-1", "/Users/me/dev/repo")
        cwd = terminal_tool._resolve_command_cwd(
            workdir=None, default_cwd="/workspace",
            session_key="sess-1", env_type="docker",
        )
        assert cwd == "/workspace"

    def test_container_record_honored_for_docker(self):
        """A legitimate in-container cd is the session's state — keep it."""
        terminal_tool.record_session_cwd("sess-1", "/workspace/subdir")
        cwd = terminal_tool._resolve_command_cwd(
            workdir=None, default_cwd="/workspace",
            session_key="sess-1", env_type="docker",
        )
        assert cwd == "/workspace/subdir"

    def test_host_record_kept_for_local_backend(self):
        terminal_tool.record_session_cwd("sess-1", "/home/me/project")
        cwd = terminal_tool._resolve_command_cwd(
            workdir=None, default_cwd="/anything",
            session_key="sess-1", env_type="local",
        )
        assert cwd == "/home/me/project"

    def test_explicit_workdir_still_wins(self):
        terminal_tool.record_session_cwd("sess-1", "/workspace/a")
        cwd = terminal_tool._resolve_command_cwd(
            workdir="/workspace/b", default_cwd="/workspace",
            session_key="sess-1", env_type="docker",
        )
        assert cwd == "/workspace/b"

    def test_no_env_type_keeps_previous_behavior(self):
        """Callers that don't pass env_type (legacy sites) are unchanged."""
        terminal_tool.record_session_cwd("sess-1", "/home/me/project")
        cwd = terminal_tool._resolve_command_cwd(
            workdir=None, default_cwd="/fallback", session_key="sess-1",
        )
        assert cwd == "/home/me/project"


class TestSessionScopedContainerLifecycle:
    def test_session_scoped_env_counts_persistent_for_turn_teardown(self, monkeypatch):
        """Session-scoped containers survive between turns (torn down at
        session close, not per-turn)."""
        _enable_isolation(monkeypatch)

        class _FakeEnv:
            _session_scoped = True
            _persistent = False

        monkeypatch.setitem(
            terminal_tool._active_environments, "tui:sess-1", _FakeEnv()
        )
        try:
            assert is_persistent_env("tui:sess-1") is True
        finally:
            terminal_tool._active_environments.pop("tui:sess-1", None)

    def test_create_environment_marks_session_scoped(self, monkeypatch):
        """_create_environment disables cross-process persist for session
        containers and stamps the marker the lifecycle paths read."""
        _enable_isolation(monkeypatch)
        captured = {}

        class _FakeDockerEnv:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(terminal_tool_backends, "_DockerEnvironment", _FakeDockerEnv)
        monkeypatch.setattr(terminal_tool, "_maybe_reap_docker_orphans", lambda cc: None)

        env = terminal_tool_backends._create_environment(
            env_type="docker", image="img:1", cwd="/workspace", timeout=60,
            container_config={"docker_persist_across_processes": True},
            task_id="tui:sess-1",
        )
        assert captured["persist_across_processes"] is False
        assert getattr(env, "_session_scoped") is True

    def test_create_environment_default_task_not_session_scoped(self, monkeypatch):
        _enable_isolation(monkeypatch)
        captured = {}

        class _FakeDockerEnv:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(terminal_tool_backends, "_DockerEnvironment", _FakeDockerEnv)
        monkeypatch.setattr(terminal_tool, "_maybe_reap_docker_orphans", lambda cc: None)

        env = terminal_tool_backends._create_environment(
            env_type="docker", image="img:1", cwd="/workspace", timeout=60,
            container_config={"docker_persist_across_processes": True},
            task_id="default",
        )
        assert captured["persist_across_processes"] is True
        assert getattr(env, "_session_scoped", False) is False


class TestRoutedScopeIdentityContract:
    """The qualified key must be the ONLY identity a routed profile can reach an environment by.
    Follow-up to #123989 (#126157 review, R1/R3): a routed miss fell back to the launch profile's
    raw slot, and the Docker builder qualified an already-qualified rollout key a second time."""

    def _routed(self, tmp_path, name="research"):
        from agent import secret_scope
        from hermes_constants import set_hermes_home_override

        home = tmp_path / "profiles" / name
        home.mkdir(parents=True)
        secret_scope.set_multiplex_active(True)
        return set_hermes_home_override(str(home))

    def _launch(self, token):
        from agent import secret_scope
        from hermes_constants import reset_hermes_home_override

        reset_hermes_home_override(token)
        secret_scope.set_multiplex_active(False)

    def test_a_routed_miss_never_falls_back_to_the_launch_profiles_raw_slot(self, monkeypatch, tmp_path):
        from types import SimpleNamespace

        _enable_isolation(monkeypatch)
        raw = "same-authorized-session"
        launch_env = SimpleNamespace(env_type="local", cwd="/launch")
        monkeypatch.setitem(terminal_tool._active_environments, raw, launch_env)  # profile A, raw key
        token = self._routed(tmp_path)
        try:
            eff = terminal_tool._resolve_container_task_id(raw)
            assert eff == f"profile:research:{raw}"
            with terminal_tool._env_lock:
                assert terminal_tool._lookup_active_env(eff, raw) is None, "B's cold miss handed out A's sandbox"
            terminal_tool.register_task_env_overrides(raw, {"cwd": "/tmp/research-workspace"})
            assert launch_env.cwd == "/launch", "B's cwd registration mutated A's live environment"
            assert terminal_tool.get_session_cwd(raw) == "/tmp/research-workspace"

            routed_env = SimpleNamespace(env_type="local", cwd="/research")
            monkeypatch.setitem(terminal_tool._active_environments, eff, routed_env)
            with terminal_tool._env_lock:
                assert terminal_tool._lookup_active_env(eff, raw) is routed_env
            terminal_tool.register_task_env_overrides(raw, {"cwd": "/tmp/research-2"})
            assert (routed_env.cwd, launch_env.cwd) == ("/tmp/research-2", "/launch")
        finally:
            self._launch(token)
            for key in (raw, f"profile:research:{raw}"):
                terminal_tool._last_activity.pop(key, None)
        # Back on the launch profile the raw slot is its own: same-owner sharing is untouched.
        with terminal_tool._env_lock:
            assert terminal_tool._lookup_active_env(raw, raw) is launch_env
        assert terminal_tool.get_session_cwd(raw) is None

    def test_an_override_rollout_under_a_routed_profile_keeps_its_lifecycle(self, monkeypatch, tmp_path):
        _enable_isolation(monkeypatch)
        built = []

        class _FakeDockerEnv:
            def __init__(self, **kwargs):
                built.append(kwargs)

        monkeypatch.setattr(terminal_tool_backends, "_DockerEnvironment", _FakeDockerEnv)
        monkeypatch.setattr(terminal_tool, "_maybe_reap_docker_orphans", lambda cc: None)
        monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
        config = {"env_type": "docker", "docker_persist_across_processes": True}

        def _plan(eff, image):
            return terminal_tool._ExecPlan(
                config=config, env_type="docker", effective_task_id=eff, image=image,
                cwd="/workspace", host_cwd=None, effective_timeout=60)

        token = self._routed(tmp_path)
        created = []
        try:
            terminal_tool.register_task_env_overrides("rollout", {"docker_image": "probe-image"})
            rollout_key = terminal_tool._resolve_container_task_id("rollout")
            assert rollout_key == "profile:research:rollout"
            created.append(rollout_key)
            rollout = terminal_tool._acquire_env(_plan(rollout_key, "probe-image"), "rollout")
            assert built[-1]["task_id"] == rollout_key and built[-1]["image"] == "probe-image"
            assert built[-1]["persist_across_processes"] is True, "the rollout became a session sandbox"
            assert getattr(rollout, "_session_scoped", False) is False

            chat_key = terminal_tool._resolve_container_task_id("chat-1")  # ordinary session: control
            created.append(chat_key)
            chat = terminal_tool._acquire_env(_plan(chat_key, "img:1"), "chat-1")
            assert built[-1]["persist_across_processes"] is False
            assert getattr(chat, "_session_scoped", False) is True
        finally:
            self._launch(token)
            for key in created:
                terminal_tool._active_environments.pop(key, None)
                terminal_tool._last_activity.pop(key, None)

    @pytest.mark.parametrize("routed", [False, True], ids=["launch", "routed"])
    @pytest.mark.parametrize("first", ["child", "parent"])
    def test_a_rollout_keeps_its_lifecycle_whoever_creates_it_first(self, monkeypatch, tmp_path, routed, first):
        """A delegated child aliases to its parent's rollout (image override); whichever of the two
        reaches the sandbox first creates the PARENT's environment, so its lifetime must follow the
        parent's registration (persistent across processes), not the initiating id's."""
        from types import SimpleNamespace

        from tools.terminal_tool_lifecycle import ensure_task_env

        _enable_isolation(monkeypatch)
        monkeypatch.setenv("TERMINAL_DOCKER_PERSIST_ACROSS_PROCESSES", "true")
        for name in ("_active_environments", "_last_activity", "_creation_locks"):
            monkeypatch.setattr(terminal_tool, name, {})
        monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
        monkeypatch.setattr(terminal_tool, "_maybe_reap_docker_orphans", lambda cc: None)
        monkeypatch.setattr(terminal_tool_backends, "_DockerEnvironment", lambda **kw: SimpleNamespace(**kw))
        token = self._routed(tmp_path) if routed else None
        try:
            terminal_tool.register_task_env_overrides("rollout", {"docker_image": "rollout-image"})
            terminal_tool.register_container_alias("rollout-child", "rollout")
            order = ("rollout-child", "rollout") if first == "child" else ("rollout", "rollout-child")
            env = ensure_task_env(order[0])
            assert env is not None and env.image == "rollout-image"
            assert env.task_id == terminal_tool._resolve_container_task_id("rollout")
            assert ensure_task_env(order[1]) is env
            assert env.persist_across_processes is True, "the rollout became a session sandbox"
            assert getattr(env, "_session_scoped", False) is False

            chat = ensure_task_env("chat-1")  # ordinary session on the same profile: still removable
            assert chat.persist_across_processes is False and chat._session_scoped is True
        finally:
            if token is not None:
                self._launch(token)

    def test_raw_readers_teardown_and_eviction_stay_within_their_owner(self, monkeypatch, tmp_path):
        """Every reader, evictor and teardown that takes a RAW id resolves it under its owner: a
        routed profile's failed startup, degraded eviction or session close must not read, evict or
        tear down the launch profile's environment that shares the id."""
        from types import SimpleNamespace

        from tools.terminal_tool_lifecycle import (
            _evict_environment_for_task, cleanup_vm, ensure_task_env, get_active_env, is_persistent_env)

        _enable_isolation(monkeypatch)
        raw = "same-session"
        cleaned = []
        for name in ("_active_environments", "_last_activity", "_creation_locks"):
            monkeypatch.setattr(terminal_tool, name, {})
        monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
        monkeypatch.setattr(terminal_tool, "_maybe_reap_docker_orphans", lambda cc: None)

        def _docker(**kwargs):
            return SimpleNamespace(cleanup=lambda: cleaned.append(kwargs["task_id"]), **kwargs)

        monkeypatch.setattr(terminal_tool_backends, "_DockerEnvironment", _docker)
        launch_env = ensure_task_env(raw)  # profile A: cached under the bare id
        assert launch_env.task_id == raw
        routed_key = f"profile:research:{raw}"
        token = self._routed(tmp_path)
        try:
            assert get_active_env(raw) is None, "B's cold miss read A's environment"
            assert is_persistent_env(raw) is False
            _evict_environment_for_task(raw)  # B's degraded-backend eviction
            cleanup_vm(raw)  # B's session close
            assert cleaned == [], "B's failure path tore down A's environment"
            assert terminal_tool._active_environments == {raw: launch_env} and raw in terminal_tool._last_activity

            routed_env = ensure_task_env(raw)  # B's own: created and re-found by its resolved key
            assert routed_env is not launch_env and routed_env.task_id == routed_key
            assert get_active_env(raw) is routed_env
            _evict_environment_for_task(raw)
            assert cleaned == [routed_key] and get_active_env(raw) is None
            ensure_task_env(raw)
            cleanup_vm(raw)
            assert cleaned == [routed_key, routed_key] and get_active_env(raw) is None
        finally:
            self._launch(token)
        assert get_active_env(raw) is launch_env and cleaned == [routed_key, routed_key]

    @pytest.mark.parametrize("routed", [False, True], ids=["launch", "routed"])
    def test_the_exit_sweep_tears_down_every_environment_in_any_scope(self, monkeypatch, tmp_path, routed):
        """``cleanup_all_environments`` holds resolved cache keys. Passing them through the raw-id
        ``cleanup_vm`` re-qualified them under a routed caller's scope and tore nothing down, so a
        sweep running in a routed profile's context left every sandbox resident."""
        from types import SimpleNamespace

        import tools.environments.base as env_base
        from tools import terminal_tool_lifecycle

        for name in ("_active_environments", "_last_activity", "_creation_locks"):
            monkeypatch.setattr(terminal_tool, name, {})
        monkeypatch.setattr(env_base, "kill_live_foreground_processes", lambda: None)
        monkeypatch.setattr(terminal_tool_lifecycle, "_scratch_paths", lambda: [])
        cleaned = []
        keys = ("default", "tui:sess-1", "profile:research", "profile:research:tui:sess-2")
        for key in keys:
            terminal_tool._active_environments[key] = SimpleNamespace(cleanup=lambda key=key: cleaned.append(key))
        token = self._routed(tmp_path) if routed else None
        try:
            terminal_tool_lifecycle.cleanup_all_environments()
        finally:
            if token is not None:
                self._launch(token)
        assert sorted(cleaned) == sorted(keys), "the sweep left sandboxes resident"
        assert terminal_tool._active_environments == {}
