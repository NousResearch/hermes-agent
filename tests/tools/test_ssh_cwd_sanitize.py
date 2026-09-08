"""Host-path cwd sanitization on the ssh backend.

``ssh`` is the one backend whose cwd is resolved by a shell on *another*
machine. A path taken from this host is not merely useless there: ``cd`` fails
and the command returns 126 before it runs. The guards that already exist for
that mistake are all scoped to container backends, and ``ssh`` is not one of
them -- and the container predicate is the wrong one for it anyway, since a
container must reject ``/Users/me`` while over ssh that is very likely exactly
where the caller means to run.

Measured on a Windows host driving a macOS peer, 2026-09-07: a worker exported
``TERMINAL_CWD`` as its own Windows workspace, and ``sw_vers`` and
``sysctl -n hw.model`` both came back ``exit 126`` until the caller passed an
explicit ``workdir``.

There are five places a cwd reaches a command or an environment builder, and
all five needed the guard:

  1. ``_get_env_config``      TERMINAL_CWD               -> env creation
  2. ``terminal_tool``        per-task cwd override      -> env creation
  3. ``_resolve_command_cwd`` session cwd record         -> every command
  4. ``file_tools``           its own builder            -> its own env
  5. ``code_execution_tool``  its own builder            -> its own env

Sites 3-5 are the ones that are easy to miss. 3 carries the failure that
sustains itself: ``cd`` dies before the cwd marker is printed, so the record is
never corrected and the same value is re-recorded after every command. 4 and 5
build their own environments and do not route through ``_resolve_command_cwd``.
"""

import tools.terminal_tool as tt
import tools.terminal_tool_config as cfg


class TestIsUnusableSSHCwd:
    def test_the_measured_case(self):
        assert cfg._is_unusable_ssh_cwd(
            r"D:\agent\workspaces\task") is True

    def test_any_drive_either_slash(self):
        for p in (r"C:\Users\me", "C:/Users/me", r"D:\work", "d:/work", r"Z:\x"):
            assert cfg._is_unusable_ssh_cwd(p) is True, p

    def test_relative_paths(self):
        for p in (".", "..", "src/", "work/project"):
            assert cfg._is_unusable_ssh_cwd(p) is True, p

    def test_tilde_is_the_peers_own_home(self):
        assert cfg._is_unusable_ssh_cwd("~") is False
        assert cfg._is_unusable_ssh_cwd("~/work") is False

    def test_absolute_posix_paths_are_kept(self):
        # We do not guess whether these exist on the peer; only reject what
        # cannot work. /Users/me is a host path for a container and a perfectly
        # good remote path here -- which is why ssh needs its own predicate.
        for p in ("/Users/me", "/home/joon", "/opt/data", "/"):
            assert cfg._is_unusable_ssh_cwd(p) is False, p

    def test_empty_is_not_flagged(self):
        assert cfg._is_unusable_ssh_cwd("") is False


class TestSessionRecordIsSanitized:
    """Site 3 -- the one that carries the self-sustaining failure."""

    def test_host_record_is_dropped_on_ssh(self):
        tt.record_session_cwd("sess-ssh-rec", r"D:\hermes\workspaces\task")
        try:
            assert tt._resolve_command_cwd(
                workdir=None, default_cwd="~",
                env_type="ssh", session_key="sess-ssh-rec") == "~"
        finally:
            tt.clear_session_cwd("sess-ssh-rec")

    def test_remote_record_is_kept(self):
        tt.record_session_cwd("sess-ssh-ok", "/Users/me/work")
        try:
            assert tt._resolve_command_cwd(
                workdir=None, default_cwd="~",
                env_type="ssh", session_key="sess-ssh-ok") == "/Users/me/work"
        finally:
            tt.clear_session_cwd("sess-ssh-ok")

    def test_explicit_workdir_still_wins(self):
        # workdir= is the caller saying where this must run; the guard must not
        # second-guess it. Passing one is also how the bug was worked around
        # before the fix existed.
        tt.record_session_cwd("sess-ssh-wd", r"D:\hermes")
        try:
            assert tt._resolve_command_cwd(
                workdir="/Users/me", default_cwd="~",
                env_type="ssh", session_key="sess-ssh-wd") == "/Users/me"
        finally:
            tt.clear_session_cwd("sess-ssh-wd")

    def test_other_backends_are_untouched(self):
        tt.record_session_cwd("sess-other", "/workspace/task")
        try:
            for backend in ("local", "docker", "modal"):
                assert tt._resolve_command_cwd(
                    workdir=None, default_cwd="/root",
                    env_type=backend, session_key="sess-other") == "/workspace/task", backend
        finally:
            tt.clear_session_cwd("sess-other")


class TestEnvironmentBuildersAreSanitized:
    """Sites 1, 2, 4 and 5 -- everything that builds an environment.

    Unit-testing the predicate is not enough: the first version of a fix like
    this can wire a guard that no test exercises, and the suite stays green.
    Each of these drives a real builder and asserts on the cwd it hands out.
    """

    @staticmethod
    def _config(cwd="~"):
        return {
            "env_type": "ssh", "docker_image": "", "singularity_image": "",
            "modal_image": "", "daytona_image": "",
            "cwd": cwd, "host_cwd": None, "timeout": 180, "lifetime_seconds": 300,
            "container_cpu": 1, "container_memory": 5120, "container_disk": 51200,
            "container_persistent": True, "docker_volumes": [], "docker_env": {},
            "docker_extra_args": [], "docker_forward_env": [],
            "docker_run_as_host_user": False, "docker_network": True,
            "docker_mount_cwd_to_workspace": False, "modal_mode": "auto",
            "local_persistent": False,
            "ssh_host": "peer", "ssh_user": "me", "ssh_port": 22, "ssh_key": "",
        }

    def _patch_builder(self, monkeypatch, captured, config):
        import tools.terminal_tool_backends as backends

        class _DummyEnv:
            cwd = config["cwd"]

            def execute(self, *a, **k):
                captured.setdefault("execute_cwds", []).append(k.get("cwd"))
                return {"output": "", "exit_code": 0}

        def fake_create_environment(**kwargs):
            captured["create_cwd"] = kwargs.get("cwd")
            return _DummyEnv()

        monkeypatch.setattr(tt, "_get_env_config", lambda: config)
        monkeypatch.setattr(backends, "_create_environment", fake_create_environment)
        monkeypatch.setattr(tt, "_start_cleanup_thread", lambda: None)
        monkeypatch.setattr(tt, "_active_environments", {})
        monkeypatch.setattr(tt, "_last_activity", {})

    def _drive(self, monkeypatch, build, override_cwd, extra_overrides=None):
        captured = {}
        config = self._config()
        self._patch_builder(monkeypatch, captured, config)
        task_id = "sess-ssh-builder"
        overrides = {"cwd": override_cwd}
        overrides.update(extra_overrides or {})
        tt.register_task_env_overrides(task_id, overrides)
        try:
            build(task_id)
        finally:
            tt.clear_task_env_overrides(task_id)
            tt.clear_session_cwd(task_id)
            tt._active_environments.pop(task_id, None)
            tt._active_environments.pop("default", None)
        return captured

    def test_file_tools_builder(self, monkeypatch):
        import tools.file_tools as ft
        cap = self._drive(monkeypatch, ft._get_file_ops, r"D:\hermes\workspaces\task")
        assert cap["create_cwd"] == "~", (
            f"file_tools built an ssh environment in {cap['create_cwd']!r}; "
            "every command it runs would die in `cd` with 126.")

    def test_file_tools_keeps_a_remote_path(self, monkeypatch):
        import tools.file_tools as ft
        cap = self._drive(monkeypatch, ft._get_file_ops, "/Users/me/work")
        assert cap["create_cwd"] == "/Users/me/work"

    # execute_code reads overrides under the COLLAPSED container id, so a
    # CWD-only override never reaches it -- it collapses to the shared
    # "default" key. An isolation key (env_type / *_image, as RL and benchmark
    # harnesses register) keeps the raw id, and that is the path on which a
    # host cwd does reach this builder today. Both cases are pinned.
    _ISOLATION = {"env_type": "ssh"}

    def test_code_execution_builder(self, monkeypatch):
        import tools.code_execution_tool as cet
        cap = self._drive(monkeypatch, cet._get_or_create_env,
                          r"D:\hermes\workspaces\task", self._ISOLATION)
        assert cap["create_cwd"] == "~"

    def test_code_execution_keeps_a_remote_path(self, monkeypatch):
        import tools.code_execution_tool as cet
        cap = self._drive(monkeypatch, cet._get_or_create_env,
                          "/Users/me/work", self._ISOLATION)
        assert cap["create_cwd"] == "/Users/me/work"

    def test_code_execution_cwd_only_override_does_not_reach_it_today(self, monkeypatch):
        # Documents the collapsed-lookup behaviour this branch does not change:
        # without an isolation key the override is not seen at all, so the
        # builder falls back to the config cwd. If that lookup is ever unified
        # with the terminal and file layers, this test should start failing and
        # the expectation becomes "/Users/me/work".
        import tools.code_execution_tool as cet
        cap = self._drive(monkeypatch, cet._get_or_create_env, "/Users/me/work")
        assert cap["create_cwd"] == "~"


class TestConfigCwdIsSanitized:
    """Site 1 -- TERMINAL_CWD itself."""

    def test_host_terminal_cwd_falls_back_to_remote_home(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "ssh")
        monkeypatch.setenv("TERMINAL_SSH_HOST", "peer")
        monkeypatch.setenv("TERMINAL_SSH_USER", "me")
        monkeypatch.setenv("TERMINAL_CWD", r"D:\hermes\workspaces\task")
        assert tt._get_env_config()["cwd"] == "~"

    def test_remote_terminal_cwd_is_kept(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "ssh")
        monkeypatch.setenv("TERMINAL_SSH_HOST", "peer")
        monkeypatch.setenv("TERMINAL_SSH_USER", "me")
        monkeypatch.setenv("TERMINAL_CWD", "/Users/me/work")
        assert tt._get_env_config()["cwd"] == "/Users/me/work"


class TestPerCallOverrideIsSanitized:
    """Site 2 -- the per-task override resolved on every terminal call.

    This one is easy to wire and leave untested: the predicate tests and the
    builder tests all stay green with this guard disabled. Mutation testing is
    what surfaced that, so it gets its own case.
    """

    @staticmethod
    def _plan(monkeypatch, override_cwd, config_cwd="~"):
        config = {
            "env_type": "ssh", "docker_image": "", "singularity_image": "",
            "modal_image": "", "daytona_image": "",
            "cwd": config_cwd, "host_cwd": None, "timeout": 180,
            "lifetime_seconds": 300, "container_cpu": 1, "container_memory": 5120,
            "container_disk": 51200, "container_persistent": True,
            "docker_volumes": [], "docker_env": {}, "docker_extra_args": [],
            "docker_forward_env": [], "docker_run_as_host_user": False,
            "docker_network": True, "docker_mount_cwd_to_workspace": False,
            "modal_mode": "auto", "local_persistent": False,
            "ssh_host": "peer", "ssh_user": "me", "ssh_port": 22, "ssh_key": "",
        }
        monkeypatch.setattr(tt, "_get_env_config", lambda: config)
        task_id = "sess-ssh-plan"
        tt.register_task_env_overrides(task_id, {"cwd": override_cwd})
        try:
            plan = tt._plan_execution(
                "pwd", task_id=task_id, timeout=None,
                background=False, _host_local=False,
            )
        finally:
            tt.clear_task_env_overrides(task_id)
            tt.clear_session_cwd(task_id)
        return plan.cwd

    def test_host_override_is_replaced_by_the_config_cwd(self, monkeypatch):
        assert self._plan(monkeypatch, r"D:\hermes\workspaces\task") == "~"

    def test_relative_override_is_replaced(self, monkeypatch):
        assert self._plan(monkeypatch, "src/") == "~"

    def test_remote_override_is_preserved(self, monkeypatch):
        assert self._plan(monkeypatch, "/Users/me/work") == "/Users/me/work"

    def test_tilde_override_is_preserved(self, monkeypatch):
        assert self._plan(monkeypatch, "~/work") == "~/work"
