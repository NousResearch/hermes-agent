"""Regression tests for host-path cwd sanitization on the ssh backend.

``ssh`` is the one backend whose cwd is resolved by a shell on *another*
machine, so a path taken from the Hermes host is not merely useless there --
``cd`` fails and the command returns 126 before it ever runs.

``tools/terminal_tool.py`` already guards two code paths against exactly this
shape of mistake, but both were scoped to ``_CONTAINER_BACKENDS`` and ``ssh``
is not one of them:

  1. ``_get_env_config()`` sanitizes the ``TERMINAL_CWD``-derived ``config["cwd"]``.
  2. ``terminal_tool()`` re-applies the guard to a *per-task cwd override*,
     which wins over ``config["cwd"]``.

Measured on SSJOON 2026-09-07: the kanban worker sets ``TERMINAL_CWD`` to its
own Windows workspace (``D:\\hermes\\kanban\\...\\workspaces\\<task_id>``) while
the ssh peer is a Mac.  ``sw_vers`` and ``sysctl -n hw.model`` both came back
``exit 126`` until the caller passed an explicit workdir.  ``_HOST_CWD_PREFIXES``
would not have caught it either -- it lists only the ``C:`` drive.

These tests pin ``_is_unusable_ssh_cwd()`` so neither path can regress, and
pin the two things it must *not* reject: ``~`` (the peer's own home, which
``_is_ssh_remote_tilde_cwd`` deliberately leaves for the remote shell) and
absolute POSIX paths that may well exist on the peer.
"""

import tools.terminal_tool as tt


class TestIsUnusableSSHCwd:
    def test_the_measured_case_windows_workspace_path(self):
        # The exact value the kanban worker exported on SSJOON.
        assert tt._is_unusable_ssh_cwd(
            r"D:\hermes\kanban\boards\hermes-multinode\workspaces\t_27e56ac3"
        ) is True

    def test_any_drive_letter_not_just_c(self):
        # _HOST_CWD_PREFIXES only lists C:, which is why the measured D: path
        # slipped through every existing guard.
        for path in (r"C:\Users\me", "C:/Users/me", r"D:\hermes", "d:/hermes", r"Z:\x"):
            assert tt._is_unusable_ssh_cwd(path) is True, path

    def test_backslash_anywhere_is_rejected(self):
        # A separator the peer's shell does not use.
        assert tt._is_unusable_ssh_cwd(r"\\server\share") is True
        assert tt._is_unusable_ssh_cwd(r"some\relative") is True

    def test_relative_paths_rejected(self):
        for path in (".", "..", "src/", "work/uservice"):
            assert tt._is_unusable_ssh_cwd(path) is True, path

    def test_tilde_is_kept(self):
        # The remote shell expands these itself; see _is_ssh_remote_tilde_cwd.
        assert tt._is_unusable_ssh_cwd("~") is False
        assert tt._is_unusable_ssh_cwd("~/work/uservice/teamwork") is False

    def test_absolute_posix_paths_are_kept(self):
        # We do not guess whether these exist on the peer -- only reject what
        # cannot possibly work.
        for path in ("/Users/ftfuture", "/home/joon", "/opt/data", "/"):
            assert tt._is_unusable_ssh_cwd(path) is False, path

    def test_empty_is_not_flagged(self):
        assert tt._is_unusable_ssh_cwd("") is False


class TestSSHCwdGuardIsWiredIn:
    """The helper is only worth having if both call sites use it."""

    def test_config_path_falls_back_to_remote_home(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "ssh")
        monkeypatch.setenv("TERMINAL_SSH_HOST", "m5")
        monkeypatch.setenv("TERMINAL_SSH_USER", "ftfuture")
        monkeypatch.setenv(
            "TERMINAL_CWD",
            r"D:\hermes\kanban\boards\hermes-multinode\workspaces\t_27e56ac3",
        )
        config = tt._get_env_config()
        assert config["env_type"] == "ssh"
        assert config["cwd"] == "~"

    def test_config_path_keeps_a_legitimate_remote_path(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "ssh")
        monkeypatch.setenv("TERMINAL_SSH_HOST", "m5")
        monkeypatch.setenv("TERMINAL_SSH_USER", "ftfuture")
        monkeypatch.setenv("TERMINAL_CWD", "/Users/ftfuture/work/uservice/teamwork")
        config = tt._get_env_config()
        assert config["cwd"] == "/Users/ftfuture/work/uservice/teamwork"

    def test_container_guard_is_untouched(self):
        # This change must not widen or narrow the container behaviour.
        assert tt._is_unusable_container_cwd(r"C:\Users\me") is True
        assert tt._is_unusable_container_cwd("/workspace") is False


class TestOverrideCwdSanitizedForSSH:
    """E2E pin for the *second* code path.

    A per-task cwd override wins over ``config["cwd"]``, so sanitizing only the
    config path leaves the hole open.  Mutation testing proved this is not
    theoretical: with the override guard disabled, every unit test above still
    passed.  These drive ``terminal_tool()`` and assert on the cwd that
    actually reaches the environment builder.
    """

    def _run_and_capture_cwd(self, monkeypatch, override_cwd, config_cwd="~"):
        captured = {}

        config = {
            "env_type": "ssh",
            "ssh_config": {"host": "m5", "user": "ftfuture", "port": 22, "key": ""},
            "docker_image": "",
            "cwd": config_cwd,
            "host_cwd": None,
            "timeout": 180,
            "lifetime_seconds": 300,
            "container_cpu": 1,
            "container_memory": 5120,
            "container_disk": 51200,
            "container_persistent": True,
            "docker_volumes": [],
            "docker_env": {},
            "docker_extra_args": [],
            "docker_mount_cwd_to_workspace": False,
            "docker_run_as_host_user": False,
            "docker_forward_env": [],
            "modal_mode": "auto",
        }

        class _DummyEnv:
            cwd = config_cwd

            def execute(self, *a, **k):
                return {"output": "", "exit_code": 0}

        def fake_create_environment(env_type, image, cwd, timeout, **kwargs):
            captured["cwd"] = cwd
            return _DummyEnv()

        monkeypatch.setattr(tt, "_get_env_config", lambda: config)
        monkeypatch.setattr(tt, "_start_cleanup_thread", lambda: None)
        monkeypatch.setattr(tt, "_check_all_guards", lambda *a, **k: {"approved": True})
        monkeypatch.setattr(tt, "_create_environment", fake_create_environment)
        monkeypatch.setattr(tt, "_active_environments", {})
        monkeypatch.setattr(tt, "_last_activity", {})

        task_id = "sess-ssh-host-cwd"
        tt.register_task_env_overrides(task_id, {"cwd": override_cwd})
        try:
            tt.terminal_tool(command="pwd", task_id=task_id)
        finally:
            tt.clear_task_env_overrides(task_id)
            tt._active_environments.pop(task_id, None)
            tt._active_environments.pop("default", None)
        return captured.get("cwd")

    def test_the_measured_case_does_not_reach_the_peer(self, monkeypatch):
        # The kanban worker's own Windows workspace, registered as an override.
        cwd = self._run_and_capture_cwd(
            monkeypatch,
            r"D:\hermes\kanban\boards\hermes-multinode\workspaces\t_27e56ac3",
        )
        assert cwd == "~", (
            f"Host cwd override leaked to the ssh peer: {cwd!r}. "
            "Every command would fail in `cd` with exit 126 before running."
        )

    def test_relative_override_does_not_reach_the_peer(self, monkeypatch):
        assert self._run_and_capture_cwd(monkeypatch, "src/") == "~"

    def test_absolute_remote_override_is_preserved(self, monkeypatch):
        # A real directory on the peer must still win — that is what overrides
        # are for.
        cwd = self._run_and_capture_cwd(monkeypatch, "/Users/ftfuture/work/uservice/teamwork")
        assert cwd == "/Users/ftfuture/work/uservice/teamwork"

    def test_tilde_override_is_preserved(self, monkeypatch):
        assert self._run_and_capture_cwd(monkeypatch, "~/work") == "~/work"
