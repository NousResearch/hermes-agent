"""Regression tests for host-path cwd sanitization on container backends.

Two code paths in ``tools/terminal_tool.py`` must reject a host (or relative)
working directory before it reaches ``docker run -w``:

  1. ``_get_env_config()`` sanitizes the ``TERMINAL_CWD``-derived ``config["cwd"]``.
  2. ``terminal_tool()`` resolves a *per-task cwd override* that WINS over
     ``config["cwd"]`` (registered by the gateway/TUI for workspace tracking,
     and by RL/benchmark envs). That override was applied RAW — never sanitized
     — so a host cwd (e.g. a Windows desktop session's ``C:\\Users\\<user>``)
     leaked straight to ``docker run -w C:\\Users\\<user>``, which fails to start
     the container (exit 125). The sanitizer at path #1 lists ``C:\\``/``C:/`` as
     host prefixes but only ever ran against ``config["cwd"]``, so the override
     bypassed the one guard that would have caught it.

Both paths now share ``_is_unusable_container_cwd()``; these tests pin its
behaviour so neither path can regress.
"""

import tools.terminal_tool as tt


class TestIsUnusableContainerCwd:
    def test_windows_backslash_host_path_rejected(self):
        # The exact shape from the bug report: a Windows host cwd reaching a
        # Linux container's -w flag.
        assert tt._is_unusable_container_cwd(r"C:\Users\someuser") is True

    def test_windows_forwardslash_host_path_rejected(self):
        assert tt._is_unusable_container_cwd("C:/Users/someuser") is True

    def test_posix_home_host_path_rejected(self):
        assert tt._is_unusable_container_cwd("/home/ben/projects") is True

    def test_macos_users_host_path_rejected(self):
        assert tt._is_unusable_container_cwd("/Users/ben/projects") is True

    def test_relative_path_rejected(self):
        assert tt._is_unusable_container_cwd(".") is True
        assert tt._is_unusable_container_cwd("src/app") is True

    def test_valid_container_workspace_accepted(self):
        # In-container paths that RL/benchmark overrides legitimately set must
        # pass through untouched.
        assert tt._is_unusable_container_cwd("/workspace") is False
        assert tt._is_unusable_container_cwd("/root") is False
        assert tt._is_unusable_container_cwd("/app") is False
        assert tt._is_unusable_container_cwd("/opt/project") is False

    def test_empty_is_not_flagged(self):
        # Empty/None-ish cwd is handled by the caller's `or config["cwd"]`
        # fallback, not by flagging it here.
        assert tt._is_unusable_container_cwd("") is False

    def test_host_prefixes_include_windows_and_posix(self):
        # Guard the constant itself — the Windows entries are the ones that
        # were load-bearing for the reported desktop bug.
        assert r"C:\\"[:2] in tt._HOST_CWD_PREFIXES or "C:\\" in tt._HOST_CWD_PREFIXES
        assert "C:/" in tt._HOST_CWD_PREFIXES
        assert "/home/" in tt._HOST_CWD_PREFIXES
        assert "/Users/" in tt._HOST_CWD_PREFIXES

    def test_container_backends_set(self):
        assert tt._CONTAINER_BACKENDS == frozenset(
            {"docker", "singularity", "modal", "daytona"}
        )


class TestOverrideCwdSanitizedAtCallSite:
    """E2E pin: a per-task cwd OVERRIDE that is a host path must NOT reach the
    container builder. This is the actual reported bug — the gateway/TUI
    registers the host launch dir as a cwd override, which previously won over
    the (sanitized) config["cwd"] and flowed raw into `docker run -w`.
    """

    def _run_and_capture_cwd(self, monkeypatch, override_cwd, config_cwd="/root"):
        """Drive terminal_tool() on the docker backend with a host-path cwd
        override registered, and return the cwd that reached _create_environment
        (i.e. the cwd that would be passed to `docker run -w`).
        """
        captured = {}

        config = {
            "env_type": "docker",
            "docker_image": "pytorch/pytorch:latest",
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
        # Force a fresh environment build so _create_environment is invoked.
        monkeypatch.setattr(tt, "_active_environments", {})
        monkeypatch.setattr(tt, "_last_activity", {})

        task_id = "sess-host-cwd"
        tt.register_task_env_overrides(task_id, {"cwd": override_cwd})
        try:
            tt.terminal_tool(command="pwd", task_id=task_id)
        finally:
            tt.clear_task_env_overrides(task_id)
            tt._active_environments.pop(task_id, None)
            tt._active_environments.pop("default", None)
        return captured.get("cwd")

    def test_windows_host_override_does_not_reach_container(self, monkeypatch):
        # The bug: C:\Users\<user> registered as override → docker run -w C:\Users\<user> → exit 125.
        cwd = self._run_and_capture_cwd(monkeypatch, r"C:\Users\someuser")
        assert cwd == "/root", (
            f"Host-path cwd override leaked to the container builder: {cwd!r}. "
            "It must be sanitized back to config['cwd']."
        )

    def test_posix_host_override_does_not_reach_container(self, monkeypatch):
        cwd = self._run_and_capture_cwd(monkeypatch, "/home/someuser/project")
        assert cwd == "/root"

    def test_valid_container_override_is_preserved(self, monkeypatch):
        # RL/benchmark envs set an in-container path; it must pass through.
        cwd = self._run_and_capture_cwd(monkeypatch, "/workspace/task42")
        assert cwd == "/workspace/task42"


class TestFileOpsCwdSanitizedAtCallSite:
    """E2E pin: file tools (_get_file_ops) must sanitize a host/relative cwd
    override before it reaches _create_environment on a container backend —
    the same guard the terminal tool got in #50636.  Without it, a Desktop/TUI
    host cwd (e.g. ``/Users/me/workspace``) leaks straight into
    ``docker run -w`` and ``search_files`` returns an empty workspace (#54447).
    """

    def _run_and_capture_cwd(self, monkeypatch, override_cwd, env_type="docker",
                             config_cwd="/workspace"):
        """Drive ``_get_file_ops()`` on a container backend with a host-path cwd
        override registered, and return the cwd that reached
        ``_create_environment`` (i.e. the cwd passed to ``docker run -w``).
        """
        import tools.terminal_tool as tt
        import tools.file_tools as ft

        captured = {}

        config = {
            "env_type": env_type,
            "docker_image": "pytorch/pytorch:latest",
            "singularity_image": "docker://pytorch/pytorch:latest",
            "modal_image": "pytorch/pytorch:latest",
            "daytona_image": "pytorch/pytorch:latest",
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
            "ssh_host": "",
            "ssh_user": "",
            "ssh_port": 22,
            "ssh_key": "",
            "ssh_persistent": False,
            "local_persistent": False,
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
        monkeypatch.setattr(tt, "_create_environment", fake_create_environment)
        # Force a fresh environment build.
        monkeypatch.setattr(tt, "_active_environments", {})
        monkeypatch.setattr(tt, "_last_activity", {})
        monkeypatch.setattr(ft, "_file_ops_cache", {})
        monkeypatch.setattr(tt, "_session_cwd", {})

        task_id = "sess-fileops-host-cwd"
        tt.register_task_env_overrides(task_id, {"cwd": override_cwd})
        try:
            ft._get_file_ops(task_id)
        finally:
            tt.clear_task_env_overrides(task_id)
        return captured.get("cwd")

    def test_macos_host_override_does_not_reach_container(self, monkeypatch):
        # Desktop/TUI registers /Users/<me>/workspace as the session cwd.
        cwd = self._run_and_capture_cwd(monkeypatch, "/Users/me/workspace")
        assert cwd == "/workspace", (
            f"Host-path cwd override leaked to the container builder: {cwd!r}. "
            "It must be sanitized back to config['cwd']."
        )

    def test_posix_home_host_override_does_not_reach_container(self, monkeypatch):
        cwd = self._run_and_capture_cwd(monkeypatch, "/home/someuser/project")
        assert cwd == "/workspace"

    def test_windows_host_override_does_not_reach_container(self, monkeypatch):
        cwd = self._run_and_capture_cwd(monkeypatch, r"C:\Users\someuser")
        assert cwd == "/workspace"

    def test_relative_cwd_override_does_not_reach_container(self, monkeypatch):
        cwd = self._run_and_capture_cwd(monkeypatch, "src/app")
        assert cwd == "/workspace"

    def test_valid_container_override_is_preserved(self, monkeypatch):
        # RL/benchmark envs set an in-container path; it must pass through.
        cwd = self._run_and_capture_cwd(monkeypatch, "/workspace/task42")
        assert cwd == "/workspace/task42"

    def test_host_override_sanitized_on_singularity(self, monkeypatch):
        cwd = self._run_and_capture_cwd(
            monkeypatch, "/Users/me/workspace", env_type="singularity")
        assert cwd == "/workspace"

    def test_host_override_sanitized_on_modal(self, monkeypatch):
        cwd = self._run_and_capture_cwd(
            monkeypatch, "/Users/me/workspace", env_type="modal")
        assert cwd == "/workspace"
