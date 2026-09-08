"""execute_code's environment builder: override lookup and container cwd.

Two defects, and fixing either alone is wrong.

The lookup read only the COLLAPSED container id. A CWD-only override is
registered under the RAW session id and deliberately collapses so sessions
share one container, so this builder silently dropped it while the terminal
and file layers -- both of which go through ``resolve_task_overrides`` --
honoured it.

The builder also had no cwd guard, unlike every other one. That looked
harmless only because of the first defect. It was not: an override carrying an
isolation key (``env_type`` or ``*_image``, as RL and benchmark harnesses
register) keeps the raw id, so a host path already reached
``docker run -w <host path>`` on exactly the rollouts that ask for their own
sandbox -- and the container fails to start with exit 125.

Fixing the lookup alone would widen that leak, since the builder would start
finding host paths it used to miss.
"""

import tools.code_execution_tool as cet
import tools.terminal_tool as tt


def _config(env_type="docker", cwd="/root"):
    return {
        "env_type": env_type, "docker_image": "pytorch/pytorch:latest",
        "singularity_image": "", "modal_image": "", "daytona_image": "",
        "cwd": cwd, "timeout": 180, "lifetime_seconds": 300,
        "container_cpu": 1, "container_memory": 5120, "container_disk": 51200,
        "container_persistent": True, "docker_volumes": [], "docker_env": {},
        "docker_extra_args": [], "docker_forward_env": [],
        "docker_run_as_host_user": False, "docker_network": True,
        "docker_mount_cwd_to_workspace": False, "modal_mode": "auto",
        "local_persistent": False, "host_cwd": None,
        "ssh_host": "", "ssh_user": "", "ssh_port": 22, "ssh_key": "",
    }


def _build(monkeypatch, task_id, overrides, env_type="docker", config_cwd="/root"):
    captured = {}
    cfg = _config(env_type, config_cwd)

    class _DummyEnv:
        cwd = config_cwd

    def fake_create_environment(**kwargs):
        captured["cwd"] = kwargs.get("cwd")
        return _DummyEnv()

    import tools.terminal_tool_backends as backends
    monkeypatch.setattr(tt, "_get_env_config", lambda: cfg)
    monkeypatch.setattr(backends, "_create_environment", fake_create_environment)
    monkeypatch.setattr(tt, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(tt, "_active_environments", {})
    monkeypatch.setattr(tt, "_last_activity", {})

    tt.register_task_env_overrides(task_id, overrides)
    try:
        cet._get_or_create_env(task_id)
    finally:
        tt.clear_task_env_overrides(task_id)
        tt._active_environments.pop(task_id, None)
        tt._active_environments.pop("default", None)
    return captured.get("cwd")


class TestOverrideLookupMatchesTheOtherLayers:
    def test_cwd_only_override_is_found(self, monkeypatch):
        # Collapses to "default"; the old lookup missed it entirely.
        assert _build(monkeypatch, "sess-cwd-only",
                      {"cwd": "/workspace/task42"}) == "/workspace/task42"


class TestHostCwdDoesNotReachTheContainerBuilder:
    def test_cwd_only_host_override_is_sanitized(self, monkeypatch):
        # Newly reachable because of the lookup fix, and therefore newly
        # dangerous without the guard.
        assert _build(monkeypatch, "sess-host", {"cwd": r"C:\Users\someuser"}) == "/root"

    def test_isolation_keyed_host_override_is_sanitized(self, monkeypatch):
        # Reachable even before the lookup fix: an isolation key keeps the raw
        # id. This is the half that was already leaking.
        assert _build(monkeypatch, "sess-iso",
                      {"cwd": "/home/someuser/project", "docker_image": "alpine"}) == "/root"

    def test_drive_path_override_is_sanitized(self, monkeypatch):
        assert _build(monkeypatch, "sess-drive", {"cwd": r"D:\work"}) == "/root"

    def test_valid_container_override_passes_through(self, monkeypatch):
        assert _build(monkeypatch, "sess-ok",
                      {"cwd": "/workspace/task42"}) == "/workspace/task42"

    def test_non_container_backend_is_untouched(self, monkeypatch):
        # The guard is container-scoped; local resolution must not narrow.
        assert _build(monkeypatch, "sess-local", {"cwd": "/home/someuser"},
                      env_type="local", config_cwd="/home/hermes") == "/home/someuser"
