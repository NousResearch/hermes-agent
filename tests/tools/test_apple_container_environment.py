from io import StringIO
import subprocess

import pytest

from tools.environments import apple_container


def _mock_subprocess_run(monkeypatch):
    calls = []

    def _run(cmd, **kwargs):
        calls.append((list(cmd) if isinstance(cmd, list) else cmd, kwargs))
        if isinstance(cmd, list) and len(cmd) >= 3 and cmd[1:3] == ["system", "status"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="{}", stderr="")
        if isinstance(cmd, list) and len(cmd) >= 2 and cmd[1] == "exec":
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="not running")
        if isinstance(cmd, list) and len(cmd) >= 2 and cmd[1] == "start":
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="not found")
        if isinstance(cmd, list) and len(cmd) >= 2 and cmd[1] == "run":
            return subprocess.CompletedProcess(cmd, 0, stdout="hermes-test\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(apple_container.subprocess, "run", _run)
    return calls


def _quiet_mount_registry(monkeypatch):
    import tools.credential_files as credential_files

    monkeypatch.setattr(credential_files, "get_credential_file_mounts", lambda: [])
    monkeypatch.setattr(credential_files, "get_skills_directory_mount", lambda: [])
    monkeypatch.setattr(credential_files, "get_cache_directory_mounts", lambda: [])


def test_ensure_apple_container_available_raises_when_missing(monkeypatch):
    monkeypatch.setattr(apple_container, "find_apple_container", lambda binary=None: None)
    monkeypatch.setattr(
        apple_container.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("subprocess.run should not be called"),
    )

    with pytest.raises(RuntimeError) as excinfo:
        apple_container._ensure_apple_container_available()

    assert "Apple container backend selected" in str(excinfo.value)
    assert "TERMINAL_APPLE_CONTAINER_BINARY" in str(excinfo.value)


def test_run_command_uses_apple_container_flags(monkeypatch, tmp_path):
    project_dir = tmp_path / "project"
    data_dir = tmp_path / "data"
    project_dir.mkdir()
    data_dir.mkdir()

    monkeypatch.setenv("TERMINAL_SANDBOX_DIR", str(tmp_path / "sandboxes"))
    monkeypatch.setattr(apple_container, "find_apple_container", lambda binary=None: "/usr/local/bin/container")
    monkeypatch.setattr(apple_container, "_get_active_profile_name", lambda: "default")
    monkeypatch.setattr(apple_container.AppleContainerEnvironment, "init_session", lambda self: None)
    _quiet_mount_registry(monkeypatch)
    calls = _mock_subprocess_run(monkeypatch)

    apple_container.AppleContainerEnvironment(
        image="python:3.11",
        cwd="/workspace",
        timeout=60,
        cpu=2,
        memory=1024,
        persistent_filesystem=True,
        task_id="test-task",
        volumes=[f"{data_dir}:/data:ro"],
        host_cwd=str(project_dir),
        auto_mount_cwd=True,
        env={"DEBUG": "1"},
        extra_args=["--rosetta"],
    )

    run_calls = [call for call in calls if isinstance(call[0], list) and call[0][1:2] == ["run"]]
    assert run_calls, "container run should have been called"
    run_args = run_calls[0][0]
    run_args_str = " ".join(run_args)

    assert run_args[:3] == ["/usr/local/bin/container", "run", "--detach"]
    assert "--init" in run_args
    assert "--workdir /workspace" in run_args_str
    assert "--cpus 2" in run_args_str
    assert "--memory 1024M" in run_args_str
    assert "--cap-drop ALL" in run_args_str
    assert "--label hermes-agent=1" in run_args_str
    assert f"type=bind,source={project_dir},target=/workspace" in run_args
    assert f"{data_dir}:/data:ro" in run_args
    assert "--env DEBUG=1" in run_args_str
    assert "--rosetta" in run_args
    assert run_args[-3:] == ["python:3.11", "sleep", "infinity"]


def test_persist_reuse_deletes_stale_name_before_fresh_run(monkeypatch, tmp_path):
    monkeypatch.setenv("TERMINAL_SANDBOX_DIR", str(tmp_path / "sandboxes"))
    monkeypatch.setattr(apple_container, "find_apple_container", lambda binary=None: "/usr/local/bin/container")
    monkeypatch.setattr(apple_container, "_get_active_profile_name", lambda: "default")
    monkeypatch.setattr(apple_container.AppleContainerEnvironment, "init_session", lambda self: None)
    _quiet_mount_registry(monkeypatch)
    calls = _mock_subprocess_run(monkeypatch)

    apple_container.AppleContainerEnvironment(
        image="python:3.11",
        task_id="test-task",
        persist_across_processes=True,
    )

    commands = [call[0][1] for call in calls if isinstance(call[0], list) and len(call[0]) > 1]
    assert commands[:5] == ["system", "exec", "start", "delete", "run"]


def test_run_bash_uses_container_exec(monkeypatch):
    captured = {}

    def _fake_popen(cmd, stdin_data=None):
        captured["cmd"] = cmd
        captured["stdin_data"] = stdin_data

        class _Proc:
            stdout = StringIO("")
            returncode = 0

            def poll(self):
                return self.returncode

        return _Proc()

    monkeypatch.setattr(apple_container, "_popen_bash", _fake_popen)

    env = apple_container.AppleContainerEnvironment.__new__(apple_container.AppleContainerEnvironment)
    env._container_exe = "/usr/local/bin/container"
    env._container_id = "hermes-test"
    env._init_env_args = ["--env", "TOKEN=value"]

    env._run_bash("echo hi", login=True, stdin_data="payload")

    assert captured["cmd"] == [
        "/usr/local/bin/container",
        "exec",
        "--interactive",
        "--env",
        "TOKEN=value",
        "hermes-test",
        "bash",
        "-l",
        "-c",
        "echo hi",
    ]
    assert captured["stdin_data"] == "payload"


def test_init_env_args_reads_hermes_dotenv_for_allowlisted_env(monkeypatch):
    env = apple_container.AppleContainerEnvironment.__new__(apple_container.AppleContainerEnvironment)
    env._env = {}
    env._forward_env = ["DATABASE_URL"]

    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(apple_container, "_load_hermes_env_vars", lambda: {"DATABASE_URL": "from_dotenv"})

    args = env._build_init_env_args()

    assert args == ["--env", "DATABASE_URL=from_dotenv"]
