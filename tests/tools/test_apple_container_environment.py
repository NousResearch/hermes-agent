"""Hermetic contract tests for the Apple Container environment."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.credential_files as credential_files
import tools.environments.apple_container as apple
from tools.environments.base import BaseEnvironment


class RunRecorder:
    def __init__(self):
        self.calls: list[list[str]] = []
        self.stop_times_out = False
        self.run_times_out = False
        self.run_returncode = 0
        self.returncodes: dict[str, int] = {}
        self.force_delete_returncode = 0

    def __call__(self, cmd, **kwargs):
        argv = list(cmd)
        self.calls.append(argv)
        if argv[-2:] == ["system", "status"]:
            return subprocess.CompletedProcess(argv, 0, "running\n", "")
        if "--version" in argv:
            return subprocess.CompletedProcess(argv, 0, "container 0.test\n", "")
        if len(argv) > 1 and argv[1] == "stop" and self.stop_times_out:
            raise subprocess.TimeoutExpired(argv, kwargs.get("timeout", 0))
        if len(argv) > 1 and argv[1] == "run":
            if self.run_times_out:
                raise subprocess.TimeoutExpired(argv, kwargs.get("timeout", 0))
            return subprocess.CompletedProcess(argv, self.run_returncode, "", "run failed")
        returncode = (
            self.force_delete_returncode
            if len(argv) > 2 and argv[1:3] == ["delete", "--force"]
            else self.returncodes.get(argv[1], 0)
        )
        return subprocess.CompletedProcess(argv, returncode, "", f"{argv[1]} failed")


@pytest.fixture
def recorder(monkeypatch, tmp_path):
    run = RunRecorder()
    monkeypatch.setattr(apple, "find_container_cli", lambda: "/usr/bin/container")
    monkeypatch.setattr(
        apple,
        "platform",
        SimpleNamespace(
            system=lambda: "Darwin",
            machine=lambda: "arm64",
            mac_ver=lambda: ("26.0", ("", "", ""), ""),
        ),
        raising=False,
    )
    monkeypatch.setattr(apple.subprocess, "run", run)
    monkeypatch.setattr(apple, "query_system_resources", lambda: {"total_cpus": 8, "total_memory_mb": 24576})
    monkeypatch.setattr(BaseEnvironment, "init_session", lambda self: None)
    monkeypatch.setattr(apple, "get_sandbox_dir", lambda: tmp_path / "sandboxes")
    monkeypatch.setattr(credential_files, "get_credential_file_mounts", lambda: [])
    monkeypatch.setattr(credential_files, "get_skills_directory_mount", lambda: [])
    monkeypatch.setattr(credential_files, "get_cache_directory_mounts", lambda: [])
    return run


def _run_args(recorder: RunRecorder) -> list[str]:
    return next(call for call in recorder.calls if call[1] == "run")


def _assert_mount(argv: list[str], source: Path, target: str, *, readonly: bool) -> None:
    spec = f"type=bind,source={source.resolve()},target={target}"
    if readonly:
        spec += ",readonly"
    index = argv.index(spec)
    assert argv[index - 1:index + 1] == ["--mount", spec]


def _mount_source_for_target(argv: list[str], target: str) -> Path:
    prefix = f"type=bind,source="
    target_marker = f",target={target}"
    spec = next(
        value
        for value in argv
        if value.startswith(prefix)
        and (value.endswith(target_marker) or f"{target_marker}," in value)
    )
    source = spec.split(",source=", 1)[1].split(",target=", 1)[0]
    return Path(source)


def test_automatic_mounts_are_readonly_and_persistent_workspace_is_writable(
    recorder, monkeypatch, tmp_path
):
    credential = tmp_path / "token.json"
    skill = tmp_path / "skills one"
    cache = tmp_path / "cache one"
    credential.write_text("secret")
    skill.mkdir()
    cache.mkdir()
    monkeypatch.setattr(
        credential_files,
        "get_credential_file_mounts",
        lambda: [{"host_path": str(credential), "container_path": "/root/.hermes/token.json"}],
    )
    monkeypatch.setattr(
        credential_files,
        "get_skills_directory_mount",
        lambda: [{"host_path": str(skill), "container_path": "/root/.hermes/skills one"}],
    )
    monkeypatch.setattr(
        credential_files,
        "get_cache_directory_mounts",
        lambda: [{"host_path": str(cache), "container_path": "/root/.hermes/cache one"}],
    )

    env = apple.AppleContainerEnvironment(persistent_filesystem=True, task_id="task one")
    argv = _run_args(recorder)

    credential_target = "/root/.hermes"
    credential_stage = _mount_source_for_target(argv, credential_target)
    _assert_mount(argv, credential_stage, credential_target, readonly=True)
    assert credential_stage != credential.parent
    assert (credential_stage / "token.json").read_text() == "secret"
    assert (credential_stage / "skills one").is_dir()
    assert (credential_stage / "cache one").is_dir()
    assert not any(call[1:3] == ["exec", env._container_name] for call in recorder.calls)
    _assert_mount(argv, skill, "/root/.hermes/skills one", readonly=True)
    _assert_mount(argv, cache, "/root/.hermes/cache one", readonly=True)
    sandbox = tmp_path / "sandboxes" / "apple_container" / "task one"
    _assert_mount(argv, sandbox / "workspace", "/workspace", readonly=False)
    _assert_mount(argv, sandbox / "root", "/root", readonly=False)
    env.cleanup()
    assert not credential_stage.exists()


def test_user_mounts_preserve_readonly_writable_and_spaces(recorder, tmp_path):
    readonly = tmp_path / "read only"
    writable = tmp_path / "write me"
    readonly.mkdir()
    writable.mkdir()

    env = apple.AppleContainerEnvironment(
        volumes=[f"{readonly}:/workspace/read only:ro", f"{writable}:/workspace/write me"]
    )
    argv = _run_args(recorder)

    _assert_mount(argv, readonly, "/workspace/read only", readonly=True)
    _assert_mount(argv, writable, "/workspace/write me", readonly=False)
    env.cleanup()


@pytest.mark.parametrize(
    "volume",
    ["missing-target", "relative:/workspace", "/host:relative", "/host:/target:rw", ":/target", "/host:"],
)
def test_malformed_user_mount_is_rejected_before_run(recorder, volume):
    with pytest.raises(ValueError, match="mount"):
        apple.AppleContainerEnvironment(volumes=[volume])
    assert not any(call[1] == "run" for call in recorder.calls)


@pytest.mark.parametrize("bad_character", [",", "\x00", "\r", "\n"])
@pytest.mark.parametrize("field", ["source", "target"])
def test_user_mount_rejects_unsafe_resolved_source_or_target_before_run(
    recorder, tmp_path, bad_character, field
):
    source = f"{tmp_path}/bad{bad_character}source" if field == "source" else str(tmp_path)
    target = f"/workspace/bad{bad_character}target" if field == "target" else "/workspace"

    with pytest.raises(ValueError, match="mount"):
        apple.AppleContainerEnvironment(volumes=[f"{source}:{target}"])

    assert not any(call[1] == "run" for call in recorder.calls)


@pytest.mark.parametrize("suffix", ["\n", "\r"])
def test_user_mount_rejects_trailing_line_break_before_run(
    recorder, tmp_path, suffix
):
    source = tmp_path / "source"
    source.mkdir()

    with pytest.raises(ValueError, match="unsafe character"):
        apple.AppleContainerEnvironment(
            volumes=[f"{source}:/workspace/data{suffix}"]
        )

    assert not any(call[1] == "run" for call in recorder.calls)


@pytest.mark.parametrize("bad_character", [",", "\x00", "\r", "\n"])
@pytest.mark.parametrize("field", ["source", "target"])
def test_automatic_mount_rejects_unsafe_resolved_source_or_target_before_run(
    recorder, monkeypatch, tmp_path, bad_character, field
):
    source = f"{tmp_path}/bad{bad_character}source" if field == "source" else str(tmp_path)
    target = f"/root/bad{bad_character}target" if field == "target" else "/root/token"
    monkeypatch.setattr(
        credential_files,
        "get_credential_file_mounts",
        lambda: [{"host_path": source, "container_path": target}],
    )

    with pytest.raises(ValueError, match="mount"):
        apple.AppleContainerEnvironment()

    assert not any(call[1] == "run" for call in recorder.calls)


def test_automatic_mount_rejects_traversal_target_before_run(
    recorder, monkeypatch, tmp_path
):
    credential = tmp_path / "token"
    credential.write_text("secret")
    monkeypatch.setattr(
        credential_files,
        "get_credential_file_mounts",
        lambda: [
            {
                "host_path": str(credential),
                "container_path": "/root/.hermes/../token",
            }
        ],
    )

    with pytest.raises(ValueError, match="mount target"):
        apple.AppleContainerEnvironment()

    assert not any(call[1] == "run" for call in recorder.calls)


def test_resource_flags_image_and_keepalive_contract(recorder):
    env = apple.AppleContainerEnvironment(
        image="python:3.11-slim-bookworm", cpu=4.0, memory=6144
    )
    argv = _run_args(recorder)
    pairs = [argv[index:index + 2] for index in range(len(argv) - 1)]
    assert ["--cpus", "4"] in pairs
    assert ["--memory", "6144M"] in pairs
    assert ["--tmpfs", "/tmp"] in pairs
    assert ["--tmpfs", "/var/tmp"] in pairs
    assert ["--tmpfs", "/run"] in pairs
    assert ["--tmpfs", "/workspace"] in pairs
    assert ["--tmpfs", "/root"] in pairs
    assert ["--tmpfs", "/home"] in pairs
    assert not any(value.startswith(("/tmp:", "/workspace:", "/root:", "/home:")) for value in argv)
    assert argv[-3:] == ["python:3.11-slim-bookworm", "sleep", "infinity"]
    env.cleanup()


def test_exec_uses_interactive_only_when_stdin_exists(recorder, monkeypatch):
    popen_calls = []
    monkeypatch.setattr(apple, "_popen_bash", lambda cmd, data: popen_calls.append((cmd, data)))
    env = apple.AppleContainerEnvironment()

    env._run_bash("pwd")
    env._run_bash("cat", stdin_data="hello")

    assert popen_calls[0][0][1:3] == ["exec", env._container_name]
    assert "--interactive" not in popen_calls[0][0]
    assert popen_calls[1][0][1:4] == ["exec", "--interactive", env._container_name]
    env.cleanup()


def test_cleanup_stops_then_deletes_and_is_idempotent(recorder):
    env = apple.AppleContainerEnvironment()
    name = env._container_name
    env.cleanup()
    env.cleanup()

    lifecycle = [call[1:] for call in recorder.calls if call[1] in {"stop", "delete", "kill"}]
    assert lifecycle == [["stop", name], ["delete", name]]


def test_stop_timeout_kills_then_still_deletes(recorder):
    env = apple.AppleContainerEnvironment()
    name = env._container_name
    recorder.stop_times_out = True
    env.cleanup()
    lifecycle = [call[1:] for call in recorder.calls if call[1] in {"stop", "delete", "kill"}]
    assert lifecycle == [
        ["stop", name],
        ["kill", name],
        ["delete", "--force", name],
    ]


def test_nonzero_stop_kills_then_force_deletes(recorder):
    env = apple.AppleContainerEnvironment()
    name = env._container_name
    recorder.returncodes["stop"] = 1
    env.cleanup()

    lifecycle = [call[1:] for call in recorder.calls if call[1] in {"stop", "delete", "kill"}]
    assert lifecycle == [["stop", name], ["kill", name], ["delete", "--force", name]]
    assert env._container_name is None


def test_nonzero_delete_falls_back_to_force_delete(recorder):
    env = apple.AppleContainerEnvironment()
    name = env._container_name
    recorder.returncodes["delete"] = 1
    env.cleanup()

    lifecycle = [call[1:] for call in recorder.calls if call[1] in {"stop", "delete", "kill"}]
    assert lifecycle == [
        ["stop", name],
        ["delete", name],
        ["delete", "--force", name],
    ]
    assert env._container_name is None


def test_cleanup_retains_name_when_forced_delete_is_unconfirmed(recorder):
    env = apple.AppleContainerEnvironment()
    name = env._container_name
    recorder.returncodes["delete"] = 1
    recorder.force_delete_returncode = 1
    env.cleanup()
    assert env._container_name == name


def test_failed_run_leaves_no_active_container(recorder):
    recorder.run_returncode = 1
    env = apple.AppleContainerEnvironment.__new__(apple.AppleContainerEnvironment)
    with pytest.raises(RuntimeError, match="Failed to start"):
        apple.AppleContainerEnvironment.__init__(env)
    assert env._container_name is None
    run_call = _run_args(recorder)
    name = run_call[run_call.index("--name") + 1]
    assert ["/usr/bin/container", "delete", "--force", name] in recorder.calls


def test_startup_timeout_force_deletes_candidate_container(recorder):
    recorder.run_times_out = True
    env = apple.AppleContainerEnvironment.__new__(apple.AppleContainerEnvironment)

    with pytest.raises(RuntimeError, match="startup timed out"):
        apple.AppleContainerEnvironment.__init__(env)

    run_call = _run_args(recorder)
    name = run_call[run_call.index("--name") + 1]
    assert ["/usr/bin/container", "delete", "--force", name] in recorder.calls
    assert env._container_name is None


def test_constructor_rejects_macos_25_arm64_before_cli_probe(monkeypatch):
    monkeypatch.setattr(
        apple,
        "platform",
        SimpleNamespace(
            system=lambda: "Darwin",
            machine=lambda: "arm64",
            mac_ver=lambda: ("25.6", ("", "", ""), ""),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        apple, "find_container_cli", lambda: pytest.fail("CLI must not be probed")
    )

    with pytest.raises(RuntimeError, match="macOS 26"):
        apple.AppleContainerEnvironment()


def test_availability_check_never_starts_system(recorder):
    apple._ensure_container_available()
    assert not any(call[1:3] == ["system", "start"] for call in recorder.calls)
