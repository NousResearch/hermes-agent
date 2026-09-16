"""Hermetic contract tests for the Apple Container environment."""

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.credential_files as credential_files
import tools.environments.apple_container as apple
from tools.environments.base import BaseEnvironment


class FakeLifetimeProcess:
    def __init__(self, recorder, argv, kwargs):
        self.recorder = recorder
        self.args = argv
        self.kwargs = kwargs
        self.stdin = io.BytesIO()
        self.returncode = recorder.run_returncode
        self.wait_calls = 0
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.wait_calls += 1
        self.recorder.events.append("client-wait")
        if self.wait_calls <= self.recorder.wait_timeouts:
            raise subprocess.TimeoutExpired(self.args, timeout or 0.0)
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9


def _run_help():
    """Small parser fixture, not a snapshot of the installed CLI's flags."""
    return {
        "serializationVersion": 0,
        "command": {
            "commandName": "run",
            "arguments": [
                {
                    "kind": kind,
                    "parsingStrategy": "default",
                    "names": [
                        {"kind": "short" if len(name) == 1 else "long", "name": name}
                        for name in names
                    ],
                }
                for kind, names in [
                    ("option", ["network"]),
                    ("option", ["tmpfs"]),
                    ("option", ["mount"]),
                    ("option", ["label"]),
                    ("option", ["volume", "v"]),
                    ("option", ["cpus", "c"]),
                    ("flag", ["read-only"]),
                    ("flag", ["init"]),
                    ("flag", ["interactive", "i"]),
                ]
            ] + [
                {"kind": "positional", "parsingStrategy": "default"},
                {"kind": "positional", "parsingStrategy": "allRemainingInput"},
            ],
        },
    }


class RunRecorder:
    def __init__(self):
        self.calls: list[list[str]] = []
        self.events: list[str] = []
        self.processes: list[FakeLifetimeProcess] = []
        self.stop_times_out = False
        self.run_times_out = False
        self.run_returncode: int | None = None
        self.returncodes: dict[str, int] = {}
        self.force_delete_returncode = 0
        self.probe_returncodes: list[int] = []
        self.probe_error: BaseException | None = None
        self.wait_timeouts = 0
        self.run_help = _run_help()
        self.help_queries: list[dict] = []

    def popen(self, cmd, **kwargs):
        argv = list(cmd)
        self.calls.append(argv)
        self.events.append("client-spawn")
        process = FakeLifetimeProcess(self, argv, kwargs)
        if self.run_returncode:
            kwargs["stderr"].write(b"run failed")
        self.processes.append(process)
        return process

    def __call__(self, cmd, **kwargs):
        argv = list(cmd)
        self.calls.append(argv)
        self.events.append(argv[1])
        if argv[-2:] == ["system", "status"]:
            return subprocess.CompletedProcess(argv, 0, "running\n", "")
        if "--version" in argv:
            return subprocess.CompletedProcess(argv, 0, "container 0.test\n", "")
        if argv[1:] == ["run", "--experimental-dump-help"]:
            self.help_queries.append(kwargs)
            return subprocess.CompletedProcess(argv, 0, json.dumps(self.run_help), "")
        if len(argv) > 1 and argv[1] == "stop" and self.stop_times_out:
            raise subprocess.TimeoutExpired(argv, kwargs.get("timeout", 0))
        if len(argv) > 1 and argv[1] == "run":
            if self.run_times_out:
                raise subprocess.TimeoutExpired(argv, kwargs.get("timeout", 0))
            code = self.run_returncode if self.run_returncode is not None else 0
            return subprocess.CompletedProcess(argv, code, "", "run failed")
        if argv[1] == "exec":
            if self.probe_error is not None:
                raise self.probe_error
            code = self.probe_returncodes.pop(0) if self.probe_returncodes else 0
            return subprocess.CompletedProcess(argv, code, "", "not ready" if code else "")
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
    monkeypatch.setattr(apple, "is_apple_container_supported_host", lambda: True)
    monkeypatch.setattr(apple.subprocess, "run", run)
    monkeypatch.setattr(apple.subprocess, "Popen", run.popen)
    monkeypatch.setattr(apple, "query_system_resources", lambda: {"total_cpus": 8, "total_memory_mb": 24576})
    monkeypatch.setattr(BaseEnvironment, "init_session", lambda self: None)
    monkeypatch.setattr(apple, "get_sandbox_dir", lambda: tmp_path / "sandboxes")
    monkeypatch.setattr(credential_files, "get_credential_file_mounts", lambda: [])
    monkeypatch.setattr(credential_files, "get_skills_directory_mount", lambda: [])
    monkeypatch.setattr(credential_files, "get_cache_directory_mounts", lambda: [])
    yield run
    for process in run.processes:
        process.stdin.close()


def _run_args(recorder: RunRecorder) -> list[str]:
    return recorder.processes[0].args


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
    assert all(
        call[2:] == [env._container_name, "bash", "-c", ":"]
        for call in recorder.calls if call[1] == "exec"
    )
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
    assert argv[-3] == "python:3.11-slim-bookworm"
    assert argv[-2] == "-c"
    assert argv[argv.index("--entrypoint") + 1] == "bash"
    env.cleanup()


@pytest.mark.parametrize(
    "extra_args",
    [
        ["python:3.11-slim-bookworm", "-c", "exec sleep infinity"],
        ["--read-only", "python:3.11-slim-bookworm", "-c", "exec sleep infinity"],
    ],
)
def test_extra_positional_image_cannot_replace_owner_keepalive(recorder, extra_args):
    with pytest.raises(ValueError, match="positional"):
        apple.AppleContainerEnvironment(extra_args=extra_args)
    assert recorder.processes == []


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--init", "other-image"],
        ["-i", "other-image"],
        ["--network", "none", "other-image"],
        ["--network=none", "other-image"],
        ["-iv", "/tmp:/mnt", "other-image"],
        ["-"], [""],
    ],
)
def test_trailing_positionals_are_rejected_before_launch(recorder, extra_args):
    with pytest.raises(ValueError, match="positional"):
        apple.AppleContainerEnvironment(extra_args=extra_args)
    assert recorder.processes == []


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--network"], ["--network", "--init"], ["--label", "-leading"],
        ["-v"], ["-iv"], ["-iv", "--network=none"], ["-c", "-1"],
    ],
)
def test_missing_or_option_valued_extra_values_are_rejected(recorder, extra_args):
    with pytest.raises(ValueError, match="requires a value; use '=' for dash-leading"):
        apple.AppleContainerEnvironment(extra_args=extra_args)
    assert recorder.processes == []


@pytest.mark.parametrize(
    "extra_args",
    [
        ["-c1"], ["-ic1"], ["-v/tmp:/mnt"], ["-iv/tmp:/mnt"],
        ["-v/tmp=a:/mnt"], ["-vi", "/tmp:/mnt"],
    ],
)
def test_joined_short_extra_values_without_equals_are_rejected(recorder, extra_args):
    with pytest.raises(ValueError, match="extra arg"):
        apple.AppleContainerEnvironment(extra_args=extra_args)
    assert recorder.processes == []


@pytest.mark.parametrize(
    "kind, extra_args",
    [
        ("flag", ["-iv=/tmp:/mnt"]), ("flag", ["-ic=1"]),
        ("flag", ["-ixv=/tmp:/mnt"]), ("option", ["-ix=value"]),
    ],
)
def test_clustered_equals_extra_values_are_rejected_before_launch(recorder, kind, extra_args):
    recorder.run_help["command"]["arguments"].append({
        "kind": kind, "parsingStrategy": "default",
        "names": [{"kind": "short", "name": "x"}],
    })
    with pytest.raises(ValueError, match="requires a separate value"):
        apple.AppleContainerEnvironment(extra_args=extra_args)
    assert recorder.processes == []


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--network="], ["--mount="], ["-c="], ["-v="], ["-iv="],
        ["--network", ""], ["--mount", ""], ["-c", ""], ["-v", ""], ["-iv", ""],
    ],
)
def test_empty_extra_option_values_are_rejected(recorder, extra_args):
    with pytest.raises(ValueError, match="requires a .*value"):
        apple.AppleContainerEnvironment(extra_args=extra_args)
    assert recorder.processes == []


@pytest.mark.parametrize("extra_args", [["--unknown=value"], ["--unknown", "value"], ["-x"], ["-ix"]])
def test_unregistered_extra_options_are_rejected(recorder, extra_args):
    with pytest.raises(ValueError, match="unknown option"):
        apple.AppleContainerEnvironment(extra_args=extra_args)
    assert recorder.processes == []


@pytest.mark.parametrize("extra_args", [["--init=true"], ["--read-only=false"]])
def test_extra_flags_do_not_take_values(recorder, extra_args):
    with pytest.raises(ValueError, match="flag does not take a value"):
        apple.AppleContainerEnvironment(extra_args=extra_args)
    assert recorder.processes == []


@pytest.mark.parametrize("extra_args", [None, []])
def test_empty_extra_args_do_not_query_parser_metadata(recorder, extra_args):
    env = apple.AppleContainerEnvironment(extra_args=extra_args)
    assert recorder.help_queries == []
    env.cleanup()


def test_extra_args_query_metadata_once_before_launch_not_per_exec(recorder, monkeypatch):
    monkeypatch.setattr(apple, "_popen_bash", lambda cmd, data: None)
    env = apple.AppleContainerEnvironment(extra_args=["--network", "none"])
    env._run_bash("pwd")
    env._run_bash("true")
    env.cleanup()
    query = [env._exe, "run", "--experimental-dump-help"]
    assert recorder.calls.count(query) == 1
    assert recorder.calls.index(query) < recorder.calls.index(_run_args(recorder))
    assert len(recorder.help_queries) == 1
    kwargs = recorder.help_queries[0]
    assert 0 < kwargs["timeout"] <= 10
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["stdin"] == subprocess.DEVNULL


@pytest.mark.parametrize("failure", ["timeout", "unavailable", "nonzero", "invalid-json"])
def test_unavailable_parser_metadata_fails_closed(recorder, monkeypatch, failure):
    original_run = recorder.__call__

    def unavailable_metadata(cmd, **kwargs):
        if cmd[1:] == ["run", "--experimental-dump-help"]:
            if failure == "timeout":
                raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
            if failure == "unavailable":
                raise OSError("unavailable")
            return subprocess.CompletedProcess(cmd, 1 if failure == "nonzero" else 0, "not json", "")
        return original_run(cmd, **kwargs)

    monkeypatch.setattr(apple.subprocess, "run", unavailable_metadata)
    with pytest.raises(ValueError, match="cannot validate extra args.*supported option metadata"):
        apple.AppleContainerEnvironment(extra_args=["--network", "none"])
    assert recorder.processes == []


@pytest.mark.parametrize(
    "path, value",
    [
        ((), None), ((), []), ((), {}),
        (("serializationVersion",), 1), (("serializationVersion",), False),
        (("command", "commandName"), "exec"),
        (("command", "arguments"), None), (("command", "arguments"), []),
        (("command", "arguments"), [{"kind": "positional"}]),
        (("command", "arguments", 0, "kind"), "unknown"),
        (("command", "arguments", 0, "parsingStrategy"), "unconditional"),
        (("command", "arguments", 0, "parsingStrategy"), None),
        (("command", "arguments", 0, "names"), []),
        (("command", "arguments", 0, "names", 0, "kind"), "longWithSingleDash"),
        (("command", "arguments", 0, "names", 0, "kind"), "short"),
        (("command", "arguments", 0, "names", 0, "name"), "bad name"),
        (("command", "arguments", 0, "names"), [
            {"kind": "long", "name": "network"}, {"kind": "long", "name": "network"},
        ]),
    ],
)
def test_malformed_or_unsupported_parser_metadata_fails_closed(recorder, path, value):
    if path:
        target = recorder.run_help
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    else:
        recorder.run_help = value
    with pytest.raises(ValueError, match="cannot validate extra args.*supported option metadata"):
        apple.AppleContainerEnvironment(extra_args=["--network", "none"])
    assert recorder.processes == []


@pytest.mark.parametrize(
    "kind, extra_args",
    [
        ("option", ["--foo=value"]), ("option", ["--foo", "value"]),
        ("option", ["-x=value"]), ("option", ["-x", "value"]),
        ("flag", ["--foo", "-x"]),
        ("flag", ["-ixv", "/tmp:/mnt"]), ("flag", ["-xiv", "/tmp:/mnt"]),
    ],
)
def test_extra_option_arity_comes_from_registered_metadata(recorder, kind, extra_args):
    recorder.run_help["command"]["arguments"].append({
        "kind": kind, "parsingStrategy": "default",
        "names": [{"kind": "long", "name": "foo"}, {"kind": "short", "name": "x"}],
    })
    env = apple.AppleContainerEnvironment(extra_args=extra_args)
    argv = _run_args(recorder)
    assert argv[-len(extra_args) - 3:-3] == extra_args
    env.cleanup()


@pytest.mark.parametrize("extra_args", [["-xd"], ["-xt"]])
def test_registered_short_flags_cannot_hide_reserved_controls(recorder, extra_args):
    recorder.run_help["command"]["arguments"].append({
        "kind": "flag", "parsingStrategy": "default",
        "names": [{"kind": "short", "name": "x"}],
    })
    with pytest.raises(ValueError, match="owns container lifetime"):
        apple.AppleContainerEnvironment(extra_args=extra_args)
    assert recorder.processes == []


def test_extra_args_are_inserted_immediately_before_image(recorder):
    env = apple.AppleContainerEnvironment(
        image="python:3.11-slim-bookworm",
        extra_args=["--network", "none", "--tmpfs", "/custom"],
    )
    argv = _run_args(recorder)

    image_index = argv.index("python:3.11-slim-bookworm")
    assert argv[image_index - 4:image_index] == [
        "--network", "none", "--tmpfs", "/custom"
    ]
    env.cleanup()


@pytest.mark.parametrize(
    "extra_args",
    ["--network=none", [123], ["--network\nnone"], ["bad\rflag"], ["bad\x00flag"]],
)
def test_extra_args_reject_non_lists_non_strings_and_control_characters_before_run(
    recorder, extra_args
):
    with pytest.raises(ValueError, match="extra arg"):
        apple.AppleContainerEnvironment(extra_args=extra_args)

    assert not any(call[1] == "run" for call in recorder.calls)


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


@pytest.mark.parametrize("stage", ["before-probe", "after-ready-probe"])
def test_successful_early_client_exit_is_not_readiness(recorder, monkeypatch, stage):
    sessions = []
    monkeypatch.setattr(BaseEnvironment, "init_session", lambda self: sessions.append(self))
    monkeypatch.setattr(apple.time, "sleep", lambda seconds: None)
    if stage == "before-probe":
        recorder.run_returncode = 0
    else:
        original_run = recorder.__call__

        def exit_after_probe(cmd, **kwargs):
            result = original_run(cmd, **kwargs)
            if cmd[1] == "exec":
                assert result.returncode == 0
                recorder.processes[0].returncode = 0
            return result

        monkeypatch.setattr(apple.subprocess, "run", exit_after_probe)
    env = apple.AppleContainerEnvironment.__new__(apple.AppleContainerEnvironment)
    try:
        with pytest.raises(RuntimeError, match=r"Failed to start Apple Container \(exit 0\)"):
            apple.AppleContainerEnvironment.__init__(env)
        assert sessions == []
        probes = [call for call in recorder.calls if call[1] == "exec"]
        assert len(probes) == (0 if stage == "before-probe" else 1)
        assert recorder.processes[0].stdin.closed
        assert recorder.events.index("client-wait") < recorder.events.index("delete")
        assert env._run_process is None
        assert env._container_name is None
    finally:
        env.cleanup()


def test_startup_timeout_force_deletes_candidate_container(recorder, monkeypatch):
    recorder.run_times_out = True
    monkeypatch.setattr(apple, "_STARTUP_TIMEOUT", 0.0)
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


def test_lifetime_client_keeps_owned_pipe_open_until_cleanup(recorder):
    env = apple.AppleContainerEnvironment()
    assert len(recorder.processes) == 1
    process = recorder.processes[0]
    argv = process.args
    assert "--interactive" in argv
    assert "--rm" in argv
    assert "--detach" not in argv
    assert "--tty" not in argv
    assert argv[argv.index("--entrypoint") + 1] == "bash"
    assert process.kwargs["stdin"] == subprocess.PIPE
    assert process.kwargs["stdout"] == subprocess.DEVNULL
    assert process.kwargs["stderr"] != subprocess.PIPE
    assert process.kwargs["close_fds"] is True
    assert process.kwargs["start_new_session"] is True
    assert not process.stdin.closed
    assert process.poll() is None
    env.cleanup()
    assert process.stdin.closed
    assert process.wait_calls >= 1
    assert env._run_process is None


def test_readiness_waits_for_successful_guest_exec(recorder):
    recorder.probe_returncodes = [1, 0]
    env = apple.AppleContainerEnvironment()
    probes = [call for call in recorder.calls if call[1] == "exec"]
    assert len(probes) == 2
    assert all(call[2:] == [env._container_name, "bash", "-c", ":"] for call in probes)
    env.cleanup()


@pytest.mark.parametrize("stage", ["probe", "session"])
@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_interrupted_constructor_releases_client_and_candidate(
    recorder, monkeypatch, stage, error_type
):
    if stage == "probe":
        recorder.probe_error = error_type()
    else:
        def interrupt_session(self):
            raise error_type()
        monkeypatch.setattr(BaseEnvironment, "init_session", interrupt_session)
    env = apple.AppleContainerEnvironment.__new__(apple.AppleContainerEnvironment)
    with pytest.raises(error_type):
        apple.AppleContainerEnvironment.__init__(env)
    assert env._run_process is None
    assert env._container_name is None
    assert recorder.processes[0].stdin.closed


def test_stubborn_lifetime_client_is_killed_and_reaped(recorder):
    recorder.wait_timeouts = 2
    env = apple.AppleContainerEnvironment()
    process = recorder.processes[0]
    env.cleanup()
    assert process.stdin.closed
    assert process.terminated
    assert process.killed
    assert process.wait_calls == 3
    assert env._run_process is None


def test_startup_timeout_reaps_creator_before_candidate_delete(recorder, monkeypatch):
    monkeypatch.setattr(apple, "_STARTUP_TIMEOUT", 0.0)
    with pytest.raises(RuntimeError, match="startup timed out"):
        apple.AppleContainerEnvironment()
    assert recorder.events.index("client-wait") < recorder.events.index("delete")
    assert recorder.processes[0].stdin.closed


def test_startup_failure_keeps_cli_diagnostic(recorder):
    recorder.run_returncode = 7
    with pytest.raises(RuntimeError, match="run failed"):
        apple.AppleContainerEnvironment()
    assert recorder.processes[0].stdin.closed


def test_unreaped_creator_retains_candidate_for_retry(recorder, monkeypatch):
    recorder.wait_timeouts = 100
    monkeypatch.setattr(apple, "_STARTUP_TIMEOUT", 0.0)
    env = apple.AppleContainerEnvironment.__new__(apple.AppleContainerEnvironment)
    with pytest.raises(RuntimeError, match="could not be reaped"):
        apple.AppleContainerEnvironment.__init__(env)
    assert env._container_name is not None
    assert env._run_process is recorder.processes[0]
    assert recorder.processes[0].stdin.closed
    assert not any(call[1] == "delete" for call in recorder.calls)
    recorder.wait_timeouts = 0
    env.cleanup()
    assert env._run_process is None
    assert env._container_name is None


def test_already_auto_removed_container_still_reaps_client(recorder, monkeypatch):
    env = apple.AppleContainerEnvironment()
    process = recorder.processes[0]
    original_run = recorder.__call__

    def auto_removed(cmd, **kwargs):
        if list(cmd)[1] == "stop":
            return subprocess.CompletedProcess(cmd, 1, "", "no such container")
        return original_run(cmd, **kwargs)

    monkeypatch.setattr(apple.subprocess, "run", auto_removed)
    env.cleanup()
    env.cleanup()
    assert env._container_name is None
    assert env._run_process is None
    assert process.stdin.closed


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--detach"], ["--detach=true"], ["-d"], ["-it"], ["-qid"],
        ["--tty"], ["--name", "other"], ["--name=other"],
        ["--entrypoint", "sleep"], ["--entrypoint=sleep"], ["--"],
        ["--label", "--name=other"], ["--label", "-dash"],
    ],
)
def test_lifecycle_overrides_are_rejected_before_launch(recorder, extra_args):
    with pytest.raises(ValueError, match="owns container lifetime"):
        apple.AppleContainerEnvironment(extra_args=extra_args)
    assert not any(call[1] == "run" for call in recorder.calls)


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--network", "none", "--tmpfs", "/custom"],
        ["-v=/tmp:/mnt"],
        ["--label", "purpose=hermes-test"],
        ["--network=none"],
        ["--mount", "type=bind,source=/tmp/with spaces,target=/mnt"],
        ["--label", "purpose=a,b c"],
        ["--label=-dash-leading"],
        ["--init", "--read-only", "-i"],
        ["-v", "/tmp/with spaces:/mnt"],
        ["-iv", "/tmp:/mnt"],
        ["-c", "1"], ["-c=1"], ["-ic", "1"],
    ],
)
def test_non_lifecycle_extra_arguments_are_preserved(recorder, extra_args):
    env = apple.AppleContainerEnvironment(extra_args=extra_args)
    argv = _run_args(recorder)
    image_index = argv.index("python:3.11-slim-bookworm")
    assert argv[image_index - len(extra_args):image_index] == extra_args
    env.cleanup()
