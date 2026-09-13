"""apple_container_extra_args: operator flags reach `container run` intact.

Covers the extra-args extension (network isolation, tmpfs shadows) end to
end: env-var bridge parsing in terminal_tool and flag placement in the
run command built by AppleContainerEnvironment.
"""

import json

import pytest

from tests.tools.test_apple_container_environment import (  # noqa: F401
    RunRecorder,
    recorder,
    _run_args,
)
from tools.environments import apple_container as apple


def test_extra_args_land_before_image(recorder):
    apple.AppleContainerEnvironment(
        image="python:3.11-slim-bookworm",
        extra_args=["--network", "none", "--tmpfs", "/repos/x/.git"],
    )
    argv = _run_args(recorder)
    image_index = argv.index("python:3.11-slim-bookworm")
    network_index = argv.index("--network")
    assert argv[network_index:network_index + 2] == ["--network", "none"]
    assert network_index < image_index
    tmpfs_indices = [i for i, a in enumerate(argv) if a == "--tmpfs"]
    assert any(argv[i + 1] == "/repos/x/.git" for i in tmpfs_indices)


def test_extra_args_default_empty_keeps_default_network(recorder):
    apple.AppleContainerEnvironment(image="python:3.11-slim-bookworm")
    argv = _run_args(recorder)
    assert "--network" not in argv


@pytest.mark.parametrize("bad", [["--net\nwork"], [42], ["ok", "b\x00ad"]])
def test_extra_args_rejects_unsafe_entries_before_run(recorder, bad):
    with pytest.raises(ValueError):
        apple.AppleContainerEnvironment(
            image="python:3.11-slim-bookworm", extra_args=bad
        )
    assert not any(call[1] == "run" for call in recorder.calls)


def test_env_bridge_parses_extra_args(monkeypatch):
    from tools import terminal_tool

    monkeypatch.setenv("TERMINAL_ENV", "apple_container")
    monkeypatch.setenv(
        "TERMINAL_APPLE_CONTAINER_EXTRA_ARGS", json.dumps(["--network", "none"])
    )
    config = terminal_tool._get_env_config()
    assert config["apple_container_extra_args"] == ["--network", "none"]


def test_env_bridge_rejects_non_list(monkeypatch):
    from tools import terminal_tool

    monkeypatch.setenv("TERMINAL_ENV", "apple_container")
    monkeypatch.setenv("TERMINAL_APPLE_CONTAINER_EXTRA_ARGS", '"oops"')
    with pytest.raises(ValueError):
        terminal_tool._get_env_config()


def test_prompt_probe_container_carries_extra_args(monkeypatch):
    """The prompt-backend-probe spawn path must pass extra_args too.

    Regression guard for the docker-lane leak (a probe container that
    mounted the volumes with DEFAULT network because network flags were
    read on only one spawn path).
    """
    from agent import prompt_builder
    from tools import terminal_tool

    monkeypatch.setenv("TERMINAL_ENV", "apple_container")
    monkeypatch.setenv(
        "TERMINAL_APPLE_CONTAINER_EXTRA_ARGS", json.dumps(["--network", "none"])
    )
    monkeypatch.setattr(prompt_builder, "_BACKEND_PROBE_CACHE", {})

    captured = {}

    class FakeEnv:
        def execute(self, cmd, timeout=4):
            return {"returncode": 0, "output": "os=Linux\n"}

        def cleanup(self):
            pass

    def fake_create_environment(**kwargs):
        captured.update(kwargs)
        return FakeEnv()

    monkeypatch.setattr("tools.terminal_tool_backends._create_environment", fake_create_environment)
    prompt_builder._probe_remote_backend("apple_container")
    assert captured["container_config"]["apple_container_extra_args"] == [
        "--network", "none",
    ]
