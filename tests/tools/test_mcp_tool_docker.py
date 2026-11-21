"""Label stamping + orphan reaping for docker-run stdio MCP servers (#138312).

``mcp_servers`` entries whose command is ``docker`` spawn the ``docker run`` CLI client as their
stdio child. When Hermes dies ungracefully the client dies too, but ``--rm`` only fires on the
client's own graceful path — the container keeps running, and gateway restart loops accumulate
one orphan per docker-based server per restart. The fix labels every spawned container with its
server and owning process, and reaps same-server containers whose owner is provably dead before
spawning a new one.
"""

from __future__ import annotations

import subprocess

import pytest

from tools.mcp_tool_docker import (
    _OWNER_LABEL,
    _SERVER_LABEL,
    inject_docker_run_labels,
    is_docker_run_command,
    reap_orphaned_docker_mcp_containers,
)


# ------------------------------------------------------------------ detection


@pytest.mark.parametrize(
    "command,args,expected",
    [
        ("docker", ["run", "-i", "--rm", "img"], True),
        ("/usr/local/bin/docker", ["run", "img"], True),
        ("Docker.exe", ["run", "img"], True),
        ("docker", ["exec", "ctr", "sh"], False),
        ("docker", [], False),
        ("podman", ["run", "img"], False),
        ("npx", ["-y", "mcp-linear"], False),
    ],
)
def test_is_docker_run_command(command, args, expected):
    assert is_docker_run_command(command, args) is expected


# ------------------------------------------------------------------- stamping


def _stamp(command, args, server="terraform"):
    return inject_docker_run_labels(server, command, args)


def test_stamp_inserts_labels_right_after_run(monkeypatch):
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "work")
    monkeypatch.setattr(
        "hermes_cli.process_identity._process_create_time", lambda pid=None: 123.456
    )
    command, args = _stamp(
        "docker", ["run", "-i", "--rm", "hashicorp/terraform-mcp-server:latest"]
    )
    assert command == "docker"
    assert args[0] == "run"
    assert args[1:7] == [
        "--label",
        "hermes-mcp-server=terraform",
        "--label",
        f"hermes-mcp-owner={_pid()}:123.456",
        "--label",
        "hermes-profile=work",
    ]
    assert args[7:] == ["-i", "--rm", "hashicorp/terraform-mcp-server:latest"]


def _pid():
    import os

    return os.getpid()


def test_stamp_sanitizes_server_name():
    _command, args = _stamp("docker", ["run", "img"], server="my server/x")
    labels = " ".join(args)
    assert "hermes-mcp-server=my_server_x" in labels


@pytest.mark.parametrize(
    "argv",
    [
        ["exec", "ctr", "sh"],
        ["build", "."],
    ],
)
def test_stamp_passes_through_non_run_subcommands(argv):
    assert _stamp("docker", argv) == ("docker", argv)


def test_stamp_passes_through_non_docker_commands():
    argv = ["-y", "mcp-linear"]
    assert _stamp("npx", argv) == ("npx", argv)
    assert _stamp("/usr/bin/uvx", ["mcp-server"]) == ("/usr/bin/uvx", ["mcp-server"])


@pytest.mark.parametrize(
    "explicit",
    [
        ["run", "--label", "hermes-mcp-server=custom", "img"],
        ["run", "--label=hermes-mcp-server=custom", "img"],
    ],
)
def test_stamp_respects_an_explicit_server_label(explicit):
    # A user managing the label themselves keeps full control (their container, their reaping).
    assert _stamp("docker", explicit) == ("docker", explicit)


# -------------------------------------------------------------------- reaping


class _FakeDocker:
    """Scripted ``docker ps``/``inspect``/``rm`` results keyed by argv shape."""

    def __init__(self, *, containers, ps_ok=True):
        self.containers = containers  # {id: owner label or None}
        self.ps_ok = ps_ok
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        if argv[1] == "ps":
            if not self.ps_ok:
                return subprocess.CompletedProcess(argv, 1, stdout="", stderr="boom")
            stdout = "\n".join(self.containers)
            return subprocess.CompletedProcess(argv, 0, stdout=stdout + "\n", stderr="")
        if argv[1] == "inspect":
            cid = argv[-1]
            owner = self.containers.get(cid)
            if owner is None:
                return subprocess.CompletedProcess(
                    argv, 1, stdout="", stderr="no such container"
                )
            return subprocess.CompletedProcess(
                argv, 0, stdout=(owner or "<no value>") + "\n", stderr=""
            )
        assert argv[1] == "rm"
        return subprocess.CompletedProcess(argv, 0, stdout=argv[-1] + "\n", stderr="")


def _reap(monkeypatch, fake, *, alive):
    monkeypatch.setattr("tools.environments.docker._docker_query", fake)
    monkeypatch.setattr(
        "hermes_cli.process_identity._pid_alive_matches",
        lambda pid, create, **kw: alive,
    )
    return reap_orphaned_docker_mcp_containers("terraform", "/usr/bin/docker")


def test_reap_removes_containers_with_dead_owners(monkeypatch):
    fake = _FakeDocker(containers={"aa111": "999:1.0", "bb222": "998:2.0"})
    removed = _reap(monkeypatch, fake, alive=False)
    assert removed == 2
    rm_calls = [c for c in fake.calls if c[1] == "rm"]
    assert rm_calls == [
        ["/usr/bin/docker", "rm", "-f", "aa111"],
        ["/usr/bin/docker", "rm", "-f", "bb222"],
    ]


def test_reap_never_touches_live_owner_containers(monkeypatch):
    fake = _FakeDocker(containers={"aa111": "999:1.0"})
    assert _reap(monkeypatch, fake, alive=True) == 0
    assert not [c for c in fake.calls if c[1] == "rm"]


def test_reap_fails_closed_on_unverifiable_owner(monkeypatch):
    # psutil unable to say, or a malformed/ownerless label: skip, never rm.
    fake = _FakeDocker(containers={"aa111": "not-a-pid", "bb222": "", "cc333": "0:1.0"})
    assert _reap(monkeypatch, fake, alive=None) == 0
    assert not [c for c in fake.calls if c[1] == "rm"]


def test_reap_skips_hermes_unlabeled_containers(monkeypatch):
    # docker inspect renders a missing label as <no value> — user/older-Hermes containers.
    fake = _FakeDocker(containers={"aa111": "<no value>"})
    assert _reap(monkeypatch, fake, alive=False) == 0
    assert not [c for c in fake.calls if c[1] == "rm"]


def test_reap_survives_ps_failure(monkeypatch):
    fake = _FakeDocker(containers={}, ps_ok=False)
    assert _reap(monkeypatch, fake, alive=False) == 0
    assert fake.calls and fake.calls[0][1] == "ps"


def test_reap_continues_past_one_bad_inspect(monkeypatch):
    fake = _FakeDocker(containers={"aa111": None, "bb222": "999:1.0"})
    assert _reap(monkeypatch, fake, alive=False) == 1
    assert [c for c in fake.calls if c[1] == "rm"] == [
        ["/usr/bin/docker", "rm", "-f", "bb222"]
    ]


# ---------------------------------------------------------------- integration


def test_preflight_stamps_docker_run_argv(monkeypatch):
    """The stdio preflight (OSV check → npx swap → stamp) labels docker-run servers end to end."""
    import asyncio

    monkeypatch.setattr("tools.osv_check.check_package_for_malware", lambda c, a: None)
    monkeypatch.setattr(
        "hermes_cli.profiles.get_active_profile_name", lambda: "default"
    )
    monkeypatch.setattr(
        "hermes_cli.process_identity._process_create_time", lambda pid=None: 7.0
    )
    from tools.mcp_tool import _preflight_stdio_command

    command, args = asyncio.run(
        _preflight_stdio_command(
            "terraform",
            "docker",
            ["run", "-i", "--rm", "hashicorp/terraform-mcp-server:latest"],
        )
    )
    assert command == "docker"
    assert f"{_SERVER_LABEL}=terraform" in args
    assert f"{_OWNER_LABEL}={_pid()}:7.000" in args
    assert args[-1] == "hashicorp/terraform-mcp-server:latest"


def test_preflight_leaves_non_docker_argv_alone(monkeypatch):
    import asyncio

    monkeypatch.setattr("tools.osv_check.check_package_for_malware", lambda c, a: None)
    from tools.mcp_tool import _preflight_stdio_command

    argv = ["-y", "mcp-linear", "--port", "7"]
    assert asyncio.run(_preflight_stdio_command("srv", "npx", argv)) == ("npx", argv)
