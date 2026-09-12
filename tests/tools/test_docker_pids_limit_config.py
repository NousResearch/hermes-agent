"""``terminal.docker_pids_limit`` — a typed knob for the container PID ceiling.

Issue #84968. ``--pids-limit`` was hard-coded to 256 with no config path. The
pids cgroup counts threads as well as processes, so legitimate multiprocessing
workloads (pytest, DataLoader workers, Chromium, parallel subagents) reach it
sooner than a process count suggests — and once exhausted the container cannot
start even a shell for diagnostics or cleanup.

The default stays 256: this is a configurability bug, not a wrong-default bug.
Raising it for everyone would weaken a real containment boundary.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.environments import docker as docker_env


# ---------------------------------------------------------------------------
# Value resolution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "configured, expected",
    [
        pytest.param(None, "256", id="unset-keeps-the-safe-default"),
        pytest.param("256", "256", id="explicit-default"),
        pytest.param("1024", "1024", id="raised-as-string"),
        pytest.param(1024, "1024", id="raised-as-int"),
        pytest.param("  512  ", "512", id="whitespace-tolerated"),
    ],
)
def test_resolve_returns_the_configured_ceiling(configured, expected):
    assert docker_env._resolve_pids_limit(configured) == expected


@pytest.mark.parametrize(
    "configured",
    [
        pytest.param(0, id="zero"),
        pytest.param("0", id="zero-string"),
        pytest.param(-1, id="minus-one"),
        pytest.param("", id="empty"),
        pytest.param("   ", id="whitespace-only"),
    ],
)
def test_non_positive_values_omit_the_flag(configured):
    """Opting out drops the flag rather than emitting ``--pids-limit -1``.

    Mirrors what ``docker_shm_size`` does with ""/"0", and avoids asserting that
    -1 behaves identically across Docker Engine, OrbStack and Colima.

    Not the same as "unlimited": omitting the flag applies the daemon's own
    default — which may itself be a finite binding an admin set (#85086). A
    user who wants literally unlimited must pass `--pids-limit -1` through
    ``docker_extra_args``, exactly as before this knob existed.
    """
    assert docker_env._resolve_pids_limit(configured) is None


def test_unparseable_value_falls_back_to_the_default(caplog):
    """A typo in one config key must not make the terminal unusable.

    Container creation is the wrong place to fail closed on a malformed number:
    the user would get no shell at all, with the cause several layers away.
    """
    with caplog.at_level("WARNING"):
        assert docker_env._resolve_pids_limit("lots") == "256"
    assert "docker_pids_limit" in caplog.text


# ---------------------------------------------------------------------------
# docker_extra_args escape hatch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "extra_args, expected",
    [
        pytest.param(["--pids-limit", "4096"], True, id="separate-value"),
        pytest.param(["--pids-limit=4096"], True, id="equals-form"),
        pytest.param(["--cpus", "2"], False, id="unrelated-flag"),
        pytest.param(["--shm-size=2g"], False, id="shm-size-not-confused"),
        pytest.param([], False, id="empty"),
        pytest.param(None, False, id="none"),
    ],
)
def test_extra_args_detection(extra_args, expected):
    assert docker_env._extra_args_set_pids_limit(extra_args) is expected


# ---------------------------------------------------------------------------
# The flag that actually reaches `docker run`
# ---------------------------------------------------------------------------

def _run_args(monkeypatch, **kwargs) -> list[str]:
    """Construct a DockerEnvironment with docker mocked and return its run args."""
    docker_env._cgroup_limits_ok = True
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")

    import subprocess

    def _run(cmd, **kw):
        out = "fake-container-id\n" if (
            isinstance(cmd, list) and len(cmd) > 1 and cmd[1] == "run"
        ) else ""
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)

    params = dict(
        image="python:3.11", cwd="/root", timeout=60, cpu=0, memory=0, disk=0,
        persistent_filesystem=False, task_id="test-task", volumes=[],
    )
    params.update(kwargs)
    env = docker_env.DockerEnvironment(**params)
    return list(env._all_run_args)


@pytest.fixture(autouse=True)
def _reset_cgroup_cache():
    docker_env._cgroup_limits_ok = None
    yield
    docker_env._cgroup_limits_ok = None


def test_configured_limit_reaches_docker_run(monkeypatch):
    args = _run_args(monkeypatch, pids_limit="1024")
    assert "--pids-limit" in args
    assert args[args.index("--pids-limit") + 1] == "1024"


def test_default_is_unchanged_when_not_configured(monkeypatch):
    """Nobody who leaves this alone sees a behaviour change."""
    args = _run_args(monkeypatch)
    assert args[args.index("--pids-limit") + 1] == "256"


def test_opting_out_removes_the_flag(monkeypatch):
    args = _run_args(monkeypatch, pids_limit="0")
    assert "--pids-limit" not in args


def test_user_extra_args_win_without_duplicating_the_flag(monkeypatch):
    """The documented workaround becomes defined behaviour.

    Previously the only way to change this was appending a second
    ``--pids-limit`` via docker_extra_args and relying on Docker's last-wins
    ordering. Ours is now skipped, so exactly one appears.
    """
    args = _run_args(monkeypatch, extra_args=["--pids-limit=4096"])
    assert args.count("--pids-limit") == 0
    assert "--pids-limit=4096" in args


# ---------------------------------------------------------------------------
# Plumbing — a knob wired at only some of its sites is worse than none
# ---------------------------------------------------------------------------

def test_setting_is_wired_through_every_config_surface():
    """The value crosses config.yaml -> env -> terminal_tool -> constructor.

    Each map is maintained by hand, so a knob can easily be added in one place
    and silently ignored in another.
    """
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["terminal"]["docker_pids_limit"] == "256"

    from hermes_cli.config import TERMINAL_CONFIG_ENV_MAP

    assert (
        TERMINAL_CONFIG_ENV_MAP["docker_pids_limit"] == "TERMINAL_DOCKER_PIDS_LIMIT"
    )


# ---------------------------------------------------------------------------
# Single-source default (post-review: Enough1122, point 2)
# ---------------------------------------------------------------------------

def test_every_hardwired_site_reads_the_same_default():
    """``"256"`` used to live in four hand-maintained places.

    config_defaults is the pure-data leaf everything can safely import, so each
    site now reads the constant from there. Importing terminal_tool wholly is
    avoided here (heavy module) — the point is the identifier's presence and
    single-source identity, not its evaluation.
    """
    import agent.prompt_builder
    import tools.terminal_tool
    from hermes_cli import config_defaults

    assert config_defaults.DEFAULT_DOCKER_PIDS_LIMIT == "256"
    assert docker_env._DEFAULT_PIDS_LIMIT == config_defaults.DEFAULT_DOCKER_PIDS_LIMIT

    for name, src in (
        ("tools/terminal_tool.py", Path(tools.terminal_tool.__file__)),
        ("agent/prompt_builder.py", Path(agent.prompt_builder.__file__)),
    ):
        text = Path(src).read_text(encoding="utf-8")
        assert "DEFAULT_DOCKER_PIDS_LIMIT" in text, (
            f"{name} must consult the shared constant, not a bare string default"
        )
        assert "\"256\"" not in [
            line.strip() for line in text.splitlines()
            if "docker_pids_limit" in line and "256" in line
        ], f"{name} still hardcodes the default beside a docker_pids_limit read"
