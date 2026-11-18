"""``terminal.docker_pids_limit`` — a typed knob for the container PID ceiling.

Issue #84968. ``--pids-limit`` was hard-coded to 256 with no config path. The
pids cgroup counts threads as well as processes, so legitimate multiprocessing
workloads (pytest, DataLoader workers, Chromium, parallel subagents) reach it
sooner than a process count suggests — and once exhausted the container cannot
start even a shell for diagnostics or cleanup.

The default follows the current upstream Docker safety ceiling (2048): this is
a configurability fix, not a default-change proposal.
"""

from __future__ import annotations

import pytest

from tools.environments import docker as docker_env


# ---------------------------------------------------------------------------
# Value resolution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "configured, expected",
    [
        pytest.param(None, "2048", id="unset-keeps-the-safe-default"),
        pytest.param("2048", "2048", id="explicit-default"),
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
        assert docker_env._resolve_pids_limit("lots") == "2048"
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
    assert args[args.index("--pids-limit") + 1] == "2048"


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

    assert DEFAULT_CONFIG["terminal"]["docker_pids_limit"] == "2048"

    from hermes_cli.config import TERMINAL_CONFIG_ENV_MAP

    assert (
        TERMINAL_CONFIG_ENV_MAP["docker_pids_limit"] == "TERMINAL_DOCKER_PIDS_LIMIT"
    )


def test_terminal_env_value_reaches_backend_config(monkeypatch):
    """The env bridge must not drop an explicit value before backend shaping."""
    from tools import terminal_tool

    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_DOCKER_PIDS_LIMIT", "1024")
    assert terminal_tool._get_env_config()["docker_pids_limit"] == "1024"


# ---------------------------------------------------------------------------
# Single-source default
# ---------------------------------------------------------------------------

def test_every_hardwired_site_reads_the_same_default():
    """``"256"`` used to live in four hand-maintained places.

    config_defaults is the pure-data leaf everything can safely import, so each
    site now reads the constant from there. Importing terminal_tool wholly is
    The backend shaper and Docker builder must both consume the same pure-data
    default rather than silently growing independent literals.
    """
    from hermes_cli import config_defaults
    from tools import terminal_tool_backends

    assert config_defaults.DEFAULT_DOCKER_PIDS_LIMIT == "2048"
    assert docker_env._DEFAULT_PIDS_LIMIT == config_defaults.DEFAULT_DOCKER_PIDS_LIMIT
    container_defaults = dict(terminal_tool_backends._CONTAINER_KEYS)
    docker_kwargs = {out: default for out, _, default in terminal_tool_backends._DOCKER_KWARGS}
    assert container_defaults["docker_pids_limit"] == config_defaults.DEFAULT_DOCKER_PIDS_LIMIT
    assert docker_kwargs["pids_limit"] == config_defaults.DEFAULT_DOCKER_PIDS_LIMIT
