"""Bot Desktop published environment: what is read back from the launcher's env file, and where it goes.

A screen hosted inside the terminal backend publishes its env file there. Only the keys the launcher writes are
read back (DISPLAY a plain X display), and those values describe the sandbox, so they are handed to processes run
inside the backend and never merged into the environment of processes started on this machine. A screen hosted
on the gateway keeps merging its values into local children exactly as before.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tools.bot_desktop import placement, runtime, sandbox_host
from tools.environments import streams

# What the launcher writes (sandbox-side paths, so a leak into a local env is unambiguous).
LAUNCHER_VALUES = {
    "DISPLAY": ":20",
    "XAUTHORITY": "/sbx/hermes-bot-desktop/default/Xauthority",
    "DBUS_SESSION_BUS_ADDRESS": "unix:path=/sbx/hermes-bot-desktop/default/bus",
    "XDG_CONFIG_HOME": "/sbx/hermes-bot-desktop/default/xdg/config",
    "XDG_CACHE_HOME": "/sbx/hermes-bot-desktop/default/xdg/cache",
    "XDG_DATA_HOME": "/sbx/hermes-bot-desktop/default/xdg/data",
}
EXTRA_LINES = {
    "LD_PRELOAD": "/sbx/lib/extra.so",
    "PATH": "/sbx/bin",
    "HTTPS_PROXY": "http://proxy.sbx.invalid:3128",
    "SOME_UNRELATED_NAME": "unrelated-4242",
}


class _FakeSandbox:
    """Stand-in for a docker terminal environment: what ``streams`` and ``sandbox_host`` read."""
    _docker_exe = "docker"
    _container_id = "c0ffee"

    def get_temp_dir(self):
        return "/sbx"


def _env_file(values: dict[str, str]) -> bytes:
    return "".join(f"{k}={v}\n" for k, v in values.items()).encode()


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "bot-desktop").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(runtime, "state_dir", lambda: home / "bot-desktop")
    monkeypatch.setattr(runtime, "_profile_name", lambda: "default")
    monkeypatch.setattr(sandbox_host, "_REPORTED_DROPS", set(), raising=False)
    return home


@pytest.fixture
def backend(monkeypatch):
    """The fake backend's env file contents and the commands run in it; ``run_in`` is the only transport."""
    state = {"env_file": _env_file({**LAUNCHER_VALUES, **EXTRA_LINES}), "calls": []}

    def run_in(env, argv, *, child_env=None, user=None, timeout=None, **_kw):
        state["calls"].append(list(argv))
        script = argv[-1] if argv else ""
        if "/env" in script and "launcher.pid" in script:
            return subprocess.CompletedProcess(argv, 0, state["env_file"], b"")
        return subprocess.CompletedProcess(argv, 1, b"", b"")

    monkeypatch.setattr(streams, "run_in", run_in)
    monkeypatch.setattr(sandbox_host, "_user_for", lambda env: None)
    return state


@pytest.fixture
def sandbox_screen(home, backend, monkeypatch):
    """A sandbox-hosted screen: the host-side marker exists and the terminal environment is registered."""
    env = _FakeSandbox()
    sandbox_host._marker().write_text(json.dumps({"display": ":20", "dir": "/sbx/hermes-bot-desktop/default",
                                                  "profile": "default", "backend": "DockerEnvironment",
                                                  "container": env._container_id, "docker": "docker"}))
    monkeypatch.setattr(runtime, "_sandbox_env", lambda *, create: env)
    monkeypatch.setattr(placement, "_setting", lambda: "terminal")
    monkeypatch.setattr(placement, "_terminal_backend", lambda: "docker")
    return env


def _leaked(env: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in env.items() if v in LAUNCHER_VALUES.values() or v in EXTRA_LINES.values()}


# ---- read boundary ------------------------------------------------------------------------------------------

def test_only_the_keys_the_launcher_writes_are_read_back(home, backend):
    published = sandbox_host.published_env(_FakeSandbox(), "default")
    assert published == LAUNCHER_VALUES


@pytest.mark.parametrize("display", ["not a display", "localhost:20", ":20; true", ":20.x", ":"])
def test_a_display_that_is_not_a_plain_x_display_means_not_published(home, backend, display):
    backend["env_file"] = _env_file({**LAUNCHER_VALUES, "DISPLAY": display})
    assert sandbox_host.published_env(_FakeSandbox(), "default") == {}


def test_empty_or_control_character_values_are_dropped(home, backend):
    backend["env_file"] = _env_file({**LAUNCHER_VALUES, "XAUTHORITY": "", "XDG_DATA_HOME": "/sbx/a\x1b[0m"})
    published = sandbox_host.published_env(_FakeSandbox(), "default")
    assert "XAUTHORITY" not in published and "XDG_DATA_HOME" not in published
    assert published["DISPLAY"] == ":20"


def test_dropped_keys_are_logged_once_by_name_without_their_value(home, backend, caplog):
    with caplog.at_level(logging.WARNING, logger=sandbox_host.logger.name):
        for _ in range(3):
            sandbox_host.published_env(_FakeSandbox(), "default")
    messages = [r.getMessage() for r in caplog.records if r.name == sandbox_host.logger.name]
    assert sum("LD_PRELOAD" in m for m in messages) == 1
    assert not any(v in m for m in messages for v in EXTRA_LINES.values())


@pytest.mark.platforms("linux")
def test_every_line_the_launcher_writes_is_read_back(tmp_path, home, caplog):
    """The real launcher.sh, X and Xfce stubbed out: the env file it publishes survives the read-back whole."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    stubs = {"Xvnc": "exec sleep 3", "dbus-run-session": 'shift; DBUS_SESSION_BUS_ADDRESS=unix:path=/run/bus exec "$@"'}
    for name in ("xdpyinfo", "setxkbmap", "xsetroot", "xset", "xauth", "xfsettingsd", "xfwm4", "xprop", "xfdesktop",
                 "xfce4-panel"):
        stubs[name] = "exit 0"
    for name, body in stubs.items():
        (bindir / name).write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
        (bindir / name).chmod(0o755)
    for tool in ("mkdir", "sed", "cat", "printf", "dirname", "bash", "sh", "rm", "ln", "touch", "chmod", "od", "tr",
                 "awk", "grep", "seq", "sleep", "kill", "mv"):
        if (real := shutil.which(tool)) and not (bindir / tool).exists():
            (bindir / tool).symlink_to(real)
    env_file = tmp_path / "env"
    subprocess.run(["bash", str(Path(sandbox_host.__file__).with_name("launcher.sh"))], check=True, timeout=30,
                   stdin=subprocess.DEVNULL, capture_output=True,
                   env={"PATH": str(bindir), "HOME": str(tmp_path), "HERMES_BD_PROFILE": "t",
                        "HERMES_BD_DISPLAY_NUM": "99", "HERMES_BD_SOCKET": str(tmp_path / "rfb.sock"),
                        "HERMES_BD_XAUTH": str(tmp_path / "Xauthority"), "HERMES_BD_ENV_FILE": str(env_file),
                        "HERMES_BD_CONFIG_HOME": str(tmp_path / "xdg")})
    written = dict(line.split("=", 1) for line in env_file.read_text(encoding="utf-8").splitlines())
    with caplog.at_level(logging.WARNING, logger=sandbox_host.logger.name):
        assert sandbox_host.parse_published(env_file.read_text(encoding="utf-8")) == written
    assert written["DISPLAY"] == ":99" and not caplog.records


# ---- agent-machine sinks (sandbox-hosted screen) ------------------------------------------------------------

def test_desktop_env_drops_this_machines_seat_for_a_sandbox_hosted_screen_but_stamps_activity(sandbox_screen):
    base = {"DISPLAY": ":0", "XAUTHORITY": "/home/u/.Xauthority", "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/u/bus",
            "WAYLAND_DISPLAY": "wayland-0", "PATH": "/usr/bin"}
    activity = runtime.state_dir() / "activity"
    assert not activity.exists()
    assert runtime.desktop_env(base) == {"PATH": "/usr/bin"}
    assert activity.exists(), "a spawn on a sandbox-hosted screen is still the agent using it"


def test_browser_and_cua_driver_envs_carry_none_of_the_backend_values(sandbox_screen, monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")
    from tools.browser_session import build_browser_env
    from tools.computer_use.cua_backend import cua_driver_child_env

    browser_env = build_browser_env()
    assert _leaked(browser_env) == {}
    assert "DISPLAY" not in browser_env
    cua_env = cua_driver_child_env({"DISPLAY": ":0", "PATH": "/usr/bin"})
    assert _leaked(cua_env) == {}
    assert "DISPLAY" not in cua_env


def test_a_host_cua_driver_never_gets_this_machines_seat_for_a_sandbox_hosted_screen(sandbox_screen, monkeypatch):
    """bounded / unrestricted computer_use runs its embedded daemon on THIS machine even under terminal placement:
    with no display it fails, where the seat would put an approval-bypassed driver on the human's own desktop."""
    for key, value in {"DISPLAY": ":0", "XAUTHORITY": "/home/u/.Xauthority",
                       "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/u/bus"}.items():
        monkeypatch.setenv(key, value)
    from tools.computer_use.cua_backend import CuaDriverBackend
    env = CuaDriverBackend("unrestricted")._embedded_daemon.child_env()
    assert not {"DISPLAY", "XAUTHORITY", "DBUS_SESSION_BUS_ADDRESS"} & set(env)


def test_local_terminal_env_carries_none_of_the_backend_values(sandbox_screen, monkeypatch):
    from tools.environments.local import _make_run_env
    monkeypatch.setenv("DISPLAY", ":0")
    env = _make_run_env({})
    assert _leaked(env) == {}
    assert env["DISPLAY"] == ":0"


# ---- controls: unchanged behavior ---------------------------------------------------------------------------

@pytest.fixture
def gateway_screen(home):
    sd = runtime.state_dir()
    (sd / "launcher.pid").write_text(f"{os.getpid()} {runtime._create_time(os.getpid())}", encoding="utf-8")
    values = {"DISPLAY": ":23", "XAUTHORITY": str(sd / "Xauthority"),
              "DBUS_SESSION_BUS_ADDRESS": f"unix:path={sd}/bus"}
    (sd / "env").write_text("".join(f"{k}={v}\n" for k, v in values.items()), encoding="utf-8")
    return values


def test_gateway_hosted_screen_still_merges_into_local_children(gateway_screen, monkeypatch):
    from tools.environments.local import _make_run_env
    env = runtime.desktop_env({"DISPLAY": ":0", "WAYLAND_DISPLAY": "wayland-0"})
    assert {k: env[k] for k in gateway_screen} == gateway_screen
    assert "WAYLAND_DISPLAY" not in env
    monkeypatch.setenv("DISPLAY", ":0")
    run_env = _make_run_env({})
    assert {k: run_env[k] for k in gateway_screen} == gateway_screen


def test_processes_inside_the_backend_still_receive_the_published_values(sandbox_screen, backend, monkeypatch):
    backend["env_file"] = _env_file(LAUNCHER_VALUES)  # exactly what the launcher writes
    assert runtime.published_env() == LAUNCHER_VALUES

    from tools import browser_session_sandbox
    env, published = browser_session_sandbox._sandbox()
    assert env is sandbox_screen and published == LAUNCHER_VALUES

    from tools.computer_use import cua_backend
    monkeypatch.setattr(streams, "exec_prefix", lambda e, *, user=None, interactive=True:
                        ["docker", "exec", "-i", e._container_id])
    invocation = cua_backend.sandbox_mcp_invocation()
    assert invocation is not None
    (command, args), child_env = invocation
    assert command == "docker" and "cua-driver mcp" in args[-1]
    for key, value in LAUNCHER_VALUES.items():
        assert f"export {key}=" in args[-1] and value in args[-1], key
    assert set(child_env) == {"PATH"}


def test_cached_driver_identity_follows_a_sandbox_hosted_screen(sandbox_screen, backend, monkeypatch):
    """The computer_use cache compares this identity to notice a screen that came up or went away."""
    from tools.computer_use import cua_backend
    monkeypatch.setenv("DISPLAY", ":0")
    assert cua_backend.desktop_identity() == ":20"
    backend["env_file"] = b""  # the screen is down: nothing published, and the seat is not the screen either
    assert cua_backend.desktop_identity() == ""
    assert cua_backend.backend_display_stale(":20", cua_backend.desktop_identity())


def test_cached_driver_identity_follows_a_gateway_hosted_screen(gateway_screen, monkeypatch):
    from tools.computer_use import cua_backend
    monkeypatch.setenv("DISPLAY", ":0")
    assert cua_backend.desktop_identity() == ":23"
