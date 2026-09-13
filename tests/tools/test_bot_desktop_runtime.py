"""Bot Desktop runtime: thumbnail and display-number allocation invariants."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import pytest

from tools.bot_desktop import runtime, thumbnail


@pytest.mark.parametrize("pm", sorted(runtime.PACKAGES))
def test_every_required_binary_maps_to_an_installed_package(pm):
    """Each binary the launcher execs must come from a package the distro list actually installs; dnf5
    refuses the whole transaction on one retired name, so the map is the contract, not the list."""
    mapping = runtime.BINARY_PACKAGES[pm]
    assert set(mapping) == set(runtime.REQUIRED_BINARIES)
    assert set(mapping.values()) <= set(runtime.PACKAGES[pm])
    assert not {"xorg-x11-server-utils", "xorg-x11-utils"} & set(runtime.PACKAGES["dnf"]), "retired on Fedora"


def test_privileged_executable_resolution_ignores_inherited_path(tmp_path, monkeypatch):
    attacker = tmp_path / "attacker"
    system = tmp_path / "system"
    attacker.mkdir()
    system.mkdir()
    for directory in (attacker, system):
        executable = directory / "sudo"
        executable.write_text("#!/bin/sh\n", encoding="utf-8")
        executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(attacker))
    monkeypatch.setattr(runtime, "_SYSTEM_PATH", str(system))

    assert runtime._system_executable("sudo") == str(system / "sudo")


def test_no_running_screen_returns_none_without_grabbing(monkeypatch):
    monkeypatch.setattr(runtime, "published_env", lambda: {"DISPLAY": ":99"})
    monkeypatch.setattr(runtime, "_launcher_pid", lambda: None)
    monkeypatch.setitem(sys.modules, "PIL.ImageGrab", None)  # an import would now fail loudly
    assert thumbnail.thumbnail_data_url() is None


def test_recycled_pid_is_not_our_launcher(tmp_path, monkeypatch):
    """launcher.pid names pid + create_time; a live pid born at another time is a stranger (recycled pid)
    and must read as not running, or stop() would killpg an unrelated session. Legacy single-number
    files and absurd digit strings are also not running."""
    import os

    monkeypatch.setattr(runtime, "state_dir", lambda: tmp_path)
    pidfile = tmp_path / "launcher.pid"
    pidfile.write_text(f"{os.getpid()} 12345.0", encoding="utf-8")  # alive, wrong birth
    assert runtime._launcher_pid() is None
    pidfile.write_text(str(os.getpid()), encoding="utf-8")  # pre-identity format
    assert runtime._launcher_pid() is None
    pidfile.write_text("9" * 40 + " 1.0", encoding="utf-8")
    assert runtime._launcher_pid() is None
    pidfile.write_text(f"{os.getpid()} {runtime._create_time(os.getpid())}", encoding="utf-8")
    assert runtime._launcher_pid() == os.getpid()


def test_recorded_display_held_by_a_live_server_is_not_reused(tmp_path, monkeypatch):
    """After profile A stops, B may take A's number; A restarting must pick another rather than
    unlink B's socket and lock."""
    import os

    monkeypatch.setattr(runtime, "state_dir", lambda: tmp_path)
    (tmp_path / "display").write_text("37", encoding="utf-8")
    live = {37: os.getpid()}  # :37 is owned by a running server (this very process stands in for it)
    monkeypatch.setattr(runtime, "_display_in_use", lambda num: num in live)
    monkeypatch.setattr(runtime, "_ALLOC_LOCK", tmp_path / "alloc.lock")
    assert runtime._allocate_display() != 37
    live.clear()
    assert runtime._allocate_display() == 37, "a free recorded number is reclaimed"


def test_desktop_launcher_does_not_inherit_credentials(tmp_path, monkeypatch):
    """The graphical session keeps benign host settings but never receives Hermes credentials."""
    import os

    from tools.bot_desktop import browser

    captured: dict[str, str] = {}

    class FakeProcess:
        pid = os.getpid()
        returncode = None

        @staticmethod
        def poll():
            return None

    def fake_popen(*_args, **kwargs):
        captured.update(kwargs["env"])
        Path(captured["HERMES_BD_ENV_FILE"]).write_text("DISPLAY=:44\n", encoding="utf-8")
        Path(captured["HERMES_BD_SOCKET"]).touch()
        return FakeProcess()

    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "github-secret")
    aws_credentials = {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN",
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_CONFIG_FILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_ROLE_ARN",
        "AWS_ROLE_SESSION_NAME",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
    }
    for key in aws_credentials:
        monkeypatch.setenv(key, "aws-secret")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
    monkeypatch.setenv("BOT_DESKTOP_BENIGN", "keep-me")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(runtime.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(runtime, "geometry", lambda: "800x600")
    monkeypatch.setattr(runtime, "status", lambda: "ready")
    monkeypatch.setattr(browser, "dock_launch", lambda: None)

    assert runtime._spawn_and_wait(tmp_path, 44, 0.1) == "ready"
    assert captured["BOT_DESKTOP_BENIGN"] == "keep-me"
    assert "DISPLAY" not in captured
    assert "OPENAI_API_KEY" not in captured
    assert "ANTHROPIC_API_KEY" not in captured
    assert "GITHUB_TOKEN" not in captured
    assert aws_credentials.isdisjoint(captured)
    assert captured["AWS_REGION"] == "us-east-1"
    assert captured["AWS_DEFAULT_REGION"] == "us-west-2"


_FAKE_LAUNCHER = """#!/usr/bin/env bash
# Stands in for launcher.sh + Xvnc: the X lock appears only after a delay (the TOCTOU window), then the
# env file + socket are published; stays alive until killed like the real supervisor.
: > "$HERMES_BD_XLOCK_DIR/spawned.$$"
sleep 0.4
echo $$ > "$HERMES_BD_XLOCK_DIR/.X${HERMES_BD_DISPLAY_NUM}-lock"
: > "$HERMES_BD_SOCKET"
printf 'DISPLAY=:%s\\n' "$HERMES_BD_DISPLAY_NUM" > "$HERMES_BD_ENV_FILE"
sleep 30
"""

# One start() per process: state_dir() is HERMES_HOME-scoped and process-global, so two profiles need two
# interpreters — which is also how two gateway profiles race on a real host.
_DRIVER = """
import json, os, sys
from pathlib import Path
sys.path.insert(0, {repo!r})
from tools.bot_desktop import runtime
scratch = Path({scratch!r})
runtime._LAUNCHER = scratch / "launcher.sh"
runtime._X_LOCK_DIR = scratch / "xlocks"
runtime._ALLOC_LOCK = scratch / "alloc.lock"
runtime.missing_binaries = lambda: []
runtime.geometry = lambda: "800x600"
os.environ["HERMES_BD_XLOCK_DIR"] = str(scratch / "xlocks")
try:
    st = runtime.start(wait_seconds=10)
    print(json.dumps({{"display": st.display, "pid": st.pid}}), flush=True)
except Exception as exc:
    print(json.dumps({{"error": str(exc)}}), flush=True)
sys.stdin.readline()  # the test releases us once every driver has reported; we own the launcher, we stop it
runtime.stop()
"""


@pytest.fixture
def start_in_fresh_process(tmp_path):
    import os
    import subprocess

    (tmp_path / "launcher.sh").write_text(_FAKE_LAUNCHER, encoding="utf-8")
    (tmp_path / "xlocks").mkdir()
    repo = str(Path(__file__).resolve().parents[2])
    procs: list[subprocess.Popen] = []

    def launch(home: Path) -> subprocess.Popen:
        env = {**os.environ, "HERMES_HOME": str(home)}
        proc = subprocess.Popen([sys.executable, "-c", _DRIVER.format(repo=repo, scratch=str(tmp_path))],
                                env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        procs.append(proc)
        return proc

    yield launch
    for proc in procs:
        with contextlib.suppress(OSError):
            proc.communicate("go\n", timeout=20)
        proc.kill()


def _collect(procs):
    import json
    out = [json.loads(p.stdout.readline()) for p in procs]  # every driver holds its launcher until released
    assert all("error" not in o for o in out), out
    return out


@pytest.mark.linux_only
def test_concurrent_cold_starts_of_two_profiles_get_distinct_displays(tmp_path, start_in_fresh_process):
    """The allocation lock must outlive the pick: Xvnc writes /tmp/.X<n>-lock well after start() chose n, so
    a second profile starting in that window used to pick the same n (and its launcher's stale-lock cleanup
    could then unlink the winner's socket)."""
    out = _collect([start_in_fresh_process(tmp_path / "a"), start_in_fresh_process(tmp_path / "b")])
    assert len({o["display"] for o in out}) == 2, out


@pytest.mark.linux_only
def test_concurrent_starts_of_one_profile_spawn_one_launcher(tmp_path, start_in_fresh_process):
    """Two start() calls for one profile spawn ONE launcher; the second used to spawn its own, overwrite
    launcher.pid and orphan the first (both callers then reported the last-written pid)."""
    out = _collect([start_in_fresh_process(tmp_path / "a"), start_in_fresh_process(tmp_path / "a")])
    assert len({o["pid"] for o in out}) == 1, out
    assert len(list((tmp_path / "xlocks").glob("spawned.*"))) == 1
