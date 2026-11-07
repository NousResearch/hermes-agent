"""Root-context scripts in the Docker image resolve commands from system directories only.

PID 1, the cont-init scripts, the static s6 services, the main wrapper and root ``docker exec``
all start as root, while ``/opt/data`` (and its ``.local/bin``) belongs to the unprivileged
hermes user. So: the image ``PATH`` lists only root-owned directories; every root script names
an absolute interpreter, pins its own ``PATH`` and calls the drop tools by absolute path; and the
hermes user's ``~/.local/bin`` is handed, after the system dirs, only to commands that already
run as hermes.

The behavioral cases run each script with a directory of planted commands first on the
inherited ``PATH`` and assert that none of them run.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.tools.docker_root_script_harness import (
    HERMES_LOCAL_BIN,
    SYSTEM_PATH,
    calls,
    dropped_command,
    records,
    redirect_root_tools,
    write_fake_id,
    write_recorders,
)

ROOT = Path(__file__).resolve().parents[2]
DOCKER = ROOT / "docker"
SYSTEM_DIRS = ("/usr/local/sbin", "/usr/local/bin", "/usr/sbin", "/usr/bin", "/sbin", "/bin")

# Every name a root-context script runs, or that with-contenv / the drop could resolve.
PLANTED = (
    "sh", "bash", "dash", "env", "id", "s6-setuidgid", "s6-svscan", "s6-envdir", "with-contenv",
    "sleep", "chown", "chmod", "chgrp", "mkdir", "basename", "dirname", "find", "grep", "stat",
    "cat", "cp", "rm", "touch", "tr", "cut", "head", "od", "sed", "getent", "usermod", "groupmod",
    "groupadd", "python", "python3", "hermes",
)
# Tools the scripts may run for real-world side effects; recorded instead of run.
RECORDED = ("chown", "chmod", "chgrp", "usermod", "groupmod", "groupadd", "s6-setuidgid", "s6-svscan", "sleep")


def _image_path() -> list[str]:
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    values = re.findall(r'^ENV PATH="([^"]+)"', text, flags=re.MULTILINE)
    assert values, "Dockerfile sets no ENV PATH"
    return values[-1].replace("${PATH}", ":".join(SYSTEM_DIRS)).split(":")


def test_image_path_lists_no_data_volume_dir() -> None:
    entries = _image_path()
    assert not [e for e in entries if e == "/opt/data" or e.startswith("/opt/data/")], entries
    assert entries[:2] == ["/opt/hermes/bin", "/opt/hermes/.venv/bin"], entries


def _root_scripts() -> list[Path]:
    out = sorted((DOCKER / "cont-init.d").iterdir())
    for svc in sorted((DOCKER / "s6-rc.d").iterdir()):
        out += [p for p in (svc / "run", svc / "finish") if p.is_file()]
    names = ("stage2-hook.sh", "main-wrapper.sh", "entrypoint-dispatch.sh", "entrypoint.sh",
             "hermes-exec-shim.sh", "tini-shim.sh")
    return out + [DOCKER / n for n in names]


def test_root_scripts_name_an_absolute_interpreter() -> None:
    bad = []
    for p in _root_scripts():
        text = p.read_text(encoding="utf-8")
        words = text.splitlines()[0].removeprefix("#!").split()
        if not words or not words[0].startswith("/"):
            bad.append(f"{p.relative_to(ROOT)}: {text.splitlines()[0]}")
        # with-contenv execs its argument through the container PATH: it must be absolute too.
        bad += [f"{p.relative_to(ROOT)}: with-contenv {arg}"
                for arg in re.findall(r"(?m)(?:^#!|exec\s+)/command/with-contenv\s+(\S+)", text) if not arg.startswith("/")]
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    generated = re.findall(r"printf '#!/command/with-contenv ([^\\\s]+)", dockerfile)
    assert generated, "Dockerfile no longer generates the 01-hermes-setup cont-init"
    bad += [f"Dockerfile 01-hermes-setup: {g}" for g in generated if not g.startswith("/")]
    assert not bad, "\n".join(bad)


def _sh() -> str:
    if not Path("/bin/sh").exists() or shutil.which("sh") is None:
        pytest.skip("sh not available")
    return "/bin/sh"


class _Harness:
    """A planted-command dir for the inherited PATH plus recorders for the redirected tools."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.planted_log = tmp_path / "planted.log"
        self.log = tmp_path / "calls.log"
        self.planted = write_recorders(tmp_path / "planted", self.planted_log, dict.fromkeys(PLANTED, ""))
        self.fake = write_recorders(tmp_path / "fake", self.log, dict.fromkeys(RECORDED, ""))
        write_fake_id(self.fake)
        self.home = tmp_path / "home"
        self.home.mkdir()

    def run(self, script: Path, *args: str, rewrite: dict[str, str] | None = None,
            env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        text = redirect_root_tools(script.read_text(encoding="utf-8"), self.fake)
        for old, new in (rewrite or {}).items():
            assert text.count(old) == 1, (script, old)
            text = text.replace(old, new)
        copy = self.tmp / f"{script.parent.name}-{script.name}"
        copy.write_text(text, encoding="utf-8")
        base = {
            "PATH": f"{self.planted}:{self.fake}:/usr/bin:/bin",
            "HOME": str(self.home),
            "HERMES_HOME": str(self.home),
            "XDG_RUNTIME_DIR": str(self.tmp / "xdg"),
            "LANG": "C.UTF-8",
        }
        return subprocess.run([_sh(), str(copy), *args], capture_output=True, text=True,
                              env={**base, **(env or {})}, timeout=60, check=False,
                              stdin=subprocess.DEVNULL, cwd=self.home)

    def planted_runs(self) -> list[list[str]]:
        return records(self.planted_log)

    def drops(self) -> list[tuple[str | None, list[str]]]:
        out = []
        for args in calls(self.log, "s6-setuidgid"):
            assert args[:1] == ["hermes"], args
            out.append(dropped_command(args))
        return out


def _assert_hermes_path(child_path: str | None, inherited: str) -> None:
    """The dropped child gets the inherited PATH, with ~/.local/bin appended after it."""
    assert child_path is not None, "the dropped child got no PATH of its own"
    entries = child_path.split(":")
    assert entries[-1] == HERMES_LOCAL_BIN, child_path
    assert ":".join(entries[:-1]) == inherited, child_path


def _supervise_tree(tmp_path: Path) -> Path:
    svc_root = tmp_path / "servicedirs"
    sup = svc_root / "dashboard" / "supervise"
    sup.mkdir(parents=True)
    os.mkfifo(sup / "control", 0o600)
    for f in ("lock", "status"):
        (sup / f).write_bytes(b"")
    return svc_root


# (script, args, extra env, {exact text: replacement}, expected dropped argv0 basename or None)
def _cases(tmp_path: Path) -> dict[str, tuple]:
    home = tmp_path / "home"
    return {
        "015-supervise-perms": (DOCKER / "cont-init.d" / "015-supervise-perms", (), {},
                                {"SVC_ROOT=/run/s6-rc/servicedirs": f"SVC_ROOT={_supervise_tree(tmp_path)}"},
                                None),
        "02-reconcile-profiles": (DOCKER / "cont-init.d" / "02-reconcile-profiles", (), {},
                                  {"GATEWAY_SCANDIR=/run/hermes-service": f"GATEWAY_SCANDIR={tmp_path / 'scan'}"},
                                  "python"),
        "dashboard-run": (DOCKER / "s6-rc.d" / "dashboard" / "run", (), {"HERMES_DASHBOARD": "1"},
                          {"\ncd /opt/data\n": f"\ncd {home}\n"}, "hermes"),
        "dashboard-finish": (DOCKER / "s6-rc.d" / "dashboard" / "finish", ("0",), {}, {}, None),
        "main-hermes-run": (DOCKER / "s6-rc.d" / "main-hermes" / "run", (), {}, {}, "sleep"),
        "profile-gateways-run": (DOCKER / "s6-rc.d" / "profile-gateways" / "run", (), {}, {}, "s6-svscan"),
        "main-wrapper-default": (DOCKER / "main-wrapper.sh", (), {}, {"\ncd /opt/data\n": f"\ncd {home}\n"},
                                 "hermes"),
        "main-wrapper-executable": (DOCKER / "main-wrapper.sh", ("sleep", "1"), {},
                                    {"\ncd /opt/data\n": f"\ncd {home}\n"}, "sleep"),
        "main-wrapper-subcommand": (DOCKER / "main-wrapper.sh", ("chat", "-q", "hi"), {},
                                    {"\ncd /opt/data\n": f"\ncd {home}\n"}, "hermes"),
    }


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("case", [
    "015-supervise-perms", "02-reconcile-profiles", "dashboard-run", "dashboard-finish", "main-hermes-run",
    "profile-gateways-run", "main-wrapper-default", "main-wrapper-executable", "main-wrapper-subcommand",
])
def test_root_script_never_runs_a_command_from_the_inherited_path(tmp_path: Path, case: str) -> None:
    h = _Harness(tmp_path)
    script, args, env, rewrite, dropped = _cases(tmp_path)[case]

    proc = h.run(script, *args, rewrite=rewrite, env=env)

    assert h.planted_runs() == [], f"root context ran planted commands: {h.planted_runs()}\n{proc.stderr}"
    # The dashboard finish script reports a disabled dashboard as a permanent failure (125).
    assert proc.returncode == (125 if case == "dashboard-finish" else 0), proc.stderr
    drops = h.drops()
    if dropped is None:
        assert drops == []
        return
    assert len(drops) == 1, drops
    child_path, child = drops[0]
    assert Path(child[0]).name == dropped, child
    if case == "main-hermes-run":
        # The no-op main service sleeps, as hermes, by absolute path.
        assert child[0].startswith("/") and child[1:] == ["infinity"], child
        return
    _assert_hermes_path(child_path, f"{h.planted}:{h.fake}:/usr/bin:/bin")


@pytest.mark.platforms("linux")
def test_stage2_hook_never_runs_a_command_from_the_inherited_path(tmp_path: Path) -> None:
    h = _Harness(tmp_path)

    proc = h.run(DOCKER / "stage2-hook.sh", env={"HERMES_UID": "4242", "HERMES_GID": "4242"})

    assert h.planted_runs() == [], f"root context ran planted commands: {h.planted_runs()}\n{proc.stderr}"
    assert proc.returncode == 0, proc.stderr
    assert "[stage2] Setup complete" in proc.stdout, proc.stdout
    assert calls(h.log, "usermod") and calls(h.log, "groupmod"), "the UID/GID remap did not run"
    drops = h.drops()
    assert drops, "stage2 ran nothing as hermes"
    for child_path, _child in drops:
        _assert_hermes_path(child_path, f"{h.planted}:{h.fake}:/usr/bin:/bin")


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("uid", ["0", "10000"])
def test_exec_shim_decides_and_drops_without_the_inherited_path(tmp_path: Path, uid: str) -> None:
    h = _Harness(tmp_path)
    real = write_recorders(tmp_path / "venv", h.log, {"hermes": ""}) / "hermes"

    proc = h.run(DOCKER / "hermes-exec-shim.sh", "--version", env={"FAKE_UID": uid},
                 rewrite={"REAL=/opt/hermes/.venv/bin/hermes": f"REAL={real}"})

    assert h.planted_runs() == [], f"the shim ran planted commands: {h.planted_runs()}\n{proc.stderr}"
    assert proc.returncode == 0, proc.stderr
    inherited = f"{h.planted}:{h.fake}:/usr/bin:/bin"
    if uid == "0":
        (child_path, child), = h.drops()
        assert child == [str(real), "--version"]
        _assert_hermes_path(child_path, inherited)
    else:
        assert h.drops() == []
        assert calls(h.log, "hermes") == [["--version"]]


# Arguments a shell would split, glob, expand or drop if any hop lost the quoting.
TRICKY_ARGS = ("", "a b", "it's", "$HOME", "*", "--flag=x y", " lead", "tab\there", "line\nbreak")


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("first", ["chat", "sleep"])
def test_main_wrapper_hands_the_dropped_child_its_argv_unchanged(tmp_path: Path, first: str) -> None:
    h = _Harness(tmp_path)
    (h.home / "glob-bait").write_text("", encoding="utf-8")

    proc = h.run(DOCKER / "main-wrapper.sh", first, *TRICKY_ARGS,
                 rewrite={"\ncd /opt/data\n": f"\ncd {h.home}\n"})

    assert proc.returncode == 0, proc.stderr
    (child_path, child), = h.drops()
    # A subcommand runs through hermes; an executable on the PATH runs directly.
    expected = ["hermes", first, *TRICKY_ARGS] if first == "chat" else [first, *TRICKY_ARGS]
    assert child == expected
    _assert_hermes_path(child_path, f"{h.planted}:{h.fake}:/usr/bin:/bin")


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("uid", ["0", "10000"])
def test_exec_shim_hands_hermes_its_argv_unchanged(tmp_path: Path, uid: str) -> None:
    h = _Harness(tmp_path)
    real = write_recorders(tmp_path / "venv", h.log, {"hermes": ""}) / "hermes"
    (h.home / "glob-bait").write_text("", encoding="utf-8")

    proc = h.run(DOCKER / "hermes-exec-shim.sh", *TRICKY_ARGS, env={"FAKE_UID": uid},
                 rewrite={"REAL=/opt/hermes/.venv/bin/hermes": f"REAL={real}"})

    assert proc.returncode == 0, proc.stderr
    if uid == "0":
        (_child_path, child), = h.drops()
        assert child == [str(real), *TRICKY_ARGS]
    else:
        assert h.drops() == []
        assert calls(h.log, "hermes") == [list(TRICKY_ARGS)]


def _login_path(tmp_path: Path, uid: str) -> list[str]:
    h = _Harness(tmp_path)
    text = redirect_root_tools((DOCKER / "profile-hermes-path.sh").read_text(encoding="utf-8"), h.fake)
    profile = tmp_path / "hermes.sh"
    profile.write_text(text, encoding="utf-8")
    login_path = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    proc = subprocess.run(
        [_sh(), "-c", f'PATH="{h.planted}:{login_path}"; . "{profile}"; printf "%s" "$PATH"'],
        capture_output=True, text=True, env={"FAKE_UID": uid, "PATH": "/usr/bin:/bin"},
        timeout=30, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert h.planted_runs() == [], h.planted_runs()
    return proc.stdout.split(":")


@pytest.mark.platforms("posix")
def test_login_profile_keeps_the_hermes_local_bin_away_from_root(tmp_path: Path) -> None:
    entries = _login_path(tmp_path, "0")
    assert entries[:2] == ["/opt/hermes/bin", "/opt/hermes/.venv/bin"], entries
    assert HERMES_LOCAL_BIN not in entries, entries


@pytest.mark.platforms("posix")
def test_login_profile_appends_the_hermes_local_bin_for_hermes(tmp_path: Path) -> None:
    entries = _login_path(tmp_path, "10000")
    assert entries[:2] == ["/opt/hermes/bin", "/opt/hermes/.venv/bin"], entries
    assert entries[-1] == HERMES_LOCAL_BIN and entries.count(HERMES_LOCAL_BIN) == 1, entries


@pytest.mark.platforms("posix")
def test_supervised_gateway_gets_the_hermes_local_bin_after_the_image_path(tmp_path: Path) -> None:
    """with-contenv restores the image PATH (no data-volume dir); the gateway, which runs as
    hermes, still finds ~/.local/bin tools, after the system dirs."""
    from hermes_cli.service_manager import S6ServiceManager

    image_bin = tmp_path / "image-bin"
    image_bin.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    lines = S6ServiceManager._render_run_script("default", {}).splitlines()
    setup = [line for line in lines[: next(i for i, line in enumerate(lines) if "gateway run" in line)]
             if line != "cd /opt/data"]
    image_path = f"{image_bin}:{SYSTEM_PATH}"

    proc = subprocess.run([_sh(), "-c", "\n".join([*setup, 'printf "%s" "$PATH"'])], capture_output=True,
                          text=True, timeout=30, check=False, cwd=home,
                          env={"PATH": image_path, "FAKE_UID": "10000", "HOME": str(home)})

    assert proc.returncode == 0, proc.stderr
    seen = proc.stdout.split(":")
    assert seen[-1] == HERMES_LOCAL_BIN and ":".join(seen[:-1]) == image_path, seen
