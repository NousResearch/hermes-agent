"""The image's root s6 supervision tree must stay out of the unprivileged ``hermes`` user's reach.

Everything under the root ``s6-svscan`` scandir (``/run/service``) is supervised by a root
``s6-supervise``, which executes each slot's ``run``/``finish``/``log/run`` as root and writes
its ``supervise/`` state as root. The cont-init scripts and the static s6-rc services are run
here against fake privileged binaries so their ownership decisions can be checked without a
container.
"""
from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER_DIR = REPO_ROOT / "docker"
ROOT_SCANDIR = "/run/service"


def _sh() -> str:
    shell = shutil.which("sh")
    if shell is None:
        pytest.skip("sh not available")
    return shell


def _fake_bin(tmp_path: Path, log: Path, names: dict[str, str], sub: str = "fakebin") -> Path:
    """Fake binaries that append ``<name> <args>`` to ``log`` and print a canned stdout."""
    bindir = tmp_path / sub
    bindir.mkdir()
    for name, stdout in names.items():
        script = bindir / name
        script.write_text(
            "#!/bin/sh\n"
            f'printf "%s\\n" "{name} $*" >> "{log}"\n'
            + (f"printf '%s\\n' '{stdout}'\n" if stdout else "")
            + "exit 0\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
    return bindir


def _run(script: Path, bindir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "HERMES_DASHBOARD"}
    env["PATH"] = f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}"
    return subprocess.run([_sh(), str(script), *args], capture_output=True, text=True, env=env, check=False)


def _calls(log: Path, name: str) -> list[list[str]]:
    if not log.exists():
        return []
    return [line.split()[1:] for line in log.read_text().splitlines() if line.split()[:1] == [name]]


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _under_root_scandir(path: str) -> bool:
    return path == ROOT_SCANDIR or path.startswith(ROOT_SCANDIR + "/")


@pytest.mark.platforms("posix")
def test_reconcile_cont_init_leaves_root_scandir_root_owned(tmp_path: Path) -> None:
    """02-reconcile-profiles must not hand /run/service (or its svscan control) to hermes:
    a hermes-writable root scandir lets hermes publish a slot that root s6-supervise executes."""
    log = tmp_path / "calls.log"
    bindir = _fake_bin(
        tmp_path, log, {"chown": "", "chmod": "", "mkdir": "", "id": "0", "s6-setuidgid": ""}
    )

    proc = _run(DOCKER_DIR / "cont-init.d" / "02-reconcile-profiles", bindir)

    assert proc.returncode == 0, proc.stderr
    for args in _calls(log, "chown") + _calls(log, "chmod"):
        assert not any(_under_root_scandir(a) for a in args), args
    # The reconciler itself still runs, dropped to hermes.
    assert any(args[:1] == ["hermes"] and "hermes_cli.container_boot" in args
               for args in _calls(log, "s6-setuidgid"))


@pytest.mark.platforms("posix")
def test_profile_gateway_scandir_is_supervised_unprivileged(tmp_path: Path) -> None:
    """Per-profile gateway slots are hermes-writable, so the s6-svscan watching them must run as
    hermes (never as root), and on the same directory the service manager registers into."""
    from hermes_agent.hermes_cli.service_manager import S6ServiceManager

    gateway_scandir = str(S6ServiceManager().scandir)
    log = tmp_path / "calls.log"
    bindir = _fake_bin(tmp_path, log, {"id": "0", "s6-setuidgid": "", "s6-svscan": "", "sleep": ""})
    rc_dir = DOCKER_DIR / "s6-rc.d"
    runners = sorted(rc_dir.glob("*/run"))

    supervisors = []
    for run in runners:
        log.unlink(missing_ok=True)
        proc = _run(run, bindir)
        assert proc.returncode == 0, (run, proc.stderr)
        for args in _calls(log, "s6-setuidgid"):
            if args[1:2] == ["s6-svscan"]:
                supervisors.append((run.parent.name, args))
        assert not _calls(log, "s6-svscan"), f"{run} runs s6-svscan as root"

    assert len(supervisors) == 1, supervisors
    name, args = supervisors[0]
    assert args[0] == "hermes"
    assert args[-1] == gateway_scandir
    assert not _under_root_scandir(gateway_scandir)
    # Wired into the user bundle so s6-rc starts it after cont-init has seeded the scandir.
    assert (rc_dir / name / "type").read_text().strip() == "longrun"
    assert (rc_dir / "user" / "contents.d" / name).exists()
    assert (rc_dir / name / "dependencies.d" / "base").exists()

    # cont-init seeds that same directory hermes-owned before s6-rc starts the supervisor.
    log.unlink(missing_ok=True)
    bindir2 = _fake_bin(
        tmp_path, log, {"chown": "", "chmod": "", "mkdir": "", "id": "0", "s6-setuidgid": ""},
        sub="fakebin-cont-init",
    )
    proc = _run(DOCKER_DIR / "cont-init.d" / "02-reconcile-profiles", bindir2)
    assert proc.returncode == 0, proc.stderr
    assert ["hermes:hermes", gateway_scandir] in _calls(log, "chown")


@pytest.mark.platforms("posix")
def test_static_supervise_trees_stay_root_owned(tmp_path: Path) -> None:
    """015-supervise-perms may let the hermes group query/control static services, but must not
    make hermes the owner of anything a root s6-supervise writes to."""
    svc_root = tmp_path / "servicedirs"
    for name in ("dashboard", "s6rc-oneshot-runner"):
        sup = svc_root / name / "supervise"
        sup.mkdir(parents=True)
        sup.chmod(0o700)
        os.mkfifo(sup / "control", 0o600)
        for f in ("lock", "status"):
            (sup / f).write_bytes(b"")
            (sup / f).chmod(0o600)
        (svc_root / name / "event").mkdir()
        (svc_root / name / "event").chmod(0o700)

    text = (DOCKER_DIR / "cont-init.d" / "015-supervise-perms").read_text(encoding="utf-8")
    script_text, n = re.subn(r"(?m)^SVC_ROOT=.*$", f"SVC_ROOT={svc_root}", text)
    assert n == 1
    script = tmp_path / "015-supervise-perms"
    script.write_text(script_text, encoding="utf-8")
    log = tmp_path / "calls.log"
    bindir = _fake_bin(tmp_path, log, {"chown": ""})

    proc = _run(script, bindir)

    assert proc.returncode == 0, proc.stderr
    chowns = _calls(log, "chown")
    assert chowns, "static services were not made reachable for the hermes group"
    for args in chowns:
        owner = next(a for a in args if not a.startswith("-"))
        assert owner.split(":")[0] == "root", args
        assert not any("/event" in a or "s6rc-" in a for a in args), args

    sup = svc_root / "dashboard" / "supervise"
    assert _mode(sup) & 0o002 == 0 and _mode(sup) & 0o020 == 0  # nobody but root may create entries
    assert _mode(sup / "control") & 0o020  # hermes group can s6-svc
    assert _mode(sup / "status") & 0o040 and _mode(sup / "lock") & 0o040  # and s6-svstat
    assert _mode(svc_root / "dashboard" / "event") == 0o700  # event fifodir untouched
    assert _mode(svc_root / "s6rc-oneshot-runner" / "supervise") == 0o700
