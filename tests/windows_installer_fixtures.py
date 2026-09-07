"""Native stage/exit/filesystem contracts; Python/package responses are synthetic.

Preserves egilewski's transaction contracts from #83149/#83194 and builds on
fangliquanflq's interrupted-retry recovery in #103771.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
INSTALLER = REPO / "scripts/install.ps1"
DRIVER = REPO / "scripts/tests/test-install-ps1-venv-retry.ps1"
ORIGINAL = "ORIGINAL_WORKING_ENV"
PARTIAL = "PARTIAL_REPLACEMENT"
VALIDATED = "VALIDATED_REPLACEMENT"
BACKUP_NAME = "venv.stale.20260905120000-" + "1" * 32


@pytest.fixture(scope="module")
def fake_uv(tmp_path_factory):
    root = tmp_path_factory.mktemp("installer-fake-uv")
    powershell = shutil.which("powershell.exe")
    assert powershell, "Native installer tests require Windows PowerShell"
    completed = subprocess.run(
        [powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(DRIVER), "-Root", str(root), "-BuildTool"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return root / "fixture-uv.exe"


@pytest.fixture(params=["powershell.exe", "pwsh.exe"])
def powershell_host(request):
    executable = shutil.which(request.param)
    assert executable, f"Native installer matrix requires {request.param}"
    return executable


@pytest.fixture
def install(tmp_path: Path, fake_uv: Path, powershell_host: str):
    (tmp_path / "fixture-host.txt").write_text(powershell_host, encoding="utf-8")
    home = tmp_path / "home"
    root = home / "hermes-agent"
    (home / "bin").mkdir(parents=True)
    shutil.copyfile(fake_uv, home / "bin/uv.exe")
    for directory in [root / ".hermes-runtime/python/fixture", root / "venv/Scripts"]:
        directory.mkdir(parents=True)
        shutil.copyfile(fake_uv, directory / "python.exe")
    (root / "venv/generation.txt").write_text(ORIGINAL, encoding="utf-8")
    return root


def _stage(install: Path, stage: str, mode: str = "ok", *, no_venv: bool = False):
    root = install.parent.parent
    command = [
        (root / "fixture-host.txt").read_text(encoding="utf-8"),
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(DRIVER),
        "-Root", str(root), "-Installer", str(INSTALLER), "-Stage", stage, "-Mode", mode,
    ]
    if no_venv:
        command.append("-NoVenv")
    completed = subprocess.run(
        command,
        cwd=root,
        env={**os.environ, "TEMP": str(root), "TMP": str(root)},
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=45,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    output = completed.stdout + completed.stderr
    with (root / "stage-output.log").open("a", encoding="utf-8") as log:
        log.write(f"\n{stage} {mode} exit={completed.returncode}\n{output}")
    frames = []
    for line in completed.stdout.splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and value.get("stage") == stage:
            frames.append(value)
    if completed.returncode in (-1, 4294967295):
        assert not frames, "bootstrap retryable process death must have no terminal frame"
    elif completed.returncode not in (91, 92, 93):
        assert len(frames) == 1, output
        assert frames[0]["ok"] == (completed.returncode == 0), output
    native_log = root / "native-events.txt"
    if completed.returncode and native_log.exists():
        output += "\nNative fixture calls:\n" + native_log.read_text(encoding="utf-8")
    return completed.returncode, output


def _pending(install: Path) -> str:
    return (install / "venv.pending-backup").read_text(encoding="ascii").strip()


def _generation(directory: Path) -> str:
    return (directory / "generation.txt").read_text(encoding="utf-8")


def _junction(install: Path, link: Path, target: Path):
    root = install.parent.parent
    result = subprocess.run(
        [(root / "fixture-host.txt").read_text(encoding="utf-8"), "-NoProfile", "-Command",
         "$ErrorActionPreference = 'Stop'; New-Item -ItemType Junction -Path $env:HERMES_TEST_LINK -Target $env:HERMES_TEST_TARGET | Out-Null"],
        env={**os.environ, "HERMES_TEST_LINK": str(link), "HERMES_TEST_TARGET": str(target)},
        capture_output=True, text=True, timeout=15, creationflags=subprocess.CREATE_NO_WINDOW,
    )
    assert result.returncode == 0, result.stdout + result.stderr
