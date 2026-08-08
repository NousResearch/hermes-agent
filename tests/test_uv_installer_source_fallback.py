"""Installer entry points must tolerate either official uv script host failing."""

import subprocess
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"

PRIMARY_PS1 = "https://astral.sh/uv/install.ps1"
FALLBACK_PS1 = (
    "https://github.com/astral-sh/uv/releases/latest/download/uv-installer.ps1"
)
PRIMARY_SH = "https://astral.sh/uv/install.sh"
FALLBACK_SH = (
    "https://github.com/astral-sh/uv/releases/latest/download/uv-installer.sh"
)


def test_managed_uv_posix_retries_official_github_source(monkeypatch):
    import hermes_cli.managed_uv as managed_uv

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[0] == "curl" and "astral.sh" in cmd[2]:
            raise subprocess.CalledProcessError(22, cmd)
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(managed_uv.subprocess, "run", fake_run)

    managed_uv._install_uv_posix({})

    download_urls = [cmd[2] for cmd in calls if cmd[0] == "curl"]
    assert download_urls == [PRIMARY_SH, FALLBACK_SH]
    assert calls[-1][0] == "sh"


def test_managed_uv_windows_retries_official_github_source(monkeypatch):
    import hermes_cli.managed_uv as managed_uv

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "astral.sh" in cmd[-1]:
            raise subprocess.CalledProcessError(1, cmd)
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(managed_uv.subprocess, "run", fake_run)

    managed_uv._install_uv_windows({})

    commands = [cmd[-1] for cmd in calls]
    assert commands == [
        f"$ErrorActionPreference = 'Stop'; irm {PRIMARY_PS1} | iex",
        f"$ErrorActionPreference = 'Stop'; irm {FALLBACK_PS1} | iex",
    ]


def test_windows_installer_retries_github_when_primary_did_not_install_uv():
    source = INSTALL_PS1.read_text(encoding="utf-8")
    install_uv = source.split("function Install-Uv", 1)[1].split(
        "function Sync-EnvPath", 1
    )[0]

    assert PRIMARY_PS1 in install_uv
    assert FALLBACK_PS1 in install_uv
    assert install_uv.index(PRIMARY_PS1) < install_uv.index(FALLBACK_PS1)
    assert "if (-not (Test-Path $managedUv))" in install_uv
    invocation_lines = [
        line
        for line in install_uv.splitlines()
        if "uv/install.ps1 | iex" in line or "uv-installer.ps1 | iex" in line
    ]
    assert len(invocation_lines) == 2
    assert all("$ErrorActionPreference = 'Stop';" in line for line in invocation_lines)


def test_posix_installer_tries_both_official_sources_before_failing():
    source = INSTALL_SH.read_text(encoding="utf-8")
    install_uv = source.split("install_uv()", 1)[1].split("check_python()", 1)[0]

    assert PRIMARY_SH in install_uv
    assert FALLBACK_SH in install_uv
    assert install_uv.index(PRIMARY_SH) < install_uv.index(FALLBACK_SH)
    assert "for _uv_installer_url in" in install_uv
    assert 'curl -LsSf "$_uv_installer_url"' in install_uv
