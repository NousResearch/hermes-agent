"""Contracts for files copied into the Linux container runtime."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_TREES = (
    REPO_ROOT / "docker" / "cont-init.d",
    REPO_ROOT / "docker" / "s6-rc.d",
)


def _runtime_files() -> list[Path]:
    return sorted(
        path
        for tree in RUNTIME_TREES
        for path in tree.rglob("*")
        if path.is_file() and path.stat().st_size
    )


def test_s6_runtime_files_declare_lf_checkout_policy() -> None:
    """Every s6 runtime file must resolve to ``eol=lf`` in Git attributes."""
    if shutil.which("git") is None or not (REPO_ROOT / ".git").exists():
        pytest.skip("Git metadata is unavailable in this source tree")

    relative_paths = [
        path.relative_to(REPO_ROOT).as_posix() for path in _runtime_files()
    ]
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={REPO_ROOT.as_posix()}",
            "check-attr",
            "eol",
            "--",
            *relative_paths,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    attributes = {
        line.split(": ", maxsplit=2)[0]: line.rsplit(": ", maxsplit=1)[-1]
        for line in result.stdout.splitlines()
    }
    missing_lf = [path for path in relative_paths if attributes.get(path) != "lf"]
    assert not missing_lf, (
        "container runtime files must have `eol=lf` in .gitattributes: "
        + ", ".join(missing_lf)
    )
