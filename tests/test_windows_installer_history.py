"""Committed installer source debt can shrink, but cannot be added or restored."""

from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path

import pytest

ORIGINAL = b"# original installer line\n" * 2003
SMALLER = b"# remaining installer line\n" * 2001
COMPLETE = b"# extracted entry point\n"
INITIAL = ({"install.ps1": ORIGINAL.count(b"\n")}, {"install.ps1": ORIGINAL})
SHRINK = ({"install.ps1": SMALLER.count(b"\n")}, {"install.ps1": SMALLER})
EMPTY = ({}, {"install.ps1": COMPLETE})
INCREASE = ({"install.ps1": 2004}, {"install.ps1": ORIGINAL + b"# added debt\n"})
NEW_SHARD = (
    {"install.ps1": 2003, "new.ps1": 2001},
    {"install.ps1": b"# original installer line\n" * 2002 + b"# @include new.ps1\n", "new.ps1": SMALLER},
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "-c", "core.hooksPath=nonexistent-fixture-hooks", *args],
        cwd=repo, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.strip()


def _commit(repo: Path, title: str) -> str:
    _git(repo, "add", "--all")
    _git(repo, "-c", "user.name=Installer Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", title)
    return _git(repo, "rev-parse", "HEAD")


def _state(repo: Path, ceilings: dict[str, int], sources: dict[str, bytes], builder) -> None:
    authoring = repo / "scripts/windows-installer"
    source = authoring / "source"
    source.mkdir(parents=True, exist_ok=True)
    for previous in source.glob("*.ps1"):
        previous.unlink()
    for name, data in sources.items():
        (source / name).write_bytes(data)
    manifest = {"version": 1, "entry": "install.ps1", "files": list(sources), "kill_track": ceilings}
    (authoring / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    # Each revision passes the local assembly gate. Only committed history can
    # reject the dishonest or regressive debt declarations below.
    (repo / "scripts/install.ps1").write_bytes(builder.assemble(authoring))


@pytest.mark.parametrize("states,base_exists,accepted", [
    ([INITIAL, SHRINK, EMPTY], True, True),
    ([INITIAL, INCREASE], True, False),
    ([INITIAL, NEW_SHARD], True, False),
    ([({"install.ps1": 2003}, {"install.ps1": ORIGINAL.replace(b"original", b"altered")})], True, False),
    ([({"install.ps1": 2004}, {"install.ps1": ORIGINAL})], True, False),
    ([INITIAL, EMPTY, INITIAL], True, False),
    ([INITIAL, INCREASE, SHRINK], True, False),
    ([INITIAL], False, False),
    ([INITIAL, SHRINK, EMPTY], "diverged", True),
], ids=[
    "mechanism-shrink-empty", "raised-ceiling", "new-oversized-shard", "dishonest-initial-source",
    "dishonest-initial-ceiling", "reintroduced-after-empty", "intermediate-increase-hidden-at-head", "missing-base-ref",
    "unrelated-upstream-progress",
])
def test_committed_history_never_increases_installer_source_debt(tmp_path: Path, states, base_exists, accepted):
    builder = importlib.import_module("scripts.build_windows_installer")
    _git(tmp_path, "init", "-b", "main")
    _git(tmp_path, "config", "core.autocrlf", "false")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/install.ps1").write_bytes(ORIGINAL)
    base = _commit(tmp_path, "Original standalone installer")
    for index, (ceilings, sources) in enumerate(states):
        _state(tmp_path, ceilings, sources, builder)
        _commit(tmp_path, f"Installer source graph revision {index}")
    head = _git(tmp_path, "rev-parse", "HEAD")
    requested_base = base if base_exists else "fixture-missing-base"
    if base_exists == "diverged":
        _git(tmp_path, "checkout", "-b", "upstream", base)
        (tmp_path / "unrelated.txt").write_text("Upstream work outside the installer\n")
        requested_base = _commit(tmp_path, "Unrelated upstream progress")
    if accepted:
        assert builder.verify_history(requested_base, head, tmp_path) is None
    else:
        with pytest.raises(builder.AssemblyError):
            builder.verify_history(requested_base, head, tmp_path)
    assert _git(tmp_path, "status", "--porcelain") == "", "history verification must not mutate its checkout"


@pytest.mark.parametrize("retired_first,restore_allowance", [(True, False), (True, True), (False, True)])
def test_merge_resolution_cannot_restore_an_allowance_from_either_parent(tmp_path: Path, retired_first, restore_allowance):
    builder = importlib.import_module("scripts.build_windows_installer")
    _git(tmp_path, "init", "-b", "main")
    _git(tmp_path, "config", "core.autocrlf", "false")
    _git(tmp_path, "config", "user.name", "Installer Fixture")
    _git(tmp_path, "config", "user.email", "fixture@example.invalid")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/install.ps1").write_bytes(ORIGINAL)
    base = _commit(tmp_path, "Original installer")
    _state(tmp_path, *INITIAL, builder)
    _commit(tmp_path, "Initial source ownership")
    _git(tmp_path, "branch", "other")
    _state(tmp_path, *EMPTY, builder)
    _commit(tmp_path, "Retire source debt")
    _git(tmp_path, "checkout", "other")
    (tmp_path / "unrelated.txt").write_text("Diverged branch\n")
    _commit(tmp_path, "Unrelated work")
    if retired_first:
        _git(tmp_path, "checkout", "main")
    _git(tmp_path, "merge", "--no-commit", "--no-ff", "other" if retired_first else "main")
    _state(tmp_path, *(INITIAL if restore_allowance else EMPTY), builder)
    head = _commit(tmp_path, "Resolve source graph in a real merge")
    assert len(_git(tmp_path, "rev-list", "--parents", "-n", "1", head).split()) == 3
    if restore_allowance:
        with pytest.raises(builder.AssemblyError):
            builder.verify_history(base, head, tmp_path)
    else:
        assert builder.verify_history(base, head, tmp_path) is None
