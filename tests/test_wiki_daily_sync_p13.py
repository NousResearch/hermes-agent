"""P13 isolation proof for scripts/wiki_daily_sync.sh.

Verifies the dry-run flag:
- WIKI_SYNC_DRY_RUN=1 prints the planned commands, exits 0, never runs git
- WIKI_SYNC_DIR can be redirected so the production wiki tree is untouched
- a missing WIKI_SYNC_DIR is reported with exit 1
"""
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "scripts" / "wiki_daily_sync.sh"


def _run(env_extra: dict | None = None):
    env = dict(os.environ)
    env.pop("WIKI_SYNC_DRY_RUN", None)
    env.pop("WIKI_SYNC_DIR", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        ["bash", str(WRAPPER)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
        timeout=15,
    )


def test_dry_run_exits_zero_without_git(tmp_path):
    """WIKI_SYNC_DRY_RUN=1 prints the plan and never touches git."""
    fake_wiki = tmp_path / "wiki"
    fake_wiki.mkdir()
    r = _run({"WIKI_SYNC_DRY_RUN": "1", "WIKI_SYNC_DIR": str(fake_wiki)})
    assert r.returncode == 0, r.stderr
    assert "dry-run:" in r.stdout
    # Nothing mutated in the fake tree (no .git operations).
    assert not (fake_wiki / ".git").exists()


def test_dry_run_works_even_when_dir_missing(tmp_path):
    """Dry-run must short-circuit BEFORE the cd guard so a missing dir
    does not cause exit 1 (the point of dry-run is to plan without
    requiring the workspace to exist)."""
    missing = tmp_path / "does-not-exist"
    r = _run({"WIKI_SYNC_DRY_RUN": "1", "WIKI_SYNC_DIR": str(missing)})
    assert r.returncode == 0, r.stderr
    assert "dry-run:" in r.stdout


def test_missing_dir_without_dry_run_exits_nonzero(tmp_path):
    """Without dry-run, a missing wiki dir must report an error and exit 1."""
    missing = tmp_path / "does-not-exist"
    r = _run({"WIKI_SYNC_DIR": str(missing)})
    assert r.returncode == 1
    assert "cannot cd" in r.stderr or "cannot cd" in r.stdout


def _git(path: Path, *args: str):
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def test_refuses_unexpected_remote_before_any_push(tmp_path):
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    _git(wiki, "init", "-b", "main")
    _git(wiki, "remote", "add", "origin", str(tmp_path / "checked-out-staging"))

    r = _run({"WIKI_SYNC_DIR": str(wiki), "GH_TOKEN": "fixture-token"})

    assert r.returncode == 1
    assert "refusing unsafe wiki origin" in r.stdout


def test_refuses_diverged_checkpoint_without_mutating_repo(tmp_path):
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    wiki = tmp_path / "wiki"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    subprocess.run(["git", "clone", str(remote), str(seed)], check=True, capture_output=True)
    _git(seed, "checkout", "-b", "main")
    _git(seed, "config", "user.email", "fixture@example.test")
    _git(seed, "config", "user.name", "Fixture")
    (seed / "base.md").write_text("base\n")
    _git(seed, "add", "base.md")
    _git(seed, "commit", "-m", "base")
    _git(seed, "push", "-u", "origin", "main")
    subprocess.run(["git", "clone", "--branch", "main", str(remote), str(wiki)], check=True, capture_output=True)
    _git(wiki, "config", "user.email", "fixture@example.test")
    _git(wiki, "config", "user.name", "Fixture")
    (wiki / "local.md").write_text("local\n")
    _git(wiki, "add", "local.md")
    _git(wiki, "commit", "-m", "local")
    (seed / "remote.md").write_text("remote\n")
    _git(seed, "add", "remote.md")
    _git(seed, "commit", "-m", "remote")
    _git(seed, "push", "origin", "main")
    before = _git(wiki, "rev-parse", "HEAD").stdout.strip()

    r = _run({
        "WIKI_SYNC_DIR": str(wiki),
        "WIKI_SYNC_EXPECTED_REMOTE": str(remote),
        "GH_TOKEN": "fixture-token",
    })

    assert r.returncode == 1
    assert "branch diverged" in r.stdout
    assert _git(wiki, "rev-parse", "HEAD").stdout.strip() == before
    assert not (wiki / "remote.md").exists()
