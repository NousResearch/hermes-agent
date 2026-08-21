"""Tests for tools/skills_repo_sync.py — git-repo-backed skills.

Uses REAL bare git remotes (the GitHub-like case) so the clone/pull/push
paths are exercised end-to-end, not mocked.
"""

import shutil
import subprocess
from pathlib import Path

from tools.skills_repo_sync import (
    _repo_slug,
    external_repo_enabled,
    get_checkout_dir,
    get_repo_skills_dir,
    get_repo_write_dir,
    maybe_push_external_repo,
    sync_external_repo,
)


def _rmtree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


# ── Helpers ────────────────────────────────────────────────────────────────


def _write_config(hermes_home: Path, *, enabled: bool = True,
                  url: str = "", branch: str = "", path: str = "") -> None:
    """Write a config.yaml under hermes_home with skills.external_repo set."""
    hermes_home.mkdir(parents=True, exist_ok=True)
    branch_line = f"    branch: {branch}\n" if branch else ""
    path_line = f"    path: {path}\n" if path else ""
    hermes_home.joinpath("config.yaml").write_text(
        f"""skills:
  external_repo:
    enabled: {str(enabled).lower()}
    url: "{url}"
{branch_line}{path_line}
""".strip(),
        encoding="utf-8",
    )


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True)


def _init_bare(root: Path) -> Path:
    """Create a bare remote repo (the GitHub-like case).

    Pins the bare HEAD to ``refs/heads/main`` so clones follow the ``main``
    branch — ``git init --bare`` defaults HEAD to the (phantom) ``master``,
    which makes clones warn and the test suite flaky.
    """
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "--bare", "-q", str(root)], check=True)
    subprocess.run(
        ["git", "-C", str(root), "symbolic-ref", "HEAD", "refs/heads/main"],
        check=True,
    )
    return root


def _seed_bare(bare: Path, *skills: str, subdir: str = "category") -> None:
    """Push skills into a bare remote from a throwaway worktree clone."""
    work = bare.parent / ("seed-" + bare.name)
    if work.exists():
        _rmtree(work)
    subprocess.run(["git", "clone", "-q", str(bare), str(work)], check=True)
    _git(work, "config", "user.email", "t@t")
    _git(work, "config", "user.name", "t")
    # The very first seed on an empty bare repo has no branch checked out;
    # create + pin `main` so the push lands on the right ref.
    head = subprocess.run(
        ["git", "-C", str(work), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()
    if head == "HEAD":
        _git(work, "checkout", "-q", "-b", "main")
    for name in skills:
        skill_dir = work / subdir / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_dir.joinpath("SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Test skill {name}\n---\n\n# {name}\n",
            encoding="utf-8",
        )
    _git(work, "add", "-A")
    if subprocess.run(["git", "-C", str(work), "diff", "--cached", "--quiet"],
                      check=False).returncode != 0:
        _git(work, "commit", "-qm", "seed skills")
    _git(work, "push", "-qu", "origin", "main")


def _worktree(bare: Path) -> Path:
    """A fresh clone of the bare remote, used to inspect pushed content."""
    work = bare.parent / ("check-" + bare.name)
    subprocess.run(["git", "clone", "-q", str(bare), str(work)], check=True)
    return work


def _head_sha(bare: Path) -> str:
    """Current HEAD sha of the bare remote's default branch."""
    return subprocess.run(
        ["git", "-C", str(bare), "rev-parse", "main"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def _setup_env(monkeypatch, tmp_path: Path) -> Path:
    """Point HERMES_HOME at a fresh temp dir and clear caches."""
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    from agent import skill_utils

    skill_utils._external_dirs_cache_clear()
    getattr(skill_utils, "_raw_config_cache_clear", lambda: None)()
    return hermes_home


# ── Slug ───────────────────────────────────────────────────────────────────


class TestRepoSlug:
    def test_https_and_ssh_slug_to_owner_repo(self):
        assert _repo_slug("https://github.com/owner/skills.git") == "owner-skills"
        assert _repo_slug("git@github.com:owner/skills.git") == "owner-skills"

    def test_same_short_name_different_owners_do_not_collide(self):
        assert _repo_slug("https://github.com/alice/skills.git") == "alice-skills"
        assert _repo_slug("https://github.com/bob/skills.git") == "bob-skills"

    def test_unsafe_urls_fall_back_to_hash(self):
        slug = _repo_slug("https://example.com/a b/c!.git")
        assert slug.startswith("repo-")


# ── Config resolution ──────────────────────────────────────────────────────


class TestConfigResolution:
    def test_disabled_returns_none_via_enabled_check(self, tmp_path, monkeypatch):
        hermes_home = _setup_env(monkeypatch, tmp_path)
        _write_config(hermes_home, enabled=False, url="https://x/y.git")
        assert external_repo_enabled() is False
        assert get_repo_write_dir() is None

    def test_enabled_without_url_is_not_active(self, tmp_path, monkeypatch):
        hermes_home = _setup_env(monkeypatch, tmp_path)
        _write_config(hermes_home, enabled=True, url="")
        assert external_repo_enabled() is False


# ── Sync (pull path) ───────────────────────────────────────────────────────


class TestSyncExternalRepo:
    def test_disabled_returns_none(self, tmp_path, monkeypatch):
        hermes_home = _setup_env(monkeypatch, tmp_path)
        _write_config(hermes_home, enabled=False, url="https://x/y.git")
        assert sync_external_repo() is None

    def test_clones_bare_repo_on_first_run(self, tmp_path, monkeypatch):
        hermes_home = _setup_env(monkeypatch, tmp_path)
        bare = _init_bare(tmp_path / "skills-repo.git")
        _seed_bare(bare, "alpha")
        _write_config(hermes_home, enabled=True, url=str(bare))

        result = sync_external_repo()
        assert result is not None
        assert (result / "category" / "alpha" / "SKILL.md").exists()
        assert result == get_checkout_dir(str(bare))

    def test_pull_brings_in_new_commits(self, tmp_path, monkeypatch):
        hermes_home = _setup_env(monkeypatch, tmp_path)
        bare = _init_bare(tmp_path / "skills-repo.git")
        _seed_bare(bare, "alpha")
        _write_config(hermes_home, enabled=True, url=str(bare))

        assert sync_external_repo() is not None
        # Another machine adds beta upstream.
        _seed_bare(bare, "beta")

        skills_dir = sync_external_repo()
        assert skills_dir is not None
        assert (skills_dir / "category" / "beta" / "SKILL.md").exists()

    def test_unreachable_url_returns_none(self, tmp_path, monkeypatch):
        hermes_home = _setup_env(monkeypatch, tmp_path)
        _write_config(hermes_home, enabled=True, url=str(tmp_path / "does-not-exist"))
        assert sync_external_repo() is None

    def test_subdir_path_is_resolved(self, tmp_path, monkeypatch):
        hermes_home = _setup_env(monkeypatch, tmp_path)
        bare = _init_bare(tmp_path / "skills-repo.git")
        _seed_bare(bare, "my-skill", subdir="skills")

        _write_config(hermes_home, enabled=True, url=str(bare), path="skills")
        skills_dir = sync_external_repo()
        assert skills_dir is not None
        assert skills_dir.name == "skills"
        assert (skills_dir / "my-skill" / "SKILL.md").exists()


# ── Push-back (write path) ─────────────────────────────────────────────────


class TestWriteBack:
    def test_new_skill_is_created_inside_checkout(self, tmp_path, monkeypatch):
        hermes_home = _setup_env(monkeypatch, tmp_path)
        bare = _init_bare(tmp_path / "skills-repo.git")
        _write_config(hermes_home, enabled=True, url=str(bare))
        assert sync_external_repo() is not None

        write_dir = get_repo_write_dir()
        assert write_dir is not None
        assert str(write_dir).startswith(str(get_checkout_dir(str(bare))))

        # simulate a skill_manage create landing in the repo
        skill_md = write_dir / "my-new-skill" / "SKILL.md"
        skill_md.parent.mkdir(parents=True, exist_ok=True)
        skill_md.write_text(
            "---\nname: my-new-skill\ndescription: New\n---\n\n# New\n",
            encoding="utf-8",
        )

        assert maybe_push_external_repo(message="write my-new-skill") is True
        # The change actually reached the remote.
        check = _worktree(bare)
        assert (check / "my-new-skill" / "SKILL.md").exists()

    def test_edit_of_repo_skill_is_pushed_back(self, tmp_path, monkeypatch):
        hermes_home = _setup_env(monkeypatch, tmp_path)
        bare = _init_bare(tmp_path / "skills-repo.git")
        _seed_bare(bare, "alpha")
        _write_config(hermes_home, enabled=True, url=str(bare))
        assert sync_external_repo() is not None

        before = _head_sha(bare)

        # Edit the skill inside the checkout (as skill_manage does on a repo
        # skill), then push.
        checkout_skill = get_checkout_dir(str(bare)) / "category" / "alpha" / "SKILL.md"
        checkout_skill.write_text(
            "---\nname: alpha\ndescription: Updated\n---\n\n# alpha v2\n",
            encoding="utf-8",
        )

        assert maybe_push_external_repo(message="write alpha") is True
        check = _worktree(bare)
        assert "Updated" in (check / "category" / "alpha" / "SKILL.md").read_text()
        assert _head_sha(bare) != before

    def test_no_changes_does_not_push(self, tmp_path, monkeypatch):
        hermes_home = _setup_env(monkeypatch, tmp_path)
        bare = _init_bare(tmp_path / "skills-repo.git")
        _seed_bare(bare, "alpha")
        _write_config(hermes_home, enabled=True, url=str(bare))
        assert sync_external_repo() is not None
        before = _head_sha(bare)

        assert maybe_push_external_repo(message="noop") is False
        assert _head_sha(bare) == before

    def test_remote_moved_reconciles_and_pushes(self, tmp_path, monkeypatch):
        """The wedge case: local commit + another install pushed meanwhile.

        A bare push would fail (non-fast-forward) and the next pull --ff-only
        would refuse too, locking the checkout forever.  The push path must
        rebase the local commit onto the remote and still deliver it.
        """
        hermes_home = _setup_env(monkeypatch, tmp_path)
        bare = _init_bare(tmp_path / "skills-repo.git")
        _seed_bare(bare, "alpha")
        _write_config(hermes_home, enabled=True, url=str(bare))
        assert sync_external_repo() is not None

        # This machine edits alpha and commits locally (no push yet).
        checkout = get_checkout_dir(str(bare))
        checkout_skill = checkout / "category" / "alpha" / "SKILL.md"
        checkout_skill.write_text(
            "---\nname: alpha\ndescription: Local edit\n---\n\n# alpha local\n",
            encoding="utf-8",
        )
        _git(checkout, "add", "-A")
        _git(
            checkout,
            "-c", "user.name=Hermes Agent",
            "-c", "user.email=hermes@nousresearch.com",
            "commit", "-qm", "hermes: local edit",
        )

        # Another machine adds beta upstream (the remote moved).
        _seed_bare(bare, "beta")

        assert maybe_push_external_repo(message="write alpha") is True
        check = _worktree(bare)
        assert (check / "category" / "beta" / "SKILL.md").exists()
        assert "Local edit" in (check / "category" / "alpha" / "SKILL.md").read_text()

    def test_conflicting_remote_edit_aborts_cleanly(self, tmp_path, monkeypatch):
        """A genuine conflict must not wedge: abort, keep local work, return False."""
        hermes_home = _setup_env(monkeypatch, tmp_path)
        bare = _init_bare(tmp_path / "skills-repo.git")
        _seed_bare(bare, "alpha")
        _write_config(hermes_home, enabled=True, url=str(bare))
        assert sync_external_repo() is not None

        checkout = get_checkout_dir(str(bare))
        checkout_skill = checkout / "category" / "alpha" / "SKILL.md"
        checkout_skill.write_text(
            "---\nname: alpha\ndescription: Local edit\n---\n\n# alpha local\n",
            encoding="utf-8",
        )
        _git(checkout, "add", "-A")
        _git(
            checkout,
            "-c", "user.name=Hermes Agent",
            "-c", "user.email=hermes@nousresearch.com",
            "commit", "-qm", "hermes: local edit",
        )

        # Another machine edits the SAME file differently upstream.
        seed_work = bare.parent / ("seed2-" + bare.name)
        _rmtree(seed_work)
        subprocess.run(["git", "clone", "-q", str(bare), str(seed_work)], check=True)
        _git(seed_work, "config", "user.email", "t@t")
        _git(seed_work, "config", "user.name", "t")
        (seed_work / "category" / "alpha" / "SKILL.md").write_text(
            "---\nname: alpha\ndescription: Remote edit\n---\n\n# alpha remote\n",
            encoding="utf-8",
        )
        _git(seed_work, "add", "-A")
        _git(seed_work, "commit", "-qm", "remote edit")
        _git(seed_work, "push", "-qu", "origin", "main")

        assert maybe_push_external_repo(message="write alpha") is False
        # Local work survives and no half-finished rebase is left behind.
        assert "Local edit" in checkout_skill.read_text()
        assert not (checkout / ".git" / "rebase-merge").exists()
        assert not (checkout / ".git" / "rebase-apply").exists()

    def test_disabled_returns_false(self, tmp_path, monkeypatch):
        hermes_home = _setup_env(monkeypatch, tmp_path)
        _write_config(hermes_home, enabled=False, url="https://x/y.git")
        assert maybe_push_external_repo() is False


# ── Discovery integration ──────────────────────────────────────────────────


class TestExternalDirsIntegration:
    def test_synced_checkout_shows_up_in_external_dirs(self, tmp_path, monkeypatch):
        hermes_home = _setup_env(monkeypatch, tmp_path)
        bare = _init_bare(tmp_path / "skills-repo.git")
        _seed_bare(bare, "alpha")
        _write_config(hermes_home, enabled=True, url=str(bare))

        from agent.skill_utils import get_external_skills_dirs

        assert get_external_skills_dirs() == []
        sync_external_repo()
        _setup_env(monkeypatch, tmp_path)  # clear caches again after sync

        dirs = get_external_skills_dirs()
        assert get_checkout_dir(str(bare)) in dirs

    def test_disabled_repo_not_listed(self, tmp_path, monkeypatch):
        hermes_home = _setup_env(monkeypatch, tmp_path)
        bare = _init_bare(tmp_path / "skills-repo.git")
        _seed_bare(bare, "alpha")
        _write_config(hermes_home, enabled=False, url=str(bare))

        from agent.skill_utils import get_external_skills_dirs

        assert get_external_skills_dirs() == []