"""Tests for tools/self_repo_guard.py — the running-source-checkout git guard."""

from pathlib import Path

import pytest

from tools.self_repo_guard import (
    detect_self_repo_git_mutation,
    get_running_source_root,
)


@pytest.fixture
def repo(tmp_path):
    """A fake source checkout acting as the running install's repo root."""
    root = tmp_path / "hermes-agent"
    (root / ".git").mkdir(parents=True)
    (root / "agent").mkdir()
    return root.resolve()


def _detect(command, cwd, root):
    return detect_self_repo_git_mutation(command, str(cwd), source_root=root)


class TestBlocksMutationsInSourceRepo:
    @pytest.mark.parametrize("sub", [
        "checkout pr-51020",
        "switch main",
        "reset --hard origin/main",
        "rebase origin/main",
        "merge origin/main",
        "pull",
        "restore .",
        "stash",
        "stash pop",
        "clean -fd",
        "cherry-pick abc123",
        "revert HEAD",
    ])
    def test_cwd_inside_repo(self, repo, sub):
        hit, msg = _detect(f"git {sub}", repo, repo)
        assert hit is True
        assert str(repo) in msg

    def test_cwd_in_repo_subdirectory(self, repo):
        hit, _ = _detect("git checkout main", repo / "agent", repo)
        assert hit is True

    def test_dash_c_targeting_repo_from_outside(self, repo, tmp_path):
        hit, _ = _detect(f"git -C {repo} checkout pr-51020", tmp_path, repo)
        assert hit is True

    def test_cd_into_repo_then_checkout(self, repo, tmp_path):
        hit, _ = _detect(f"cd {repo} && git checkout pr-51020", tmp_path, repo)
        assert hit is True

    def test_relative_cd_into_repo(self, repo):
        hit, _ = _detect("cd hermes-agent && git pull", repo.parent, repo)
        assert hit is True

    def test_mutation_after_safe_command(self, repo):
        hit, _ = _detect("git status; git reset --hard HEAD~1", repo, repo)
        assert hit is True

    def test_wrapped_in_sudo_env(self, repo):
        hit, _ = _detect("sudo env GIT_PAGER=cat git checkout main", repo, repo)
        assert hit is True

    def test_tilde_dash_c_path(self, repo, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(repo.parent))
        hit, _ = _detect("git -C ~/hermes-agent checkout main", tmp_path, repo)
        assert hit is True


class TestAllowsSafeCommands:
    @pytest.mark.parametrize("cmd", [
        "git status",
        "git log --oneline -5",
        "git diff main...HEAD",
        "git branch --show-current",
        "git stash list",
        "git stash show -p",
        "git commit -m 'msg'",
        "git add -A",
        "git fetch origin main",
        "git worktree add /tmp/wt feature-branch",
        "git push fork feature-branch",
        "ls -la",
        "grep -rn checkout tools/",
    ])
    def test_read_only_and_dev_loop_in_repo(self, repo, cmd):
        hit, _ = _detect(cmd, repo, repo)
        assert hit is False

    def test_mutation_in_other_repo(self, repo, tmp_path):
        other = tmp_path / "other-project"
        other.mkdir()
        hit, _ = _detect("git checkout main", other, repo)
        assert hit is False

    def test_dash_c_redirects_out_of_repo(self, repo, tmp_path):
        hit, _ = _detect(f"git -C {tmp_path} checkout main", repo, repo)
        assert hit is False

    def test_cd_out_of_repo_then_checkout(self, repo, tmp_path):
        hit, _ = _detect(f"cd {tmp_path} && git checkout main", repo, repo)
        assert hit is False

    def test_mentioning_repo_path_without_targeting_it(self, repo, tmp_path):
        hit, _ = _detect(f"echo {repo} && git checkout main", tmp_path, repo)
        assert hit is False

    def test_checkout_as_grep_pattern_not_git(self, repo):
        hit, _ = _detect("grep checkout file.txt", repo, repo)
        assert hit is False

    def test_empty_command(self, repo):
        hit, _ = _detect("", repo, repo)
        assert hit is False

    def test_packaged_install_is_inert(self, monkeypatch, tmp_path):
        import tools.self_repo_guard as mod
        monkeypatch.setattr(mod, "get_running_source_root", lambda: None)
        hit, msg = mod.detect_self_repo_git_mutation("git checkout main", str(tmp_path))
        assert hit is False
        assert msg is None


class TestSourceRootResolution:
    def test_resolves_to_repo_when_git_dir_present(self):
        # The test suite itself runs from a source checkout, so the resolver
        # must find a root whose .git exists.
        root = get_running_source_root()
        if root is not None:
            assert (root / ".git").exists()

    def test_worktree_git_file_counts(self, tmp_path, monkeypatch):
        import tools.self_repo_guard as mod
        root = tmp_path / "wt"
        root.mkdir()
        (root / ".git").write_text("gitdir: /somewhere/.git/worktrees/wt\n")
        (root / "tools").mkdir()
        fake_file = root / "tools" / "self_repo_guard.py"
        fake_file.write_text("")
        monkeypatch.setattr(mod, "__file__", str(fake_file))
        assert mod.get_running_source_root() == root.resolve()


class TestUnparseableCommands:
    def test_unbalanced_quotes_fall_back(self, repo):
        hit, _ = _detect('git checkout "unterminated', repo, repo)
        assert hit is True

    def test_subshell_syntax_does_not_crash(self, repo):
        hit, _ = _detect("VAL=$(git rev-parse HEAD) git checkout main", repo, repo)
        assert hit is True
