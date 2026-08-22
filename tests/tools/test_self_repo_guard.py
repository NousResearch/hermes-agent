"""Tests for tools/self_repo_guard.py — the running-source-checkout git guard."""

import os
import shlex
import subprocess
from pathlib import Path

import pytest

from tools.self_repo_guard import (
    detect_self_repo_git_mutation,
    get_running_source_root,
)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "hermes-agent"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "agent").mkdir()
    return root.resolve()


def _detect(command, cwd, root):
    return detect_self_repo_git_mutation(command, str(cwd), source_root=root)


def _shell_path(path: Path) -> str:
    """Render a native path as one shell-safe command argument."""
    return shlex.quote(path.as_posix())


def _msys_path(path: Path) -> str:
    """Render a resolved Windows path using Git Bash's drive-root spelling."""
    resolved = path.resolve()
    return f"/{resolved.drive[0].lower()}{resolved.as_posix()[2:]}"


class TestBlocksMutationsInSourceRepo:
    @pytest.mark.parametrize(
        "sub",
        [
            "checkout pr-51020",
            "switch main",
            "bisect start",
            "bisect good HEAD~10",
            "reset --hard origin/main",
            "reset --har origin/main",
            "rebase origin/main",
            "merge origin/main",
            "pull",
            "restore .",
            "stash",
            "stash pop",
            "clean -fd",
            "cherry-pick abc123",
            "revert HEAD",
        ],
    )
    def test_cwd_inside_repo(self, repo, sub):
        hit, msg = _detect(f"git {sub}", repo, repo)
        assert hit is True
        assert str(repo) in msg

    def test_cwd_in_repo_subdirectory(self, repo):
        hit, _ = _detect("git checkout main", repo / "agent", repo)
        assert hit is True

    def test_dash_c_targeting_repo_from_outside(self, repo, tmp_path):
        hit, _ = _detect(f"git -C {_shell_path(repo)} checkout pr-51020", tmp_path, repo)
        assert hit is True

    def test_cd_into_repo_then_checkout(self, repo, tmp_path):
        hit, _ = _detect(f"cd {_shell_path(repo)} && git checkout pr-51020", tmp_path, repo)
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

    @pytest.mark.parametrize(
        "command",
        [
            "sudo -u root git checkout main",
            "env -u GIT_PAGER git switch main",
            "/usr/bin/git checkout main",
            "sh -c 'git checkout main'",
            "bash -lc 'git switch main'",
            "bash -o pipefail -c 'git checkout main'",
            "bash +O extglob -c 'git checkout main'",
            "zsh -yc 'git checkout main'",
            "dash -Vc 'git checkout main'",
            "ksh -Gc 'git checkout main'",
        ],
    )
    def test_wrappers_and_nested_shells(self, repo, command):
        hit, _ = _detect(command, repo, repo)
        assert hit is True

    @pytest.mark.parametrize(
        "command",
        [
            "gh pr checkout 51020",
            "hub pr checkout 51020",
        ],
    )
    def test_pr_checkout_clients(self, repo, command):
        hit, _ = _detect(command, repo, repo)
        assert hit is True

    def test_explicit_work_tree_targeting_repo(self, repo, tmp_path):
        command = (
            f"git --git-dir={_shell_path(repo / '.git')} "
            f"--work-tree={_shell_path(repo)} checkout main"
        )
        hit, _ = _detect(command, tmp_path, repo)
        assert hit is True

    def test_git_environment_targeting_repo(self, repo, tmp_path):
        command = (
            f"GIT_DIR={_shell_path(repo / '.git')} "
            f"GIT_WORK_TREE={_shell_path(repo)} git checkout main"
        )
        hit, _ = _detect(command, tmp_path, repo)
        assert hit is True

    def test_inline_git_alias(self, repo):
        hit, _ = _detect("git -c alias.co=checkout co main", repo, repo)
        assert hit is True

    def test_configured_git_alias(self, repo):
        subprocess.run(
            ["git", "-C", str(repo), "config", "alias.co", "checkout"],
            check=True,
        )
        hit, _ = _detect("git co main", repo, repo)
        assert hit is True

    def test_mutation_in_command_substitution(self, repo):
        hit, _ = _detect('echo "$(git checkout main)"', repo, repo)
        assert hit is True

    @pytest.mark.parametrize(
        "command",
        [
            'echo "$(echo ready && git checkout main)"',
            "echo `git checkout main`",
            'echo "`git checkout main`"',
        ],
    )
    def test_nested_command_lists(self, repo, command):
        hit, _ = _detect(command, repo, repo)
        assert hit is True

    def test_shell_heredoc_is_executed(self, repo):
        command = "bash <<'EOF'\ngit checkout main\nEOF\n"
        hit, _ = _detect(command, repo, repo)
        assert hit is True

    def test_tilde_dash_c_path(self, repo, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(repo.parent))
        monkeypatch.setenv("USERPROFILE", str(repo.parent))
        hit, _ = _detect("git -C ~/hermes-agent checkout main", tmp_path, repo)
        assert hit is True


class TestAllowsSafeCommands:
    @pytest.mark.parametrize(
        "cmd",
        [
            "git status",
            "git log --oneline -5",
            "git diff main...HEAD",
            "git branch --show-current",
            "git stash list",
            "git stash show -p",
            "git stash create",
            "git stash store abc123",
            "git stash drop",
            "git stash clear",
            "git reset --soft HEAD~1",
            "git reset --mixed HEAD~1",
            "git restore --staged pyproject.toml",
            "git clean --dry-run -fd",
            "git clean -nd",
            "git commit -m 'msg'",
            "git add -A",
            "git fetch origin main",
            "git worktree add /tmp/wt feature-branch",
            "git push fork feature-branch",
            "ls -la",
            "grep -rn checkout tools/",
        ],
    )
    def test_read_only_and_dev_loop_in_repo(self, repo, cmd):
        hit, _ = _detect(cmd, repo, repo)
        assert hit is False

    def test_mutation_in_other_repo(self, repo, tmp_path):
        other = tmp_path / "other-project"
        other.mkdir()
        hit, _ = _detect("git checkout main", other, repo)
        assert hit is False

    def test_dash_c_redirects_out_of_repo(self, repo, tmp_path):
        hit, _ = _detect(f"git -C {_shell_path(tmp_path)} checkout main", repo, repo)
        assert hit is False

    def test_cd_out_of_repo_then_checkout(self, repo, tmp_path):
        hit, _ = _detect(f"cd {_shell_path(tmp_path)} && git checkout main", repo, repo)
        assert hit is False

    def test_mentioning_repo_path_without_targeting_it(self, repo, tmp_path):
        hit, _ = _detect(f"echo {repo} && git checkout main", tmp_path, repo)
        assert hit is False

    def test_checkout_as_grep_pattern_not_git(self, repo):
        hit, _ = _detect("grep checkout file.txt", repo, repo)
        assert hit is False

    def test_pr_checkout_words_in_other_gh_command_are_safe(self, repo):
        hit, _ = _detect("gh api /repos/example/pr/checkout", repo, repo)
        assert hit is False

    @pytest.mark.parametrize(
        "command",
        [
            'echo "safe | git checkout main"',
            "echo '$(git checkout main)'",
            "printf '%s\\n' 'git checkout main'",
        ],
    )
    def test_quoted_git_text_is_not_executed(self, repo, command):
        hit, _ = _detect(command, repo, repo)
        assert hit is False

    @pytest.mark.parametrize(
        "command",
        [
            "cat > script.sh <<'EOF'\ngit checkout main\nEOF\n",
            "python - <<'PY'\nprint('git checkout main')\nPY\n",
        ],
    )
    def test_data_heredoc_is_not_executed_as_shell(self, repo, command):
        hit, _ = _detect(command, repo, repo)
        assert hit is False

    def test_subshell_cd_does_not_leak(self, repo):
        command = f"(cd {repo} && git status); git checkout main"
        hit, _ = _detect(command, repo.parent, repo)
        assert hit is False

    def test_pipeline_cd_does_not_leak(self, repo):
        command = f"cd {repo} | cat; git checkout main"
        hit, _ = _detect(command, repo.parent, repo)
        assert hit is False

    def test_successful_cd_or_branch_does_not_run(self, repo):
        command = f"cd {repo} || git checkout main"
        hit, _ = _detect(command, repo.parent, repo)
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


class TestWorktreeTargetingSourceRoot:
    @pytest.mark.parametrize(
        "sub",
        [
            "remove .",
            "remove -f .",
            "remove --force .",
            "remove -- .",
            "move . {other}",
            "move -f . {other}",
        ],
    )
    def test_blocks_relative_target_from_inside(self, repo, tmp_path, sub):
        command = f"git worktree {sub.format(other=tmp_path / 'moved')}"
        hit, msg = _detect(command, repo, repo)
        assert hit is True
        assert str(repo) in msg

    @pytest.mark.parametrize("action", ["remove", "remove -f", "remove --force"])
    def test_blocks_absolute_target_from_outside(self, repo, tmp_path, action):
        hit, _ = _detect(
            f"git worktree {action} {_shell_path(repo)}", tmp_path, repo
        )
        assert hit is True

    def test_blocks_move_of_root_from_outside(self, repo, tmp_path):
        command = (
            f"git worktree move {_shell_path(repo)} "
            f"{_shell_path(tmp_path / 'moved')}"
        )
        hit, _ = _detect(command, tmp_path, repo)
        assert hit is True

    def test_blocks_dash_c_worktree_remove(self, repo, tmp_path):
        hit, _ = _detect(
            f"git -C {_shell_path(tmp_path)} worktree remove {_shell_path(repo)}",
            tmp_path,
            repo,
        )
        assert hit is True

    def test_blocks_parent_relative_target_from_subdirectory(self, repo):
        hit, _ = _detect("git worktree remove ..", repo / "agent", repo)
        assert hit is True

    def test_blocks_sibling_relative_target(self, repo):
        hit, _ = _detect(f"git worktree remove ../{repo.name}", repo, repo)
        assert hit is True

    @pytest.mark.parametrize(
        "sub",
        [
            "add {other}",
            "add -b feature {other}",
            "list",
            "list --porcelain",
            "prune",
            "lock {other}",
            "unlock {other}",
            "remove {other}",
            "move {other} {other}-dest",
        ],
    )
    def test_allows_other_worktrees_and_add(self, repo, tmp_path, sub):
        command = f"git worktree {sub.format(other=tmp_path / 'other-wt')}"
        hit, _ = _detect(command, repo, repo)
        assert hit is False

    @pytest.mark.parametrize("sub", ["", "remove", "move", "-f"])
    def test_incomplete_worktree_command_is_not_blocked(self, repo, sub):
        hit, _ = _detect(f"git worktree {sub}".strip(), repo, repo)
        assert hit is False


@pytest.mark.skipif(os.name != "nt", reason="Git Bash drive paths are Windows-only")
class TestWindowsMsysPathResolution:
    def test_dash_c_targeting_repo(self, repo, tmp_path):
        command = f"git -C {_shell_path(Path(_msys_path(repo)))} checkout main"
        hit, _ = _detect(command, tmp_path, repo)
        assert hit is True

    def test_cd_targeting_repo(self, repo, tmp_path):
        command = f"cd {_shell_path(Path(_msys_path(repo)))} && git checkout main"
        hit, _ = _detect(command, tmp_path, repo)
        assert hit is True

    def test_dash_c_targeting_repo_with_spaces(self, tmp_path):
        repo = tmp_path / "hermes agent"
        repo.mkdir()
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        repo = repo.resolve()
        command = f"git -C {_shell_path(Path(_msys_path(repo)))} checkout main"
        hit, _ = _detect(command, tmp_path, repo)
        assert hit is True

    def test_dash_c_redirecting_outside_repo(self, repo, tmp_path):
        command = f"git -C {_shell_path(Path(_msys_path(tmp_path)))} checkout main"
        hit, _ = _detect(command, repo, repo)
        assert hit is False

    def test_worktree_remove_targeting_repo(self, repo, tmp_path):
        command = f"git worktree remove {_shell_path(Path(_msys_path(repo)))}"
        hit, _ = _detect(command, tmp_path, repo)
        assert hit is True

    def test_tilde_uses_git_bash_home(self, repo, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(repo.parent))
        monkeypatch.setenv("USERPROFILE", str(tmp_path / "different-profile"))
        hit, _ = _detect("git -C ~/hermes-agent checkout main", tmp_path, repo)
        assert hit is True


@pytest.mark.skipif(os.name != "nt", reason="Native backslash paths are Windows-only")
class TestWindowsQuotedNativePathResolution:
    @pytest.mark.parametrize("quote", ["'", '"'])
    def test_dash_c_targeting_repo(self, repo, tmp_path, quote):
        command = f"git -C {quote}{repo}{quote} checkout main"
        hit, _ = _detect(command, tmp_path, repo)
        assert hit is True

    def test_dash_c_echo_literal_substitution_targeting_repo(self, repo, tmp_path):
        # Git Bash executes POSIX substitutions on Windows before launching Git.
        command = f'git -C "$(echo {repo.as_posix()})" checkout main'
        hit, _ = _detect(command, tmp_path, repo)
        assert hit is True

    def test_dash_c_more_than_eight_nested_echo_substitutions(self, repo, tmp_path):
        # Nine levels cross the production evaluator's ``depth > 8`` cap and
        # must therefore fail closed for a mutating Git subcommand.
        substitution = f"'{repo.as_posix()}'"
        for _ in range(9):
            substitution = f"$(echo {substitution})"
        command = f'git -C "{substitution}" checkout main'
        hit, _ = _detect(command, tmp_path, repo)
        assert hit is True

    def test_explicit_work_tree_targeting_repo(self, repo, tmp_path):
        command = f"git --work-tree='{repo}' checkout main"
        hit, _ = _detect(command, tmp_path, repo)
        assert hit is True

    def test_environment_work_tree_targeting_repo(self, repo, tmp_path):
        command = f"GIT_WORK_TREE='{repo}' git checkout main"
        hit, _ = _detect(command, tmp_path, repo)
        assert hit is True

    @pytest.mark.parametrize("style", ["printf", "printf-format", "backtick"])
    def test_dash_c_literal_command_substitution_targeting_repo(
        self, repo, tmp_path, style
    ):
        if style == "printf":
            substitution = f"$(printf '{repo}')"
        elif style == "printf-format":
            substitution = f"$(printf '%s' '{repo}')"
        else:
            substitution = f"`printf '{repo}'`"
        command = f'git -C "{substitution}" checkout main'
        hit, _ = _detect(command, tmp_path, repo)
        assert hit is True

    def test_literal_command_substitution_preserves_unc_path(self):
        import tools.self_repo_guard as mod

        command = r'''git -C "$(printf '\\server\share\repo')" checkout main'''
        assert mod._shell_words_at(command, 0)[2] == r"\\server\share\repo"

    def test_literal_command_substitution_avoids_placeholder_collision(self):
        import tools.self_repo_guard as mod

        command = (
            r'''git -C "__HERMES_WINDOWS_PATH_0__'''
            r'''$(printf 'C:\repo')" checkout main'''
        )
        assert mod._shell_words_at(command, 0)[2] == (
            r"__HERMES_WINDOWS_PATH_0__C:\repo"
        )

    def test_literal_command_substitution_avoids_private_sentinel_collision(self):
        import tools.self_repo_guard as mod

        literal_sentinel = "\ue000"
        command = f'''git -C "{literal_sentinel}$(printf 'C:\\repo')" checkout main'''
        assert mod._shell_words_at(command, 0)[2] == (
            f"{literal_sentinel}C:\\repo"
        )

    @pytest.mark.parametrize(
        "substitution",
        [
            '$(printf "{path}")',
            '`printf "{path}"`',
        ],
    )
    def test_single_quoted_substitution_remains_literal(
        self, repo, tmp_path, substitution
    ):
        literal = substitution.format(path=repo.as_posix())
        command = f"git -C '{literal}' checkout main"
        hit, _ = _detect(command, tmp_path, repo)
        assert hit is False

    def test_double_quoted_substitution_is_evaluated(self, repo, tmp_path):
        command = f'''git -C "$(printf '{repo}')" checkout main'''
        hit, _ = _detect(command, tmp_path, repo)
        assert hit is True

    @pytest.mark.parametrize(
        "command_template",
        [
            '''git -C "$(printf '{path}')$(printf '')" checkout main''',
            '''git -C "$(printf "$(printf '{path}')")" checkout main''',
            '''git -C "$(printf '{path}')`printf ''`" checkout main''',
        ],
    )
    def test_composed_literal_substitution_targets_repo(
        self, repo, tmp_path, command_template
    ):
        command = command_template.format(path=repo)
        hit, _ = _detect(command, tmp_path, repo)
        assert hit is True

    @pytest.mark.parametrize(
        "literal",
        [
            r'''\$(printf '{path}')''',
            r'''\`printf '{path}'\`''',
        ],
    )
    def test_escaped_substitution_remains_literal(self, repo, tmp_path, literal):
        command = f'''git -C "{literal.format(path=repo.as_posix())}" checkout main'''
        hit, _ = _detect(command, tmp_path, repo)
        assert hit is False

    @pytest.mark.parametrize("variable", ["HOME", "PWD"])
    def test_dynamic_parameter_targeting_repo_fails_closed(
        self, repo, tmp_path, monkeypatch, variable
    ):
        monkeypatch.setenv(variable, str(repo))
        command = f'''git -C "$(printf "${variable}")" checkout main'''
        hit, _ = _detect(command, tmp_path, repo)
        assert hit is True

    def test_dynamic_environment_work_tree_fails_closed(
        self, repo, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HOME", str(repo))
        command = '''GIT_WORK_TREE="$HOME" git checkout main'''
        hit, _ = _detect(command, tmp_path, repo)
        assert hit is True

    def test_dynamic_worktree_victim_fails_closed(
        self, repo, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HOME", str(repo))
        command = '''git worktree remove "$HOME"'''
        hit, _ = _detect(command, tmp_path, repo)
        assert hit is True

    def test_dynamic_parameter_read_only_git_is_allowed(
        self, repo, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HOME", str(repo))
        command = '''git -C "$(printf "$HOME")" status'''
        hit, _ = _detect(command, tmp_path, repo)
        assert hit is False

    def test_single_quoted_parameter_remains_literal(self, repo, tmp_path):
        command = "git -C '$HOME' checkout main"
        hit, _ = _detect(command, tmp_path, repo)
        assert hit is False

    def test_unquoted_backslashes_follow_shell_escape_semantics(self, repo, tmp_path):
        hit, _ = _detect(f"git -C {repo} checkout main", tmp_path, repo)
        assert hit is False

    @pytest.mark.parametrize("quote", ["'", '"'])
    def test_quoted_relative_native_path_targeting_repo(
        self, repo, tmp_path, quote
    ):
        nested = repo / "nested"
        nested.mkdir()
        relative = f"{repo.name}\\nested"
        command = f"git -C {quote}{relative}{quote} checkout main"
        hit, _ = _detect(command, repo.parent, repo)
        assert hit is True


class TestSourceRootResolution:
    def test_resolves_to_repo_when_git_dir_present(self):
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


class TestBlockMessageGuidance:
    """The block message must steer agents to a disk-backed scratch clone,
    not a bare "temporary clone" (agents defaulted to /tmp, which is tmpfs
    on most distros — parallel salvage clones running npm ci filled a 32GB
    tmpfs to 97% in one campaign)."""

    def test_message_recommends_shared_clone_on_disk(self, repo):
        hit, msg = _detect("git rebase origin/main", repo, repo)
        assert hit is True
        assert "git clone --shared" in msg
        assert "scratch" in msg

    def test_message_warns_against_tmp_for_dep_installs(self, repo):
        hit, msg = _detect("git rebase origin/main", repo, repo)
        assert hit is True
        assert "tmpfs" in msg
        assert "Delete the clone" in msg

    def test_scratch_hint_honors_hermes_home(self, repo, monkeypatch, tmp_path):
        hermes_home = tmp_path / "custom-hermes-home"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        hit, msg = _detect("git rebase origin/main", repo, repo)
        assert hit is True
        assert str(hermes_home / "scratch") in msg
