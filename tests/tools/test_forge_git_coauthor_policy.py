"""Forge git co-author policy smoke tests."""

import os
import subprocess

from tools.forge_git_coauthor import (
    RAZ_COAUTHOR_TRAILER,
    apply_forge_git_coauthor_policy,
    forge_git_coauthor_prelude,
    should_enable_forge_git_coauthor,
)


def _run(cmd: str, cwd, env=None) -> subprocess.CompletedProcess:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(
        ["bash", "-lc", cmd],
        cwd=cwd,
        env=merged,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )


def test_policy_enabled_only_for_forge_kanban_worker(monkeypatch):
    monkeypatch.setenv("HERMES_PROFILE", "forge")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_example")
    monkeypatch.delenv("HERMES_FORGE_COAUTHOR_DISABLED", raising=False)
    assert should_enable_forge_git_coauthor()

    monkeypatch.setenv("HERMES_FORGE_COAUTHOR_DISABLED", "1")
    assert not should_enable_forge_git_coauthor()

    monkeypatch.setenv("HERMES_FORGE_COAUTHOR_DISABLED", "0")
    monkeypatch.setenv("HERMES_PROFILE", "default")
    assert not should_enable_forge_git_coauthor()


def test_terminal_tool_policy_application_is_scoped(monkeypatch):
    command = "git commit -m 'feat: example'"

    monkeypatch.setenv("HERMES_PROFILE", "forge")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_example")
    wrapped = apply_forge_git_coauthor_policy(command)
    assert wrapped != command
    assert RAZ_COAUTHOR_TRAILER in wrapped
    assert wrapped.endswith(command)

    monkeypatch.setenv("HERMES_FORGE_COAUTHOR_DISABLED", "1")
    assert apply_forge_git_coauthor_policy(command) == command


def test_forge_git_commit_gets_raz_trailer_exactly_once(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _run("git init && git config user.name Forge && git config user.email forge@example.invalid", repo)
    (repo / "file.txt").write_text("hello\n", encoding="utf-8")

    env = {
        "HERMES_PROFILE": "forge",
        "HERMES_KANBAN_TASK": "t_example",
    }
    _run(f"{forge_git_coauthor_prelude()}\ngit add file.txt && git commit -m 'feat: add file'", repo, env)

    message = _run("git log -1 --format=%B", repo).stdout
    assert message.count(RAZ_COAUTHOR_TRAILER) == 1
    assert message.rstrip().endswith(RAZ_COAUTHOR_TRAILER)


def test_existing_commit_message_trailers_remain_valid_and_deduped(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _run("git init && git config user.name Forge && git config user.email forge@example.invalid", repo)
    (repo / "file.txt").write_text("hello\n", encoding="utf-8")

    env = {
        "HERMES_PROFILE": "forge",
        "HERMES_KANBAN_TASK": "t_example",
    }
    other_coauthor = "Co-authored-by: Other Person <other@example.com>"
    message_file = repo / "message.txt"
    message_file.write_text(
        "feat: add file\n\nBody line.\n\nRefs: #123\n"
        f"{other_coauthor}\n"
        f"{RAZ_COAUTHOR_TRAILER}\n"
        f"{RAZ_COAUTHOR_TRAILER}\n",
        encoding="utf-8",
    )

    _run(f"{forge_git_coauthor_prelude()}\ngit add file.txt && git commit -F message.txt", repo, env)

    message = _run("git log -1 --format=%B", repo).stdout
    assert message.count(RAZ_COAUTHOR_TRAILER) == 1
    assert other_coauthor in message
    assert "Refs: #123" in message
    parsed = _run("git log -1 --format=%B | git interpret-trailers --parse", repo).stdout
    assert RAZ_COAUTHOR_TRAILER in parsed
    assert other_coauthor in parsed
    assert message.rstrip().endswith(RAZ_COAUTHOR_TRAILER)


def test_forge_git_push_guard_rejects_raz_line_outside_trailer_block(tmp_path):
    remote = tmp_path / "remote.git"
    repo = tmp_path / "repo"
    _run(f"git init --bare {remote}", tmp_path)
    _run(
        f"git init {repo} && cd {repo} && git config user.name Forge && "
        "git config user.email forge@example.invalid && git remote add origin "
        f"{remote}",
        tmp_path,
    )
    (repo / "file.txt").write_text("hello\n", encoding="utf-8")
    message_file = repo / "message.txt"
    message_file.write_text(
        f"feat: body-only trailer\n\n{RAZ_COAUTHOR_TRAILER}\n\nMore body text after the line.\n",
        encoding="utf-8",
    )
    _run("git add file.txt && git commit -F message.txt", repo)

    env = {
        "HERMES_PROFILE": "forge",
        "HERMES_KANBAN_TASK": "t_example",
    }
    result = subprocess.run(
        ["bash", "-lc", f"{forge_git_coauthor_prelude()}\ngit push -u origin HEAD"],
        cwd=repo,
        env={**os.environ, **env},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 1
    assert "Forge commit policy blocked git push" in result.stdout
    assert "missing Raz co-author trailer" in result.stdout


def test_forge_git_push_guard_blocks_missing_trailer(tmp_path):
    remote = tmp_path / "remote.git"
    repo = tmp_path / "repo"
    _run(f"git init --bare {remote}", tmp_path)
    _run(
        f"git init {repo} && cd {repo} && git config user.name Forge && "
        "git config user.email forge@example.invalid && git remote add origin "
        f"{remote}",
        tmp_path,
    )
    (repo / "file.txt").write_text("hello\n", encoding="utf-8")
    _run("git add file.txt && git commit -m 'feat: missing trailer'", repo)

    env = {
        "HERMES_PROFILE": "forge",
        "HERMES_KANBAN_TASK": "t_example",
    }
    result = subprocess.run(
        ["bash", "-lc", f"{forge_git_coauthor_prelude()}\ngit push -u origin HEAD"],
        cwd=repo,
        env={**os.environ, **env},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 1
    assert "Forge commit policy blocked git push" in result.stdout
    assert "missing Raz co-author trailer" in result.stdout
