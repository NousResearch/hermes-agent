import json
import subprocess
from pathlib import Path

import toolsets
from tools import codex_workflow_run_tool as workflow
from tools.registry import registry


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _clean_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "initial")
    return repo


def _call(**kwargs):
    defaults = {
        "task": "make a small change",
        "allowed_files": ["README.md"],
        "allowed_globs": [],
        "verify_cmd_ids": ["diff-check"],
        "continue_policy": "stop-on-review-needed",
        "dirty_baseline_policy": "require-clean",
        "mode": "execute",
    }
    defaults.update(kwargs)
    return json.loads(workflow.codex_workflow_run(defaults))


def test_clean_repo_calls_staged_implementation(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    calls = []

    def fake_staged(args):
        calls.append(args)
        return json.dumps({"status": "ready_for_review", "resolved_workdir": args["workdir"]})

    monkeypatch.setattr(workflow.staged, "codex_staged_implement", fake_staged)

    result = _call(workdir=str(repo))

    assert result["status"] == "staged_called"
    assert result["dirty_recovery"]["strategy"] == "none"
    assert result["codex_staged_result"]["status"] == "ready_for_review"
    assert calls == [
        {
            "workdir": str(repo),
            "task": "make a small change",
            "allowed_files": ["README.md"],
            "allowed_globs": [],
            "verify_cmd_ids": ["diff-check"],
            "continue_policy": "stop-on-review-needed",
            "dirty_baseline_policy": "require-clean",
            "mode": "execute",
        }
    ]


def test_cache_only_dirty_cleanup_then_calls_staged(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    cache_path = repo / ".pytest_cache" / "v" / "cache" / "nodeids"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("cached\n", encoding="utf-8")
    calls = []

    def fake_staged(args):
        calls.append(args)
        return json.dumps({"status": "ready_for_review"})

    monkeypatch.setattr(workflow.staged, "codex_staged_implement", fake_staged)

    result = _call(
        workdir=str(repo),
        standing_authorization=True,
        auto_clean_cache=True,
    )

    assert result["status"] == "staged_called"
    assert result["dirty_recovery"]["strategy"] == "cache_cleanup"
    assert result["dirty_recovery"]["cache_cleaned_paths"] == [".pytest_cache/v/cache/nodeids"]
    assert result["dirty_recovery"]["post_cleanup_dirty_check"]["is_clean"] is True
    assert not cache_path.exists()
    assert len(calls) == 1


def test_source_unknown_dirty_uses_isolated_worktree_when_authorized(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    (repo / "tools").mkdir()
    (repo / "tools" / "dirty_tool.py").write_text("dirty\n", encoding="utf-8")
    original_dirty = repo / "scratch.tmp"
    original_dirty.write_text("dirty\n", encoding="utf-8")
    isolated = tmp_path / ".hermes-worktrees" / "repo-phase-abc123"
    calls = []
    worktree_calls = []

    def fake_create(repo_arg, *, stage_id, git_head):
        worktree_calls.append((repo_arg, stage_id, git_head))
        isolated.mkdir(parents=True)
        (isolated / "README.md").write_text("hello\n", encoding="utf-8")
        return {"path": str(isolated), "branch": "work/phase-20260606-abc123", "source_head": git_head}

    def fake_staged(args):
        calls.append(args)
        return json.dumps({"status": "ready_for_review", "resolved_workdir": args["workdir"]})

    monkeypatch.setattr(workflow, "_create_isolated_worktree", fake_create)
    monkeypatch.setattr(workflow.staged, "codex_staged_implement", fake_staged)

    result = _call(
        workdir=str(repo),
        standing_authorization=True,
        allow_isolated_worktree=True,
        stage_id="phase",
    )

    assert result["status"] == "staged_called"
    assert result["dirty_recovery"]["strategy"] == "isolated_worktree"
    assert result["dirty_recovery"]["isolated_worktree"]["path"] == str(isolated)
    assert calls[0]["workdir"] == str(isolated)
    assert worktree_calls[0][0] == repo
    assert (repo / "tools" / "dirty_tool.py").exists()
    assert original_dirty.exists()


def test_dirty_without_authorization_requires_recovery_and_does_not_call_staged(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    (repo / "scratch.tmp").write_text("dirty\n", encoding="utf-8")
    calls = []

    def fake_staged(args):
        calls.append(args)
        raise AssertionError("staged implementation should not be called")

    monkeypatch.setattr(workflow.staged, "codex_staged_implement", fake_staged)

    result = _call(workdir=str(repo))

    assert result["status"] == "dirty_recovery_required"
    assert result["dirty_recovery"]["initial_dirty_check"]["is_clean"] is False
    assert result["dirty_recovery"]["requires_user_decision"] is True
    assert result["codex_staged_result"] is None
    assert calls == []


def test_registration_and_core_toolset_exposure():
    schema = registry.get_schema("codex_workflow_run")

    assert schema is not None
    assert registry.get_toolset_for_tool("codex_workflow_run") == "codex_staged_implement"
    assert "codex_workflow_run" in toolsets._HERMES_CORE_TOOLS
    assert "codex_workflow_run" in toolsets.resolve_toolset("codex_staged_implement")
    assert "codex_staged_implement" in toolsets.resolve_toolset("codex_staged_implement")
    assert "codex_workflow_run" in toolsets.resolve_toolset("codex_workflow_run")


def test_schema_has_no_executable_command_suggestions():
    encoded = json.dumps(registry.get_schema("codex_workflow_run"))

    assert "git worktree add" not in encoded
    assert "codex exec" not in encoded
    assert "codex-yuna exec" not in encoded
    assert "stash" not in encoded
    assert "reset" not in encoded


def test_cache_cleanup_never_deletes_original_dirty_source(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    source_path = repo / "tools" / "dirty_tool.py"
    source_path.parent.mkdir()
    source_path.write_text("dirty\n", encoding="utf-8")
    cache_path = repo / ".pytest_cache" / "v" / "cache" / "nodeids"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("cached\n", encoding="utf-8")
    calls = []

    def fake_staged(args):
        calls.append(args)
        raise AssertionError("staged implementation should not be called")

    monkeypatch.setattr(workflow.staged, "codex_staged_implement", fake_staged)

    result = _call(
        workdir=str(repo),
        standing_authorization=True,
        auto_clean_cache=True,
    )

    assert result["status"] == "dirty_recovery_required"
    assert source_path.exists()
    assert cache_path.exists()
    assert calls == []
