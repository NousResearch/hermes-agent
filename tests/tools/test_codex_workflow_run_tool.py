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
    branch = _git(repo, "branch", "--show-current")
    head = _git(repo, "rev-parse", "HEAD")
    event = workflow.provenance.provenance_event(
        repo=repo,
        branch=branch,
        head_sha=head,
        stage_id="phase12a0",
        session_id="session-a",
        actor="hermes",
        tool="codex_workflow_run",
        operation="create",
        path=".pytest_cache/v/cache/nodeids",
        before_hash=None,
        after_hash=workflow.provenance.file_hash(cache_path),
    )
    calls = []

    def fake_staged(args):
        calls.append(args)
        return json.dumps({"status": "ready_for_review"})

    monkeypatch.setattr(workflow.staged, "codex_staged_implement", fake_staged)

    result = _call(
        workdir=str(repo),
        standing_authorization=True,
        auto_clean_cache=True,
        cleanup_allowed_globs=[".pytest_cache/**"],
        session_id="session-a",
        provenance_events=[event],
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
    properties = schema["parameters"]["properties"]
    assert "session_id" in properties
    assert "provenance_events" in properties


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


def _verified_evidence(repo: Path, *, touched_files: list[str] | None = None, dirty_state_id: str | None = None) -> dict:
    dirty = workflow.staged._dirty_check(repo)
    return {
        "stage_id": "phase-4",
        "allowed_files": ["README.md"],
        "allowed_globs": [],
        "touched_files": touched_files if touched_files is not None else dirty["dirty_paths"],
        "dirty_state_id": dirty_state_id if dirty_state_id is not None else dirty["dirty_state_id"],
        "codex_implementation_status": "completed",
        "codex_review_status": "packet_only_passed",
        "hermes_verification_commands": [{"id": "diff-check", "status": "passed"}],
        "verified_at": "2026-06-06T00:00:00Z",
    }


def test_checkpoint_valid_evidence_commits_touched_files(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    evidence = {}

    def fake_staged(args):
        (Path(args["workdir"]) / "README.md").write_text("hello\nchanged\n", encoding="utf-8")
        evidence.update(_verified_evidence(repo))
        return json.dumps({"status": "ready_for_review", "candidate_id": "cand-1"})

    monkeypatch.setattr(workflow.staged, "codex_staged_implement", fake_staged)

    result = _call(
        workdir=str(repo),
        standing_authorization=True,
        checkpoint_verified_diff=True,
        verification_evidence=evidence,
        checkpoint_message="checkpoint phase 4",
        stage_id="phase-4",
    )

    assert result["status"] == "staged_called"
    assert result["checkpoint"]["status"] == "committed"
    assert result["checkpoint"]["message"] == "checkpoint phase 4"
    assert result["checkpoint"]["touched_files"] == ["README.md"]
    assert _git(repo, "status", "--porcelain=v1", "--untracked-files=all") == ""
    assert _git(repo, "log", "-1", "--pretty=%s") == "checkpoint phase 4"


def test_checkpoint_without_evidence_does_not_commit(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)

    def fake_staged(args):
        (Path(args["workdir"]) / "README.md").write_text("hello\nchanged\n", encoding="utf-8")
        return json.dumps({"status": "ready_for_review"})

    monkeypatch.setattr(workflow.staged, "codex_staged_implement", fake_staged)

    result = _call(
        workdir=str(repo),
        standing_authorization=True,
        checkpoint_verified_diff=True,
    )

    assert result["status"] == "checkpoint_blocked"
    assert result["checkpoint"]["status"] == "blocked"
    assert result["checkpoint"]["reason"] == "missing_verification_evidence"
    assert _git(repo, "log", "--oneline").count("\n") == 0
    assert "README.md" in _git(repo, "status", "--porcelain=v1", "--untracked-files=all")


def test_checkpoint_dirty_state_id_mismatch_blocks(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    evidence = {}

    def fake_staged(args):
        (Path(args["workdir"]) / "README.md").write_text("hello\nchanged\n", encoding="utf-8")
        evidence.update(_verified_evidence(repo, dirty_state_id="stale"))
        return json.dumps({"status": "ready_for_review"})

    monkeypatch.setattr(workflow.staged, "codex_staged_implement", fake_staged)

    result = _call(
        workdir=str(repo),
        standing_authorization=True,
        checkpoint_verified_diff=True,
        verification_evidence=evidence,
    )

    assert result["status"] == "checkpoint_blocked"
    assert result["checkpoint"]["reason"] == "dirty_state_id_mismatch"
    assert "README.md" in _git(repo, "status", "--porcelain=v1", "--untracked-files=all")


def test_checkpoint_without_standing_authorization_blocks(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    evidence = {}

    def fake_staged(args):
        (Path(args["workdir"]) / "README.md").write_text("hello\nchanged\n", encoding="utf-8")
        evidence.update(_verified_evidence(repo))
        return json.dumps({"status": "ready_for_review"})

    monkeypatch.setattr(workflow.staged, "codex_staged_implement", fake_staged)

    result = _call(
        workdir=str(repo),
        checkpoint_verified_diff=True,
        verification_evidence=evidence,
    )

    assert result["status"] == "checkpoint_blocked"
    assert result["checkpoint"]["reason"] == "authorization_required"
    assert result["checkpoint"]["authorization_required"] is True
    assert "README.md" in _git(repo, "status", "--porcelain=v1", "--untracked-files=all")


def test_leftover_candidate_reported_after_staged_leaves_dirty(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)

    def fake_staged(args):
        (Path(args["workdir"]) / "README.md").write_text("hello\ncandidate\n", encoding="utf-8")
        return json.dumps(
            {
                "status": "ready_for_review",
                "candidate_id": "cand-left",
                "candidate_disposition": "pending_review",
                "completion_trusted": False,
            }
        )

    monkeypatch.setattr(workflow.staged, "codex_staged_implement", fake_staged)

    result = _call(workdir=str(repo))

    assert result["status"] == "staged_called"
    assert result["leftover_candidate"]["requires_review"] is True
    assert result["leftover_candidate"]["requires_hermes_verification"] is True
    assert result["leftover_candidate"]["candidate_id"] == "cand-left"
    assert result["leftover_candidate"]["candidate_disposition"] == "pending_review"
    assert result["leftover_candidate"]["completion_trusted"] is False
    assert result["leftover_candidate"]["touched_files"] == ["README.md"]


def test_checkpoint_touched_files_outside_current_dirty_blocks(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    evidence = {}

    def fake_staged(args):
        (Path(args["workdir"]) / "README.md").write_text("hello\nchanged\n", encoding="utf-8")
        evidence.update(_verified_evidence(repo, touched_files=["README.md", "missing.txt"]))
        return json.dumps({"status": "ready_for_review"})

    monkeypatch.setattr(workflow.staged, "codex_staged_implement", fake_staged)

    result = _call(
        workdir=str(repo),
        standing_authorization=True,
        checkpoint_verified_diff=True,
        verification_evidence=evidence,
    )

    assert result["status"] == "checkpoint_blocked"
    assert result["checkpoint"]["reason"] == "touched_files_do_not_match_dirty_paths"
    assert "README.md" in _git(repo, "status", "--porcelain=v1", "--untracked-files=all")


def test_checkpoint_touched_files_outside_allowlist_blocks(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    (repo / "other.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "other.txt")
    _git(repo, "commit", "-m", "add other")
    evidence = {}

    def fake_staged(args):
        (Path(args["workdir"]) / "README.md").write_text("hello\nchanged\n", encoding="utf-8")
        (Path(args["workdir"]) / "other.txt").write_text("base\nchanged\n", encoding="utf-8")
        evidence.update(_verified_evidence(repo, touched_files=["README.md", "other.txt"]))
        return json.dumps({"status": "ready_for_review"})

    monkeypatch.setattr(workflow.staged, "codex_staged_implement", fake_staged)

    result = _call(
        workdir=str(repo),
        standing_authorization=True,
        checkpoint_verified_diff=True,
        verification_evidence=evidence,
    )

    assert result["status"] == "checkpoint_blocked"
    assert result["checkpoint"]["reason"] == "touched_files_outside_allowlist"
    assert "README.md" in _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    assert "other.txt" in _git(repo, "status", "--porcelain=v1", "--untracked-files=all")


def test_dry_run_clean_repo_does_not_call_staged_or_checkpoint(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    calls = []

    def fake_staged(args):
        calls.append(args)
        raise AssertionError("dry_run must not call staged implementation")

    monkeypatch.setattr(workflow.staged, "codex_staged_implement", fake_staged)

    result = _call(
        workdir=str(repo),
        mode="dry_run",
        standing_authorization=True,
        checkpoint_verified_diff=True,
        verification_evidence={"stage_id": "phase-4"},
    )

    assert result["status"] == "dry_run"
    assert result["would_call_staged"] is True
    assert result["codex_staged_result"] is None
    assert "checkpoint" not in result
    assert calls == []
    assert _git(repo, "status", "--porcelain=v1", "--untracked-files=all") == ""


def test_dry_run_cache_dirty_does_not_clean_cache_or_call_staged(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    cache_path = repo / ".pytest_cache" / "v" / "cache" / "nodeids"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("cached\n", encoding="utf-8")
    calls = []

    def fake_staged(args):
        calls.append(args)
        raise AssertionError("dry_run must not call staged implementation")

    monkeypatch.setattr(workflow.staged, "codex_staged_implement", fake_staged)

    result = _call(
        workdir=str(repo),
        mode="dry_run",
        standing_authorization=True,
        auto_clean_cache=True,
    )

    assert result["status"] == "dry_run"
    assert result["would_call_staged"] is False
    assert cache_path.exists()
    assert result["dirty_recovery"]["initial_dirty_check"]["dirty_path_classes"]["cache"] == [
        ".pytest_cache/v/cache/nodeids"
    ]
    assert calls == []


def test_dry_run_source_dirty_does_not_create_isolated_worktree(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    source_path = repo / "tools" / "dirty_tool.py"
    source_path.parent.mkdir()
    source_path.write_text("dirty\n", encoding="utf-8")
    staged_calls = []
    worktree_calls = []

    def fake_staged(args):
        staged_calls.append(args)
        raise AssertionError("dry_run must not call staged implementation")

    def fake_create(*args, **kwargs):
        worktree_calls.append((args, kwargs))
        raise AssertionError("dry_run must not create isolated worktrees")

    monkeypatch.setattr(workflow.staged, "codex_staged_implement", fake_staged)
    monkeypatch.setattr(workflow, "_create_isolated_worktree", fake_create)

    result = _call(
        workdir=str(repo),
        mode="dry_run",
        standing_authorization=True,
        allow_isolated_worktree=True,
    )

    assert result["status"] == "dry_run"
    assert result["would_call_staged"] is False
    assert source_path.exists()
    assert staged_calls == []
    assert worktree_calls == []


def test_unknown_untracked_doc_plan_is_preserved(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    plan_path = repo / "docs" / "plans" / "phase12-notes.md"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text("notes from another session\n", encoding="utf-8")
    calls = []

    def fake_staged(args):
        calls.append(args)
        raise AssertionError("unknown docs/plans dirty must not call staged implementation")

    monkeypatch.setattr(workflow.staged, "codex_staged_implement", fake_staged)

    result = _call(
        workdir=str(repo),
        mode="dry_run",
        standing_authorization=True,
        auto_clean_cache=True,
        allowed_files=["README.md"],
    )

    assert result["status"] == "dry_run"
    assert plan_path.exists()
    assert calls == []
    assert result["would_delete_paths"] == []
    assert result["would_overwrite_paths"] == []
    assert result["dirty_ownership"][0]["path"] == "docs/plans/phase12-notes.md"
    assert result["dirty_ownership"][0]["owner_policy"] == "unknown_unowned"
    assert result["dirty_ownership"][0]["default_behavior"] == "preserve"
    assert "docs_plans_default_preserve" in result["cleanup_blocking_reasons"]


def test_dry_run_outputs_dirty_ownership_and_cleanup_block_reasons(tmp_path):
    repo = _clean_repo(tmp_path)
    cache_path = repo / ".pytest_cache" / "v" / "cache" / "nodeids"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("cached\n", encoding="utf-8")

    result = _call(
        workdir=str(repo),
        mode="dry_run",
        standing_authorization=True,
        auto_clean_cache=True,
        session_id="session-a",
    )

    assert result["status"] == "dry_run"
    assert result["dirty_ownership"] == [
        {
            "path": ".pytest_cache/v/cache/nodeids",
            "path_class": "cache",
            "owner_policy": "generated_cache",
            "owner_session_id": None,
            "default_behavior": "preserve",
            "cleanup_allowed": False,
            "blocking_reasons": ["dry_run_non_mutating", "owner_policy_not_current_session", "path_not_in_allowlist"],
        }
    ]
    assert result["cleanup_allowed"] is False
    assert result["cleanup_blocking_reasons"] == [
        "dry_run_non_mutating",
        "owner_policy_not_current_session",
        "path_not_in_allowlist",
    ]
    assert result["would_delete_paths"] == []
    assert result["would_overwrite_paths"] == []
    assert cache_path.exists()


def test_execute_cache_cleanup_blocks_other_session_owner(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    cache_path = repo / ".pytest_cache" / "v" / "cache" / "nodeids"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("cached\n", encoding="utf-8")
    calls = []

    def fake_staged(args):
        calls.append(args)
        raise AssertionError("cache cleanup without provenance should not call staged")

    monkeypatch.setattr(workflow.staged, "codex_staged_implement", fake_staged)

    result = _call(
        workdir=str(repo),
        standing_authorization=True,
        auto_clean_cache=True,
        session_id="session-a",
        provenance_events=[
            {
                "schema_version": 1,
                "event_id": "evt-other",
                "repo_id": workflow.provenance.repo_id(repo),
                "branch": _git(repo, "branch", "--show-current"),
                "head_sha": _git(repo, "rev-parse", "HEAD"),
                "stage_id": "phase12a0",
                "session_id": "session-b",
                "actor": "hermes",
                "tool": "codex_workflow_run",
                "operation": "create",
                "path": ".pytest_cache/v/cache/nodeids",
                "path_class": "generated_cache",
                "before_hash": None,
                "after_hash": workflow.provenance.file_hash(cache_path),
                "owner_session_id": "session-b",
                "owner_policy": "current_session",
                "authorization": {"explicit": False, "reason": ""},
                "timestamp": "2026-06-06T00:00:00Z",
            }
        ],
    )

    assert result["status"] == "dirty_recovery_required"
    assert cache_path.exists()
    assert result["dirty_recovery"]["strategy"] == "cache_cleanup_blocked_by_provenance"
    assert result["dirty_recovery"]["provenance_cleanup"][".pytest_cache/v/cache/nodeids"]["cleanup_allowed"] is False
    assert calls == []


def test_execute_cache_cleanup_blocks_unprovenanced_cache(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    cache_path = repo / ".pytest_cache" / "v" / "cache" / "nodeids"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("cached\n", encoding="utf-8")
    calls = []

    def fake_staged(args):
        calls.append(args)
        raise AssertionError("unprovenanced cache cleanup must not call staged")

    monkeypatch.setattr(workflow.staged, "codex_staged_implement", fake_staged)

    result = _call(
        workdir=str(repo),
        standing_authorization=True,
        auto_clean_cache=True,
        cleanup_allowed_globs=[".pytest_cache/**"],
        session_id="session-a",
    )

    assert result["status"] == "dirty_recovery_required"
    assert result["dirty_recovery"]["strategy"] == "cache_cleanup_blocked_by_provenance"
    assert "owner_policy_not_current_session" in result["dirty_recovery"]["cleanup_blocking_reasons"]
    assert cache_path.exists()
    assert calls == []


def test_execute_cache_cleanup_blocks_current_session_cache_outside_allowlist(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    cache_path = repo / ".pytest_cache" / "v" / "cache" / "nodeids"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("cached\n", encoding="utf-8")
    branch = _git(repo, "branch", "--show-current")
    head = _git(repo, "rev-parse", "HEAD")
    event = workflow.provenance.provenance_event(
        repo=repo,
        branch=branch,
        head_sha=head,
        stage_id="phase12a0",
        session_id="session-a",
        actor="hermes",
        tool="codex_workflow_run",
        operation="create",
        path=".pytest_cache/v/cache/nodeids",
        before_hash=None,
        after_hash=workflow.provenance.file_hash(cache_path),
    )
    calls = []

    def fake_staged(args):
        calls.append(args)
        raise AssertionError("out-of-allowlist cache cleanup must not call staged")

    monkeypatch.setattr(workflow.staged, "codex_staged_implement", fake_staged)

    result = _call(
        workdir=str(repo),
        # Keep execution scope valid but intentionally exclude cache cleanup path.
        allowed_files=["README.md"],
        allowed_globs=[],
        standing_authorization=True,
        auto_clean_cache=True,
        session_id="session-a",
        provenance_events=[event],
    )

    assert result["status"] == "dirty_recovery_required"
    assert result["dirty_recovery"]["strategy"] == "cache_cleanup_blocked_by_provenance"
    assert "path_not_in_allowlist" in result["dirty_recovery"]["cleanup_blocking_reasons"]
    assert cache_path.exists()
    assert calls == []


def test_cleanup_allowlist_is_separate_from_codex_write_scope():
    cleanup_scope = workflow._cleanup_allowlist(
        {"cleanup_allowed_globs": [".pytest_cache/**"]},
        {"files": ["README.md"], "globs": ["tools/**"]},
    )

    assert cleanup_scope == {"files": [], "globs": [".pytest_cache/**"]}
