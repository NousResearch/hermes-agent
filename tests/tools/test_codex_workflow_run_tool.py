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
    assert "must_fix_loop" in properties
    assert properties["max_fix_rounds"]["default"] == 2


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
        "risk_classes": ["docs_only"],
        "hermes_verification_commands": [
            {
                "cmd_id": "diff-check",
                "argv": ["git", "diff", "--check", "--", "README.md"],
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
                "start_time": "2026-06-06T00:00:00Z",
                "end_time": "2026-06-06T00:00:01Z",
                "status": "passed",
            }
        ],
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
    ledger_root = Path(result["ledger_path"]).parents[3]
    assert not ledger_root.exists()


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


def test_dry_run_ledger_event_preview_without_writing_ledger(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    result = _call(workdir=str(repo), mode="dry_run", stage_id="phase12a-ledger-resume")

    assert result["status"] == "dry_run"
    assert result["would_write_ledger"] is True
    assert result["would_record_ledger_events"] is True
    assert result["ledger_event_preview"][0]["operation"] == "dry_run_plan"
    assert result["ledger_event_preview"][0]["stage_id"] == "phase12a-ledger-resume"
    assert not Path(result["ledger_path"]).exists()
    assert not (hermes_home / "runtime" / "codex_workflows").exists()


def test_dry_run_no_mutation_even_when_write_authorized(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    source_path = repo / "README.md"
    source_path.write_text("authorized dirty\n", encoding="utf-8")
    calls = []

    def fake_staged(args):
        calls.append(args)
        raise AssertionError("dry_run must not call staged implementation")

    monkeypatch.setattr(workflow.staged, "codex_staged_implement", fake_staged)

    result = _call(
        workdir=str(repo),
        mode="dry_run",
        standing_authorization=True,
        allowed_files=["README.md"],
        cleanup_allowed_files=["README.md"],
        session_id="session-a",
    )

    assert result["status"] == "dry_run"
    assert result["would_call_staged"] is False
    assert result["would_delete_paths"] == []
    assert result["would_overwrite_paths"] == []
    assert result["cleanup_allowed"] is False
    assert source_path.read_text(encoding="utf-8") == "authorized dirty\n"
    assert calls == []


def test_dry_run_blocks_overlap_with_unknown_source_dirty(tmp_path):
    repo = _clean_repo(tmp_path)
    (repo / "README.md").write_text("unknown dirty\n", encoding="utf-8")

    result = _call(workdir=str(repo), mode="dry_run", allowed_files=["README.md"])

    assert result["status"] == "dry_run"
    assert result["would_call_staged"] is False
    assert "unknown_dirty_overlaps_write_scope" in result["blocking_reasons"]
    assert result["recommended_next_stage"]["stage_id"] == "stop_for_user"
    assert result["authorization_required"] == ["write_stage"]


def test_dry_run_recommends_isolated_worktree_for_non_overlap_unknown_dirty(tmp_path):
    repo = _clean_repo(tmp_path)
    scratch = repo / "scratch.tmp"
    scratch.write_text("dirty\n", encoding="utf-8")

    result = _call(workdir=str(repo), mode="dry_run", allowed_files=["README.md"])

    assert result["status"] == "dry_run"
    assert result["would_create_isolated_worktree"] is False
    assert "unknown_dirty_non_overlap_recommend_isolated_worktree" in result["blocking_reasons"]
    assert result["recommended_next_stage"]["stage_id"] == "isolated_worktree"
    assert scratch.exists()


def test_dry_run_cleanup_block_reasons_include_ownership_and_cleanup_scope(tmp_path):
    repo = _clean_repo(tmp_path)
    cache_path = repo / ".pytest_cache" / "v" / "cache" / "nodeids"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("cached\n", encoding="utf-8")

    result = _call(
        workdir=str(repo),
        mode="dry_run",
        standing_authorization=True,
        auto_clean_cache=True,
        allowed_files=["README.md"],
        cleanup_allowed_globs=[],
        session_id="session-a",
    )

    assert result["cleanup_allowed"] is False
    assert "dry_run_non_mutating" in result["cleanup_blocking_reasons"]
    assert "owner_policy_not_current_session" in result["cleanup_blocking_reasons"]
    assert "path_not_in_allowlist" in result["cleanup_blocking_reasons"]
    assert result["dirty_ownership"][0]["cleanup_allowed"] is False
    assert cache_path.exists()


def test_dry_run_no_delete_or_overwrite_for_unknown_unowned(tmp_path):
    repo = _clean_repo(tmp_path)
    (repo / "README.md").write_text("unknown dirty\n", encoding="utf-8")

    result = _call(
        workdir=str(repo),
        mode="dry_run",
        standing_authorization=True,
        cleanup_allowed_files=["README.md"],
        session_id="session-a",
    )

    assert result["dirty_ownership"][0]["owner_policy"] == "unknown_unowned"
    assert result["would_delete_paths"] == []
    assert result["would_overwrite_paths"] == []
    assert result["cleanup_allowed"] is False


def test_dry_run_never_commit_push_deploy_or_restart(tmp_path):
    repo = _clean_repo(tmp_path)

    result = _call(
        workdir=str(repo),
        mode="dry_run",
        standing_authorization=True,
        checkpoint_verified_diff=True,
    )

    assert result["would_commit"] is False
    assert result["would_push"] is False
    assert result["would_deploy_or_restart"] is False


def test_dry_run_authorization_required_for_write_stage(tmp_path):
    repo = _clean_repo(tmp_path)

    result = _call(workdir=str(repo), mode="dry_run")

    assert result["authorization_required"] == ["write_stage"]


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


def test_dry_run_does_not_call_mutating_or_cleanup_helpers(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    (repo / "README.md").write_text("dirty\n", encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("dry_run must not call mutating helper")

    monkeypatch.setattr(workflow, "_call_staged", forbidden)
    monkeypatch.setattr(workflow, "_create_isolated_worktree", forbidden)
    monkeypatch.setattr(workflow, "_clean_cache_dirty_paths", forbidden)
    monkeypatch.setattr(workflow.ledger, "write_ledger", forbidden)
    monkeypatch.setattr(workflow.provenance, "cleanup_decision", forbidden)
    monkeypatch.setattr(workflow.provenance, "provenance_event", forbidden)

    result = _call(
        workdir=str(repo),
        mode="dry_run",
        standing_authorization=True,
        checkpoint_verified_diff=True,
        allowed_files=["README.md"],
        cleanup_allowed_files=["README.md"],
        session_id="session-a",
    )

    assert result["status"] == "dry_run"
    assert result["would_call_staged"] is False
    assert result["would_create_isolated_worktree"] is False
    assert result["would_delete_paths"] == []
    assert result["would_overwrite_paths"] == []
    assert (repo / "README.md").read_text(encoding="utf-8") == "dirty\n"




def _write_guard_final(argv, payload: str) -> str:
    final_path = Path(argv[argv.index("--final-file") + 1])
    final_path.write_text(payload, encoding="utf-8")
    return payload


def _passed_guard_json() -> str:
    return json.dumps(
        {
            "status": "passed",
            "reason": "ok",
            "terminated_by_guard": False,
            "source_flood_detected": False,
            "diff_flood_detected": False,
            "json_field_flood_detected": False,
            "review": {
                "verdict": "passed",
                "summary": "ok",
                "must_fix": [],
                "suggested_fixes": [],
                "verification_commands": [],
                "final_judgment": "ok",
            },
        }
    )


def _failed_guard_json() -> str:
    return json.dumps(
        {
            "status": "failed",
            "reason": "must_fix_non_empty",
            "terminated_by_guard": False,
            "source_flood_detected": False,
            "diff_flood_detected": False,
            "json_field_flood_detected": False,
            "review": {
                "verdict": "failed",
                "summary": "needs fix",
                "must_fix": ["add regression coverage"],
                "suggested_fixes": ["refactor later"],
                "verification_commands": [],
                "final_judgment": "needs fix",
            },
        }
    )


def test_review_autopilot_uses_existing_packet_and_guard_commands(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    commands = []

    def fake_staged(args):
        (Path(args["workdir"]) / "README.md").write_text("hello\ncandidate\n", encoding="utf-8")
        return json.dumps({"status": "ready_for_review", "candidate_id": "cand-1", "completion_trusted": False})

    def fake_review_command(argv, *, timeout=workflow._REVIEW_TIMEOUT_SECONDS):
        commands.append(argv)
        if str(workflow._review_packet_script()) in argv:
            return subprocess.CompletedProcess(argv, 0, stdout="# packet\n", stderr="")
        if str(workflow._review_guard_script()) in argv:
            return subprocess.CompletedProcess(argv, 0, stdout=_write_guard_final(argv, _passed_guard_json()), stderr="")
        raise AssertionError(f"unexpected command: {argv}")

    monkeypatch.setattr(workflow.staged, "codex_staged_implement", fake_staged)
    monkeypatch.setattr(workflow, "_run_review_command", fake_review_command)

    result = _call(workdir=str(repo), review_autopilot=True, review_autopilot_authorized=True)

    assert result["status"] == "staged_called"
    assert result["review"]["status"] == "passed"
    assert str(workflow._review_packet_script()) in commands[0]
    assert str(workflow._review_guard_script()) in commands[1]
    assert "--max-total-chars" in commands[0]
    assert "--file" in commands[0]
    assert "README.md" in commands[0]
    assert "--review-packet-file" in commands[1]
    assert result["leftover_candidate"]["requires_review"] is False


def test_workflow_tool_optionally_outputs_next_stage_recommendation(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)

    def fake_staged(args):
        (Path(args["workdir"]) / "README.md").write_text("hello\ncandidate\n", encoding="utf-8")
        return json.dumps({"status": "ready_for_review", "candidate_id": "cand-1", "completion_trusted": False})

    def fake_review_command(argv, *, timeout=workflow._REVIEW_TIMEOUT_SECONDS):
        if str(workflow._review_packet_script()) in argv:
            return subprocess.CompletedProcess(argv, 0, stdout="# packet\n", stderr="")
        if str(workflow._review_guard_script()) in argv:
            return subprocess.CompletedProcess(argv, 0, stdout=_write_guard_final(argv, _passed_guard_json()), stderr="")
        raise AssertionError(f"unexpected command: {argv}")

    def forbidden_commit(*args, **kwargs):
        raise AssertionError("recommendation must not checkpoint or advance")

    evidence = {
        "risk_classes": ["docs_only"],
        "hermes_verification_commands": [
            {
                "cmd_id": "diff-check",
                "argv": ["git", "diff", "--check"],
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
                "start_time": "2026-06-07T00:00:00Z",
                "end_time": "2026-06-07T00:00:01Z",
                "status": "passed",
            }
        ],
    }

    monkeypatch.setattr(workflow.staged, "codex_staged_implement", fake_staged)
    monkeypatch.setattr(workflow, "_run_review_command", fake_review_command)
    monkeypatch.setattr(workflow, "_commit_checkpoint", forbidden_commit)

    result = _call(
        workdir=str(repo),
        review_autopilot=True,
        review_autopilot_authorized=True,
        recommend_next_stage=True,
        advance_next_stage=True,
        verification_evidence=evidence,
        next_stage_candidate={
            "stage_id": "phase12f-next-stage-recommender",
            "why": "review and verification passed",
            "allowed_files": ["agent/codex_workflow_recommender.py"],
            "verify_cmd_ids": ["workflow-tool-pytest", "py-compile", "diff-check"],
        },
    )

    assert result["status"] == "staged_called"
    assert result["next_stage_recommendation"]["status"] == "recommended"
    recommendation = result["next_stage_recommendation"]["recommendation"]
    assert recommendation["authorization_required"] is True
    assert recommendation["non_goals"] == ["commit", "push", "deploy", "restart", "force-push"]
    assert result["next_stage_recommendation"]["advance"]["status"] == "blocked"
    assert "checkpoint" not in result


def test_review_unavailable_timeout_is_fail_closed(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)

    def fake_staged(args):
        (Path(args["workdir"]) / "README.md").write_text("hello\ncandidate\n", encoding="utf-8")
        return json.dumps({"status": "ready_for_review"})

    def fake_review_command(argv, *, timeout=workflow._REVIEW_TIMEOUT_SECONDS):
        if str(workflow._review_packet_script()) in argv:
            return subprocess.CompletedProcess(argv, 0, stdout="# packet\n", stderr="")
        raise subprocess.TimeoutExpired(argv, timeout)

    monkeypatch.setattr(workflow.staged, "codex_staged_implement", fake_staged)
    monkeypatch.setattr(workflow, "_run_review_command", fake_review_command)

    result = _call(workdir=str(repo), review_autopilot=True, review_autopilot_authorized=True)

    assert result["status"] == "review_failed"
    assert result["review"]["status"] == "unavailable"
    assert result["review"]["reason"] == "review_timeout"


def test_review_invalid_json_is_fail_closed(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)

    def fake_staged(args):
        (Path(args["workdir"]) / "README.md").write_text("hello\ncandidate\n", encoding="utf-8")
        return json.dumps({"status": "ready_for_review"})

    def fake_review_command(argv, *, timeout=workflow._REVIEW_TIMEOUT_SECONDS):
        if str(workflow._review_packet_script()) in argv:
            return subprocess.CompletedProcess(argv, 0, stdout="# packet\n", stderr="")
        return subprocess.CompletedProcess(argv, 2, stdout=_write_guard_final(argv, "{not json"), stderr="")

    monkeypatch.setattr(workflow.staged, "codex_staged_implement", fake_staged)
    monkeypatch.setattr(workflow, "_run_review_command", fake_review_command)

    result = _call(workdir=str(repo), review_autopilot=True, review_autopilot_authorized=True)

    assert result["status"] == "review_failed"
    assert result["review"]["status"] == "unavailable"
    assert result["review"]["reason"] == "invalid_guard_json"


def test_review_missing_final_file_is_fail_closed(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)

    def fake_staged(args):
        (Path(args["workdir"]) / "README.md").write_text("hello\ncandidate\n", encoding="utf-8")
        return json.dumps({"status": "ready_for_review"})

    def fake_review_command(argv, *, timeout=workflow._REVIEW_TIMEOUT_SECONDS):
        if str(workflow._review_packet_script()) in argv:
            return subprocess.CompletedProcess(argv, 0, stdout="# packet\n", stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout=_passed_guard_json(), stderr="")

    monkeypatch.setattr(workflow.staged, "codex_staged_implement", fake_staged)
    monkeypatch.setattr(workflow, "_run_review_command", fake_review_command)

    result = _call(workdir=str(repo), review_autopilot=True, review_autopilot_authorized=True)

    assert result["status"] == "review_failed"
    assert result["review"]["status"] == "unavailable"
    assert result["review"]["reason"] == "missing_final_file"


def test_review_guard_metadata_fail_closed():
    cases = [
        ({"status": "passed", "review": {"verdict": "passed", "must_fix": []}, "terminated_by_guard": True, "reason": "terminated"}, "terminated"),
        ({"status": "passed", "review": {"verdict": "passed", "must_fix": []}, "process_exited_before_guard": True, "reason": "process_exited_before_guard"}, "process_exited_before_guard"),
        ({"status": "passed", "review": {"verdict": "passed", "must_fix": []}, "source_flood_detected": True, "reason": "source_flood"}, "source_flood"),
        ({"status": "passed", "review": {"verdict": "passed", "must_fix": []}, "diff_flood_detected": True, "reason": "diff_flood"}, "diff_flood"),
        ({"status": "passed", "review": {"verdict": "passed", "must_fix": []}, "json_field_flood_detected": True, "reason": "json_field_flood"}, "json_field_flood"),
        ({"status": "passed", "review": {"verdict": "passed", "must_fix": []}, "reason": "codex_bin_not_found"}, "codex_bin_not_found"),
        ({"status": "passed", "review": {"verdict": "passed", "must_fix": []}, "reason": "provider_5xx"}, "provider_5xx"),
    ]
    for guard, reason in cases:
        result = workflow._classify_review_guard_result(guard)
        assert result["status"] == "unavailable"
        assert result["reason"] == reason


def test_review_pass_requires_no_must_fix(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)

    def fake_staged(args):
        (Path(args["workdir"]) / "README.md").write_text("hello\ncandidate\n", encoding="utf-8")
        return json.dumps({"status": "ready_for_review"})

    def fake_review_command(argv, *, timeout=workflow._REVIEW_TIMEOUT_SECONDS):
        if str(workflow._review_packet_script()) in argv:
            return subprocess.CompletedProcess(argv, 0, stdout="# packet\n", stderr="")
        guard = json.loads(_passed_guard_json())
        guard["review"]["must_fix"] = ["fix this"]
        return subprocess.CompletedProcess(argv, 1, stdout=_write_guard_final(argv, json.dumps(guard)), stderr="")

    monkeypatch.setattr(workflow.staged, "codex_staged_implement", fake_staged)
    monkeypatch.setattr(workflow, "_run_review_command", fake_review_command)

    result = _call(workdir=str(repo), review_autopilot=True, review_autopilot_authorized=True)

    assert result["status"] == "review_failed"
    assert result["review"]["status"] == "failed"
    assert result["review"]["reason"] == "must_fix_non_empty"


def test_review_dirty_after_review_contaminated_stops(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)

    def fake_staged(args):
        (Path(args["workdir"]) / "README.md").write_text("hello\ncandidate\n", encoding="utf-8")
        return json.dumps({"status": "ready_for_review"})

    def fake_review_command(argv, *, timeout=workflow._REVIEW_TIMEOUT_SECONDS):
        if str(workflow._review_packet_script()) in argv:
            return subprocess.CompletedProcess(argv, 0, stdout="# packet\n", stderr="")
        (repo / "README.md").write_text("hello\ncandidate\nreview mutation\n", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, stdout=_write_guard_final(argv, _passed_guard_json()), stderr="")

    monkeypatch.setattr(workflow.staged, "codex_staged_implement", fake_staged)
    monkeypatch.setattr(workflow, "_run_review_command", fake_review_command)

    result = _call(workdir=str(repo), review_autopilot=True, review_autopilot_authorized=True)

    assert result["status"] == "review_failed"
    assert result["review"]["status"] == "failed"
    assert result["review"]["reason"] == "review_changed_worktree"
    assert result["review"]["contaminated"] is True


def test_review_packet_excludes_unknown_dirty_diff(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    (repo / "scratch.tmp").write_text("unknown dirty\n", encoding="utf-8")
    isolated = tmp_path / "isolated"
    commands = []

    def fake_create(repo_arg, *, stage_id, git_head):
        isolated.mkdir()
        (isolated / "README.md").write_text("hello\n", encoding="utf-8")
        _git(isolated, "init")
        _git(isolated, "config", "user.email", "test@example.com")
        _git(isolated, "config", "user.name", "Test User")
        _git(isolated, "add", "README.md")
        _git(isolated, "commit", "-m", "initial")
        return {"path": str(isolated), "branch": "work/review", "source_head": git_head}

    def fake_staged(args):
        (Path(args["workdir"]) / "README.md").write_text("hello\ncandidate\n", encoding="utf-8")
        return json.dumps({"status": "ready_for_review"})

    def fake_review_command(argv, *, timeout=workflow._REVIEW_TIMEOUT_SECONDS):
        commands.append(argv)
        if str(workflow._review_packet_script()) in argv:
            return subprocess.CompletedProcess(argv, 0, stdout="# packet\n", stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout=_write_guard_final(argv, _passed_guard_json()), stderr="")

    monkeypatch.setattr(workflow, "_create_isolated_worktree", fake_create)
    monkeypatch.setattr(workflow.staged, "codex_staged_implement", fake_staged)
    monkeypatch.setattr(workflow, "_run_review_command", fake_review_command)

    result = _call(
        workdir=str(repo),
        standing_authorization=True,
        allow_isolated_worktree=True,
        review_autopilot=True,
        review_autopilot_authorized=True,
    )

    packet_command = commands[0]
    assert result["status"] == "staged_called"
    assert result["review"]["status"] == "passed"
    assert "README.md" in packet_command
    assert "scratch.tmp" not in packet_command
    assert repo.joinpath("scratch.tmp").exists()


def test_must_fix_loop_without_review_autopilot_stops_as_review_unavailable(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)

    def fake_staged(args):
        (Path(args["workdir"]) / "README.md").write_text("hello\ncandidate\n", encoding="utf-8")
        return json.dumps({"status": "ready_for_review"})

    monkeypatch.setattr(workflow.staged, "codex_staged_implement", fake_staged)

    result = _call(
        workdir=str(repo),
        must_fix_loop=True,
        must_fix_loop_authorized=True,
    )

    assert result["status"] == "staged_called"
    assert result["must_fix_loop"]["status"] == "stopped"
    assert result["must_fix_loop"]["reason"] == "review_unavailable"
    assert result["must_fix_loop"]["blocks_continuation"] is True


def test_must_fix_loop_ready_for_round_after_failed_review_autopilot(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)

    def fake_staged(args):
        (Path(args["workdir"]) / "README.md").write_text("hello" + chr(10) + "candidate" + chr(10), encoding="utf-8")
        return json.dumps({"status": "ready_for_review"})

    def fake_review_command(argv, *, timeout=workflow._REVIEW_TIMEOUT_SECONDS):
        if str(workflow._review_packet_script()) in argv:
            return subprocess.CompletedProcess(argv, 0, stdout="# packet" + chr(10), stderr="")
        return subprocess.CompletedProcess(argv, 1, stdout=_write_guard_final(argv, _failed_guard_json()), stderr="")

    monkeypatch.setattr(workflow.staged, "codex_staged_implement", fake_staged)
    monkeypatch.setattr(workflow, "_run_review_command", fake_review_command)

    result = _call(
        workdir=str(repo),
        review_autopilot=True,
        review_autopilot_authorized=True,
        must_fix_loop=True,
        must_fix_loop_authorized=True,
    )

    assert result["status"] == "review_failed"
    assert result["review"]["status"] == "failed"
    assert result["must_fix_loop"]["status"] == "ready_for_round"
    assert result["must_fix_loop"]["must_fix"] == ["add regression coverage"]
    assert result["must_fix_loop"]["suggested_fixes_recorded"] == ["refactor later"]
    assert result["must_fix_loop"]["auto_implements_suggested_fixes"] is False


def test_must_fix_loop_max_round_exhaustion_through_workflow_tool(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)

    def fake_staged(args):
        (Path(args["workdir"]) / "README.md").write_text("hello" + chr(10) + "candidate" + chr(10), encoding="utf-8")
        return json.dumps({"status": "ready_for_review"})

    def fake_review_command(argv, *, timeout=workflow._REVIEW_TIMEOUT_SECONDS):
        if str(workflow._review_packet_script()) in argv:
            return subprocess.CompletedProcess(argv, 0, stdout="# packet" + chr(10), stderr="")
        return subprocess.CompletedProcess(argv, 1, stdout=_write_guard_final(argv, _failed_guard_json()), stderr="")

    monkeypatch.setattr(workflow.staged, "codex_staged_implement", fake_staged)
    monkeypatch.setattr(workflow, "_run_review_command", fake_review_command)

    result = _call(
        workdir=str(repo),
        review_autopilot=True,
        review_autopilot_authorized=True,
        must_fix_loop=True,
        must_fix_loop_authorized=True,
        current_fix_round=2,
    )

    assert result["status"] == "review_failed"
    assert result["must_fix_loop"]["status"] == "stopped"
    assert result["must_fix_loop"]["reason"] == "max_fix_rounds_exhausted"
    assert result["must_fix_loop"]["max_fix_rounds"] == 2
