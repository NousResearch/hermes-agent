import hashlib
import json
import subprocess
from pathlib import Path

import pytest

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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _preflight(status: str = "passed", *, blockers: list[str] | None = None) -> dict:
    checks = {
        "impl_guard_exists": True,
        "stage_runner_exists": True,
        "review_guard_exists": True,
        "review_packet_exists": True,
        "codex_bin_found": True,
        "node_bin_found": True,
        "sandbox_verified_env": True,
    }
    for blocker in blockers or []:
        if blocker == "missing_codex_bin":
            checks["codex_bin_found"] = False
        elif blocker == "missing_node_bin":
            checks["node_bin_found"] = False
        elif blocker == "sandbox_not_verified":
            checks["sandbox_verified_env"] = False
    return {"status": status, "checks": checks, "blockers": blockers or []}


@pytest.fixture(autouse=True)
def _default_preflight_passed(monkeypatch, request):
    if request.node.name.startswith("test_real_codex_preflight"):
        return
    monkeypatch.setattr(workflow, "_codex_preflight", lambda repo: _preflight(), raising=False)


def test_real_codex_preflight_reports_bounded_blockers(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)

    monkeypatch.delenv("HERMES_CODEX_IMPL_GUARD_SANDBOX_VERIFIED", raising=False)
    monkeypatch.setattr(
        workflow.shutil,
        "which",
        lambda name, path=None: None if name == "codex-yuna" else "/tmp/fake-node",
    )

    result = workflow._codex_preflight(repo)
    encoded = json.dumps(result)

    assert result["status"] == "blocked"
    assert result["checks"]["codex_bin_found"] is False
    assert result["checks"]["node_bin_found"] is True
    assert result["checks"]["sandbox_verified_env"] is False
    assert "missing_codex_bin" in result["blockers"]
    assert "sandbox_not_verified" in result["blockers"]
    assert all(isinstance(value, bool) for value in result["checks"].values())
    assert "PATH" not in encoded
    assert "HERMES_CODEX_IMPL_GUARD_SANDBOX_VERIFIED" not in encoded
    assert "/tmp/fake-node" not in encoded


def test_preflight_passed_requires_consistent_checks_and_no_blockers():
    assert workflow._preflight_passed(_preflight()) is True
    assert workflow._preflight_passed(_preflight("blocked", blockers=["sandbox_not_verified"])) is False
    assert workflow._preflight_passed({"status": "passed", "checks": {}, "blockers": []}) is False
    assert workflow._preflight_passed({"status": "passed", "checks": {"codex_bin_found": False}, "blockers": []}) is False
    assert workflow._preflight_passed({"status": "passed", "checks": {"codex_bin_found": True}, "blockers": ["x"]}) is False


def test_preflight_blocked_clean_repo_does_not_call_staged(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    calls = []

    def fake_staged(args):
        calls.append(args)
        raise AssertionError("blocked preflight must not call staged implementation")

    monkeypatch.setattr(workflow, "_codex_preflight", lambda repo: _preflight("blocked", blockers=["sandbox_not_verified"]))
    monkeypatch.setattr(workflow.staged, "codex_staged_implement", fake_staged)

    result = _call(workdir=str(repo))

    assert result["status"] == "preflight_blocked"
    assert result["preflight"]["status"] == "blocked"
    assert result["preflight"]["blockers"] == ["sandbox_not_verified"]
    assert result["codex_staged_result"] is None
    assert result["codex_packet_review"]["status"] == "not_run"
    assert result["codex_packet_review"]["reason"] == "preflight_blocked"
    assert calls == []


def test_preflight_blocked_cache_dirty_does_not_clean_cache(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    cache_path = repo / ".pytest_cache" / "v" / "cache" / "nodeids"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("cached\n", encoding="utf-8")

    monkeypatch.setattr(workflow, "_codex_preflight", lambda repo: _preflight("blocked", blockers=["missing_codex_bin"]))
    monkeypatch.setattr(
        workflow.staged,
        "codex_staged_implement",
        lambda args: (_ for _ in ()).throw(AssertionError("staged must not run")),
    )

    result = _call(workdir=str(repo), standing_authorization=True, auto_clean_cache=True)

    assert result["status"] == "preflight_blocked"
    assert cache_path.exists()
    assert result["dirty_recovery"]["strategy"] == "none"


def test_preflight_blocked_source_dirty_does_not_create_worktree(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    source_path = repo / "tools" / "dirty_tool.py"
    source_path.parent.mkdir()
    source_path.write_text("dirty\n", encoding="utf-8")
    worktree_calls = []

    def fake_create(*args, **kwargs):
        worktree_calls.append((args, kwargs))
        raise AssertionError("blocked preflight must not create isolated worktree")

    monkeypatch.setattr(workflow, "_codex_preflight", lambda repo: _preflight("blocked", blockers=["sandbox_not_verified"]))
    monkeypatch.setattr(workflow, "_create_isolated_worktree", fake_create)

    result = _call(workdir=str(repo), standing_authorization=True, allow_isolated_worktree=True)

    assert result["status"] == "preflight_blocked"
    assert source_path.exists()
    assert worktree_calls == []


def test_preflight_blocked_checkpoint_does_not_commit(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    initial_head = _git(repo, "rev-parse", "HEAD")

    monkeypatch.setattr(workflow, "_codex_preflight", lambda repo: _preflight("blocked", blockers=["sandbox_not_verified"]))
    monkeypatch.setattr(
        workflow.staged,
        "codex_staged_implement",
        lambda args: (_ for _ in ()).throw(AssertionError("staged must not run")),
    )

    result = _call(
        workdir=str(repo),
        standing_authorization=True,
        checkpoint_verified_diff=True,
        verification_evidence=_verified_evidence(repo),
        checkpoint_message="should not commit",
    )

    assert result["status"] == "preflight_blocked"
    assert _git(repo, "rev-parse", "HEAD") == initial_head


def test_dry_run_reports_blocked_preflight_without_mutation(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    cache_path = repo / ".pytest_cache" / "v" / "cache" / "nodeids"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("cached\n", encoding="utf-8")

    monkeypatch.setattr(workflow, "_codex_preflight", lambda repo: _preflight("blocked", blockers=["missing_node_bin"]))
    monkeypatch.setattr(
        workflow,
        "_run_packet_only_review",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("dry_run must not review")),
        raising=False,
    )

    result = _call(
        workdir=str(repo),
        mode="dry_run",
        standing_authorization=True,
        auto_clean_cache=True,
        allow_isolated_worktree=True,
        checkpoint_verified_diff=True,
    )

    assert result["status"] == "dry_run"
    assert result["preflight"]["status"] == "blocked"
    assert result["codex_staged_result"] is None
    assert result["codex_packet_review"] is None
    assert cache_path.exists()


def test_dry_run_passed_preflight_predicts_staged_without_mutation(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)

    monkeypatch.setattr(
        workflow.staged,
        "codex_staged_implement",
        lambda args: (_ for _ in ()).throw(AssertionError("dry_run must not call staged")),
    )
    monkeypatch.setattr(
        workflow,
        "_run_packet_only_review",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("dry_run must not review")),
        raising=False,
    )

    result = _call(workdir=str(repo), mode="dry_run")

    assert result["status"] == "dry_run"
    assert result["preflight"]["status"] == "passed"
    assert result["would_call_staged"] is True
    assert result["codex_staged_result"] is None
    assert result["codex_packet_review"] is None


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
        _git(isolated, "init")
        _git(isolated, "config", "user.email", "test@example.com")
        _git(isolated, "config", "user.name", "Test User")
        _git(isolated, "add", "README.md")
        _git(isolated, "commit", "-m", "isolated base")
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


def test_dirty_recovery_reports_conservative_ownership_classes(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    cache_path = repo / ".pytest_cache" / "v" / "cache" / "nodeids"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("cached\n", encoding="utf-8")
    source_path = repo / "tools" / "dirty_tool.py"
    source_path.parent.mkdir()
    source_path.write_text("dirty source\n", encoding="utf-8")
    docs_path = repo / "docs" / "plan.md"
    docs_path.parent.mkdir()
    docs_path.write_text("dirty docs\n", encoding="utf-8")
    unknown_path = repo / "scratch.tmp"
    unknown_path.write_text("dirty unknown\n", encoding="utf-8")

    monkeypatch.setattr(
        workflow.staged,
        "codex_staged_implement",
        lambda args: (_ for _ in ()).throw(AssertionError("dirty recovery must not call staged")),
    )

    result = _call(workdir=str(repo))
    ownership = result["dirty_recovery"]["dirty_ownership"]

    assert set(ownership) == {
        "current_session",
        "other_known_session",
        "unknown_unowned",
        "generated_cache",
        "review_artifact_current_session",
        "dangerous_conflict",
    }
    assert ownership["generated_cache"] == [".pytest_cache/v/cache/nodeids"]
    assert "tools/dirty_tool.py" in ownership["unknown_unowned"]
    assert "docs/plan.md" in ownership["unknown_unowned"]
    assert "scratch.tmp" in ownership["unknown_unowned"]
    assert ownership["current_session"] == []
    assert ownership["review_artifact_current_session"] == []


def test_dangerous_cache_status_is_not_generated_cache(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    cache_path = repo / ".pytest_cache" / "v" / "cache" / "nodeids"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("cached\n", encoding="utf-8")
    dirty = workflow.staged._dirty_check(repo)
    dirty["unsafe_reasons"] = ["delete_status"]

    monkeypatch.setattr(workflow.staged, "_dirty_check", lambda repo_arg: dirty)

    result = _call(workdir=str(repo), standing_authorization=True, auto_clean_cache=True)
    ownership = result["dirty_recovery"]["dirty_ownership"]

    assert result["status"] == "dirty_recovery_required"
    assert ".pytest_cache/v/cache/nodeids" in ownership["dangerous_conflict"]
    assert ownership["generated_cache"] == []
    assert result["dirty_recovery"]["cleanup_allowed"] is False
    assert result["dirty_recovery"]["would_mutate_paths"] == []
    assert result["dirty_recovery"]["cache_cleaned_paths"] == []
    assert cache_path.exists()


def test_global_unsafe_reason_blocks_all_cache_cleanup(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    cache_path = repo / ".pytest_cache" / "v" / "cache" / "nodeids"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("cached\n", encoding="utf-8")
    source_path = repo / "tools" / "dirty_tool.py"
    source_path.parent.mkdir()
    source_path.write_text("dirty source\n", encoding="utf-8")
    dirty = workflow.staged._dirty_check(repo)
    dirty["unsafe_reasons"] = ["secret_path_evidence"]

    monkeypatch.setattr(workflow.staged, "_dirty_check", lambda repo_arg: dirty)
    monkeypatch.setattr(
        workflow.staged,
        "codex_staged_implement",
        lambda args: (_ for _ in ()).throw(AssertionError("unsafe dirty must not call staged")),
    )

    result = _call(workdir=str(repo), standing_authorization=True, auto_clean_cache=True)
    ownership = result["dirty_recovery"]["dirty_ownership"]

    assert result["status"] == "dirty_recovery_required"
    assert ".pytest_cache/v/cache/nodeids" in ownership["dangerous_conflict"]
    assert "tools/dirty_tool.py" in ownership["dangerous_conflict"]
    assert ownership["generated_cache"] == []
    assert result["dirty_recovery"]["cleanup_allowed"] is False
    assert result["dirty_recovery"]["cache_cleaned_paths"] == []
    assert cache_path.exists()
    assert source_path.exists()


def test_mixed_cache_and_unknown_dirty_does_not_partially_clean_cache(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    cache_path = repo / ".pytest_cache" / "v" / "cache" / "nodeids"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("cached\n", encoding="utf-8")
    unknown_path = repo / "scratch.tmp"
    unknown_path.write_text("dirty\n", encoding="utf-8")

    monkeypatch.setattr(
        workflow.staged,
        "codex_staged_implement",
        lambda args: (_ for _ in ()).throw(AssertionError("mixed dirty must not call staged")),
    )

    result = _call(workdir=str(repo), standing_authorization=True, auto_clean_cache=True)

    assert result["status"] == "dirty_recovery_required"
    assert result["dirty_recovery"]["cleanup_allowed"] is False
    assert result["dirty_recovery"]["cache_cleanup_preview_allowed"] is False
    assert result["dirty_recovery"]["cache_cleaned_paths"] == []
    assert cache_path.exists()
    assert unknown_path.exists()


def test_plan_paths_without_marker_are_unknown_unowned(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    hermes_plan_path = repo / ".hermes" / "plans" / "foo.md"
    hermes_plan_path.parent.mkdir(parents=True)
    hermes_plan_path.write_text("plan\n", encoding="utf-8")
    docs_plan_path = repo / "docs" / "plans" / "bar.md"
    docs_plan_path.parent.mkdir(parents=True)
    docs_plan_path.write_text("plan\n", encoding="utf-8")

    monkeypatch.setattr(
        workflow.staged,
        "codex_staged_implement",
        lambda args: (_ for _ in ()).throw(AssertionError("dirty recovery must not call staged")),
    )

    result = _call(workdir=str(repo))
    ownership = result["dirty_recovery"]["dirty_ownership"]

    assert ".hermes/plans/foo.md" in ownership["unknown_unowned"]
    assert "docs/plans/bar.md" in ownership["unknown_unowned"]
    assert ownership["review_artifact_current_session"] == []


def test_secret_data_binary_or_large_file_evidence_is_dangerous(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    secret_path = repo / "secrets.json"
    secret_path.write_text("secret-ish\n", encoding="utf-8")
    dirty = workflow.staged._dirty_check(repo)
    dirty["unsafe_reasons"] = ["secret_path_evidence"]

    monkeypatch.setattr(workflow.staged, "_dirty_check", lambda repo_arg: dirty)

    result = _call(workdir=str(repo), standing_authorization=True, auto_clean_cache=True)
    ownership = result["dirty_recovery"]["dirty_ownership"]

    assert "secrets.json" in ownership["dangerous_conflict"]
    assert result["dirty_recovery"]["cleanup_allowed"] is False
    assert result["dirty_recovery"]["would_mutate_paths"] == []


def test_unknown_unsafe_reason_blocks_cleanup_even_without_dangerous_class(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    cache_path = repo / ".pytest_cache" / "v" / "cache" / "nodeids"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("cached\n", encoding="utf-8")
    dirty = workflow.staged._dirty_check(repo)
    dirty["unsafe_reasons"] = ["future_unscoped_unsafe_reason"]

    monkeypatch.setattr(workflow.staged, "_dirty_check", lambda repo_arg: dirty)
    monkeypatch.setattr(
        workflow.staged,
        "codex_staged_implement",
        lambda args: (_ for _ in ()).throw(AssertionError("unsafe dirty must not call staged")),
    )

    result = _call(workdir=str(repo), standing_authorization=True, auto_clean_cache=True)

    assert result["status"] == "dirty_recovery_required"
    assert result["dirty_recovery"]["cleanup_allowed"] is False
    assert result["dirty_recovery"]["cache_cleanup_preview_allowed"] is False
    assert result["dirty_recovery"]["cache_cleaned_paths"] == []
    assert result["dirty_recovery"]["would_mutate_paths"] == []
    assert cache_path.exists()


def test_dry_run_cache_only_reports_cleanup_preview_but_keeps_cache(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    cache_path = repo / ".pytest_cache" / "v" / "cache" / "nodeids"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("cached\n", encoding="utf-8")
    before_hash = _sha256(cache_path)
    staged_calls = []

    def fake_staged(args):
        staged_calls.append(args)
        raise AssertionError("dry_run must not call staged implementation")

    monkeypatch.setattr(workflow.staged, "codex_staged_implement", fake_staged)

    result = _call(workdir=str(repo), mode="dry_run", standing_authorization=True, auto_clean_cache=True)

    assert result["status"] == "dry_run"
    assert result["dirty_recovery"]["cleanup_allowed"] is False
    assert result["dirty_recovery"]["cache_cleanup_preview_allowed"] is True
    assert result["dirty_recovery"]["would_cleanup_cache_in_execute"] is True
    assert result["dirty_recovery"]["would_mutate_paths"] == []
    assert cache_path.exists()
    assert _sha256(cache_path) == before_hash
    assert staged_calls == []


def test_isolated_worktree_preserves_original_dirty_hashes(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    source_path = repo / "tools" / "dirty_tool.py"
    source_path.parent.mkdir()
    source_path.write_text("dirty source\n", encoding="utf-8")
    docs_path = repo / "docs" / "plan.md"
    docs_path.parent.mkdir()
    docs_path.write_text("dirty docs\n", encoding="utf-8")
    unknown_path = repo / "scratch.tmp"
    unknown_path.write_text("dirty unknown\n", encoding="utf-8")
    cache_path = repo / ".pytest_cache" / "v" / "cache" / "nodeids"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("cached\n", encoding="utf-8")
    original_hashes = {path: _sha256(path) for path in (source_path, docs_path, unknown_path, cache_path)}
    isolated = tmp_path / ".hermes-worktrees" / "repo-p2-abc123"

    def fake_create(repo_arg, *, stage_id, git_head):
        isolated.mkdir(parents=True)
        (isolated / "README.md").write_text("hello\n", encoding="utf-8")
        _git(isolated, "init")
        _git(isolated, "config", "user.email", "test@example.com")
        _git(isolated, "config", "user.name", "Test User")
        _git(isolated, "add", "README.md")
        _git(isolated, "commit", "-m", "isolated base")
        return {"path": str(isolated), "branch": "work/p2-abc123", "source_head": git_head}

    def fake_staged(args):
        return json.dumps({"status": "ready_for_review", "resolved_workdir": args["workdir"]})

    monkeypatch.setattr(workflow, "_create_isolated_worktree", fake_create)
    monkeypatch.setattr(workflow.staged, "codex_staged_implement", fake_staged)

    result = _call(
        workdir=str(repo),
        standing_authorization=True,
        auto_clean_cache=True,
        allow_isolated_worktree=True,
        stage_id="p2",
    )

    assert result["status"] == "staged_called"
    assert result["dirty_recovery"]["strategy"] == "isolated_worktree"
    assert result["dirty_recovery"]["cache_cleaned_paths"] == []
    for path, before_hash in original_hashes.items():
        assert path.exists()
        assert _sha256(path) == before_hash


def test_preflight_blocked_flags_do_not_mutate_dirty_recovery(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    cache_path = repo / ".pytest_cache" / "v" / "cache" / "nodeids"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("cached\n", encoding="utf-8")
    before_hash = _sha256(cache_path)
    worktree_calls = []

    def fake_create(*args, **kwargs):
        worktree_calls.append((args, kwargs))
        raise AssertionError("blocked preflight must not create isolated worktree")

    monkeypatch.setattr(workflow, "_codex_preflight", lambda repo: _preflight("blocked", blockers=["sandbox_not_verified"]))
    monkeypatch.setattr(workflow, "_create_isolated_worktree", fake_create)
    monkeypatch.setattr(
        workflow.staged,
        "codex_staged_implement",
        lambda args: (_ for _ in ()).throw(AssertionError("blocked preflight must not call staged")),
    )

    result = _call(
        workdir=str(repo),
        standing_authorization=True,
        auto_clean_cache=True,
        allow_isolated_worktree=True,
        mode="execute",
    )

    assert result["status"] == "preflight_blocked"
    assert result["dirty_recovery"]["cleanup_allowed"] is False
    assert result["dirty_recovery"]["cache_cleaned_paths"] == []
    assert result["dirty_recovery"]["would_create_isolated_worktree"] is False
    assert cache_path.exists()
    assert _sha256(cache_path) == before_hash
    assert worktree_calls == []


def test_dirty_recovery_metadata_has_no_destructive_option_words(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    (repo / "scratch.tmp").write_text("dirty\n", encoding="utf-8")

    monkeypatch.setattr(
        workflow.staged,
        "codex_staged_implement",
        lambda args: (_ for _ in ()).throw(AssertionError("dirty recovery must not call staged")),
    )

    result = _call(workdir=str(repo))
    text = "\n".join(_strings(result["dirty_recovery"])).lower()

    for banned in ("stash", "reset", "revert", "drop", "overwrite", "force-push", "force push", "deploy", "restart"):
        assert banned not in text
    assert "remove dirty files" not in text


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
    monkeypatch.setattr(
        workflow,
        "_run_packet_only_review",
        lambda **kwargs: _packet_review("packet_only_unusable", must_fix_count=None),
        raising=False,
    )

    result = _call(workdir=str(repo))

    assert result["status"] == "staged_review_unavailable"
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


def _packet_review(status: str, *, must_fix_count: int | None = 0, summary: str = "review summary") -> dict:
    return {
        "status": status,
        "reason": None,
        "must_fix_count": must_fix_count,
        "final_judgment": "可以继续" if status == "packet_only_passed" else "需要先修",
        "summary": summary,
        "raw_log_path": "/tmp/codex-review.raw.log",
        "final_file": "/tmp/codex-review.final.txt",
    }


def test_auto_packet_review_passes_ready_candidate(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)
    review_calls = []

    def fake_staged(args):
        (Path(args["workdir"]) / "README.md").write_text("hello\nreviewed\n", encoding="utf-8")
        return json.dumps(
            {
                "status": "ready_for_review",
                "candidate_id": "cand-p0",
                "candidate_disposition": "pending_review",
                "completion_trusted": True,
            }
        )

    def fake_review(**kwargs):
        review_calls.append(kwargs)
        return _packet_review("packet_only_passed", must_fix_count=0)

    monkeypatch.setattr(workflow.staged, "codex_staged_implement", fake_staged)
    monkeypatch.setattr(workflow, "_run_packet_only_review", fake_review, raising=False)

    result = _call(workdir=str(repo))

    assert result["status"] == "staged_reviewed"
    assert result["preflight"]["status"] == "passed"
    assert result["codex_packet_review"]["status"] == "packet_only_passed"
    assert result["codex_packet_review"]["must_fix_count"] == 0
    assert result["leftover_candidate"]["requires_review"] is False
    assert result["leftover_candidate"]["requires_fixes"] is False
    assert result["leftover_candidate"]["requires_hermes_verification"] is True
    assert result["leftover_candidate"]["packet_review_status"] == "packet_only_passed"
    assert review_calls[0]["repo"] == repo
    assert review_calls[0]["touched_files"] == ["README.md"]
    assert review_calls[0]["dirty_baseline_paths"] == []
    assert "dirty" not in review_calls[0]


def test_auto_packet_review_blocks_on_must_fix(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)

    def fake_staged(args):
        (Path(args["workdir"]) / "README.md").write_text("hello\nblocked\n", encoding="utf-8")
        return json.dumps({"status": "review_needed", "candidate_id": "cand-blocked"})

    monkeypatch.setattr(workflow.staged, "codex_staged_implement", fake_staged)
    monkeypatch.setattr(
        workflow,
        "_run_packet_only_review",
        lambda **kwargs: _packet_review("packet_only_failed", must_fix_count=1),
        raising=False,
    )

    result = _call(workdir=str(repo))

    assert result["status"] == "staged_review_blocked"
    assert result["codex_packet_review"]["status"] == "packet_only_failed"
    assert result["codex_packet_review"]["must_fix_count"] == 1
    assert result["leftover_candidate"]["requires_review"] is False
    assert result["leftover_candidate"]["requires_fixes"] is True
    assert result["leftover_candidate"]["requires_hermes_verification"] is True


def test_auto_packet_review_unusable_fail_closed(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)

    def fake_staged(args):
        (Path(args["workdir"]) / "README.md").write_text("hello\nunusable\n", encoding="utf-8")
        return json.dumps({"status": "ready_for_review", "candidate_id": "cand-unusable"})

    monkeypatch.setattr(workflow.staged, "codex_staged_implement", fake_staged)
    monkeypatch.setattr(
        workflow,
        "_run_packet_only_review",
        lambda **kwargs: _packet_review("packet_only_unusable", must_fix_count=None),
        raising=False,
    )

    result = _call(workdir=str(repo))

    assert result["status"] == "staged_review_unavailable"
    assert result["codex_packet_review"]["status"] == "packet_only_unusable"
    assert result["leftover_candidate"]["requires_review"] is True
    assert result["leftover_candidate"]["requires_fixes"] is False
    assert result["leftover_candidate"]["requires_hermes_verification"] is True


def test_takeover_candidate_review_pass_does_not_become_trusted_completion(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)

    def fake_staged(args):
        (Path(args["workdir"]) / "README.md").write_text("hello\ntakeover\n", encoding="utf-8")
        return json.dumps(
            {
                "status": "takeover_candidate",
                "candidate_id": "cand-takeover",
                "candidate_disposition": "takeover_required",
                "completion_trusted": False,
            }
        )

    monkeypatch.setattr(workflow.staged, "codex_staged_implement", fake_staged)
    monkeypatch.setattr(
        workflow,
        "_run_packet_only_review",
        lambda **kwargs: _packet_review("packet_only_passed", must_fix_count=0),
        raising=False,
    )

    result = _call(workdir=str(repo))

    assert result["status"] == "staged_reviewed"
    assert result["codex_staged_result"]["status"] == "takeover_candidate"
    assert result["leftover_candidate"]["completion_trusted"] is False
    assert result["leftover_candidate"]["candidate_disposition"] == "takeover_required"
    assert result["leftover_candidate"]["requires_hermes_verification"] is True
    assert "approved" not in json.dumps(result)


def test_blocked_staged_status_does_not_run_packet_review(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)

    def fake_staged(args):
        (Path(args["workdir"]) / "README.md").write_text("hello\nblocked by allowlist\n", encoding="utf-8")
        return json.dumps({"status": "blocked_by_allowlist", "candidate_id": "cand-blocked"})

    def fake_review(**kwargs):
        raise AssertionError("blocked staged status must not run packet review")

    monkeypatch.setattr(workflow.staged, "codex_staged_implement", fake_staged)
    monkeypatch.setattr(workflow, "_run_packet_only_review", fake_review, raising=False)

    result = _call(workdir=str(repo))

    assert result["status"] == "staged_called"
    assert result["codex_packet_review"]["status"] == "not_run"
    assert result["codex_packet_review"]["reason"] == "staged_status_blocked_by_allowlist"
    assert result["leftover_candidate"]["requires_review"] is True


def test_checkpoint_path_does_not_auto_review_or_commit_without_evidence(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)

    def fake_staged(args):
        (Path(args["workdir"]) / "README.md").write_text("hello\ncheckpoint\n", encoding="utf-8")
        return json.dumps({"status": "ready_for_review", "candidate_id": "cand-checkpoint"})

    def fake_review(**kwargs):
        raise AssertionError("checkpoint path must not run packet review")

    monkeypatch.setattr(workflow.staged, "codex_staged_implement", fake_staged)
    monkeypatch.setattr(workflow, "_run_packet_only_review", fake_review, raising=False)

    result = _call(workdir=str(repo), standing_authorization=True, checkpoint_verified_diff=True)

    assert result["status"] == "checkpoint_blocked"
    assert result["checkpoint"]["reason"] == "missing_verification_evidence"


def test_dry_run_never_runs_packet_review(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)

    def fake_review(**kwargs):
        raise AssertionError("dry_run must not run packet review")

    monkeypatch.setattr(workflow, "_run_packet_only_review", fake_review, raising=False)

    result = _call(workdir=str(repo), mode="dry_run")

    assert result["status"] == "dry_run"
    assert result["codex_packet_review"] is None


def test_review_output_is_bounded(tmp_path, monkeypatch):
    repo = _clean_repo(tmp_path)

    def fake_staged(args):
        (Path(args["workdir"]) / "README.md").write_text("hello\nbounded\n", encoding="utf-8")
        return json.dumps({"status": "ready_for_review", "candidate_id": "cand-bounded"})

    def fake_review(**kwargs):
        result = _packet_review("packet_only_passed", must_fix_count=0)
        result["raw_review_text"] = "diff --git a/README.md b/README.md\n" + "source line\n" * 100
        result["full_diff"] = "@@ should not be returned @@"
        return result

    monkeypatch.setattr(workflow.staged, "codex_staged_implement", fake_staged)
    monkeypatch.setattr(workflow, "_run_packet_only_review", fake_review, raising=False)

    result = _call(workdir=str(repo))
    encoded = json.dumps(result)

    assert result["codex_packet_review"]["status"] == "packet_only_passed"
    assert "diff --git" not in encoded
    assert "@@ should not be returned @@" not in encoded


def test_malformed_structured_review_fails_closed():
    malformed_reviews = [
        {"status": "passed", "review": {}},
        {
            "status": "passed",
            "review": {
                "verdict": "passed",
                "must_fix": "not-a-list",
                "final_judgment": "可以继续",
                "summary": "bad must_fix type",
            },
        },
        {
            "status": "passed",
            "review": {
                "verdict": "unknown",
                "must_fix": [],
                "final_judgment": "可以继续",
                "summary": "bad verdict",
            },
        },
        {
            "status": "passed",
            "review": {
                "verdict": "passed",
                "must_fix": [],
                "summary": "missing final judgment",
            },
        },
        {
            "status": "passed",
            "review": {
                "verdict": "passed",
                "must_fix": [],
                "final_judgment": "不能继续",
                "summary": "contradictory final judgment",
            },
        },
        {
            "status": "passed",
            "review": {
                "verdict": "failed",
                "must_fix": [],
                "final_judgment": "可以继续",
                "summary": "contradictory failed verdict",
            },
        },
        {
            "status": "passed",
            "review": {
                "verdict": "failed",
                "must_fix": [],
                "final_judgment": "需要先修",
                "summary": "failed verdict without actionable blockers",
            },
        },
    ]

    for raw in malformed_reviews:
        normalized = workflow._normalize_packet_review_result(raw)

        assert normalized["status"] == "packet_only_unusable"
        assert normalized["reason"] == "review_guard_schema_invalid"


def test_direct_packet_review_passed_shape_is_validated():
    malformed_passes = [
        {"status": "packet_only_passed"},
        {
            "status": "packet_only_passed",
            "must_fix_count": 1,
            "final_judgment": "可以继续",
            "summary": "contradictory pass",
        },
        {
            "status": "packet_only_passed",
            "must_fix_count": 0,
            "final_judgment": "需要先修",
            "summary": "contradictory judgment",
        },
        {
            "status": "packet_only_passed",
            "must_fix_count": 0,
            "final_judgment": "可以继续",
            "summary": "",
        },
    ]

    for raw in malformed_passes:
        normalized = workflow._normalize_packet_review_result(raw)

        assert normalized["status"] == "packet_only_unusable"
        assert normalized["reason"] == "review_guard_schema_invalid"


def test_direct_packet_review_failed_requires_must_fix_count():
    normalized = workflow._normalize_packet_review_result(
        {
            "status": "packet_only_failed",
            "must_fix_count": 0,
            "final_judgment": "需要先修",
            "summary": "failed without blockers",
        }
    )

    assert normalized["status"] == "packet_only_unusable"
    assert normalized["reason"] == "review_guard_schema_invalid"


def test_packet_review_summary_omits_diff_like_content():
    summarized = workflow._summarize_packet_review(
        {
            "status": "packet_only_passed",
            "must_fix_count": 0,
            "final_judgment": "可以继续",
            "summary": "diff --git a/file.py b/file.py\n@@ giant source excerpt @@",
            "raw_log_path": "/tmp/raw.log",
            "final_file": "/tmp/final.txt",
        }
    )

    assert summarized["status"] == "packet_only_passed"
    assert "diff --git" not in summarized["summary"]
    assert "@@" not in summarized["summary"]


def test_packet_review_reason_omits_diff_like_content():
    summarized = workflow._summarize_packet_review(
        {
            "status": "packet_only_unusable",
            "reason": "diff --git a/file.py b/file.py\n@@ giant source excerpt @@",
            "summary": "review unavailable",
            "raw_log_path": "/tmp/raw.log",
            "final_file": "/tmp/final.txt",
        }
    )
    encoded = json.dumps(summarized)

    assert summarized["status"] == "packet_only_unusable"
    assert "diff --git" not in encoded
    assert "@@" not in encoded
