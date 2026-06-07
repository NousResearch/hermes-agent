from pathlib import Path

from agent import codex_workflow_provenance as provenance


def test_other_session_dirty_is_preserved(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    path = repo / "README.md"
    path.write_text("owned elsewhere\n", encoding="utf-8")
    event = provenance.provenance_event(
        repo=repo,
        branch="main",
        head_sha="abc123",
        stage_id="phase12a0",
        session_id="session-b",
        actor="hermes",
        tool="codex_workflow_run",
        operation="modify",
        path="README.md",
        before_hash=None,
        after_hash=provenance.file_hash(path),
    )

    ownership = provenance.classify_path(
        repo=repo,
        rel_path="README.md",
        current_session_id="session-a",
        branch="main",
        head_sha="abc123",
        events=[event],
    )

    assert ownership["owner_policy"] == "other_known_session"
    assert ownership["default_behavior"] == "preserve"
    assert ownership["cleanup_allowed"] is False


def test_cleanup_owner_mismatch_blocks_delete(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    path = repo / ".pytest_cache" / "nodeids"
    path.parent.mkdir()
    path.write_text("cache\n", encoding="utf-8")
    event = provenance.provenance_event(
        repo=repo,
        branch="main",
        head_sha="abc123",
        stage_id="phase12a0",
        session_id="other-session",
        actor="hermes",
        tool="codex_workflow_run",
        operation="create",
        path=".pytest_cache/nodeids",
        before_hash=None,
        after_hash=provenance.file_hash(path),
    )

    decision = provenance.cleanup_decision(
        repo=repo,
        rel_path=".pytest_cache/nodeids",
        current_session_id="current-session",
        branch="main",
        head_sha="abc123",
        events=[event],
        allowed_files=[],
        allowed_globs=[".pytest_cache/**"],
        explicit_authorization=True,
        operation="delete",
    )

    assert decision["cleanup_allowed"] is False
    assert "owner_policy_not_current_session" in decision["blocking_reasons"]
    assert path.exists()


def test_delete_hash_mismatch_blocks_delete(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    path = repo / ".pytest_cache" / "nodeids"
    path.parent.mkdir()
    path.write_text("cache v1\n", encoding="utf-8")
    event = provenance.provenance_event(
        repo=repo,
        branch="main",
        head_sha="abc123",
        stage_id="phase12a0",
        session_id="current-session",
        actor="hermes",
        tool="codex_workflow_run",
        operation="create",
        path=".pytest_cache/nodeids",
        before_hash=None,
        after_hash=provenance.file_hash(path),
    )
    path.write_text("cache v2\n", encoding="utf-8")

    decision = provenance.cleanup_decision(
        repo=repo,
        rel_path=".pytest_cache/nodeids",
        current_session_id="current-session",
        branch="main",
        head_sha="abc123",
        events=[event],
        allowed_files=[],
        allowed_globs=[".pytest_cache/**"],
        explicit_authorization=True,
        operation="delete",
    )

    assert decision["owner_policy"] == "unknown_unowned"
    assert decision["cleanup_allowed"] is False
    assert "hash_mismatch" in decision["blocking_reasons"]
    assert path.exists()


def test_review_artifact_cleanup_does_not_delete_foreign_doc(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    path = repo / "docs" / "plans" / "review.md"
    path.parent.mkdir(parents=True)
    path.write_text("foreign doc\n", encoding="utf-8")
    event = provenance.provenance_event(
        repo=repo,
        branch="main",
        head_sha="abc123",
        stage_id="phase12a0",
        session_id="other-session",
        actor="review_guard",
        tool="codex_review_guard",
        operation="review_artifact",
        path="docs/plans/review.md",
        before_hash=None,
        after_hash=provenance.file_hash(path),
    )

    decision = provenance.cleanup_decision(
        repo=repo,
        rel_path="docs/plans/review.md",
        current_session_id="current-session",
        branch="main",
        head_sha="abc123",
        events=[event],
        allowed_files=["docs/plans/review.md"],
        allowed_globs=[],
        explicit_authorization=True,
        operation="delete",
    )

    assert decision["cleanup_allowed"] is False
    assert decision["owner_policy"] == "other_known_session"
    assert "docs_plans_default_preserve" in decision["blocking_reasons"]
    assert path.exists()


def test_missing_session_id_degrades_to_unknown_unowned(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    path = repo / "README.md"
    path.write_text("unknown\n", encoding="utf-8")

    ownership = provenance.classify_path(
        repo=repo,
        rel_path="README.md",
        current_session_id=None,
        branch="main",
        head_sha="abc123",
        events=[],
    )

    assert ownership["owner_policy"] == "unknown_unowned"
    assert "missing_session_id" in ownership["blocking_reasons"]


def test_docs_plans_default_preserve_without_current_session_artifact_proof(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    path = repo / "docs" / "plans" / "phase.md"
    path.parent.mkdir(parents=True)
    path.write_text("plan\n", encoding="utf-8")

    decision = provenance.cleanup_decision(
        repo=repo,
        rel_path="docs/plans/phase.md",
        current_session_id="current-session",
        branch="main",
        head_sha="abc123",
        events=[],
        allowed_files=["docs/plans/phase.md"],
        allowed_globs=[],
        explicit_authorization=True,
        operation="delete",
    )

    assert decision["owner_policy"] == "unknown_unowned"
    assert decision["cleanup_allowed"] is False
    assert "docs_plans_default_preserve" in decision["blocking_reasons"]
    assert path.exists()


def test_repo_id_mismatch_blocks_current_session_ownership(tmp_path):
    repo = tmp_path / "repo"
    other_repo = tmp_path / "other"
    repo.mkdir()
    other_repo.mkdir()
    path = repo / ".pytest_cache" / "nodeids"
    path.parent.mkdir()
    path.write_text("cache\n", encoding="utf-8")
    event = provenance.provenance_event(
        repo=other_repo,
        branch="main",
        head_sha="abc123",
        stage_id="phase12a0",
        session_id="current-session",
        actor="hermes",
        tool="codex_workflow_run",
        operation="create",
        path=".pytest_cache/nodeids",
        before_hash=None,
        after_hash=provenance.file_hash(path),
    )

    decision = provenance.cleanup_decision(
        repo=repo,
        rel_path=".pytest_cache/nodeids",
        current_session_id="current-session",
        branch="main",
        head_sha="abc123",
        events=[event],
        allowed_files=[],
        allowed_globs=[".pytest_cache/**"],
        explicit_authorization=True,
        operation="delete",
    )

    assert decision["owner_policy"] == "unknown_unowned"
    assert decision["cleanup_allowed"] is False
    assert "repo_id_mismatch" in decision["blocking_reasons"]


def test_current_session_cleanup_requires_allowlist_and_authorization(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    path = repo / ".pytest_cache" / "nodeids"
    path.parent.mkdir()
    path.write_text("cache\n", encoding="utf-8")
    event = provenance.provenance_event(
        repo=repo,
        branch="main",
        head_sha="abc123",
        stage_id="phase12a0",
        session_id="current-session",
        actor="hermes",
        tool="codex_workflow_run",
        operation="create",
        path=".pytest_cache/nodeids",
        before_hash=None,
        after_hash=provenance.file_hash(path),
    )

    not_authorized = provenance.cleanup_decision(
        repo=repo,
        rel_path=".pytest_cache/nodeids",
        current_session_id="current-session",
        branch="main",
        head_sha="abc123",
        events=[event],
        allowed_files=[],
        allowed_globs=[".pytest_cache/**"],
        explicit_authorization=False,
        operation="delete",
    )
    not_allowlisted = provenance.cleanup_decision(
        repo=repo,
        rel_path=".pytest_cache/nodeids",
        current_session_id="current-session",
        branch="main",
        head_sha="abc123",
        events=[event],
        allowed_files=[],
        allowed_globs=[],
        explicit_authorization=True,
        operation="delete",
    )
    allowed = provenance.cleanup_decision(
        repo=repo,
        rel_path=".pytest_cache/nodeids",
        current_session_id="current-session",
        branch="main",
        head_sha="abc123",
        events=[event],
        allowed_files=[],
        allowed_globs=[".pytest_cache/**"],
        explicit_authorization=True,
        operation="delete",
    )

    assert not_authorized["cleanup_allowed"] is False
    assert "authorization_required" in not_authorized["blocking_reasons"]
    assert not_allowlisted["cleanup_allowed"] is False
    assert "path_not_in_allowlist" in not_allowlisted["blocking_reasons"]
    assert allowed["cleanup_allowed"] is True
