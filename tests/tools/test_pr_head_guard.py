"""Regression tests for the PR-head invariant gate."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

from agent.pr_head_guard import command_requires_pr_head_guard, enforce_pr_head_invariant, resolve_repo_root_for_path
from agent.runtime_evidence import clear_runtime_evidence, current_runtime_evidence, seed_turn_runtime_evidence
from tools.approval import check_all_command_guards

_REAL_RUN = subprocess.run


def _init_repo(tmp_path: Path, branch: str) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "README.md").write_text("seed\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=test@example.com", "-c", "user.name=Test", "commit", "-q", "-m", "seed"],
        cwd=repo,
        check=True,
    )
    subprocess.run(["git", "branch", "-M", branch], cwd=repo, check=True)
    head_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repo, head_sha


def _mock_gh_pr_view(number: int, branch: str, sha: str, repo_name: str):
    payload = json.dumps(
        {
            "number": number,
            "headRefName": branch,
            "headRefOid": sha,
            "repository": {"nameWithOwner": repo_name},
        }
    )

    def _run(cmd, *args, **kwargs):
        if isinstance(cmd, list) and cmd[:3] == ["gh", "pr", "view"]:
            return subprocess.CompletedProcess(cmd, 0, payload, "")
        return _REAL_RUN(cmd, *args, **kwargs)

    return _run


def _mock_gh_pr_view_failure():
    def _run(cmd, *args, **kwargs):
        if isinstance(cmd, list) and cmd[:3] == ["gh", "pr", "view"]:
            return subprocess.CompletedProcess(cmd, 1, "", "gh pr view failed")
        return _REAL_RUN(cmd, *args, **kwargs)

    return _run


def _mock_gh_pr_view_and_diff(
    number: int,
    branch: str,
    sha: str,
    repo_name: str,
    diff_text: str,
):
    payload = json.dumps(
        {
            "number": number,
            "headRefName": branch,
            "headRefOid": sha,
            "repository": {"nameWithOwner": repo_name},
        }
    )

    def _run(cmd, *args, **kwargs):
        if isinstance(cmd, list) and cmd[:3] == ["gh", "pr", "view"]:
            return subprocess.CompletedProcess(cmd, 0, payload, "")
        if isinstance(cmd, list) and cmd[:3] == ["gh", "pr", "diff"]:
            return subprocess.CompletedProcess(cmd, 0, diff_text, "")
        return _REAL_RUN(cmd, *args, **kwargs)

    return _run


def _mock_gh_pr_view_with_target(
    expected_target: str,
    number: int,
    branch: str,
    sha: str,
    repo_name: str,
    fallback_sha: str,
):
    matching_payload = json.dumps(
        {
            "number": number,
            "headRefName": branch,
            "headRefOid": sha,
            "repository": {"nameWithOwner": repo_name},
        }
    )
    fallback_payload = json.dumps(
        {
            "number": 99,
            "headRefName": "main",
            "headRefOid": fallback_sha,
            "repository": {"nameWithOwner": repo_name},
        }
    )

    def _run(cmd, *args, **kwargs):
        if isinstance(cmd, list) and cmd[:3] == ["gh", "pr", "view"]:
            target = cmd[3] if len(cmd) > 3 and not cmd[3].startswith("-") else ""
            payload = matching_payload if target == str(expected_target) else fallback_payload
            return subprocess.CompletedProcess(cmd, 0, payload, "")
        return _REAL_RUN(cmd, *args, **kwargs)

    return _run


def test_repo_root_resolution_finds_git_root(tmp_path):
    repo, _ = _init_repo(tmp_path, "feat/runtime-guardrails")
    assert resolve_repo_root_for_path(str(repo / "README.md")) == repo


def test_command_classifier_flags_pr_sensitive_actions():
    assert command_requires_pr_head_guard("git push origin HEAD")[0] is True
    assert command_requires_pr_head_guard("git -C /repo push origin HEAD")[0] is True
    assert command_requires_pr_head_guard("git -c core.sshCommand=ssh push origin HEAD")[0] is True
    assert command_requires_pr_head_guard("gh pr edit 17 --title foo")[0] is True
    assert command_requires_pr_head_guard("gh -R org/repo pr merge 17 --squash")[0] is True
    assert command_requires_pr_head_guard("gh pr checks 17")[0] is True
    assert command_requires_pr_head_guard("echo hello")[0] is False


def test_matching_live_pr_head_allows_pr_mutation(tmp_path, monkeypatch):
    repo, head_sha = _init_repo(tmp_path, "feat/runtime-guardrails")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    clear_runtime_evidence()

    with patch("agent.pr_head_guard.subprocess.run", side_effect=_mock_gh_pr_view(17, "feat/runtime-guardrails", head_sha, "org/repo")):
        result = check_all_command_guards(
            "gh pr edit 17 --add-label ready",
            "local",
            repo_root=str(repo),
        )

    assert result["approved"] is True
    evidence = current_runtime_evidence()
    assert evidence is not None
    assert evidence["pr_head_verified"] is True
    assert evidence["pr_number"] == 17
    assert evidence["live_pr_head_sha"] == head_sha
    clear_runtime_evidence()


def test_explicit_pr_target_is_used_for_live_head_evidence(tmp_path):
    repo, head_sha = _init_repo(tmp_path, "feat/runtime-guardrails")
    clear_runtime_evidence()

    with patch(
        "agent.pr_head_guard.subprocess.run",
        side_effect=_mock_gh_pr_view_with_target(
            "17",
            17,
            "feat/runtime-guardrails",
            head_sha,
            "org/repo",
            f"{head_sha[:-1]}0",
        ),
    ):
        result = check_all_command_guards(
            "gh -R org/repo pr merge 17 --squash",
            "local",
            repo_root=str(repo),
        )

    assert result["approved"] is True
    clear_runtime_evidence()


def test_explicit_pr_target_blocks_when_live_head_evidence_is_missing(tmp_path):
    repo, _ = _init_repo(tmp_path, "feat/runtime-guardrails")
    clear_runtime_evidence()

    with patch("agent.pr_head_guard.subprocess.run", side_effect=_mock_gh_pr_view_failure()):
        result = check_all_command_guards(
            "gh pr merge 17 --squash",
            "local",
            repo_root=str(repo),
        )

    assert result["approved"] is False
    assert "PR mutation" in result["message"]
    assert "verified live PR-head evidence is unavailable" in result["message"]
    clear_runtime_evidence()


def test_mismatched_local_head_blocks_pr_mutation(tmp_path, monkeypatch):
    repo, head_sha = _init_repo(tmp_path, "feat/runtime-guardrails")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    clear_runtime_evidence()

    with patch("agent.pr_head_guard.subprocess.run", side_effect=_mock_gh_pr_view(17, "feat/runtime-guardrails", f"{head_sha[:-1]}0", "org/repo")):
        result = check_all_command_guards(
            "gh pr merge 17 --squash",
            "local",
            repo_root=str(repo),
        )

    assert result["approved"] is False
    assert "local checkout does not match the live PR head" in result["message"]
    assert "PR: #17" in result["message"]
    clear_runtime_evidence()


def test_mismatched_local_head_blocks_ci_diagnosis(tmp_path):
    repo, head_sha = _init_repo(tmp_path, "feat/runtime-guardrails")
    clear_runtime_evidence()

    with patch("agent.pr_head_guard.subprocess.run", side_effect=_mock_gh_pr_view(17, "feat/runtime-guardrails", f"{head_sha[:-1]}0", "org/repo")):
        result = check_all_command_guards(
            "gh pr checks 17",
            "local",
            repo_root=str(repo),
        )

    assert result["approved"] is False
    assert "CI diagnosis" in result["message"]
    clear_runtime_evidence()


def test_mismatched_local_head_blocks_review_thread_resolution(tmp_path):
    repo, head_sha = _init_repo(tmp_path, "feat/runtime-guardrails")
    clear_runtime_evidence()

    command = "gh api graphql -f query='mutation { resolveReviewThread(input:{threadId:\"T123\"}) { clientMutationId } }'"
    with patch("agent.pr_head_guard.subprocess.run", side_effect=_mock_gh_pr_view(17, "feat/runtime-guardrails", f"{head_sha[:-1]}0", "org/repo")):
        result = check_all_command_guards(command, "local", repo_root=str(repo))

    assert result["approved"] is False
    assert "review-thread resolution" in result["message"]
    clear_runtime_evidence()


def test_missing_thread_id_blocks_review_thread_resolution(tmp_path):
    repo, head_sha = _init_repo(tmp_path, "feat/runtime-guardrails")
    clear_runtime_evidence()

    seed_turn_runtime_evidence(
        failing_file="src/app.py",
        failing_line=17,
    )
    with patch(
        "agent.pr_head_guard.subprocess.run",
        side_effect=_mock_gh_pr_view(17, "feat/runtime-guardrails", head_sha, "org/repo"),
    ):
        result = check_all_command_guards(
            "gh api graphql -f query='mutation { resolveReviewThread(input:{}) { clientMutationId } }'",
            "local",
            repo_root=str(repo),
        )

    assert result["approved"] is False
    assert "no review thread/comment ID is available" in result["message"]
    clear_runtime_evidence()


def test_missing_live_pr_head_evidence_blocks_review_thread_resolution(tmp_path):
    repo, _ = _init_repo(tmp_path, "feat/runtime-guardrails")
    clear_runtime_evidence()
    seed_turn_runtime_evidence(
        review_thread_comment_id="T123",
        failing_file="src/app.py",
        failing_line=17,
    )

    with patch("agent.pr_head_guard.subprocess.run", side_effect=_mock_gh_pr_view_failure()):
        result = check_all_command_guards(
            "gh api graphql -f query='mutation { resolveReviewThread(input:{threadId:\"T123\"}) { clientMutationId } }'",
            "local",
            repo_root=str(repo),
        )

    assert result["approved"] is False
    assert "verified live PR-head evidence is unavailable" in result["message"]
    clear_runtime_evidence()


def test_missing_ci_evidence_blocks_pr_ci_diagnosis(tmp_path):
    repo, head_sha = _init_repo(tmp_path, "feat/runtime-guardrails")
    clear_runtime_evidence()

    with patch(
        "agent.pr_head_guard.subprocess.run",
        side_effect=_mock_gh_pr_view(17, "feat/runtime-guardrails", head_sha, "org/repo"),
    ):
        result = check_all_command_guards(
            "gh pr checks 17",
            "local",
            repo_root=str(repo),
        )

    assert result["approved"] is False
    assert "missing CI evidence" in result["message"]
    clear_runtime_evidence()


def test_stale_ci_sha_blocks_diagnosis(tmp_path):
    repo, head_sha = _init_repo(tmp_path, "feat/runtime-guardrails")
    clear_runtime_evidence()
    seed_turn_runtime_evidence(
        ci_evidence_fetched_for_live_pr_head=True,
        ci_run_id="987654321",
        ci_checkout_sha=f"{head_sha[:-1]}0",
        ci_status="completed",
        ci_conclusion="failure",
        failing_file="src/app.py",
        failing_blob_sha="deadbeef",
        failing_line=17,
    )

    with patch(
        "agent.pr_head_guard.subprocess.run",
        side_effect=_mock_gh_pr_view(17, "feat/runtime-guardrails", head_sha, "org/repo"),
    ):
        result = check_all_command_guards(
            "gh pr checks 17",
            "local",
            repo_root=str(repo),
        )

    assert result["approved"] is False
    assert "CI checkout/head SHA does not match" in result["message"]
    clear_runtime_evidence()


def test_stale_ci_sha_blocks_rerun(tmp_path):
    repo, head_sha = _init_repo(tmp_path, "feat/runtime-guardrails")
    clear_runtime_evidence()
    seed_turn_runtime_evidence(
        ci_evidence_fetched_for_live_pr_head=True,
        ci_run_id="987654321",
        ci_checkout_sha=f"{head_sha[:-1]}0",
        ci_status="completed",
        ci_conclusion="failure",
        failing_file="src/app.py",
        failing_blob_sha="deadbeef",
        failing_line=17,
    )

    with patch(
        "agent.pr_head_guard.subprocess.run",
        side_effect=_mock_gh_pr_view(17, "feat/runtime-guardrails", head_sha, "org/repo"),
    ):
        result = check_all_command_guards(
            "gh run rerun 987654321",
            "local",
            repo_root=str(repo),
        )

    assert result["approved"] is False
    assert "CI checkout/head SHA does not match" in result["message"]
    clear_runtime_evidence()


def test_matching_live_pr_head_and_ci_sha_allows_diagnosis_and_rerun(tmp_path):
    repo, head_sha = _init_repo(tmp_path, "feat/runtime-guardrails")
    clear_runtime_evidence()
    seed_turn_runtime_evidence(
        ci_evidence_fetched_for_live_pr_head=True,
        ci_run_id="987654321",
        ci_checkout_sha=head_sha,
        ci_status="completed",
        ci_conclusion="failure",
        failing_file="src/app.py",
        failing_blob_sha="deadbeef",
        failing_line=17,
    )

    with patch(
        "agent.pr_head_guard.subprocess.run",
        side_effect=_mock_gh_pr_view(17, "feat/runtime-guardrails", head_sha, "org/repo"),
    ):
        diagnosis = check_all_command_guards(
            "gh pr checks 17",
            "local",
            repo_root=str(repo),
        )
        rerun = check_all_command_guards(
            "gh run rerun 987654321",
            "local",
            repo_root=str(repo),
        )

    assert diagnosis["approved"] is True
    assert rerun["approved"] is True
    evidence = current_runtime_evidence()
    assert evidence is not None
    assert evidence["ci_evidence_ci_sha_matches_live_pr_head"] is True
    clear_runtime_evidence()


def test_local_only_fix_blocks_review_thread_resolution(tmp_path):
    repo, head_sha = _init_repo(tmp_path, "feat/runtime-guardrails")
    clear_runtime_evidence()
    seed_turn_runtime_evidence(
        review_thread_comment_id="T123",
        failing_file="src/app.py",
        failing_line=17,
    )

    with patch(
        "agent.pr_head_guard.subprocess.run",
        side_effect=_mock_gh_pr_view(17, "feat/runtime-guardrails", f"{head_sha[:-1]}0", "org/repo"),
    ):
        result = check_all_command_guards(
            "gh api graphql -f query='mutation { resolveReviewThread(input:{threadId:\"T123\"}) { clientMutationId } }'",
            "local",
            repo_root=str(repo),
        )

    assert result["approved"] is False
    assert "local checkout does not match the live PR head" in result["message"]
    clear_runtime_evidence()


def test_fix_not_pushed_blocks_review_thread_resolution(tmp_path):
    repo, head_sha = _init_repo(tmp_path, "feat/runtime-guardrails")
    clear_runtime_evidence()
    seed_turn_runtime_evidence(
        review_thread_comment_id="T123",
        failing_file="src/app.py",
        failing_line=17,
    )

    with patch(
        "agent.pr_head_guard.subprocess.run",
        side_effect=_mock_gh_pr_view(17, "feat/runtime-guardrails", f"{head_sha[:-1]}0", "org/repo"),
    ):
        result = check_all_command_guards(
            "gh api graphql -f query='mutation { resolveReviewThread(input:{threadId:\"T123\"}) { clientMutationId } }'",
            "local",
            repo_root=str(repo),
        )

    assert result["approved"] is False
    assert "live PR head" in result["message"]
    clear_runtime_evidence()


def test_fetched_patch_still_contains_issue_blocks_resolution(tmp_path):
    repo, head_sha = _init_repo(tmp_path, "feat/runtime-guardrails")
    clear_runtime_evidence()
    seed_turn_runtime_evidence(
        review_thread_comment_id="T123",
        failing_file="src/app.py",
        failing_line=17,
        failing_blob_sha="deadbeef",
    )

    patch_text = """diff --git a/src/app.py b/src/app.py
index 1111111..2222222 100644
--- a/src/app.py
+++ b/src/app.py
@@ -15,4 +15,4 @@
 line 15
 line 16
-line 17
+line 17 fixed
 line 18
"""
    with patch(
        "agent.pr_head_guard.subprocess.run",
        side_effect=_mock_gh_pr_view_and_diff(17, "feat/runtime-guardrails", head_sha, "org/repo", patch_text),
    ):
        result = check_all_command_guards(
            "gh api graphql -f query='mutation { resolveReviewThread(input:{threadId:\"T123\"}) { clientMutationId } }'",
            "local",
            repo_root=str(repo),
        )

    assert result["approved"] is False
    assert "fetched PR patch still shows the reported issue" in result["message"]
    clear_runtime_evidence()


def test_live_pr_head_patch_allowing_resolution(tmp_path):
    repo, head_sha = _init_repo(tmp_path, "feat/runtime-guardrails")
    clear_runtime_evidence()
    seed_turn_runtime_evidence(
        review_thread_comment_id="T123",
        failing_file="src/app.py",
        failing_line=17,
        failing_blob_sha="deadbeef",
    )

    patch_text = """diff --git a/src/app.py b/src/app.py
index 1111111..2222222 100644
--- a/src/app.py
+++ b/src/app.py
@@ -1,4 +1,4 @@
 line 1
 line 2
-line 3
+line 3 fixed
 line 4
"""
    with patch(
        "agent.pr_head_guard.subprocess.run",
        side_effect=_mock_gh_pr_view_and_diff(17, "feat/runtime-guardrails", head_sha, "org/repo", patch_text),
    ):
        result = check_all_command_guards(
            "gh api graphql -f query='mutation { resolveReviewThread(input:{threadId:\"T123\"}) { clientMutationId } }'",
            "local",
            repo_root=str(repo),
        )

    assert result["approved"] is True
    evidence = current_runtime_evidence()
    assert evidence is not None
    assert evidence["review_thread_patch_clean"] is True
    assert evidence["fetched_pr_patch_checked"] is True
    clear_runtime_evidence()


def test_mismatched_local_head_blocks_file_writes(tmp_path):
    repo, head_sha = _init_repo(tmp_path, "feat/runtime-guardrails")
    clear_runtime_evidence()

    with patch("agent.pr_head_guard.subprocess.run", side_effect=_mock_gh_pr_view(17, "feat/runtime-guardrails", f"{head_sha[:-1]}0", "org/repo")):
        message = enforce_pr_head_invariant(repo_root=repo, action_label="file write")

    assert message is not None
    assert "file write" in message
    assert "Local HEAD" in message
    clear_runtime_evidence()


def test_stale_local_branch_alias_is_blocked(tmp_path):
    repo, head_sha = _init_repo(tmp_path, "stale-alias")
    clear_runtime_evidence()

    with patch("agent.pr_head_guard.subprocess.run", side_effect=_mock_gh_pr_view(17, "feat/runtime-guardrails", head_sha, "org/repo")):
        result = check_all_command_guards(
            "gh pr edit 17 --add-label ready",
            "local",
            repo_root=str(repo),
        )

    assert result["approved"] is False
    assert "Local branch: stale-alias" in result["message"]
    clear_runtime_evidence()


def test_unrelated_non_pr_commands_are_not_blocked(tmp_path):
    repo, _ = _init_repo(tmp_path, "feat/runtime-guardrails")
    clear_runtime_evidence()

    def _fail_if_gh(*args, **kwargs):
        if isinstance(args[0], list) and args[0][:3] == ["gh", "pr", "view"]:
            raise AssertionError("gh pr view should not be called for unrelated commands")
        return _REAL_RUN(*args, **kwargs)

    with patch("agent.pr_head_guard.subprocess.run", side_effect=_fail_if_gh):
        result = check_all_command_guards("echo hello", "local", repo_root=str(repo))

    assert result["approved"] is True
    assert current_runtime_evidence() is None
    clear_runtime_evidence()
