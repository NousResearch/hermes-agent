import datetime as _datetime
import fnmatch
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from agent import codex_workflow_ledger as ledger
from agent import codex_workflow_provenance as provenance
from tools import codex_staged_implement_tool as staged
from tools.registry import registry


_SUPPORTED_MODES = {"execute", "dry_run"}
_SUPPORTED_CONTINUE_POLICY = "stop-on-review-needed"
_SUPPORTED_DIRTY_POLICY = "require-clean"
_ALLOWED_VERIFY_IDS = {"diff-check", "none"}
_REVIEW_TIMEOUT_SECONDS = 900
_REVIEW_PACKET_LIMITS = {
    "max_stat_chars": 4_000,
    "max_name_chars": 4_000,
    "max_diff_chars": 30_000,
    "max_total_chars": 40_000,
}


def _json_result(payload: dict[str, Any]) -> str:
    return json.dumps(staged._bound(payload), ensure_ascii=False)


def _base(
    status: str,
    *,
    repo: Path | None = None,
    git_head: str | None = None,
    dirty: dict[str, Any] | None = None,
    codex_staged_result: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    result = {
        "status": status,
        "resolved_workdir": str(repo) if repo is not None else None,
        "git_head": git_head,
        "dirty_recovery": {
            "initial_dirty_check": dirty,
            "strategy": "none",
            "cache_cleaned_paths": [],
            "isolated_worktree": None,
        },
        "codex_staged_result": codex_staged_result,
    }
    result.update(extra)
    return result


def _validate_common(args: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    mode = args.get("mode", "execute")
    if mode not in _SUPPORTED_MODES:
        return None, _base("unsupported_mode", mode=mode)

    continue_policy = args.get("continue_policy", _SUPPORTED_CONTINUE_POLICY)
    if continue_policy != _SUPPORTED_CONTINUE_POLICY:
        return None, _base("rejected_policy", reason="unsupported_continue_policy")

    dirty_policy = args.get("dirty_baseline_policy", _SUPPORTED_DIRTY_POLICY)
    if dirty_policy != _SUPPORTED_DIRTY_POLICY:
        return None, _base("rejected_policy", reason="unsupported_dirty_baseline_policy")

    verify_ids = args.get("verify_cmd_ids", ["diff-check"])
    if verify_ids is None:
        verify_ids = ["diff-check"]
    if (
        not isinstance(verify_ids, list)
        or not verify_ids
        or any(not isinstance(item, str) or item not in _ALLOWED_VERIFY_IDS for item in verify_ids)
        or ("none" in verify_ids and verify_ids != ["none"])
    ):
        return None, _base("rejected_policy", reason="unsupported_verify_cmd_id")

    task = args.get("task")
    if not isinstance(task, str) or not task.strip():
        return None, _base("rejected_scope", reason="missing_task")

    repo, git_head = staged._resolve_repo_root(args.get("workdir"))
    if repo is None:
        return None, _base("rejected_repo", reason="invalid_workdir")

    allowlist, scope_error = staged._validate_scope(args, repo)
    if scope_error:
        return None, _base("rejected_scope", repo=repo, git_head=git_head, reason=scope_error)

    normalized = {
        "workdir": str(repo),
        "task": task.strip(),
        "allowed_files": allowlist["files"],
        "allowed_globs": allowlist["globs"],
        "verify_cmd_ids": verify_ids,
        "continue_policy": continue_policy,
        "dirty_baseline_policy": dirty_policy,
        "mode": mode,
    }
    return {
        "repo": repo,
        "git_head": git_head,
        "mode": mode,
        "normalized": normalized,
        "allowlist": allowlist,
    }, None


def _call_staged(normalized: dict[str, Any]) -> dict[str, Any]:
    payload = staged.codex_staged_implement(dict(normalized))
    try:
        decoded = json.loads(payload)
    except Exception:
        return {"status": "malformed", "raw": payload}
    return decoded if isinstance(decoded, dict) else {"status": "malformed", "raw": decoded}


def _review_packet_script() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "runtime" / "codex_review_packet.py"


def _review_guard_script() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "runtime" / "codex_review_guard.py"


def _run_review_command(argv: list[str], *, timeout: int = _REVIEW_TIMEOUT_SECONDS) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


def _current_dirty_paths(dirty: dict[str, Any]) -> list[str]:
    paths = dirty.get("dirty_paths")
    return list(paths) if isinstance(paths, list) else []


def _current_branch(repo: Path) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(repo), "branch", "--show-current"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _path_in_allowlist(path: str, allowlist: dict[str, list[str]]) -> bool:
    if path in allowlist.get("files", []):
        return True
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in allowlist.get("globs", []))


def _review_prompt() -> str:
    return (
        "Review this Hermes candidate implementation for correctness, regressions, and missing tests. "
        "Return only JSON matching the provided schema. Mark verdict failed if any must-fix issue remains."
    )


def _review_scope_files(
    *,
    dirty_before: dict[str, Any],
    dirty_after: dict[str, Any],
    allowlist: dict[str, list[str]],
) -> list[str]:
    before_paths = set(_current_dirty_paths(dirty_before))
    scoped: list[str] = []
    for path in _current_dirty_paths(dirty_after):
        if path in before_paths:
            continue
        if not _path_in_allowlist(path, allowlist):
            continue
        scoped.append(path)
    return sorted(scoped)


def _review_unavailable(reason: str, **extra: Any) -> dict[str, Any]:
    result = {
        "status": "unavailable",
        "verdict": "unavailable",
        "reason": reason,
        "passed": False,
    }
    result.update(extra)
    return result


def _review_failed(reason: str, **extra: Any) -> dict[str, Any]:
    result = {
        "status": "failed",
        "verdict": "failed",
        "reason": reason,
        "passed": False,
    }
    result.update(extra)
    return result


def _review_has_guard_metadata(guard: dict[str, Any]) -> bool:
    if guard.get("terminated_by_guard") or guard.get("process_exited_before_guard"):
        return True
    if guard.get("source_flood_detected") or guard.get("diff_flood_detected") or guard.get("json_field_flood_detected"):
        return True
    reason = str(guard.get("reason") or "").lower()
    return any(
        marker in reason
        for marker in (
            "timeout",
            "flood",
            "limit_exceeded",
            "codex_bin_not_found",
            "provider",
            "auth",
            "quota",
            "unavailable",
        )
    )


def _classify_review_guard_result(guard: Any) -> dict[str, Any]:
    if not isinstance(guard, dict):
        return _review_unavailable("invalid_guard_json", guard_result=guard)
    if _review_has_guard_metadata(guard):
        return _review_unavailable(str(guard.get("reason") or "guard_metadata_unavailable"), guard_result=guard)

    status = guard.get("status")
    review = guard.get("review")
    if status == "passed" and isinstance(review, dict):
        verdict = str(review.get("verdict") or "").strip().lower()
        must_fix = review.get("must_fix")
        if verdict == "passed" and must_fix == []:
            return {
                "status": "passed",
                "verdict": "passed",
                "reason": "ok",
                "passed": True,
                "review": review,
                "guard_result": guard,
            }
        if isinstance(must_fix, list) and must_fix:
            return _review_failed("must_fix_non_empty", review=review, guard_result=guard)
        return _review_failed("review_not_strict_pass", review=review, guard_result=guard)

    if status == "failed":
        return _review_failed(str(guard.get("reason") or "verdict_failed"), review=review, guard_result=guard)
    return _review_unavailable(str(guard.get("reason") or "review_unavailable"), review=review, guard_result=guard)


def _run_review_autopilot(
    *,
    args: dict[str, Any],
    repo: Path,
    dirty_before_review: dict[str, Any],
    dirty_after_staged: dict[str, Any],
    normalized: dict[str, Any],
    allowlist: dict[str, list[str]],
    staged_result: dict[str, Any],
) -> dict[str, Any]:
    if not bool(args.get("review_autopilot")):
        return {"status": "not_requested", "passed": False}
    if not bool(args.get("review_autopilot_authorized")):
        return _review_unavailable("authorization_required", authorization_required=True)
    if dirty_after_staged.get("dirty_paths_truncated"):
        return _review_unavailable("dirty_paths_truncated")

    scope_files = _review_scope_files(
        dirty_before=dirty_before_review,
        dirty_after=dirty_after_staged,
        allowlist=allowlist,
    )
    if not scope_files:
        return _review_unavailable("missing_review_scope")

    review_dir = Path(tempfile.mkdtemp(prefix="hermes-codex-review-"))
    packet_path = review_dir / "review_packet.md"
    final_path = review_dir / "final.json"
    raw_log_path = review_dir / "raw.log"
    packet_cmd = [
        sys.executable,
        str(_review_packet_script()),
        "--workdir",
        str(repo),
        "--max-stat-chars",
        str(_REVIEW_PACKET_LIMITS["max_stat_chars"]),
        "--max-name-chars",
        str(_REVIEW_PACKET_LIMITS["max_name_chars"]),
        "--max-diff-chars",
        str(_REVIEW_PACKET_LIMITS["max_diff_chars"]),
        "--max-total-chars",
        str(_REVIEW_PACKET_LIMITS["max_total_chars"]),
    ]
    for path in scope_files:
        packet_cmd.extend(["--file", path])
    for path in normalized.get("allowed_files", []):
        packet_cmd.extend(["--allowed-file", path])
    for pattern in normalized.get("allowed_globs", []):
        packet_cmd.extend(["--allowed-glob", pattern])
    for path in _current_dirty_paths(dirty_before_review):
        packet_cmd.extend(["--dirty-baseline", path])
    if staged_result.get("candidate_id"):
        packet_cmd.extend(["--candidate-id", str(staged_result.get("candidate_id"))])
    if staged_result.get("candidate_disposition"):
        packet_cmd.extend(["--candidate-disposition", str(staged_result.get("candidate_disposition"))])
    completion_trusted = staged_result.get("completion_trusted")
    if completion_trusted is True:
        packet_cmd.extend(["--completion-trusted", "true"])
    elif completion_trusted is False:
        packet_cmd.extend(["--completion-trusted", "false"])
    else:
        packet_cmd.extend(["--completion-trusted", "unknown"])

    try:
        packet_proc = _run_review_command(packet_cmd, timeout=60)
    except subprocess.TimeoutExpired:
        return _review_unavailable("review_packet_timeout", packet_command=packet_cmd, scope_files=scope_files)
    if packet_proc.returncode != 0:
        return _review_unavailable(
            "review_packet_failed",
            packet_command=packet_cmd,
            packet_stderr=packet_proc.stderr,
            scope_files=scope_files,
        )
    packet_path.write_text(packet_proc.stdout, encoding="utf-8")

    guard_cmd = [
        sys.executable,
        str(_review_guard_script()),
        "--workdir",
        str(repo),
        "--prompt",
        _review_prompt(),
        "--review-packet-file",
        str(packet_path),
        "--final-file",
        str(final_path),
        "--raw-log",
        str(raw_log_path),
        "--timeout-seconds",
        str(int(args.get("review_timeout_seconds") or _REVIEW_TIMEOUT_SECONDS)),
    ]
    if isinstance(args.get("codex_review_bin"), str) and args["codex_review_bin"].strip():
        guard_cmd.extend(["--codex-bin", args["codex_review_bin"].strip()])

    try:
        guard_proc = _run_review_command(guard_cmd, timeout=int(args.get("review_timeout_seconds") or _REVIEW_TIMEOUT_SECONDS) + 30)
    except subprocess.TimeoutExpired:
        return _review_unavailable("review_timeout", packet_command=packet_cmd, guard_command=guard_cmd, scope_files=scope_files)

    if not final_path.exists():
        return _review_unavailable(
            "missing_final_file",
            packet_command=packet_cmd,
            guard_command=guard_cmd,
            guard_stdout=guard_proc.stdout,
            guard_stderr=guard_proc.stderr,
            scope_files=scope_files,
        )
    try:
        guard_json = json.loads(final_path.read_text(encoding="utf-8"))
    except Exception:
        return _review_unavailable(
            "invalid_guard_json",
            packet_command=packet_cmd,
            guard_command=guard_cmd,
            guard_stdout=guard_proc.stdout,
            guard_stderr=guard_proc.stderr,
            scope_files=scope_files,
        )

    review_result = _classify_review_guard_result(guard_json)
    review_result.update(
        {
            "packet_command": packet_cmd,
            "guard_command": guard_cmd,
            "packet_file": str(packet_path),
            "final_file": str(final_path),
            "raw_log": str(raw_log_path),
            "scope_files": scope_files,
        }
    )
    return review_result


def _checkpoint_blocked(
    reason: str,
    *,
    authorization_required: bool = False,
    dirty_state_id: str | None = None,
) -> dict[str, Any]:
    return {
        "status": "blocked",
        "reason": reason,
        "authorization_required": authorization_required,
        "dirty_state_id": dirty_state_id,
    }


def _validate_checkpoint_evidence(
    *,
    evidence: Any,
    normalized: dict[str, Any],
    allowlist: dict[str, list[str]],
    dirty: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    dirty_paths = _current_dirty_paths(dirty)
    dirty_state_id = dirty.get("dirty_state_id")
    if not isinstance(evidence, dict) or not evidence:
        return None, _checkpoint_blocked("missing_verification_evidence", dirty_state_id=dirty_state_id)

    if not (isinstance(evidence.get("stage_id"), str) and evidence["stage_id"].strip()) and not (
        isinstance(evidence.get("task_id"), str) and evidence["task_id"].strip()
    ):
        return None, _checkpoint_blocked("missing_stage_or_task_id", dirty_state_id=dirty_state_id)

    if evidence.get("allowed_files") != normalized.get("allowed_files") or evidence.get(
        "allowed_globs"
    ) != normalized.get("allowed_globs"):
        return None, _checkpoint_blocked("allowlist_mismatch", dirty_state_id=dirty_state_id)

    touched_files = evidence.get("touched_files")
    if not isinstance(touched_files, list) or any(not isinstance(path, str) or not path for path in touched_files):
        return None, _checkpoint_blocked("invalid_touched_files", dirty_state_id=dirty_state_id)
    if sorted(touched_files) != sorted(dirty_paths):
        return None, _checkpoint_blocked("touched_files_do_not_match_dirty_paths", dirty_state_id=dirty_state_id)
    if any(not _path_in_allowlist(path, allowlist) for path in touched_files):
        return None, _checkpoint_blocked("touched_files_outside_allowlist", dirty_state_id=dirty_state_id)

    if evidence.get("dirty_state_id") != dirty_state_id:
        return None, _checkpoint_blocked("dirty_state_id_mismatch", dirty_state_id=dirty_state_id)

    if not evidence.get("codex_implementation_status"):
        return None, _checkpoint_blocked("missing_codex_implementation_status", dirty_state_id=dirty_state_id)
    if evidence.get("codex_review_status") not in {"packet_only_passed", "full_review_passed"}:
        return None, _checkpoint_blocked("codex_review_not_passed", dirty_state_id=dirty_state_id)
    commands = evidence.get("hermes_verification_commands")
    if not isinstance(commands, list) or not commands:
        return None, _checkpoint_blocked("missing_hermes_verification_commands", dirty_state_id=dirty_state_id)
    if not evidence.get("verified_at"):
        return None, _checkpoint_blocked("missing_verified_at", dirty_state_id=dirty_state_id)

    return {"touched_files": list(touched_files), "dirty_state_id": dirty_state_id}, None


def _commit_checkpoint(repo: Path, *, touched_files: list[str], message: str | None) -> dict[str, Any]:
    commit_message = message.strip() if isinstance(message, str) and message.strip() else "Codex verified checkpoint"
    add_proc = subprocess.run(
        ["git", "-C", str(repo), "add", "--", *touched_files],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    if add_proc.returncode != 0:
        return _checkpoint_blocked("git_add_failed")
    commit_proc = subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", commit_message],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    if commit_proc.returncode != 0:
        return _checkpoint_blocked("git_commit_failed")
    sha_proc = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    return {
        "status": "committed",
        "commit_sha": sha_proc.stdout.strip() if sha_proc.returncode == 0 else None,
        "message": commit_message,
        "touched_files": list(touched_files),
    }


def _leftover_candidate(dirty: dict[str, Any], codex_staged_result: dict[str, Any]) -> dict[str, Any]:
    result = {
        "requires_review": True,
        "requires_hermes_verification": True,
        "touched_files": _current_dirty_paths(dirty),
        "dirty_state_id": dirty.get("dirty_state_id"),
    }
    for key in ("candidate_id", "candidate_disposition", "completion_trusted"):
        if key in codex_staged_result:
            result[key] = codex_staged_result.get(key)
    return result


def _finish_after_staged(
    *,
    args: dict[str, Any],
    repo: Path,
    git_head: str | None,
    initial_dirty: dict[str, Any],
    review_baseline_dirty: dict[str, Any] | None = None,
    normalized: dict[str, Any],
    allowlist: dict[str, list[str]],
    staged_result: dict[str, Any],
    dirty_recovery_update: dict[str, Any] | None = None,
) -> str:
    result = _base(
        "staged_called",
        repo=repo,
        git_head=git_head,
        dirty=initial_dirty,
        codex_staged_result=staged_result,
    )
    if dirty_recovery_update:
        result["dirty_recovery"].update(dirty_recovery_update)

    post_dirty = staged._dirty_check(repo)
    review_result = _run_review_autopilot(
        args=args,
        repo=repo,
        dirty_before_review=review_baseline_dirty or initial_dirty,
        dirty_after_staged=post_dirty,
        normalized=normalized,
        allowlist=allowlist,
        staged_result=staged_result,
    )
    if review_result.get("status") != "not_requested":
        result["review"] = review_result
        if review_result.get("status") == "passed":
            result["leftover_candidate"] = {
                "requires_review": False,
                "requires_hermes_verification": True,
                "touched_files": review_result.get("scope_files", []),
                "dirty_state_id": post_dirty.get("dirty_state_id"),
                "candidate_id": staged_result.get("candidate_id"),
                "candidate_disposition": staged_result.get("candidate_disposition"),
                "completion_trusted": staged_result.get("completion_trusted", False),
            }
        else:
            result["status"] = "review_failed"
            return _json_result(result)

        after_review_dirty = staged._dirty_check(repo)
        result["review"]["post_review_dirty_check"] = after_review_dirty
        if after_review_dirty.get("dirty_state_id") != post_dirty.get("dirty_state_id"):
            result["status"] = "review_failed"
            result["review"] = _review_failed(
                "review_changed_worktree",
                contaminated=True,
                pre_review_dirty_state_id=post_dirty.get("dirty_state_id"),
                post_review_dirty_state_id=after_review_dirty.get("dirty_state_id"),
                post_review_dirty_check=after_review_dirty,
            )
            return _json_result(result)

    checkpoint_requested = bool(args.get("checkpoint_verified_diff"))
    if checkpoint_requested:
        if not bool(args.get("standing_authorization")):
            result["status"] = "checkpoint_blocked"
            result["checkpoint"] = _checkpoint_blocked(
                "authorization_required",
                authorization_required=True,
                dirty_state_id=post_dirty.get("dirty_state_id"),
            )
            return _json_result(result)
        if post_dirty.get("is_clean"):
            result["status"] = "checkpoint_blocked"
            result["checkpoint"] = _checkpoint_blocked(
                "no_dirty_diff_to_checkpoint",
                dirty_state_id=post_dirty.get("dirty_state_id"),
            )
            return _json_result(result)
        verified, blocked = _validate_checkpoint_evidence(
            evidence=args.get("verification_evidence"),
            normalized=normalized,
            allowlist=allowlist,
            dirty=post_dirty,
        )
        if blocked is not None:
            result["status"] = "checkpoint_blocked"
            result["checkpoint"] = blocked
            return _json_result(result)
        assert verified is not None
        checkpoint = _commit_checkpoint(
            repo,
            touched_files=verified["touched_files"],
            message=args.get("checkpoint_message"),
        )
        if checkpoint.get("status") != "committed":
            result["status"] = "checkpoint_blocked"
        result["checkpoint"] = checkpoint
        return _json_result(result)

    if not post_dirty.get("is_clean") and result.get("review", {}).get("status") != "passed":
        result["leftover_candidate"] = _leftover_candidate(post_dirty, staged_result)
    return _json_result(result)


def _dirty_buckets(dirty: dict[str, Any]) -> set[str]:
    classes = dirty.get("dirty_path_classes", {})
    return {name for name, paths in classes.items() if paths}


def _cache_only(dirty: dict[str, Any]) -> bool:
    return (
        _dirty_buckets(dirty) == {"cache"}
        and not dirty.get("unsafe_reasons")
        and not dirty.get("dirty_paths_truncated")
    )


def _safe_cache_path(repo: Path, rel_path: str) -> Path | None:
    if not isinstance(rel_path, str) or not rel_path.strip():
        return None
    path = Path(rel_path)
    if path.is_absolute() or ".." in path.parts:
        return None
    if staged._dirty_path_class(rel_path) != "cache":
        return None
    target = (repo / rel_path).resolve(strict=False)
    if not staged._is_relative_to(target, repo):
        return None
    return target


def _clean_cache_dirty_paths(repo: Path, dirty: dict[str, Any], *, approved_paths: set[str] | None = None) -> list[str]:
    approved = approved_paths or set()
    cleaned: list[str] = []
    for rel_path in dirty.get("dirty_path_classes", {}).get("cache", []):
        if rel_path not in approved:
            continue
        target = _safe_cache_path(repo, rel_path)
        if target is None or not (target.exists() or target.is_symlink()):
            continue
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
        cleaned.append(rel_path)
    return cleaned


def _provenance_events(args: dict[str, Any]) -> list[dict[str, Any]]:
    events = args.get("provenance_events")
    if not isinstance(events, list):
        return []
    return [event for event in events if isinstance(event, dict)]


def _cleanup_allowlist(args: dict[str, Any], allowlist: dict[str, list[str]]) -> dict[str, list[str]]:
    # Cleanup/delete scope is intentionally separate from Codex implementation
    # write scope. A file being safe for Codex to edit does not make it safe to
    # delete or overwrite during dirty recovery.
    files: list[str] = []
    globs: list[str] = []
    extra_files = args.get("cleanup_allowed_files")
    extra_globs = args.get("cleanup_allowed_globs")
    if isinstance(extra_files, list):
        files.extend(path for path in extra_files if isinstance(path, str) and path.strip())
    if isinstance(extra_globs, list):
        globs.extend(pattern for pattern in extra_globs if isinstance(pattern, str) and pattern.strip())
    return {"files": files, "globs": globs}


def _cleanup_decisions(
    *,
    args: dict[str, Any],
    repo: Path,
    git_head: str | None,
    dirty: dict[str, Any],
    allowlist: dict[str, list[str]],
    dry_run: bool,
) -> dict[str, dict[str, Any]]:
    branch = _current_branch(repo)
    cleanup_allowlist = _cleanup_allowlist(args, allowlist)
    return {
        rel_path: provenance.cleanup_decision(
            repo=repo,
            rel_path=rel_path,
            current_session_id=args.get("session_id") if isinstance(args.get("session_id"), str) else None,
            branch=branch,
            head_sha=git_head,
            events=_provenance_events(args),
            allowed_files=cleanup_allowlist.get("files", []),
            allowed_globs=cleanup_allowlist.get("globs", []),
            explicit_authorization=bool(args.get("standing_authorization")),
            operation="delete",
            dry_run=dry_run,
        )
        for rel_path in _current_dirty_paths(dirty)
    }


def _dry_run_preview_decisions(
    *,
    args: dict[str, Any],
    repo: Path,
    git_head: str | None,
    dirty: dict[str, Any],
    allowlist: dict[str, list[str]],
) -> dict[str, dict[str, Any]]:
    """Build ownership/cleanup preview without invoking cleanup helpers.

    Dry-run may classify current dirty paths, but it must not call code paths
    that clean, delete, overwrite, write ledgers/provenance, create worktrees,
    or invoke Codex/review guards.
    """
    branch = _current_branch(repo)
    cleanup_allowlist = _cleanup_allowlist(args, allowlist)
    decisions: dict[str, dict[str, Any]] = {}
    for rel_path in _current_dirty_paths(dirty):
        decision = provenance.classify_path(
            repo=repo,
            rel_path=rel_path,
            current_session_id=args.get("session_id") if isinstance(args.get("session_id"), str) else None,
            branch=branch,
            head_sha=git_head,
            events=_provenance_events(args),
        )
        reasons = set(decision.get("blocking_reasons") or [])
        reasons.add("dry_run_non_mutating")
        if decision.get("owner_policy") not in {"current_session", "review_artifact_current_session"}:
            reasons.add("owner_policy_not_current_session")
        if not _path_in_allowlist(rel_path, cleanup_allowlist):
            reasons.add("path_not_in_allowlist")
        if not bool(args.get("standing_authorization")):
            reasons.add("authorization_required")
        decision["cleanup_allowed"] = False
        decision["blocking_reasons"] = sorted(reasons)
        decisions[rel_path] = decision
    return decisions


def _cleanup_blocking_reasons(decisions: dict[str, dict[str, Any]]) -> list[str]:
    reasons: set[str] = set()
    for decision in decisions.values():
        reasons.update(decision.get("blocking_reasons") or [])
    return sorted(reasons)


def _ownership_preview_from_decisions(decisions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [decisions[path] for path in sorted(decisions)]


def _dirty_overlaps_scope(dirty_paths: list[str], allowlist: dict[str, list[str]]) -> list[str]:
    return sorted(path for path in dirty_paths if _path_in_allowlist(path, allowlist))


def _dry_run_blocking_reasons(
    *,
    dirty: dict[str, Any],
    allowlist: dict[str, list[str]],
    cleanup_decisions: dict[str, dict[str, Any]],
) -> list[str]:
    if dirty.get("is_clean"):
        return []
    dirty_paths = _current_dirty_paths(dirty)
    unknown_or_foreign = [
        path
        for path, decision in cleanup_decisions.items()
        if decision.get("owner_policy") in {"unknown_unowned", "other_known_session", "dangerous_conflict"}
    ]
    reasons: set[str] = set()
    overlapping = _dirty_overlaps_scope(unknown_or_foreign, allowlist)
    if overlapping:
        reasons.add("unknown_dirty_overlaps_write_scope")
    elif unknown_or_foreign or dirty_paths:
        reasons.add("unknown_dirty_non_overlap_recommend_isolated_worktree")
    return sorted(reasons)


def _recommended_next_stage(
    *,
    args: dict[str, Any],
    normalized: dict[str, Any],
    blocking_reasons: list[str],
) -> dict[str, Any]:
    if "unknown_dirty_overlaps_write_scope" in blocking_reasons:
        stage_id = "stop_for_user"
    elif "unknown_dirty_non_overlap_recommend_isolated_worktree" in blocking_reasons:
        stage_id = "isolated_worktree"
    else:
        stage_id = args.get("stage_id") or "phase12a0-provenance-contract"
    return {
        "stage_id": stage_id,
        "allowed_files": normalized.get("allowed_files", []),
        "allowed_globs": normalized.get("allowed_globs", []),
        "verify_cmd_ids": normalized.get("verify_cmd_ids", []),
    }


def _ledger_event_preview(
    *,
    repo: Path,
    git_head: str | None,
    args: dict[str, Any],
    dirty: dict[str, Any],
    blocking_reasons: list[str],
) -> list[dict[str, Any]]:
    branch = _current_branch(repo)
    stage_id = args.get("stage_id") or "phase12a0-provenance-contract"
    return [
        ledger.redact(
            {
                "schema_version": 1,
                "repo_id": ledger.repo_id(repo),
                "branch": branch,
                "head_sha": git_head,
                "stage_id": stage_id,
                "session_id": args.get("session_id") if isinstance(args.get("session_id"), str) else "unknown",
                "actor": "hermes",
                "tool": "codex_workflow_run",
                "operation": "dry_run_plan",
                "dirty_state_id": dirty.get("dirty_state_id"),
                "dirty_paths": _current_dirty_paths(dirty),
                "blocking_reasons": blocking_reasons,
            }
        )
    ]


def _stage_slug(stage_id: Any) -> str:
    raw = str(stage_id or "codex-workflow").strip().lower()
    slug = re.sub(r"[^a-z0-9._-]+", "-", raw).strip("-._")
    return slug[:48] or "codex-workflow"


def _create_isolated_worktree(repo: Path, *, stage_id: Any, git_head: str | None) -> dict[str, str]:
    shortsha = (git_head or "head")[:12]
    stage = _stage_slug(stage_id)
    date = _datetime.datetime.now(_datetime.UTC).strftime("%Y%m%d")
    base = repo.parent / ".hermes-worktrees"
    worktree = base / f"{repo.name}-{stage}-{shortsha}"
    branch = f"work/{stage}-{date}-{shortsha}"
    if worktree.exists():
        raise RuntimeError("isolated_worktree_path_exists")
    base.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["git", "-C", str(repo), "worktree", "add", "-b", branch, str(worktree), "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError("isolated_worktree_create_failed")
    return {"path": str(worktree), "branch": branch, "source_head": git_head}


def codex_workflow_run(args: dict[str, Any]) -> str:
    if not isinstance(args, dict):
        args = {}

    validated, error = _validate_common(args)
    if error is not None:
        return _json_result(error)
    assert validated is not None

    repo = validated["repo"]
    git_head = validated["git_head"]
    normalized = validated["normalized"]
    dirty = staged._dirty_check(repo)

    if validated["mode"] == "dry_run":
        cleanup_decisions = _dry_run_preview_decisions(
            args=args,
            repo=repo,
            git_head=git_head,
            dirty=dirty,
            allowlist=validated["allowlist"],
        )
        blocking_reasons = _dry_run_blocking_reasons(
            dirty=dirty,
            allowlist=validated["allowlist"],
            cleanup_decisions=cleanup_decisions,
        )
        stage_id = args.get("stage_id") or "phase12a0-provenance-contract"
        branch = _current_branch(repo)
        return _json_result(
            _base(
                "dry_run",
                repo=repo,
                git_head=git_head,
                dirty=dirty,
                mode="dry_run",
                resolved_allowlist=validated["allowlist"],
                would_call_staged=bool(dirty.get("is_clean")),
                would_run_review=bool(args.get("review_autopilot") and args.get("review_autopilot_authorized")),
                would_run_verification=False,
                would_create_isolated_worktree=False,
                would_commit=False,
                would_push=False,
                would_deploy_or_restart=False,
                would_write_ledger=True,
                would_record_ledger_events=True,
                ledger_path=str(ledger.ledger_path(repo=repo, branch=branch, stage_id=stage_id)),
                ledger_event_preview=_ledger_event_preview(
                    repo=repo,
                    git_head=git_head,
                    args=args,
                    dirty=dirty,
                    blocking_reasons=blocking_reasons,
                ),
                dirty_ownership=_ownership_preview_from_decisions(cleanup_decisions),
                cleanup_allowed=False,
                cleanup_blocking_reasons=_cleanup_blocking_reasons(cleanup_decisions),
                would_delete_paths=[],
                would_overwrite_paths=[],
                blocking_reasons=blocking_reasons,
                authorization_required=["write_stage"],
                recommended_next_stage=_recommended_next_stage(
                    args=args,
                    normalized=normalized,
                    blocking_reasons=blocking_reasons,
                ),
            )
        )

    if dirty.get("is_clean"):
        staged_result = _call_staged(normalized)
        return _finish_after_staged(
            args=args,
            repo=repo,
            git_head=git_head,
            initial_dirty=dirty,
            normalized=normalized,
            allowlist=validated["allowlist"],
            staged_result=staged_result,
        )

    standing_auth = bool(args.get("standing_authorization"))
    if _cache_only(dirty) and standing_auth and bool(args.get("auto_clean_cache")):
        cleanup_decisions = _cleanup_decisions(
            args=args,
            repo=repo,
            git_head=git_head,
            dirty=dirty,
            allowlist=validated["allowlist"],
            dry_run=False,
        )
        blocked_cleanup = {
            path: decision for path, decision in cleanup_decisions.items() if not decision.get("cleanup_allowed")
        }
        if blocked_cleanup:
            result = _base("dirty_recovery_required", repo=repo, git_head=git_head, dirty=dirty)
            result["dirty_recovery"].update(
                {
                    "strategy": "cache_cleanup_blocked_by_provenance",
                    "provenance_cleanup": cleanup_decisions,
                    "cleanup_blocking_reasons": _cleanup_blocking_reasons(cleanup_decisions),
                }
            )
            return _json_result(result)
        cleaned_paths = _clean_cache_dirty_paths(
            repo,
            dirty,
            approved_paths={
                path for path, decision in cleanup_decisions.items() if decision.get("cleanup_allowed")
            },
        )
        after = staged._dirty_check(repo)
        if after.get("is_clean"):
            staged_result = _call_staged(normalized)
            return _finish_after_staged(
                args=args,
                repo=repo,
                git_head=git_head,
                initial_dirty=dirty,
                review_baseline_dirty=after,
                normalized=normalized,
                allowlist=validated["allowlist"],
                staged_result=staged_result,
                dirty_recovery_update={
                    "strategy": "cache_cleanup",
                    "cache_cleaned_paths": cleaned_paths,
                    "post_cleanup_dirty_check": after,
                },
            )
        result = _base("dirty_recovery_required", repo=repo, git_head=git_head, dirty=dirty)
        result["dirty_recovery"].update(
            {
                "strategy": "cache_cleanup_incomplete",
                "cache_cleaned_paths": cleaned_paths,
                "post_cleanup_dirty_check": after,
            }
        )
        return _json_result(result)

    if standing_auth and bool(args.get("allow_isolated_worktree")):
        try:
            worktree_info = _create_isolated_worktree(repo, stage_id=args.get("stage_id"), git_head=git_head)
        except Exception as exc:
            result = _base("dirty_recovery_required", repo=repo, git_head=git_head, dirty=dirty)
            result["dirty_recovery"].update({"strategy": "isolated_worktree_failed", "error": str(exc)})
            return _json_result(result)
        isolated_args = dict(normalized)
        isolated_args["workdir"] = worktree_info["path"]
        isolated_baseline = staged._dirty_check(Path(worktree_info["path"]))
        staged_result = _call_staged(isolated_args)
        return _finish_after_staged(
            args=args,
            repo=Path(worktree_info["path"]),
            git_head=git_head,
            initial_dirty=dirty,
            review_baseline_dirty=isolated_baseline,
            normalized=isolated_args,
            allowlist=validated["allowlist"],
            staged_result=staged_result,
            dirty_recovery_update={"strategy": "isolated_worktree", "isolated_worktree": worktree_info},
        )

    result = _base("dirty_recovery_required", repo=repo, git_head=git_head, dirty=dirty)
    result["dirty_recovery"].update(staged._dirty_decision_metadata(dirty))
    return _json_result(result)


_SCHEMA = {
    "name": "codex_workflow_run",
    "description": (
        "High-level Hermes orchestrator for dirty worktree recovery before guarded Codex "
        "candidate implementation. It validates repo, scope, and policies; may remove "
        "cache-only dirty paths with standing authorization; may create an isolated clean "
        "worktree with standing authorization; then delegates to codex_staged_implement."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "workdir": {"type": "string", "description": "Git repository root."},
            "task": {"type": "string", "description": "Implementation task."},
            "allowed_files": {"type": "array", "items": {"type": "string"}},
            "allowed_globs": {"type": "array", "items": {"type": "string"}},
            "verify_cmd_ids": {
                "type": "array",
                "items": {"type": "string", "enum": ["diff-check", "none"]},
            },
            "continue_policy": {"type": "string", "enum": ["stop-on-review-needed"]},
            "dirty_baseline_policy": {"type": "string", "enum": ["require-clean"]},
            "standing_authorization": {"type": "boolean"},
            "session_id": {"type": "string"},
            "provenance_events": {"type": "array", "items": {"type": "object"}},
            "auto_clean_cache": {"type": "boolean"},
            "cleanup_allowed_files": {"type": "array", "items": {"type": "string"}},
            "cleanup_allowed_globs": {"type": "array", "items": {"type": "string"}},
            "allow_isolated_worktree": {"type": "boolean"},
            "stage_id": {"type": "string"},
            "checkpoint_verified_diff": {"type": "boolean"},
            "verification_evidence": {"type": "object"},
            "checkpoint_message": {"type": "string"},
            "review_autopilot": {"type": "boolean"},
            "review_autopilot_authorized": {"type": "boolean"},
            "review_timeout_seconds": {"type": "integer"},
            "codex_review_bin": {"type": "string"},
            "mode": {"type": "string", "enum": ["execute", "dry_run"]},
        },
        "required": ["workdir", "task"],
    },
}


registry.register(
    name="codex_workflow_run",
    toolset="codex_staged_implement",
    schema=_SCHEMA,
    handler=lambda args, **kwargs: codex_workflow_run(args),
    description=_SCHEMA["description"],
)
