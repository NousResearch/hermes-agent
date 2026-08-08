import datetime as _datetime
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from tools import codex_staged_implement_tool as staged
from tools.registry import registry


_SUPPORTED_MODES = {"execute", "dry_run"}
_SUPPORTED_CONTINUE_POLICY = "stop-on-review-needed"
_SUPPORTED_DIRTY_POLICY = "require-clean"
_ALLOWED_VERIFY_IDS = {"diff-check", "none"}
_AUTO_PACKET_REVIEW_STATUSES = {"ready_for_review", "review_needed", "takeover_candidate"}
_PACKET_REVIEW_STATUSES = {"packet_only_passed", "packet_only_failed", "packet_only_unusable"}
_PACKET_REVIEW_SUMMARY_KEYS = {
    "status",
    "reason",
    "must_fix_count",
    "final_judgment",
    "summary",
    "raw_log_path",
    "final_file",
}
_MAX_PACKET_REVIEW_SUMMARY_CHARS = 1200
_SUMMARY_LEAK_MARKERS = ("diff --git", "\n@@", "@@ ", "--- a/", "+++ b/")
_OWNER_CLASSES = (
    "current_session",
    "other_known_session",
    "unknown_unowned",
    "generated_cache",
    "review_artifact_current_session",
    "dangerous_conflict",
)
_DANGEROUS_UNSAFE_REASONS = {
    "conflict_status",
    "rename_status",
    "delete_status",
    "typechange_or_chmod_status",
    "submodule_status_or_metadata",
    "secret_path_evidence",
    "real_data_path_evidence",
    "binary_path_evidence",
    "large_file_evidence",
}


def _json_result(payload: dict[str, Any]) -> str:
    return json.dumps(staged._bound(payload), ensure_ascii=False)


def _base(
    status: str,
    *,
    repo: Path | None = None,
    git_head: str | None = None,
    dirty: dict[str, Any] | None = None,
    preflight: dict[str, Any] | None = None,
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
        "preflight": preflight,
        "codex_staged_result": codex_staged_result,
        "codex_packet_review": None,
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


def _current_dirty_paths(dirty: dict[str, Any]) -> list[str]:
    paths = dirty.get("dirty_paths")
    return list(paths) if isinstance(paths, list) else []


def _path_in_allowlist(path: str, allowlist: dict[str, list[str]]) -> bool:
    if path in allowlist.get("files", []):
        return True
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in allowlist.get("globs", []))


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
        "requires_fixes": False,
        "requires_hermes_verification": True,
        "touched_files": _current_dirty_paths(dirty),
        "dirty_state_id": dirty.get("dirty_state_id"),
    }
    for key in ("candidate_id", "candidate_disposition", "completion_trusted"):
        if key in codex_staged_result:
            result[key] = codex_staged_result.get(key)
    return result


def _bounded_review_summary(summary: str | None) -> str | None:
    if not isinstance(summary, str):
        return None
    stripped = summary.strip()
    if not stripped:
        return None
    if any(marker in stripped for marker in _SUMMARY_LEAK_MARKERS):
        return "[summary omitted: looked like diff/source output]"
    if len(stripped) > _MAX_PACKET_REVIEW_SUMMARY_CHARS:
        return stripped[: _MAX_PACKET_REVIEW_SUMMARY_CHARS - 24].rstrip() + "...[truncated]"
    return stripped


def _bounded_review_reason(reason: str | None) -> str | None:
    if not isinstance(reason, str):
        return None
    stripped = reason.strip()
    if not stripped:
        return None
    if "\n" in stripped or any(marker in stripped for marker in _SUMMARY_LEAK_MARKERS):
        return "review_reason_omitted"
    if len(stripped) > 160:
        return stripped[:136].rstrip() + "...[truncated]"
    return stripped


def _bounded_review_final_judgment(final_judgment: str | None) -> str | None:
    if not isinstance(final_judgment, str):
        return None
    stripped = final_judgment.strip()
    if not stripped:
        return None
    if any(marker in stripped for marker in _SUMMARY_LEAK_MARKERS):
        return "[final judgment omitted: looked like diff/source output]"
    if len(stripped) > 240:
        return stripped[:216].rstrip() + "...[truncated]"
    return stripped


def _safe_review_artifact_path(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped or "\n" in stripped or any(marker in stripped for marker in _SUMMARY_LEAK_MARKERS):
        return None
    if len(stripped) > 500:
        return None
    return stripped


def _packet_review_result(
    status: str,
    *,
    reason: str | None = None,
    must_fix_count: int | None = None,
    final_judgment: str | None = None,
    summary: str | None = None,
    raw_log_path: str | None = None,
    final_file: str | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "reason": _bounded_review_reason(reason),
        "must_fix_count": must_fix_count,
        "final_judgment": _bounded_review_final_judgment(final_judgment),
        "summary": _bounded_review_summary(summary),
        "raw_log_path": _safe_review_artifact_path(raw_log_path),
        "final_file": _safe_review_artifact_path(final_file),
    }


def _packet_review_not_run(reason: str) -> dict[str, Any]:
    return _packet_review_result("not_run", reason=reason)


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _normalize_packet_review_result(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return _packet_review_result("packet_only_unusable", reason="malformed_review_result")

    status = raw.get("status")
    if status in _PACKET_REVIEW_STATUSES:
        must_fix_count = _safe_int(raw.get("must_fix_count"))
        final_judgment = raw.get("final_judgment") if isinstance(raw.get("final_judgment"), str) else None
        summary = raw.get("summary") if isinstance(raw.get("summary"), str) else None
        if status == "packet_only_passed" and (
            must_fix_count != 0
            or not (isinstance(final_judgment, str) and final_judgment.strip())
            or "需要先修" in final_judgment
            or "可以继续" not in final_judgment
            or not _bounded_review_summary(summary)
        ):
            return _packet_review_result(
                "packet_only_unusable",
                reason="review_guard_schema_invalid",
                raw_log_path=raw.get("raw_log_path") if isinstance(raw.get("raw_log_path"), str) else None,
                final_file=raw.get("final_file") if isinstance(raw.get("final_file"), str) else None,
            )
        if status == "packet_only_failed" and (
            must_fix_count is None
            or must_fix_count <= 0
            or not (isinstance(final_judgment, str) and "需要先修" in final_judgment)
            or not _bounded_review_summary(summary)
        ):
            return _packet_review_result(
                "packet_only_unusable",
                reason="review_guard_schema_invalid",
                raw_log_path=raw.get("raw_log_path") if isinstance(raw.get("raw_log_path"), str) else None,
                final_file=raw.get("final_file") if isinstance(raw.get("final_file"), str) else None,
            )
        return _packet_review_result(
            status,
            reason=raw.get("reason") if isinstance(raw.get("reason"), str) else None,
            must_fix_count=must_fix_count,
            final_judgment=final_judgment,
            summary=summary,
            raw_log_path=raw.get("raw_log_path") if isinstance(raw.get("raw_log_path"), str) else None,
            final_file=raw.get("final_file") if isinstance(raw.get("final_file"), str) else None,
        )

    if any(
        raw.get(flag)
        for flag in (
            "diff_flood_detected",
            "source_flood_detected",
            "json_field_flood_detected",
            "terminated_by_guard",
        )
    ):
        return _packet_review_result(
            "packet_only_unusable",
            reason=str(raw.get("reason") or "review_guard_flood_or_terminated"),
            raw_log_path=raw.get("raw_log_path") if isinstance(raw.get("raw_log_path"), str) else None,
            final_file=raw.get("final_file") if isinstance(raw.get("final_file"), str) else None,
        )

    if status != "passed":
        return _packet_review_result(
            "packet_only_unusable",
            reason=f"review_guard_status_{status or 'missing'}",
            raw_log_path=raw.get("raw_log_path") if isinstance(raw.get("raw_log_path"), str) else None,
            final_file=raw.get("final_file") if isinstance(raw.get("final_file"), str) else None,
        )

    review = raw.get("review")
    if not isinstance(review, dict):
        return _packet_review_result(
            "packet_only_unusable",
            reason="missing_structured_review",
            raw_log_path=raw.get("raw_log_path") if isinstance(raw.get("raw_log_path"), str) else None,
            final_file=raw.get("final_file") if isinstance(raw.get("final_file"), str) else None,
        )

    must_fix = review.get("must_fix")
    verdict = review.get("verdict") if isinstance(review.get("verdict"), str) else None
    final_judgment = review.get("final_judgment") if isinstance(review.get("final_judgment"), str) else None
    summary = review.get("summary") if isinstance(review.get("summary"), str) else None
    if (
        verdict not in {"passed", "failed"}
        or not isinstance(must_fix, list)
        or any(not isinstance(item, str) for item in must_fix)
        or not (isinstance(final_judgment, str) and final_judgment.strip())
        or (verdict == "passed" and "可以继续" not in final_judgment)
        or (verdict == "failed" and "需要先修" not in final_judgment)
        or (verdict == "failed" and isinstance(must_fix, list) and not must_fix)
        or any(marker in final_judgment for marker in _SUMMARY_LEAK_MARKERS)
        or not (isinstance(summary, str) and summary.strip())
    ):
        return _packet_review_result(
            "packet_only_unusable",
            reason="review_guard_schema_invalid",
            raw_log_path=raw.get("raw_log_path") if isinstance(raw.get("raw_log_path"), str) else None,
            final_file=raw.get("final_file") if isinstance(raw.get("final_file"), str) else None,
        )

    must_fix_count = len(must_fix)
    failed = must_fix_count > 0 or verdict == "failed" or "需要先修" in final_judgment

    return _packet_review_result(
        "packet_only_failed" if failed else "packet_only_passed",
        reason=None,
        must_fix_count=must_fix_count,
        final_judgment=final_judgment,
        summary=summary,
        raw_log_path=raw.get("raw_log_path") if isinstance(raw.get("raw_log_path"), str) else None,
        final_file=raw.get("final_file") if isinstance(raw.get("final_file"), str) else None,
    )


def _should_auto_packet_review(
    *,
    staged_result: dict[str, Any],
    post_dirty: dict[str, Any],
    checkpoint_requested: bool,
) -> tuple[bool, str]:
    if checkpoint_requested:
        return False, "checkpoint_requested"
    if post_dirty.get("is_clean"):
        return False, "no_candidate_diff"
    status = staged_result.get("status")
    if status not in _AUTO_PACKET_REVIEW_STATUSES:
        return False, f"staged_status_{status or 'missing'}"
    return True, "candidate_ready"


def _completion_trusted_arg(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "unknown"


def _candidate_disposition_arg(value: Any) -> str | None:
    if value in {"pending_review", "takeover_required", "rejected", "unavailable"}:
        return str(value)
    return None


def _review_env() -> dict[str, str]:
    env = dict(os.environ)
    home = Path.home()
    extra = [
        str(home / ".local" / "node-v22.21.1-linux-x64" / "bin"),
        str(home / ".local" / "bin"),
    ]
    env["PATH"] = ":".join([*extra, env.get("PATH", "")])
    return env


def _tool_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _codex_preflight(repo: Path) -> dict[str, Any]:
    _ = repo
    root = _tool_root()
    workflow_path = _review_env().get("PATH", "")
    checks = {
        "impl_guard_exists": (root / "scripts" / "runtime" / "codex_impl_guard.py").is_file(),
        "stage_runner_exists": (root / "scripts" / "runtime" / "codex_stage_runner.py").is_file(),
        "review_guard_exists": (root / "scripts" / "runtime" / "codex_review_guard.py").is_file(),
        "review_packet_exists": (root / "scripts" / "runtime" / "codex_review_packet.py").is_file(),
        "codex_bin_found": shutil.which("codex-yuna", path=workflow_path) is not None,
        "node_bin_found": shutil.which("node", path=workflow_path) is not None,
        "sandbox_verified_env": os.environ.get("HERMES_CODEX_IMPL_GUARD_SANDBOX_VERIFIED") == "1",
    }
    blocker_codes = {
        "impl_guard_exists": "missing_impl_guard",
        "stage_runner_exists": "missing_stage_runner",
        "review_guard_exists": "missing_review_guard",
        "review_packet_exists": "missing_review_packet",
        "codex_bin_found": "missing_codex_bin",
        "node_bin_found": "missing_node_bin",
        "sandbox_verified_env": "sandbox_not_verified",
    }
    blockers = [code for name, code in blocker_codes.items() if checks.get(name) is not True]
    return {"status": "passed" if not blockers else "blocked", "checks": checks, "blockers": blockers}


def _preflight_passed(preflight: dict[str, Any] | None) -> bool:
    if not isinstance(preflight, dict) or preflight.get("status") != "passed":
        return False
    checks = preflight.get("checks")
    blockers = preflight.get("blockers")
    required_checks = {
        "impl_guard_exists",
        "stage_runner_exists",
        "review_guard_exists",
        "review_packet_exists",
        "codex_bin_found",
        "node_bin_found",
        "sandbox_verified_env",
    }
    return (
        isinstance(checks, dict)
        and set(checks) == required_checks
        and all(value is True for value in checks.values())
        and blockers == []
    )


def _preflight_blocked_result(
    *,
    repo: Path,
    git_head: str | None,
    dirty: dict[str, Any],
    preflight: dict[str, Any],
    args: dict[str, Any] | None = None,
) -> str:
    args = args or {}
    result = _base(
        "preflight_blocked",
        repo=repo,
        git_head=git_head,
        dirty=dirty,
        preflight=preflight,
        codex_packet_review=_packet_review_not_run("preflight_blocked"),
    )
    result["dirty_recovery"].update(
        _dirty_recovery_context(args, dirty=dirty, mode=str(args.get("mode", "execute")), preflight=preflight)
    )
    return _json_result(result)


def _run_packet_only_review(
    *,
    repo: Path,
    touched_files: list[str],
    allowlist: dict[str, list[str]],
    dirty_baseline_paths: list[str],
    staged_result: dict[str, Any],
) -> dict[str, Any]:
    root = _tool_root()
    packet_script = root / "scripts" / "runtime" / "codex_review_packet.py"
    guard_script = root / "scripts" / "runtime" / "codex_review_guard.py"
    review_dir = Path(tempfile.mkdtemp(prefix="codex-workflow-review-"))
    packet_file = review_dir / "packet.md"
    prompt_file = review_dir / "prompt.txt"
    raw_log = review_dir / "raw.log"
    final_file = review_dir / "final.txt"

    packet_cmd = [sys.executable, str(packet_script), "--workdir", str(repo)]
    for path in touched_files:
        packet_cmd.extend(["--file", path])
    for path in allowlist.get("files", []):
        packet_cmd.extend(["--allowed-file", path])
    for pattern in allowlist.get("globs", []):
        packet_cmd.extend(["--allowed-glob", pattern])
    for path in dirty_baseline_paths:
        packet_cmd.extend(["--dirty-baseline", path])
    candidate_id = staged_result.get("candidate_id")
    if isinstance(candidate_id, str) and candidate_id.strip():
        packet_cmd.extend(["--candidate-id", candidate_id.strip()])
    disposition = _candidate_disposition_arg(staged_result.get("candidate_disposition"))
    if disposition:
        packet_cmd.extend(["--candidate-disposition", disposition])
    packet_cmd.extend(["--completion-trusted", _completion_trusted_arg(staged_result.get("completion_trusted"))])

    try:
        packet_proc = subprocess.run(
            packet_cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
    except Exception as exc:
        return _packet_review_result("packet_only_unusable", reason=f"packet_build_exception:{type(exc).__name__}")
    if packet_proc.returncode != 0:
        return _packet_review_result("packet_only_unusable", reason="packet_build_failed")
    packet_file.write_text(packet_proc.stdout, encoding="utf-8")

    prompt_file.write_text(
        "Review this bounded packet only. Do not edit files. Do not run shell commands. "
        "Do not request full source or full diffs. Focus on requirements fit, fail-closed "
        "semantics, scope creep, and whether Hermes verification is still required. Return "
        "structured Chinese review sections with must-fix items and final judgment.\n",
        encoding="utf-8",
    )
    guard_cmd = [
        sys.executable,
        str(guard_script),
        "--workdir",
        str(repo),
        "--prompt-file",
        str(prompt_file),
        "--review-packet-file",
        str(packet_file),
        "--raw-log",
        str(raw_log),
        "--final-file",
        str(final_file),
        "--timeout-seconds",
        "420",
    ]
    try:
        guard_proc = subprocess.run(
            guard_cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=480,
            env=_review_env(),
        )
    except subprocess.TimeoutExpired:
        return _packet_review_result(
            "packet_only_unusable",
            reason="review_guard_timeout",
            raw_log_path=str(raw_log),
            final_file=str(final_file),
        )
    except Exception as exc:
        return _packet_review_result(
            "packet_only_unusable",
            reason=f"review_guard_exception:{type(exc).__name__}",
            raw_log_path=str(raw_log),
            final_file=str(final_file),
        )
    if guard_proc.returncode != 0:
        return _packet_review_result(
            "packet_only_unusable",
            reason="review_guard_failed",
            raw_log_path=str(raw_log),
            final_file=str(final_file),
        )
    try:
        decoded = json.loads(guard_proc.stdout)
    except Exception:
        return _packet_review_result(
            "packet_only_unusable",
            reason="review_guard_malformed_json",
            raw_log_path=str(raw_log),
            final_file=str(final_file),
        )
    return _normalize_packet_review_result(decoded)


def _summarize_packet_review(raw: Any) -> dict[str, Any]:
    normalized = _normalize_packet_review_result(raw)
    return {key: normalized.get(key) for key in _PACKET_REVIEW_SUMMARY_KEYS}


def _apply_packet_review_to_leftover(
    leftover: dict[str, Any],
    packet_review: dict[str, Any],
) -> None:
    status = packet_review.get("status")
    leftover["packet_review_status"] = status
    if status == "packet_only_passed":
        leftover["requires_review"] = False
        leftover["requires_fixes"] = False
    elif status == "packet_only_failed":
        leftover["requires_review"] = False
        leftover["requires_fixes"] = True
    else:
        leftover["requires_review"] = True
        leftover["requires_fixes"] = False


def _status_after_packet_review(packet_review: dict[str, Any]) -> str:
    status = packet_review.get("status")
    if status == "packet_only_passed":
        return "staged_reviewed"
    if status == "packet_only_failed":
        return "staged_review_blocked"
    if status == "packet_only_unusable":
        return "staged_review_unavailable"
    return "staged_called"


def _finish_after_staged(
    *,
    args: dict[str, Any],
    repo: Path,
    git_head: str | None,
    initial_dirty: dict[str, Any],
    preflight: dict[str, Any] | None,
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
        preflight=preflight,
        codex_staged_result=staged_result,
    )
    if dirty_recovery_update:
        result["dirty_recovery"].update(dirty_recovery_update)

    post_dirty = staged._dirty_check(repo)
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

    if not post_dirty.get("is_clean"):
        leftover = _leftover_candidate(post_dirty, staged_result)
        should_review, reason = _should_auto_packet_review(
            staged_result=staged_result,
            post_dirty=post_dirty,
            checkpoint_requested=checkpoint_requested,
        )
        if should_review:
            packet_review = _summarize_packet_review(
                _run_packet_only_review(
                    repo=repo,
                    touched_files=leftover["touched_files"],
                    allowlist=allowlist,
                    dirty_baseline_paths=[],
                    staged_result=staged_result,
                )
            )
        else:
            packet_review = _packet_review_not_run(reason)
        result["codex_packet_review"] = packet_review
        _apply_packet_review_to_leftover(leftover, packet_review)
        result["leftover_candidate"] = leftover
        result["status"] = _status_after_packet_review(packet_review)
    else:
        result["codex_packet_review"] = _packet_review_not_run("no_candidate_diff")
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


def _dangerous_unsafe_reasons(dirty: dict[str, Any]) -> list[str]:
    reasons = dirty.get("unsafe_reasons", [])
    if not isinstance(reasons, list):
        return []
    return [str(reason) for reason in reasons if reason in _DANGEROUS_UNSAFE_REASONS]


def _dirty_ownership_metadata(dirty: dict[str, Any]) -> dict[str, list[str]]:
    ownership = {name: [] for name in _OWNER_CLASSES}
    classes = dirty.get("dirty_path_classes") if isinstance(dirty.get("dirty_path_classes"), dict) else {}
    all_paths = _current_dirty_paths(dirty)
    dangerous_reasons = _dangerous_unsafe_reasons(dirty)
    if dangerous_reasons:
        ownership["dangerous_conflict"] = all_paths
        return ownership

    known: set[str] = set()
    for rel_path in classes.get("cache", []) or []:
        if isinstance(rel_path, str):
            ownership["generated_cache"].append(rel_path)
            known.add(rel_path)
    for bucket in ("source", "test", "docs", "unknown"):
        for rel_path in classes.get(bucket, []) or []:
            if isinstance(rel_path, str) and rel_path not in known:
                ownership["unknown_unowned"].append(rel_path)
                known.add(rel_path)
    for rel_path in all_paths:
        if rel_path not in known:
            ownership["unknown_unowned"].append(rel_path)
            known.add(rel_path)
    return ownership


def _dirty_recovery_preview(
    dirty: dict[str, Any],
    *,
    mode: str,
    standing_authorization: bool,
    auto_clean_cache: bool,
    allow_isolated_worktree: bool,
    preflight_passed: bool,
) -> dict[str, Any]:
    ownership = _dirty_ownership_metadata(dirty)
    execute_cache_cleanup_allowed = bool(
        preflight_passed
        and standing_authorization
        and auto_clean_cache
        and _cache_only(dirty)
        and ownership.get("generated_cache")
    )
    cleanup_allowed = bool(mode != "dry_run" and execute_cache_cleanup_allowed)
    dirty_present = not bool(dirty.get("is_clean"))
    return {
        **staged._dirty_decision_metadata(dirty),
        "dirty_ownership": ownership,
        "cleanup_allowed": cleanup_allowed,
        "cache_cleanup_preview_allowed": execute_cache_cleanup_allowed,
        "would_cleanup_cache_in_execute": execute_cache_cleanup_allowed,
        "would_create_isolated_worktree": bool(
            mode != "dry_run"
            and preflight_passed
            and dirty_present
            and standing_authorization
            and allow_isolated_worktree
            and not cleanup_allowed
        ),
        "would_mutate_paths": [],
    }


def _dirty_recovery_context(args: dict[str, Any], *, dirty: dict[str, Any], mode: str, preflight: dict[str, Any] | None) -> dict[str, Any]:
    return _dirty_recovery_preview(
        dirty,
        mode=mode,
        standing_authorization=bool(args.get("standing_authorization")),
        auto_clean_cache=bool(args.get("auto_clean_cache")),
        allow_isolated_worktree=bool(args.get("allow_isolated_worktree")),
        preflight_passed=_preflight_passed(preflight),
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


def _clean_cache_dirty_paths(repo: Path, dirty: dict[str, Any]) -> list[str]:
    cleaned: list[str] = []
    for rel_path in dirty.get("dirty_path_classes", {}).get("cache", []):
        target = _safe_cache_path(repo, rel_path)
        if target is None or not (target.exists() or target.is_symlink()):
            continue
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
        cleaned.append(rel_path)
    return cleaned


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
    preflight = _codex_preflight(repo)

    if validated["mode"] == "dry_run":
        result = _base(
            "dry_run",
            repo=repo,
            git_head=git_head,
            dirty=dirty,
            preflight=preflight,
            resolved_allowlist=validated["allowlist"],
            would_call_staged=bool(dirty.get("is_clean")) and _preflight_passed(preflight),
        )
        result["dirty_recovery"].update(
            _dirty_recovery_context(args, dirty=dirty, mode="dry_run", preflight=preflight)
        )
        return _json_result(result)

    if not _preflight_passed(preflight):
        return _preflight_blocked_result(repo=repo, git_head=git_head, dirty=dirty, preflight=preflight, args=args)

    if dirty.get("is_clean"):
        staged_result = _call_staged(normalized)
        return _finish_after_staged(
            args=args,
            repo=repo,
            git_head=git_head,
            initial_dirty=dirty,
            preflight=preflight,
            normalized=normalized,
            allowlist=validated["allowlist"],
            staged_result=staged_result,
        )

    standing_auth = bool(args.get("standing_authorization"))
    if _cache_only(dirty) and standing_auth and bool(args.get("auto_clean_cache")):
        cleaned_paths = _clean_cache_dirty_paths(repo, dirty)
        after = staged._dirty_check(repo)
        if after.get("is_clean"):
            staged_result = _call_staged(normalized)
            return _finish_after_staged(
                args=args,
                repo=repo,
                git_head=git_head,
                initial_dirty=dirty,
                preflight=preflight,
                normalized=normalized,
                allowlist=validated["allowlist"],
                staged_result=staged_result,
                dirty_recovery_update={
                    **_dirty_recovery_context(args, dirty=dirty, mode="execute", preflight=preflight),
                    "strategy": "cache_cleanup",
                    "cache_cleaned_paths": cleaned_paths,
                    "post_cleanup_dirty_check": after,
                },
            )
        result = _base("dirty_recovery_required", repo=repo, git_head=git_head, dirty=dirty, preflight=preflight)
        result["dirty_recovery"].update(
            {
                **_dirty_recovery_context(args, dirty=dirty, mode="execute", preflight=preflight),
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
            result = _base("dirty_recovery_required", repo=repo, git_head=git_head, dirty=dirty, preflight=preflight)
            result["dirty_recovery"].update(
                {
                    **_dirty_recovery_context(args, dirty=dirty, mode="execute", preflight=preflight),
                    "strategy": "isolated_worktree_failed",
                    "error": str(exc),
                }
            )
            return _json_result(result)
        isolated_args = dict(normalized)
        isolated_args["workdir"] = worktree_info["path"]
        staged_result = _call_staged(isolated_args)
        return _finish_after_staged(
            args=args,
            repo=Path(worktree_info["path"]),
            git_head=git_head,
            initial_dirty=dirty,
            preflight=preflight,
            normalized=isolated_args,
            allowlist=validated["allowlist"],
            staged_result=staged_result,
            dirty_recovery_update={
                **_dirty_recovery_context(args, dirty=dirty, mode="execute", preflight=preflight),
                "strategy": "isolated_worktree",
                "isolated_worktree": worktree_info,
                "cache_cleaned_paths": [],
            },
        )

    result = _base("dirty_recovery_required", repo=repo, git_head=git_head, dirty=dirty, preflight=preflight)
    result["dirty_recovery"].update(_dirty_recovery_context(args, dirty=dirty, mode="execute", preflight=preflight))
    return _json_result(result)


_SCHEMA = {
    "name": "codex_workflow_run",
    "description": (
        "High-level Hermes orchestrator for dirty worktree recovery before guarded Codex "
        "candidate implementation. It validates repo, scope, policies, and local Codex "
        "workflow preflight; may remove cache-only dirty paths with standing authorization; "
        "may create an isolated clean worktree with standing authorization; delegates to "
        "codex_staged_implement; then runs bounded packet-only review for safe staged candidates."
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
            "auto_clean_cache": {"type": "boolean"},
            "allow_isolated_worktree": {"type": "boolean"},
            "stage_id": {"type": "string"},
            "checkpoint_verified_diff": {"type": "boolean"},
            "verification_evidence": {"type": "object"},
            "checkpoint_message": {"type": "string"},
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
