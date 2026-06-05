import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from tools.registry import registry


_ALLOWED_VERIFY_IDS = {"diff-check", "none"}
_SUPPORTED_MODES = {"execute", "dry_run_plan"}
_SUPPORTED_CONTINUE_POLICY = "stop-on-review-needed"
_SUPPORTED_DIRTY_POLICIES = {"require-clean"}
_DANGER_PARTS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    "dist",
    "build",
}
_WIDE_GLOBS = {"*", "**", "/*", "/**", "**/*", "./*", "./**", "**/*.py"}
_HIGH_RISK_VERIFY_NONE_WORDS = {
    "delete",
    "remove",
    "restart",
    "deploy",
    "migration",
    "migrate",
    "permission",
    "auth",
    "secret",
    "token",
    "key",
    "credential",
}
_PREVIEW_LIMIT = 8000
_LIST_LIMIT = 200
_DICT_LIMIT = 80
_STRING_LIMIT = 4000
_DIRTY_PATHS_LIMIT = 100


def _json_result(payload: dict[str, Any]) -> str:
    return json.dumps(_bound(payload), ensure_ascii=False)


def _bound(value: Any) -> Any:
    if isinstance(value, str):
        if len(value) <= _STRING_LIMIT:
            return value
        return value[: _STRING_LIMIT - 14] + "...[truncated]"
    if isinstance(value, list):
        items = [_bound(item) for item in value[:_LIST_LIMIT]]
        if len(value) > _LIST_LIMIT:
            items.append(f"...[{len(value) - _LIST_LIMIT} more]")
        return items
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in list(value.items())[:_DICT_LIMIT]:
            out[str(key)[:_STRING_LIMIT]] = _bound(item)
        if len(value) > _DICT_LIMIT:
            out["...[truncated_keys]"] = len(value) - _DICT_LIMIT
        return out
    return value


def _preview(text: str | None) -> str:
    return (text or "")[:_PREVIEW_LIMIT]


def _base_result(
    *,
    status: str,
    resolved_workdir: str | None = None,
    git_head: str | None = None,
    resolved_allowlist: dict[str, list[str]] | None = None,
    dirty_baseline_policy: str | None = None,
    dirty_check: dict[str, Any] | None = None,
    stage_plan_path: str | None = None,
    raw_dir: str | None = None,
    runner_exit_code: int | None = None,
    changed_files: list[str] | None = None,
    stopped_slice: Any = None,
    verification_policy: str | None = None,
    candidate_id: str | None = None,
    completion_trusted: bool = False,
    final_present: bool = False,
    limit_exceeded: bool = False,
    out_of_scope_files: list[str] | None = None,
    candidate_disposition: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    if candidate_disposition is None:
        if status in {"ready_for_review", "review_needed"}:
            candidate_disposition = "pending_review"
        elif status == "takeover_candidate":
            candidate_disposition = "takeover_required"
        elif status in {"runner_unusable", "timeout"}:
            candidate_disposition = "unavailable"
        else:
            candidate_disposition = "rejected"
    result = {
        "status": status,
        "candidate_id": candidate_id,
        "candidate_disposition": candidate_disposition,
        "completion_trusted": completion_trusted,
        "final_present": final_present,
        "limit_exceeded": limit_exceeded,
        "out_of_scope_files": out_of_scope_files or [],
        "resolved_workdir": resolved_workdir,
        "git_head": git_head,
        "resolved_allowlist": resolved_allowlist or {"files": [], "globs": []},
        "dirty_baseline_policy": dirty_baseline_policy,
        "dirty_check": dirty_check or {"is_clean": None, "porcelain_count": None},
        "stage_plan_path": stage_plan_path,
        "raw_dir": raw_dir,
        "runner_exit_code": runner_exit_code,
        "changed_files": changed_files or [],
        "stopped_slice": stopped_slice,
        "verification_policy": verification_policy,
    }
    result.update(extra)
    return result


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )


def _resolve_repo_root(workdir: Any) -> tuple[Path | None, str | None]:
    if not isinstance(workdir, str) or not workdir.strip():
        return None, None
    candidate = Path(workdir).expanduser().resolve()
    proc = _git(candidate, "rev-parse", "--show-toplevel")
    if proc.returncode != 0:
        return None, None
    root = Path(proc.stdout.strip()).resolve()
    if candidate != root:
        return None, None
    head_proc = _git(root, "rev-parse", "HEAD")
    git_head = head_proc.stdout.strip() if head_proc.returncode == 0 else None
    return root, git_head


def _has_danger_part(path: str) -> bool:
    return any(part in _DANGER_PARTS for part in Path(path).parts)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _literal_scope_prefix_parts(value: str, *, is_glob: bool) -> list[str]:
    parts = list(Path(value).parts)
    if not is_glob:
        return parts[:-1]
    literal_parts: list[str] = []
    for part in parts:
        if any(ch in part for ch in "*?["):
            break
        literal_parts.append(part)
    return literal_parts


def _scope_prefix_stays_inside_repo(value: str, *, repo: Path, is_glob: bool) -> bool:
    current = repo
    for part in _literal_scope_prefix_parts(value, is_glob=is_glob):
        current = current / part
        if current.is_symlink() or current.exists():
            try:
                resolved = current.resolve(strict=False)
            except OSError:
                return False
            if not _is_relative_to(resolved, repo):
                return False
        elif is_glob:
            # A glob under a non-existing directory prefix is too ambiguous for
            # this write-capable entry point. Keep v1 fail-closed.
            return False
        else:
            break
    return True


def _validate_scope_item(item: Any, *, repo: Path, is_glob: bool) -> str | None:
    if not isinstance(item, str) or not item.strip():
        return None
    value = item.strip()
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or _has_danger_part(value):
        return None
    if not _scope_prefix_stays_inside_repo(value, repo=repo, is_glob=is_glob):
        return None
    if is_glob:
        if value in _WIDE_GLOBS or value.startswith("/") or value.startswith("**/"):
            return None
        return value

    target = repo / value
    if target.exists() or target.is_symlink():
        resolved = target.resolve(strict=False)
        if not _is_relative_to(resolved, repo):
            return None
    return value


def _validate_scope(args: dict[str, Any], repo: Path) -> tuple[dict[str, list[str]] | None, str | None]:
    raw_files = args.get("allowed_files") or []
    raw_globs = args.get("allowed_globs") or []
    if not isinstance(raw_files, list) or not isinstance(raw_globs, list):
        return None, "scope must be lists"

    files: list[str] = []
    globs: list[str] = []
    for item in raw_files:
        value = _validate_scope_item(item, repo=repo, is_glob=False)
        if value is None:
            return None, "invalid allowed_files entry"
        files.append(value)
    for item in raw_globs:
        value = _validate_scope_item(item, repo=repo, is_glob=True)
        if value is None:
            return None, "invalid allowed_globs entry"
        globs.append(value)

    if not files and not globs:
        return None, "scope is required"
    return {"files": files, "globs": globs}, None


def _dirty_check(repo: Path) -> dict[str, Any]:
    proc = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    lines = [line for line in proc.stdout.splitlines() if line]
    paths: list[str] = []
    for line in lines:
        raw_path = line[3:] if len(line) > 3 else line
        if " -> " in raw_path:
            raw_path = raw_path.split(" -> ", 1)[1]
        paths.append(raw_path.strip())
    return {
        "is_clean": proc.returncode == 0 and not lines,
        "porcelain_count": len(lines),
        "dirty_count": len(paths),
        "dirty_paths": paths[:_DIRTY_PATHS_LIMIT],
        "dirty_paths_truncated": len(paths) > _DIRTY_PATHS_LIMIT,
    }


def _verification_policy(verify_ids: list[str]) -> str:
    if verify_ids == ["none"]:
        return "deferred_to_hermes"
    return "diff-check"


def _verify_none_high_risk(task: str) -> bool:
    lowered = task.lower()
    return any(word in lowered for word in _HIGH_RISK_VERIFY_NONE_WORDS)


def _runner_script() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "runtime" / "codex_stage_runner.py"


def _run_runner(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60 * 60,
    )


def _write_stage_files(
    *,
    repo: Path,
    task: str,
    allowlist: dict[str, list[str]],
    verify_ids: list[str],
    continue_policy: str,
    dirty_policy: str,
) -> tuple[Path, Path, Path]:
    temp_base = Path("/tmp") if Path("/tmp").is_dir() else Path(tempfile.gettempdir())
    temp_root = Path(tempfile.mkdtemp(prefix="hermes-codex-stage-", dir=str(temp_base)))
    if _is_relative_to(temp_root.resolve(strict=False), repo):
        raise RuntimeError("unsafe_stage_temp_dir")
    raw_dir = temp_root / "raw"
    raw_dir.mkdir(mode=0o700)
    plan_path = temp_root / "stage_plan.json"
    prompt_path = temp_root / "prompt.txt"
    prompt_path.write_text(task, encoding="utf-8")
    plan = {
        "repo": str(repo),
        "continue_policy": continue_policy,
        "dirty_baseline_policy": dirty_policy,
        "slices": [
            {
                "id": "slice-1",
                "prompt_file": str(prompt_path),
                "allowed_files": allowlist["files"],
                "allowed_globs": allowlist["globs"],
                "verify_cmd_ids": verify_ids,
                "dirty_baseline_policy": dirty_policy,
            }
        ],
    }
    plan_path.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    return plan_path, prompt_path, raw_dir


def _dry_run_stage_plan(
    *,
    repo: Path,
    task: str,
    allowlist: dict[str, list[str]],
    verify_ids: list[str],
    continue_policy: str,
    dirty_policy: str,
) -> dict[str, Any]:
    return {
        "repo": str(repo),
        "continue_policy": continue_policy,
        "dirty_baseline_policy": dirty_policy,
        "slices": [
            {
                "id": "slice-1",
                "goal": task,
                "prompt_file": "<create-outside-repo-before-execution>",
                "allowed_files": allowlist["files"],
                "allowed_globs": allowlist["globs"],
                "verify_cmd_ids": verify_ids,
                "dirty_baseline_policy": dirty_policy,
            }
        ],
    }


def _dry_run_plan_result(
    *,
    repo: Path,
    git_head: str | None,
    task_text: str,
    verify_ids: list[str],
    continue_policy: str,
    dirty_policy: str,
    verification_policy: str,
    allowlist: dict[str, list[str]] | None,
    scope_error: str | None,
) -> dict[str, Any]:
    dirty = _dirty_check(repo)
    empty_allowlist = {"files": [], "globs": []}
    if scope_error:
        risk = "needs_scope" if scope_error == "scope is required" else "unsupported"
        next_action = "provide_explicit_scope" if risk == "needs_scope" else "provide_narrow_explicit_scope"
        return _base_result(
            status="dry_run_plan",
            resolved_workdir=str(repo),
            git_head=git_head,
            resolved_allowlist=empty_allowlist,
            dirty_baseline_policy=dirty_policy,
            dirty_check=dirty,
            verification_policy=verification_policy,
            risk_classification=risk,
            needs_user_confirmation=True,
            proposed_allowlist=empty_allowlist,
            proposed_stage_plan=None,
            next_required_action=next_action,
            reason=scope_error,
        )
    assert allowlist is not None
    if not dirty["is_clean"]:
        return _base_result(
            status="dry_run_plan",
            resolved_workdir=str(repo),
            git_head=git_head,
            resolved_allowlist=allowlist,
            dirty_baseline_policy=dirty_policy,
            dirty_check=dirty,
            verification_policy=verification_policy,
            risk_classification="unsupported",
            needs_user_confirmation=True,
            proposed_allowlist=allowlist,
            proposed_stage_plan=None,
            next_required_action="clean_worktree_before_execution",
            reason="dirty_worktree",
        )
    return _base_result(
        status="dry_run_plan",
        resolved_workdir=str(repo),
        git_head=git_head,
        resolved_allowlist=allowlist,
        dirty_baseline_policy=dirty_policy,
        dirty_check=dirty,
        verification_policy=verification_policy,
        risk_classification="low",
        needs_user_confirmation=True,
        proposed_allowlist=allowlist,
        proposed_stage_plan=_dry_run_stage_plan(
            repo=repo,
            task=task_text,
            allowlist=allowlist,
            verify_ids=verify_ids,
            continue_policy=continue_policy,
            dirty_policy=dirty_policy,
        ),
        next_required_action="confirm_or_execute_with_explicit_scope",
    )


def _runner_impl_result(runner_payload: dict[str, Any]) -> dict[str, Any]:
    slice_results = runner_payload.get("slice_results")
    if isinstance(slice_results, list) and slice_results:
        first = slice_results[-1]
        if isinstance(first, dict):
            impl = first.get("impl_guard_result")
            if isinstance(impl, dict):
                return impl
    return {}


def _runner_changed_files(runner_payload: dict[str, Any]) -> list[str]:
    changed = runner_payload.get("changed_files")
    if isinstance(changed, list):
        return [str(item) for item in changed if isinstance(item, str)]
    impl_changed = _runner_impl_result(runner_payload).get("changed_files")
    if isinstance(impl_changed, list):
        return [str(item) for item in impl_changed if isinstance(item, str)]
    return []


def _candidate_id(
    *,
    git_head: str | None,
    task_text: str,
    allowlist: dict[str, list[str]],
    stage_plan_path: Path,
    stopped_slice: Any = None,
) -> str:
    slice_id = stopped_slice if isinstance(stopped_slice, str) and stopped_slice else "slice-1"
    seed = {
        "git_head": git_head or str(stage_plan_path),
        "slice_id": slice_id,
        "task_hash": hashlib.sha256(task_text.encode("utf-8")).hexdigest(),
        "allowlist": allowlist,
    }
    return hashlib.sha256(json.dumps(seed, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _truthy_flag(payload: dict[str, Any], names: set[str]) -> bool:
    return any(bool(payload.get(name)) for name in names)


def _limit_exceeded(runner_payload: dict[str, Any], *, malformed_oversized: bool = False) -> bool:
    if malformed_oversized:
        return True
    impl = _runner_impl_result(runner_payload)
    for payload in (runner_payload, impl):
        if _truthy_flag(
            payload,
            {
                "limit_exceeded",
                "stdout_flood",
                "diff_flood",
                "source_flood",
                "json_flood",
                "output_limited",
                "diff_flood_detected",
                "source_flood_detected",
                "json_field_flood_detected",
                "terminated_by_guard",
            },
        ):
            return True
        reason = str(
            payload.get("reason")
            or payload.get("limit_reason")
            or payload.get("codex_reason")
            or payload.get("error")
            or ""
        ).lower()
        if "limit" in reason or "flood" in reason or "oversized" in reason:
            return True
    return False


def _has_final_issue(runner_payload: dict[str, Any]) -> bool:
    impl = _runner_impl_result(runner_payload)
    for payload in (runner_payload, impl):
        if _truthy_flag(payload, {"final_missing", "missing_final", "final_error"}):
            return True
        reason = str(payload.get("reason") or payload.get("error") or "").lower()
        if "final" in reason and ("missing" in reason or "error" in reason or "failed" in reason):
            return True
    return False


def _final_present(runner_payload: dict[str, Any]) -> bool:
    if _has_final_issue(runner_payload):
        return False
    impl = _runner_impl_result(runner_payload)
    for payload in (impl, runner_payload):
        for name in ("final_file", "final", "final_output", "output_last_message"):
            value = payload.get(name)
            if isinstance(value, str) and value.strip():
                return True
            if value is True:
                return True
    return False


def _out_of_scope_files(runner_payload: dict[str, Any]) -> list[str]:
    impl = _runner_impl_result(runner_payload)
    files: list[str] = []
    for payload in (runner_payload, impl):
        for name in ("out_of_scope_files", "allowlist_violations"):
            value = payload.get(name)
            if isinstance(value, list):
                files.extend(str(item) for item in value if isinstance(item, str))
    return sorted(set(files))


def _candidate_disposition(
    status: str,
    *,
    limit_exceeded: bool,
    out_of_scope_files: list[str] | None = None,
    final_issue: bool = False,
) -> str:
    if status in {"runner_unusable", "timeout"}:
        return "unavailable"
    if status == "takeover_candidate" or limit_exceeded or final_issue or bool(out_of_scope_files):
        return "takeover_required"
    if status in {"ready_for_review", "review_needed"}:
        return "pending_review"
    return "rejected"


def _completion_trusted(
    runner_payload: dict[str, Any],
    *,
    status: str,
    runner_exit_code: int,
    limit_exceeded: bool,
    out_of_scope_files: list[str] | None = None,
) -> bool:
    if status not in {"ready_for_review", "review_needed"} or runner_exit_code != 0:
        return False
    if limit_exceeded or _has_final_issue(runner_payload) or bool(out_of_scope_files):
        return False
    impl = _runner_impl_result(runner_payload)
    if impl.get("trusted_completion") is False:
        return False
    impl_status = str(impl.get("status") or "")
    runner_status = str(runner_payload.get("status") or "")
    return runner_status in {"completed", "passed"} or impl_status in {"completed", "passed"}


def _dry_run_contract_rejection(
    *,
    reason: str,
    next_action: str = "provide_supported_policy",
    resolved_workdir: str | None = None,
    git_head: str | None = None,
    dirty_baseline_policy: str | None = None,
    dirty_check: dict[str, Any] | None = None,
    verification_policy: str | None = None,
) -> dict[str, Any]:
    empty_allowlist = {"files": [], "globs": []}
    return _base_result(
        status="dry_run_plan",
        resolved_workdir=resolved_workdir,
        git_head=git_head,
        resolved_allowlist=empty_allowlist,
        dirty_baseline_policy=dirty_baseline_policy,
        dirty_check=dirty_check,
        verification_policy=verification_policy,
        risk_classification="unsupported",
        needs_user_confirmation=True,
        proposed_allowlist=empty_allowlist,
        proposed_stage_plan=None,
        next_required_action=next_action,
        reason=reason,
    )


def _dry_run_repo_context(args: dict[str, Any], *, dirty_policy: str | None) -> tuple[str | None, str | None, dict[str, Any] | None]:
    repo, git_head = _resolve_repo_root(args.get("workdir"))
    if repo is None:
        return None, None, None
    return str(repo), git_head, _dirty_check(repo)


def _codex_staged_dry_run_plan(args: dict[str, Any]) -> dict[str, Any]:
    default_verification_policy = _verification_policy(["diff-check"])
    default_dirty_policy = "require-clean"
    resolved_workdir, git_head, dirty_check = _dry_run_repo_context(args, dirty_policy=default_dirty_policy)

    continue_policy = args.get("continue_policy", _SUPPORTED_CONTINUE_POLICY)
    if continue_policy != _SUPPORTED_CONTINUE_POLICY:
        return _dry_run_contract_rejection(
            reason="unsupported_continue_policy",
            resolved_workdir=resolved_workdir,
            git_head=git_head,
            dirty_baseline_policy=default_dirty_policy,
            dirty_check=dirty_check,
            verification_policy=default_verification_policy,
        )

    dirty_policy = args.get("dirty_baseline_policy", default_dirty_policy)
    if dirty_policy not in _SUPPORTED_DIRTY_POLICIES:
        return _dry_run_contract_rejection(
            reason="unsupported_dirty_policy",
            resolved_workdir=resolved_workdir,
            git_head=git_head,
            dirty_baseline_policy=dirty_policy,
            dirty_check=dirty_check,
            verification_policy=default_verification_policy,
        )

    verify_ids = args.get("verify_cmd_ids", ["diff-check"])
    if verify_ids is None:
        verify_ids = ["diff-check"]
    if not isinstance(verify_ids, list) or not verify_ids:
        return _dry_run_contract_rejection(
            reason="rejected_verify_policy",
            resolved_workdir=resolved_workdir,
            git_head=git_head,
            dirty_baseline_policy=dirty_policy,
            dirty_check=dirty_check,
        )
    if any(not isinstance(item, str) or item not in _ALLOWED_VERIFY_IDS for item in verify_ids):
        return _dry_run_contract_rejection(
            reason="unsupported_verify_cmd_id",
            resolved_workdir=resolved_workdir,
            git_head=git_head,
            dirty_baseline_policy=dirty_policy,
            dirty_check=dirty_check,
        )
    if "none" in verify_ids and verify_ids != ["none"]:
        return _dry_run_contract_rejection(
            reason="rejected_verify_policy",
            resolved_workdir=resolved_workdir,
            git_head=git_head,
            dirty_baseline_policy=dirty_policy,
            dirty_check=dirty_check,
        )
    verification_policy = _verification_policy(verify_ids)

    task = args.get("task")
    if not isinstance(task, str) or not task.strip():
        return _dry_run_contract_rejection(
            reason="missing_task",
            next_action="provide_task_and_explicit_scope",
            resolved_workdir=resolved_workdir,
            git_head=git_head,
            dirty_baseline_policy=dirty_policy,
            dirty_check=dirty_check,
            verification_policy=verification_policy,
        )
    task_text = task.strip()
    if verify_ids == ["none"] and _verify_none_high_risk(task_text):
        return _dry_run_contract_rejection(
            reason="verify_none_high_risk_task",
            resolved_workdir=resolved_workdir,
            git_head=git_head,
            dirty_baseline_policy=dirty_policy,
            dirty_check=dirty_check,
            verification_policy=verification_policy,
        )

    repo, git_head = _resolve_repo_root(args.get("workdir"))
    if repo is None:
        return _dry_run_contract_rejection(
            reason="invalid_workdir",
            next_action="provide_valid_workdir",
            dirty_baseline_policy=dirty_policy,
            verification_policy=verification_policy,
        )

    allowlist, scope_error = _validate_scope(args, repo)
    return _dry_run_plan_result(
        repo=repo,
        git_head=git_head,
        task_text=task_text,
        verify_ids=verify_ids,
        continue_policy=continue_policy,
        dirty_policy=dirty_policy,
        verification_policy=verification_policy,
        allowlist=allowlist,
        scope_error=scope_error,
    )


def _map_runner_status(
    runner_payload: dict[str, Any],
    *,
    verify_ids: list[str],
    runner_exit_code: int,
) -> tuple[str, str | None]:
    runner_status = str(runner_payload.get("status") or "")
    runner_reason = str(runner_payload.get("reason") or "")
    verification_status = str(runner_payload.get("verification_status") or "")
    impl_status = str(_runner_impl_result(runner_payload).get("status") or "")
    if runner_status == "stopped":
        if "review_needed" in runner_reason or impl_status == "review_needed":
            return "review_needed", "run_read_only_review"
        if "blocked_by_allowlist" in runner_reason or impl_status == "blocked_by_allowlist":
            return "blocked_by_allowlist", None
        if "takeover_candidate" in runner_reason or impl_status == "takeover_candidate":
            return "takeover_candidate", None
        if "failed" in runner_reason or impl_status == "failed":
            return "implementation_failed", None
        if "unusable" in runner_reason or impl_status == "unusable":
            return "runner_unusable", None
        return "runner_unusable", None
    if runner_exit_code != 0 and runner_status not in {
        "stopped",
        "blocked_by_allowlist",
        "takeover_candidate",
        "failed",
        "unusable",
        "malformed",
        "timeout",
    }:
        return "runner_unusable", None
    if runner_status == "review_needed":
        return "review_needed", "run_read_only_review"
    if runner_status in {"completed", "passed"} and verification_status in {"", "passed"}:
        if verify_ids == ["none"]:
            return "review_needed", "run_read_only_review"
        return "ready_for_review", "run_hermes_verification"
    if runner_status == "blocked_by_allowlist":
        return "blocked_by_allowlist", None
    if runner_status == "takeover_candidate":
        return "takeover_candidate", None
    if runner_status == "failed":
        return "implementation_failed", None
    if runner_status == "unusable":
        return "runner_unusable", None
    if runner_status in {"malformed", "timeout"}:
        return runner_status, None
    return "malformed", None


def codex_staged_implement(args: dict[str, Any]) -> str:
    if not isinstance(args, dict):
        args = {}

    mode = args.get("mode", "execute")
    if mode not in _SUPPORTED_MODES:
        return _json_result(_base_result(status="unsupported_mode", mode=mode))
    if mode == "dry_run_plan":
        return _json_result(_codex_staged_dry_run_plan(args))

    continue_policy = args.get("continue_policy", _SUPPORTED_CONTINUE_POLICY)
    if continue_policy != _SUPPORTED_CONTINUE_POLICY:
        return _json_result(_base_result(status="rejected_verify_policy"))

    dirty_policy = args.get("dirty_baseline_policy", "require-clean")
    if dirty_policy not in _SUPPORTED_DIRTY_POLICIES:
        return _json_result(_base_result(status="unsupported_dirty_policy", dirty_baseline_policy=dirty_policy))

    verify_ids = args.get("verify_cmd_ids", ["diff-check"])
    if verify_ids is None:
        verify_ids = ["diff-check"]
    if not isinstance(verify_ids, list) or not verify_ids:
        return _json_result(_base_result(status="rejected_verify_policy"))
    if any(not isinstance(item, str) or item not in _ALLOWED_VERIFY_IDS for item in verify_ids):
        return _json_result(_base_result(status="unsupported_verify_cmd_id"))
    if "none" in verify_ids and verify_ids != ["none"]:
        return _json_result(_base_result(status="rejected_verify_policy"))

    task = args.get("task")
    if not isinstance(task, str) or not task.strip():
        return _json_result(_base_result(status="rejected_scope"))
    task_text = task.strip()
    if verify_ids == ["none"] and _verify_none_high_risk(task_text):
        return _json_result(_base_result(status="rejected_verify_policy", reason="verify_none_high_risk_task"))

    repo, git_head = _resolve_repo_root(args.get("workdir"))
    if repo is None:
        return _json_result(_base_result(status="rejected_scope"))
    resolved_workdir = str(repo)
    verification_policy = _verification_policy(verify_ids)

    allowlist, scope_error = _validate_scope(args, repo)
    if scope_error:
        return _json_result(
            _base_result(
                status="rejected_scope",
                resolved_workdir=resolved_workdir,
                git_head=git_head,
                verification_policy=verification_policy,
                error=scope_error,
            )
        )

    dirty = _dirty_check(repo)
    if not dirty["is_clean"] and dirty_policy == "require-clean":
        return _json_result(
            _base_result(
                status="dirty_worktree",
                resolved_workdir=resolved_workdir,
                git_head=git_head,
                resolved_allowlist=allowlist,
                dirty_baseline_policy=dirty_policy,
                dirty_check=dirty,
                verification_policy=verification_policy,
            )
        )

    try:
        plan_path, prompt_path, raw_dir = _write_stage_files(
            repo=repo,
            task=task_text,
            allowlist=allowlist,
            verify_ids=verify_ids,
            continue_policy=continue_policy,
            dirty_policy=dirty_policy,
        )
    except Exception as exc:
        return _json_result(
            _base_result(
                status="runner_unusable",
                reason="stage_file_creation_failed",
                resolved_workdir=resolved_workdir,
                git_head=git_head,
                resolved_allowlist=allowlist,
                dirty_baseline_policy=dirty_policy,
                dirty_check=dirty,
                runner_exit_code=None,
                verification_policy=verification_policy,
                next_required_action=None,
                error=type(exc).__name__,
            )
        )
    argv = [
        sys.executable,
        str(_runner_script()),
        "--plan-file",
        str(plan_path),
        "--raw-dir",
        str(raw_dir),
    ]

    try:
        proc = _run_runner(argv)
    except subprocess.TimeoutExpired as exc:
        return _json_result(
            _base_result(
                status="timeout",
                resolved_workdir=resolved_workdir,
                git_head=git_head,
                resolved_allowlist=allowlist,
                dirty_baseline_policy=dirty_policy,
                dirty_check=dirty,
                stage_plan_path=str(plan_path),
                raw_dir=str(raw_dir),
                runner_exit_code=None,
                verification_policy=verification_policy,
                runner_stdout_preview=_preview(exc.stdout if isinstance(exc.stdout, str) else ""),
                runner_stderr_preview=_preview(exc.stderr if isinstance(exc.stderr, str) else ""),
            )
        )

    stdout_preview = _preview(proc.stdout)
    stderr_preview = _preview(proc.stderr)
    try:
        runner_payload = json.loads(proc.stdout or "{}")
        if not isinstance(runner_payload, dict):
            raise ValueError("runner JSON was not an object")
    except Exception:
        malformed_oversized = len(proc.stdout or "") > _PREVIEW_LIMIT or len(proc.stderr or "") > _PREVIEW_LIMIT
        return _json_result(
            _base_result(
                status="malformed",
                resolved_workdir=resolved_workdir,
                git_head=git_head,
                resolved_allowlist=allowlist,
                dirty_baseline_policy=dirty_policy,
                dirty_check=dirty,
                stage_plan_path=str(plan_path),
                raw_dir=str(raw_dir),
                runner_exit_code=proc.returncode,
                verification_policy=verification_policy,
                candidate_id=_candidate_id(
                    git_head=git_head,
                    task_text=task_text,
                    allowlist=allowlist,
                    stage_plan_path=plan_path,
                ),
                limit_exceeded=_limit_exceeded({}, malformed_oversized=malformed_oversized),
                candidate_disposition=_candidate_disposition("malformed", limit_exceeded=malformed_oversized),
                runner_stdout_preview=stdout_preview,
                runner_stderr_preview=stderr_preview,
            )
        )

    status, next_action = _map_runner_status(runner_payload, verify_ids=verify_ids, runner_exit_code=proc.returncode)
    limit_exceeded = _limit_exceeded(runner_payload)
    final_present = _final_present(runner_payload)
    final_issue = _has_final_issue(runner_payload)
    out_of_scope_files = _out_of_scope_files(runner_payload)
    candidate_id = _candidate_id(
        git_head=git_head,
        task_text=task_text,
        allowlist=allowlist,
        stage_plan_path=plan_path,
        stopped_slice=runner_payload.get("stopped_slice"),
    )
    return _json_result(
        _base_result(
            status=status,
            resolved_workdir=resolved_workdir,
            git_head=git_head,
            resolved_allowlist=allowlist,
            dirty_baseline_policy=dirty_policy,
            dirty_check=dirty,
            stage_plan_path=str(plan_path),
            raw_dir=str(raw_dir),
            runner_exit_code=proc.returncode,
            changed_files=_runner_changed_files(runner_payload),
            stopped_slice=runner_payload.get("stopped_slice"),
            verification_policy=verification_policy,
            next_required_action=next_action,
            candidate_id=candidate_id,
            completion_trusted=_completion_trusted(
                runner_payload,
                status=status,
                runner_exit_code=proc.returncode,
                limit_exceeded=limit_exceeded,
                out_of_scope_files=out_of_scope_files,
            ),
            final_present=final_present,
            limit_exceeded=limit_exceeded,
            out_of_scope_files=out_of_scope_files,
            candidate_disposition=_candidate_disposition(
                status,
                limit_exceeded=limit_exceeded,
                out_of_scope_files=out_of_scope_files,
                final_issue=final_issue,
            ),
            runner_stdout_preview=stdout_preview
            if status in {"malformed", "implementation_failed", "runner_unusable"}
            else "",
            runner_stderr_preview=stderr_preview
            if status in {"malformed", "implementation_failed", "runner_unusable"}
            else "",
        )
    )


_SCHEMA = {
    "name": "codex_staged_implement",
    "description": (
        "Default Hermes channel for Codex candidate implementation work. Use this instead of "
        "calling raw `codex-yuna exec` / `codex exec` through terminal. Requires "
        "explicit `workdir`, `task`, and narrow `allowed_files` / `allowed_globs`; "
        "dirty worktrees require an explicit dirty_baseline_policy and the result is "
        "candidate work only. `completion_trusted` means the runner/implementation guard "
        "appeared to finish without known limit/flood/final-output issues, not that the task "
        "is complete. Hermes must verify the diff, run tests, and review before accepting."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "workdir": {"type": "string", "description": "Git repository root to modify."},
            "task": {"type": "string", "description": "Implementation task."},
            "allowed_files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Relative file paths Codex may modify.",
            },
            "allowed_globs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Narrow relative glob patterns Codex may modify.",
            },
            "verify_cmd_ids": {
                "type": "array",
                "items": {"type": "string", "enum": ["diff-check", "none"]},
                "description": "Verification policy ids.",
            },
            "continue_policy": {
                "type": "string",
                "enum": ["stop-on-review-needed"],
            },
            "dirty_baseline_policy": {
                "type": "string",
                "enum": ["require-clean"],
                "description": "Dirty baseline policy. Phase 13 v1 is fail-closed and only supports clean worktrees.",
            },
            "mode": {
                "type": "string",
                "enum": ["execute", "dry_run_plan"],
                "description": "Use dry_run_plan to propose a bounded stage plan without invoking runner/Codex or writing repo files.",
            },
        },
        "required": ["workdir", "task"],
    },
}


registry.register(
    name="codex_staged_implement",
    toolset="codex_staged_implement",
    schema=_SCHEMA,
    handler=lambda args, **kwargs: codex_staged_implement(args),
    description=_SCHEMA["description"],
)
