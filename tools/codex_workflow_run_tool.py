import datetime as _datetime
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from tools import codex_staged_implement_tool as staged
from tools.registry import registry


_SUPPORTED_MODES = {"execute", "dry_run"}
_SUPPORTED_CONTINUE_POLICY = "stop-on-review-needed"
_SUPPORTED_DIRTY_POLICY = "require-clean"
_ALLOWED_VERIFY_IDS = {"diff-check", "none"}


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
        "mode": "execute",
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

    if validated["mode"] == "dry_run":
        return _json_result(
            _base(
                "dry_run",
                repo=repo,
                git_head=git_head,
                dirty=dirty,
                resolved_allowlist=validated["allowlist"],
                would_call_staged=bool(dirty.get("is_clean")),
            )
        )

    if dirty.get("is_clean"):
        staged_result = _call_staged(normalized)
        return _json_result(
            _base(
                "staged_called",
                repo=repo,
                git_head=git_head,
                dirty=dirty,
                codex_staged_result=staged_result,
            )
        )

    standing_auth = bool(args.get("standing_authorization"))
    if _cache_only(dirty) and standing_auth and bool(args.get("auto_clean_cache")):
        cleaned_paths = _clean_cache_dirty_paths(repo, dirty)
        after = staged._dirty_check(repo)
        if after.get("is_clean"):
            staged_result = _call_staged(normalized)
            result = _base(
                "staged_called",
                repo=repo,
                git_head=git_head,
                dirty=dirty,
                codex_staged_result=staged_result,
            )
            result["dirty_recovery"].update(
                {
                    "strategy": "cache_cleanup",
                    "cache_cleaned_paths": cleaned_paths,
                    "post_cleanup_dirty_check": after,
                }
            )
            return _json_result(result)
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
        staged_result = _call_staged(isolated_args)
        result = _base(
            "staged_called",
            repo=repo,
            git_head=git_head,
            dirty=dirty,
            codex_staged_result=staged_result,
        )
        result["dirty_recovery"].update({"strategy": "isolated_worktree", "isolated_worktree": worktree_info})
        return _json_result(result)

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
            "auto_clean_cache": {"type": "boolean"},
            "allow_isolated_worktree": {"type": "boolean"},
            "stage_id": {"type": "string"},
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
