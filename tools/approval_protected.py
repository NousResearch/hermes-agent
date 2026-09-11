"""Experimental protected-edit review at the actual complete write boundary.

Default remains one-operation human approval. Only single local write_file and
replace operations may defer that gate; V4A/multi-file/delete/move and remote
backends retain it. Fuzzy matching and newline/BOM preparation finish BEFORE
review, so the reviewer never approves a guessed patch. Not a semantic sandbox:
recognizing instruction risk still depends on a fallible independent reviewer.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import inspect
import json
import os
from pathlib import Path
import re
import stat

from tools.approval_task import current_task, task_revoked

_MAX_BYTES = 32768
# A conservative backstop, not a complete instruction-language risk classifier.
_HIGH_RISK = re.compile(
    r"(?i)\b(ignore|bypass|disable|override|weaken|exfiltrat\w*|credentials?|secrets?|"
    r"passwords?|tokens?|permissions?|security|approvals?|allowlist|yolo)\b")


@dataclass
class _EditScope:
    path: str
    task_id: str
    deferred: bool = False
    target: str | None = None
    preimage: str | None = None


_scope: ContextVar[_EditScope | None] = ContextVar("protected_edit_scope", default=None)


def protected_edit_scope(kind):
    """No tool parameter can mint intent. This only carries exact write plumbing."""
    def decorate(fn):
        signature = inspect.signature(fn)

        @wraps(fn)
        def wrapped(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            params = bound.arguments
            supported = kind == "write" or params.get("mode") == "replace"
            path = params.get("path")
            scope = _EditScope(path, params["task_id"]) if supported and isinstance(path, str) else None
            token = _scope.set(scope)
            try:
                return fn(*args, **kwargs)
            finally:
                _scope.reset(token)
        return wrapped
    return decorate


def _smart_enabled():
    try:
        from hermes_cli.config import load_config_readonly, cfg_get
        return cfg_get(load_config_readonly(), "security", "protected_instruction_files",
                       "review_mode", default=None) == "smart"
    except Exception:
        return False


def defer_protected_gate(paths, task_id):
    scope = _scope.get()
    if (scope is None or paths != [scope.path] or task_id != scope.task_id
            or current_task() is None or not _smart_enabled()):
        return False
    try:
        from tools.file_tools import _get_file_ops, _resolve_path_for_task
        from tools.environments.local import LocalEnvironment
        if not isinstance(_get_file_ops(task_id).env, LocalEnvironment):
            return False
        scope.target = str(Path(_resolve_path_for_task(scope.path, task_id)).resolve())
        scope.deferred = True
        return True
    except Exception:
        return False


def record_patch_preimage(path, raw_content):
    scope = _scope.get()
    if scope is not None and scope.deferred:
        scope.preimage = raw_content


def _snapshot(path):
    try:
        metadata = os.stat(path)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MAX_BYTES:
        raise ValueError("unbounded or non-regular target")
    # No unbounded read even if the file grows after stat.
    with open(path, "rb") as source:
        data = source.read(_MAX_BYTES + 1)
    if len(data) > _MAX_BYTES:
        raise ValueError("oversized target")
    return data


def _human(scope, path):
    from tools.file_tools_write_guards import _request_protected_instruction_approval
    return _request_protected_instruction_approval([path], scope.task_id)


@contextmanager
def review_atomic_write(file_ops, path, content):
    """Yield the pinned canonical target or raise before the real atomic writer.

    File tools already hold their resolved-path lock. Check the preimage again
    after model latency and pin the canonical target so symlink aliases cannot
    silently retarget the reviewed operation. External filesystem writers are
    not transactionally locked by Hermes; this is not an OS compare-and-swap.
    """
    scope = _scope.get()
    if scope is None or not scope.deferred:
        yield path
        return
    from tools.file_tools_write_guards import _check_sensitive_path, _check_cross_profile_path
    from tools.approval_smart import _smart_approve
    from tools.environments.local import LocalEnvironment
    if not isinstance(file_ops.env, LocalEnvironment):
        raise PermissionError("Protected review backend changed; submit again for human approval")
    canonical = str(Path(path).resolve())
    hard_error = _check_sensitive_path(canonical, scope.task_id) or _check_cross_profile_path(canonical, scope.task_id)
    if hard_error:
        raise PermissionError(hard_error)
    approved = False
    record = current_task()
    try:
        before_bytes = _snapshot(canonical)
        before = before_bytes.decode("utf-8") if before_bytes is not None else None
        # A replace operation must not overwrite a preimage changed since matching.
        preimage_matches = scope.preimage is None or before == scope.preimage
        payload = json.dumps({"target": canonical, "before": before, "after": content}, ensure_ascii=False)
        # Full canonical target must be explicit in this experimental slice. A
        # basename alone is insufficient. This substring filter is NOT authorization:
        # prefix matches and negated references still require independent intent review.
        if (record is not None and canonical == scope.target and preimage_matches
                and canonical.replace("\\", "/") in record.raw_text.replace("\\", "/")
                and not _HIGH_RISK.search(content) and not _HIGH_RISK.search(before or "")
                and len(payload) <= _MAX_BYTES):
            verdict = _smart_approve("protected file edit", "Complete proposed instruction edit", proposed_edit=payload)
            approved = (verdict == "approve" and current_task() == record and not task_revoked()
                        and str(Path(path).resolve()) == canonical and _snapshot(canonical) == before_bytes)
    except (OSError, ValueError, UnicodeError):
        approved = False
    if task_revoked():
        raise PermissionError("Task ended or was cancelled before the protected write")
    if not approved:
        error = _human(scope, canonical)
        if error:
            raise PermissionError(error)
    # Liveness is checked after either reviewer or human latency, immediately
    # before handing the pinned path to the atomic writer.
    if task_revoked():
        raise PermissionError("Task ended or was cancelled before the protected write")
    yield canonical
