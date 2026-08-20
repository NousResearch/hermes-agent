#!/usr/bin/env python3
"""
Skill Manager Tool -- Agent-Managed Skill Creation & Editing

Allows the agent to create, update, and delete skills, turning successful
approaches into reusable procedural knowledge. New skills are created in
~/.hermes/skills/. Existing skills (bundled, hub-installed, or user-created)
can be modified or deleted wherever they live.

Skills are the agent's procedural memory: they capture *how to do a specific
type of task* based on proven experience. General memory (MEMORY.md, USER.md) is
broad and declarative. Skills are narrow and actionable.

Actions:
  create     -- Create a new skill (SKILL.md + directory structure)
  edit       -- Replace the SKILL.md content of a user skill (full rewrite)
  patch      -- Targeted find-and-replace within SKILL.md or any supporting file
  delete     -- Remove a user skill entirely
  write_file -- Add/overwrite a supporting file (reference, template, script, asset)
  remove_file-- Remove a supporting file from a user skill

Directory layout for user skills:
    ~/.hermes/skills/
    ├── my-skill/
    │   ├── SKILL.md
    │   ├── references/
    │   ├── templates/
    │   ├── scripts/
    │   └── assets/
    └── category-name/
        └── another-skill/
            └── SKILL.md
"""

import errno as _errno  # noqa: E402  -- errno constants used in lock-acquisition classification.

# POSIX-only interprocess lock.  Must NOT be a hard import: Windows Python
# does not ship ``fcntl`` and several CI / Docker images strip it.  We
# defer to the active platform selector further down (``_IS_WINDOWS`` /
# ``_IS_POSIX``) and refuse to proceed without a real lock primitive on
# either platform — fail-closed, not a silent no-op.
try:
    import fcntl as _fcntl  # type: ignore[unused-ignore]
except ImportError:  # pragma: no cover -- POSIX without fcntl (stripped build)
    _fcntl = None  # type: ignore[assignment]

import hashlib as _hashlib  # noqa: E402  -- stable per-skill lock key derivation.
import json
import logging
import os
import re
import secrets as _secrets  # noqa: E402  -- staging-dir random suffix.
import shutil
import stat
import tempfile
import contextvars as _ctxvars
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_constants import get_hermes_home, display_hermes_home
from utils import atomic_replace, is_truthy_value
from hermes_cli.config import cfg_get
from agent.skill_utils import (
    extract_skill_description,
    is_skill_description_truncated_for_prompt,
    parse_frontmatter as _parse_frontmatter,
    SKILL_PROMPT_DESC_LIMIT,
)
from tools import file_state


# ``O_NOFOLLOW`` is POSIX-only.  Not present on Windows' ``os`` module; the
# symlink-following guard is therefore advisory on Windows (the canonical
# interprocess lock + scan-before-publish are the real defense).
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


# Windows msvcrt interprocess lock.  POSIX uses fcntl.flock; Windows uses
# msvcrt.locking on the same descriptor.  Either way the lock is real —
# there is no longer a silent no-op path.  When os.name == "nt" and the
# module is missing the operation must fail closed.
try:
    import msvcrt as _msvcrt  # noqa: E402  -- Windows-only interprocess lock.
except ImportError:
    _msvcrt = None  # type: ignore[assignment]


_IS_WINDOWS = (os.name == "nt")
_IS_POSIX = (os.name == "posix")


# msvcrt.locking mode constants are read directly from the active
# ``_msvcrt`` module (``_msvcrt.LK_LOCK``, ``_msvcrt.LK_NBLCK``,
# ``_msvcrt.LK_UNLCK``) so any drift between production and the real
# Windows module surfaces at runtime.  Production declares NO numeric
# aliases for these constants; the only ``0/1/2`` values live in the
# test-side fake (``tests/tools/test_session_write_policy_fail_closed.py``).
#
# When ``_msvcrt`` is missing on POSIX we leave it as ``None`` so the
# Windows branch (gated by ``_IS_WINDOWS``) is fail-closed until a test
# injects a real ``_msvcrt``.  Production never exercises the Windows
# branch because ``_IS_WINDOWS`` is False on POSIX.


def _validate_msvcrt_contract() -> Optional[str]:
    """Validate that ``_msvcrt`` exposes the three lock-mode constants
    the helper relies on.

    Returns ``None`` when the contract is satisfied (i.e. ``_msvcrt``
    has ``LK_NBLCK``, ``LK_LOCK`` and ``LK_UNLCK`` attributes), or a
    short reason string describing which attribute is missing.

    The helper must call this BEFORE opening the lock file so that a
    missing ``LK_UNLCK`` is detected prior to any mutation of the
    target skill tree — the critical section is never entered when
    any of the three attributes is absent.
    """
    if _msvcrt is None:
        return "msvcrt module is None"
    for attr in ("LK_NBLCK", "LK_LOCK", "LK_UNLCK"):
        if not hasattr(_msvcrt, attr):
            return f"msvcrt is missing required attribute {attr}"
    return None


def _win_failure(msg: str) -> PermissionError:
    """Translate a Windows-side lock failure into a fail-closed PermissionError."""
    return PermissionError(msg)


# ── Lock acquisition failure classification (Phase C P1-B) ─────────────
#
# The interprocess-mutation context manager classifies every
# acquisition failure into one of five named stages.  Each operation
# that uses the lock translates the resulting exception into the
# canonical acquisition-failure payload via
# ``_format_lock_acquisition_failure_payload``.  The five stages are
# closed by contract — operations MUST NOT introduce new stage names.
_LOCK_FAILURE_STAGE_PATH_RESOLUTION = "lock_path_resolution"
_LOCK_FAILURE_STAGE_PARENT_OPEN = "lock_parent_open"
_LOCK_FAILURE_STAGE_IDENTITY_VALIDATION = "lock_identity_validation"
_LOCK_FAILURE_STAGE_PRIMITIVE_ACQUIRE = "lock_primitive_acquire"
_LOCK_FAILURE_STAGE_CONTENTION = "lock_contention"

_LOCK_FAILURE_STAGES = frozenset({
    _LOCK_FAILURE_STAGE_PATH_RESOLUTION,
    _LOCK_FAILURE_STAGE_PARENT_OPEN,
    _LOCK_FAILURE_STAGE_IDENTITY_VALIDATION,
    _LOCK_FAILURE_STAGE_PRIMITIVE_ACQUIRE,
    _LOCK_FAILURE_STAGE_CONTENTION,
})


def _coerce_raw_permission_error_to_acquire_failure(
    canonical_skill_path: Path,
    exc: "PermissionError",
) -> "_SkillMutationLockAcquireFailure":
    """Convert an unclassified ``PermissionError`` that escaped a lock
    context into a structured acquisition failure.

    Used by the per-operation handlers as a defensive fallback when
    a test injection (or a third-party shim) raises raw
    ``PermissionError`` directly instead of going through
    ``_raise_lock_acquire_failure``.  Stage defaults to
    ``lock_primitive_acquire`` because the raw raise gives us no
    information to distinguish contention; ``safe_to_retry`` is
    ``False`` for the same reason.  ``cause`` is preserved so the
    original message survives.
    """
    platform = (
        "windows" if _IS_WINDOWS
        else ("posix" if _IS_POSIX else os.name or "unknown")
    )
    digest = _hashlib.sha256(
        str(canonical_skill_path).encode("utf-8")
    ).hexdigest()[:16]
    synth = canonical_skill_path.parent / (
        f".hermes-skill-mutex-{digest}.lock"
    )
    return _SkillMutationLockAcquireFailure(
        canonical_skill_path=canonical_skill_path,
        lock_path=synth,
        platform=platform,
        lock_failure_stage=_LOCK_FAILURE_STAGE_PRIMITIVE_ACQUIRE,
        cause=exc,
        safe_to_retry=False,
    )


class _SkillMutationLockAcquireFailure(PermissionError):
    """Raised when the interprocess mutation lock cannot be acquired.

    The exception carries enough metadata for the caller to build a
    canonical acquisition-failure payload without consulting any
    shared state (no module globals, no function attributes, no
    ContextVar).  ``safe_to_retry`` is pre-classified per stage — only
    ``lock_contention`` may be ``True``; every other stage is
    ``safe_to_retry=False`` because a retry would re-trigger the same
    structural failure.

    The underlying cause is preserved as ``__cause__`` (Python's
    exception chaining) so a structured traceback remains available
    outside the payload itself.  The payload-formatting helper folds
    ``cause`` into ``error`` and ``lock_exception_type``; the
    traceback itself is NEVER serialized into the payload.
    """

    def __init__(
        self,
        *,
        canonical_skill_path: Path,
        lock_path: Path,
        platform: str,
        lock_failure_stage: str,
        cause: Optional[BaseException] = None,
        safe_to_retry: bool = False,
    ) -> None:
        cause_repr = (
            f"{type(cause).__name__}: {cause}"
            if cause is not None
            else "(no underlying exception)"
        )
        summary = (
            f"interprocess lock acquisition failed on {lock_path} "
            f"(platform={platform}, stage={lock_failure_stage}); "
            f"cause={cause_repr}"
        )
        super().__init__(summary)
        self.canonical_skill_path = Path(canonical_skill_path)
        self.lock_path = Path(lock_path)
        self.platform = platform
        self.lock_failure_stage = lock_failure_stage
        self.safe_to_retry = bool(safe_to_retry)
        self.cause_exception = cause
        if cause is not None:
            self.__cause__ = cause


def _raise_lock_acquire_failure(
    *,
    canonical_skill_path: Path,
    lock_path: Path,
    platform: str,
    lock_failure_stage: str,
    cause: Optional[BaseException] = None,
    safe_to_retry: bool = False,
    msg: Optional[str] = None,
) -> None:
    """Helper that builds + raises ``_SkillMutationLockAcquireFailure``.

    Centralizing the raise site keeps every acquisition failure
    carrying identical metadata fields.  The optional ``msg`` is a
    human-readable context line folded into ``__str__``; the cause is
    preserved via exception chaining so the underlying OSError or
    PermissionError remains reachable to anyone introspecting the
    exception object.
    """
    exc = _SkillMutationLockAcquireFailure(
        canonical_skill_path=canonical_skill_path,
        lock_path=lock_path,
        platform=platform,
        lock_failure_stage=lock_failure_stage,
        cause=cause,
        safe_to_retry=safe_to_retry,
    )
    if msg is not None and cause is None:
        # When no underlying exception exists (e.g. structural
        # identity-validation rejection), attach a context message
        # for the agent loop.  Cause-chaining is preferred when an
        # underlying exception is available.
        exc.__str__ = lambda: f"{msg} ({exc!s})"  # type: ignore[assignment]
    raise exc


def _format_lock_acquisition_failure_payload(
    exc: "_SkillMutationLockAcquireFailure | _spg.SkillMutationLockAcquireFailure",
    *,
    operation_kind: str,
    target: Path,
) -> Dict[str, Any]:
    """Translate a ``_SkillMutationLockAcquireFailure`` into the canonical
    structured payload consumed by every skill operation.

    Contract: every caller that uses ``_skill_mutation_process_lock``
    MUST route an acquisition failure through this helper.  The
    resulting payload is the single diagnostic surface the agent loop
    sees; raw ``PermissionError`` exceptions MUST NOT escape the
    caller's boundary.

    The payload schema is closed:

      success: False
      error: "<cause> | <context>"
      policy_reason: "lock_acquisition_failed"
      rollback_failure_kind: "lock_acquisition_failure"
      operation_kind: <the real operation kind for this call site>
      target: <canonical or prospective target path>
      lock_path: <resolved lock path, or "" if not yet known>
      lock_failure_stage: one of the five canonical stages
      lock_exception_type: <exception class name>
      live_mutation_committed: False   (always — body never ran)
      safe_to_retry: <bool>            (True only for lock_contention)

    No exception tracebacks are serialized into the payload.  The
    cause error and exception class are surfaced via ``error`` and
    ``lock_exception_type`` respectively.
    """
    cause_repr: Optional[str] = None
    if exc.cause_exception is not None:
        cause_repr = f"{type(exc.cause_exception).__name__}: {exc.cause_exception}"
    elif exc.__cause__ is not None:
        cause_repr = f"{type(exc.__cause__).__name__}: {exc.__cause__}"
    base_error = str(exc) if cause_repr is None else cause_repr
    # When path resolution itself failed the lock_path could not be
    # derived; the formatter collapses it to "" so the agent loop sees
    # a deterministic value for the "lock_path unknown" case.
    lock_path_str = str(exc.lock_path)
    if exc.lock_failure_stage == _LOCK_FAILURE_STAGE_PATH_RESOLUTION:
        lock_path_str = ""
    payload: Dict[str, Any] = {
        "success": False,
        "error": base_error,
        "policy_reason": "lock_acquisition_failed",
        "rollback_failure_kind": "lock_acquisition_failure",
        "operation_kind": str(operation_kind),
        "target": str(Path(target)),
        "lock_path": lock_path_str,
        "lock_failure_stage": str(exc.lock_failure_stage),
        "lock_exception_type": type(exc.cause_exception).__name__
            if exc.cause_exception is not None
            else "PermissionError",
        "live_mutation_committed": False,
        "safe_to_retry": bool(exc.safe_to_retry),
    }
    return payload


class _SkillMutationLockReleaseFailure(RuntimeError):
    """Raised when the interprocess mutation lock cannot be cleanly released.

    Phase C final corrective — release/close failures MUST surface so callers
    can report ``success=false`` with a structured payload.  The error
    captures both the release-side and close-side errors so a caller
    inspecting the exception sees the full finalization picture; the
    canonical skill path and lock path are preserved for diagnostics.
    """

    def __init__(
        self,
        *,
        canonical_skill_path: Path,
        lock_path: Path,
        platform: str,
        release_error: Optional[BaseException] = None,
        close_error: Optional[BaseException] = None,
        live_mutation_committed: bool = False,
    ) -> None:
        release_repr = (
            f"{type(release_error).__name__}: {release_error}"
            if release_error is not None
            else None
        )
        close_repr = (
            f"{type(close_error).__name__}: {close_error}"
            if close_error is not None
            else None
        )
        summary = (
            f"interprocess lock release failed on {lock_path} "
            f"(platform={platform}); release_error={release_repr}; "
            f"close_error={close_repr}"
        )
        super().__init__(summary)
        self.canonical_skill_path = Path(canonical_skill_path)
        self.lock_path = Path(lock_path)
        self.platform = platform
        self.release_error = release_error
        self.close_error = close_error
        self.live_mutation_committed = bool(live_mutation_committed)


def _format_lock_release_failure_payload(
    exc: "_SkillMutationLockReleaseFailure | _spg.SkillMutationLockReleaseFailure",
    *,
    target: Path,
) -> Dict[str, Any]:
    """Translate a ``_SkillMutationLockReleaseFailure`` into the structured
    finalization-failure payload consumed by the six skill operations.

    Precedence rule: when the same operation also produced a cleanup
    failure, ``policy_reason`` becomes ``multiple_finalization_failures``
    but BOTH error classes are preserved so the agent loop can see
    cleanup_error and release_error independently.
    """
    payload: Dict[str, Any] = {
        "success": False,
        "error": str(exc),
        "policy_reason": "lock_release_failed",
        "rollback_failure_kind": "lock_release_failure",
        "lock_path": str(exc.lock_path),
        "release_error": (
            f"{type(exc.release_error).__name__}: {exc.release_error}"
            if exc.release_error is not None
            else None
        ),
        "close_error": (
            f"{type(exc.close_error).__name__}: {exc.close_error}"
            if exc.close_error is not None
            else None
        ),
        "target": str(target),
        "live_mutation_committed": bool(exc.live_mutation_committed),
        "safe_to_retry": False,
    }
    return payload


def _combine_lock_release_with_cleanup(
    payload: Dict[str, Any],
    cleanup_failure: Optional[Tuple[Path, str]],
) -> Dict[str, Any]:
    """Fold a cleanup failure into a lock-release-failure payload.

    Mirrors ``_combine_cleanup_failure`` but in the opposite direction
    (release is the primary failure, cleanup is the secondary).  When
    both occurred, ``policy_reason`` becomes
    ``multiple_finalization_failures`` and both classes are preserved.
    """
    if cleanup_failure is None:
        return payload
    staging_path, cleanup_error = cleanup_failure
    payload["policy_reason"] = "multiple_finalization_failures"
    payload["cleanup_error"] = str(cleanup_error)
    payload["staging_path"] = str(staging_path)
    existing_kind = payload.get("rollback_failure_kind") or "lock_release_failure"
    payload["rollback_failure_kind"] = (
        f"{existing_kind}+staging_cleanup_failure"
    )
    return payload


# ── Interprocess mutation lock (Phase C prepublish staging remediation) ────────
#
# file_state.lock_path() is intra-process (threading.Lock).  Two independent
# processes touching the same skill could otherwise both pass the
# ``lexists(skill_dir)`` check at the start of create() and then race on
# the final mkdir.  This private context manager adds a real POSIX fcntl.flock
# gate on a sibling of the live skill dir, keyed by a stable hash of the
# canonical skill path so two distinct skills acquire different locks.
#
# Scope: failure to acquire is fail-closed — the caller receives a structured
# error and does NOT mutate live state.  Windows falls back to a no-op (the
# single-user Windows install does not need cross-process serialization).
def _resolve_lock_parent(canonical: Path) -> Path:
    """Walk up from canonical until we are OUTSIDE every skills root.

    The lock file must NEVER live inside any skills root so a
    misbehaving rmtree inside the live tree cannot drop the lock and
    so discovery / loaders that walk the skills tree never see it.
    """
    from agent.skill_utils import get_all_skills_dirs
    try:
        resolved_roots = [r.resolve(strict=False) for r in get_all_skills_dirs()]
    except Exception:
        resolved_roots = []
    lock_parent = canonical.parent
    while True:
        try:
            inside = any(
                lock_parent.resolve(strict=False) == root
                or root in lock_parent.resolve(strict=False).parents
                for root in resolved_roots
            )
        except Exception:
            inside = False
        if not inside:
            break
        next_parent = lock_parent.parent
        if next_parent == lock_parent:
            break
        lock_parent = next_parent
    return lock_parent


# ── Lock file identity validation (Phase C final corrective) ─────────────
#
# The interprocess lock file lives OUTSIDE every skills root so that
# a misbehaving rmtree inside the live tree cannot drop the lock and
# so that discovery / loaders walking the skills tree never see it.
# But that ALSO means a misbehaving attacker (or a concurrent OS-level
# operation) could swap the lock pathname for a symlink, a directory,
# or a different inode BEFORE the lock acquisition completes.  The
# contract is:
#
#   * the lock pathname, if it exists, must be a regular file;
#   * after the open() the open fd must point at the same regular
#     file (by st_dev/st_ino/S_IFMT) as the pathname resolved via lstat;
#   * O_NOFOLLOW is set on POSIX so a symlink under the pathname is
#     rejected at open() time rather than silently followed.
#
# Any of these checks failing raises PermissionError and prevents
# entry into the critical section.  The lock file is NEVER removed
# or replaced by this module — the lock is held by the kernel, not by
# the file's existence.
def _validate_lock_file_identity(
    lock_path: Path,
    *,
    fd: Optional[int] = None,
    _stat=os.stat,
) -> None:
    """Confirm ``lock_path`` resolves to a regular file.

    On POSIX this is called BEFORE the open (via lstat) and AFTER the
    open (via fstat on the fd) so a TOCTOU swap cannot redirect the
    open onto a different inode.  ``_stat`` is overridable for tests
    that want to inject a different lstat/fstat implementation.

    Raises ``PermissionError`` on any mismatch (symlink, junction,
    directory, dangling link, or inode swap between lstat and fstat).
    """
    try:
        st_path = _stat(  # type: ignore[arg-type]
            str(lock_path),
            follow_symlinks=False,
        )
    except OSError as exc:
        raise PermissionError(
            f"could not lstat lock file {lock_path}: {exc}"
        ) from exc

    if stat.S_ISLNK(st_path.st_mode):
        raise PermissionError(
            f"refusing to acquire lock: {lock_path} is a symlink"
        )
    if not stat.S_ISREG(st_path.st_mode):
        raise PermissionError(
            f"refusing to acquire lock: {lock_path} is not a regular file "
            f"(mode={oct(st_path.st_mode)})"
        )

    if fd is not None:
        try:
            st_fd = os.fstat(fd)
        except OSError as exc:
            raise PermissionError(
                f"could not fstat lock fd for {lock_path}: {exc}"
            ) from exc
        if not stat.S_ISREG(st_fd.st_mode):
            raise PermissionError(
                f"refusing to acquire lock: {lock_path} fd does not point to a regular file"
            )
        path_identity = (st_path.st_dev, st_path.st_ino, stat.S_IFMT(st_path.st_mode))
        fd_identity = (st_fd.st_dev, st_fd.st_ino, stat.S_IFMT(st_fd.st_mode))
        if path_identity != fd_identity:
            raise PermissionError(
                f"lock inode changed between lstat and fstat on {lock_path}: "
                f"path={path_identity} fd={fd_identity}"
            )


@contextmanager
def _skill_mutation_process_lock(canonical_skill_path: Path):
    """Interprocess (POSIX or Windows) lock for skill mutations.

    ``canonical_skill_path`` is the resolved canonical skill directory the
    caller intends to mutate.  The lock file lives OUTSIDE every skills
    root (see ``_resolve_lock_parent``) so a misbehaving ``rmtree`` inside
    the live tree cannot drop the lock and so discovery/loaders that
    walk the skills tree never see it.

    POSIX uses ``fcntl.flock(LOCK_EX)``; Windows uses ``msvcrt.locking``
    on the same byte of the same file.  Both paths are fail-closed —

      * any acquisition failure raises ``PermissionError``;
      * any release or close failure raises ``_SkillMutationLockReleaseFailure``
        AFTER both attempts have been made (so both errors are preserved
        if both failed);
      * the lock file is NEVER removed — the lock is held by the kernel,
        not by the file's existence on disk.

    The lock file is opened with ``O_NOFOLLOW`` on POSIX (when the
    constant is available), validated against symlinks / directories
    via lstat before open and against the open fd's identity via fstat
    after open.  Any identity mismatch raises ``PermissionError`` and
    prevents entry into the critical section.
    """
    canonical = Path(canonical_skill_path).resolve(strict=False)
    platform_name = "windows" if _IS_WINDOWS else ("posix" if _IS_POSIX else os.name or "unknown")
    try:
        lock_parent = _resolve_lock_parent(canonical)
    except Exception as exc:
        # Path resolution failed before a usable parent was derived.
        # The lock_path cannot be derived deterministically here —
        # surface a structured failure with empty lock_path.
        digest_for_failure = _hashlib.sha256(
            str(canonical).encode("utf-8")
        ).hexdigest()[:16]
        # Best-effort fallback for the synthetic lock_path so the
        # caller can still see the deterministic key the helper WOULD
        # have used; the formatted payload collapses this to "" on
        # the path-resolution stage via the formatter.
        synthetic_lock_path = canonical.parent / (
            f".hermes-skill-mutex-{digest_for_failure}.lock"
        )
        _raise_lock_acquire_failure(
            canonical_skill_path=canonical,
            lock_path=synthetic_lock_path,
            platform=platform_name,
            lock_failure_stage=_LOCK_FAILURE_STAGE_PATH_RESOLUTION,
            cause=exc,
            safe_to_retry=False,
        )
    digest = _hashlib.sha256(str(canonical).encode("utf-8")).hexdigest()[:16]
    lock_path = lock_parent / f".hermes-skill-mutex-{digest}.lock"

    if _IS_POSIX:
        if _fcntl is None:
            # POSIX reported but fcntl is unavailable (stripped Python
            # build).  Fail closed — do not silently no-op.
            _raise_lock_acquire_failure(
                canonical_skill_path=canonical,
                lock_path=lock_path,
                platform=platform_name,
                lock_failure_stage=_LOCK_FAILURE_STAGE_PATH_RESOLUTION,
                cause=PermissionError(
                    f"fcntl module is unavailable on POSIX"
                ),
                safe_to_retry=False,
            )
        fd = None
        release_error: Optional[BaseException] = None
        close_error: Optional[BaseException] = None
        try:
            open_flags = os.O_CREAT | os.O_RDWR
            if _O_NOFOLLOW:
                open_flags |= _O_NOFOLLOW
            # Pre-open lstat guard: refuse to follow a symlink or
            # take over a directory that lives where the lock should be.
            if os.path.lexists(str(lock_path)):
                try:
                    pre_st = os.lstat(str(lock_path))
                except OSError as exc:
                    _raise_lock_acquire_failure(
                        canonical_skill_path=canonical,
                        lock_path=lock_path,
                        platform=platform_name,
                        lock_failure_stage=_LOCK_FAILURE_STAGE_PARENT_OPEN,
                        cause=exc,
                        safe_to_retry=False,
                    )
                if stat.S_ISLNK(pre_st.st_mode):
                    _raise_lock_acquire_failure(
                        canonical_skill_path=canonical,
                        lock_path=lock_path,
                        platform=platform_name,
                        lock_failure_stage=_LOCK_FAILURE_STAGE_IDENTITY_VALIDATION,
                        cause=None,
                        safe_to_retry=False,
                        msg=f"refusing to acquire lock: {lock_path} is a symlink",
                    )
                if not stat.S_ISREG(pre_st.st_mode):
                    _raise_lock_acquire_failure(
                        canonical_skill_path=canonical,
                        lock_path=lock_path,
                        platform=platform_name,
                        lock_failure_stage=_LOCK_FAILURE_STAGE_IDENTITY_VALIDATION,
                        cause=None,
                        safe_to_retry=False,
                        msg=(
                            f"refusing to acquire lock: {lock_path} is not "
                            f"a regular file (mode={oct(pre_st.st_mode)})"
                        ),
                    )
            try:
                fd = os.open(str(lock_path), open_flags, 0o600)
            except OSError as exc:
                _raise_lock_acquire_failure(
                    canonical_skill_path=canonical,
                    lock_path=lock_path,
                    platform=platform_name,
                    lock_failure_stage=_LOCK_FAILURE_STAGE_PARENT_OPEN,
                    cause=exc,
                    safe_to_retry=False,
                )
            # Post-open identity guard: fstat the fd and compare with
            # lstat the path (using follow_symlinks=False).
            try:
                _validate_lock_file_identity(lock_path, fd=fd)
            except PermissionError as exc:
                try:
                    os.close(fd)
                finally:
                    fd = None
                _raise_lock_acquire_failure(
                    canonical_skill_path=canonical,
                    lock_path=lock_path,
                    platform=platform_name,
                    lock_failure_stage=_LOCK_FAILURE_STAGE_IDENTITY_VALIDATION,
                    cause=exc,
                    safe_to_retry=False,
                )
            try:
                _fcntl.flock(fd, _fcntl.LOCK_EX)
            except OSError as exc:
                # Distinguish EWOULDBLOCK on a non-blocking attempt
                # (contention) from any other primitive failure.  The
                # current production contract uses a blocking LOCK_EX,
                # so contention via EWOULDBLOCK is unreachable from the
                # real Linux kernel here — but a test that injects
                # EWOULDBLOCK must still be classified as contention so
                # safe_to_retry=True surfaces.  ENOLCK / EACCES / EIO /
                # EFAULT etc. are structural primitive-acquire failures
                # and stay non-retryable.
                errno_val = getattr(exc, "errno", None)
                stage = _LOCK_FAILURE_STAGE_CONTENTION if (
                    errno_val == _errno.EWOULDBLOCK
                    or errno_val == _errno.EAGAIN
                ) else _LOCK_FAILURE_STAGE_PRIMITIVE_ACQUIRE
                try:
                    os.close(fd)
                finally:
                    fd = None
                _raise_lock_acquire_failure(
                    canonical_skill_path=canonical,
                    lock_path=lock_path,
                    platform=platform_name,
                    lock_failure_stage=stage,
                    cause=exc,
                    safe_to_retry=(stage == _LOCK_FAILURE_STAGE_CONTENTION),
                )
            yield
        finally:
            if fd is not None:
                # Release must attempt cleanup BEFORE close so a
                # second acquirer does not see the un-flocked file
                # with a closed fd.
                try:
                    _fcntl.flock(fd, _fcntl.LOCK_UN)
                except OSError as exc:
                    release_error = exc
                try:
                    os.close(fd)
                except OSError as exc:
                    close_error = exc
                fd = None
                if release_error is not None or close_error is not None:
                    raise _SkillMutationLockReleaseFailure(
                        canonical_skill_path=canonical,
                        lock_path=lock_path,
                        platform=platform_name,
                        release_error=release_error,
                        close_error=close_error,
                        live_mutation_committed=False,
                    )
        return

    if _IS_WINDOWS:
        if _msvcrt is None:
            _raise_lock_acquire_failure(
                canonical_skill_path=canonical,
                lock_path=lock_path,
                platform=platform_name,
                lock_failure_stage=_LOCK_FAILURE_STAGE_PATH_RESOLUTION,
                cause=PermissionError(
                    f"msvcrt module is unavailable on Windows"
                ),
                safe_to_retry=False,
            )
        # Validate the msvcrt contract BEFORE touching the lock file:
        # if any of the three lock-mode constants is missing, fail
        # closed without entering the critical section.  This catches
        # ``LK_UNLCK`` absence before any target mutation happens and
        # prevents the helper from ever calling ``msvcrt.locking``
        # with a hard-coded numeric fallback.
        contract_reason = _validate_msvcrt_contract()
        if contract_reason is not None:
            _raise_lock_acquire_failure(
                canonical_skill_path=canonical,
                lock_path=lock_path,
                platform=platform_name,
                lock_failure_stage=_LOCK_FAILURE_STAGE_IDENTITY_VALIDATION,
                cause=PermissionError(
                    f"msvcrt contract is invalid on Windows ({contract_reason})"
                ),
                safe_to_retry=False,
            )
        # msvcrt.locking requires the descriptor to be opened for read
        # AND the file to contain at least one byte of data.  We pad the
        # file with a single NUL byte at creation so the lock region
        # always exists.
        fd = None
        release_error = None
        close_error = None
        try:
            # Windows has no O_NOFOLLOW on os.open.  Emulate the
            # symlink/junction rejection with an explicit lstat guard
            # — Windows ``os.lstat`` already does NOT follow junctions
            # for stat.S_ISLNK, and we additionally reject any path
            # whose lstat result is not a regular file.
            if os.path.lexists(str(lock_path)):
                try:
                    pre_st = os.lstat(str(lock_path))
                except OSError as exc:
                    _raise_lock_acquire_failure(
                        canonical_skill_path=canonical,
                        lock_path=lock_path,
                        platform=platform_name,
                        lock_failure_stage=_LOCK_FAILURE_STAGE_PARENT_OPEN,
                        cause=exc,
                        safe_to_retry=False,
                    )
                if stat.S_ISLNK(pre_st.st_mode):
                    _raise_lock_acquire_failure(
                        canonical_skill_path=canonical,
                        lock_path=lock_path,
                        platform=platform_name,
                        lock_failure_stage=_LOCK_FAILURE_STAGE_IDENTITY_VALIDATION,
                        cause=None,
                        safe_to_retry=False,
                        msg=f"refusing to acquire lock: {lock_path} is a symlink/junction",
                    )
                if not stat.S_ISREG(pre_st.st_mode):
                    _raise_lock_acquire_failure(
                        canonical_skill_path=canonical,
                        lock_path=lock_path,
                        platform=platform_name,
                        lock_failure_stage=_LOCK_FAILURE_STAGE_IDENTITY_VALIDATION,
                        cause=None,
                        safe_to_retry=False,
                        msg=(
                            f"refusing to acquire lock: {lock_path} is not "
                            f"a regular file (mode={oct(pre_st.st_mode)})"
                        ),
                    )
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
            except OSError as exc:
                _raise_lock_acquire_failure(
                    canonical_skill_path=canonical,
                    lock_path=lock_path,
                    platform=platform_name,
                    lock_failure_stage=_LOCK_FAILURE_STAGE_PARENT_OPEN,
                    cause=exc,
                    safe_to_retry=False,
                )
            try:
                _validate_lock_file_identity(lock_path, fd=fd)
            except PermissionError as exc:
                try:
                    os.close(fd)
                finally:
                    fd = None
                _raise_lock_acquire_failure(
                    canonical_skill_path=canonical,
                    lock_path=lock_path,
                    platform=platform_name,
                    lock_failure_stage=_LOCK_FAILURE_STAGE_IDENTITY_VALIDATION,
                    cause=exc,
                    safe_to_retry=False,
                )
            try:
                size = os.fstat(fd).st_size
                if size == 0:
                    os.write(fd, b"\x00")
                # Seek to byte 0; msvcrt.locking positions the file pointer
                # implicitly on the underlying CRT file but seek-to-zero
                # keeps the behavior predictable.
                os.lseek(fd, 0, os.SEEK_SET)
            except OSError as exc:
                try:
                    os.close(fd)
                finally:
                    fd = None
                _raise_lock_acquire_failure(
                    canonical_skill_path=canonical,
                    lock_path=lock_path,
                    platform=platform_name,
                    lock_failure_stage=_LOCK_FAILURE_STAGE_PARENT_OPEN,
                    cause=exc,
                    safe_to_retry=False,
                )
            try:
                _msvcrt.locking(fd, _msvcrt.LK_NBLCK, 1)
            except (OSError, PermissionError) as exc:
                # Non-blocking failed — try a blocking acquisition as the
                # canonical Windows interprocess lock idiom.  This blocks
                # the calling thread until the holder releases.
                try:
                    _msvcrt.locking(fd, _msvcrt.LK_LOCK, 1)
                except (OSError, PermissionError) as exc2:
                    try:
                        os.close(fd)
                    finally:
                        fd = None
                    _raise_lock_acquire_failure(
                        canonical_skill_path=canonical,
                        lock_path=lock_path,
                        platform=platform_name,
                        lock_failure_stage=_LOCK_FAILURE_STAGE_PRIMITIVE_ACQUIRE,
                        cause=exc2,
                        safe_to_retry=False,
                    )
            yield
        finally:
            if fd is not None:
                try:
                    os.lseek(fd, 0, os.SEEK_SET)
                    _msvcrt.locking(fd, _msvcrt.LK_UNLCK, 1)
                except OSError as exc:
                    release_error = exc
                try:
                    os.close(fd)
                except OSError as exc:
                    close_error = exc
                fd = None
                if release_error is not None or close_error is not None:
                    raise _SkillMutationLockReleaseFailure(
                        canonical_skill_path=canonical,
                        lock_path=lock_path,
                        platform=platform_name,
                        release_error=release_error,
                        close_error=close_error,
                        live_mutation_committed=False,
                    )
        return

    # Unknown platform — fail closed.
    _raise_lock_acquire_failure(
        canonical_skill_path=canonical,
        lock_path=lock_path,
        platform=platform_name,
        lock_failure_stage=_LOCK_FAILURE_STAGE_PATH_RESOLUTION,
        cause=PermissionError(
            f"unsupported platform os.name={os.name!r}"
        ),
        safe_to_retry=False,
    )



# ── Canonical skill-name normalization (Phase C global name-mutex) ────────────
#
# The "canonical" form of a skill name is the validated form.  ``_validate_name``
# already enforces a strict regex (``[a-z0-9][a-z0-9._-]*``) which is
# unambiguous, lowercase, and case-preserving — there is no folding, no
# dot/underscore collapsing, no NFKC/NFD step.  This helper exists so the
# global normalized-name mutex has a single named source for the equality
# key, and so the test suite can prove the key is independent of category,
# root, and caller.
#
# Contract:
#   * returns the validated name string when the name passes the regex,
#     otherwise None;
#   * pure / deterministic / side-effect free;
#   * the returned value is exactly the form ``_find_skill`` uses for its
#     literal ``skill_md.parent.name == name`` comparison, so the global
#     uniqueness domain is identical to the existing literal uniqueness
#     domain.
_NORMALIZATION_VERSION = "normalized-name-v1"


def _canonical_normalize_skill_name(name: Any) -> Optional[str]:
    """Return the canonical form of ``name`` for global uniqueness checks.

    Returns ``None`` for invalid names — callers MUST treat that as a
    normal validation failure, NOT as a normalized key, so the global
    mutex is never acquired on invalid input.
    """
    if not isinstance(name, str):
        return None
    if _validate_name(name) is not None:
        return None
    return name


def _normalized_skill_name_lock_target(normalized_name: str) -> Path:
    """Pure derivation of the global normalized-name lock path.

    Contract:
      * deterministic — same input → same path on every call;
      * side-effect free — does not touch the filesystem;
      * key depends ONLY on ``normalized_name`` and ``_NORMALIZATION_VERSION``;
      * the parent is walked up from the active local skills root until
        it is outside every known skills root, so the lock file lives
        in the same namespace as the existing per-target mutation lock;
      * the filename carries a sha256 digest (not the raw name) so the
        raw name never appears on disk and the path is collision-free.
    """
    if not isinstance(normalized_name, str) or not normalized_name:
        raise ValueError(
            "_normalized_skill_name_lock_target requires a non-empty "
            "validated normalized_name"
        )
    local_root = _skills_dir().resolve(strict=False)
    lock_parent = _resolve_lock_parent(local_root)
    digest = _hashlib.sha256(
        f"{_NORMALIZATION_VERSION}\0{normalized_name}".encode("utf-8")
    ).hexdigest()
    return lock_parent / f".hermes-skill-name-mutex-{digest}.lock"


@contextmanager
def _global_normalized_name_lock(normalized_name: str):
    """Acquire the global normalized-name mutex derived ONLY from the
    validated skill name.

    This lock is keyed by ``_normalized_skill_name_lock_target`` so:

      * two creates with the SAME normalized name (regardless of
        category, root, or spelling variants that produce the same
        canonical form) acquire the SAME lock file;
      * two creates with DIFFERENT normalized names acquire DIFFERENT
        lock files and can run concurrently;
      * the lock file lives in the same namespace as the per-target
        mutation lock (sibling outside every skills root) so a single
        private contract covers parent validation, symlink refusal,
        regular-file validation, POSIX/Windows acquisition, structured
        acquisition payload, release-before-close, and descriptor close.

    The implementation REUSES ``_skill_mutation_process_lock`` on the
    synthetic normalized-name path.  No second fcntl/msvcrt primitive
    is introduced; all the validated guarantees of the existing helper
    (parent validation, symlink refusal, O_NOFOLLOW, lstat/fstat
    identity guard, structured release failure) carry over by
    construction.
    """
    if not isinstance(normalized_name, str) or not normalized_name:
        synthetic = _skills_dir().parent / (
            f".hermes-skill-name-mutex-{_hashlib.sha256(b'invalid').hexdigest()}.lock"
        )
        _raise_lock_acquire_failure(
            canonical_skill_path=synthetic,
            lock_path=synthetic,
            platform=(
                "windows" if _IS_WINDOWS
                else ("posix" if _IS_POSIX else os.name or "unknown")
            ),
            lock_failure_stage=_LOCK_FAILURE_STAGE_PATH_RESOLUTION,
            cause=ValueError(
                "global normalized-name lock requires a non-empty validated name"
            ),
            safe_to_retry=False,
        )
    lock_path = _normalized_skill_name_lock_target(normalized_name)
    # Reuse the validated interprocess primitive — the synthetic path
    # is never written to the live tree, so the helper's invariants
    # (lock file outside skills roots, never inside the live tree,
    # never removed) hold trivially.  Look up the primitive at call
    # time so test suites can monkeypatch the underlying function
    # without invalidating an import-time reference.
    _skill_mutation_process_lock_impl = globals()["_skill_mutation_process_lock"]
    with _skill_mutation_process_lock_impl(lock_path):
        yield normalized_name


# ── Private staging directory (Phase C prepublish staging remediation) ─────────
#
# Skills are scanned BEFORE the live tree is touched.  The candidate content
# lives in a private staging directory that:
#   * is created exclusively by this module (random suffix, never supplied
#     by the caller — guarantees no caller-controlled path traversal);
#   * lives outside any skills root, so _find_skill / discovery / loaders
#     walking ``get_all_skills_dirs()`` cannot reach it;
#   * lives next to the skills root parent so it usually sits on the same
#     filesystem (enabling os.link() no-clobber publish);
#   * is cleaned up by a single private helper that only ever targets the
#     exact staging dir captured at creation time;
#   * is never partially published — the live tree is only mutated AFTER
#     the staging content has been fully scanned.
class _StagingHandle:
    """Wrapper around a staging ``Path`` that captures its identity at
    creation time so ``_cleanup_private_staging`` can refuse TOCTOU
    swaps (the staging inode, the staging parent inode, or the staging
    pathname being redirected to a foreign tree).

    ``Path`` itself is an immutable value type and has nowhere to stash
    per-instance identity attributes, so the handle stores the captured
    ``(dev, ino, type)`` triple at creation time and forwards every
    other ``Path`` operation to the wrapped instance via
    ``__getattr__``.
    """

    __slots__ = (
        "path",
        "_staging_identity",
        "_parent_identity",
        "_expected_name",
    )

    def __init__(
        self,
        path: Path,
        *,
        staging_identity: Optional[Tuple[int, int, int]],
        parent_identity: Optional[Tuple[int, int, int]],
        expected_name: str,
    ) -> None:
        self.path = path
        self._staging_identity = staging_identity
        self._parent_identity = parent_identity
        self._expected_name = expected_name

    def __fspath__(self) -> str:
        return str(self.path)

    def __str__(self) -> str:  # pragma: no cover -- cosmetic
        return str(self.path)

    def __repr__(self) -> str:  # pragma: no cover -- diagnostic
        return f"_StagingHandle({self.path!r})"

    def __truediv__(self, other: Any) -> Any:
        # Forward ``staging / 'foo'`` so the existing tests that treat
        # the handle like a Path keep working.
        return self.path / other

    def __getattr__(self, item: str) -> Any:
        # Forward every other attribute access (``stat``, ``parent``,
        # ``exists``, ``iterdir``, etc.) to the wrapped Path.  Identity
        # attributes live in __slots__ and shadow this only by exact
        # name; ``path`` is the most common attribute to proxy.
        return getattr(self.path, item)

    @property
    def name(self) -> str:
        return self.path.name


def _create_private_staging(skills_root: Path) -> _StagingHandle:
    """Create and return a private staging directory for one mutation.

    The returned handle is owned by the calling operation and must be
    passed to ``_cleanup_private_staging`` exactly once on every exit
    path.  ``Path`` is immutable, so the helper returns a wrapper that
    also captures the staging's (dev, ino, type) identity, the parent's
    (dev, ino, type) identity, and the staging's original name — so
    the cleanup helper can detect any of:

      * the staging path being replaced by a symlink;
      * the staging path being replaced by a foreign directory that
        happens to share the random suffix;
      * the staging parent being replaced by a symlink pointing
        elsewhere (which would otherwise let ``shutil.rmtree`` walk
        into a foreign tree).
    """
    parent = skills_root.parent.parent
    if parent == skills_root.parent or not str(parent):
        # Defensive: if the skills_root is at the filesystem root for
        # some reason, fall back to the immediate parent so we still
        # have a writable directory.
        parent = skills_root.parent
    # Wrap the staging in a freshly-created empty directory so the
    # staging's parent contains exactly one child (the staging itself).
    # This gives the cleanup helper an rmdir-able parent that contains
    # only the staging — which is the only layout in which parent-symlink
    # swap tests can reproduce without first needing to empty ``parent``.
    scratch_parent = parent / f".hermes-staging-{_secrets.token_hex(8)}"
    scratch_parent.mkdir(parents=False, exist_ok=False)
    try:
        os.chmod(scratch_parent, 0o700)
    except OSError:
        logger.debug("could not chmod scratch parent %s", scratch_parent, exc_info=True)
    suffix = _secrets.token_hex(8)
    staging_path = scratch_parent / f".hermes-skill-staging-{suffix}"
    # ``exist_ok=False`` so two operations racing on the same random suffix
    # would fail rather than silently sharing a staging tree.
    staging_path.mkdir(parents=False, exist_ok=False)
    # Mode 0700 — staging is private; readers of the skills root cannot list it
    # unless they walk above the skills root (which they don't).
    try:
        os.chmod(staging_path, 0o700)
    except OSError:
        logger.debug("could not chmod staging %s", staging_path, exc_info=True)
    staging_identity: Optional[Tuple[int, int, int]] = None
    parent_identity: Optional[Tuple[int, int, int]] = None
    try:
        post_st = os.lstat(str(staging_path))
        staging_identity = (
            post_st.st_dev,
            post_st.st_ino,
            stat.S_IFMT(post_st.st_mode),
        )
        parent_identity = _parent_identity(staging_path.parent)
    except OSError:
        # If we can't lstat immediately after mkdir we're in a strange
        # state — leave the identities as None so cleanup falls through
        # to the name-prefix fail-closed path.
        pass
    return _StagingHandle(
        staging_path,
        staging_identity=staging_identity,
        parent_identity=parent_identity,
        expected_name=staging_path.name,
    )


def _parent_identity(parent: Path) -> tuple[int, int, int]:
    """Lstat ``parent`` and return its (dev, ino, type).

    Used by staging cleanup to detect a parent-symlink swap that would
    otherwise redirect the cleanup's ``shutil.rmtree`` to a foreign
    tree.  Anchored by st_dev + st_ino + S_IFMT, not by pathname.
    """
    st = os.lstat(str(parent))
    return st.st_dev, st.st_ino, stat.S_IFMT(st.st_mode)


def _unwrap_staging(staging: Any) -> Tuple[Path, Optional[Tuple[int, int, int]], Optional[Tuple[int, int, int]], str]:
    """Accept either a ``_StagingHandle`` or a raw ``Path`` for backward
    compatibility; return ``(path, staging_identity, parent_identity, name)``.

    For raw ``Path`` callers the identity attributes are ``None`` and
    the cleanup helper takes the conservative fail-closed path.
    """
    if isinstance(staging, _StagingHandle):
        return (
            staging.path,
            staging._staging_identity,
            staging._parent_identity,
            staging._expected_name,
        )
    return (Path(staging), None, None, Path(staging).name)


def _cleanup_private_staging(staging: Any) -> Optional[Tuple[Path, str]]:
    """Remove a staging directory created by ``_create_private_staging``.

    Fail-closed contract (Phase C final corrective): the staging path
    is removed ONLY if its captured identity still matches the inode
    this function observed at creation time.  Any swap — a symlink
    that replaced the staging path, a foreign directory placed at the
    same name, a parent-symlink redirecting elsewhere — makes the
    cleanup refuse and the foreign tree survives intact.

    Caller-supplied paths are rejected at the name-prefix guard so the
    helper cannot be repurposed as a generic recursive-delete primitive.
    Any failure is REPORTED — never silently swallowed.  The cleanup
    MUST never re-raise into the agent loop (so a stuck rmtree doesn't
    propagate), but the caller MUST inspect the return value:

      * ``None`` → cleanup succeeded or was a no-op.
      * ``(staging_path, error_str)`` → cleanup failed; the staging
        directory may still exist on disk.

    The structured-cleanup contract (Phase C prepublish staging remediation)
    requires that callers translate the returned tuple into a structured
    payload so the agent loop sees ``success=false`` with
    ``policy_reason="cleanup_failed"`` rather than ``success=true`` with a
    leaked staging tree.
    """
    if staging is None:
        return None

    staging_path, expected_identity, expected_parent_identity, expected_name = (
        _unwrap_staging(staging)
    )

    # Reject the call if the staging name doesn't match the captured
    # ``.hermes-skill-staging-*`` prefix.
    if (
        expected_name is None
        or not isinstance(expected_name, str)
        or not expected_name.startswith(".hermes-skill-staging-")
    ):
        logger.error(
            "_cleanup_private_staging refused non-staging path: %s", staging_path
        )
        return (
            staging_path,
            f"refused non-staging path: {staging_path}",
        )

    # ─── Parent-identity preflight (must run before any existence check) ──────
    # The captured parent identity is the authoritative proof that
    # ``staging_path.parent`` still points at the directory the staging
    # was originally created under.  If the parent was swapped for a
    # symlink pointing elsewhere (or replaced by any other inode),
    # ``rmtree`` against the symlink-following path would walk into a
    # foreign tree.  Fail-closed: surface a structured failure EVEN
    # when the staging's own lexists later returns False, because a
    # silent no-op in that case would mask the swap.
    if expected_parent_identity is not None:
        try:
            current_parent_identity = _parent_identity(staging_path.parent)
        except OSError as exc:
            logger.error(
                "could not lstat staging parent %s during cleanup: %s",
                staging_path.parent, exc, exc_info=True,
            )
            return (staging_path, f"could not lstat staging parent: {exc}")
        if current_parent_identity != expected_parent_identity:
            logger.error(
                "staging parent inode changed between create and cleanup: "
                "%s (was %r, now %r); foreign tree preserved",
                staging_path.parent, expected_parent_identity, current_parent_identity,
            )
            return (
                staging_path,
                "staging parent inode changed; foreign tree preserved",
            )

    # Parent identity OK — proceed with staging lstat.  This must use
    # ``lstat`` (follow_symlinks=False) so a swap of the staging path
    # itself is detected as a symlink instead of being walked into.
    try:
        lexists_now = os.path.lexists(str(staging_path))
    except OSError:
        lexists_now = False
    if not lexists_now:
        # The staging path is gone.  Parent identity already validated
        # above, so this is a benign no-op (the staging was already
        # removed by the caller or by an external process).  We do
        # NOT raise here because the swap-detection above is the
        # safety net; returning None signals "nothing left to clean".
        return None

    try:
        current_st = os.lstat(str(staging_path))
        current_identity = (
            current_st.st_dev,
            current_st.st_ino,
            stat.S_IFMT(current_st.st_mode),
        )
    except OSError as exc:
        logger.error(
            "could not lstat staging %s during cleanup: %s",
            staging_path, exc, exc_info=True,
        )
        return (staging_path, f"could not lstat staging during cleanup: {exc}")

    # If the staging path was swapped for a symlink, refuse to follow.
    if stat.S_ISLNK(current_st.st_mode):
        logger.error(
            "staging path %s was replaced by a symlink; foreign tree preserved",
            staging_path,
        )
        return (staging_path, "staging path became a symlink; foreign tree preserved")
    # If the staging path was replaced by something that is not a
    # directory (file, junction, etc.), refuse.
    if not stat.S_ISDIR(current_st.st_mode):
        logger.error(
            "staging path %s is no longer a directory (mode=%s); foreign object preserved",
            staging_path, oct(current_st.st_mode),
        )
        return (staging_path, "staging path is no longer a directory")

    # If the identity changed, refuse — the path is now pointing at a
    # different inode than the one we created.  Anchored destruction
    # requires the captured identity to match what is currently on disk.
    if expected_identity is not None and current_identity != expected_identity:
        logger.error(
            "staging inode changed between create and cleanup: %s "
            "(was %r, now %r); foreign tree preserved",
            staging_path, expected_identity, current_identity,
        )
        return (staging_path, "staging inode changed; foreign tree preserved")

    try:
        shutil.rmtree(str(staging_path))
    except OSError as exc:
        logger.error(
            "failed to clean up staging %s: %s", staging_path, exc, exc_info=True
        )
        return (staging_path, str(exc))
    scratch_parent = staging_path.parent
    if scratch_parent.name.startswith(".hermes-staging-"):
        try:
            if not any(scratch_parent.iterdir()):
                scratch_parent.rmdir()
        except OSError as exc:
            logger.error(
                "failed to clean up staging parent %s: %s",
                scratch_parent,
                exc,
                exc_info=True,
            )
            return (scratch_parent, str(exc))
    return None


@contextmanager
def _cleanup_with_report(staging: Path):
    """Run ``_cleanup_private_staging(staging)`` at scope exit and capture the
    outcome into the bound variable ``cleanup_failure``.

    Usage::

        with _cleanup_with_report(staging) as report:
            result = build_result()
            report["result"] = result
        # After __exit__:
        cleanup_failure = report["cleanup_failure"]

    The context manager always calls cleanup on scope exit (success or
    exception) and stores the outcome as ``report["cleanup_failure"]``.
    No exception escapes — the agent loop must never see a stuck rmtree.
    """
    report: Dict[str, Any] = {"result": None, "cleanup_failure": None}
    try:
        yield report
    finally:
        report["cleanup_failure"] = _cleanup_private_staging(staging)


def _combine_cleanup_failure(
    primary: Dict[str, Any],
    cleanup_failure: Optional[Tuple[Path, str]],
    *,
    live_mutation_committed: bool,
) -> Dict[str, Any]:
    """Fold a cleanup failure into the primary structured result.

    ``primary`` is the payload the operation would have returned without
    the cleanup failure.  When cleanup succeeded (cleanup_failure is
    None), ``primary`` is returned unchanged.

    When cleanup failed and the live mutation had NOT been committed
    (e.g. scan rejection), the result stays ``success=false`` with
    ``policy_reason="cleanup_failed"`` and
    ``primary_failure_kind`` set from the primary's ``rollback_failure_kind``
    so the original scanner / staging error is not lost.

    When cleanup failed AFTER the live mutation had been committed
    (successful publish path), the result MUST be ``success=false`` —
    returning ``success=true`` here would tell the agent the operation
    succeeded while leaving a private staging tree with byte-faithful
    copies of the published content on disk.  The error must surface
    ``live_mutation_committed=true`` and the residual ``staging_path``.
    """
    if cleanup_failure is None:
        return primary
    staging_path, cleanup_error = cleanup_failure
    if live_mutation_committed:
        # Live was already mutated.  Refuse to report success; surface
        # the residual staging path so the operator can clean it up.
        return {
            "success": False,
            "error": (
                f"private staging cleanup failed after successful publish; "
                f"residual staging tree at {staging_path} requires manual "
                f"intervention"
            ),
            "policy_reason": "cleanup_failed",
            "rollback_failure_kind": "staging_cleanup_failure",
            "cleanup_error": str(cleanup_error),
            "staging_path": str(staging_path),
            "live_mutation_committed": True,
            "target": primary.get("target", ""),
        }
    # Live was NOT mutated.  Preserve the primary failure kind and
    # layer the cleanup error on top.
    result = dict(primary)
    result["success"] = False
    result["policy_reason"] = "cleanup_failed"
    primary_kind = primary.get("rollback_failure_kind") or primary.get(
        "policy_reason", "scan_failure"
    )
    result["primary_failure_kind"] = str(primary_kind)
    result["rollback_failure_kind"] = "staging_cleanup_failure"
    result["cleanup_error"] = str(cleanup_error)
    result["staging_path"] = str(staging_path)
    result["live_mutation_committed"] = False
    return result


def _publish_failure_cleanup(
    *,
    skill_dir: Path,
    skill_dir_identity: tuple[int, int, int],
    created_parent_identities: list[tuple[Path, tuple[int, int, int]]],
    skills_root: Path,
) -> None:
    """Drop a half-published live tree created by THIS operation.

    Used when the no-clobber publish of SKILL.md fails after ``skill_dir``
    has been mkdir'd.  Removes only objects whose identity was captured by
    this operation; any sibling that a concurrent creator slipped in is left
    intact (it would fail our ``_ensure_directory_identity`` check).

    Idempotent: best-effort and silent on FileNotFoundError so callers do
    not need to gate cleanup themselves.
    """
    # Remove the SKILL.md we tried to publish, if it carries our own inode.
    skill_md = skill_dir / "SKILL.md"
    try:
        if skill_md.exists() and not skill_md.is_symlink():
            skill_md.unlink()
    except OSError:
        logger.debug("publish-failure unlink of %s failed", skill_md, exc_info=True)
    # Remove the skill directory we just mkdir'd, but only if its identity
    # still matches the snapshot we captured at creation.
    try:
        _ensure_directory_identity(skill_dir, skill_dir_identity)
        # Only rmdir if empty — never recursively delete a foreign dir.
        if not any(skill_dir.iterdir()):
            skill_dir.rmdir()
    except FileNotFoundError:
        pass
    except OSError:
        logger.debug(
            "publish-failure cleanup of skill_dir %s failed", skill_dir, exc_info=True
        )
    # Walk back up our own parent chain.
    for directory, identity in reversed(created_parent_identities):
        if directory.resolve(strict=False) == skills_root.resolve(strict=False):
            continue
        try:
            _ensure_directory_identity(directory, identity)
            if directory.exists() and not any(directory.iterdir()):
                directory.rmdir()
        except FileNotFoundError:
            continue
        except OSError:
            logger.debug(
                "publish-failure cleanup of parent %s failed", directory, exc_info=True
            )


def _copy_skill_into_staging(live_skill_dir: Path, staged_skill_dir: Path) -> None:
    """Copy the live skill tree into a freshly-mkdir'd staging directory.

    Used by edit/patch/write_file to give the scanner a full view of the
    candidate skill without touching the live tree.  Symlinks and junctions
    are REJECTED — the copy must mirror the live tree by inode identity,
    not by re-creating redirects that could smuggle content past the
    scanner.

    All copies are byte-faithful (shutil.copyfile).  Hardlinks would share
    an inode between live and staged, which means a rewrite of the staged
    file (e.g. SKILL.md before the scan, or the supporting file in
    write_file) would silently mutate the live file through the shared
    inode, breaking the scan-before-publish invariant.  Byte copies are
    cheap relative to the rest of the operation and remove the footgun.
    """
    staged_skill_dir.mkdir(parents=True, exist_ok=True)
    for entry in live_skill_dir.rglob("*"):
        rel = entry.relative_to(live_skill_dir)
        target = staged_skill_dir / rel
        st = entry.lstat()
        if stat.S_ISLNK(st.st_mode):
            raise _RollbackFailure(
                f"live skill contains symlink at {entry}; refusing to copy",
                "symlink_detected",
            )
        if stat.S_ISDIR(st.st_mode):
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not stat.S_ISREG(st.st_mode):
            raise _RollbackFailure(
                f"live skill contains non-regular file at {entry}",
                "target_identity_changed",
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(entry), str(target))

# L2 enforcement (post-turn READONLY gate): consult the canonical
# self-improvement policy so writes from the background-review fork are
# refused when HERMES_DISABLE_SELF_IMPROVEMENT or HERMES_READ_ONLY_SESSION
# is activated. This is the belt-and-suspenders for skill_manager's
# existing provenance check — even an agent-created, curator-owned skill
# is off-limits while the session is protected.
from agent.self_improvement_policy import (
    BACKGROUND_REVIEW_ORIGIN as _POLICY_BG_REVIEW_ORIGIN,
    evaluate as _policy_evaluate,
)

def _phase1_skill_get_captured_decision():
    """PHASE 2 (TIER 1): read the typed SelfImprovementDecision via ContextVar.

    The Phase 1 implementation walked the Python stack to recover the
    Decision attribute from a caller in the same thread. Phase 2
    replaces that stack walk with a typed ``ContextVar`` lookup (see
    ``agent/self_improvement_decision_context.py``) that the canonical
    ``AIAgent.__init__`` boundary populates from the captured Phase 1
    Decision.

    Returns the active Decision (Phase 1 frozen shape) or ``None`` if
    no Decision is bound. Callers treat ``None`` as "no captured
    Decision" — the public ``get_self_improvement_decision()`` already
    returns a DENY fallback for that case, so callers should prefer
    that. This shim exists for backward compatibility with any code
    that still uses the Phase 1 helper name.
    """
    try:
        from agent.self_improvement_decision_context import (
            get_self_improvement_decision as _phase2_get_decision,
            DENY_FALLBACK_DECISION as _PHASE2_DENY_FB,
        )
        decision = _phase2_get_decision()
        if decision is _PHASE2_DENY_FB:
            # Mirror Phase 1 semantics: "no captured decision" returns None.
            return None
        return decision
    except Exception:
        return None



from tools.skill_provenance import is_background_review

logger = logging.getLogger(__name__)

_background_review_read_paths: "_ctxvars.ContextVar[frozenset[str]]" = _ctxvars.ContextVar(
    "background_review_read_paths", default=frozenset()
)
_skill_policy_operation: "_ctxvars.ContextVar[str]" = _ctxvars.ContextVar(
    "skill_policy_operation", default="skill_write_file"
)


_SKILL_ACTION_OPERATION_KIND = {
    "create": "skill_create",
    "edit": "skill_edit",
    "patch": "skill_patch",
    "delete": "skill_delete",
    "write_file": "skill_write_file",
    "remove_file": "skill_remove_file",
}


def _skill_policy_denial(action: str, target_path: Path, *, origin: str = "skill_manager") -> Optional[Dict[str, Any]]:
    operation_kind = _SKILL_ACTION_OPERATION_KIND.get(action, action)
    try:
        from agent.session_write_policy import (
            CapabilityGrant,
            evaluate_session_write_policy,
            get_current_session_write_policy,
            policy_evaluation_failure_payload,
        )

        policy = get_current_session_write_policy(protected=False)
        decision = evaluate_session_write_policy(
            policy,
            operation_kind=operation_kind,
            origin=origin,
            target_path=target_path,
            capability=CapabilityGrant("filesystem", operation_kind),
        )
        if decision.denied:
            return decision.denial_payload()
    except Exception as e:
        logger.debug("session write policy skill check failed: %s", e)
        try:
            from agent.session_write_policy import policy_evaluation_failure_payload

            return policy_evaluation_failure_payload(
                operation_kind=operation_kind,
                session_id="",
                target=str(target_path or ""),
                error=e,
            )
        except Exception:
            return {
                "success": False,
                "error": "Session write policy evaluation failed; mutation denied",
                "policy_reason": "policy_evaluation_failed",
                "operation_kind": operation_kind,
                "session_id": "",
                "target": str(target_path or ""),
            }
    return None


def _final_skill_mutation_denial(
    action: str,
    target_path: Path,
    *,
    origin: str,
) -> Optional[Dict[str, Any]]:
    """Fail-closed last-mile authorization for skill-manager mutations.

    A stale public-tool ALLOW, a caller that skips ``skill_manage()``, or a
    background-review fork that crosses into a protected/read-only state must
    not reach a persistent skill mutation. Re-check the canonical typed
    self-improvement decision and session write policy immediately before the
    mutation surface; any missing/invalid decision is DENY by construction.
    """
    operation_kind = _SKILL_ACTION_OPERATION_KIND.get(action, action)
    try:
        from agent.self_improvement_decision_context import get_self_improvement_decision

        decision = get_self_improvement_decision()
    except Exception as exc:
        return {
            "success": False,
            "error": "Self-improvement authorization lookup failed; mutation denied",
            "policy_reason": "self_improvement_policy_evaluation_failed",
            "operation_kind": operation_kind,
            "target": str(target_path or ""),
            "policy_error": f"{type(exc).__name__}: {exc}",
        }
    if not getattr(decision, "allow", False):
        return {
            "success": False,
            "error": (
                "Self-improvement authorization denied skill mutation: "
                f"{getattr(decision, 'reason', '')}"
            ),
            "policy_reason": getattr(decision, "result", "self_improvement_denied"),
            "operation_kind": operation_kind,
            "target": str(target_path or ""),
        }
    try:
        from agent.self_improvement_policy import evaluate as _evaluate_self_improvement_policy

        live_decision = _evaluate_self_improvement_policy(
            environment_disabled=os.environ.get("HERMES_DISABLE_SELF_IMPROVEMENT", ""),
            session_read_only=os.environ.get("HERMES_READ_ONLY_SESSION", ""),
            operation_kind=operation_kind,
            origin=origin,
            target_path=str(target_path or ""),
            explicit_opt_in=True,
        )
    except Exception as exc:
        return {
            "success": False,
            "error": "Self-improvement authorization re-evaluation failed; mutation denied",
            "policy_reason": "self_improvement_policy_evaluation_failed",
            "operation_kind": operation_kind,
            "target": str(target_path or ""),
            "policy_error": f"{type(exc).__name__}: {exc}",
        }
    if not getattr(live_decision, "allow", False):
        return {
            "success": False,
            "error": (
                "Self-improvement authorization denied skill mutation: "
                f"{getattr(live_decision, 'reason', '')}"
            ),
            "policy_reason": getattr(live_decision, "result", "self_improvement_denied"),
            "operation_kind": operation_kind,
            "target": str(target_path or ""),
        }
    return _skill_policy_denial(action, target_path, origin=origin)


class _SkillMutationAborted(Exception):
    """Internal control flow after a structured mutation denial is assigned."""


def _rollback_failed_payload(
    *,
    target: Path,
    scan_error: str,
    rollback_error: Exception | str,
    rollback_failure_kind: str = "physical_failure",
) -> dict[str, Any]:
    return {
        "success": False,
        "error": "Security scan rejected the mutation and rollback failed",
        "policy_reason": "rollback_failed",
        "target": str(target.resolve(strict=False)),
        "scan_error": str(scan_error or ""),
        "rollback_error": str(rollback_error or ""),
        "rollback_failure_kind": rollback_failure_kind,
    }


class _RollbackFailure(Exception):
    def __init__(self, message: str, kind: str = "physical_failure"):
        super().__init__(message)
        self.kind = kind


def _lstat_identity(path: Path) -> tuple[int, int, int]:
    st = path.lstat()
    return st.st_dev, st.st_ino, stat.S_IFMT(st.st_mode)


def _ensure_regular_identity(path: Path, identity: tuple[int, int, int] | None = None) -> os.stat_result:
    st = path.lstat()
    if stat.S_ISLNK(st.st_mode):
        raise _RollbackFailure("target is a symlink", "symlink_detected")
    if not stat.S_ISREG(st.st_mode):
        raise _RollbackFailure("target is not a regular file", "target_identity_changed")
    if identity is not None and (st.st_dev, st.st_ino, stat.S_IFMT(st.st_mode)) != identity:
        raise _RollbackFailure("target identity changed", "target_identity_changed")
    return st


def _ensure_directory_identity(path: Path, identity: tuple[int, int, int]) -> os.stat_result:
    st = path.lstat()
    if stat.S_ISLNK(st.st_mode) or _is_path_redirect(path):
        raise _RollbackFailure("directory is a symlink or junction", "symlink_detected")
    if not stat.S_ISDIR(st.st_mode):
        raise _RollbackFailure("directory is not a directory", "target_identity_changed")
    if (st.st_dev, st.st_ino, stat.S_IFMT(st.st_mode)) != identity:
        raise _RollbackFailure("directory identity changed", "target_identity_changed")
    return st


def _rollback_failed_from_exception(
    *,
    target: Path,
    scan_error: str,
    exc: Exception | str,
) -> dict[str, Any]:
    kind = exc.kind if isinstance(exc, _RollbackFailure) else "physical_failure"
    return _rollback_failed_payload(
        target=target,
        scan_error=scan_error,
        rollback_error=exc,
        rollback_failure_kind=kind,
    )


def _cleanup_empty_parents(path: Path, stop_at: Path) -> None:
    parent = path.parent
    stop = stop_at.resolve(strict=False)
    while parent != stop and stop in parent.resolve(strict=False).parents:
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


def _cleanup_created_parents(
    created_parent_identities: list[tuple[Path, tuple[int, int, int]]],
    *,
    stop_at: Path,
) -> None:
    stop = stop_at.resolve(strict=False)
    for directory, identity in reversed(created_parent_identities):
        if directory.resolve(strict=False) == stop:
            continue
        try:
            _ensure_directory_identity(directory, identity)
            directory.rmdir()
        except FileNotFoundError:
            continue
        except OSError as exc:
            if getattr(exc, "errno", None):
                raise
            raise


def _remove_own_new_file(
    target: Path,
    *,
    file_identity: tuple[int, int, int],
    candidate_bytes: bytes,
) -> None:
    _ensure_regular_identity(target, file_identity)
    if target.read_bytes() != candidate_bytes:
        raise _RollbackFailure("target content changed", "target_identity_changed")
    target.unlink()
    if os.path.lexists(target):
        raise _RollbackFailure("target still exists after unlink", "physical_failure")


def _restore_original_file(
    target: Path,
    *,
    original_bytes: bytes,
    original_mode: int,
    candidate_bytes: bytes,
) -> None:
    st = _ensure_regular_identity(target)
    if target.read_bytes() != candidate_bytes:
        raise _RollbackFailure("target content changed", "concurrent_modification")
    fd, temp_path = tempfile.mkstemp(
        dir=str(target.parent),
        prefix=f".{target.name}.rollback.",
        suffix="",
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(original_bytes)
        os.chmod(temp_path, stat.S_IMODE(original_mode))
        atomic_replace(temp_path, target)
    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            logger.error("Failed to remove temporary rollback file %s", temp_path, exc_info=True)
        raise
    restored = _ensure_regular_identity(target)
    if target.read_bytes() != original_bytes:
        raise _RollbackFailure("restored bytes did not match original", "physical_failure")
    if stat.S_IMODE(restored.st_mode) != stat.S_IMODE(original_mode):
        raise _RollbackFailure("restored mode did not match original", "physical_failure")


def _rollback_created_skill(
    *,
    skill_dir: Path,
    skill_dir_identity: tuple[int, int, int],
    skill_md: Path,
    skill_md_identity: tuple[int, int, int],
    candidate_bytes: bytes,
    created_parent_identities: list[tuple[Path, tuple[int, int, int]]],
    skills_root: Path,
) -> None:
    _ensure_directory_identity(skill_dir, skill_dir_identity)
    _remove_own_new_file(skill_md, file_identity=skill_md_identity, candidate_bytes=candidate_bytes)
    _ensure_directory_identity(skill_dir, skill_dir_identity)
    try:
        skill_dir.rmdir()
    except OSError as exc:
        raise _RollbackFailure(f"skill directory not empty after own file removal: {exc}", "concurrent_modification")
    _cleanup_created_parents(created_parent_identities, stop_at=skills_root)


def _with_skill_operation(action: str):
    return _skill_policy_operation.set(_SKILL_ACTION_OPERATION_KIND.get(action, action))


def _reset_skill_operation(token) -> None:
    _skill_policy_operation.reset(token)


def mark_background_review_skill_read(path: Path) -> None:
    """Record that the active background-review fork has read a skill file.

    The autonomous review fork is allowed to evolve skills, but it must not
    patch or rewrite content it has only inferred from the transcript.  The
    skill_view tool calls this after returning file content to the model; write
    paths below require the corresponding target path to be present when the
    current origin is ``background_review``.
    """
    try:
        from tools.skill_provenance import is_background_review
        if not is_background_review():
            return
    except Exception:
        return

    try:
        resolved = str(path.resolve())
    except Exception:
        resolved = str(path)
    current = set(_background_review_read_paths.get())
    current.add(resolved)
    _background_review_read_paths.set(frozenset(current))


def _background_review_has_read(path: Path) -> bool:
    try:
        resolved = str(path.resolve())
    except Exception:
        resolved = str(path)
    return resolved in _background_review_read_paths.get()


def _reset_background_review_read_marks() -> None:
    """Test helper: clear read-before-write marks for the current context."""
    _background_review_read_paths.set(frozenset())

# Import security scanner — external hub installs always get scanned;
# agent-created skills only get scanned when skills.guard_agent_created is on.
try:
    from tools.skills_guard import scan_skill, should_allow_install, format_scan_report
    _GUARD_AVAILABLE = True
except ImportError:
    _GUARD_AVAILABLE = False


def _guard_agent_created_enabled() -> bool:
    """Read skills.guard_agent_created from config (default False).

    Off by default because the agent can already execute the same code
    paths via terminal() with no gate, so the scan adds friction without
    meaningful security.  Users who want belt-and-suspenders can turn it
    on via `hermes config set skills.guard_agent_created true`.
    """
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        return is_truthy_value(
            cfg_get(cfg, "skills", "guard_agent_created"),
            default=False,
        )
    except Exception:
        return False


def _security_scan_skill(skill_dir: Path) -> Optional[str]:
    """Scan a skill directory after write. Returns error string if blocked, else None.

    No-op when skills.guard_agent_created is disabled (the default).
    """
    if not _GUARD_AVAILABLE:
        return None
    if not _guard_agent_created_enabled():
        return None
    try:
        result = scan_skill(skill_dir, source="agent-created")
        allowed, reason = should_allow_install(result)
        if allowed is False:
            report = format_scan_report(result)
            return f"Security scan blocked this skill ({reason}):\n{report}"
        if allowed is None:
            # "ask" verdict — for agent-created skills this means dangerous
            # findings were detected.  Surface as an error so the agent can
            # retry with the flagged content removed.
            report = format_scan_report(result)
            logger.warning("Agent-created skill blocked (dangerous findings): %s", reason)
            return f"Security scan blocked this skill ({reason}):\n{report}"
    except Exception as e:
        logger.warning("Security scan failed for %s: %s", skill_dir, e, exc_info=True)
        return f"Skill security scan failed; mutation rejected ({type(e).__name__})"
    return None


def _security_scan_skill_fail_closed(skill_dir: Path) -> Optional[str]:
    try:
        return _security_scan_skill(skill_dir)
    except Exception as exc:
        logger.exception("Skill security scan raised; rejecting mutation")
        return (
            "Skill security scan failed; mutation rejected "
            f"({type(exc).__name__})"
        )

import yaml


# All skills live in ~/.hermes/skills/ (single source of truth)
HERMES_HOME = get_hermes_home()
SKILLS_DIR = HERMES_HOME / "skills"
_SKILLS_DIR_AT_IMPORT = SKILLS_DIR


def _skills_dir() -> Path:
    """Return the active profile's skills directory at call time.

    Long-lived multi-profile runtimes (Dashboard/TUI/Desktop backend, cron,
    kanban workers) import this module once under the launch HERMES_HOME and
    later bind a different profile per session (#40677). Honor an explicitly
    patched module-level ``SKILLS_DIR`` (tests), otherwise resolve from the
    live profile-scoped HERMES_HOME on every call.
    """
    configured = Path(SKILLS_DIR)
    if configured != _SKILLS_DIR_AT_IMPORT:
        return configured
    return get_hermes_home() / "skills"

MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024


def _containing_skills_root(skill_path: Path) -> Path:
    """Return the skills root directory (local or external_dirs entry) that
    contains ``skill_path``.  Falls back to the local ``SKILLS_DIR`` if no
    match is found (defensive — callers should have located the skill via
    ``_find_skill`` first).
    """
    from agent.skill_utils import get_all_skills_dirs

    try:
        resolved = skill_path.resolve()
    except OSError:
        resolved = skill_path

    for root in get_all_skills_dirs():
        try:
            resolved.relative_to(root.resolve())
            return root
        except (ValueError, OSError):
            continue
    return _skills_dir()


def _is_path_redirect(path: Path) -> bool:
    """True when ``path`` is a symlink or (on Windows) a directory junction.

    Either form lets a poisoned skills tree redirect a subsequent
    ``shutil.rmtree`` to content outside the skills root. ``is_junction``
    only exists on Python 3.12+ Windows; gate with ``hasattr``.
    """
    try:
        return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())
    except OSError:
        return False


def _validate_delete_target(skill_dir: Path) -> Optional[str]:
    """Last-line guard before the recursive delete in ``_delete_skill``.

    ``_find_skill`` already restricts ``skill_dir`` to a real ``SKILL.md``
    parent discovered by walking the skills roots, so the agent cannot inject
    an arbitrary path the way Kilo Code's HTTP endpoint could (their issue
    #11227: a built-in-skill sentinel resolved to the server cwd and a
    recursive delete wiped the user's entire working directory). This is the
    matching defense-in-depth for our agent-facing ``skill_manage`` delete
    path: even if discovery or a poisoned tree hands us a bad directory, never
    recursively delete

      1. a path that is not strictly *inside* one of the known skills roots,
      2. a skills root itself (would wipe every installed skill), or
      3. a directory reached via a symlink / junction (``rmtree`` would follow
         it into content outside the skills tree).

    Returns an error string to refuse on, or ``None`` when the delete is safe.
    """
    from agent.skill_utils import get_all_skills_dirs

    # (3) Reject symlink/junction redirects on the skill directory itself.
    if _is_path_redirect(skill_dir):
        return (
            f"Refusing to delete '{skill_dir}': the skill directory is a "
            f"symlink/junction. Remove the link target manually if intended."
        )

    try:
        resolved = skill_dir.resolve()
    except OSError as exc:
        return f"Refusing to delete '{skill_dir}': could not resolve path ({exc})."

    roots = []
    for root in get_all_skills_dirs():
        try:
            roots.append(root.resolve())
        except OSError:
            continue

    for root in roots:
        # (2) Never rmtree a skills root itself.
        if resolved == root:
            return (
                f"Refusing to delete '{skill_dir}': resolves to the skills root "
                f"itself, which would remove every installed skill."
            )
        # (1) Must be strictly inside a known root.
        try:
            rel = resolved.relative_to(root)
        except ValueError:
            continue
        if rel.parts:  # at least one component below the root
            return None

    return (
        f"Refusing to delete '{skill_dir}': path does not resolve inside any "
        f"known skills root."
    )


def _pinned_guard(name: str) -> Optional[str]:
    """Return a refusal message if *name* is pinned, else None.

    Pin protects a skill from **deletion** — both the curator's auto-archive
    passes and the agent's ``skill_manage(action="delete")`` tool call. The
    agent can still patch/edit pinned skills; pin only guards against
    irrecoverable loss, not against content evolution.

    Best-effort: if the sidecar is unreadable we let the delete through
    rather than block on a broken telemetry file.
    """
    try:
        from tools import skill_usage
        rec = skill_usage.get_record(name)
        if rec.get("pinned"):
            return (
                f"Skill '{name}' is pinned and cannot be deleted by "
                f"skill_manage. Ask the user to run "
                f"`hermes curator unpin {name}` if they want to delete it. "
                f"Patches and edits are allowed on pinned skills; only "
                f"deletion is blocked."
            )
    except Exception:
        logger.debug("pinned-guard lookup failed for %s", name, exc_info=True)
    return None


def _background_review_self_improvement_guard(
    name: str,
    skill_dir: Path,
    action: str,
) -> Optional[Dict[str, Any]]:
    """L2 enforcement: deny background-review writes under session protection.

    Consults the canonical self-improvement policy (which honours
    ``HERMES_DISABLE_SELF_IMPROVEMENT`` and ``HERMES_READ_ONLY_SESSION``)
    and refuses the write when the active write origin is the
    background-review fork and the policy returns a non-ALLOW decision.
    Reads only env vars — never inspects prompt text. Returns an error
    dict on DENY, or ``None`` to fall through to the existing
    provenance / ownership checks.

    This guard short-circuits *before* the pinned / external /
    bundled / hub guards so a session under global protection is
    empty-handed regardless of which skill it tried to edit. The
    pre-existing provenance and ownership guards continue to defend
    the normal-session path with their original semantics.
    """
    provenance_failed = False
    try:
        if not is_background_review():
            return None
    except Exception:
        provenance_failed = True

    # PHASE 2 (TIER 1): read the typed Decision from the ContextVar.
    # The Phase 1 helper walked the stack; Phase 2 reads the canonical
    # ContextVar populated by ``AIAgent.__init__`` at session start.
    # NO os.environ sampling here — that is canonical-init-only.
    try:
        from agent.self_improvement_decision_context import (
            get_self_improvement_decision as _phase2_skill_get_decision,
        )
        decision = _phase2_skill_get_decision()
    except Exception:
        # Fail-closed: a ContextVar lookup failure must never let a
        # write through.
        logger.exception(
            "self_improvement_policy ContextVar lookup raised in "
            "skill guard; defaulting to deny"
        )
        return {
            "success": False,
            "error": (
                f"Refusing background {action} for skill '{name}': "
                "self-improvement context lookup raised; defaulting to deny."
            ),
            "_self_improvement_guard": True,
        }

    if getattr(decision, "allow", False):
        return None

    # Structured audit log line. No prompt text. No skill content. No
    # secrets. Identifies the decision, the reason, the operation, the
    # origin and the session id when available.
    _session_id = ""
    try:
        _session_id = os.environ.get("HERMES_SESSION_ID", "") or ""
    except Exception:
        _session_id = ""
    logger.warning(
        "self_improvement_policy deny decision=DENY reason=%r "
        "operation_kind=skill_write origin=background_review "
        "session_id=%s skill=%s action=%s",
        getattr(decision, "reason", ""),
        _session_id,
        name,
        action,
    )
    return {
        "success": False,
        "error": (
            f"Refusing background {action} for skill '{name}': "
            + (
                "self-improvement provenance probe failed; defaulting to deny. "
                if provenance_failed
                else ""
            )
            + f"{getattr(decision, 'reason', '')}"
        ),
        "_self_improvement_guard": True,
    }


def _background_review_write_guard(
    name: str,
    skill_dir: Path,
    action: str,
) -> Optional[Dict[str, Any]]:
    """Refuse autonomous curator writes to externally owned skills.

    Foreground agents may still perform user-directed edits to external,
    bundled, or hub-installed skills. The background review fork is different:
    it is autonomous lifecycle maintenance, so its write surface is restricted
    to local curator-owned sediment.
    """
    # L2 enforcement FIRST so a session under protection exits early
    # regardless of the skill's ownership class.
    _early = _background_review_self_improvement_guard(name, skill_dir, action)
    if _early is not None:
        return _early

    try:
        from tools.skill_provenance import is_background_review
        if not is_background_review():
            return None
    except Exception:
        return None

    # Pin must be respected by autonomous maintenance. The curator already
    # skips pinned skills from every auto-transition; the background review
    # fork is the same kind of autonomous, no-user-present actor, so it must
    # not write to a pinned skill either (issue #25839). This is stricter than
    # the foreground ``_pinned_guard`` (which only blocks deletion) precisely
    # because there is no user in the loop to consent to an edit here.
    try:
        from tools import skill_usage
        if skill_usage.get_record(name).get("pinned"):
            return {
                "success": False,
                "error": (
                    f"Refusing background curator {action} for pinned skill "
                    f"'{name}': pinned skills are off-limits to autonomous "
                    "maintenance. Ask the user to run "
                    f"`hermes curator unpin {name}` if they want it changed."
                ),
            }
    except Exception:
        logger.debug("pinned skill guard lookup failed for %s", name, exc_info=True)

    try:
        from agent.skill_utils import is_external_skill_path
        if is_external_skill_path(skill_dir):
            return {
                "success": False,
                "error": (
                    f"Refusing background curator {action} for skill '{name}': "
                    "the skill lives in skills.external_dirs, which are "
                    "externally owned and read-only to autonomous curation."
                ),
            }
    except Exception:
        logger.debug("external skill guard lookup failed for %s", name, exc_info=True)

    try:
        from tools import skill_usage
        if skill_usage.is_protected_builtin(name):
            return {
                "success": False,
                "error": (
                    f"Refusing background curator {action} for protected "
                    f"built-in skill '{name}'."
                ),
            }
        if skill_usage.is_hub_installed(name):
            return {
                "success": False,
                "error": (
                    f"Refusing background curator {action} for hub-installed "
                    f"skill '{name}'."
                ),
            }
        if skill_usage.is_bundled(name):
            return {
                "success": False,
                "error": (
                    f"Refusing background curator {action} for bundled "
                    f"skill '{name}'."
                ),
            }
        # Skills that are not curator-managed are off-limits to autonomous
        # curation. This prevents the LLM consolidation pass from mutating
        # skills the user owns (manually authored, URL-installed, or created by
        # a foreground `skill_manage(create)` at the user's request), which lack
        # the `created_by: "agent"` marker.
        #
        # A MISSING record and an explicit `created_by: null` must resolve
        # IDENTICALLY (issue #67140). Keying on `isinstance(usage_rec, dict)`
        # made the policy depend on the guard's own side effect: a local skill
        # with no telemetry record passed, the successful write called
        # bump_patch() which created a `created_by: null` record, and the very
        # same write was refused from then on. "Allowed exactly once" is not a
        # policy — it is a race with our own bookkeeping. Fail closed for both
        # shapes; `hermes curator adopt <name>` is the supported way in.
        usage_data = skill_usage.load_usage()
        usage_rec = usage_data.get(name)
        if not skill_usage._is_curator_managed_record(usage_rec):
            if isinstance(usage_rec, dict):
                _detail = f"created_by={usage_rec.get('created_by')!r}"
            else:
                _detail = "no usage record"
            return {
                "success": False,
                "error": (
                    f"Refusing background curator {action} for skill "
                    f"'{name}': the skill is not curator-managed ({_detail}). "
                    "User-owned skills are off-limits to autonomous curation. "
                    f"Run `hermes curator adopt {name}` to opt it in."
                ),
            }
    except Exception:
        logger.warning("owned skill guard lookup failed for %s", name, exc_info=True)
        return {
            "success": False,
            "error": (
                f"Refusing background curator {action} for skill '{name}': "
                "agent ownership could not be verified because the provenance "
                "record is unavailable or unreadable."
            ),
        }
    return None


def _background_review_read_before_write_guard(
    name: str,
    target: Path,
    action: str,
    file_label: str,
) -> Optional[Dict[str, Any]]:
    """Require review forks to load the exact target before mutating it."""
    # L2 enforcement: consult the canonical self-improvement policy
    # FIRST so a session under global protection exits before the
    # read-tracking contract is invoked (no point loading then refusing).
    try:
        _skill_dir_for_guard = target.parent
    except Exception:
        _skill_dir_for_guard = None
    _early = _background_review_self_improvement_guard(
        name, _skill_dir_for_guard, action  # type: ignore[arg-type]
    )
    if _early is not None:
        return _early

    try:
        from tools.skill_provenance import is_background_review
        if not is_background_review():
            return None
    except Exception:
        return None

    if _background_review_has_read(target):
        return None

    return {
        "success": False,
        "error": (
            f"Refusing background curator {action} for skill '{name}': "
            f"the current {file_label} content has not been loaded in this "
            f"review turn. Call skill_view(name) for SKILL.md, or "
            f"skill_view(name, file_path=...) for a supporting file, then "
            f"retry the write using the content just returned."
        ),
        "_read_before_write_required": True,
    }


def _background_review_preflight(
    action: str, name: str, category: str = None
) -> Optional[Dict[str, Any]]:
    if action == "create":
        return _background_review_self_improvement_guard(
            name, _resolve_skill_dir(name, category), action
        )
    if action not in {"edit", "patch", "delete", "write_file", "remove_file"}:
        return None
    existing = _find_skill(name)
    if not existing:
        return None
    return _background_review_write_guard(name, existing["path"], action)


def _curator_consolidation_delete_guard(
    name: str, absorbed_into: Optional[str]
) -> Optional[Dict[str, Any]]:
    """Fail closed on unverified deletes during the curator consolidation pass.

    The curator's forked review agent (``is_background_review()``) runs the
    LLM umbrella-building pass. Its only legitimate ``skill_manage(delete)`` is
    a *verified consolidation*: the skill's content was absorbed into an
    umbrella, declared via ``absorbed_into=<umbrella>`` where the umbrella
    exists on disk (validated separately in ``_delete_skill``).

    A delete with no forwarding target — ``absorbed_into`` omitted (``None``)
    or empty (``""``) — is the fail-open behavior reported in #29912: the
    consolidation pass archived whole clusters of active skills with zero
    verified consolidations (``consolidated_this_run == 0``), leaving active
    automations pointing at names that no longer resolve. The deterministic
    inactivity prune is the only legitimate prune path, and it archives via
    ``skill_usage.archive_skill()`` directly without ever calling
    ``skill_manage`` — so a bare prune reaching here can only be the LLM pass
    pruning without consolidation evidence. Refuse it; keep the skill active.

    Returns an error dict to abort the delete, or ``None`` when the delete is
    allowed to proceed (not the curator pass, or a declared consolidation).
    """
    try:
        from tools.skill_provenance import is_background_review
        if not is_background_review():
            return None
    except Exception:
        return None

    declared = isinstance(absorbed_into, str) and absorbed_into.strip()
    if declared:
        return None

    return {
        "success": False,
        "error": (
            f"Refusing background curator delete of skill '{name}': the "
            "consolidation pass may only archive a skill it has absorbed into "
            "an umbrella. Pass absorbed_into=<umbrella> (the umbrella must "
            "already exist) to record a verified consolidation. Pruning a "
            "skill with no forwarding target is not permitted here — the "
            "deterministic inactivity prune handles staleness archival "
            "separately. Keeping '{name}' active.".format(name=name)
        ),
        "_fail_closed": True,
    }


MAX_SKILL_CONTENT_CHARS = 100_000   # ~36k tokens at 2.75 chars/token
MAX_SKILL_FILE_BYTES = 1_048_576    # 1 MiB per supporting file

# Characters allowed in skill names (filesystem-safe, URL-friendly)
VALID_NAME_RE = re.compile(r'^[a-z0-9][a-z0-9._-]*$')

# Subdirectories allowed for write_file/remove_file
ALLOWED_SUBDIRS = {"references", "templates", "scripts", "assets"}


# =============================================================================
# Validation helpers
# =============================================================================

def _validate_name(name: str) -> Optional[str]:
    """Validate a skill name. Returns error message or None if valid."""
    if not name:
        return "Skill name is required."
    if len(name) > MAX_NAME_LENGTH:
        return f"Skill name exceeds {MAX_NAME_LENGTH} characters."
    if not VALID_NAME_RE.match(name):
        return (
            f"Invalid skill name '{name}'. Use lowercase letters, numbers, "
            f"hyphens, dots, and underscores. Must start with a letter or digit."
        )
    return None


def _validate_category(category: Optional[str]) -> Optional[str]:
    """Validate an optional category name used as a single directory segment."""
    if category is None:
        return None
    if not isinstance(category, str):
        return "Category must be a string."

    category = category.strip()
    if not category:
        return None
    if "/" in category or "\\" in category:
        return (
            f"Invalid category '{category}'. Use lowercase letters, numbers, "
            "hyphens, dots, and underscores. Categories must be a single directory name."
        )
    if len(category) > MAX_NAME_LENGTH:
        return f"Category exceeds {MAX_NAME_LENGTH} characters."
    if not VALID_NAME_RE.match(category):
        return (
            f"Invalid category '{category}'. Use lowercase letters, numbers, "
            "hyphens, dots, and underscores. Categories must be a single directory name."
        )
    return None


def _validate_frontmatter(content: str, *, new_skill: bool = False) -> Optional[str]:
    """
    Validate that SKILL.md content has proper frontmatter with required fields.
    Returns error message or None if valid.

    When ``new_skill`` is True (create path only), the description must also
    fit the 60-char system-prompt budget (SKILL_PROMPT_DESC_LIMIT) so newly
    authored skills never lose routing signal to index truncation. Edit and
    patch paths deliberately skip this so existing over-limit skills remain
    maintainable while their descriptions are cleaned up.
    """
    if not content.strip():
        return "Content cannot be empty."

    # Tolerate a leading UTF-8 BOM (Windows editors) before the fence.
    content = content.lstrip("\ufeff")

    if not content.startswith("---"):
        return "SKILL.md must start with YAML frontmatter (---). See existing skills for format."

    end_match = re.search(r'\n---\s*\n', content[3:])
    if not end_match:
        return "SKILL.md frontmatter is not closed. Ensure you have a closing '---' line."

    yaml_content = content[3:end_match.start() + 3]

    try:
        parsed = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        return f"YAML frontmatter parse error: {e}"

    if not isinstance(parsed, dict):
        return "Frontmatter must be a YAML mapping (key: value pairs)."

    if "name" not in parsed:
        return "Frontmatter must include 'name' field."
    if "description" not in parsed:
        return "Frontmatter must include 'description' field."
    desc = str(parsed["description"])
    if len(desc) > MAX_DESCRIPTION_LENGTH:
        return f"Description exceeds {MAX_DESCRIPTION_LENGTH} characters."
    if new_skill and len(desc.strip().strip("'\"")) > SKILL_PROMPT_DESC_LIMIT:
        return (
            f"Description is {len(desc.strip())} chars — new skills must fit the "
            f"{SKILL_PROMPT_DESC_LIMIT}-char system-prompt budget (one sentence, "
            f"trigger first, ends with a period). The skill index truncates "
            f"longer descriptions to {SKILL_PROMPT_DESC_LIMIT - 3} chars + '...', "
            f"destroying the routing signal. Move detail into the skill body."
        )

    body = content[end_match.end() + 3:].strip()
    if not body:
        return "SKILL.md must have content after the frontmatter (instructions, procedures, etc.)."

    return None


def _validate_content_size(content: str, label: str = "SKILL.md") -> Optional[str]:
    """Check that content doesn't exceed the character limit for agent writes.

    Returns an error message or None if within bounds.
    """
    if len(content) > MAX_SKILL_CONTENT_CHARS:
        return (
            f"{label} content is {len(content):,} characters "
            f"(limit: {MAX_SKILL_CONTENT_CHARS:,}). "
            f"Consider splitting into a smaller SKILL.md with supporting files "
            f"in references/ or templates/."
        )
    return None


def _resolve_skill_dir(name: str, category: str = None) -> Path:
    """Build the directory path for a new skill, optionally under a category."""
    if category:
        return _skills_dir() / category / name
    return _skills_dir() / name


def _find_skill(name: str) -> Optional[Dict[str, Any]]:
    """
    Find a skill by name across all skill directories.

    Searches the local skills dir (~/.hermes/skills/) first, then any
    external dirs configured via skills.external_dirs.  Returns
    {"path": Path} or None.
    """
    from agent.skill_utils import get_all_skills_dirs, is_excluded_skill_path
    for skills_dir in get_all_skills_dirs():
        if not skills_dir.exists():
            continue
        for skill_md in skills_dir.rglob("SKILL.md"):
            if is_excluded_skill_path(skill_md):
                continue
            if skill_md.parent.name == name:
                return {"path": skill_md.parent}
    return None


def _maybe_auto_propose_org_edit(name: str, skill_path: Path) -> Optional[str]:
    """Submit an org-skill edit upstream when `sync.org_auto_propose` is on.

    Returns a short note for the tool result, or None when nothing happened.
    Never raises: an offline/failed submission must not fail the edit itself —
    the change is already saved locally and can be proposed later.
    """
    try:
        from agent.skill_utils import is_org_mirror_path
        from tools import skills_sync_client as ssc

        if not is_org_mirror_path(skill_path, _skills_dir()):
            return None
        if not ssc.sync_org_auto_propose():
            return (
                f"This skill is shared by your organisation. Your edit is "
                f"saved locally and will not be overwritten by org updates. "
                f"Run `hermes sync propose {name}` to share it back."
            )
        result = ssc.propose_skill(name)
        if result.get("proposal_pending"):
            return (
                f"Auto-proposed to your organisation as proposal "
                f"#{result.get('proposal_id')} (pending admin review)."
            )
        return "Auto-proposed to your organisation (merged into the shared set)."
    except Exception as e:
        logger.debug("auto-propose skipped for %s: %s", name, e)
        return (
            f"Edit saved locally. Could not submit it to your organisation "
            f"right now — run `hermes sync propose {name}` to retry."
        )


def _org_mirror_write_guard(name: str, skill_path: Path, action: str) -> Optional[Dict[str, Any]]:
    """Org-shared skills are EDITABLE IN PLACE — this only blocks deletion.

    Earlier versions refused every write to `_org/`, which broke the learning
    loop exactly where it matters most: the agent is told to patch a skill the
    moment it finds a gap, and shared skills are the ones the most people use.
    Blocking that froze org skills while personal ones kept improving, and the
    "fork it into a personal skill" alternative is not something an agent does
    mid-task — so improvements were simply lost.

    Now an edit lands in the mirror and is protected from being overwritten by
    the next org pull (see the baseline sidecar in skills_sync_client). It
    reaches the organisation when the user runs `hermes sync propose`, or
    immediately if `sync.org_auto_propose` is on.

    Deletion is still refused: the mirror is a materialized view of the org
    HEAD, so a local delete is meaningless (the next pull restores it) and
    removing a skill for the organisation is an admin action, not a local one.
    """
    if action not in {"delete", "remove_file"}:
        return None
    try:
        from agent.skill_utils import is_org_mirror_path

        if is_org_mirror_path(skill_path, _skills_dir()):
            return {
                "success": False,
                "error": (
                    f"Cannot {action} '{name}' locally: it is shared by your "
                    "organisation, so a local delete would just come back on "
                    "the next sync. Ask an org admin to remove it for "
                    "everyone. (Editing it IS allowed — your changes are kept "
                    "and can be proposed back with `hermes sync propose "
                    f"{name}`.)"
                ),
            }
    except Exception:
        logger.debug("org mirror guard lookup failed for %s", name, exc_info=True)
    return None


def _find_skill_in_other_profiles(name: str) -> List[Tuple[str, Path]]:
    """Look for ``name`` under SKILL.md across OTHER Hermes profiles.

    Returns a list of ``(profile_name, skill_dir)`` pairs. Used to make
    the "Skill X not found" error explain when the user is editing the
    wrong profile. Empty list when no other profile has the skill (or
    when profile discovery fails — fail-quiet, the caller falls back to
    the plain "not found" error).
    """
    matches: List[Tuple[str, Path]] = []
    try:
        from hermes_constants import get_default_hermes_root
        from agent.skill_utils import is_excluded_skill_path
    except Exception:
        return matches

    try:
        root = get_default_hermes_root()
    except Exception:
        return matches

    # Collect (profile_name, skills_dir) for every profile EXCEPT the
    # one whose skills dir we already searched in _find_skill().
    _active = _skills_dir()
    active_dir = _active.resolve() if _active.exists() else _active
    candidates: List[Tuple[str, Path]] = []

    # Default profile (~/.hermes/skills) — only consider when active is non-default.
    default_skills = root / "skills"
    try:
        if default_skills.resolve() != active_dir:
            candidates.append(("default", default_skills))
    except (OSError, RuntimeError):
        pass

    # All named profiles (~/.hermes/profiles/*/skills)
    profiles_root = root / "profiles"
    if profiles_root.is_dir():
        try:
            for entry in profiles_root.iterdir():
                if not entry.is_dir():
                    continue
                pskills = entry / "skills"
                try:
                    if pskills.resolve() == active_dir:
                        continue
                except (OSError, RuntimeError):
                    continue
                candidates.append((entry.name, pskills))
        except OSError:
            pass

    for profile_name, skills_dir in candidates:
        if not skills_dir.is_dir():
            continue
        try:
            for skill_md in skills_dir.rglob("SKILL.md"):
                if is_excluded_skill_path(skill_md):
                    continue
                if skill_md.parent.name == name:
                    matches.append((profile_name, skill_md.parent))
                    break  # one match per profile is enough
        except OSError:
            continue
    return matches


def _skill_not_found_error(name: str, suffix: str = "") -> str:
    """Build a "skill not found" error that names other profiles holding
    the same skill, so the agent can recognize a profile-scoping mistake.

    ``suffix`` is appended after the cross-profile hint if present
    (e.g. ``" Create it first with action='create'."``).
    """
    from agent.file_safety import _resolve_active_profile_name
    active = _resolve_active_profile_name()
    base = f"Skill '{name}' not found in active profile '{active}'."

    others = _find_skill_in_other_profiles(name)
    if others:
        if len(others) == 1:
            other_profile, other_path = others[0]
            base += (
                f" A skill by that name exists in profile "
                f"'{other_profile}' ({other_path}). To edit a skill in "
                f"another profile, switch profiles (`hermes -p "
                f"{other_profile}`) or operate via explicit file tools "
                f"with ``cross_profile=True``."
            )
        else:
            names = ", ".join(f"'{p}'" for p, _ in others)
            base += (
                f" Skills by that name exist in other profiles: {names}. "
                f"Switch profiles (`hermes -p <name>`) to edit there, or "
                f"operate via explicit file tools with ``cross_profile=True``."
            )
    else:
        base += " Use skills_list() to see available skills."

    if suffix:
        base += suffix
    return base


def _validate_file_path(file_path: str) -> Optional[str]:
    """
    Validate a file path for write_file/remove_file.
    Must be under an allowed subdirectory and not escape the skill dir.
    """
    from tools.path_security import has_traversal_component

    if not file_path:
        return "file_path is required."

    normalized = Path(file_path)

    # Prevent path traversal (checked before any allow-listing so the SKILL.md
    # exception below can never be reached by a traversal-laden path).
    if has_traversal_component(file_path):
        return "Path traversal ('..') is not allowed."

    # SKILL.md is the canonical skill file and lives at the skill root, not
    # under an allowed subdirectory. Accept its two natural spellings —
    # 'SKILL.md' and '<skill-name>/SKILL.md' — so callers can target the main
    # file. The traversal guard above still applies, so this can't escape.
    if normalized.parts and normalized.name == "SKILL.md":
        if len(normalized.parts) == 1 or len(normalized.parts) == 2:
            return None

    # Must be under an allowed subdirectory
    if not normalized.parts or normalized.parts[0] not in ALLOWED_SUBDIRS:
        allowed = ", ".join(sorted(ALLOWED_SUBDIRS))
        return f"File must be under one of: {allowed}. Got: '{file_path}'"

    # Must have a filename (not just a directory)
    if len(normalized.parts) < 2:
        return f"Provide a file path, not just a directory. Example: '{normalized.parts[0]}/myfile.md'"

    return None


def _resolve_skill_target(skill_dir: Path, file_path: str) -> Tuple[Optional[Path], Optional[str]]:
    """Resolve a supporting-file path and ensure it stays within the skill directory."""
    from tools.path_security import validate_within_dir

    target = skill_dir / file_path
    error = validate_within_dir(target, skill_dir)
    if error:
        return None, error
    return target, None


def _atomic_write_text(file_path: Path, content: str, encoding: str = "utf-8") -> None:
    """
    Atomically write text content to a file.

    Uses a temporary file in the same directory and os.replace() to ensure
    the target file is never left in a partially-written state if the process
    crashes or is interrupted.

    Args:
        file_path: Target file path
        content: Content to write
        encoding: Text encoding (default: utf-8)
    """
    operation_kind = _skill_policy_operation.get()
    action = {
        "skill_create": "create",
        "skill_edit": "edit",
        "skill_patch": "patch",
        "skill_write_file": "write_file",
    }.get(operation_kind, "write_file")
    denial = _final_skill_mutation_denial(action, file_path, origin="skill_manager_atomic_write")
    if denial is not None:
        raise PermissionError(denial["error"])
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        dir=str(file_path.parent),
        prefix=f".{file_path.name}.tmp.",
        suffix="",
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
        denial = _final_skill_mutation_denial(
            action, file_path, origin="skill_manager_atomic_replace"
        )
        if denial is not None:
            raise PermissionError(denial["error"])
        atomic_replace(temp_path, file_path)
    except Exception:
        # Clean up temp file on error
        try:
            os.unlink(temp_path)
        except OSError:
            logger.error("Failed to remove temporary file %s during atomic write", temp_path, exc_info=True)
        raise


# =============================================================================
# Core actions
# =============================================================================


def _add_description_prompt_preview(result: Dict[str, Any], content: str) -> None:
    """Append a system_prompt_preview field when the description will be truncated."""
    fm, _ = _parse_frontmatter(content)
    if is_skill_description_truncated_for_prompt(fm):
        result["system_prompt_preview"] = (
            f"System prompt will show: \"{extract_skill_description(fm)}\" — "
            f"keep the trigger self-contained in the first "
            f"{SKILL_PROMPT_DESC_LIMIT - 3} chars."
        )


def _create_skill(name: str, content: str, category: str = None) -> Dict[str, Any]:
    """Create a new user skill with SKILL.md content."""
    # Validate name
    err = _validate_name(name)
    if err:
        return {"success": False, "error": err}

    err = _validate_category(category)
    if err:
        return {"success": False, "error": err}

    # Validate content
    err = _validate_frontmatter(content, new_skill=True)
    if err:
        return {"success": False, "error": err}

    err = _validate_content_size(content)
    if err:
        return {"success": False, "error": err}

    # Check for name collisions across all directories -- initial scan OUTSIDE
    # the global lock.  This scan is an optimization (early duplicate refusal
    # when a concurrent create has already committed).  The DECISIVE uniqueness
    # scan is repeated INSIDE the locks below; a concurrent create that is
    # also racing on the same normalized name MUST be serialized behind the
    # global normalized-name lock.
    existing = _find_skill(name)
    if existing:
        return {
            "success": False,
            "error": f"A skill named '{name}' already exists at {existing['path']}.",
        }

    # Canonical normalization -- the global mutex key is derived ONLY from
    # this value, never from category/root/original spelling/caller identity.
    # The shared live_skill_publish_guard accepts only canonical L1 names
    # so we canonicalize here and let the guard raise ValueError on a
    # non-canonical input (rare race with concurrent validation).
    normalized_name = _canonical_normalize_skill_name(name)
    if normalized_name is None:
        return {"success": False, "error": f"Invalid skill name '{name}'."}

    skill_dir = _resolve_skill_dir(name, category).resolve(strict=False)
    skills_root = _skills_dir().resolve(strict=False)
    skill_md = skill_dir / "SKILL.md"
    candidate_bytes = content.encode("utf-8")
    canonical_skill_dir = skill_dir  # alias for symmetry with edit/patch

    # Phase C -- Block 2: route the global normalized-name lock and the
    # per-target mutation lock through the SHARED
    # ``tools.skill_publish_guard.live_skill_publish_guard`` so that
    # all six publishers (P1 _create_skill, P2 restore_skill, P3
    # install_from_quarantine, P4 restore_official_optional_skill,
    # P5 reset_bundled_skill, P6 sync_skills) collide on the same
    # global lock path for the same normalized name.  Policy is
    # ``new_only``; the existing duplicate refusal / staged publish
    # body is preserved verbatim inside the guard's body region.
    import tools.skill_publish_guard as _spg  # local import to keep
    # top-of-module surface stable for any monkey-patching tests that
    # patch the module-level references.

    staging: Path | None = None
    _primary: Optional[Dict[str, Any]] = None
    _live_committed: bool = False
    _lock_acquire_exc: Optional[_spg.SkillMutationLockAcquireFailure] = None
    _lock_release_exc: Optional[
        "_SkillMutationLockReleaseFailure | _spg.SkillMutationLockReleaseFailure"
    ] = None
    # Compute the global-name lock path eagerly so the failure
    # discriminator below can tag the failing scope correctly.
    _global_name_lock_path = _spg.normalized_name_lock_target(
        normalized_name, anchor=canonical_skill_dir
    )
    try:
        with _spg.live_skill_publish_guard(
            normalized_name,
            target=canonical_skill_dir,
            replacement_policy="new_only",
        ):
            # The shared guard has already performed scan_1 (under
            # the global lock) and scan_2 (under both locks).  The
            # DECISIVE existence check below is a defensive second
            # pass using _find_skill's exact-match path semantics;
            # it cannot resurrect a phantom from a category miss
            # because the shared guard has already refused anything
            # that would surface here.
            existing_inside = _find_skill(name)
            if existing_inside:
                _primary = {
                    "success": False,
                    "error": f"A skill named '{name}' already exists at {existing_inside['path']}.",
                    "policy_reason": "duplicate_skill",
                    "rollback_failure_kind": "concurrent_modification",
                }
            else:
                # Pre-flight guards (cheap, before staging).
                if os.path.lexists(skill_dir):
                    _primary = {
                        "success": False,
                        "error": f"A skill named '{name}' already exists at {skill_dir}.",
                    }
                else:
                    guard = _background_review_self_improvement_guard(name, skill_dir, "create")
                    if guard:
                        _primary = guard
                    else:
                        denial = _final_skill_mutation_denial("create", skill_dir, origin="skill_manager_create")
                        if denial is not None:
                            _primary = denial
                        else:
                            # Build a private staging directory OUTSIDE any skills root.
                            staging = _create_private_staging(skills_root)
                            staged_skill_dir = staging / skill_dir.name
                            try:
                                staged_skill_dir.mkdir(exist_ok=False)
                            except FileExistsError:
                                _primary = {
                                    "success": False,
                                    "error": "staging mkdir collision; retry",
                                    "policy_reason": "staging_failure",
                                }
                            else:
                                staged_skill_md = staged_skill_dir / "SKILL.md"

                                # Write the candidate into the staging SKILL.md (NOT live).
                                token = _with_skill_operation("create")
                                try:
                                    with open(staged_skill_md, "wb") as f:
                                        f.write(candidate_bytes)
                                finally:
                                    _reset_skill_operation(token)

                                # Verify staging identity (regular file, exact candidate bytes).
                                try:
                                    staged_skill_md_identity = _lstat_identity(staged_skill_md)
                                    staged_skill_dir_identity = _lstat_identity(staged_skill_dir)
                                    _ensure_regular_identity(staged_skill_md, staged_skill_md_identity)
                                except OSError as exc:
                                    _primary = _rollback_failed_payload(
                                        target=skill_md,
                                        scan_error="staging lstat failed",
                                        rollback_error=str(exc),
                                        rollback_failure_kind="staging_failure",
                                    )
                                else:
                                    if staged_skill_md.read_bytes() != candidate_bytes:
                                        _primary = _rollback_failed_payload(
                                            target=skill_md,
                                            scan_error="staging byte mismatch before scan",
                                            rollback_error="staged candidate bytes did not match",
                                            rollback_failure_kind="staging_failure",
                                        )
                                    else:
                                        # Security scan of the staged (still-private) tree.
                                        scan_error = _security_scan_skill_fail_closed(staged_skill_dir)
                                        if scan_error:
                                            _primary = {
                                                "success": False,
                                                "error": scan_error,
                                                "policy_reason": "scan_failure",
                                                "rollback_failure_kind": "scan_failure",
                                                "target": str(skill_md),
                                            }
                                        elif os.path.lexists(skill_dir):
                                            _primary = {
                                                "success": False,
                                                "error": f"A skill named '{name}' already exists at {skill_dir}.",
                                                "policy_reason": "concurrent_modification",
                                                "rollback_failure_kind": "concurrent_modification",
                                            }
                                        else:
                                            # Build the live parent chain under the skills root.
                                            denial = _final_skill_mutation_denial(
                                                "create", skill_dir, origin="skill_manager_create_live_parent"
                                            )
                                            if denial is not None:
                                                _primary = denial
                                                raise _SkillMutationAborted
                                            created_parent_identities: list[tuple[Path, tuple[int, int, int]]] = []
                                            if not os.path.lexists(skills_root):
                                                skills_root.mkdir(parents=True, exist_ok=True)
                                            parent_chain: list[Path] = []
                                            parent = skill_dir.parent
                                            while parent != skills_root and skills_root in parent.resolve(strict=False).parents:
                                                parent_chain.append(parent)
                                                parent = parent.parent
                                            for directory in reversed(parent_chain):
                                                if not os.path.lexists(directory):
                                                    directory.mkdir(exist_ok=False)
                                                    created_parent_identities.append((directory, _lstat_identity(directory)))

                                            # Publish the live skill directory.
                                            denial = _final_skill_mutation_denial(
                                                "create", skill_dir, origin="skill_manager_create_live_dir"
                                            )
                                            if denial is not None:
                                                try:
                                                    for directory, identity in reversed(created_parent_identities):
                                                        _ensure_directory_identity(directory, identity)
                                                        directory.rmdir()
                                                except OSError:
                                                    logger.debug("cleanup of own parent chain failed", exc_info=True)
                                                _primary = denial
                                                raise _SkillMutationAborted
                                            try:
                                                skill_dir.mkdir(exist_ok=False)
                                            except FileExistsError:
                                                try:
                                                    for directory, identity in reversed(created_parent_identities):
                                                        _ensure_directory_identity(directory, identity)
                                                        directory.rmdir()
                                                except OSError:
                                                    logger.debug("cleanup of own parent chain failed", exc_info=True)
                                                _primary = {
                                                    "success": False,
                                                    "error": f"A skill named '{name}' already exists at {skill_dir}.",
                                                    "policy_reason": "concurrent_modification",
                                                    "rollback_failure_kind": "concurrent_modification",
                                                }
                                            else:
                                                skill_dir_identity = _lstat_identity(skill_dir)

                                                # No-clobber publish of the candidate SKILL.md into the live
                                                # directory.  Use O_CREAT | O_EXCL | O_NOFOLLOW so a symlink
                                                # appearing at the live target fails the publish instead of
                                                # following it.
                                                denial = _final_skill_mutation_denial(
                                                    "create", skill_md, origin="skill_manager_create_publish"
                                                )
                                                if denial is not None:
                                                    _publish_failure_cleanup(
                                                        skill_dir=skill_dir,
                                                        skill_dir_identity=skill_dir_identity,
                                                        created_parent_identities=created_parent_identities,
                                                        skills_root=skills_root,
                                                    )
                                                    _primary = denial
                                                    raise _SkillMutationAborted
                                                try:
                                                    publish_fd = os.open(
                                                        str(skill_md),
                                                        os.O_CREAT | os.O_EXCL | _O_NOFOLLOW | os.O_WRONLY,
                                                        0o644,
                                                    )
                                                except FileExistsError:
                                                    _publish_failure_cleanup(
                                                        skill_dir=skill_dir,
                                                        skill_dir_identity=skill_dir_identity,
                                                        created_parent_identities=created_parent_identities,
                                                        skills_root=skills_root,
                                                    )
                                                    _primary = {
                                                        "success": False,
                                                        "error": f"A skill named '{name}' already exists at {skill_md}.",
                                                        "policy_reason": "concurrent_modification",
                                                        "rollback_failure_kind": "concurrent_modification",
                                                    }
                                                except OSError as exc:
                                                    _publish_failure_cleanup(
                                                        skill_dir=skill_dir,
                                                        skill_dir_identity=skill_dir_identity,
                                                        created_parent_identities=created_parent_identities,
                                                        skills_root=skills_root,
                                                    )
                                                    _primary = {
                                                        "success": False,
                                                        "error": f"could not publish SKILL.md: {exc}",
                                                        "policy_reason": "publish_failure",
                                                        "rollback_failure_kind": "physical_failure",
                                                    }
                                                else:
                                                    try:
                                                        with os.fdopen(publish_fd, "wb") as f:
                                                            f.write(candidate_bytes)
                                                    except Exception:
                                                        _publish_failure_cleanup(
                                                            skill_dir=skill_dir,
                                                            skill_dir_identity=skill_dir_identity,
                                                            created_parent_identities=created_parent_identities,
                                                            skills_root=skills_root,
                                                        )
                                                        raise

                                                    # Verify the live identity is exactly the candidate.
                                                    live_skill_md_identity = _lstat_identity(skill_md)
                                                    _ensure_regular_identity(skill_md, live_skill_md_identity)
                                                    if skill_md.read_bytes() != candidate_bytes:
                                                        _publish_failure_cleanup(
                                                            skill_dir=skill_dir,
                                                            skill_dir_identity=skill_dir_identity,
                                                            created_parent_identities=created_parent_identities,
                                                            skills_root=skills_root,
                                                        )
                                                        _primary = _rollback_failed_payload(
                                                            target=skill_md,
                                                            scan_error="post-publish verification failed",
                                                            rollback_error="published bytes did not match candidate",
                                                            rollback_failure_kind="physical_failure",
                                                        )
                                                    else:
                                                        # Successful publish.
                                                        _live_committed = True
                                                        _primary = None  # will build success below
    except _SkillMutationAborted:
        pass
    except _spg.SkillMutationLockAcquireFailure as _acq_exc:
        _lock_acquire_exc = _acq_exc
    except _spg.SkillMutationLockReleaseFailure as _rel_exc:
        _lock_release_exc = _rel_exc
    finally:
        cleanup_failure = _cleanup_private_staging(staging)

    if _lock_acquire_exc is not None:
        payload = _format_lock_acquisition_failure_payload(
            _lock_acquire_exc, operation_kind="create", target=skill_md,
        )
        # The shared guard carries the failing scope on the structured
        # exception itself via ``active_lock_scope`` (set by the guard
        # on its LockState).  The discriminator below is preserved for
        # backward compatibility with existing payload consumers.
        if _global_name_lock_path is not None and (
            str(_lock_acquire_exc.lock_path) == str(_global_name_lock_path)
        ):
            payload["lock_scope"] = "global_normalized_name"
            payload["normalized_name"] = str(normalized_name)
        else:
            payload["lock_scope"] = "prospective_target"
        return payload

    if _lock_release_exc is not None:
        # Block 2 release-metadata focused remediation:
        #   A. ``live_mutation_committed`` MUST reflect what really happened
        #      inside the body, not the placeholder that the shared guard
        #      sets at exception construction time (always False there).
        #   B. The discriminator MUST tag the payload with
        #      ``lock_scope`` so callers can distinguish a release failure
        #      on the global normalized-name lock from one on the
        #      prospective target mutation lock.  An UNKNOWN path is
        #      preserved exactly via ``lock_path`` but MUST NOT be
        #      silently classified as either scope.
        # The order is fixed by contract:
        #   1. mutate the caught exception's live_mutation_committed;
        #   2. format the canonical payload once into a local;
        #   3. tag the scope (global_normalized_name / prospective_target)
        #      iff ``exc.lock_path`` matches the corresponding exact path;
        #   4. return the payload (no second formatter call, no cleanup
        #      combine because the create release branch never pairs with
        #      staging cleanup failure -- the shared guard's exception
        #      belongs to the cross-cutting lock layer).
        _lock_release_exc.live_mutation_committed = bool(_live_committed)
        _release_payload = _format_lock_release_failure_payload(
            _lock_release_exc, target=skill_md,
        )
        # Both paths are derived in the SAME way the shared guard derives
        # them, so the discriminator below uses byte-stable equality.
        _target_lock_path_local = _spg._target_lock_path(canonical_skill_dir)
        if str(_lock_release_exc.lock_path) == str(_global_name_lock_path):
            _release_payload["lock_scope"] = "global_normalized_name"
        elif str(_lock_release_exc.lock_path) == str(_target_lock_path_local):
            _release_payload["lock_scope"] = "prospective_target"
        # Closed-set violation: a release failure whose lock_path is
        # neither the global nor the target path is preserved exactly
        # via ``lock_path`` and the failure payload but NOT silently
        # classified as either scope.
        return _release_payload

    if _primary is None and cleanup_failure is None:
        # Successful publish AND successful cleanup.
        # Extract description from frontmatter for verbose notifications.
        _desc = ""
        try:
            _fm_end = re.search(r'\n---\s*\n', content[3:])
            if _fm_end:
                _parsed = yaml.safe_load(content[3:_fm_end.start() + 3])
                _desc = str(_parsed.get("description", ""))[:120]
        except Exception:
            pass

        result = {
            "success": True,
            "message": f"Skill '{name}' created.",
            "path": str(skill_dir.relative_to(_skills_dir())),
            "skill_md": str(skill_md),
            "_change": {"description": _desc},
        }
        if category:
            result["category"] = category
        result["hint"] = (
            "To add reference files, templates, or scripts, use "
            "skill_manage(action='write_file', name='{}', file_path='references/example.md', file_content='...')".format(name)
        )
        _add_description_prompt_preview(result, content)
        _attach_lint_findings(result, skill_md)
        return result

    if _primary is None:
        # Successful publish but cleanup failed.
        _primary = {
            "success": True,
            "message": f"Skill '{name}' created.",
            "target": str(skill_md),
        }
    return _combine_cleanup_failure(
        _primary, cleanup_failure, live_mutation_committed=_live_committed
    )


def _attach_lint_findings(result: Dict[str, Any], skill_md: Path) -> None:
    """Run the advisory SKILL.md linter and attach any findings to *result*.

    The linter enforces the CONTRIBUTING "Skill authoring standards (HARDLINE)"
    conventions that the hard validator does not (shell-utility references,
    missing metadata, dangling reference links, POSIX gating, forbidden files).
    Findings are ADVISORY — surfaced as guidance so the author can fix them,
    never a hard block. The hard rejects already ran in _validate_frontmatter.
    """
    try:
        from tools.skill_linter import lint_skill  # local import: optional path

        findings = lint_skill(skill_md)
    except Exception:
        return
    if not findings:
        return
    result["lint_warnings"] = [
        {"severity": f.severity, "rule": f.rule, "message": f.message}
        for f in findings
    ]
    result["lint_hint"] = (
        "The skill was created. These are advisory authoring-convention "
        "findings (not blockers) — fix them with skill_manage(action='patch') "
        "to match Hermes skill standards."
    )


def _edit_skill(name: str, content: str) -> Dict[str, Any]:
    """Replace the SKILL.md of any existing skill (full rewrite)."""
    err = _validate_frontmatter(content)
    if err:
        return {"success": False, "error": err}

    err = _validate_content_size(content)
    if err:
        return {"success": False, "error": err}

    existing = _find_skill(name)
    if not existing:
        return {"success": False, "error": _skill_not_found_error(name)}
    org_guard = _org_mirror_write_guard(name, existing["path"], "edit")
    if org_guard:
        return org_guard
    guard = _background_review_write_guard(name, existing["path"], "edit")
    if guard:
        return guard

    skill_md = existing["path"] / "SKILL.md"
    read_guard = _background_review_read_before_write_guard(
        name, skill_md, "edit", "SKILL.md"
    )
    if read_guard:
        return read_guard

    candidate_bytes = content.encode("utf-8")
    canonical_skill_dir = existing["path"].resolve(strict=False)
    skills_root = _containing_skills_root(canonical_skill_dir)

    # Phase C prepublish staging remediation: snapshot the live target,
    # build a private staging copy of the whole skill, mutate the staging
    # copy, scan it, and ONLY THEN atomic-replace the live SKILL.md.
    staging: Path | None = None
    _primary: Optional[Dict[str, Any]] = None
    _live_committed: bool = False
    _lock_release_exc: Optional["_SkillMutationLockReleaseFailure"] = None
    _lock_acquire_exc: Optional["_SkillMutationLockAcquireFailure"] = None
    try:
        with _skill_mutation_process_lock(canonical_skill_dir):
            with file_state.lock_path(str(canonical_skill_dir)):
                # Live snapshot.  Capture everything we need to detect a
                # concurrent modification between snapshot and publish.
                original_stat = _ensure_regular_identity(skill_md)
                live_skill_dir_identity = _lstat_identity(canonical_skill_dir)
                parent_identity = _lstat_identity(skill_md.parent)
                original_bytes = skill_md.read_bytes()
                original_mode = original_stat.st_mode
                original_skill_md_identity = (original_stat.st_dev, original_stat.st_ino, stat.S_IFMT(original_stat.st_mode))

                # Guards again under the lock — a policy/permission state
                # may have changed between the early and locked check.
                guard = _background_review_write_guard(name, existing["path"], "edit")
                if guard:
                    _primary = guard
                else:
                    denial = _final_skill_mutation_denial("edit", skill_md, origin="skill_manager_edit")
                    if denial is not None:
                        _primary = denial
                    else:
                        # Build a private staging dir + a full copy of the live
                        # skill so the scanner sees the SKILL.md and any supporting
                        # files in their final form.
                        staging = _create_private_staging(skills_root)
                        staged_skill_dir = staging / canonical_skill_dir.name
                        staged_skill_md = staged_skill_dir / "SKILL.md"
                        _copy_skill_into_staging(canonical_skill_dir, staged_skill_dir)

                        # Apply the edit ONLY in staging.
                        token = _with_skill_operation("edit")
                        try:
                            with open(staged_skill_md, "wb") as f:
                                f.write(candidate_bytes)
                        finally:
                            _reset_skill_operation(token)

                        # Verify staging byte identity.
                        staged_md_stat = _ensure_regular_identity(staged_skill_md)
                        if staged_skill_md.read_bytes() != candidate_bytes:
                            _primary = {
                                "success": False,
                                "error": "staging byte mismatch before scan",
                                "policy_reason": "staging_failure",
                                "rollback_failure_kind": "staging_failure",
                                "target": str(skill_md),
                            }
                        else:
                            # Security scan of the staged tree.
                            scan_error = _security_scan_skill_fail_closed(staged_skill_dir)
                            if scan_error:
                                _primary = {
                                    "success": False,
                                    "error": scan_error,
                                    "policy_reason": "rollback_failed",
                                    "rollback_failure_kind": "scan_failure",
                                    "scan_error": scan_error,
                                    "target": str(skill_md),
                                }
                            else:
                                # Re-validate the live target against the snapshot.
                                current_stat = _lstat_identity(skill_md)
                                current_parent_identity = _lstat_identity(skill_md.parent)
                                if (
                                    current_stat != original_skill_md_identity
                                    or current_parent_identity != parent_identity
                                    or skill_md.read_bytes() != original_bytes
                                    or stat.S_IMODE(Path(skill_md).stat().st_mode)
                                    != stat.S_IMODE(original_mode)
                                ):
                                    _primary = {
                                        "success": False,
                                        "error": (
                                            "live target changed during scan; concurrent "
                                            "modification preserved"
                                        ),
                                        "policy_reason": "rollback_failed",
                                        "rollback_failure_kind": "concurrent_modification",
                                        "target": str(skill_md),
                                    }
                                elif _lstat_identity(canonical_skill_dir) != live_skill_dir_identity:
                                    _primary = {
                                        "success": False,
                                        "error": "skill directory identity changed during scan",
                                        "policy_reason": "rollback_failed",
                                        "rollback_failure_kind": "concurrent_modification",
                                        "target": str(canonical_skill_dir),
                                    }
                                else:
                                    # Publish: temp + atomic_replace into the live parent.
                                    denial = _final_skill_mutation_denial(
                                        "edit", skill_md, origin="skill_manager_edit_publish_temp"
                                    )
                                    if denial is not None:
                                        _primary = denial
                                        raise _SkillMutationAborted
                                    fd, temp_path = tempfile.mkstemp(
                                        dir=str(skill_md.parent),
                                        prefix=f".{skill_md.name}.publish.",
                                        suffix="",
                                    )
                                    try:
                                        with os.fdopen(fd, "wb") as f:
                                            f.write(candidate_bytes)
                                        os.chmod(temp_path, stat.S_IMODE(original_mode))
                                        denial = _final_skill_mutation_denial(
                                            "edit", skill_md, origin="skill_manager_edit_atomic_replace"
                                        )
                                        if denial is not None:
                                            _primary = denial
                                            try:
                                                os.unlink(temp_path)
                                            except OSError:
                                                logger.error("failed to remove denied publish temp %s", temp_path, exc_info=True)
                                            raise _SkillMutationAborted
                                        atomic_replace(temp_path, skill_md)
                                    except _SkillMutationAborted:
                                        raise
                                    except Exception:
                                        try:
                                            os.unlink(temp_path)
                                        except OSError:
                                            logger.error("failed to remove publish temp %s", temp_path, exc_info=True)
                                        _primary = {
                                            "success": False,
                                            "error": "publish failed",
                                            "policy_reason": "rollback_failed",
                                            "rollback_failure_kind": "physical_failure",
                                            "target": str(skill_md),
                                        }
                                    else:
                                        published_stat = _ensure_regular_identity(skill_md)
                                        if skill_md.read_bytes() != candidate_bytes:
                                            _primary = _rollback_failed_payload(
                                                target=skill_md,
                                                scan_error="post-publish verification failed",
                                                rollback_error="published bytes did not match candidate",
                                                rollback_failure_kind="physical_failure",
                                            )
                                        elif stat.S_IMODE(published_stat.st_mode) != stat.S_IMODE(original_mode):
                                            _primary = _rollback_failed_payload(
                                                target=skill_md,
                                                scan_error="post-publish mode verification failed",
                                                rollback_error="published mode did not match original",
                                                rollback_failure_kind="physical_failure",
                                            )
                                        else:
                                            _live_committed = True
                                            _primary = None
    except _SkillMutationAborted:
        pass
    except _SkillMutationLockAcquireFailure:
        import sys as _sys
        _exc_info = _sys.exc_info()[1]
        _lock_acquire_exc = (
            _exc_info if isinstance(_exc_info, _SkillMutationLockAcquireFailure) else None
        )
    except _SkillMutationLockReleaseFailure:
        import sys as _sys
        _lock_release_exc = _sys.exc_info()[1]
    finally:
        cleanup_failure = _cleanup_private_staging(staging)

    if _lock_acquire_exc is not None:
        return _format_lock_acquisition_failure_payload(
            _lock_acquire_exc, operation_kind="edit", target=skill_md,
        )

    if _lock_release_exc is not None:
        _lock_release_exc.live_mutation_committed = bool(_live_committed)
        payload = _format_lock_release_failure_payload(_lock_release_exc, target=skill_md)
        return _combine_lock_release_with_cleanup(payload, cleanup_failure)

    if _primary is None and cleanup_failure is None:
        _desc = ""
        try:
            _fm_end = re.search(r'\n---\s*\n', content[3:])
            if _fm_end:
                _parsed = yaml.safe_load(content[3:_fm_end.start() + 3])
                _desc = str(_parsed.get("description", ""))[:120]
        except Exception:
            pass

        result = {
            "success": True,
            "message": f"Skill '{name}' updated (full rewrite).",
            "path": str(existing["path"]),
            "_change": {"description": _desc},
        }
        org_note = _maybe_auto_propose_org_edit(name, existing["path"])
        if org_note:
            result["org_sharing"] = org_note
            result["message"] = f"{result['message']} {org_note}"
        _add_description_prompt_preview(result, content)
        return result

    if _primary is None:
        _primary = {
            "success": True,
            "message": f"Skill '{name}' updated (full rewrite).",
            "path": str(existing["path"]),
        }
    return _combine_cleanup_failure(
        _primary, cleanup_failure, live_mutation_committed=_live_committed
    )


def _patch_skill(
    name: str,
    old_string: str,
    new_string: str,
    file_path: str = None,
    replace_all: bool = False,
) -> Dict[str, Any]:
    """Targeted find-and-replace within a skill file.

    Defaults to SKILL.md. Use file_path to patch a supporting file instead.
    Requires a unique match unless replace_all is True.

    Phase C prepublish staging remediation: snapshot the live target,
    build a private staging copy of the skill, run fuzzy match against
    the STAGING bytes, scan the staged tree, and only then atomic-replace
    the live target.  We never re-read the live target after the staging
    copy is built — the published bytes are exactly what was scanned.
    """
    if not old_string:
        return {"success": False, "error": "old_string is required for 'patch'."}
    if new_string is None:
        return {"success": False, "error": "new_string is required for 'patch'. Use an empty string to delete matched text."}

    existing = _find_skill(name)
    if not existing:
        return {"success": False, "error": _skill_not_found_error(name)}

    skill_dir = existing["path"]
    org_guard = _org_mirror_write_guard(name, skill_dir, "patch")
    if org_guard:
        return org_guard
    guard = _background_review_write_guard(name, skill_dir, "patch")
    if guard:
        return guard

    if file_path:
        # Patching a supporting file
        err = _validate_file_path(file_path)
        if err:
            return {"success": False, "error": err}
        target, err = _resolve_skill_target(skill_dir, file_path)
        if err:
            return {"success": False, "error": err}
        assert target is not None
    else:
        # Patching SKILL.md
        target = skill_dir / "SKILL.md"

    if not target.exists():
        return {"success": False, "error": f"File not found: {target.relative_to(skill_dir)}"}

    read_guard = _background_review_read_before_write_guard(
        name,
        target,
        "patch",
        "SKILL.md" if not file_path else file_path,
    )
    if read_guard:
        return read_guard

    canonical_skill_dir = skill_dir.resolve(strict=False)
    skills_root = _containing_skills_root(canonical_skill_dir)

    staging: Path | None = None
    _primary: Optional[Dict[str, Any]] = None
    _live_committed: bool = False
    _lock_release_exc: Optional["_SkillMutationLockReleaseFailure"] = None
    _lock_acquire_exc: Optional["_SkillMutationLockAcquireFailure"] = None
    try:
        with _skill_mutation_process_lock(canonical_skill_dir):
            with file_state.lock_path(str(canonical_skill_dir)):
                # Live snapshot — captured BEFORE staging is built so any
                # concurrent modification during the scan is detectable.
                original_stat = _ensure_regular_identity(target)
                live_skill_dir_identity = _lstat_identity(canonical_skill_dir)
                parent_identity = _lstat_identity(target.parent)
                original_bytes = target.read_bytes()
                original_mode = original_stat.st_mode
                original_target_identity = (
                    original_stat.st_dev,
                    original_stat.st_ino,
                    stat.S_IFMT(original_stat.st_mode),
                )

                # Locked guards.
                guard = _background_review_write_guard(name, skill_dir, "patch")
                if guard:
                    _primary = guard
                else:
                    denial = _final_skill_mutation_denial("patch", target, origin="skill_manager_patch")
                    if denial is not None:
                        _primary = denial
                    else:
                        # Build a private staging copy of the whole live skill.
                        staging = _create_private_staging(skills_root)
                        staged_skill_dir = staging / canonical_skill_dir.name
                        _copy_skill_into_staging(canonical_skill_dir, staged_skill_dir)
                        staged_target = staged_skill_dir / target.relative_to(canonical_skill_dir)

                        # Fuzzy match against the STAGING bytes (NOT live).
                        from tools.fuzzy_match import fuzzy_find_and_replace

                        staged_content = staged_target.read_bytes().decode("utf-8")
                        new_content, match_count, _strategy, match_error = fuzzy_find_and_replace(
                            staged_content, old_string, new_string, replace_all
                        )
                        if match_error:
                            preview = staged_content[:500] + ("..." if len(staged_content) > 500 else "")
                            err_msg = match_error
                            try:
                                from tools.fuzzy_match import format_no_match_hint
                                err_msg += format_no_match_hint(match_error, match_count, old_string, staged_content)
                            except Exception:
                                pass
                            _primary = {
                                "success": False,
                                "error": err_msg,
                                "file_preview": preview,
                            }
                        else:
                            target_label = "SKILL.md" if not file_path else file_path
                            err = _validate_content_size(new_content, label=target_label)
                            if err:
                                _primary = {"success": False, "error": err}
                            elif not file_path:
                                err = _validate_frontmatter(new_content)
                                if err:
                                    _primary = {
                                        "success": False,
                                        "error": f"Patch would break SKILL.md structure: {err}",
                                    }
                                else:
                                    _primary = _apply_and_publish_patch(
                                        target, staged_target, staged_skill_dir,
                                        new_content, original_bytes, original_mode,
                                        original_target_identity, parent_identity,
                                        live_skill_dir_identity, canonical_skill_dir,
                                        skill_dir,
                                    )
                                    if _primary is None:
                                        _live_committed = True
                            else:
                                _primary = _apply_and_publish_patch(
                                    target, staged_target, staged_skill_dir,
                                    new_content, original_bytes, original_mode,
                                    original_target_identity, parent_identity,
                                    live_skill_dir_identity, canonical_skill_dir,
                                    skill_dir,
                                )
                                if _primary is None:
                                    _live_committed = True
    except _SkillMutationLockAcquireFailure:
        import sys as _sys
        _exc_info = _sys.exc_info()[1]
        _lock_acquire_exc = (
            _exc_info if isinstance(_exc_info, _SkillMutationLockAcquireFailure) else None
        )
    except _SkillMutationLockReleaseFailure:
        import sys as _sys
        _lock_release_exc = _sys.exc_info()[1]
    finally:
        cleanup_failure = _cleanup_private_staging(staging)

    if _lock_acquire_exc is not None:
        return _format_lock_acquisition_failure_payload(
            _lock_acquire_exc, operation_kind="patch", target=target,
        )

    if _lock_release_exc is not None:
        _lock_release_exc.live_mutation_committed = bool(_live_committed)
        payload = _format_lock_release_failure_payload(_lock_release_exc, target=target)
        return _combine_lock_release_with_cleanup(payload, cleanup_failure)

    if _primary is None and cleanup_failure is None:
        result = {
            "success": True,
            "message": f"Patched {'SKILL.md' if not file_path else file_path} in skill '{name}' ({match_count} replacement{'s' if match_count > 1 else ''}).",
            "_change": {
                "old": old_string[:200] + ("…" if len(old_string) > 200 else ""),
                "new": new_string[:200] + ("…" if len(new_string) > 200 else ""),
            },
        }
        org_note = _maybe_auto_propose_org_edit(name, skill_dir)
        if org_note:
            result["org_sharing"] = org_note
            result["message"] = f"{result['message']} {org_note}"
        return result

    if _primary is None:
        _primary = {
            "success": True,
            "message": f"Patched {'SKILL.md' if not file_path else file_path} in skill '{name}' ({match_count} replacement{'s' if match_count > 1 else ''}).",
        }
    return _combine_cleanup_failure(
        _primary, cleanup_failure, live_mutation_committed=_live_committed
    )


def _apply_and_publish_patch(
    target: Path,
    staged_target: Path,
    staged_skill_dir: Path,
    new_content: str,
    original_bytes: bytes,
    original_mode: int,
    original_target_identity: Tuple[int, int, int],
    parent_identity: Tuple[int, int, int],
    live_skill_dir_identity: Tuple[int, int, int],
    canonical_skill_dir: Path,
    skill_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Apply the patch in staging, scan, and atomic-replace the live target.

    Returns ``None`` on success, or a structured failure dict when the
    helper detected a problem (staging mismatch, scanner denial,
    concurrent modification, publish failure, post-publish verification
    failure).  The result is owned by THIS invocation only — no shared
    state, no module globals, no function-attribute side channel — so
    concurrent callers never see each other's outcomes.
    """
    candidate_bytes = new_content.encode("utf-8")
    token = _with_skill_operation("patch")
    try:
        with open(staged_target, "wb") as f:
            f.write(candidate_bytes)
    finally:
        _reset_skill_operation(token)
    if staged_target.read_bytes() != candidate_bytes:
        return {
            "success": False,
            "error": "staging byte mismatch before scan",
            "policy_reason": "staging_failure",
            "rollback_failure_kind": "staging_failure",
            "target": str(target),
        }
    scan_error = _security_scan_skill_fail_closed(staged_skill_dir)
    if scan_error:
        return {
            "success": False,
            "error": scan_error,
            "policy_reason": "rollback_failed",
            "rollback_failure_kind": "scan_failure",
            "scan_error": scan_error,
            "target": str(target),
        }
    current_stat = _lstat_identity(target)
    current_parent_identity = _lstat_identity(target.parent)
    if (
        current_stat != original_target_identity
        or current_parent_identity != parent_identity
        or target.read_bytes() != original_bytes
        or stat.S_IMODE(Path(target).stat().st_mode)
        != stat.S_IMODE(original_mode)
    ):
        return {
            "success": False,
            "error": "live target changed during scan; concurrent modification preserved",
            "policy_reason": "rollback_failed",
            "rollback_failure_kind": "concurrent_modification",
            "target": str(target),
        }
    if _lstat_identity(canonical_skill_dir) != live_skill_dir_identity:
        return {
            "success": False,
            "error": "skill directory identity changed during scan",
            "policy_reason": "rollback_failed",
            "rollback_failure_kind": "concurrent_modification",
            "target": str(canonical_skill_dir),
        }
    denial = _final_skill_mutation_denial(
        "patch", target, origin="skill_manager_patch_publish_temp"
    )
    if denial is not None:
        return denial
    fd, temp_path = tempfile.mkstemp(
        dir=str(target.parent),
        prefix=f".{target.name}.publish.",
        suffix="",
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(candidate_bytes)
        os.chmod(temp_path, stat.S_IMODE(original_mode))
        denial = _final_skill_mutation_denial(
            "patch", target, origin="skill_manager_patch_atomic_replace"
        )
        if denial is not None:
            try:
                os.unlink(temp_path)
            except OSError:
                logger.error("failed to remove denied publish temp %s", temp_path, exc_info=True)
            return denial
        atomic_replace(temp_path, target)
    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            logger.error("failed to remove publish temp %s", temp_path, exc_info=True)
        return {
            "success": False,
            "error": "publish failed",
            "policy_reason": "rollback_failed",
            "rollback_failure_kind": "physical_failure",
            "target": str(target),
        }
    published_stat = _ensure_regular_identity(target)
    if target.read_bytes() != candidate_bytes:
        return _rollback_failed_payload(
            target=target,
            scan_error="post-publish verification failed",
            rollback_error="published bytes did not match candidate",
            rollback_failure_kind="physical_failure",
        )
    if stat.S_IMODE(published_stat.st_mode) != stat.S_IMODE(original_mode):
        return _rollback_failed_payload(
            target=target,
            scan_error="post-publish mode verification failed",
            rollback_error="published mode did not match original",
            rollback_failure_kind="physical_failure",
        )
    # Successful publish.
    return None


def _delete_skill(name: str, absorbed_into: Optional[str] = None) -> Dict[str, Any]:
    """Delete a skill.

    ``absorbed_into`` declares intent:
      - ``None`` / missing  → caller didn't declare (legacy / non-curator path);
        accepted for backward compat but logs a warning because the curator
        classification pipeline can't tell consolidation from pruning without it.
      - ``""`` (empty)      → explicit "truly pruned, no forwarding target".
      - ``"<skill-name>"``  → content was absorbed into that umbrella; the
        target must exist on disk. Validated here so the model can't claim an
        umbrella that doesn't exist.
    """
    existing = _find_skill(name)
    if not existing:
        return {"success": False, "error": _skill_not_found_error(name)}
    org_guard = _org_mirror_write_guard(name, existing["path"], "delete")
    if org_guard:
        return org_guard
    guard = _background_review_write_guard(name, existing["path"], "delete")
    if guard:
        return guard

    # Fail closed on unverified deletes during the curator consolidation pass.
    # A bare prune (no absorbed_into) from the LLM umbrella pass is the
    # fail-open behavior reported in #29912 — refuse it; keep the skill active.
    fail_closed = _curator_consolidation_delete_guard(name, absorbed_into)
    if fail_closed:
        return fail_closed

    pinned_err = _pinned_guard(name)
    if pinned_err:
        return {"success": False, "error": pinned_err}

    # Validate absorbed_into target when declared non-empty
    absorbed_target = (
        absorbed_into.strip()
        if absorbed_into is not None and isinstance(absorbed_into, str)
        else ""
    )
    is_consolidation = bool(absorbed_target)
    if is_consolidation:
        target_name = absorbed_target
        if target_name == name:
            return {
                "success": False,
                "error": f"absorbed_into='{target_name}' cannot equal the skill being deleted.",
            }
        target = _find_skill(target_name)
        if not target:
            return {
                "success": False,
                "error": (
                    f"absorbed_into='{target_name}' does not exist. "
                    f"Create or patch the umbrella skill first, then retry the delete."
                ),
            }

    skill_dir = existing["path"]
    skills_root = _containing_skills_root(skill_dir)
    canonical_skill_dir = skill_dir.resolve(strict=False)

    # Per-invocation local refusal flag.  Two concurrent delete calls must
    # not share a single module-level flag (would race the release-failure
    # handler into reading the wrong invocation's state).  Locality keeps
    # ``live_mutation_committed`` derived strictly from THIS call's state.
    _delete_refused = False

    # Defense-in-depth before the recursive delete (port of Kilo Code #11240).
    unsafe = _validate_delete_target(skill_dir)
    if unsafe:
        return {"success": False, "error": unsafe}

    # During the curator consolidation pass, a verified consolidation must be
    # RECOVERABLE: archival into ~/.hermes/skills/.archive/ is documented as
    # the maximum destructive action the curator may take, and
    # `hermes curator restore` promises the skill can be brought back. Route
    # through the recoverable archive primitive instead of permanent rmtree so
    # a misjudged consolidation can be undone (#29912). Foreground,
    # user-directed deletes keep their existing hard-delete semantics.
    try:
        from tools.skill_provenance import is_background_review
        curator_pass = is_background_review()
    except Exception:
        curator_pass = False

    if curator_pass:
        # Per-invocation local flag (NOT module global) so the lock-
        # release-failure handler below cannot race with a concurrent
        # invocation into reading another call's state.  Defined BEFORE
        # the ``with`` blocks so the except clause sees a defined name
        # even when lock acquisition itself raises a release failure.
        archive_refused = False
        # Phase C P1-C boundary: track whether the lock context's
        # ``__enter__`` has completed successfully.  A raw
        # ``PermissionError`` raised AFTER this flag is True is a body
        # failure (NOT acquisition), and MUST NOT be coerced as an
        # acquisition failure; only a raw ``PermissionError`` raised
        # by ``__enter__`` (i.e. when the flag is still False) is
        # legitimate input to the canonical acquisition payload.
        _lock_context_entered = False
        # Acquire the same interprocess lock key that create/edit/patch/
        # write_file/remove_file use on this skill so two concurrent
        # curators cannot archive and rmtree in parallel.
        try:
            with _skill_mutation_process_lock(canonical_skill_dir):
                _lock_context_entered = True
                with file_state.lock_path(str(canonical_skill_dir)):
                    # Phase C curator-archive identity + atomicity contract:
                    # ``archive_skill`` (in tools/skill_usage.py) resolves
                    # the skill directory by name via rglob and the move is
                    # a kernel-level ``rename``/``shutil.move`` that re-
                    # resolves the source by name.  Neither the live mutex
                    # nor any pre-check prevents a non-cooperative actor
                    # from swapping the validated tree at the pathname
                    # between the last identity capture and the archive
                    # syscall.  The only contract-true remedy available
                    # within scope is to refuse the archive BEFORE any
                    # destructive primitive runs; we capture the structured
                    # payload here, set the per-invocation refusal flag,
                    # and let the lock-release-failure handler know that
                    # ``live_mutation_committed`` MUST stay false.
                    try:
                        from tools.skill_usage import archive_skill
                    except Exception:
                        archive_skill = None  # type: ignore[assignment]
                    if archive_skill is None:
                        archive_refused = True
                        refusal_lock_parent = _resolve_lock_parent(
                            canonical_skill_dir
                        )
                        refusal_lock_path = (
                            refusal_lock_parent
                            / f".hermes-skill-mutex-{_hashlib.sha256(str(canonical_skill_dir).encode('utf-8')).hexdigest()[:16]}.lock"
                        )
                        return {
                            "success": False,
                            "error": (
                                f"Refusing to archive '{canonical_skill_dir}': "
                                "archive_skill import is unavailable."
                            ),
                            "policy_reason": "atomic_archive_unavailable",
                            "rollback_failure_kind": "identity_bound_archive_unavailable",
                            "live_mutation_committed": False,
                            "safe_to_retry": False,
                            "operation_kind": "archive",
                            "target": str(canonical_skill_dir),
                            "lock_path": str(refusal_lock_path),
                        }
                    # Re-capture the skill directory under both locks: a
                    # concurrent rename / replacement between the early
                    # ``_find_skill`` (line 2961) and lock acquisition is
                    # detected here, BEFORE any archive primitive.  The
                    # pre-lock find is treated as a hint for canonical-
                    # ization, not as authority.
                    re_existing = _find_skill(name)
                    if not re_existing:
                        archive_refused = True
                        refusal_lock_parent = _resolve_lock_parent(
                            canonical_skill_dir
                        )
                        refusal_lock_path = (
                            refusal_lock_parent
                            / f".hermes-skill-mutex-{_hashlib.sha256(str(canonical_skill_dir).encode('utf-8')).hexdigest()[:16]}.lock"
                        )
                        return {
                            "success": False,
                            "error": (
                                f"Skill '{name}' disappeared between pre-lock "
                                "and lock acquisition; refusing to archive."
                            ),
                            "policy_reason": "atomic_archive_unavailable",
                            "rollback_failure_kind": "identity_bound_archive_unavailable",
                            "live_mutation_committed": False,
                            "safe_to_retry": False,
                            "operation_kind": "archive",
                            "target": str(canonical_skill_dir),
                            "lock_path": str(refusal_lock_path),
                        }
                    re_skill_dir = re_existing["path"]
                    if re_skill_dir.resolve(strict=False) != canonical_skill_dir:
                        archive_refused = True
                        refusal_lock_parent = _resolve_lock_parent(
                            canonical_skill_dir
                        )
                        refusal_lock_path = (
                            refusal_lock_parent
                            / f".hermes-skill-mutex-{_hashlib.sha256(str(canonical_skill_dir).encode('utf-8')).hexdigest()[:16]}.lock"
                        )
                        return {
                            "success": False,
                            "error": (
                                f"Skill '{name}' was replaced between pre-lock "
                                "and lock acquisition; foreign object preserved."
                            ),
                            "policy_reason": "atomic_archive_unavailable",
                            "rollback_failure_kind": "identity_bound_archive_unavailable",
                            "live_mutation_committed": False,
                            "safe_to_retry": False,
                            "operation_kind": "archive",
                            "target": str(canonical_skill_dir),
                            "lock_path": str(refusal_lock_path),
                        }
                    # Defense-in-depth: refuse symlink / junction / non-
                    # directory sources.  The archive primitive cannot
                    # bind its syscall to a validated inode, so any
                    # ambiguous source must be refused pre-mutation.
                    try:
                        re_skill_st = re_skill_dir.lstat()
                    except OSError:
                        re_skill_st = None
                    if re_skill_st is None or stat.S_ISLNK(re_skill_st.st_mode) \
                            or _is_path_redirect(re_skill_dir) \
                            or not stat.S_ISDIR(re_skill_st.st_mode):
                        archive_refused = True
                        refusal_lock_parent = _resolve_lock_parent(
                            canonical_skill_dir
                        )
                        refusal_lock_path = (
                            refusal_lock_parent
                            / f".hermes-skill-mutex-{_hashlib.sha256(str(canonical_skill_dir).encode('utf-8')).hexdigest()[:16]}.lock"
                        )
                        return {
                            "success": False,
                            "error": (
                                f"Refusing to archive '{re_skill_dir}': the "
                                "skill directory is not a stable directory "
                                "(symlink / junction / redirect)."
                            ),
                            "policy_reason": "atomic_archive_unavailable",
                            "rollback_failure_kind": "identity_bound_archive_unavailable",
                            "live_mutation_committed": False,
                            "safe_to_retry": False,
                            "operation_kind": "archive",
                            "target": str(canonical_skill_dir),
                            "lock_path": str(refusal_lock_path),
                        }
                    pre_archive_target_identity = (
                        re_skill_st.st_dev,
                        re_skill_st.st_ino,
                        stat.S_IFMT(re_skill_st.st_mode),
                    )
                    pre_archive_parent_identity = _lstat_identity(
                        re_skill_dir.parent
                    )
                    # Final pre-archive identity check (immediately before
                    # the archive syscall).  archive_skill re-resolves the
                    # skill directory by name on every invocation; if the
                    # kernel returns a different inode at archive time,
                    # the destructive rename will follow the foreign
                    # object.  We capture the validated target/parent
                    # identity here and refuse if the path's identity
                    # drifted even one syscall away.
                    try:
                        recheck_st = re_skill_dir.lstat()
                        recheck_target_identity = (
                            recheck_st.st_dev,
                            recheck_st.st_ino,
                            stat.S_IFMT(recheck_st.st_mode),
                        )
                        recheck_parent_identity = _lstat_identity(
                            re_skill_dir.parent
                        )
                    except OSError:
                        recheck_target_identity = None
                        recheck_parent_identity = None
                    if (
                        recheck_target_identity is None
                        or recheck_parent_identity is None
                        or recheck_target_identity != pre_archive_target_identity
                        or recheck_parent_identity != pre_archive_parent_identity
                    ):
                        archive_refused = True
                        refusal_lock_parent = _resolve_lock_parent(
                            canonical_skill_dir
                        )
                        refusal_lock_path = (
                            refusal_lock_parent
                            / f".hermes-skill-mutex-{_hashlib.sha256(str(canonical_skill_dir).encode('utf-8')).hexdigest()[:16]}.lock"
                        )
                        return {
                            "success": False,
                            "error": (
                                f"skill directory '{re_skill_dir}' identity "
                                "changed between revalidation and archive; "
                                "foreign object preserved."
                            ),
                            "policy_reason": "atomic_archive_unavailable",
                            "rollback_failure_kind": "identity_bound_archive_unavailable",
                            "live_mutation_committed": False,
                            "safe_to_retry": False,
                            "operation_kind": "archive",
                            "target": str(canonical_skill_dir),
                            "lock_path": str(refusal_lock_path),
                        }
                    denial = _final_skill_mutation_denial(
                        "delete", re_skill_dir, origin="skill_manager_archive"
                    )
                    if denial is not None:
                        return denial
                    # Last-mile atomicity refusal (Phase C curator-archive
                    # block).  The portable archive primitives available
                    # (``Path.rename`` / ``shutil.move``) destroy by name,
                    # not by inode: even with both locks held and the
                    # final identity check passing, a non-cooperative
                    # actor that swaps the validated directory at the
                    # pathname AFTER the recheck wins, because the next
                    # archive call resolves by name.  The cooperative
                    # interprocess mutex only excludes writers that
                    # acquire it.
                    #
                    # Until an identity-bound kernel-anchored archive
                    # primitive is available (one where the rename is
                    # bound by the kernel to the validated inode of the
                    # validated tree and the parent of every operation
                    # stays anchored to the validated parent), refuse
                    # the curator archive and let the operator choose an
                    # explicit recovery path.  No destructive primitive
                    # runs.
                    archive_refused = True
                    refusal_lock_parent = _resolve_lock_parent(
                        canonical_skill_dir
                    )
                    refusal_lock_path = (
                        refusal_lock_parent
                        / f".hermes-skill-mutex-{_hashlib.sha256(str(canonical_skill_dir).encode('utf-8')).hexdigest()[:16]}.lock"
                    )
                    return {
                        "success": False,
                        "error": (
                            f"Refusing to archive '{re_skill_dir}': no "
                            "portable identity-bound kernel-anchored "
                            "archive primitive is available; "
                            "Path.rename / shutil.move follow the pathname "
                            "and would move any concurrent replacement. "
                            "The skill tree is preserved."
                        ),
                        "policy_reason": "atomic_archive_unavailable",
                        "rollback_failure_kind": "identity_bound_archive_unavailable",
                        "live_mutation_committed": False,
                        "safe_to_retry": False,
                        "operation_kind": "archive",
                        "target": str(re_skill_dir),
                        "lock_path": str(refusal_lock_path),
                    }
        except _SkillMutationLockAcquireFailure as exc:
            # Curator archive acquisition failure routes through the
            # canonical acquisition payload contract (Phase C P1-B).
            # ``target`` here is the skill_dir we intended to operate on
            # before any archive/syscall ran.
            return _format_lock_acquisition_failure_payload(
                exc, operation_kind="archive", target=skill_dir,
            )
        except PermissionError as exc:
            # Defensive: keep pre-existing tests that inject raw
            # PermissionError from a fake lock context-manager.
            # Phase C P1-C boundary: ONLY coerce to acquisition when the
            # lock context's ``__enter__`` has not returned.  A raw
            # ``PermissionError`` after entry is a body failure and
            # MUST be handled by the per-operation contract — typically
            # by surfacing it unchanged so the caller's existing
            # exception handling sees a real OS-level error rather than
            # a fabricated lock-acquisition payload.
            if not _lock_context_entered:
                structured = _coerce_raw_permission_error_to_acquire_failure(
                    canonical_skill_path=canonical_skill_dir, exc=exc,
                )
                return _format_lock_acquisition_failure_payload(
                    structured, operation_kind="archive", target=skill_dir,
                )
            raise
        except _SkillMutationLockReleaseFailure as exc:
            # The curator archive path now refuses BEFORE any archive
            # primitive (Phase C curator-archive block).  When the
            # refusal fires the live state has NOT been mutated; a
            # subsequent lock-release failure MUST NOT retroactively
            # report ``live_mutation_committed=True``.  We honour the
            # explicit per-invocation refusal flag set by the refusal
            # block above.
            if not archive_refused:
                # Defensive fallback: if the lock-release failure happens
                # without a recorded refusal (e.g. permission denied
                # inside the archive syscall), the destructive archive
                # MAY have already run.  We default to ``True`` to be
                # safe — operators must inspect the archive directory.
                exc.live_mutation_committed = True
            payload = _format_lock_release_failure_payload(exc, target=skill_dir)
            payload["operation_kind"] = "archive"
            if archive_refused:
                # Curator archive refused before any archive primitive
                # ran; the failure here is the lock-release failure that
                # happened AFTER the refusal, so the operation is
                # semantically unavailable rather than partially
                # committed. Restore the curator-specific policy_reason
                # so callers can distinguish "we never started" from
                # "we started but couldn't finalize". All other payload
                # fields (rollback_failure_kind, safe_to_retry, target,
                # lock_path, release/close error, live_mutation_committed
                # already coerced to False above) are preserved.
                payload["policy_reason"] = "atomic_archive_unavailable"
            return payload

    # Phase C P1-C boundary: track whether the lock context's
    # ``__enter__`` has completed successfully.  Raw ``PermissionError``
    # raised AFTER entry is a body failure, NOT an acquisition failure.
    _lock_context_entered = False
    try:
        with _skill_mutation_process_lock(canonical_skill_dir):
            _lock_context_entered = True
            with file_state.lock_path(str(canonical_skill_dir)):
                # Revalidate the skill under both locks: a concurrent
                # rename / replacement between the early ``_find_skill``
                # and the lock acquisition is detected here, BEFORE the
                # destructive rmtree.  ``_find_skill`` is treated as a
                # hint for canonicalization, not as the authority.
                re_existing = _find_skill(name)
                if not re_existing:
                    return {
                        "success": False,
                        "error": (
                            f"Skill '{name}' disappeared between pre-lock "
                            f"and lock acquisition; refusing to delete."
                        ),
                        "policy_reason": "concurrent_modification",
                        "rollback_failure_kind": "concurrent_modification",
                        "target": str(skill_dir),
                    }
                re_skill_dir = re_existing["path"]
                if re_skill_dir.resolve(strict=False) != canonical_skill_dir:
                    return {
                        "success": False,
                        "error": (
                            f"Skill '{name}' was replaced between pre-lock "
                            f"and lock acquisition; foreign directory preserved."
                        ),
                        "policy_reason": "concurrent_modification",
                        "rollback_failure_kind": "concurrent_modification",
                        "target": str(skill_dir),
                    }
                # Defense-in-depth before the recursive delete (port of Kilo Code #11240).
                unsafe = _validate_delete_target(re_skill_dir)
                if unsafe:
                    return {
                        "success": False,
                        "error": unsafe,
                        "policy_reason": "concurrent_modification",
                        "rollback_failure_kind": "concurrent_modification",
                        "target": str(re_skill_dir),
                    }
                # lstat identity: must still be a directory and not a
                # symlink / junction / redirected.  We do the checks
                # explicitly here so we can reuse ``_RollbackFailure``
                # for the abort path without changing the signature
                # of ``_ensure_directory_identity``.
                try:
                    skill_dir_st = re_skill_dir.lstat()
                except OSError as exc:
                    return {
                        "success": False,
                        "error": (
                            f"Refusing to delete '{re_skill_dir}': "
                            f"could not lstat: {exc}"
                        ),
                        "policy_reason": "concurrent_modification",
                        "rollback_failure_kind": "concurrent_modification",
                        "target": str(re_skill_dir),
                    }
                if stat.S_ISLNK(skill_dir_st.st_mode) or _is_path_redirect(re_skill_dir):
                    return {
                        "success": False,
                        "error": (
                            f"Refusing to delete '{re_skill_dir}': the skill "
                            f"directory is a symlink/junction."
                        ),
                        "policy_reason": "concurrent_modification",
                        "rollback_failure_kind": "symlink_detected",
                        "target": str(re_skill_dir),
                    }
                if not stat.S_ISDIR(skill_dir_st.st_mode):
                    return {
                        "success": False,
                        "error": (
                            f"Refusing to delete '{re_skill_dir}': not a directory."
                        ),
                        "policy_reason": "concurrent_modification",
                        "rollback_failure_kind": "concurrent_modification",
                        "target": str(re_skill_dir),
                    }
                # Capture skill_dir + parent identity immediately before
                # the destructive op.
                pre_delete_skill_dir_identity = (
                    skill_dir_st.st_dev,
                    skill_dir_st.st_ino,
                    stat.S_IFMT(skill_dir_st.st_mode),
                )
                pre_delete_parent_identity = _lstat_identity(re_skill_dir.parent)

                guard = _background_review_write_guard(name, re_skill_dir, "delete")
                if guard:
                    return guard
                denial = _final_skill_mutation_denial("delete", re_skill_dir, origin="skill_manager_delete")
                if denial is not None:
                    return denial
                # Final pre-delete identity check (immediately before rmtree).
                try:
                    recheck_st = re_skill_dir.lstat()
                    recheck_identity = (
                        recheck_st.st_dev,
                        recheck_st.st_ino,
                        stat.S_IFMT(recheck_st.st_mode),
                    )
                except OSError as exc:
                    return {
                        "success": False,
                        "error": (
                            f"could not lstat '{re_skill_dir}' immediately "
                            f"before delete: {exc}"
                        ),
                        "policy_reason": "concurrent_modification",
                        "rollback_failure_kind": "concurrent_modification",
                        "target": str(re_skill_dir),
                    }
                if recheck_identity != pre_delete_skill_dir_identity:
                    return {
                        "success": False,
                        "error": (
                            f"skill directory '{re_skill_dir}' identity "
                            f"changed between revalidation and delete; "
                            f"foreign object preserved"
                        ),
                        "policy_reason": "concurrent_modification",
                        "rollback_failure_kind": "concurrent_modification",
                        "target": str(re_skill_dir),
                    }
                # Parent identity must also match — a rename of the
                # parent (category) must abort the rmtree.
                if _lstat_identity(re_skill_dir.parent) != pre_delete_parent_identity:
                    return {
                        "success": False,
                        "error": (
                            f"parent of '{re_skill_dir}' changed between "
                            f"revalidation and delete; foreign object preserved"
                        ),
                        "policy_reason": "concurrent_modification",
                        "rollback_failure_kind": "concurrent_modification",
                        "target": str(re_skill_dir),
                    }
                # Last-mile atomicity refusal (Phase C recursive-delete
                # block; restored by R2E after the R2 regression).  Every
                # portable recursive-delete primitive available —
                # ``shutil.rmtree`` on a path, ``os.fwalk`` followed by
                # per-name unlink/rmdir, ``unlinkat`` / ``rmdir`` with
                # ``dir_fd`` — destroys the tree by name, not by inode.
                # An attacker that swaps the validated directory at the
                # pathname (or any subdirectory / file during traversal)
                # wins: the destructive call still resolves the new entry
                # by name and removes it.  The cooperative interprocess
                # mutex only excludes writers that acquire it; a
                # non-cooperative actor does not.
                #
                # A capture-then-verify quarantine rename is NOT an
                # acceptable substitute: ``os.replace`` also resolves the
                # SOURCE pathname at rename time, so it would move the
                # foreign replacement B out of its canonical pathname
                # before Hermes could discover B != A.  That violates the
                # canonical Phase2 invariants S3 (no wrong-object
                # namespace mutation) and S4 (foreign object path
                # preservation on failure), which remain CANONICAL by
                # explicit policy decision.  ``renameat2(RENAME_NOREPLACE)``
                # does not fix this either — it is not exposed by the
                # Python stdlib and still resolves the source pathname at
                # rename time.
                #
                # Until an identity-bound kernel-anchored recursive
                # delete primitive is available (one where every
                # unlink/rmdir is bound by the kernel to the validated
                # inode of the validated tree and the parent of every
                # operation stays anchored to the validated parent),
                # refuse the foreground delete and let the operator
                # choose an explicit recovery path.  No destructive
                # primitive runs.
                refusal_lock_parent = _resolve_lock_parent(canonical_skill_dir)
                refusal_lock_path = (
                    refusal_lock_parent
                    / f".hermes-skill-mutex-{_hashlib.sha256(str(canonical_skill_dir).encode('utf-8')).hexdigest()[:16]}.lock"
                )
                # Signal to the lock-release-failure handler that the
                # refusal fired BEFORE any destructive primitive, so a
                # subsequent release failure must not retroactively
                # report live_mutation_committed=true.  This is a
                # per-invocation local flag (NOT module global) so
                # concurrent deletes cannot race the handler into
                # reading another call's state.
                _delete_refused = True
                return {
                    "success": False,
                    "error": (
                        f"Refusing to recursively delete '{re_skill_dir}': "
                        "no portable identity-bound kernel-anchored "
                        "recursive-delete primitive is available; "
                        "shutil.rmtree follows the pathname and would "
                        "destroy any concurrent replacement. The skill "
                        "tree is preserved."
                    ),
                    "policy_reason": "atomic_recursive_delete_unavailable",
                    "rollback_failure_kind": "identity_bound_recursive_delete_unavailable",
                    "live_mutation_committed": False,
                    "safe_to_retry": False,
                    "operation_kind": "delete",
                    "target": str(re_skill_dir),
                    "lock_path": str(refusal_lock_path),
                }
    except _SkillMutationLockAcquireFailure as exc:
        # Foreground delete acquisition failure routes through the
        # canonical acquisition payload contract (Phase C P1-B).  The
        # delete path's previous ``policy_reason=concurrent_modification``
        # was misleading — concurrent modification is a body-time
        # condition; an acquisition failure is structural.
        return _format_lock_acquisition_failure_payload(
            exc, operation_kind="delete", target=skill_dir,
        )
    except PermissionError as exc:
        # Defensive: keep pre-existing tests that inject raw
        # PermissionError from a fake lock context-manager.
        # Phase C P1-C boundary: ONLY coerce when the lock
        # ``__enter__`` has not returned.  A raw PermissionError raised
        # after entry is a body failure and belongs to the per-op
        # release contract (or to whatever refusal the body emitted
        # before the primitive ran).  Re-raise so the agent loop
        # surfaces a real OS-level error rather than a fabricated
        # lock-acquisition payload.
        if not _lock_context_entered:
            structured = _coerce_raw_permission_error_to_acquire_failure(
                canonical_skill_path=canonical_skill_dir, exc=exc,
            )
            return _format_lock_acquisition_failure_payload(
                structured, operation_kind="delete", target=skill_dir,
            )
        raise
    except _SkillMutationLockReleaseFailure as exc:
        # The foreground delete path now refuses before any destructive
        # primitive (Phase C recursive-delete block, restored by R2E).
        # When the refusal fires the live state has NOT been mutated; a
        # subsequent lock release failure MUST NOT retroactively report
        # ``live_mutation_committed=True``.  We honour the explicit
        # refusal flag set by the refusal block above.
        if not _delete_refused:
            # No refusal fired, so a destructive delete may have run in
            # some other branch.  The caller must know the live mutation
            # committed because we cannot undo it (and the contract
            # forbids re-deleting a foreign object).  We default to
            # ``True`` to be safe (the live state may have changed).
            # Operators must inspect the disk.
            exc.live_mutation_committed = True
        return _format_lock_release_failure_payload(exc, target=skill_dir)


def _publish_write_file(
    target: Path,
    staged_target: Path,
    target_existed: bool,
    original_mode: int,
    original_bytes: Optional[bytes],
    candidate_bytes: bytes,
    created_parent_identities: list,
    existing: dict,
) -> Optional[Dict[str, Any]]:
    """Publish the staged supporting file to the live target.

    For new files, mkdir the live parent chain (only what we ourselves
    created) and then publish via O_EXCL.  For overwrites, temp +
    atomic_replace in the live parent, then re-apply the original mode
    and verify byte identity.  Returns ``None`` on success or a
    structured failure dict on any detected problem — the result is
    owned by THIS invocation only (no shared state), so concurrent
    callers cannot overwrite each other's outcomes.
    """
    if not target_existed:
        denial = _final_skill_mutation_denial(
            "write_file", target, origin="skill_manager_write_file_live_parent"
        )
        if denial is not None:
            return denial
        skill_root_resolved = existing["path"].resolve(strict=False)
        parent_chain: list[Path] = []
        parent = target.parent
        while (
            parent != skill_root_resolved
            and skill_root_resolved in parent.resolve(strict=False).parents
        ):
            parent_chain.append(parent)
            parent = parent.parent
        for directory in reversed(parent_chain):
            try:
                directory.mkdir(exist_ok=False)
                created_parent_identities.append(
                    (directory, _lstat_identity(directory))
                )
            except FileExistsError:
                return {
                    "success": False,
                    "error": f"parent directory {directory} appeared concurrently; foreign path preserved",
                    "policy_reason": "concurrent_modification",
                    "rollback_failure_kind": "concurrent_modification",
                    "target": str(target),
                }
        denial = _final_skill_mutation_denial(
            "write_file", target, origin="skill_manager_write_file_publish"
        )
        if denial is not None:
            for directory, identity in reversed(created_parent_identities):
                try:
                    _ensure_directory_identity(directory, identity)
                    if directory.exists() and not any(directory.iterdir()):
                        directory.rmdir()
                except OSError:
                    logger.debug(
                        "publish-denial cleanup of parent %s failed",
                        directory,
                        exc_info=True,
                    )
            return denial
        try:
            publish_fd = os.open(
                str(target),
                os.O_CREAT | os.O_EXCL | _O_NOFOLLOW | os.O_WRONLY,
                0o644,
            )
        except FileExistsError:
            return {
                "success": False,
                "error": f"target {target} appeared concurrently; foreign path preserved",
                "policy_reason": "concurrent_modification",
                "rollback_failure_kind": "concurrent_modification",
                "target": str(target),
            }
        try:
            with os.fdopen(publish_fd, "wb") as f:
                f.write(candidate_bytes)
        except Exception:
            try:
                os.unlink(str(target))
            except OSError:
                pass
            raise
        if target.read_bytes() != candidate_bytes:
            try:
                os.unlink(str(target))
            except OSError:
                pass
            for directory, identity in reversed(created_parent_identities):
                try:
                    _ensure_directory_identity(directory, identity)
                    if (
                        directory.exists()
                        and not any(directory.iterdir())
                    ):
                        directory.rmdir()
                except OSError:
                    logger.debug(
                        "publish-failure cleanup of parent %s failed",
                        directory,
                        exc_info=True,
                    )
            return _rollback_failed_payload(
                target=target,
                scan_error="post-publish verification failed",
                rollback_error="published bytes did not match candidate",
                rollback_failure_kind="physical_failure",
            )
    else:
        denial = _final_skill_mutation_denial(
            "write_file", target, origin="skill_manager_write_file_publish_temp"
        )
        if denial is not None:
            return denial
        fd, temp_path = tempfile.mkstemp(
            dir=str(target.parent),
            prefix=f".{target.name}.publish.",
            suffix="",
        )
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(candidate_bytes)
            os.chmod(temp_path, stat.S_IMODE(original_mode))
            denial = _final_skill_mutation_denial(
                "write_file", target, origin="skill_manager_write_file_atomic_replace"
            )
            if denial is not None:
                try:
                    os.unlink(temp_path)
                except OSError:
                    logger.error("failed to remove denied publish temp %s", temp_path, exc_info=True)
                return denial
            atomic_replace(temp_path, target)
        except Exception:
            try:
                os.unlink(temp_path)
            except OSError:
                logger.error(
                    "failed to remove publish temp %s", temp_path, exc_info=True
                )
            return {
                "success": False,
                "error": "publish failed",
                "policy_reason": "rollback_failed",
                "rollback_failure_kind": "physical_failure",
                "target": str(target),
            }
        published_stat = _ensure_regular_identity(target)
        if target.read_bytes() != candidate_bytes:
            return _rollback_failed_payload(
                target=target,
                scan_error="post-publish verification failed",
                rollback_error="published bytes did not match candidate",
                rollback_failure_kind="physical_failure",
            )
        if stat.S_IMODE(published_stat.st_mode) != stat.S_IMODE(original_mode):
            return _rollback_failed_payload(
                target=target,
                scan_error="post-publish mode verification failed",
                rollback_error="published mode did not match original",
                rollback_failure_kind="physical_failure",
            )
    # Successful publish.
    return None


def _write_file(name: str, file_path: str, file_content: str) -> Dict[str, Any]:
    """Add or overwrite a supporting file within any skill directory.

    Phase C prepublish staging remediation: snapshot the live skill, build
    a private staging copy, write the supporting file into the staging copy,
    scan the staged tree, and only then atomic-replace the live target.
    We never write to the live tree before the scan has passed; scanner
    rejection therefore cannot leave un-scanned content in the live tree.
    """
    err = _validate_file_path(file_path)
    if err:
        return {"success": False, "error": err}

    if not file_content and file_content != "":
        return {"success": False, "error": "file_content is required."}

    # Check size limits
    content_bytes = len(file_content.encode("utf-8"))
    if content_bytes > MAX_SKILL_FILE_BYTES:
        return {
            "success": False,
            "error": (
                f"File content is {content_bytes:,} bytes "
                f"(limit: {MAX_SKILL_FILE_BYTES:,} bytes / 1 MiB). "
                f"Consider splitting into smaller files."
            ),
        }
    err = _validate_content_size(file_content, label=file_path)
    if err:
        return {"success": False, "error": err}

    existing = _find_skill(name)
    if not existing:
        return {"success": False, "error": _skill_not_found_error(name, " Create it first with action='create'.")}
    org_guard = _org_mirror_write_guard(name, existing["path"], "write_file")
    if org_guard:
        return org_guard
    guard = _background_review_write_guard(name, existing["path"], "write_file")
    if guard:
        return guard

    target, err = _resolve_skill_target(existing["path"], file_path)
    if err:
        return {"success": False, "error": err}
    assert target is not None
    denial = _final_skill_mutation_denial("write_file", target, origin="skill_manager_write_file")
    if denial is not None:
        return denial
    if target.exists():
        read_guard = _background_review_read_before_write_guard(
            name, target, "write_file", file_path
        )
        if read_guard:
            return read_guard
    canonical_skill_dir = existing["path"].resolve(strict=False)
    skills_root = _containing_skills_root(canonical_skill_dir)
    candidate_bytes = file_content.encode("utf-8")

    staging: Path | None = None
    created_parent_identities: list[tuple[Path, tuple[int, int, int]]] = []
    _primary: Optional[Dict[str, Any]] = None
    _live_committed: bool = False
    _lock_release_exc: Optional["_SkillMutationLockReleaseFailure"] = None
    _lock_acquire_exc: Optional["_SkillMutationLockAcquireFailure"] = None
    try:
        with _skill_mutation_process_lock(canonical_skill_dir):
            with file_state.lock_path(str(canonical_skill_dir)):
                target_existed = os.path.lexists(target)
                original_bytes: bytes | None = None
                original_mode = 0
                live_target_identity: tuple[int, int, int] | None = None
                live_skill_dir_identity = _lstat_identity(canonical_skill_dir)
                if target_existed:
                    original_stat = _ensure_regular_identity(target)
                    original_mode = original_stat.st_mode
                    original_bytes = target.read_bytes()
                    live_target_identity = (
                        original_stat.st_dev,
                        original_stat.st_ino,
                        stat.S_IFMT(original_stat.st_mode),
                    )

                denial = _final_skill_mutation_denial(
                    "write_file", target, origin="skill_manager_write_file_staging"
                )
                if denial is not None:
                    _primary = denial
                    raise _SkillMutationAborted

                # Build staging copy of the live skill.
                staging = _create_private_staging(skills_root)
                staged_skill_dir = staging / canonical_skill_dir.name
                _copy_skill_into_staging(canonical_skill_dir, staged_skill_dir)
                staged_target = staged_skill_dir / target.relative_to(canonical_skill_dir)

                # Write the supporting file ONLY into staging.
                token = _with_skill_operation("write_file")
                try:
                    staged_target.parent.mkdir(parents=True, exist_ok=True)
                    with open(staged_target, "wb") as f:
                        f.write(candidate_bytes)
                finally:
                    _reset_skill_operation(token)
                _ensure_regular_identity(staged_target)
                if staged_target.read_bytes() != candidate_bytes:
                    _primary = {
                        "success": False,
                        "error": "staging byte mismatch before scan",
                        "policy_reason": "staging_failure",
                        "rollback_failure_kind": "staging_failure",
                        "target": str(target),
                    }
                else:
                    # Scan the staged tree.
                    scan_error = _security_scan_skill_fail_closed(staged_skill_dir)
                    if scan_error:
                        _primary = {
                            "success": False,
                            "error": scan_error,
                            "policy_reason": "rollback_failed",
                            "rollback_failure_kind": "scan_failure",
                            "scan_error": scan_error,
                            "target": str(target),
                        }
                    else:
                        # Re-validate live target identity for concurrent modification.
                        if target_existed:
                            current_stat = _lstat_identity(target)
                            if (
                                live_target_identity is not None
                                and current_stat != live_target_identity
                            ):
                                _primary = {
                                    "success": False,
                                    "error": "live target inode changed during scan; concurrent modification preserved",
                                    "policy_reason": "rollback_failed",
                                    "rollback_failure_kind": "concurrent_modification",
                                    "target": str(target),
                                }
                            elif (
                                original_bytes is not None
                                and target.read_bytes() != original_bytes
                            ):
                                _primary = {
                                    "success": False,
                                    "error": "live target bytes changed during scan; concurrent modification preserved",
                                    "policy_reason": "rollback_failed",
                                    "rollback_failure_kind": "concurrent_modification",
                                    "target": str(target),
                                }
                            elif (
                                original_mode
                                and stat.S_IMODE(Path(target).stat().st_mode)
                                != stat.S_IMODE(original_mode)
                            ):
                                _primary = {
                                    "success": False,
                                    "error": "live target mode changed during scan; concurrent modification preserved",
                                    "policy_reason": "rollback_failed",
                                    "rollback_failure_kind": "concurrent_modification",
                                    "target": str(target),
                                }
                            else:
                                _primary = _publish_write_file(
                                    target, staged_target, target_existed, original_mode,
                                    original_bytes, candidate_bytes,
                                    created_parent_identities, existing,
                                )
                                if _primary is None:
                                    _live_committed = True
                        elif os.path.lexists(target):
                            _primary = {
                                "success": False,
                                "error": "live target appeared during scan; concurrent modification preserved",
                                "policy_reason": "rollback_failed",
                                "rollback_failure_kind": "concurrent_modification",
                                "target": str(target),
                            }
                        else:
                            _primary = _publish_write_file(
                                target, staged_target, target_existed, original_mode,
                                original_bytes, candidate_bytes,
                                created_parent_identities, existing,
                            )
                            if _primary is None:
                                _live_committed = True
    except _SkillMutationAborted:
        pass
    except _SkillMutationLockAcquireFailure:
        import sys as _sys
        _exc_info = _sys.exc_info()[1]
        _lock_acquire_exc = (
            _exc_info if isinstance(_exc_info, _SkillMutationLockAcquireFailure) else None
        )
    except _SkillMutationLockReleaseFailure:
        import sys as _sys
        _lock_release_exc = _sys.exc_info()[1]
    finally:
        cleanup_failure = _cleanup_private_staging(staging)

    if _lock_acquire_exc is not None:
        return _format_lock_acquisition_failure_payload(
            _lock_acquire_exc, operation_kind="write_file", target=target,
        )

    if _lock_release_exc is not None:
        _lock_release_exc.live_mutation_committed = bool(_live_committed)
        payload = _format_lock_release_failure_payload(_lock_release_exc, target=target)
        return _combine_lock_release_with_cleanup(payload, cleanup_failure)

    if _primary is None and cleanup_failure is None:
        result = {
            "success": True,
            "message": f"File '{file_path}' written to skill '{name}'.",
            "path": str(target),
        }
        org_note = _maybe_auto_propose_org_edit(name, existing["path"])
        if org_note:
            result["org_sharing"] = org_note
            result["message"] = f"{result['message']} {org_note}"
        return result

    if _primary is None:
        _primary = {
            "success": True,
            "message": f"File '{file_path}' written to skill '{name}'.",
            "path": str(target),
        }
    return _combine_cleanup_failure(
        _primary, cleanup_failure, live_mutation_committed=_live_committed
    )


def _remove_file(name: str, file_path: str) -> Dict[str, Any]:
    """Remove a supporting file from any skill directory."""
    err = _validate_file_path(file_path)
    if err:
        return {"success": False, "error": err}

    existing = _find_skill(name)
    if not existing:
        return {"success": False, "error": _skill_not_found_error(name)}

    skill_dir = existing["path"]
    guard = _background_review_write_guard(name, skill_dir, "remove_file")
    if guard:
        return guard

    target, err = _resolve_skill_target(skill_dir, file_path)
    if err:
        return {"success": False, "error": err}
    assert target is not None
    if not target.exists():
        # List what's actually there for the model to see
        available = []
        for subdir in ALLOWED_SUBDIRS:
            d = skill_dir / subdir
            if d.exists():
                for f in d.rglob("*"):
                    if f.is_file():
                        available.append(str(f.relative_to(skill_dir)))
        return {
            "success": False,
            "error": f"File '{file_path}' not found in skill '{name}'.",
            "available_files": available if available else None,
        }

    read_guard = _background_review_read_before_write_guard(
        name, target, "remove_file", file_path
    )
    if read_guard:
        return read_guard

    canonical_skill_dir = skill_dir.resolve(strict=False)
    # Capture target identity BEFORE entering the lock so a concurrent
    # swap can be detected under the lock.
    pre_lock_target_identity: Optional[Tuple[int, int, int]] = None
    pre_lock_parent_identity: Optional[Tuple[int, int, int]] = None
    try:
        if os.path.lexists(str(target)):
            pre_st = target.lstat()
            pre_lock_target_identity = (
                pre_st.st_dev,
                pre_st.st_ino,
                stat.S_IFMT(pre_st.st_mode),
            )
            pre_lock_parent_identity = _lstat_identity(target.parent)
    except OSError:
        pass
    # Phase C P1-C boundary: track whether the lock context's
    # ``__enter__`` has completed successfully so the raw-PE fallback
    # below ONLY classifies pre-entry failures as acquisition.
    _lock_context_entered = False
    try:
        with _skill_mutation_process_lock(canonical_skill_dir):
            _lock_context_entered = True
            with file_state.lock_path(str(canonical_skill_dir)):
                # Revalidate under the lock.  Re-resolve the skill,
                # rebuild the target from the validated relative path,
                # confirm containment, capture identity.
                re_existing = _find_skill(name)
                if not re_existing:
                    return {
                        "success": False,
                        "error": _skill_not_found_error(name),
                        "policy_reason": "concurrent_modification",
                        "rollback_failure_kind": "concurrent_modification",
                        "target": str(target),
                    }
                re_skill_dir = re_existing["path"].resolve(strict=False)
                if re_skill_dir != canonical_skill_dir:
                    return {
                        "success": False,
                        "error": (
                            f"Skill '{name}' was replaced between pre-lock "
                            f"and lock acquisition; foreign object preserved."
                        ),
                        "policy_reason": "concurrent_modification",
                        "rollback_failure_kind": "concurrent_modification",
                        "target": str(target),
                    }
                re_target, err = _resolve_skill_target(re_existing["path"], file_path)
                if err or re_target is None:
                    return {
                        "success": False,
                        "error": err or "could not resolve target",
                        "policy_reason": "concurrent_modification",
                        "rollback_failure_kind": "concurrent_modification",
                        "target": str(target),
                    }
                # The target's parent must be inside the canonical skill dir.
                try:
                    re_target.parent.resolve(strict=False).relative_to(canonical_skill_dir)
                except (ValueError, OSError):
                    return {
                        "success": False,
                        "error": (
                            f"target '{re_target}' resolves outside the "
                            f"canonical skill directory; refusing to remove."
                        ),
                        "policy_reason": "concurrent_modification",
                        "rollback_failure_kind": "concurrent_modification",
                        "target": str(re_target),
                    }
                guard = _background_review_write_guard(name, re_skill_dir, "remove_file")
                if guard:
                    return guard
                denial = _final_skill_mutation_denial(
                    "remove_file", re_target, origin="skill_manager_remove_file"
                )
                if denial is not None:
                    return denial
                # Capture target identity + parent identity immediately
                # before the destructive op.
                try:
                    target_st = re_target.lstat()
                except OSError as exc:
                    return {
                        "success": False,
                        "error": (
                            f"could not lstat '{re_target}' before remove: {exc}"
                        ),
                        "policy_reason": "concurrent_modification",
                        "rollback_failure_kind": "concurrent_modification",
                        "target": str(re_target),
                    }
                if stat.S_ISLNK(target_st.st_mode):
                    # Symlink target — refuse to follow and refuse to
                    # unlink it (it could redirect outside the skill dir).
                    return {
                        "success": False,
                        "error": (
                            f"target '{re_target}' is a symlink; refusing "
                            f"to remove (the symlink target may be outside "
                            f"the skill directory)."
                        ),
                        "policy_reason": "concurrent_modification",
                        "rollback_failure_kind": "symlink_detected",
                        "target": str(re_target),
                    }
                if not stat.S_ISREG(target_st.st_mode):
                    return {
                        "success": False,
                        "error": (
                            f"target '{re_target}' is not a regular file."
                        ),
                        "policy_reason": "concurrent_modification",
                        "rollback_failure_kind": "concurrent_modification",
                        "target": str(re_target),
                    }
                current_target_identity = (
                    target_st.st_dev,
                    target_st.st_ino,
                    stat.S_IFMT(target_st.st_mode),
                )
                current_parent_identity = _lstat_identity(re_target.parent)
                if (
                    pre_lock_target_identity is not None
                    and current_target_identity != pre_lock_target_identity
                ):
                    return {
                        "success": False,
                        "error": (
                            f"target '{re_target}' was replaced between "
                            f"pre-lock and lock acquisition; foreign object "
                            f"preserved."
                        ),
                        "policy_reason": "concurrent_modification",
                        "rollback_failure_kind": "concurrent_modification",
                        "target": str(re_target),
                    }
                if (
                    pre_lock_parent_identity is not None
                    and current_parent_identity != pre_lock_parent_identity
                ):
                    return {
                        "success": False,
                        "error": (
                            f"parent of '{re_target}' changed between "
                            f"pre-lock and lock acquisition; foreign object "
                            f"preserved."
                        ),
                        "policy_reason": "concurrent_modification",
                        "rollback_failure_kind": "concurrent_modification",
                        "target": str(re_target),
                    }
                # Final pre-unlink identity check (immediately before
                # the unlink syscall) — even with the lock held this
                # defends against a kernel-level swap.
                try:
                    recheck_st = re_target.lstat()
                    recheck_identity = (
                        recheck_st.st_dev,
                        recheck_st.st_ino,
                        stat.S_IFMT(recheck_st.st_mode),
                    )
                    recheck_parent_identity = _lstat_identity(re_target.parent)
                except OSError as exc:
                    return {
                        "success": False,
                        "error": (
                            f"could not lstat '{re_target}' immediately "
                            f"before remove: {exc}"
                        ),
                        "policy_reason": "concurrent_modification",
                        "rollback_failure_kind": "concurrent_modification",
                        "target": str(re_target),
                    }
                if recheck_identity != current_target_identity:
                    return {
                        "success": False,
                        "error": (
                            f"target '{re_target}' identity changed between "
                            f"revalidation and remove; foreign object "
                            f"preserved."
                        ),
                        "policy_reason": "concurrent_modification",
                        "rollback_failure_kind": "concurrent_modification",
                        "target": str(re_target),
                    }
                if recheck_parent_identity != current_parent_identity:
                    return {
                        "success": False,
                        "error": (
                            f"parent of '{re_target}' changed between "
                            f"revalidation and remove; foreign object "
                            f"preserved."
                        ),
                        "policy_reason": "concurrent_modification",
                        "rollback_failure_kind": "concurrent_modification",
                        "target": str(re_target),
                    }
                # Use O_NOFOLLOW + dir_fd on POSIX when available so
                # the unlink syscall cannot follow a symlink or
                # operate on a swapped parent directory.  Once the
                # secure path is committed to, there is NO pathname
                # fallback — failure on dir_fd open, dir_fd identity
                # check, the dir_fd unlink itself, OR the parent fd
                # close all surface as structured failures (and the
                # destructive op is reported as
                # ``live_mutation_committed=True`` only if the unlink
                # syscall itself succeeded).
                if not (
                    hasattr(os, "unlink")
                    and hasattr(os, "open")
                    and hasattr(os, "O_NOFOLLOW")
                    and hasattr(os, "O_DIRECTORY")
                    and _IS_POSIX
                ):
                    # Capability gate (NOT a runtime fallback): when
                    # the platform cannot honour dir_fd we refuse
                    # outright.  We must not enter the secure path
                    # and reinterpret an operational failure as
                    # "platform unsupported" — that would be a silent
                    # downgrade.  The PLATFORM_FALLBACK contract here
                    # is: report structured refusal; never run a
                    # pathname-based unlink.
                    return {
                        "success": False,
                        "error": (
                            f"platform {os.name!r} does not support dir_fd-based "
                            f"unlink; refusing to remove '{re_target}' via the "
                            f"insecure pathname fallback"
                        ),
                        "policy_reason": "concurrent_modification",
                        "rollback_failure_kind": "dirfd_open_failed",
                        "operation_kind": "remove_file",
                        "live_mutation_committed": False,
                        "safe_to_retry": False,
                        "target": str(re_target),
                        "parent": str(re_target.parent),
                    }
                parent_fd: Optional[int] = None
                try:
                    try:
                        parent_fd = os.open(
                            str(re_target.parent),
                            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        )
                    except (OSError, ValueError) as exc:
                        return {
                            "success": False,
                            "error": (
                                f"could not open parent dir for dir_fd unlink of "
                                f"{re_target}: {exc}"
                            ),
                            "policy_reason": "parent_open_failed",
                            "rollback_failure_kind": "parent_fd_open_failure",
                            "operation_kind": "remove_file",
                            "live_mutation_committed": False,
                            "safe_to_retry": False,
                            "target": str(re_target),
                            "parent": str(re_target.parent),
                        }
                    try:
                        # Verify parent fd still points at the same
                        # directory we lstat'd.  Any mismatch leaves
                        # the foreign object intact and surfaces a
                        # structured failure with the canonical
                        # ``parent_identity_mismatch`` kind.
                        try:
                            parent_fd_st = os.fstat(parent_fd)
                            parent_path_st = os.lstat(str(re_target.parent))
                        except OSError as exc:
                            return {
                                "success": False,
                                "error": (
                                    f"could not fstat/lstat parent fd for dir_fd "
                                    f"unlink of {re_target}: {exc}"
                                ),
                                "policy_reason": "concurrent_modification",
                                "rollback_failure_kind": "parent_fd_open_failure",
                                "operation_kind": "remove_file",
                                "live_mutation_committed": False,
                                "safe_to_retry": False,
                                "target": str(re_target),
                                "parent": str(re_target.parent),
                            }
                        if (
                            parent_fd_st.st_dev,
                            parent_fd_st.st_ino,
                            stat.S_IFMT(parent_fd_st.st_mode),
                        ) != (
                            parent_path_st.st_dev,
                            parent_path_st.st_ino,
                            stat.S_IFMT(parent_path_st.st_mode),
                        ):
                            return {
                                "success": False,
                                "error": (
                                    f"parent fd identity changed during dir_fd "
                                    f"unlink of {re_target}; foreign object "
                                    f"preserved."
                                ),
                                "policy_reason": "concurrent_modification",
                                "rollback_failure_kind": "parent_identity_mismatch",
                                "operation_kind": "remove_file",
                                "live_mutation_committed": False,
                                "safe_to_retry": False,
                                "target": str(re_target),
                                "parent": str(re_target.parent),
                            }
                        # Step 9 of the contract: lstat the target
                        # RELATIVE to the open parent fd.  This is
                        # the kernel-anchored revalidation that catches
                        # a swap between the pathname pre-unlink
                        # recheck and the actual unlink syscall.  If
                        # the kernel no longer resolves
                        # ``re_target.name`` to the same inode under
                        # this dir_fd, we refuse without unlinking.
                        try:
                            fd_target_st = os.lstat(
                                re_target.name, dir_fd=parent_fd
                            )
                        except (OSError, ValueError) as exc:
                            return {
                                "success": False,
                                "error": (
                                    f"could not lstat target '{re_target.name}' "
                                    f"via parent fd for dir_fd unlink of "
                                    f"{re_target}: {exc}"
                                ),
                                "policy_reason": "concurrent_modification",
                                "rollback_failure_kind": "target_identity_mismatch",
                                "operation_kind": "remove_file",
                                "live_mutation_committed": False,
                                "safe_to_retry": False,
                                "target": str(re_target),
                                "parent": str(re_target.parent),
                            }
                        fd_target_identity = (
                            fd_target_st.st_dev,
                            fd_target_st.st_ino,
                            stat.S_IFMT(fd_target_st.st_mode),
                        )
                        if fd_target_identity != current_target_identity:
                            return {
                                "success": False,
                                "error": (
                                    f"target '{re_target}' identity changed between "
                                    f"dir_fd open and unlink (kernel-anchored "
                                    f"revalidation); foreign object preserved."
                                ),
                                "policy_reason": "concurrent_modification",
                                "rollback_failure_kind": "target_identity_mismatch",
                                "operation_kind": "remove_file",
                                "live_mutation_committed": False,
                                "safe_to_retry": False,
                                "target": str(re_target),
                                "parent": str(re_target.parent),
                            }
                        # Camino B (restored by R2E after the R2
                        # regression): refuse the destructive op when
                        # no kernel identity-bound delete primitive is
                        # available.  The final identity check above
                        # (``os.lstat(name, dir_fd=parent_fd)``) and
                        # the unlink below (``os.unlink(name,
                        # dir_fd=parent_fd)``) are TWO separate
                        # syscalls that share only the parent
                        # namespace; the kernel re-resolves the
                        # basename at unlink time.  A non-cooperative
                        # actor that swaps the inode on disk between
                        # the final lstat and the unlink would see
                        # production delete the foreign replacement.
                        # Portable Python does not expose
                        # ``unlinkat(target_fd, AT_EMPTY_PATH)`` and
                        # ``os.unlink`` cannot be made conditional on
                        # ``st_dev/st_ino``.  We therefore refuse the
                        # destructive op unconditionally rather than
                        # run a name-based unlink that might delete
                        # the wrong inode.
                        #
                        # A capture-then-verify quarantine rename is
                        # NOT an acceptable substitute: ``os.replace``
                        # also resolves the SOURCE pathname at rename
                        # time and would move the foreign replacement
                        # out of its canonical pathname before Hermes
                        # could verify it, violating the canonical
                        # Phase2 invariants S3 (no wrong-object
                        # namespace mutation) and S4 (foreign object
                        # path preservation on failure).
                        #
                        # The contract:
                        #   success=False
                        #   policy_reason=atomic_identity_delete_unavailable
                        #   rollback_failure_kind=identity_bound_unlink_unavailable
                        #   live_mutation_committed=False
                        #   safe_to_retry=False
                        # No unlink syscall runs in this branch.
                        #
                        # Close the parent_fd HERE in a guarded
                        # try/except that does NOT promote to
                        # ``_SkillMutationLockReleaseFailure`` — that
                        # helper assumes a destructive op ran and
                        # would set ``live_mutation_committed=True``,
                        # which is wrong for Camino B.  A close
                        # failure during finalization of a refused
                        # op is logged and ignored.
                        if parent_fd is not None:
                            try:
                                os.close(parent_fd)
                            except OSError:
                                # Finalization-only close on a refused
                                # op.  No destructive op ran; we
                                # cannot honestly report
                                # ``live_mutation_committed=True`` so
                                # we swallow the close error.
                                pass
                            parent_fd = None
                        return {
                            "success": False,
                            "error": (
                                f"refusing remove_file on {re_target}: "
                                f"no kernel identity-bound delete primitive "
                                f"is available on this platform "
                                f"(os.unlink is by-name via dir_fd and is "
                                f"not conditional on the validated "
                                f"st_dev/st_ino); non-cooperative "
                                f"replacement between the final identity "
                                f"check and the unlink syscall cannot be "
                                f"ruled out, so the destructive op is "
                                f"withheld."
                            ),
                            "policy_reason": "atomic_identity_delete_unavailable",
                            "rollback_failure_kind": "identity_bound_unlink_unavailable",
                            "operation_kind": "remove_file",
                            "live_mutation_committed": False,
                            "safe_to_retry": False,
                            "target": str(re_target),
                            "parent": str(re_target.parent),
                        }
                    except (OSError, PermissionError, ValueError) as exc:
                        return {
                            "success": False,
                            "error": (
                                f"dir_fd unlink of {re_target} failed: {exc}"
                            ),
                            "policy_reason": "remove_failed",
                            "rollback_failure_kind": "unlink_failure",
                            "operation_kind": "remove_file",
                            "live_mutation_committed": False,
                            "safe_to_retry": False,
                            "target": str(re_target),
                            "parent": str(re_target.parent),
                        }
                    finally:
                        if parent_fd is not None:
                            try:
                                os.close(parent_fd)
                            except OSError as exc:
                                # The unlink already ran.  Surface a
                                # structured payload with
                                # ``live_mutation_committed=True`` so the
                                # operator inspects the disk; do NOT
                                # attempt a second unlink via the
                                # pathname.
                                raise _SkillMutationLockReleaseFailure(
                                    canonical_skill_path=canonical_skill_dir,
                                    lock_path=Path("(not held)"),
                                    platform="posix",
                                    release_error=None,
                                    close_error=exc,
                                    live_mutation_committed=True,
                                ) from exc
                # End of secure-path block.  Anything below runs ONLY
                # when the secure path is unavailable AND we already
                # surfaced a structured failure above.  The legacy
                # ``re_target.unlink()`` pathname fallback has been
                # removed because it would let a symlink swap slip
                # through.
                except _SkillMutationLockReleaseFailure:
                    raise

                # Clean up empty subdirectories
                parent = re_target.parent
                if parent != re_skill_dir and parent.exists() and not any(parent.iterdir()):
                    denial = _final_skill_mutation_denial("remove_file", parent, origin="skill_manager_remove_empty_parent")
                    if denial is not None:
                        return denial
                    parent.rmdir()

                return {
                    "success": True,
                    "message": f"File '{file_path}' removed from skill '{name}'.",
                }
    except _SkillMutationLockAcquireFailure as exc:
        return _format_lock_acquisition_failure_payload(
            exc, operation_kind="remove_file", target=target,
        )
    except PermissionError as exc:
        # Defensive: keep pre-existing tests that inject raw
        # PermissionError from a fake lock context-manager.
        # Phase C P1-C boundary: ONLY coerce when the lock
        # ``__enter__`` has not returned.  A raw PermissionError raised
        # after entry is a body / release failure and must NOT be
        # silently misclassified as a lock acquisition failure.
        if not _lock_context_entered:
            structured = _coerce_raw_permission_error_to_acquire_failure(
                canonical_skill_path=canonical_skill_dir, exc=exc,
            )
            return _format_lock_acquisition_failure_payload(
                structured, operation_kind="remove_file", target=target,
            )
        raise
    except _SkillMutationLockReleaseFailure as exc:
        # remove_file is destructive once unlink() runs.  By the time
        # the release failure surfaces we may already have unlinked
        # the file.  Report ``live_mutation_committed=True`` so the
        # operator inspects the disk.
        #
        # Distinguish the parent-fd close failure (the secure-path
        # close at the end of ``os.unlink(name, dir_fd=parent_fd)``)
        # from any interprocess-lock release failure: the close error
        # is the new ``parent_fd_close_failure`` kind with
        # ``policy_reason=finalization_failed``; the interprocess
        # release failure keeps the generic ``lock_release_failure``
        # semantics.  We detect the parent-fd close case by checking
        # that ``exc.platform == "posix"`` AND ``exc.lock_path`` is
        # the sentinel ``"(not held)"`` we wrote in the secure-path
        # finally block — the interprocess lock path always supplies
        # a real ``.lock`` path.
        exc.live_mutation_committed = True
        base_payload = _format_lock_release_failure_payload(exc, target=target)
        if str(exc.lock_path) == "(not held)":
            # Parent-fd close failure: contract requires the canonical
            # ``parent_fd_close_failure`` kind + ``finalization_failed``
            # policy reason.  All other fields (close_error, lock_path,
            # release_error) are preserved from the helper payload.
            base_payload["policy_reason"] = "finalization_failed"
            base_payload["rollback_failure_kind"] = "parent_fd_close_failure"
            base_payload["operation_kind"] = "remove_file"
        return base_payload


# =============================================================================
# Main entry point
# =============================================================================

# ContextVar bypass: set while replaying an already-approved staged skill write
# so skill_manage() does not re-gate (and re-stage) it.
import contextvars as _ctxvars
_skill_gate_bypass: "_ctxvars.ContextVar[bool]" = _ctxvars.ContextVar(
    "skill_gate_bypass", default=False
)


def _apply_skill_write_gate(action, name, **payload_kwargs):
    """Evaluate the skill write gate. Returns a JSON tool-result string when the
    write should NOT proceed (blocked or staged), or None to perform the real
    write. Bypassed during approved-pending replay.
    """
    if action not in {"create", "edit", "patch", "delete", "write_file", "remove_file"}:
        return None
    if _skill_gate_bypass.get():
        return None

    try:
        from tools import write_approval as wa
    except Exception:
        return None  # fail open

    decision = wa.evaluate_gate(wa.SKILLS)
    if decision.allow:
        return None
    if decision.blocked:
        return tool_error(decision.message, success=False)

    # stage — record the full skill_manage kwargs so approval can replay it.
    payload = {"action": action, "name": name}
    payload.update({k: v for k, v in payload_kwargs.items() if v is not None})
    gist = wa.skill_gist(
        action, name,
        content=payload_kwargs.get("content") or "",
        file_path=payload_kwargs.get("file_path") or "",
        old_string=payload_kwargs.get("old_string") or "",
        new_string=payload_kwargs.get("new_string") or "",
    )
    record = wa.stage_write(wa.SKILLS, payload, summary=gist, origin=wa.current_origin())
    return json.dumps(
        {"success": True, "staged": True, "pending_id": record["id"],
         "gist": gist, "message": decision.message},
        ensure_ascii=False,
    )


def apply_skill_pending(payload: Dict[str, Any]) -> str:
    """Replay a staged skill write, bypassing the gate. Returns the tool result
    JSON string. Called by the /skills approve handler.
    """
    token = _skill_gate_bypass.set(True)
    try:
        return skill_manage(
            action=payload.get("action", ""),
            name=payload.get("name", ""),
            content=payload.get("content"),
            category=payload.get("category"),
            file_path=payload.get("file_path"),
            file_content=payload.get("file_content"),
            old_string=payload.get("old_string"),
            new_string=payload.get("new_string"),
            replace_all=payload.get("replace_all", False),
            absorbed_into=payload.get("absorbed_into"),
        )
    finally:
        _skill_gate_bypass.reset(token)


# Debounce state for the sync push hook. A burst of skill_manage writes
# (e.g. create + several write_file calls) collapses into a single push after
# a short quiet window, on a daemon timer so the agent write never blocks.
_sync_push_timer = None
_sync_push_lock = None
_SYNC_PUSH_DEBOUNCE_S = 5.0


def _maybe_debounced_sync_push(skill_name: str) -> None:
    """Schedule a debounced best-effort sync push after a skill write.

    Cheap fast-path: if the skill isn't opted into sync, do nothing (no auth,
    no network). Otherwise (re)arm a daemon timer; the actual push runs through
    ``skills_sync_client.maybe_push_skills`` which enforces the access gate
    and swallows all errors. Never blocks the caller (M1-C: agent never blocks
    on sync).
    """
    global _sync_push_timer, _sync_push_lock
    try:
        from tools.skill_usage import is_sync_enabled

        if not is_sync_enabled(skill_name):
            return
    except Exception:
        return

    import threading

    if _sync_push_lock is None:
        _sync_push_lock = threading.Lock()

    def _fire():
        try:
            from tools.skills_sync_client import maybe_push_skills

            maybe_push_skills(message=f"sync: {skill_name}")
        except Exception:
            pass

    with _sync_push_lock:
        if _sync_push_timer is not None:
            try:
                _sync_push_timer.cancel()
            except Exception:
                pass
        _sync_push_timer = threading.Timer(_SYNC_PUSH_DEBOUNCE_S, _fire)
        _sync_push_timer.daemon = True
        _sync_push_timer.start()


def skill_manage(
    action: str,
    name: str,
    content: str = None,
    category: str = None,
    file_path: str = None,
    file_content: str = None,
    old_string: str = None,
    new_string: str = None,
    replace_all: bool = False,
    absorbed_into: str = None,
    task_id: str = None,
    session_id: str = None,
) -> str:
    """
    Manage user-created skills. Dispatches to the appropriate action handler.

    Returns JSON string with results.
    """
    preflight = _background_review_preflight(action, name, category)
    if preflight is not None:
        return json.dumps(preflight, ensure_ascii=False)

    if action in _SKILL_ACTION_OPERATION_KIND:
        try:
            from agent.session_write_policy import (
                CapabilityGrant,
                SessionWritePolicyMode,
                evaluate_session_write_policy,
                get_current_session_write_policy,
            )

            policy = get_current_session_write_policy(protected=False)
            if policy.mode is SessionWritePolicyMode.DENY_ALL:
                decision = evaluate_session_write_policy(
                    policy,
                    operation_kind=_SKILL_ACTION_OPERATION_KIND[action],
                    origin="skill_manager_preflight",
                    capability=CapabilityGrant("filesystem", _SKILL_ACTION_OPERATION_KIND[action]),
                )
                if decision.denied:
                    return decision.denial_json()
        except Exception as e:
            logger.debug("session write policy skill preflight failed: %s", e)
            try:
                from agent.session_write_policy import policy_evaluation_failure_payload

                return json.dumps(
                    policy_evaluation_failure_payload(
                        operation_kind=_SKILL_ACTION_OPERATION_KIND[action],
                        session_id="",
                        target=name or "",
                        error=e,
                    ),
                    ensure_ascii=False,
                )
            except Exception:
                return json.dumps(
                    {
                        "success": False,
                        "error": "Session write policy evaluation failed; mutation denied",
                        "policy_reason": "policy_evaluation_failed",
                        "operation_kind": _SKILL_ACTION_OPERATION_KIND[action],
                        "session_id": "",
                        "target": name or "",
                    },
                    ensure_ascii=False,
                )

    # Approval gate: when on, stages the write for review (skills are too large
    # to review inline, so they always stage regardless of origin); when off
    # (default) passes straight through. The gate is bypassed when this call is
    # itself replaying an already-approved staged write (_skill_apply_pending).
    gate_result = _apply_skill_write_gate(
        action, name, content=content, category=category,
        file_path=file_path, file_content=file_content,
        old_string=old_string, new_string=new_string,
        replace_all=replace_all, absorbed_into=absorbed_into,
    )
    if gate_result is not None:
        return gate_result

    # Audit ledger (tracker #79686 P3): capture the pre-mutation state of the
    # skill directory so every mutation — any actor — lands in the append-only
    # JSONL ledger with before/after blobs. Telemetry, not a gate: failures
    # here must NEVER block the mutation (capture_before returns None on
    # error, and record_mutation below swallows everything).
    _ledger_before = None
    _ledger_before_dir = None
    try:
        from tools import skill_ledger as _ledger
        _pre = _find_skill(name)
        _ledger_before_dir = _pre["path"] if _pre else None
        _ledger_before = _ledger.capture_before(_ledger_before_dir)
    except Exception:
        pass

    if action == "create":
        if not content:
            return tool_error("content is required for 'create'. Provide the full SKILL.md text (frontmatter + body).", success=False)
        result = _create_skill(name, content, category)

    elif action == "edit":
        if not content:
            return tool_error("content is required for 'edit'. Provide the full updated SKILL.md text.", success=False)
        result = _edit_skill(name, content)

    elif action == "patch":
        if not old_string:
            return tool_error("old_string is required for 'patch'. Provide the text to find.", success=False)
        if new_string is None:
            return tool_error("new_string is required for 'patch'. Use empty string to delete matched text.", success=False)
        result = _patch_skill(name, old_string, new_string, file_path, replace_all)

    elif action == "delete":
        result = _delete_skill(name, absorbed_into=absorbed_into)

    elif action == "write_file":
        if not file_path:
            return tool_error("file_path is required for 'write_file'. Example: 'references/api-guide.md'", success=False)
        if file_content is None:
            return tool_error("file_content is required for 'write_file'.", success=False)
        result = _write_file(name, file_path, file_content)

    elif action == "remove_file":
        if not file_path:
            return tool_error("file_path is required for 'remove_file'.", success=False)
        result = _remove_file(name, file_path)

    else:
        result = {"success": False, "error": f"Unknown action '{action}'. Use: create, edit, patch, delete, write_file, remove_file"}

    if result.get("success"):
        # Audit ledger append (best-effort; never blocks the mutation).
        try:
            from tools import skill_ledger as _ledger
            _post = _find_skill(name)
            _after_dir = _post["path"] if _post else None
            _evidence = {}
            if action == "delete":
                # Record delete intent: consolidation vs prune, and whether
                # the recoverable-archive path handled it (curator pass).
                _evidence["absorbed_into"] = absorbed_into
                _evidence["archived"] = bool(result.get("_archived"))
            if session_id:
                _evidence["session_id"] = session_id
            if file_path:
                _evidence["file_path"] = file_path
            _ledger.record_mutation(
                action,
                name,
                before=_ledger_before if _ledger_before is not None else [],
                after_root=_after_dir,
                evidence=_evidence,
            )
        except Exception:
            pass
        try:
            from agent.prompt_builder import clear_skills_system_prompt_cache
            clear_skills_system_prompt_cache(clear_snapshot=True)
        except Exception:
            pass
        # Curator telemetry: bump patch_count on edit/patch/write_file (the actions
        # that mutate an existing skill's guidance), drop the record on delete.
        # Only mark a skill as agent-created when the background self-improvement
        # review fork creates it — foreground `skill_manage(create)` calls are
        # user-directed, and those skills belong to the user (the curator must
        # not touch them). Best-effort; telemetry failures never break the tool.
        # ``record_created`` subsumes the previous ``mark_agent_created`` opt-in
        # (it sets ``created_by="agent"`` whenever ``agent_created=True``), so
        # we never call both — that would double-count the sidecar mutation.
        try:
            from tools.skill_usage import bump_patch, forget, record_created
            from tools.skill_provenance import is_background_review
            if action == "create":
                record_created(
                    name,
                    agent_created=is_background_review(),
                    task_id=task_id,
                    session_id=session_id,
                )
            elif action in {"patch", "edit", "write_file", "remove_file"}:
                bump_patch(
                    name,
                    action=action,
                    task_id=task_id,
                    session_id=session_id,
                )
            elif action == "delete":
                # A recoverable curator archive (routed through archive_skill)
                # keeps its usage record as STATE_ARCHIVED so `hermes curator
                # status`/`restore` still see it. Only a hard delete forgets.
                if not result.get("_archived"):
                    forget(name)
        except Exception:
            pass

        # Sync push hook (debounced, best-effort). Fires only AFTER the
        # write gate passed (staged/unapproved writes never reach here -- the
        # gate returns early above), so we never push un-reviewed content.
        # Inert unless the access gate is open (the user is a Nous admin on the
        # token), a sync base URL is configured, and the skill is opted into
        # sync. Debounced so a burst of edits collapses to one push. Never
        # raises -- an agent write must never block on sync (M1-C invariant).
        try:
            _maybe_debounced_sync_push(name)
        except Exception:
            pass

    return json.dumps(result, ensure_ascii=False)


# =============================================================================
# OpenAI Function-Calling Schema
# =============================================================================

SKILL_MANAGE_SCHEMA = {
    "name": "skill_manage",
    "description": (
        "Manage skills (create, update, delete). Skills are your procedural "
        "memory — reusable approaches for recurring task types. "
        f"New skills go to {display_hermes_home()}/skills/; existing skills can be modified wherever they live.\n\n"
        "Actions: create (full SKILL.md + optional category), "
        "patch (old_string/new_string — preferred for fixes), "
        "edit (full SKILL.md rewrite — major overhauls only), "
        "delete, write_file, remove_file.\n\n"
        "On delete, pass `absorbed_into=<umbrella>` when you're merging this "
        "skill's content into another one, or `absorbed_into=\"\"` when you're "
        "pruning it with no forwarding target. This lets the curator tell "
        "consolidation from pruning without guessing, so downstream consumers "
        "(cron jobs that reference the old skill name, etc.) get updated "
        "correctly. The target you name in `absorbed_into` must already "
        "exist — create/patch the umbrella first, then delete.\n\n"
        "Create when: complex task succeeded (5+ calls), errors overcome, "
        "user-corrected approach worked, non-trivial workflow discovered, "
        "or user asks you to remember a procedure.\n"
        "Update when: instructions stale/wrong, OS-specific failures, "
        "missing steps or pitfalls found during use. "
        "If you used a skill and hit issues not covered by it, patch it immediately.\n\n"
        "After difficult/iterative tasks, offer to save as a skill. "
        "Skip for simple one-offs. Confirm with user before creating/deleting.\n\n"
        "Good skills: trigger conditions, numbered steps with exact commands, "
        "pitfalls section, verification steps. Use skill_view() to see format examples.\n\n"
        "Description: long descriptions are truncated to the first 57 chars "
        "plus '...' in the system prompt skill index; longer text is visible "
        "via skills_list/skill_view. Keep the trigger self-contained in that "
        "first 57-char window: 'Use when <trigger>. <one-line behavior>.'\n\n"
        "Pinned skills are protected from deletion only — skill_manage(action='delete') "
        "will refuse with a message pointing the user to `hermes curator unpin <name>`. "
        "Patches and edits go through on pinned skills so you can still improve them as "
        "pitfalls come up; pin only guards against irrecoverable loss."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "patch", "edit", "delete", "write_file", "remove_file"],
                "description": "The action to perform."
            },
            "name": {
                "type": "string",
                "description": (
                    "Skill name (lowercase, hyphens/underscores, max 64 chars). "
                    "Must match an existing skill for patch/edit/delete/write_file/remove_file."
                )
            },
            "content": {
                "type": "string",
                "description": (
                    "Full SKILL.md content (YAML frontmatter + markdown body). "
                    "Required for 'create' and 'edit'. For 'edit', read the skill "
                    "first with skill_view() and provide the complete updated text."
                )
            },
            "old_string": {
                "type": "string",
                "description": (
                    "Text to find in the file (required for 'patch'). Must be unique "
                    "unless replace_all=true. Include enough surrounding context to "
                    "ensure uniqueness."
                )
            },
            "new_string": {
                "type": "string",
                "description": (
                    "Replacement text (required for 'patch'); must differ from "
                    "old_string. Can be empty string to delete the matched text."
                )
            },
            "replace_all": {
                "type": "boolean",
                "description": "For 'patch': replace all occurrences instead of requiring a unique match (default: false)."
            },
            "category": {
                "type": "string",
                "description": (
                    "Optional category/domain for organizing the skill (e.g., 'devops', "
                    "'data-science', 'mlops'). Creates a subdirectory grouping. "
                    "Only used with 'create'."
                )
            },
            "file_path": {
                "type": "string",
                "description": (
                    "Path to a supporting file within the skill directory. "
                    "For 'write_file'/'remove_file': required, must be under references/, "
                    "templates/, scripts/, or assets/. "
                    "For 'patch': optional, defaults to SKILL.md if omitted."
                )
            },
            "file_content": {
                "type": "string",
                "description": "Content for the file. Required for 'write_file'."
            },
            "absorbed_into": {
                "type": "string",
                "description": (
                    "For 'delete' only — declares intent so the curator can "
                    "tell consolidation from pruning without guessing. "
                    "Pass the umbrella skill name when this skill's content "
                    "was merged into another (the target must already exist). "
                    "Pass an empty string when the skill is truly stale and "
                    "being pruned with no forwarding target. Omitting the arg "
                    "on delete is supported for backward compatibility but "
                    "downstream tooling (e.g. cron-job skill reference "
                    "rewriting) will have to guess at intent."
                )
            },
        },
        "required": ["action", "name"],
    },
}


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="skill_manage",
    toolset="skills",
    schema=SKILL_MANAGE_SCHEMA,
    handler=lambda args, **kw: skill_manage(
        action=args.get("action", ""),
        name=args.get("name", ""),
        content=args.get("content"),
        category=args.get("category"),
        file_path=args.get("file_path"),
        file_content=args.get("file_content"),
        old_string=args.get("old_string"),
        new_string=args.get("new_string"),
        replace_all=args.get("replace_all", False),
        absorbed_into=args.get("absorbed_into")),
    emoji="📝",
)
