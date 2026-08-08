"""Neutral shared guard for live skill publishers.

This module is the canonical, import-neutral primitive that every live
skill publisher (P1..P6 in the Phase C architecture freeze) will use
in Blocks 2..5 to protect the live skill tree.

It is intentionally NOT wired to any specific publisher yet; Block 1
delivers the primitives + isolated tests, and Block 2 onwards performs
the per-publisher migration.

Architectural contract (frozen, see
HERMES_SESSION_WRITE_POLICY_PHASE_C_SHARED_LIVE_PUBLISHER_GUARD_AND_LOCK_SCOPE_ARCHITECTURE_FREEZE_PASS_AFTER_OPERATOR_RATIFICATION
and the per-block design docs):

  * Import-neutrality:
      FORBIDDEN imports (would create a cycle / coupling):
        tools.skill_manager_tool, tools.skill_usage, tools.skills_hub,
        tools.skills_sync, model_tools, run_agent,
        agent.background_review, agent.self_improvement_policy,
        agent.session_write_policy.
      ALLOWED imports: stdlib, hermes_constants (default home),
      agent.skill_utils (discovery helper shared with publishers;
      it does NOT import back into publishers or into this module).

  * Canonicalization is L1 strict (^[a-z0-9][a-z0-9._-]*$). No
    silent lowercase conversion, no Unicode normalization, no path
    traversal admission. D1-D4 of the operator-ratified contract are
    encoded in the rejection paths.

  * The global normalized-name lock is a separate namespace from the
    per-target mutation lock (.hermes-skill-name-mutex-<digest>.lock
    vs .hermes-skill-mutex-<digest>.lock). It lives OUTSIDE every
    known skills root and is derived deterministically from
    sha256(NORMALIZATION_VERSION + NUL + canonical_name).

  * Lock state is per-invocation. The discriminator distinguishes
    raw acquisition failures on the GLOBAL lock from raw acquisition
    failures on the TARGET lock from body/release failures. The
    discriminator fields are populated on the exception object so
    no caller relies on string equality of lock_path.

  * Duplicate scan covers the flat layout (<root>/<name>), the
    category layout (<root>/<category>/<name>), and configured
    external_dirs (read-only discovery). 0 / 1 / >1 matches are
    distinguished and produce different outcomes (allow / policy /
    duplicate refusal / invariant violation).

  * Replacement policies are represented as a closed literal:
    new_only, replace_same_target, replace_with_backup.
    No publisher logic is integrated in Block 1; the guard yields to
    a callback inside the locked region and Block 2 wires the real
    publish action into that callback.

This module is a TRANSITIONAL DUPLICATION of the interprocess
primitive currently in tools/skill_manager_tool.py. The duplication
is intentional and documented: Block 2 will be the migration step
that removes the old implementation in favour of this module. The two
implementations MUST remain behaviorally equivalent for the duration
of the migration.
"""

from __future__ import annotations

import dataclasses
import errno as _errno
import hashlib as _hashlib
import os
import re
import stat
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
)

# ---------------------------------------------------------------------------
# Platform detection (POSIX fcntl + Windows msvcrt; both fall back to fail-closed).
# ---------------------------------------------------------------------------
_IS_WINDOWS = sys.platform.startswith("win") or os.name == "nt"
_IS_POSIX = os.name == "posix"

try:  # POSIX-only
    import fcntl as _fcntl  # type: ignore[unused-ignore]
except ImportError:  # pragma: no cover -- POSIX without fcntl (stripped build)
    _fcntl = None  # type: ignore[assignment]

try:  # Windows-only
    import msvcrt as _msvcrt  # type: ignore[unused-ignore]
except ImportError:
    _msvcrt = None  # type: ignore[assignment]

_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)

# ---------------------------------------------------------------------------
# Canonical normalization (L1 strict)
# ---------------------------------------------------------------------------
NORMALIZATION_VERSION = "normalized-name-v1"

# L1 strict regex (Phase C frozen contract):
#   ^[a-z0-9][a-z0-9._-]*$
#   - lowercase ASCII only
#   - starts with letter or digit
#   - body letters, digits, '.', '_', '-'
#   - no spaces, no Unicode, no path separators, no traversal
VALID_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

MAX_NAME_LENGTH = 64


def canonical_normalize_skill_name(name):
    # type: (Any) -> Optional[str]
    """Return the canonical form of ``name`` for global uniqueness checks.

    Returns ``None`` for invalid names -- callers MUST treat that as a
    normal validation failure, NOT a normalized key, so the global
    mutex is never acquired on invalid input.

    Ratified D1-D4 contract decisions are encoded in the rejection
    paths:

      D1  Uppercase / non-L1 hub bundle names -> rejected (no
          silent lowercase conversion).
      D2  Archived legacy names failing L1 -> rejected (the source
          archive is preserved on disk by the caller; this helper
          does NOT touch storage).
      D3  external_dir skill names failing L1 -> caller treats the
          match as a read-only conflict; this helper itself does not
          perform any I/O.
      D4  L1-compatible names -> unchanged.
    """
    if not isinstance(name, str):
        return None
    if not name:
        return None
    if len(name) > MAX_NAME_LENGTH:
        return None
    if not VALID_NAME_RE.match(name):
        return None
    return name


def validate_name_message(name):
    # type: (str) -> Optional[str]
    """Return an error message for ``name`` if invalid, else ``None``."""
    if not name:
        return "Skill name is required."
    if len(name) > MAX_NAME_LENGTH:
        return "Skill name exceeds {0} characters.".format(MAX_NAME_LENGTH)
    if not VALID_NAME_RE.match(name):
        return (
            "Invalid skill name '{0}'. Use lowercase letters, numbers, "
            "hyphens, dots, and underscores. Must start with a letter or digit."
        ).format(name)
    return None


# ---------------------------------------------------------------------------
# Lock-failure stages (frozen, closed set).
# ---------------------------------------------------------------------------
LOCK_FAILURE_STAGE_PATH_RESOLUTION = "lock_path_resolution"
LOCK_FAILURE_STAGE_PARENT_OPEN = "lock_parent_open"
LOCK_FAILURE_STAGE_IDENTITY_VALIDATION = "lock_identity_validation"
LOCK_FAILURE_STAGE_PRIMITIVE_ACQUIRE = "lock_primitive_acquire"
LOCK_FAILURE_STAGE_CONTENTION = "lock_contention"

LOCK_FAILURE_STAGES = frozenset({
    LOCK_FAILURE_STAGE_PATH_RESOLUTION,
    LOCK_FAILURE_STAGE_PARENT_OPEN,
    LOCK_FAILURE_STAGE_IDENTITY_VALIDATION,
    LOCK_FAILURE_STAGE_PRIMITIVE_ACQUIRE,
    LOCK_FAILURE_STAGE_CONTENTION,
})


# ---------------------------------------------------------------------------
# Lock-acquire / lock-release exceptions.
#
# These re-declared here are intentionally identical to the
# _SkillMutationLockAcquireFailure / _SkillMutationLockReleaseFailure
# classes already present in tools/skill_manager_tool.py. Block 2
# will retire the old class names in favour of these public ones;
# until then the two pairs of classes are behaviorally equivalent.
# ---------------------------------------------------------------------------
class SkillMutationLockAcquireFailure(PermissionError):
    """Structured interprocess-lock acquisition failure.

    The exception carries enough metadata for the caller to build a
    canonical acquisition-failure payload without consulting any
    shared state. ``safe_to_retry`` is pre-classified per stage --
    only ``lock_contention`` may be ``True``; every other stage is
    ``safe_to_retry=False`` because a retry would re-trigger the
    same structural failure.
    """

    def __init__(
        self,
        *,
        canonical_skill_path,
        lock_path,
        platform,
        lock_failure_stage,
        cause=None,
        safe_to_retry=False,
        msg=None,
    ):
        cause_repr = (
            "{0}: {1}".format(type(cause).__name__, cause)
            if cause is not None
            else "(no underlying exception)"
        )
        summary = (
            "interprocess lock acquisition failed on {0} "
            "(platform={1}, stage={2}); "
            "cause={3}"
        ).format(lock_path, platform, lock_failure_stage, cause_repr)
        if msg is not None:
            summary = "{0} ({1})".format(msg, summary)
        super(SkillMutationLockAcquireFailure, self).__init__(summary)
        self.canonical_skill_path = Path(canonical_skill_path)
        self.lock_path = Path(lock_path)
        self.platform = platform
        self.lock_failure_stage = lock_failure_stage
        self.safe_to_retry = bool(safe_to_retry)
        self.cause_exception = cause
        if cause is not None:
            self.__cause__ = cause


class SkillMutationLockReleaseFailure(RuntimeError):
    """Structured interprocess-lock release failure.

    Captures both the release-side and close-side errors so a caller
    inspecting the exception sees the full finalization picture; the
    canonical skill path and lock path are preserved for diagnostics.
    """

    def __init__(
        self,
        *,
        canonical_skill_path,
        lock_path,
        platform,
        release_error=None,
        close_error=None,
        live_mutation_committed=False,
    ):
        release_repr = (
            "{0}: {1}".format(type(release_error).__name__, release_error)
            if release_error is not None
            else None
        )
        close_repr = (
            "{0}: {1}".format(type(close_error).__name__, close_error)
            if close_error is not None
            else None
        )
        summary = (
            "interprocess lock release failed on {0} "
            "(platform={1}); release_error={2}; "
            "close_error={3}"
        ).format(lock_path, platform, release_repr, close_repr)
        super(SkillMutationLockReleaseFailure, self).__init__(summary)
        self.canonical_skill_path = Path(canonical_skill_path)
        self.lock_path = Path(lock_path)
        self.platform = platform
        self.release_error = release_error
        self.close_error = close_error
        self.live_mutation_committed = bool(live_mutation_committed)


def _platform_name():
    if _IS_WINDOWS:
        return "windows"
    if _IS_POSIX:
        return "posix"
    return os.name or "unknown"


def _raise_lock_acquire_failure(
    *,
    canonical_skill_path,
    lock_path,
    platform,
    lock_failure_stage,
    cause=None,
    safe_to_retry=False,
    msg=None,
):
    raise SkillMutationLockAcquireFailure(
        canonical_skill_path=canonical_skill_path,
        lock_path=lock_path,
        platform=platform,
        lock_failure_stage=lock_failure_stage,
        cause=cause,
        safe_to_retry=safe_to_retry,
        msg=msg,
    )


def _validate_msvcrt_contract():
    """Return ``None`` if the msvcrt contract is satisfied, else a reason."""
    if _msvcrt is None:
        return "msvcrt module is None"
    for attr in ("LK_NBLCK", "LK_LOCK", "LK_UNLCK"):
        if not hasattr(_msvcrt, attr):
            return "msvcrt is missing required attribute {0}".format(attr)
    return None


def _validate_lock_file_identity(lock_path, *, fd=None, _stat=os.stat):
    """Confirm ``lock_path`` resolves to a regular file.

    Raises :class:`PermissionError` on any mismatch (symlink, junction,
    directory, dangling link, or inode swap between lstat and fstat).
    """
    try:
        st_path = _stat(str(lock_path), follow_symlinks=False)
    except OSError as exc:
        raise PermissionError(
            "could not lstat lock file {0}: {1}".format(lock_path, exc)
        ) from exc

    if stat.S_ISLNK(st_path.st_mode):
        raise PermissionError(
            "refusing to acquire lock: {0} is a symlink".format(lock_path)
        )
    if not stat.S_ISREG(st_path.st_mode):
        raise PermissionError(
            "refusing to acquire lock: {0} is not a regular file "
            "(mode={1})".format(lock_path, oct(st_path.st_mode))
        )

    if fd is not None:
        try:
            st_fd = os.fstat(fd)
        except OSError as exc:
            raise PermissionError(
                "could not fstat lock fd for {0}: {1}".format(lock_path, exc)
            ) from exc
        if not stat.S_ISREG(st_fd.st_mode):
            raise PermissionError(
                "refusing to acquire lock: {0} fd does not point "
                "to a regular file".format(lock_path)
            )
        path_identity = (
            st_path.st_dev,
            st_path.st_ino,
            stat.S_IFMT(st_path.st_mode),
        )
        fd_identity = (
            st_fd.st_dev,
            st_fd.st_ino,
            stat.S_IFMT(st_fd.st_mode),
        )
        if path_identity != fd_identity:
            raise PermissionError(
                "lock inode changed between lstat and fstat on {0}: "
                "path={1} fd={2}".format(lock_path, path_identity, fd_identity)
            )


def _resolve_lock_parent(canonical):
    """Walk up from ``canonical`` until OUTSIDE every skills root."""
    try:
        from agent.skill_utils import get_all_skills_dirs
    except Exception:
        return Path(canonical).parent

    try:
        resolved_roots = [r.resolve(strict=False) for r in get_all_skills_dirs()]
    except Exception:
        resolved_roots = []

    lock_parent = Path(canonical).parent
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


def normalized_name_lock_target(normalized_name, *, anchor=None):
    """Pure derivation of the global normalized-name lock path.

    Contract:

      * deterministic -- same input -> same path on every call;
      * side-effect free -- does not touch the filesystem;
      * key depends ONLY on ``normalized_name`` and
        NORMALIZATION_VERSION;
      * the parent is walked up from the ``anchor`` (or, when no
        anchor is given, from a synthetic skills root) until it is
        outside every known skills root, so the lock file lives in
        the same sibling namespace as the per-target mutation lock;
      * the filename carries the full SHA-256 digest (64 hex chars)
        so the raw name never appears on disk and the path is
        collision-free;
      * the namespace is SEPARATE from per-target mutation locks
        (.hermes-skill-name-mutex- vs .hermes-skill-mutex-).

    The optional ``anchor`` argument lets the caller (typically
    :func:`live_skill_publish_guard`) supply the canonical skill
    target path so the lock_parent resolution starts from the real
    publisher target rather than a synthetic skills-root path. This
    is the standard call path; the no-anchor form is retained only
    for tests that need a side-effect-free derivation.
    """
    if not isinstance(normalized_name, str) or not normalized_name:
        raise ValueError(
            "normalized_name_lock_target requires a non-empty validated name"
        )
    if canonical_normalize_skill_name(normalized_name) is None:
        raise ValueError(
            "normalized_name_lock_target requires an L1-validated name"
        )
    if anchor is not None:
        anchor_path = Path(anchor).resolve(strict=False)
    else:
        # Side-effect-free derivation: anchor on a synthetic skill
        # path so the resolution walks outside any skills root.
        anchor_path = Path("/skills") / normalized_name
    lock_parent = _resolve_lock_parent(anchor_path)
    digest = _hashlib.sha256(
        (NORMALIZATION_VERSION + "\0" + normalized_name).encode("utf-8")
    ).hexdigest()
    return lock_parent / (".hermes-skill-name-mutex-" + digest + ".lock")


# ---------------------------------------------------------------------------
# Low-level validated interprocess primitive.
#
# This is a thin specialization of the same logic that currently lives in
# tools.skill_manager_tool.py._skill_mutation_process_lock. The two
# implementations are INTENTIONALLY behaviorally equivalent; Block 2
# migrates the per-publisher callers onto this public helper and removes
# the private one in skill_manager_tool.py.
#
# The helper takes an explicit lock path so it can be reused for both
# the per-target lock (digest of the canonical skill path) and the
# global-name lock (digest of NORMALIZATION_VERSION + name).
# ---------------------------------------------------------------------------
@contextmanager
def _acquire_lock_at_path(*, lock_path, canonical_skill_path):
    platform_name = _platform_name()
    lock_path = Path(lock_path)
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _raise_lock_acquire_failure(
            canonical_skill_path=canonical_skill_path,
            lock_path=lock_path,
            platform=platform_name,
            lock_failure_stage=LOCK_FAILURE_STAGE_PATH_RESOLUTION,
            cause=exc,
            safe_to_retry=False,
        )

    if _IS_POSIX:
        if _fcntl is None:
            _raise_lock_acquire_failure(
                canonical_skill_path=canonical_skill_path,
                lock_path=lock_path,
                platform=platform_name,
                lock_failure_stage=LOCK_FAILURE_STAGE_PATH_RESOLUTION,
                cause=PermissionError("fcntl module is unavailable on POSIX"),
                safe_to_retry=False,
            )
        fd = None
        release_error = None
        close_error = None
        try:
            open_flags = os.O_CREAT | os.O_RDWR
            if _O_NOFOLLOW:
                open_flags |= _O_NOFOLLOW
            if os.path.lexists(str(lock_path)):
                try:
                    pre_st = os.lstat(str(lock_path))
                except OSError as exc:
                    _raise_lock_acquire_failure(
                        canonical_skill_path=canonical_skill_path,
                        lock_path=lock_path,
                        platform=platform_name,
                        lock_failure_stage=LOCK_FAILURE_STAGE_PARENT_OPEN,
                        cause=exc,
                        safe_to_retry=False,
                    )
                if stat.S_ISLNK(pre_st.st_mode):
                    msg_symlink = "refusing to acquire lock: " + str(lock_path) + " is a symlink"
                    _raise_lock_acquire_failure(
                        canonical_skill_path=canonical_skill_path,
                        lock_path=lock_path,
                        platform=platform_name,
                        lock_failure_stage=LOCK_FAILURE_STAGE_IDENTITY_VALIDATION,
                        cause=None,
                        safe_to_retry=False,
                        msg=msg_symlink,
                    )
                if not stat.S_ISREG(pre_st.st_mode):
                    mode_str = oct(pre_st.st_mode)
                    msg_reg = (
                        "refusing to acquire lock: " + str(lock_path)
                        + " is not a regular file (mode=" + mode_str + ")"
                    )
                    _raise_lock_acquire_failure(
                        canonical_skill_path=canonical_skill_path,
                        lock_path=lock_path,
                        platform=platform_name,
                        lock_failure_stage=LOCK_FAILURE_STAGE_IDENTITY_VALIDATION,
                        cause=None,
                        safe_to_retry=False,
                        msg=msg_reg,
                    )
            try:
                fd = os.open(str(lock_path), open_flags, 0o600)
            except OSError as exc:
                _raise_lock_acquire_failure(
                    canonical_skill_path=canonical_skill_path,
                    lock_path=lock_path,
                    platform=platform_name,
                    lock_failure_stage=LOCK_FAILURE_STAGE_PARENT_OPEN,
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
                    canonical_skill_path=canonical_skill_path,
                    lock_path=lock_path,
                    platform=platform_name,
                    lock_failure_stage=LOCK_FAILURE_STAGE_IDENTITY_VALIDATION,
                    cause=exc,
                    safe_to_retry=False,
                )
            try:
                _fcntl.flock(fd, _fcntl.LOCK_EX)
            except OSError as exc:
                errno_val = getattr(exc, "errno", None)
                stage = (
                    LOCK_FAILURE_STAGE_CONTENTION
                    if errno_val in (_errno.EWOULDBLOCK, _errno.EAGAIN)
                    else LOCK_FAILURE_STAGE_PRIMITIVE_ACQUIRE
                )
                try:
                    os.close(fd)
                finally:
                    fd = None
                _raise_lock_acquire_failure(
                    canonical_skill_path=canonical_skill_path,
                    lock_path=lock_path,
                    platform=platform_name,
                    lock_failure_stage=stage,
                    cause=exc,
                    safe_to_retry=(stage == LOCK_FAILURE_STAGE_CONTENTION),
                )
            yield
        finally:
            if fd is not None:
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
                    raise SkillMutationLockReleaseFailure(
                        canonical_skill_path=canonical_skill_path,
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
                canonical_skill_path=canonical_skill_path,
                lock_path=lock_path,
                platform=platform_name,
                lock_failure_stage=LOCK_FAILURE_STAGE_PATH_RESOLUTION,
                cause=PermissionError("msvcrt module is unavailable on Windows"),
                safe_to_retry=False,
            )
        contract_reason = _validate_msvcrt_contract()
        if contract_reason is not None:
            msg_msvcrt = (
                "msvcrt contract is invalid on Windows ("
                + str(contract_reason) + ")"
            )
            _raise_lock_acquire_failure(
                canonical_skill_path=canonical_skill_path,
                lock_path=lock_path,
                platform=platform_name,
                lock_failure_stage=LOCK_FAILURE_STAGE_IDENTITY_VALIDATION,
                cause=PermissionError(msg_msvcrt),
                safe_to_retry=False,
            )
        fd = None
        release_error = None
        close_error = None
        try:
            if os.path.lexists(str(lock_path)):
                try:
                    pre_st = os.lstat(str(lock_path))
                except OSError as exc:
                    _raise_lock_acquire_failure(
                        canonical_skill_path=canonical_skill_path,
                        lock_path=lock_path,
                        platform=platform_name,
                        lock_failure_stage=LOCK_FAILURE_STAGE_PARENT_OPEN,
                        cause=exc,
                        safe_to_retry=False,
                    )
                if stat.S_ISLNK(pre_st.st_mode):
                    msg_symlink_win = (
                        "refusing to acquire lock: " + str(lock_path)
                        + " is a symlink/junction"
                    )
                    _raise_lock_acquire_failure(
                        canonical_skill_path=canonical_skill_path,
                        lock_path=lock_path,
                        platform=platform_name,
                        lock_failure_stage=LOCK_FAILURE_STAGE_IDENTITY_VALIDATION,
                        cause=None,
                        safe_to_retry=False,
                        msg=msg_symlink_win,
                    )
                if not stat.S_ISREG(pre_st.st_mode):
                    mode_str_win = oct(pre_st.st_mode)
                    msg_reg_win = (
                        "refusing to acquire lock: " + str(lock_path)
                        + " is not a regular file (mode=" + mode_str_win + ")"
                    )
                    _raise_lock_acquire_failure(
                        canonical_skill_path=canonical_skill_path,
                        lock_path=lock_path,
                        platform=platform_name,
                        lock_failure_stage=LOCK_FAILURE_STAGE_IDENTITY_VALIDATION,
                        cause=None,
                        safe_to_retry=False,
                        msg=msg_reg_win,
                    )
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
            except OSError as exc:
                _raise_lock_acquire_failure(
                    canonical_skill_path=canonical_skill_path,
                    lock_path=lock_path,
                    platform=platform_name,
                    lock_failure_stage=LOCK_FAILURE_STAGE_PARENT_OPEN,
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
                    canonical_skill_path=canonical_skill_path,
                    lock_path=lock_path,
                    platform=platform_name,
                    lock_failure_stage=LOCK_FAILURE_STAGE_IDENTITY_VALIDATION,
                    cause=exc,
                    safe_to_retry=False,
                )
            try:
                size = os.fstat(fd).st_size
                if size == 0:
                    os.write(fd, b"\x00")
                os.lseek(fd, 0, os.SEEK_SET)
            except OSError as exc:
                try:
                    os.close(fd)
                finally:
                    fd = None
                _raise_lock_acquire_failure(
                    canonical_skill_path=canonical_skill_path,
                    lock_path=lock_path,
                    platform=platform_name,
                    lock_failure_stage=LOCK_FAILURE_STAGE_PARENT_OPEN,
                    cause=exc,
                    safe_to_retry=False,
                )
            try:
                _msvcrt.locking(fd, _msvcrt.LK_NBLCK, 1)
            except (OSError, PermissionError):
                try:
                    _msvcrt.locking(fd, _msvcrt.LK_LOCK, 1)
                except (OSError, PermissionError) as exc2:
                    try:
                        os.close(fd)
                    finally:
                        fd = None
                    _raise_lock_acquire_failure(
                        canonical_skill_path=canonical_skill_path,
                        lock_path=lock_path,
                        platform=platform_name,
                        lock_failure_stage=LOCK_FAILURE_STAGE_PRIMITIVE_ACQUIRE,
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
                    raise SkillMutationLockReleaseFailure(
                        canonical_skill_path=canonical_skill_path,
                        lock_path=lock_path,
                        platform=platform_name,
                        release_error=release_error,
                        close_error=close_error,
                        live_mutation_committed=False,
                    )
        return

    _raise_lock_acquire_failure(
        canonical_skill_path=canonical_skill_path,
        lock_path=lock_path,
        platform=platform_name,
        lock_failure_stage=LOCK_FAILURE_STAGE_PATH_RESOLUTION,
        cause=PermissionError("unsupported platform os.name=" + repr(os.name)),
        safe_to_retry=False,
    )


# ---------------------------------------------------------------------------
# Discovery (delegates to the neutral agent.skill_utils helper).
# ---------------------------------------------------------------------------
def find_skill(name):
    """Find a skill by name across every skills root.

    Returns ``{"path": Path}`` or ``None``.
    """
    try:
        from agent.skill_utils import get_all_skills_dirs, is_excluded_skill_path
    except Exception:
        return None
    for skills_dir in get_all_skills_dirs():
        if not skills_dir.exists():
            continue
        for skill_md in skills_dir.rglob("SKILL.md"):
            if is_excluded_skill_path(skill_md):
                continue
            if skill_md.parent.name == name:
                return {"path": skill_md.parent}
    return None


def is_external_path(path):
    """Return True if ``path`` lives under any configured external_dir."""
    try:
        from agent.skill_utils import is_external_skill_path
    except Exception:
        return False
    try:
        return bool(is_external_skill_path(Path(path)))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Replacement policy (representation only -- no real publish in Block 1).
# ---------------------------------------------------------------------------
ReplacementPolicy = Literal["new_only", "replace_same_target", "replace_with_backup"]
_REPLACEMENT_POLICIES = frozenset(
    {"new_only", "replace_same_target", "replace_with_backup"}
)


def validate_replacement_policy(value):
    """Return ``value`` if it is a valid replacement policy; else ``ValueError``."""
    if not isinstance(value, str):
        raise ValueError(
            "replacement_policy must be a string; got "
            + type(value).__name__
        )
    if value not in _REPLACEMENT_POLICIES:
        raise ValueError(
            "replacement_policy must be one of "
            + str(sorted(_REPLACEMENT_POLICIES))
            + "; got "
            + repr(value)
        )
    return value


# ---------------------------------------------------------------------------
# Lock state -- per-invocation, never global.
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class LockState:
    """Discriminating state for the combined global+target acquisition.

    Fields:
      global_entered    True after the global normalized-name lock
                        has been acquired successfully.
      target_entered    True after the per-target mutation lock has
                        been acquired successfully.
      active_lock_scope One of '' (no lock held),
                        'global_normalized_name', or
                        'prospective_target'. Used by the payload
                        formatter to discriminate raw acquisition
                        failures by which lock they originated from.
      active_lock_path  The lock path associated with the current scope.
                        Set eagerly so the structured payload can tag
                        failures correctly even when the global lock
                        fails BEFORE entering the critical section.
                        NEVER overwritten retroactively from a global
                        path to a target path or vice versa.
    """

    global_entered: bool = False
    target_entered: bool = False
    active_lock_scope: str = ""
    active_lock_path: "Optional[Path]" = None


# ---------------------------------------------------------------------------
# Global duplicate scan.
# ---------------------------------------------------------------------------
def global_duplicate_scan(
    name,
    *,
    approved_replacement_target=None,
    exclude_paths=None,
):
    """Discover every live tree path carrying a skill named ``name``.

    Walks every skills root from
    agent.skill_utils.get_all_skills_dirs (which includes the local
    writable skills dir AND configured external_dirs). external paths
    are returned but are NEVER written to by this guard.

    The two layouts are both covered:

      * flat:   <root>/<name>
      * cat:    <root>/<category>/<name>

    Returns the FULL list of conflicting paths.

    exclude_paths lets the caller ignore its own staging directory
    (or any other transient path) so it does not get classified as
    a duplicate of itself.
    """
    if not isinstance(name, str) or not name:
        return []
    try:
        from agent.skill_utils import get_all_skills_dirs
    except Exception:
        return []

    excluded = []
    if exclude_paths:
        for p in exclude_paths:
            try:
                excluded.append(Path(p).resolve(strict=False))
            except Exception:
                excluded.append(Path(p))

    matches = []
    for skills_dir in get_all_skills_dirs():
        if not skills_dir.exists():
            continue
        # flat layout
        candidate_flat = skills_dir / name
        if candidate_flat.exists() and candidate_flat.is_dir():
            try:
                resolved = candidate_flat.resolve(strict=False)
            except Exception:
                resolved = candidate_flat
            if resolved not in excluded:
                matches.append(resolved)
        # category layout: walk one level
        try:
            for entry in skills_dir.iterdir():
                if not entry.is_dir():
                    continue
                candidate_cat = entry / name
                if candidate_cat.exists() and candidate_cat.is_dir():
                    try:
                        resolved = candidate_cat.resolve(strict=False)
                    except Exception:
                        resolved = candidate_cat
                    if resolved not in excluded:
                        matches.append(resolved)
        except Exception:
            continue

    # Sticky note for downstream readers: external_dir matches surface
    # here for visibility (D3), but this helper does not mutate any
    # external path.
    return matches


# ---------------------------------------------------------------------------
# Shared guard -- the combined global+target lock sequence.
# ---------------------------------------------------------------------------
def _raise_duplicate_or_policy_failure(
    *,
    state,
    canonical_skill_path,
    lock_path,
    message,
    same_target_ok,
    replacement_policy,
    conflicts,
):
    """Raise the canonical duplicate / policy refusal exception."""
    raise SkillMutationLockAcquireFailure(
        canonical_skill_path=canonical_skill_path,
        lock_path=lock_path,
        platform=_platform_name(),
        lock_failure_stage=LOCK_FAILURE_STAGE_PRIMITIVE_ACQUIRE,
        cause=ValueError(message),
        safe_to_retry=False,
    )


def _target_lock_path(canonical_skill_path):
    """Derive the per-target mutation lock path (separate namespace)."""
    canonical = Path(canonical_skill_path).resolve(strict=False)
    lock_parent = _resolve_lock_parent(canonical)
    digest = _hashlib.sha256(
        str(canonical).encode("utf-8")
    ).hexdigest()[:16]
    return lock_parent / (".hermes-skill-mutex-" + digest + ".lock")


@contextmanager
def live_skill_publish_guard(
    name,
    *,
    target,
    replacement_policy="new_only",
):
    """Combined global-name + per-target protection for a live publish.

    Sequence (frozen):

      1. canonicalize ``name`` (L1 strict);
      2. derive the global normalized-name lock path;
      3. acquire the global normalized-name lock;
      4. global duplicate scan #1 (with the global lock held);
      5. acquire the per-target mutation lock;
      6. global duplicate scan #2 (with both locks held);
      7. yield ``LockState`` to the caller so publisher-specific
         logic can run under the combined protection;
      8. release the per-target lock;
      9. release the global lock.

    Body or release failures are NOT coerced into acquisition failures.
    The discriminator fields (``global_entered``, ``target_entered``) on
    ``LockState`` are populated so a structured payload can tag the
    failing scope correctly without relying on string equality of
    ``lock_path``.

    No real publish action is integrated in Block 1. The context manager
    only protects the region; the publisher integration arrives in
    Block 2 onwards.
    """
    # 1. canonicalize
    canonical = canonical_normalize_skill_name(name)
    if canonical is None:
        raise ValueError(
            "refusing to publish: name " + repr(name) + " is not L1-valid"
        )

    replacement_policy = validate_replacement_policy(replacement_policy)

    canonical_skill_path = target
    state = LockState()

    # 2. derive the global lock path eagerly. The anchor argument
    # routes the resolution through the real publisher target so
    # the lock_parent sits outside every known skills root AND on a
    # writable filesystem (no /skills synthetic root).
    global_lock_path = normalized_name_lock_target(
        canonical, anchor=canonical_skill_path
    )

    # 3. acquire the global lock. A raw PermissionError here is an
    # acquisition failure on the GLOBAL lock.
    state.active_lock_scope = "global_normalized_name"
    state.active_lock_path = global_lock_path
    # Late-binding lookups so test suites can monkey-patch the
    # underlying helpers without invalidating an import-time
    # reference. The names are resolved at call time through the
    # module's namespace.
    _acquire = globals()["_acquire_lock_at_path"]

    @contextmanager
    def _classified_global_lock_context():
        # Acquire the global normalized-name lock, classify any
        # PermissionError raised either by the factory or by the
        # context-manager __enter__ as a global acquisition failure,
        # and yield into the protected body. PermissionError raised
        # after global_entered has been flipped to True (body or
        # release) propagates verbatim and is NOT reclassified.
        try:
            _global_ctx = _acquire(
                lock_path=global_lock_path,
                canonical_skill_path=canonical_skill_path,
            )
            with _global_ctx:
                state.global_entered = True
                yield
        except PermissionError as exc:
            if state.global_entered:
                # body or release PermissionError: propagate verbatim
                # and do not reclasify as an acquisition failure.
                raise
            raise SkillMutationLockAcquireFailure(
                canonical_skill_path=canonical_skill_path,
                lock_path=global_lock_path,
                platform=_platform_name(),
                lock_failure_stage=LOCK_FAILURE_STAGE_PRIMITIVE_ACQUIRE,
                cause=exc,
                safe_to_retry=False,
            ) from exc

    with _classified_global_lock_context():

            # 4. global duplicate scan #1 (with the global lock held)
            conflicts = global_duplicate_scan(
                canonical,
                approved_replacement_target=canonical_skill_path,
            )
            if len(conflicts) > 1:
                raise SkillMutationLockAcquireFailure(
                    canonical_skill_path=canonical_skill_path,
                    lock_path=global_lock_path,
                    platform=_platform_name(),
                    lock_failure_stage=LOCK_FAILURE_STAGE_PRIMITIVE_ACQUIRE,
                    cause=ValueError(
                        "more than one live skill named "
                        + repr(canonical)
                        + " found across skills roots: "
                        + str(conflicts)
                    ),
                    safe_to_retry=False,
                )
            if len(conflicts) == 1:
                only = conflicts[0]
                try:
                    only_resolved = only.resolve(strict=False)
                except Exception:
                    only_resolved = only
                try:
                    target_resolved = canonical_skill_path.resolve(strict=False)
                except Exception:
                    target_resolved = canonical_skill_path
                same_target = only_resolved == target_resolved
                if not same_target:
                    raise SkillMutationLockAcquireFailure(
                        canonical_skill_path=canonical_skill_path,
                        lock_path=global_lock_path,
                        platform=_platform_name(),
                        lock_failure_stage=LOCK_FAILURE_STAGE_PRIMITIVE_ACQUIRE,
                        cause=ValueError(
                            "duplicate live skill named "
                            + repr(canonical)
                            + " at "
                            + str(only_resolved)
                            + "; existing skill blocks publish under different target "
                            + str(target_resolved)
                        ),
                        safe_to_retry=False,
                    )
                if replacement_policy == "new_only":
                    raise SkillMutationLockAcquireFailure(
                        canonical_skill_path=canonical_skill_path,
                        lock_path=global_lock_path,
                        platform=_platform_name(),
                        lock_failure_stage=LOCK_FAILURE_STAGE_PRIMITIVE_ACQUIRE,
                        cause=ValueError(
                            "skill "
                            + repr(canonical)
                            + " already exists at "
                            + str(target_resolved)
                            + " and replacement_policy="
                            + repr(replacement_policy)
                            + " does not permit replacement"
                        ),
                        safe_to_retry=False,
                    )

            # 5. acquire the per-target mutation lock. Symmetric to the
            # global helper above: a raw PermissionError raised either by
            # the factory (PermissionError raised before __enter__ runs)
            # or by the context-manager __enter__ BEFORE target_entered
            # flips to True is a target acquisition failure and is
            # classified as such. PermissionError raised AFTER
            # target_entered (body or __exit__) propagates verbatim and
            # is NOT reclassified.
            target_lock_path = _target_lock_path(canonical_skill_path)
            state.active_lock_scope = "prospective_target"
            state.active_lock_path = target_lock_path

            @contextmanager
            def _classified_target_lock_context():
                _target_ctx = _acquire(
                    lock_path=target_lock_path,
                    canonical_skill_path=canonical_skill_path,
                )
                try:
                    with _target_ctx:
                        state.target_entered = True
                        yield
                except PermissionError as exc:
                    if state.target_entered:
                        # body or release PermissionError: propagate
                        # verbatim and do not reclassify as an
                        # acquisition failure.
                        raise
                    raise SkillMutationLockAcquireFailure(
                        canonical_skill_path=canonical_skill_path,
                        lock_path=target_lock_path,
                        platform=_platform_name(),
                        lock_failure_stage=LOCK_FAILURE_STAGE_PRIMITIVE_ACQUIRE,
                        cause=exc,
                        safe_to_retry=False,
                    ) from exc

            with _classified_target_lock_context():

                # 6. global duplicate scan #2 (with both locks held).
                conflicts2 = global_duplicate_scan(
                    canonical,
                    approved_replacement_target=canonical_skill_path,
                )
                if len(conflicts2) > 1:
                    raise SkillMutationLockAcquireFailure(
                        canonical_skill_path=canonical_skill_path,
                        lock_path=target_lock_path,
                        platform=_platform_name(),
                        lock_failure_stage=LOCK_FAILURE_STAGE_PRIMITIVE_ACQUIRE,
                        cause=ValueError(
                            "more than one live skill named "
                            + repr(canonical)
                            + " appeared during scan #2: "
                            + str(conflicts2)
                        ),
                        safe_to_retry=False,
                    )
                if len(conflicts2) == 1:
                    only = conflicts2[0]
                    try:
                        only_resolved = only.resolve(strict=False)
                    except Exception:
                        only_resolved = only
                    try:
                        target_resolved = canonical_skill_path.resolve(strict=False)
                    except Exception:
                        target_resolved = canonical_skill_path
                    same_target = only_resolved == target_resolved
                    if not same_target:
                        raise SkillMutationLockAcquireFailure(
                            canonical_skill_path=canonical_skill_path,
                            lock_path=target_lock_path,
                            platform=_platform_name(),
                            lock_failure_stage=LOCK_FAILURE_STAGE_PRIMITIVE_ACQUIRE,
                            cause=ValueError(
                                "duplicate live skill named "
                                + repr(canonical)
                                + " appeared during scan #2 at "
                                + str(only_resolved)
                            ),
                            safe_to_retry=False,
                        )
                    if replacement_policy == "new_only":
                        raise SkillMutationLockAcquireFailure(
                            canonical_skill_path=canonical_skill_path,
                            lock_path=target_lock_path,
                            platform=_platform_name(),
                            lock_failure_stage=LOCK_FAILURE_STAGE_PRIMITIVE_ACQUIRE,
                            cause=ValueError(
                                "skill "
                                + repr(canonical)
                                + " already exists at "
                                + str(target_resolved)
                                + " and replacement_policy="
                                + repr(replacement_policy)
                                + " does not permit replacement"
                            ),
                            safe_to_retry=False,
                        )

                # 7. yield under combined protection. Body failures are
                # NOT coerced into acquisition failures.
                yield state

            # 8. release target lock (handled by _target_ctx exit).
            state.target_entered = False
            state.active_lock_scope = "global_normalized_name"
            state.active_lock_path = global_lock_path

    # 9. release global lock (handled by _global_ctx exit).
    state.global_entered = False
    state.active_lock_scope = ""
    state.active_lock_path = None
class PublishGuard:
    """OO-style wrapper around :func:`live_skill_publish_guard`.

    Block 1 only validates that this class is constructible and
    delegates correctly; Block 2 onwards wires real publishers into
    it.
    """

    __slots__ = ("_name", "_target", "_replacement_policy", "_cm", "_state")

    def __init__(self, name, *, target, replacement_policy="new_only"):
        self._name = name
        self._target = Path(target)
        self._replacement_policy = replacement_policy
        self._cm = None
        self._state = None

    def __enter__(self):
        self._cm = live_skill_publish_guard(
            self._name,
            target=self._target,
            replacement_policy=self._replacement_policy,
        )
        self._state = self._cm.__enter__()
        return self._state

    def __exit__(self, exc_type, exc, tb):
        if self._cm is None:
            return False
        result = self._cm.__exit__(exc_type, exc, tb)
        self._cm = None
        self._state = None
        return result
