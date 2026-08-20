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
        secondary_failures=None,
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
        self.secondary_failures = list(secondary_failures or [])


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
        from agent.skill_utils import (
            get_all_skills_dirs,
            is_excluded_skill_path,
        )
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
                # Archived/cache/support trees are not live publishers.
                # Use the same exclusion authority as normal skill discovery.
                try:
                    _excluded_candidate = is_excluded_skill_path(
                        candidate_cat / "SKILL.md",
                        root=skills_dir,
                    )
                except Exception:
                    # Classification failure must not hide a possible
                    # live duplicate.
                    _excluded_candidate = False
                if _excluded_candidate:
                    continue
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
    identity = _path_lock_identity(canonical)
    digest = _hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()[:16]
    return lock_parent / (".hermes-skill-mutex-" + digest + ".lock")


def _canonical_path(path):
    """Resolve a path with the module's existing non-strict convention."""
    try:
        return Path(path).resolve(strict=False)
    except Exception:
        return Path(path)


def _normalize_path_identity_text(path):
    """Return the platform lock-identity spelling for a path string.

    Windows path identity is case-insensitive; POSIX path identity is not.
    Tests may simulate Windows by patching ``_IS_WINDOWS`` on a POSIX host,
    so this helper performs the fold directly instead of relying on the
    host implementation of ``os.path.normcase``.
    """
    text = str(path).replace(os.sep, "/")
    if os.altsep:
        text = text.replace(os.altsep, "/")
    if _IS_WINDOWS:
        text = text.lower()
    return text


def _path_lock_identity(path):
    """Canonical mutation-lock identity shared by dedupe, sort, and hashing."""
    return _normalize_path_identity_text(_canonical_path(path))


def _path_live_identity(path):
    """Lexical/live pathname identity that does not follow final symlinks."""
    return _normalize_path_identity_text(Path(path))


def _dedupe_lock_identity_paths(paths):
    seen = set()
    result = []
    for path in paths:
        canonical = _canonical_path(path)
        key = _path_lock_identity(canonical)
        if key in seen:
            continue
        seen.add(key)
        result.append(canonical)
    return result


def _dedupe_live_paths(paths):
    seen = set()
    result = []
    for path in paths:
        live = Path(path)
        key = _path_live_identity(live)
        if key in seen:
            continue
        seen.add(key)
        result.append(live)
    return result


def _dedupe_canonical_paths(paths):
    return _dedupe_lock_identity_paths(paths)


def _path_sort_key(path):
    """Stable path-content lock ordering key.

    The key is based on Path.resolve(strict=False), os.path.normcase,
    slash-normalized text, and the resolved path text as a deterministic
    tie-breaker. It never depends on object identity.
    """
    identity = _path_lock_identity(path)
    return (identity, identity)


def _read_skill_frontmatter_name(skill_dir):
    """Return (name, error) for SKILL.md frontmatter in a skill directory."""
    skill_md = Path(skill_dir) / "SKILL.md"
    if not skill_md.exists():
        return None, None
    try:
        text = skill_md.read_text(encoding="utf-8")
    except Exception as exc:
        return None, exc
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None, None
    end = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end = idx
            break
    if end is None:
        return None, ValueError("unterminated SKILL.md frontmatter")
    for line in lines[1:end]:
        stripped = line.strip()
        if stripped.startswith("name:"):
            raw = stripped.split(":", 1)[1].strip()
            if (raw.startswith('"') and raw.endswith('"')) or (
                raw.startswith("'") and raw.endswith("'")
            ):
                raw = raw[1:-1]
            return raw, None
    return None, None


def _is_path_redirect(path):
    """True when ``path`` is a symlink or supported Windows junction."""
    path = Path(path)
    try:
        return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())
    except OSError:
        return False


def _path_identity(path, *, require_frontmatter_identity=False):
    """Capture filesystem identity for an approved existing repair path."""
    path = Path(path)
    try:
        st = path.lstat()
    except OSError as exc:
        raise ValueError("approved repair path disappeared: {0}".format(path)) from exc
    if stat.S_ISLNK(st.st_mode) or _is_path_redirect(path):
        raise ValueError(
            "approved repair path became a symlink/junction: {0}".format(path)
        )
    if not stat.S_ISDIR(st.st_mode):
        raise ValueError(
            "approved repair path is not a directory: {0}".format(path)
        )
    frontmatter_name = None
    if require_frontmatter_identity:
        frontmatter_name, error = _read_skill_frontmatter_name(path)
        if error is not None:
            raise ValueError(
                "could not re-read approved repair path identity at {0}: {1}".format(
                    path, error
                )
            ) from error
    return (
        st.st_dev,
        st.st_ino,
        stat.S_IFMT(st.st_mode),
        frontmatter_name if require_frontmatter_identity else None,
    )


def _capture_approved_path_identity(
    path,
    *,
    canonical_skill_path,
    lock_path,
    require_frontmatter_identity=False,
):
    try:
        return _path_identity(
            path,
            require_frontmatter_identity=require_frontmatter_identity,
        )
    except (ValueError, OSError) as exc:
        _raise_repair_failure(
            canonical_skill_path=canonical_skill_path,
            lock_path=lock_path,
            message="approved repair path identity could not be captured: {0}".format(path),
            cause=exc,
        )


def _maintenance_duplicate_scan(name, *, identity_names=()):
    """Maintenance-only same-identity scan for live skill repair.

    This intentionally does NOT change ordinary ``global_duplicate_scan``
    semantics. Repair identity may come from directory basename,
    SKILL.md frontmatter ``name``, or explicit ``identity_names`` aliases.
    The aliases are scan identity only: callers must canonicalize aliases
    to one canonical ``name`` before entering the guard, and aliases never
    create additional global lock keys.
    """
    identities = {name}
    for identity in identity_names or ():
        if not isinstance(identity, str) or not identity:
            raise ValueError("identity_names entries must be non-empty strings")
        identities.add(identity)

    try:
        from agent.skill_utils import get_all_skills_dirs, is_excluded_skill_path
    except Exception:
        return []

    matches = []
    by_key = set()
    for skills_dir in get_all_skills_dirs():
        if not skills_dir.exists():
            continue
        candidates = []
        # Flat layout candidate.
        try:
            for entry in skills_dir.iterdir():
                if entry.is_dir():
                    candidates.append(entry)
        except Exception:
            continue
        # One-level category layout candidates.
        for entry in list(candidates):
            try:
                for child in entry.iterdir():
                    if child.is_dir():
                        candidates.append(child)
            except Exception:
                continue

        for candidate in candidates:
            try:
                if is_excluded_skill_path(candidate / "SKILL.md", root=skills_dir):
                    continue
            except Exception:
                pass
            basis = []
            if candidate.name in identities:
                basis.append("basename")
            frontmatter_name, error = _read_skill_frontmatter_name(candidate)
            if error is not None and candidate.name in identities:
                raise ValueError(
                    "malformed/unreadable identity state at approved candidate {0}: {1}".format(
                        candidate, error
                    )
                ) from error
            if frontmatter_name in identities:
                basis.append("frontmatter")
            if not basis:
                continue
            key = _path_live_identity(candidate)
            if key in by_key:
                continue
            by_key.add(key)
            matches.append({"path": candidate, "basis": tuple(basis)})
    return matches


def _raise_repair_failure(*, canonical_skill_path, lock_path, message, cause=None):
    if cause is None:
        cause = ValueError(message)
    raise SkillMutationLockAcquireFailure(
        canonical_skill_path=canonical_skill_path,
        lock_path=lock_path,
        platform=_platform_name(),
        lock_failure_stage=LOCK_FAILURE_STAGE_PRIMITIVE_ACQUIRE,
        cause=cause,
        safe_to_retry=False,
    )


def _validate_repair_scan_exact(
    *,
    canonical_skill_path,
    lock_path,
    approved_paths,
    scan_records,
    label,
):
    observed = {_path_live_identity(record["path"]) for record in scan_records}
    approved = {_path_live_identity(path) for path in approved_paths}
    if observed != approved:
        unexpected = [str(record["path"]) for record in scan_records if _path_live_identity(record["path"]) not in approved]
        missing = [str(path) for path in approved_paths if _path_live_identity(path) not in observed]
        _raise_repair_failure(
            canonical_skill_path=canonical_skill_path,
            lock_path=lock_path,
            message=(
                "repair approved set mismatch during {0}; unexpected={1}; missing={2}".format(
                    label, unexpected, missing
                )
            ),
        )


def _enter_classified_lock(ctx, *, canonical_skill_path, lock_path):
    try:
        ctx.__enter__()
    except SkillMutationLockAcquireFailure:
        raise
    except PermissionError as exc:
        raise SkillMutationLockAcquireFailure(
            canonical_skill_path=canonical_skill_path,
            lock_path=lock_path,
            platform=_platform_name(),
            lock_failure_stage=LOCK_FAILURE_STAGE_PRIMITIVE_ACQUIRE,
            cause=exc,
            safe_to_retry=False,
        ) from exc


def _release_lock_contexts(records, *, exc_type=None, exc=None, tb=None):
    first_release_failure = None
    secondary_release_failures = []
    for record in reversed(records):
        try:
            record["ctx"].__exit__(exc_type, exc, tb)
        except SkillMutationLockReleaseFailure as release_exc:
            if exc is not None:
                release_exc.__context__ = exc
            if first_release_failure is None:
                first_release_failure = release_exc
            else:
                secondary_release_failures.append(release_exc)
        except Exception as release_exc:
            wrapped = SkillMutationLockReleaseFailure(
                canonical_skill_path=record["canonical_skill_path"],
                lock_path=record["lock_path"],
                platform=_platform_name(),
                release_error=release_exc,
                close_error=None,
                live_mutation_committed=False,
            )
            if exc is not None:
                wrapped.__context__ = exc
            if first_release_failure is None:
                first_release_failure = wrapped
            else:
                secondary_release_failures.append(wrapped)
    if first_release_failure is not None:
        first_release_failure.secondary_failures.extend(secondary_release_failures)
        raise first_release_failure


@contextmanager
def _live_skill_transaction_guard(
    name,
    *,
    target,
    replacement_policy="new_only",
    mode="publish",
    approved_existing_paths=None,
    mutation_paths=None,
    identity_names=(),
):
    """Shared global-name transaction guard for publish and repair.

    The canonical normalized-name lock is always the first and only global
    serialization key. ``identity_names`` is used only by repair scanning;
    it MUST NOT create extra global lock keys. Callers are responsible for
    canonicalizing aliases to a single canonical ``name`` before entry.
    """
    canonical = canonical_normalize_skill_name(name)
    if canonical is None:
        raise ValueError(
            "refusing to publish: name " + repr(name) + " is not L1-valid"
        )

    canonical_skill_path = _canonical_path(target)
    state = LockState()
    global_lock_path = normalized_name_lock_target(
        canonical, anchor=canonical_skill_path
    )
    _acquire = globals()["_acquire_lock_at_path"]

    state.active_lock_scope = "global_normalized_name"
    state.active_lock_path = global_lock_path
    try:
        global_ctx = _acquire(
            lock_path=global_lock_path,
            canonical_skill_path=canonical_skill_path,
        )
    except SkillMutationLockAcquireFailure:
        raise
    except PermissionError as exc:
        raise SkillMutationLockAcquireFailure(
            canonical_skill_path=canonical_skill_path,
            lock_path=global_lock_path,
            platform=_platform_name(),
            lock_failure_stage=LOCK_FAILURE_STAGE_PRIMITIVE_ACQUIRE,
            cause=exc,
            safe_to_retry=False,
        ) from exc
    body_exc = None
    try:
        _enter_classified_lock(
            global_ctx,
            canonical_skill_path=canonical_skill_path,
            lock_path=global_lock_path,
        )
        state.global_entered = True
        if mode == "publish":
            replacement_policy = validate_replacement_policy(replacement_policy)
            yield from _live_skill_publish_locked(
                canonical=canonical,
                canonical_skill_path=canonical_skill_path,
                replacement_policy=replacement_policy,
                global_lock_path=global_lock_path,
                state=state,
                acquire=_acquire,
            )
        elif mode == "repair":
            yield from _live_skill_repair_locked(
                canonical=canonical,
                canonical_skill_path=canonical_skill_path,
                approved_existing_paths=approved_existing_paths,
                mutation_paths=mutation_paths,
                identity_names=identity_names,
                global_lock_path=global_lock_path,
                state=state,
                acquire=_acquire,
            )
        elif mode == "delete":
            # Deletion is its own live-skill trust boundary: callers must
            # perform deletion-specific provenance/content checks inside this
            # guard, but the generic same-name + mutation-path serialization is
            # intentionally shared with the repair primitive so A1G lock
            # ordering and identity failure behaviour stay identical.
            yield from _live_skill_repair_locked(
                canonical=canonical,
                canonical_skill_path=canonical_skill_path,
                approved_existing_paths=approved_existing_paths,
                mutation_paths=mutation_paths,
                identity_names=identity_names,
                global_lock_path=global_lock_path,
                state=state,
                acquire=_acquire,
            )
        else:
            raise ValueError("unknown live skill transaction mode: {0}".format(mode))
    except BaseException as exc:
        body_exc = exc
        if state.global_entered:
            try:
                global_ctx.__exit__(type(exc), exc, exc.__traceback__)
            except SkillMutationLockReleaseFailure as release_exc:
                release_exc.__context__ = exc
                raise
            except Exception as release_exc:
                wrapped = SkillMutationLockReleaseFailure(
                    canonical_skill_path=canonical_skill_path,
                    lock_path=global_lock_path,
                    platform=_platform_name(),
                    release_error=release_exc,
                    close_error=None,
                    live_mutation_committed=False,
                )
                wrapped.__context__ = exc
                raise wrapped from release_exc
        raise
    finally:
        try:
            if state.global_entered and body_exc is None:
                global_ctx.__exit__(None, None, None)
        finally:
            state.global_entered = False
            state.active_lock_scope = ""
            state.active_lock_path = None


def _live_skill_publish_locked(
    *,
    canonical,
    canonical_skill_path,
    replacement_policy,
    global_lock_path,
    state,
    acquire,
):
    conflicts = global_duplicate_scan(
        canonical,
        approved_replacement_target=canonical_skill_path,
    )
    _validate_publish_conflicts(
        conflicts,
        canonical=canonical,
        canonical_skill_path=canonical_skill_path,
        lock_path=global_lock_path,
        replacement_policy=replacement_policy,
        scan_label="across skills roots",
    )

    target_lock_path = _target_lock_path(canonical_skill_path)
    state.active_lock_scope = "prospective_target"
    state.active_lock_path = target_lock_path
    try:
        target_ctx = acquire(
            lock_path=target_lock_path,
            canonical_skill_path=canonical_skill_path,
        )
    except SkillMutationLockAcquireFailure:
        raise
    except PermissionError as exc:
        raise SkillMutationLockAcquireFailure(
            canonical_skill_path=canonical_skill_path,
            lock_path=target_lock_path,
            platform=_platform_name(),
            lock_failure_stage=LOCK_FAILURE_STAGE_PRIMITIVE_ACQUIRE,
            cause=exc,
            safe_to_retry=False,
        ) from exc
    try:
        _enter_classified_lock(
            target_ctx,
            canonical_skill_path=canonical_skill_path,
            lock_path=target_lock_path,
        )
        state.target_entered = True
        conflicts2 = global_duplicate_scan(
            canonical,
            approved_replacement_target=canonical_skill_path,
        )
        _validate_publish_conflicts(
            conflicts2,
            canonical=canonical,
            canonical_skill_path=canonical_skill_path,
            lock_path=target_lock_path,
            replacement_policy=replacement_policy,
            scan_label="appeared during scan #2",
        )
        try:
            yield state
        except BaseException as exc:
            _release_lock_contexts(
                [{"ctx": target_ctx, "canonical_skill_path": canonical_skill_path, "lock_path": target_lock_path}],
                exc_type=type(exc), exc=exc, tb=exc.__traceback__,
            )
            raise
        else:
            _release_lock_contexts(
                [{"ctx": target_ctx, "canonical_skill_path": canonical_skill_path, "lock_path": target_lock_path}],
            )
    finally:
        state.target_entered = False
        state.active_lock_scope = "global_normalized_name"
        state.active_lock_path = global_lock_path


def _validate_publish_conflicts(
    conflicts,
    *,
    canonical,
    canonical_skill_path,
    lock_path,
    replacement_policy,
    scan_label,
):
    if len(conflicts) > 1:
        raise SkillMutationLockAcquireFailure(
            canonical_skill_path=canonical_skill_path,
            lock_path=lock_path,
            platform=_platform_name(),
            lock_failure_stage=LOCK_FAILURE_STAGE_PRIMITIVE_ACQUIRE,
            cause=ValueError(
                "more than one live skill named "
                + repr(canonical)
                + " found "
                + str(scan_label)
                + ": "
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
                lock_path=lock_path,
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
                lock_path=lock_path,
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


def _live_skill_repair_locked(
    *,
    canonical,
    canonical_skill_path,
    approved_existing_paths,
    mutation_paths,
    identity_names,
    global_lock_path,
    state,
    acquire,
):
    if approved_existing_paths is None:
        raise ValueError("approved_existing_paths is mandatory for repair")
    if mutation_paths is None:
        raise ValueError("mutation_paths is mandatory for repair")

    approved_paths = _dedupe_live_paths(approved_existing_paths)
    mutation_lock_paths = _dedupe_lock_identity_paths(mutation_paths)

    scan1 = _maintenance_duplicate_scan(canonical, identity_names=identity_names)
    _validate_repair_scan_exact(
        canonical_skill_path=canonical_skill_path,
        lock_path=global_lock_path,
        approved_paths=approved_paths,
        scan_records=scan1,
        label="scan #1",
    )
    basis_by_key = {_path_live_identity(record["path"]): record["basis"] for record in scan1}
    identities1 = {}
    for path in approved_paths:
        key = _path_live_identity(path)
        identities1[key] = _capture_approved_path_identity(
            path,
            canonical_skill_path=canonical_skill_path,
            lock_path=global_lock_path,
            require_frontmatter_identity="frontmatter" in basis_by_key.get(key, ()),
        )

    ordered_mutation_paths = sorted(mutation_lock_paths, key=_path_sort_key)
    held = []
    try:
        for mutation_path in ordered_mutation_paths:
            lock_path = _target_lock_path(mutation_path)
            try:
                ctx = acquire(lock_path=lock_path, canonical_skill_path=mutation_path)
            except SkillMutationLockAcquireFailure as exc:
                _release_lock_contexts(
                    held,
                    exc_type=type(exc),
                    exc=exc,
                    tb=exc.__traceback__,
                )
                raise
            except PermissionError as exc:
                classified = SkillMutationLockAcquireFailure(
                    canonical_skill_path=mutation_path,
                    lock_path=lock_path,
                    platform=_platform_name(),
                    lock_failure_stage=LOCK_FAILURE_STAGE_PRIMITIVE_ACQUIRE,
                    cause=exc,
                    safe_to_retry=False,
                )
                _release_lock_contexts(
                    held,
                    exc_type=type(classified),
                    exc=classified,
                    tb=classified.__traceback__,
                )
                raise classified from exc
            state.active_lock_scope = "prospective_target"
            state.active_lock_path = lock_path
            try:
                _enter_classified_lock(
                    ctx,
                    canonical_skill_path=mutation_path,
                    lock_path=lock_path,
                )
            except BaseException as exc:
                _release_lock_contexts(held, exc_type=type(exc), exc=exc, tb=exc.__traceback__)
                raise
            held.append({
                "ctx": ctx,
                "canonical_skill_path": mutation_path,
                "lock_path": lock_path,
            })
        state.target_entered = bool(held)

        scan2 = _maintenance_duplicate_scan(canonical, identity_names=identity_names)
        _validate_repair_scan_exact(
            canonical_skill_path=canonical_skill_path,
            lock_path=global_lock_path,
            approved_paths=approved_paths,
            scan_records=scan2,
            label="scan #2",
        )
        basis2_by_key = {_path_live_identity(record["path"]): record["basis"] for record in scan2}
        for path in approved_paths:
            key = _path_live_identity(path)
            identity2 = _capture_approved_path_identity(
                path,
                canonical_skill_path=canonical_skill_path,
                lock_path=global_lock_path,
                require_frontmatter_identity="frontmatter" in basis2_by_key.get(key, ()),
            )
            if identities1.get(key) != identity2:
                _raise_repair_failure(
                    canonical_skill_path=canonical_skill_path,
                    lock_path=global_lock_path,
                    message="approved repair path identity changed before yield: {0}".format(path),
                )

        try:
            yield state
        except BaseException as exc:
            _release_lock_contexts(held, exc_type=type(exc), exc=exc, tb=exc.__traceback__)
            raise
        else:
            _release_lock_contexts(held)
    finally:
        state.target_entered = False
        state.active_lock_scope = "global_normalized_name"
        state.active_lock_path = global_lock_path


@contextmanager
def live_skill_publish_guard(
    name,
    *,
    target,
    replacement_policy="new_only",
):
    """Combined global-name + per-target protection for a live publish.

    This public API remains source-compatible with the original Phase C
    primitive. Normal publishing still uses ordinary ``global_duplicate_scan``
    semantics, the same replacement_policy values, and the same global-name
    then target-lock ordering.
    """
    with _live_skill_transaction_guard(
        name,
        target=target,
        replacement_policy=replacement_policy,
        mode="publish",
    ) as state:
        yield state


@contextmanager
def live_skill_repair_guard(
    name,
    *,
    target,
    approved_existing_paths,
    mutation_paths,
    identity_names=(),
):
    """Guard a same-name live-skill repair transaction.

    ``approved_existing_paths`` is an explicit prerequisite set and is
    revalidated exactly before and after mutation path locks are acquired.
    ``mutation_paths`` is the independent set that receives per-path locks;
    approved external survivors may be omitted from it. In that case the
    global normalized-name lock serializes Hermes-managed same-name operations,
    but it does NOT prevent an external process from mutating the unlocked
    survivor during the repair body. External survivor state is therefore an
    observed/revalidated prerequisite, not a filesystem lock guarantee.

    ``identity_names`` supplies repair scan aliases only. It never creates
    additional global lock keys; callers must canonicalize aliases to the one
    canonical ``name`` before entering this guard.
    """
    with _live_skill_transaction_guard(
        name,
        target=target,
        mode="repair",
        approved_existing_paths=approved_existing_paths,
        mutation_paths=mutation_paths,
        identity_names=identity_names,
    ) as state:
        yield state


@contextmanager
def live_skill_delete_guard(
    name,
    *,
    target,
    approved_existing_paths,
    mutation_paths,
    identity_names=(),
):
    """Guard a same-name live-skill deletion transaction.

    The guard is deliberately narrow: it serializes the canonical skill name,
    locks caller-supplied mutation paths, and fails closed if unexpected
    same-name live state appears before the deletion body. Caller-specific
    provenance/content checks remain with the caller and must be re-run inside
    the guarded body immediately before removing anything.
    """
    with _live_skill_transaction_guard(
        name,
        target=target,
        mode="delete",
        approved_existing_paths=approved_existing_paths,
        mutation_paths=mutation_paths,
        identity_names=identity_names,
    ) as state:
        yield state


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
