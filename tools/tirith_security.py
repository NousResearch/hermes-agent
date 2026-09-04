"""Tirith pre-exec security scanning wrapper.

Runs the tirith binary as a subprocess to scan commands for content-level
threats (homograph URLs, pipe-to-interpreter, terminal injection, etc.).

Exit code is the verdict source of truth:
  0 = allow, 1 = block, 2 = warn

JSON stdout enriches findings/summary but never overrides the verdict.
Operational failures (spawn error, timeout, unknown exit code) respect
the fail_open config setting. Programming errors propagate.

Auto-install: if tirith is not found on PATH or at the configured path,
it is automatically downloaded from GitHub releases to Hermes' private cache.
The download always verifies SHA-256 checksums.  When cosign is available on
PATH, provenance verification (GitHub Actions workflow signature) is also
performed.  If cosign is not installed, the download proceeds with SHA-256
verification only — still secure via HTTPS + checksum, just without supply
chain provenance proof.  Installation runs in a background thread so startup
never blocks.

Managed updates: only Hermes' private Tirith cache is maintained. This is
$HERMES_HOME/bin/tirith for normal installs and a platform-qualified cache for
immutable images whose data volume may also be mounted by a different host OS.
Startup and later scans return that working binary immediately while due update
checks run in a failure-isolated background thread. User-configured, PATH,
package-manager, and development builds are never modified.
Every automatic replacement requires signed release provenance; if cosign or
the release signature is unavailable, Hermes keeps the working binary and
retries later instead of silently falling back to checksum-only verification.
"""

import errno
import functools
import gzip
import hashlib
import io
import json
import logging
import math
import os
import platform
import re
import secrets
import shutil
import stat
import subprocess
import tarfile
import tempfile
import threading
import time
import urllib.request
from contextvars import copy_context
from dataclasses import dataclass, field
from typing import Protocol, TypedDict, cast

from hermes_cli.urllib_security import open_credentialed_url
from hermes_constants import get_hermes_home, is_termux

logger = logging.getLogger(__name__)

_REPO = "sheeki03/tirith"

# Cosign provenance verification — pinned to one stable release tag from the
# specific release workflow. The authenticated tag is also the update version.
_COSIGN_IDENTITY_RE = re.compile(
    rf"^https://github\.com/{re.escape(_REPO)}/\.github/workflows/"
    r"release\.yml@refs/tags/v"
    r"((?:0|[1-9][0-9]{0,9}))\."
    r"((?:0|[1-9][0-9]{0,9}))\."
    r"((?:0|[1-9][0-9]{0,9}))$"
)
_COSIGN_ISSUER = "https://token.actions.githubusercontent.com"

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes"}


def _env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _load_security_config() -> dict:
    """Load security settings from config.yaml, with env var overrides."""
    defaults = {
        "tirith_enabled": True,
        "tirith_path": "tirith",
        "tirith_timeout": 5,
        "tirith_fail_open": True,
    }
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly().get("security", {}) or {}
    except Exception:
        cfg = {}

    return {
        "tirith_enabled": _env_bool("TIRITH_ENABLED", cfg.get("tirith_enabled", defaults["tirith_enabled"])),
        "tirith_path": os.getenv("TIRITH_BIN", cfg.get("tirith_path", defaults["tirith_path"])),
        "tirith_timeout": _env_int("TIRITH_TIMEOUT", cfg.get("tirith_timeout", defaults["tirith_timeout"])),
        "tirith_fail_open": _env_bool("TIRITH_FAIL_OPEN", cfg.get("tirith_fail_open", defaults["tirith_fail_open"])),
    }


# ---------------------------------------------------------------------------
# Auto-install
# ---------------------------------------------------------------------------


def _tirith_auto_install_allowed() -> bool:
    """Return whether Hermes may download Tirith at runtime.

    Tirith is a binary rather than a Python dependency, but it is still a
    runtime install and therefore obeys the same user-facing kill switch as
    every lazy backend dependency.
    """
    from tools.lazy_deps import _allow_lazy_installs

    return _allow_lazy_installs()


_INSTALL_FAILED = False  # sentinel: distinct from "not yet tried"


@dataclass
class _RuntimeState:
    """Process-local Tirith state isolated by profile and scanner config."""

    key: tuple[str, str]
    resolved_path: str | None | bool = None
    install_failure_reason: str = ""
    install_thread: threading.Thread | None = None
    update_thread: threading.Thread | None = None
    crash_count: int = 0
    circuit_open: bool = False
    circuit_opened_at: float | None = None
    circuit_probe_in_flight: bool = False
    circuit_lock: threading.Lock = field(default_factory=threading.Lock)
    install_lock: threading.Lock = field(default_factory=threading.Lock)
    update_schedule_lock: threading.Lock = field(default_factory=threading.Lock)


_runtime_states: dict[tuple[str, str], _RuntimeState] = {}
_runtime_states_lock = threading.Lock()


def _runtime_scope_key(configured_path: str) -> tuple[str, str]:
    home = os.path.normcase(os.path.abspath(os.path.expanduser(str(get_hermes_home()))))
    return home, configured_path


def _runtime_state(configured_path: str) -> _RuntimeState:
    key = _runtime_scope_key(configured_path)
    with _runtime_states_lock:
        state = _runtime_states.get(key)
        if state is None:
            state = _RuntimeState(key=key)
            _runtime_states[key] = state
        return state


def _reset_runtime_states_for_tests() -> None:
    """Reset process-local state; test fixtures only."""
    with _runtime_states_lock:
        _runtime_states.clear()

# Circuit breaker: after _CRASH_LIMIT consecutive spawn/execution failures,
# pause Tirith briefly to prevent agent hangs (#41400), then permit one
# half-open recovery probe. Any recognized Tirith verdict closes the breaker.
#
# Each profile has its own lock. It protects state transitions and the explicit
# in-flight claim, never the subprocess call.
_CRASH_LIMIT = 3
_CIRCUIT_RETRY_SECONDS = 60.0


def _reset_tirith_crash_state(state: _RuntimeState) -> None:
    """Close the circuit after Tirith or its managed install recovers."""
    with state.circuit_lock:
        state.crash_count = 0
        state.circuit_open = False
        state.circuit_opened_at = None
        state.circuit_probe_in_flight = False


def _record_tirith_crash(state: _RuntimeState) -> None:
    """Increment the crash counter and open the circuit breaker if needed."""
    with state.circuit_lock:
        state.crash_count += 1
        if state.crash_count >= _CRASH_LIMIT:
            state.circuit_open = True
            state.circuit_opened_at = time.monotonic()
            logger.warning(
                "tirith circuit breaker opened after %d consecutive failures; "
                "retrying after %.0fs",
                state.crash_count,
                _CIRCUIT_RETRY_SECONDS,
            )


def _circuit_scan_admission(state: _RuntimeState) -> tuple[bool, bool]:
    """Atomically decide whether a scan may run and claim a recovery probe.

    Returns ``(scan_allowed, probe_claimed)``. A single locked decision avoids
    a stale open/closed observation racing with another scan that recovers the
    same profile's breaker.
    """
    with state.circuit_lock:
        if not state.circuit_open:
            return True, False
        if state.circuit_opened_at is None or state.circuit_probe_in_flight:
            return False, False
        now = time.monotonic()
        if now - state.circuit_opened_at < _CIRCUIT_RETRY_SECONDS:
            return False, False
        state.circuit_probe_in_flight = True
        return True, True


def _finish_circuit_probe(state: _RuntimeState) -> None:
    """Release a half-open claim even when resolution or scanning raises."""
    with state.circuit_lock:
        state.circuit_probe_in_flight = False

# Hermes-managed Tirith updates. Tirith 0.4.1 introduced Hermes-aware
# provenance for its self-updater; older managed release binaries are
# bootstrapped through Hermes' checksum-verified installer before future
# updates are delegated to Tirith.
_SELF_UPDATE_MIN_VERSION = (0, 4, 1)
_UPDATE_CHECK_TTL = 86400  # 24 hours after a successful check
_UPDATE_FAILURE_TTL = 3600  # 1 hour after an operational failure
_UPDATE_TIMEOUT = 120
_UPDATE_PROBE_TIMEOUT = 30
_UPDATE_STATE_SCHEMA = 1
_UPDATE_STATE_MAX_BYTES = 4096
_MAX_ARCHIVE_DOWNLOAD_BYTES = 64 * 1024 * 1024
_MAX_METADATA_DOWNLOAD_BYTES = 256 * 1024
_MAX_TIRITH_BINARY_BYTES = 64 * 1024 * 1024
_MAX_RELEASE_ARCHIVE_MEMBERS = 128
_MAX_RELEASE_ARCHIVE_UNPACKED_BYTES = 128 * 1024 * 1024
_UPDATE_SUCCESS_OUTCOMES = frozenset(
    {"bootstrapped", "current", "installed", "skipped", "updated"}
)

class _UpdateState(TypedDict):
    schema_version: int
    checked_at: float
    outcome: str


_in_process_update_states: dict[str, _UpdateState] = {}
_in_process_update_state_lock = threading.Lock()

# Warning de-duplication. The spawn/path warnings live in the hot path —
# without this dedupe set, a Windows install where ``tirith`` isn't on PATH
# (e.g. background install thread still running, or install marked failed)
# spams ``tirith spawn failed: [WinError 2]...`` once per terminal command,
# easily filling errors.log with hundreds of identical lines.
_warned_messages: set[str] = set()
_warned_lock = threading.Lock()


def _warn_once(key: str, message: str, *args) -> None:
    """``logger.warning`` but at-most-once per ``key`` for the process
    lifetime. Used to avoid drowning the log when a fail-open tirith
    misconfiguration fires on every command."""
    with _warned_lock:
        if key in _warned_messages:
            return
        _warned_messages.add(key)
    logger.warning(message, *args)


def _reset_spawn_warning_state() -> None:
    """Clear the warn-once dedupe set. Called when tirith is freshly
    (re)installed so a subsequent failure surfaces again — e.g. user
    deletes the binary mid-session.
    """
    with _warned_lock:
        _warned_messages.clear()

# Disk-persistent failure marker — avoids retry across process restarts
_MARKER_TTL = 86400  # 24 hours


def _get_hermes_home() -> str:
    """Return the Hermes home directory, respecting HERMES_HOME env var."""
    return str(get_hermes_home())


@functools.lru_cache(maxsize=1)
def _uses_image_managed_tirith_root() -> bool:
    """Return whether an immutable image owns this Hermes runtime."""
    try:
        from hermes_cli.image_provenance import read_image_provenance

        # Presence is authoritative even when the immutable marker is invalid;
        # absence preserves the historical path for source/package installs.
        return read_image_provenance() is not None
    except Exception:
        return False


def _managed_tirith_home() -> str:
    """Return the provenance root for Hermes' platform-specific cache."""
    home = _get_hermes_home()
    if _uses_image_managed_tirith_root():
        target = _detect_target()
        if target is not None:
            # The official image's /opt/data is commonly the host ~/.hermes.
            # Never let a Linux container replace or execute a host macOS
            # binary (or let mixed-architecture images fight over one cache).
            return os.path.join(home, ".tirith-managed", target)
    return home


def _managed_tirith_path() -> str:
    """Return the only Tirith binary Hermes is allowed to update."""
    return os.path.join(_managed_tirith_home(), "bin", "tirith")


_DENIED_MANAGED_TIRITH_ROOTS = frozenset(
    {
        "/",
        "/bin",
        "/sbin",
        "/usr",
        "/usr/bin",
        "/usr/sbin",
        "/usr/local",
        "/usr/local/bin",
        "/usr/local/sbin",
        "/System",
        "/Library",
        "/Applications",
        "/opt/homebrew",
        "/home/linuxbrew/.linuxbrew",
        "/nix",
    }
)
_DENIED_MANAGED_TIRITH_PREFIXES = (
    "/nix/store",
    "/nix/var/nix/profiles",
    "/opt/homebrew/Cellar",
    "/home/linuxbrew/.linuxbrew/Cellar",
)


def _managed_tirith_root_is_denied(path: str) -> bool:
    """Reject system and package-manager roots as Hermes-owned storage.

    This mirrors Tirith 0.4.1+'s self-update ownership boundary. It is also
    required for pre-0.4.1 bootstraps, where Hermes performs the replacement
    itself after proving the old bytes match an official release.
    """
    # ``normcase`` is intentionally a no-op in Python's POSIX path module,
    # even on the normally case-insensitive macOS filesystems.  Fold the
    # comparison keys explicitly so spelling /OPT/HOMEBREW/... cannot grant
    # Hermes ownership over the real /opt/homebrew tree.  Being conservative
    # on a case-sensitive POSIX volume is preferable to ever self-replacing a
    # package-manager binary.
    def comparison_key(value: str) -> str:
        return os.path.normcase(os.path.normpath(os.path.abspath(value))).casefold()

    lexical_root = comparison_key(path)
    canonical_root = comparison_key(os.path.realpath(path))
    roots = {lexical_root, canonical_root}
    denied_roots = {
        comparison_key(value)
        for value in _DENIED_MANAGED_TIRITH_ROOTS
    }
    if roots & denied_roots:
        return True
    for prefix in _DENIED_MANAGED_TIRITH_PREFIXES:
        denied_prefix = comparison_key(prefix)
        for root in roots:
            if root == denied_prefix or root.startswith(denied_prefix + os.sep):
                return True
    return False


def _is_owned_private_directory(path: str) -> bool:
    """Return whether ``path`` is a real directory safe for executable data."""
    try:
        directory_stat = os.lstat(path)
    except OSError:
        return False
    if not stat.S_ISDIR(directory_stat.st_mode):
        return False
    effective_uid = getattr(os, "geteuid", None)
    if os.name == "posix" and effective_uid is not None:
        if directory_stat.st_uid != effective_uid():
            return False
        if directory_stat.st_mode & 0o022:
            return False
        if not _trusted_unix_acl_is_private(path, directory=True):
            return False
    return True


def _linux_posix_acl_blob_is_private(
    value: bytes,
    *,
    owner_uid: int,
    effective_uid: int,
) -> bool:
    """Validate one Linux ``system.posix_acl_*`` xattr value."""
    acl_ea_version = 0x0002
    acl_user_obj = 0x01
    acl_user = 0x02
    acl_group_obj = 0x04
    acl_group = 0x08
    acl_mask = 0x10
    acl_other = 0x20
    acl_write = 0x02

    if len(value) < 4 or (len(value) - 4) % 8 != 0:
        return False
    if int.from_bytes(value[:4], "little") != acl_ea_version:
        return False
    for offset in range(4, len(value), 8):
        entry = value[offset:offset + 8]
        tag = int.from_bytes(entry[:2], "little")
        permissions = int.from_bytes(entry[2:4], "little")
        principal_id = int.from_bytes(entry[4:8], "little")
        if tag in {acl_user_obj, acl_group_obj, acl_mask, acl_other}:
            continue
        if tag == acl_user:
            if permissions & acl_write and principal_id not in {
                0,
                owner_uid,
                effective_uid,
            }:
                return False
            continue
        if tag == acl_group:
            if permissions & acl_write:
                return False
            continue
        return False
    return True


def _linux_acl_is_private(path: str, *, directory: bool) -> bool:
    """Reject Linux access/default ACLs that grant foreign write authority."""
    getxattr = getattr(os, "getxattr", None)
    listxattr = getattr(os, "listxattr", None)
    effective_uid = getattr(os, "geteuid", None)
    if getxattr is None or listxattr is None or effective_uid is None:
        return False
    try:
        owner_uid = os.lstat(path).st_uid
        attribute_names = set(listxattr(path, follow_symlinks=False))
    except (OSError, TypeError):
        return False

    expected_names = {
        "system.posix_acl_access",
        "system.posix_acl_default",
    }
    if any(
        name.startswith("system.") and "acl" in name.lower()
        and name not in expected_names
        for name in attribute_names
    ):
        # NFSv4/rich ACL semantics are not equivalent to POSIX ACL xattrs.
        return False

    names = ["system.posix_acl_access"]
    if directory:
        names.append("system.posix_acl_default")
    missing_xattr_errors = {errno.ENODATA}
    enoattr = getattr(errno, "ENOATTR", None)
    if enoattr is not None:
        missing_xattr_errors.add(enoattr)
    for name in names:
        try:
            value = getxattr(path, name, follow_symlinks=False)
        except OSError as exc:
            if exc.errno in missing_xattr_errors:
                continue
            return False
        except TypeError:
            return False
        if len(value) > 64 * 1024 or not _linux_posix_acl_blob_is_private(
            value,
            owner_uid=owner_uid,
            effective_uid=effective_uid(),
        ):
            return False
    return True


def _darwin_acl_is_private(path: str) -> bool:
    """Reject mutating allow entries in a macOS extended ACL."""
    import ctypes

    acl_type_extended = 0x0000_0100
    acl_first_entry = 0
    acl_next_entry = -1
    acl_extended_allow = 1
    mutating_permissions = (
        (1 << 2)
        | (1 << 4)
        | (1 << 5)
        | (1 << 6)
        | (1 << 8)
        | (1 << 10)
        | (1 << 12)
        | (1 << 13)
    )
    encoded_path = os.fsencode(path)
    if b"\0" in encoded_path:
        return False
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        acl_get_file = libc.acl_get_file
        acl_get_entry = libc.acl_get_entry
        acl_get_tag_type = libc.acl_get_tag_type
        acl_get_permset_mask_np = libc.acl_get_permset_mask_np
        acl_free = libc.acl_free
        acl_get_file.argtypes = [ctypes.c_char_p, ctypes.c_int]
        acl_get_file.restype = ctypes.c_void_p
        acl_get_entry.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        acl_get_entry.restype = ctypes.c_int
        acl_get_tag_type.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        acl_get_tag_type.restype = ctypes.c_int
        acl_get_permset_mask_np.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        acl_get_permset_mask_np.restype = ctypes.c_int
        acl_free.argtypes = [ctypes.c_void_p]
        acl_free.restype = ctypes.c_int
    except (AttributeError, OSError):
        return False

    ctypes.set_errno(0)
    acl = acl_get_file(encoded_path, acl_type_extended)
    if not acl:
        return ctypes.get_errno() == errno.ENOENT
    try:
        entry_id = acl_first_entry
        while True:
            entry = ctypes.c_void_p()
            ctypes.set_errno(0)
            if acl_get_entry(acl, entry_id, ctypes.byref(entry)) != 0:
                return ctypes.get_errno() == errno.EINVAL
            tag_type = ctypes.c_int()
            permissions = ctypes.c_uint64()
            if (
                acl_get_tag_type(entry, ctypes.byref(tag_type)) != 0
                or acl_get_permset_mask_np(entry, ctypes.byref(permissions)) != 0
            ):
                return False
            if (
                tag_type.value == acl_extended_allow
                and permissions.value & mutating_permissions
            ):
                return False
            entry_id = acl_next_entry
    finally:
        acl_free(acl)


def _trusted_unix_acl_is_private(path: str, *, directory: bool) -> bool:
    """Validate ACL mutation authority on supported managed Unix targets."""
    system = platform.system()
    if system == "Linux":
        return _linux_acl_is_private(path, directory=directory)
    if system == "Darwin":
        return _darwin_acl_is_private(path)
    return False


def _is_owned_private_executable(path: str) -> bool:
    """Return whether a managed executable is owner-only mutable and runnable."""
    try:
        executable_stat = os.lstat(path)
    except OSError:
        return False
    if not stat.S_ISREG(executable_stat.st_mode):
        return False
    effective_uid = getattr(os, "geteuid", None)
    if os.name == "posix" and effective_uid is not None:
        return (
            executable_stat.st_uid == effective_uid()
            and executable_stat.st_mode & 0o022 == 0
            and executable_stat.st_mode & stat.S_IXUSR != 0
            and _trusted_unix_acl_is_private(path, directory=False)
        )
    return os.access(path, os.X_OK)


def _managed_tirith_directory_chain() -> list[str]:
    """Return every directory from HERMES_HOME through managed ``bin``.

    Immutable images add ``.tirith-managed/<target>`` below HERMES_HOME. Each
    component is an ownership boundary: validating only the final target and
    ``bin`` would let an intermediate symlink redirect replacement elsewhere.
    """
    base = os.path.abspath(_get_hermes_home())
    managed_home = os.path.abspath(_managed_tirith_home())
    try:
        if os.path.commonpath((base, managed_home)) != base:
            return []
    except ValueError:
        return []

    relative = os.path.relpath(managed_home, base)
    parts = [] if relative == os.curdir else relative.split(os.sep)
    if any(part in {"", os.curdir, os.pardir} for part in parts):
        return []

    directories = [base]
    current = base
    for part in parts:
        current = os.path.join(current, part)
        directories.append(current)
    directories.append(os.path.join(managed_home, "bin"))
    return directories


def _managed_install_directory_is_real() -> bool:
    """Reject package-manager, redirected, or peer-writable managed roots."""
    directories = _managed_tirith_directory_chain()
    if not directories or any(
        _managed_tirith_root_is_denied(directory) for directory in directories
    ):
        return False
    return all(_is_owned_private_directory(directory) for directory in directories)


def _is_managed_tirith_location(path: str) -> bool:
    """Return whether ``path`` names or aliases Hermes' managed Tirith file.

    Lexical comparison also covers an absent installation destination. For an
    existing file, identity comparison prevents a PATH symlink, hard link, or
    case alias from shedding the managed-cache trust policy.
    """
    expected = os.path.normcase(os.path.abspath(_managed_tirith_path()))
    candidate = os.path.normcase(os.path.abspath(path))
    if candidate == expected:
        return True
    try:
        return os.path.samefile(candidate, expected)
    except OSError:
        return False


def _is_managed_tirith(path: str) -> bool:
    """Return whether ``path`` is inside Hermes' real managed-bin boundary."""
    managed_path = os.path.abspath(_managed_tirith_path())
    return (
        _is_managed_tirith_location(path)
        and _managed_install_directory_is_real()
        and _is_owned_private_executable(managed_path)
    )


def _validated_tirith_path(path: str) -> str | None:
    """Return a usable path, normalizing managed aliases to the owned path."""
    if not os.path.isfile(path) or not os.access(path, os.X_OK):
        return None
    if not _is_managed_tirith_location(path):
        return path
    managed_path = os.path.abspath(_managed_tirith_path())
    return managed_path if _is_managed_tirith(managed_path) else None


def _update_state_path() -> str:
    return os.path.join(_managed_tirith_home(), ".tirith-update-state.json")


def _update_process_lock_path() -> str:
    return os.path.join(_managed_tirith_home(), ".tirith-update.lock")


def _read_small_regular_file(path: str, max_bytes: int) -> str | None:
    """Read a bounded regular file without following a final-component symlink."""
    try:
        file_stat = os.lstat(path)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size > max_bytes:
            return None
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags)
        try:
            with os.fdopen(fd, "r", encoding="utf-8") as handle:
                fd = -1
                value = handle.read(max_bytes + 1)
        finally:
            if fd >= 0:
                os.close(fd)
        if len(value.encode("utf-8")) > max_bytes:
            return None
        return value
    except (OSError, UnicodeError):
        return None


def _valid_update_state(state: object) -> _UpdateState | None:
    """Return a validated update-state object, or ``None``."""
    if not isinstance(state, dict):
        return None
    state_dict = cast(dict[str, object], state)
    schema_version = state_dict.get("schema_version")
    if type(schema_version) is not int or schema_version != _UPDATE_STATE_SCHEMA:
        return None
    checked_at = state_dict.get("checked_at")
    outcome = state_dict.get("outcome")
    if not isinstance(checked_at, (int, float)) or isinstance(checked_at, bool):
        return None
    try:
        normalized_checked_at = float(checked_at)
    except (OverflowError, TypeError, ValueError):
        return None
    if not math.isfinite(normalized_checked_at) or normalized_checked_at < 0:
        return None
    if not isinstance(outcome, str) or (
        outcome != "failed" and outcome not in _UPDATE_SUCCESS_OUTCOMES
    ):
        return None
    return {
        "schema_version": schema_version,
        "checked_at": normalized_checked_at,
        "outcome": outcome,
    }


def _update_state_key() -> str:
    return os.path.normcase(os.path.abspath(_update_state_path()))


def _read_update_state() -> _UpdateState | None:
    disk_state = None
    raw = _read_small_regular_file(_update_state_path(), _UPDATE_STATE_MAX_BYTES)
    if raw is not None:
        try:
            disk_state = _valid_update_state(json.loads(raw))
        except (json.JSONDecodeError, TypeError):
            pass

    with _in_process_update_state_lock:
        memory_state = _in_process_update_states.get(_update_state_key())
        if memory_state is not None:
            memory_state = memory_state.copy()

    if disk_state is None:
        return memory_state
    if memory_state is None:
        return disk_state
    if memory_state["checked_at"] >= disk_state["checked_at"]:
        return memory_state
    return disk_state


def _write_update_state(outcome: str, *, now: float | None = None) -> bool:
    """Record update throttling state in memory and atomically on disk."""
    payload = _valid_update_state(
        {
            "schema_version": _UPDATE_STATE_SCHEMA,
            "checked_at": time.time() if now is None else now,
            "outcome": outcome,
        }
    )
    if payload is None:
        return False

    # A read-only/full home must not turn every later command in a long-lived
    # process into another release-network attempt. Record the backoff before
    # best-effort persistence; another process still relies on the disk copy.
    with _in_process_update_state_lock:
        _in_process_update_states[_update_state_key()] = payload.copy()

    home = _managed_tirith_home()
    tmp_path = ""
    fd = -1
    try:
        os.makedirs(home, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".tirith-update-state-", dir=home)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, _update_state_path())
        tmp_path = ""
        return True
    except (OSError, TypeError, ValueError):
        return False
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _update_is_due(*, now: float | None = None) -> bool:
    """Return whether Hermes should launch a managed Tirith update check."""
    state = _read_update_state()
    if state is None:
        return True
    current_time = time.time() if now is None else now
    checked_at = state["checked_at"]
    if checked_at > current_time:
        return True
    ttl = _UPDATE_FAILURE_TTL if state["outcome"] == "failed" else _UPDATE_CHECK_TTL
    return (current_time - checked_at) >= ttl


def _acquire_update_lock_with_status() -> tuple[int | None, str]:
    """Acquire the advisory lock and distinguish contention from I/O errors.

    Tirith only ships on POSIX platforms. ``flock`` releases automatically on
    process death, so there is no stale-file reclamation race and a suspended
    updater cannot be mistaken for a dead one.
    """
    if os.name == "nt":
        return None, "error"

    import fcntl

    path = _update_process_lock_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except OSError:
        return None, "error"

    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
    except OSError:
        return None, "error"
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            return None, "error"
        os.fchmod(fd, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None, "busy"
    except OSError as exc:
        os.close(fd)
        if exc.errno in {errno.EACCES, errno.EAGAIN}:
            return None, "busy"
        return None, "error"
    return fd, "acquired"


def _acquire_update_lock():
    """Acquire a process-bound advisory lock, or return ``None``."""
    fd, _status = _acquire_update_lock_with_status()
    return fd


def _release_update_lock(fd: int) -> None:
    """Release a lock returned by :func:`_acquire_update_lock`."""
    if os.name != "nt":
        import fcntl

        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
    try:
        os.close(fd)
    except OSError:
        pass


def _failure_marker_path() -> str:
    """Return the path to the install-failure marker file."""
    return os.path.join(_managed_tirith_home(), ".tirith-install-failed")


def _read_failure_reason() -> str | None:
    """Read the failure reason from the disk marker.

    Returns the reason string, or None if the marker doesn't exist or is
    older than _MARKER_TTL.
    """
    try:
        p = _failure_marker_path()
        mtime = os.path.getmtime(p)
        if (time.time() - mtime) >= _MARKER_TTL:
            return None
        with open(p, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def _is_install_failed_on_disk() -> bool:
    """Check if a recent install failure was persisted to disk.

    Returns False (allowing retry) when:
    - No marker exists
    - Marker is older than _MARKER_TTL (24h)
    - Marker reason is 'cosign_missing' and cosign is now on PATH
    """
    reason = _read_failure_reason()
    if reason is None:
        return False
    if reason == "cosign_missing" and shutil.which("cosign"):
        _clear_install_failed()
        return False
    return True


def _mark_install_failed(reason: str = ""):
    """Persist install failure to disk to avoid retry on next process.

    Args:
        reason: Short tag identifying the failure cause. Use "cosign_missing"
                when cosign is not on PATH so the marker can be auto-cleared
                once cosign becomes available.
    """
    try:
        p = _failure_marker_path()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(reason)
    except OSError:
        pass


def _clear_install_failed():
    """Remove the install-failure marker after local recovery."""
    # Reset the warn-once dedupe set so a subsequent failure (e.g. user
    # deletes the binary) surfaces in the log again instead of being
    # silently suppressed by a stale dedupe key from before the fix.
    _reset_spawn_warning_state()
    try:
        os.unlink(_failure_marker_path())
    except OSError:
        pass


def _hermes_bin_dir() -> str:
    """Return Hermes' private Tirith bin directory, creating it if needed."""
    directories = _managed_tirith_directory_chain()
    if not directories or any(
        _managed_tirith_root_is_denied(directory) for directory in directories
    ):
        raise OSError("Hermes managed Tirith directory is outside its trusted root")

    # HERMES_HOME is the configured trust anchor. Once it is real, owned, and
    # not peer-writable, create each descendant separately and lstat it before
    # proceeding. A symlink can therefore never be traversed by a later mkdir.
    os.makedirs(directories[0], mode=0o700, exist_ok=True)
    if not _is_owned_private_directory(directories[0]):
        raise OSError("Hermes home is redirected or peer-writable")
    for directory in directories[1:]:
        try:
            os.mkdir(directory, 0o755)
        except FileExistsError:
            pass
        if not _is_owned_private_directory(directory):
            raise OSError("Hermes managed Tirith directory is redirected or peer-writable")
    return directories[-1]


def _detect_target() -> str | None:
    """Return the Rust target triple for the current platform, or None.

    Hermes-managed install/update is currently limited to Tirith's macOS and
    Linux tarballs. Native Windows has a Tirith release, but its ZIP packaging
    and Hermes self-update ownership proof need a separate integration. Callers
    treat ``None`` as unsupported by this manager and fall back to Hermes'
    pattern-matching guards.
    """
    system = platform.system()
    machine = platform.machine().lower()

    # Termux uses Android's Bionic libc, so Tirith's glibc Linux build cannot
    # run there.  The release publishes a statically linked musl build for
    # AArch64 Termux; no x86_64 Android artifact is currently published.
    if is_termux():
        return (
            "aarch64-unknown-linux-musl"
            if machine in {"aarch64", "arm64"}
            else None
        )

    if system == "Darwin":
        plat = "apple-darwin"
    elif system == "Linux":
        plat = "unknown-linux-gnu"
    else:
        return None

    if machine in {"x86_64", "amd64"}:
        arch = "x86_64"
    elif machine in {"aarch64", "arm64"}:
        arch = "aarch64"
    else:
        return None

    return f"{arch}-{plat}"


def is_platform_supported() -> bool:
    """True when tirith ships a prebuilt binary for this OS+arch.

    Used by callers (CLI banner, etc.) to distinguish "tirith failed to
    install" from "tirith was never going to install here" — the latter
    is silent because there is nothing the user can do about it.
    """
    return _detect_target() is not None


def _download_file(
    url: str,
    dest: str,
    timeout: int = 10,
    *,
    max_bytes: int,
) -> None:
    """Download a URL without allowing the response to exceed ``max_bytes``."""
    req = urllib.request.Request(url)
    from agent.secret_scope import get_secret
    token = get_secret("GITHUB_TOKEN")
    if token:
        # ``urllib`` copies ordinary headers to redirect requests, including
        # cross-origin GitHub release-asset redirects. Keep the credential on
        # the initial github.com request only.
        req.add_unredirected_header("Authorization", f"token {token}")
    written = 0
    try:
        with open_credentialed_url(req, timeout=timeout) as resp, open(dest, "wb") as f:
            while chunk := resp.read(min(1024 * 1024, max_bytes - written + 1)):
                written += len(chunk)
                if written > max_bytes:
                    raise ValueError(
                        f"download exceeds {max_bytes}-byte limit: {url}"
                    )
                f.write(chunk)
    except Exception:
        try:
            os.unlink(dest)
        except OSError:
            pass
        raise


def _release_identity_from_certificate(
    cert_path: str,
) -> tuple[tuple[int, int, int], str] | None:
    """Return the one stable Tirith release identity carried by a certificate."""
    try:
        from cryptography import x509
    except ImportError as exc:
        logger.warning("cannot inspect cosign certificate: %s", exc)
        return None

    try:
        with open(cert_path, "rb") as cert_file:
            certificate = x509.load_pem_x509_certificate(cert_file.read())
        san = certificate.extensions.get_extension_for_class(
            x509.SubjectAlternativeName
        ).value
        identities = san.get_values_for_type(x509.UniformResourceIdentifier)
    except (OSError, ValueError, x509.ExtensionNotFound, x509.DuplicateExtension) as exc:
        logger.warning("cannot inspect cosign certificate identity: %s", exc)
        return None

    matches = []
    for identity in identities:
        match = _COSIGN_IDENTITY_RE.fullmatch(identity)
        if match is not None:
            matches.append((tuple(int(part) for part in match.groups()), identity))
    if len(matches) != 1:
        logger.warning(
            "cosign certificate must contain exactly one stable Tirith release identity"
        )
        return None
    return matches[0]


def _verify_cosign(
    checksums_path: str,
    sig_path: str,
    cert_path: str,
) -> tuple[bool | None, tuple[int, int, int] | None]:
    """Verify cosign provenance signature on checksums.txt.

    Returns:
        (True, version) — cosign verified one exact stable release identity
        (False, None)   — cosign found but verification failed
        (None, None)    — cosign not available or could not execute

    ``False`` is an explicit verification rejection. ``None`` lets the caller
    use Hermes' documented SHA-256-only fallback.
    """
    cosign = shutil.which("cosign")
    if not cosign:
        logger.info("cosign not found on PATH")
        return None, None

    release_identity = _release_identity_from_certificate(cert_path)
    if release_identity is None:
        return False, None
    release_version, certificate_identity = release_identity

    try:
        result = subprocess.run(
            [cosign, "verify-blob",
             "--certificate", cert_path,
             "--signature", sig_path,
             "--certificate-identity", certificate_identity,
             "--certificate-oidc-issuer", _COSIGN_ISSUER,
             checksums_path],
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            timeout=15,
            stdin=subprocess.DEVNULL,
            env=_tirith_subprocess_env(),
        )
        if result.returncode == 0:
            logger.info("cosign provenance verification passed")
            return True, release_version
        else:
            logger.warning("cosign verification failed (exit %d): %s",
                          result.returncode, result.stderr.strip())
            return False, None
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("cosign execution failed: %s", exc)
        return None, None


def _verify_checksum(archive_path: str, checksums_path: str, archive_name: str) -> bool:
    """Verify SHA-256 of the archive against checksums.txt."""
    expected = None
    with open(checksums_path, encoding="utf-8") as f:
        for line in f:
            # Format: "<hash>  <filename>"
            parts = line.strip().split("  ", 1)
            if len(parts) == 2 and parts[1] == archive_name:
                expected = parts[0]
                break
    if not expected:
        logger.warning("No checksum entry for %s", archive_name)
        return False

    actual = _sha256_file(archive_path)
    if actual != expected:
        logger.warning("Checksum mismatch: expected %s, got %s", expected, actual)
        return False
    return True


def _sha256_file(path: str) -> str:
    """Return the SHA-256 digest of a regular filesystem path."""
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


class _ArchiveExpansionLimitExceeded(ValueError):
    """Raised before a release archive expands past its process budget."""


class _ReadableBytes(Protocol):
    def read(self, size: int = -1) -> bytes: ...


class _BoundedDecompressedReader(io.RawIOBase):
    """Count bytes below ``tarfile`` so metadata bodies count toward the cap."""

    def __init__(self, source: _ReadableBytes, max_bytes: int):
        self._source = source
        self._max_bytes = max_bytes
        self._bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        remaining_with_probe = self._max_bytes - self._bytes_read + 1
        if size < 0 or size > remaining_with_probe:
            size = remaining_with_probe
        chunk = self._source.read(size)
        self._bytes_read += len(chunk)
        if self._bytes_read > self._max_bytes:
            raise _ArchiveExpansionLimitExceeded(
                "tirith release archive exceeds "
                f"{self._max_bytes}-byte decompression limit"
            )
        return chunk

    def readable(self) -> bool:
        return True


def _extract_tirith_binary(
    tar: tarfile.TarFile,
    dest_dir: str,
    log,
) -> tuple[str | None, str]:
    """Extract the tirith binary from a release archive into dest_dir."""
    # Iterate lazily instead of calling getmembers(). The latter scans and
    # decompresses the complete archive before we can enforce member limits.
    for member_number, member in enumerate(tar, start=1):
        if member_number > _MAX_RELEASE_ARCHIVE_MEMBERS:
            log(
                "tirith archive exceeds %d-member limit",
                _MAX_RELEASE_ARCHIVE_MEMBERS,
            )
            return None, "too_many_archive_members"
        if member.size > _MAX_TIRITH_BINARY_BYTES:
            log(
                "tirith archive member exceeds %d-byte limit: %s",
                _MAX_TIRITH_BINARY_BYTES,
                member.name,
            )
            return None, "archive_member_too_large"
        if member.name == "tirith" or member.name.endswith("/tirith"):
            if ".." in member.name:
                continue
            if not member.isfile():
                log("tirith archive member is not a regular file: %s", member.name)
                return None, "binary_not_regular_file"
            src_file = tar.extractfile(member)
            if src_file is None:
                log("tirith binary could not be read from archive")
                return None, "binary_extract_failed"

            dest_path = os.path.join(dest_dir, "tirith")
            try:
                with open(dest_path, "wb") as out:
                    shutil.copyfileobj(src_file, out)
            finally:
                src_file.close()
            return dest_path, ""

    log("tirith binary not found in archive")
    return None, "binary_not_in_archive"


def _extract_release_archive(
    archive_path: str,
    dest_dir: str,
    log,
) -> tuple[str | None, str]:
    """Stream a gzip release through a total decompression budget.

    Bounding only yielded ``TarInfo`` members is insufficient: GNU long-name
    and PAX extension records are inflated and consumed inside ``tarfile``
    before iteration yields a member. The reader therefore enforces the cap
    beneath the tar parser, covering headers, extension bodies, padding, and
    ordinary member contents together.
    """
    try:
        with gzip.open(archive_path, "rb") as decompressed:
            bounded = _BoundedDecompressedReader(
                decompressed,
                _MAX_RELEASE_ARCHIVE_UNPACKED_BYTES,
            )
            with tarfile.open(fileobj=bounded, mode="r|") as tar:
                return _extract_tirith_binary(tar, dest_dir, log)
    except _ArchiveExpansionLimitExceeded as exc:
        log("tirith release archive is too large when unpacked: %s", exc)
        return None, "archive_too_large"
    except (EOFError, OSError, tarfile.TarError) as exc:
        log("tirith release archive could not be read: %s", exc)
        return None, "archive_invalid"


def _atomic_replace_binary(
    source: str,
    destination: str,
    *,
    expected_existing_sha256: str | None = None,
    require_destination_absent: bool = False,
) -> None:
    """Stage ``source`` beside ``destination`` and atomically commit it.

    Download extraction happens in the system temporary directory, which may
    be on another filesystem. Copying into a sibling first makes the final
    commit atomic and preserves an existing working scanner if any staging or
    commit step fails. Initial installs use an atomic hard-link commit so a
    concurrent installer can win without being overwritten.
    """
    destination_dir = os.path.abspath(os.path.dirname(destination))
    os.makedirs(destination_dir, exist_ok=True)
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    directory_fd = os.open(destination_dir, directory_flags)
    staged_name = ""
    staged_fd = -1
    try:
        if not stat.S_ISDIR(os.fstat(directory_fd).st_mode):
            raise NotADirectoryError(destination_dir)
        destination_name = os.path.basename(destination)
        for _attempt in range(32):
            candidate = f".tirith-install-{secrets.token_hex(8)}"
            try:
                staged_fd = os.open(
                    candidate,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                    dir_fd=directory_fd,
                )
            except FileExistsError:
                continue
            staged_name = candidate
            break
        else:
            raise FileExistsError("could not allocate a unique Tirith staging file")

        with os.fdopen(staged_fd, "wb") as output, open(source, "rb") as input_file:
            staged_fd = -1
            shutil.copyfileobj(input_file, output)
            os.fchmod(output.fileno(), 0o755)
            output.flush()
            os.fsync(output.fileno())

        staged_path = os.path.join(destination_dir, staged_name)
        if not _is_owned_private_executable(staged_path):
            raise PermissionError(
                "Tirith staging file is not owner-only mutable and executable"
            )

        if expected_existing_sha256 is not None:
            current_flags = os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                current_flags |= os.O_NOFOLLOW
            current_fd = os.open(destination_name, current_flags, dir_fd=directory_fd)
            try:
                with os.fdopen(current_fd, "rb") as current:
                    current_fd = -1
                    sha = hashlib.sha256()
                    for chunk in iter(lambda: current.read(8192), b""):
                        sha.update(chunk)
                if sha.hexdigest() != expected_existing_sha256:
                    raise OSError(
                        "managed Tirith changed after its release bytes were verified"
                    )
            finally:
                if current_fd >= 0:
                    os.close(current_fd)

        if require_destination_absent:
            # A hard link is the portable POSIX no-clobber commit primitive:
            # destination creation and the EEXIST check are one filesystem
            # operation. The stage is a regular sibling on the same filesystem.
            os.link(
                staged_name,
                destination_name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
            os.unlink(staged_name, dir_fd=directory_fd)
        else:
            os.replace(
                staged_name,
                destination_name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
        staged_name = ""
        try:
            os.fsync(directory_fd)
        except OSError:
            # Some otherwise-supported filesystems reject directory fsync.
            pass
    finally:
        if staged_fd >= 0:
            try:
                os.close(staged_fd)
            except OSError:
                pass
        if staged_name:
            try:
                os.unlink(staged_name, dir_fd=directory_fd)
            except OSError:
                pass
        os.close(directory_fd)


def _download_verified_tirith(
    base_url: str,
    target: str,
    workdir: str,
    log,
    *,
    require_signed_release: bool = False,
) -> tuple[str | None, str, bool, tuple[int, int, int] | None]:
    """Download, verify, and extract one Tirith release into ``workdir``.

    Initial installation retains the historical HTTPS + SHA-256 fallback.
    Callers replacing an existing managed executable set
    ``require_signed_release`` so an unavailable signature fails closed.
    """
    archive_name = f"tirith-{target}.tar.gz"
    archive_path = os.path.join(workdir, archive_name)
    checksums_path = os.path.join(workdir, "checksums.txt")
    sig_path = os.path.join(workdir, "checksums.txt.sig")
    cert_path = os.path.join(workdir, "checksums.txt.pem")

    try:
        _download_file(
            f"{base_url}/{archive_name}",
            archive_path,
            max_bytes=_MAX_ARCHIVE_DOWNLOAD_BYTES,
        )
        _download_file(
            f"{base_url}/checksums.txt",
            checksums_path,
            max_bytes=_MAX_METADATA_DOWNLOAD_BYTES,
        )
    except Exception as exc:
        log("tirith download failed: %s", exc)
        return None, "download_failed", False, None

    cosign_verified = False
    signed_version = None
    if shutil.which("cosign"):
        try:
            _download_file(
                f"{base_url}/checksums.txt.sig",
                sig_path,
                max_bytes=_MAX_METADATA_DOWNLOAD_BYTES,
            )
            _download_file(
                f"{base_url}/checksums.txt.pem",
                cert_path,
                max_bytes=_MAX_METADATA_DOWNLOAD_BYTES,
            )
        except Exception as exc:
            if require_signed_release:
                log("tirith release rejected: cosign artifacts unavailable: %s", exc)
                return None, "cosign_artifacts_unavailable", False, None
            logger.info(
                "cosign artifacts unavailable (%s), proceeding with SHA-256 only", exc
            )
        else:
            cosign_result, signed_version = _verify_cosign(
                checksums_path, sig_path, cert_path
            )
            if cosign_result is True:
                cosign_verified = True
            elif cosign_result is False:
                log("tirith release rejected: cosign provenance verification failed")
                return None, "cosign_verification_failed", False, None
            else:
                if require_signed_release:
                    log("tirith release rejected: cosign provenance could not be verified")
                    return None, "cosign_exec_failed", False, None
                signed_version = None
                logger.info("cosign execution failed, proceeding with SHA-256 only")
    else:
        if require_signed_release:
            log("tirith release replacement requires cosign provenance verification")
            return None, "cosign_missing", False, None
        logger.info(
            "cosign not on PATH — using SHA-256 verification only "
            "(install cosign for full supply chain verification)"
        )

    if not _verify_checksum(archive_path, checksums_path, archive_name):
        return None, "checksum_failed", cosign_verified, signed_version

    src, reason = _extract_release_archive(archive_path, workdir, log)
    return src, reason, cosign_verified, signed_version


def _install_tirith(
    *,
    log_failures: bool = True,
    expected_existing_sha256: str | None = None,
    current_version: tuple[int, int, int] | None = None,
    minimum_candidate_version: tuple[int, int, int] | None = None,
    allow_same_version_replacement: bool = False,
) -> tuple[str | None, str]:
    """Download and install Tirith to Hermes' private managed cache.

    Always verifies the SHA-256 checksum. Initial installation verifies cosign
    provenance when available; replacement of an existing managed executable
    requires it.
    Returns (installed_path, failure_reason).  On success failure_reason is "".
    failure_reason is a short tag used by the disk marker to decide if the
    failure is retryable (e.g. "cosign_missing" clears when cosign appears).
    """
    if not _tirith_auto_install_allowed():
        return None, "lazy_installs_disabled"

    replacing_existing = expected_existing_sha256 is not None
    if replacing_existing != (current_version is not None):
        return None, "invalid_replacement_request"
    if (
        minimum_candidate_version is not None or allow_same_version_replacement
    ) and not replacing_existing:
        return None, "invalid_replacement_request"

    log = logger.warning if log_failures else logger.debug

    target = _detect_target()
    if not target:
        logger.info("tirith auto-install: unsupported platform %s/%s",
                     platform.system(), platform.machine())
        return None, "unsupported_platform"
    if (
        allow_same_version_replacement
        and target != "aarch64-unknown-linux-musl"
    ):
        return None, "invalid_replacement_request"

    base_url = f"https://github.com/{_REPO}/releases/latest/download"
    managed_dest = _managed_tirith_path()
    destination_was_absent = not os.path.lexists(managed_dest)
    # Initial installers are create-only. An existing path belongs to the
    # process that installed it and cannot silently become a replacement call.
    if not destination_was_absent and not replacing_existing:
        return None, "destination_exists"

    try:
        tmpdir = tempfile.mkdtemp(prefix="tirith-install-")
    except OSError as exc:
        log("tirith install failed: cannot create temp dir: %s", exc)
        return None, "no_space"
    try:
        logger.info("tirith not found — downloading latest release for %s...", target)
        src, reason, cosign_verified, signed_version = _download_verified_tirith(
            base_url,
            target,
            tmpdir,
            log,
            require_signed_release=replacing_existing,
        )
        if src is None:
            return None, reason

        if replacing_existing:
            # The exact stable tag came from the certificate identity that
            # cosign verified over the checksum manifest. It is therefore
            # authenticated without executing downloaded bytes from /tmp.
            if not cosign_verified or signed_version is None:
                return None, "candidate_version_unverified"
            assert current_version is not None
            candidate_is_older = signed_version < current_version
            candidate_is_same = signed_version == current_version
            if candidate_is_older or (
                candidate_is_same and not allow_same_version_replacement
            ):
                logger.info(
                    "tirith replacement skipped: signed candidate v%s cannot "
                    "replace installed v%s",
                    ".".join(str(part) for part in signed_version),
                    ".".join(str(part) for part in current_version),
                )
                return None, "candidate_not_newer"
            if (
                minimum_candidate_version is not None
                and signed_version < minimum_candidate_version
            ):
                return None, "candidate_below_minimum"

        # Config is live and the verified download may have taken seconds.
        # Re-check before creating or replacing anything under HERMES_HOME.
        if not _tirith_auto_install_allowed():
            return None, "lazy_installs_disabled"

        try:
            dest = os.path.join(_hermes_bin_dir(), "tirith")
        except OSError as exc:
            log("tirith install aborted: untrusted managed directory: %s", exc)
            return None, "managed_directory_untrusted"
        # A different process may have installed a binary while this download
        # was in flight. Never let an initial checksum-only download become an
        # automatic replacement because of that race.
        if os.path.lexists(dest) and not cosign_verified:
            log("tirith replacement aborted: signed provenance is required")
            return None, "cosign_required_for_replacement"
        if not _managed_install_directory_is_real():
            log("tirith install aborted: Hermes managed-bin directory is redirected")
            return None, "managed_directory_untrusted"
        try:
            _atomic_replace_binary(
                src,
                dest,
                expected_existing_sha256=expected_existing_sha256,
                require_destination_absent=destination_was_absent,
            )
        except OSError as exc:
            log("tirith install failed while replacing %s: %s", dest, exc)
            return None, "install_replace_failed"
        if not _is_managed_tirith(dest):
            log("tirith install boundary changed while replacing %s", dest)
            return None, "managed_directory_changed"

        verification = "cosign + SHA-256" if cosign_verified else "SHA-256 only"
        logger.info("tirith installed to %s (%s)", dest, verification)
        return dest, ""

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


_TIRITH_VERSION_RE = re.compile(r"^tirith ([0-9]+)\.([0-9]+)\.([0-9]+)$")
_TIRITH_EMBEDDED_VERSION_RES = (
    # Tirith 0.3+ keeps the exact clap --version output in the release binary.
    re.compile(rb"tirith ([0-9]{1,10})\.([0-9]{1,10})\.([0-9]{1,10})(?:\r?\n|\x00)"),
    # Tirith 0.2.12, the release current when Hermes first enabled Termux,
    # predates that stable string but embeds its shell-hook version marker.
    re.compile(
        rb"shell-version([0-9]{1,10})\.([0-9]{1,10})\."
        rb"([0-9]{1,10})tirith\.sh"
    ),
)


def _parse_tirith_version(output: str) -> tuple[int, int, int] | None:
    """Parse only Tirith's stable ``tirith X.Y.Z`` version format."""
    if len(output) > 128:
        return None
    match = _TIRITH_VERSION_RE.fullmatch(output.strip())
    if match is None:
        return None
    try:
        major, minor, patch = match.groups()
        return int(major), int(minor), int(patch)
    except ValueError:
        return None


def _read_embedded_tirith_version(
    path: str,
) -> tuple[tuple[int, int, int] | None, str]:
    """Read a release version without executing an incompatible binary.

    Historical Hermes builds downloaded Tirith's glibc AArch64 artifact on
    Termux. Android's Bionic loader cannot execute it, so ``--version`` is not
    available during migration. Only stable, release-generated markers are
    accepted, and the caller must still byte-match the tagged release before
    replacing anything.
    """
    if not _is_managed_tirith(path):
        return None, "managed_path_untrusted"
    try:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags)
        with os.fdopen(fd, "rb") as binary:
            binary_stat = os.fstat(binary.fileno())
            if not stat.S_ISREG(binary_stat.st_mode):
                return None, "binary_read_failed"
            size = binary_stat.st_size
            if size <= 0 or size > _MAX_TIRITH_BINARY_BYTES:
                return None, "binary_size_invalid"
            payload = binary.read(_MAX_TIRITH_BINARY_BYTES + 1)
    except OSError:
        return None, "binary_read_failed"
    if len(payload) != size or len(payload) > _MAX_TIRITH_BINARY_BYTES:
        return None, "binary_size_invalid"

    versions: set[tuple[int, int, int]] = set()
    for pattern in _TIRITH_EMBEDDED_VERSION_RES:
        for match in pattern.finditer(payload):
            try:
                major, minor, patch = match.groups()
                versions.add((int(major), int(minor), int(patch)))
            except ValueError:
                return None, "unparseable"
    if len(versions) != 1:
        return None, "unparseable"
    return versions.pop(), ""


def _tirith_subprocess_env() -> dict[str, str]:
    from tools.environments.local import hermes_subprocess_env

    env = hermes_subprocess_env(inherit_credentials=False)
    # Tirith 0.4.1+ proves Hermes ownership against this exact root before it
    # self-replaces. In image installs it deliberately differs from the shared
    # data root so host/container binaries cannot collide.
    env["HERMES_HOME"] = _managed_tirith_home()
    return env


def _probe_tirith_version(path: str) -> tuple[tuple[int, int, int] | None, str]:
    """Return a stable Tirith version or an operational/parse reason."""
    if not _is_managed_tirith(path):
        return None, "managed_path_untrusted"
    path = os.path.abspath(_managed_tirith_path())
    try:
        result = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_UPDATE_PROBE_TIMEOUT,
            stdin=subprocess.DEVNULL,
            env=_tirith_subprocess_env(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, "probe_failed"
    if result.returncode != 0:
        return None, "probe_failed"
    version = _parse_tirith_version(result.stdout)
    return (version, "") if version is not None else (None, "unparseable")


def _verify_legacy_release_binary(
    path: str,
    version: tuple[int, int, int],
    *,
    target: str | None = None,
    log_failures: bool = True,
) -> tuple[str | None, str]:
    """Prove a pre-Hermes-provenance binary matches published release bytes.

    Legacy Tirith cannot attest whether it is a release or local build. Hermes
    therefore verifies the matching tagged archive before replacing it. The
    returned digest binds that proof to the later atomic swap.
    """
    target = target or _detect_target()
    if target is None:
        return None, "unsupported_platform"
    log = logger.warning if log_failures else logger.debug
    release = ".".join(str(part) for part in version)
    base_url = f"https://github.com/{_REPO}/releases/download/v{release}"
    try:
        tmpdir = tempfile.mkdtemp(prefix="tirith-legacy-proof-")
    except OSError:
        return None, "no_space"
    try:
        released, reason, _cosign_verified, _signed_version = _download_verified_tirith(
            base_url, target, tmpdir, log
        )
        if released is None:
            return None, reason
        try:
            expected = _sha256_file(released)
            actual = _sha256_file(path)
        except OSError:
            return None, "binary_read_failed"
        if actual != expected:
            logger.info(
                "tirith background update skipped: legacy binary does not match "
                "the v%s release artifact",
                release,
            )
            return None, "binary_mismatch"
        return actual, ""
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _probe_tirith_provenance(path: str) -> tuple[dict | None, str]:
    """Read provenance that Tirith 0.4.1+ exposes for updater ownership."""
    if not _is_managed_tirith(path):
        return None, "managed_path_untrusted"
    path = os.path.abspath(_managed_tirith_path())
    try:
        result = subprocess.run(
            [path, "version", "--provenance", "--format", "json"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_UPDATE_PROBE_TIMEOUT,
            stdin=subprocess.DEVNULL,
            env=_tirith_subprocess_env(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, "provenance_failed"
    if result.returncode != 0:
        return None, "provenance_failed"
    try:
        provenance = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
        return None, "provenance_failed"
    if not isinstance(provenance, dict):
        return None, "provenance_failed"
    return provenance, ""


def _provenance_allows_managed_update(
    path: str, version: tuple[int, int, int], provenance: dict
) -> bool:
    """Validate that Tirith itself recognizes this exact binary as Hermes-owned."""
    reported_path = provenance.get("binary_path")
    reported_version = provenance.get("version")
    if not isinstance(reported_path, str) or not os.path.isabs(reported_path):
        return False
    try:
        same_binary = os.path.samefile(path, reported_path)
    except OSError:
        same_binary = False
    provenance_version = (
        _parse_tirith_version(f"tirith {reported_version}")
        if isinstance(reported_version, str)
        else None
    )
    return (
        same_binary
        and provenance_version == version
        and provenance.get("install_method") == "hermes"
        and provenance.get("install_method_resolved") is True
        and provenance.get("dev_build") is False
    )


def _run_tirith_update(path: str) -> str:
    """Run Tirith's noninteractive updater and classify its JSON result."""
    if not _is_managed_tirith(path):
        return "failed"
    path = os.path.abspath(_managed_tirith_path())
    # The user may disable runtime installs while the background worker is
    # probing version/provenance. Re-check at the mutating/network sink so the
    # live opt-out takes effect without waiting for the worker to finish.
    if not _tirith_auto_install_allowed():
        return "deferred"
    try:
        result = subprocess.run(
            [path, "update", "--yes", "--format", "json"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_UPDATE_TIMEOUT,
            stdin=subprocess.DEVNULL,
            env=_tirith_subprocess_env(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.info("tirith background update failed; keeping current binary: %s", exc)
        return "failed"

    if result.returncode != 0:
        logger.info(
            "tirith background update exited with %d; keeping current binary: %s",
            result.returncode,
            result.stderr.strip()[:500],
        )
        return "failed"
    try:
        payload = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
        return "failed"
    if not isinstance(payload, dict):
        return "failed"
    if payload.get("action") == "none":
        return "current"
    if payload.get("action") == "updated":
        return "updated"
    return "failed"


def _verify_termux_release_binary(
    path: str,
    version: tuple[int, int, int],
    *,
    log_failures: bool,
) -> tuple[str | None, str | None, str]:
    """Match a managed Termux binary to its tagged GNU or musl release."""
    targets = ["aarch64-unknown-linux-gnu"]
    # Tirith first published a Termux-compatible AArch64 musl artifact in
    # v0.3.0. Earlier tags legitimately return 404 for that asset.
    if version >= (0, 3, 0):
        targets.append("aarch64-unknown-linux-musl")

    failures: list[str] = []
    for release_target in targets:
        expected_sha256, reason = _verify_legacy_release_binary(
            path,
            version,
            target=release_target,
            log_failures=log_failures,
        )
        if expected_sha256 is not None:
            return expected_sha256, release_target, ""
        failures.append(reason)
    if failures and all(reason == "binary_mismatch" for reason in failures):
        return None, None, "binary_mismatch"
    return None, None, next(
        (reason for reason in failures if reason != "binary_mismatch"),
        "binary_mismatch",
    )


def _maintain_managed_tirith(path: str, *, log_failures: bool = True) -> str:
    """Bootstrap or update an existing Hermes-managed Tirith binary."""
    if not _is_managed_tirith(path):
        return "skipped"
    path = os.path.abspath(_managed_tirith_path())

    target = _detect_target()
    termux_musl_target = target == "aarch64-unknown-linux-musl"
    version_was_embedded = False
    version, reason = _probe_tirith_version(path)
    if version is None:
        if termux_musl_target and reason == "probe_failed":
            version, reason = _read_embedded_tirith_version(path)
            version_was_embedded = version is not None
        if version is None and reason == "unparseable":
            logger.info("tirith background update skipped: unrecognized build version")
            return "skipped"
        if version is None:
            return "failed"

    if termux_musl_target:
        # A future Tirith release that identifies its native musl target can
        # safely use its own updater. v0.4.1 reports GNU for both AArch64
        # builds, so older releases take the byte-matched migration below.
        if not version_was_embedded and version >= _SELF_UPDATE_MIN_VERSION:
            provenance, _ = _probe_tirith_provenance(path)
            if provenance is not None and _provenance_allows_managed_update(
                path, version, provenance
            ):
                if provenance.get("target") == "aarch64-unknown-linux-musl":
                    return _run_tirith_update(path)

        expected_sha256, matched_target, verification_reason = (
            _verify_termux_release_binary(
                path,
                version,
                log_failures=log_failures,
            )
        )
        if expected_sha256 is None:
            return "skipped" if verification_reason == "binary_mismatch" else "failed"
        installed, install_reason = _install_tirith(
            log_failures=log_failures,
            expected_existing_sha256=expected_sha256,
            current_version=version,
            minimum_candidate_version=_SELF_UPDATE_MIN_VERSION,
            allow_same_version_replacement=(
                matched_target == "aarch64-unknown-linux-gnu"
            ),
        )
        if installed and _is_managed_tirith(installed):
            return "bootstrapped" if version < _SELF_UPDATE_MIN_VERSION else "updated"
        if install_reason == "lazy_installs_disabled":
            return "deferred"
        if install_reason == "candidate_not_newer":
            return "current"
        return "failed"

    if version < _SELF_UPDATE_MIN_VERSION:
        expected_sha256, verification_reason = _verify_legacy_release_binary(
            path, version, log_failures=log_failures
        )
        if expected_sha256 is None:
            return "skipped" if verification_reason == "binary_mismatch" else "failed"
        installed, install_reason = _install_tirith(
            log_failures=log_failures,
            expected_existing_sha256=expected_sha256,
            current_version=version,
            minimum_candidate_version=_SELF_UPDATE_MIN_VERSION,
        )
        if installed and _is_managed_tirith(installed):
            return "bootstrapped"
        if install_reason == "lazy_installs_disabled":
            return "deferred"
        if install_reason == "candidate_not_newer":
            return "current"
        return "failed"

    provenance, _ = _probe_tirith_provenance(path)
    if provenance is None:
        return "failed"
    if not _provenance_allows_managed_update(path, version, provenance):
        logger.info(
            "tirith background update skipped: binary is not a verified "
            "Hermes-managed release build"
        )
        return "skipped"

    return _run_tirith_update(path)


def _background_update(path: str, *, log_failures: bool = True) -> None:
    """Failure-isolated worker for managed Tirith maintenance."""
    if not _tirith_auto_install_allowed() or not _is_managed_tirith(path):
        return
    lock_fd, lock_status = _acquire_update_lock_with_status()
    if lock_fd is None:
        # Contention means another worker owns the outcome and will persist
        # its own freshness state. Operational lock errors have no such owner;
        # record a short failure backoff so every command cannot spawn a new
        # doomed worker in a long-lived process.
        if lock_status == "error":
            _write_update_state("failed")
        return
    try:
        # Another process may have completed maintenance before we acquired the
        # lock, so check the shared state again inside the critical section.
        if not _update_is_due():
            return
        outcome = _maintain_managed_tirith(path, log_failures=log_failures)
        if outcome != "deferred":
            _write_update_state(
                outcome if outcome in _UPDATE_SUCCESS_OUTCOMES else "failed"
            )
    except Exception as exc:
        if log_failures:
            logger.warning(
                "tirith background update failed unexpectedly; keeping current binary: %s",
                exc,
            )
        else:
            logger.debug("tirith background update failed unexpectedly", exc_info=True)
        _write_update_state("failed")
    finally:
        _release_update_lock(lock_fd)


def _schedule_managed_update(
    path: str,
    configured_path: str,
    *,
    log_failures: bool = True,
    state: _RuntimeState | None = None,
) -> None:
    """Launch at most one non-blocking managed update worker at a time."""
    state = state or _runtime_state(configured_path)
    if (
        not is_platform_supported()
        or _is_explicit_path(configured_path)
        or not _is_managed_tirith(path)
        or not _tirith_auto_install_allowed()
        or not _update_is_due()
    ):
        return
    with state.update_schedule_lock:
        if state.update_thread is not None and state.update_thread.is_alive():
            return
        # Re-check after serializing schedulers. A worker in this process or
        # another Hermes process may have refreshed the shared state while we
        # were waiting for the lock.
        if not _update_is_due():
            return
        worker_context = copy_context()
        thread = threading.Thread(
            target=worker_context.run,
            args=(_background_update, path),
            kwargs={"log_failures": log_failures},
            daemon=True,
        )
        try:
            thread.start()
        except RuntimeError:
            logger.debug(
                "could not start tirith background update thread", exc_info=True
            )
            return
        state.update_thread = thread


def _is_explicit_path(configured_path: str) -> bool:
    """Return True if the user explicitly configured a non-default tirith path."""
    return configured_path != "tirith"


def _resolve_tirith_path(
    configured_path: str,
    *,
    background_only: bool = False,
    state: _RuntimeState | None = None,
) -> str | None:
    """Resolve the tirith binary path, auto-installing if necessary.

    If the user explicitly set a path (anything other than the bare "tirith"
    default), that path is authoritative — we never fall through to
    auto-download a different binary.

    For the default "tirith":
    1. PATH lookup via shutil.which
    2. Hermes' private Tirith cache (previously auto-installed)
    3. Auto-install from GitHub releases into that cache

    Failed installs are cached for the process lifetime (and persisted to
    disk for 24h) to avoid repeated network attempts.
    """
    state = state or _runtime_state(configured_path)

    # Fast path: successfully resolved on a previous call.
    if isinstance(state.resolved_path, str):
        path = state.resolved_path
        if _is_managed_tirith_location(path):
            validated_path = _validated_tirith_path(path)
            if validated_path is None:
                state.resolved_path = _INSTALL_FAILED
                state.install_failure_reason = "managed_cache_untrusted"
            else:
                path = validated_path
                state.resolved_path = path
        if isinstance(state.resolved_path, str):
            # The resolver is exercised for every scan, including in long-lived
            # gateways. Reconsider completed workers here so a Tirith release made
            # after Hermes startup is discovered once the shared TTL expires.
            _schedule_managed_update(
                path,
                configured_path,
                log_failures=False,
                state=state,
            )
            return path

    expanded = os.path.expanduser(configured_path)
    explicit = _is_explicit_path(configured_path)
    install_failed = state.resolved_path is _INSTALL_FAILED

    # Explicit path: check it and stop. Never auto-download a replacement.
    if explicit:
        validated_path = _validated_tirith_path(expanded)
        if validated_path is not None:
            state.resolved_path = validated_path
            return validated_path
        # Also try shutil.which in case it's a bare name on PATH
        found = shutil.which(expanded)
        validated_path = _validated_tirith_path(found) if found else None
        if validated_path is not None:
            state.resolved_path = validated_path
            return validated_path
        logger.warning("Configured tirith path %r not found; scanning disabled", configured_path)
        state.resolved_path = _INSTALL_FAILED
        state.install_failure_reason = "explicit_path_missing"
        return None if background_only else expanded

    # Default "tirith" — always re-run cheap local checks so a manual
    # install is picked up even after a previous network failure (P2 fix:
    # long-lived gateway/CLI recovers without restart).
    found = shutil.which("tirith")
    validated_path = _validated_tirith_path(found) if found else None
    if validated_path is not None:
        state.resolved_path = validated_path
        state.install_failure_reason = ""
        _clear_install_failed()
        _schedule_managed_update(
            validated_path,
            configured_path,
            log_failures=False,
            state=state,
        )
        return validated_path
    if found and _is_managed_tirith_location(found):
        state.resolved_path = _INSTALL_FAILED
        state.install_failure_reason = "managed_cache_untrusted"
        return None if background_only else expanded

    # Platform support controls Hermes' managed cache and installer, not
    # whether an operator-provided Tirith binary may scan commands. Explicit
    # paths and PATH discovery above therefore remain available everywhere.
    if not is_platform_supported():
        state.resolved_path = _INSTALL_FAILED
        state.install_failure_reason = "unsupported_platform"
        return None if background_only else expanded

    hermes_bin = _managed_tirith_path()
    if _is_managed_tirith(hermes_bin):
        state.resolved_path = hermes_bin
        state.install_failure_reason = ""
        _clear_install_failed()
        _schedule_managed_update(
            hermes_bin,
            configured_path,
            log_failures=False,
            state=state,
        )
        return hermes_bin
    if os.path.lexists(hermes_bin):
        state.resolved_path = _INSTALL_FAILED
        state.install_failure_reason = "managed_cache_untrusted"
        return None if background_only else expanded

    # A policy opt-out is not an installation failure. Do not cache or persist
    # it, so changing the setting can take effect immediately in this process.
    if not _tirith_auto_install_allowed():
        return None if background_only else expanded

    # Local checks failed.  If a previous install attempt already failed,
    # skip the network retry — UNLESS the failure was "cosign_missing" and
    # cosign is now available (retryable cause resolved in-process).
    if install_failed:
        if (
            state.install_failure_reason == "cosign_missing"
            and shutil.which("cosign")
        ):
            # Retryable cause resolved — clear sentinel and fall through to retry
            state.resolved_path = None
            state.install_failure_reason = ""
            _clear_install_failed()
            install_failed = False
        else:
            return None if background_only else expanded

    # If a background install thread is running, don't start a parallel one —
    # return the configured path; the OSError handler in check_command_security
    # will apply fail_open until the thread finishes.
    if state.install_thread is not None and state.install_thread.is_alive():
        return None if background_only else expanded

    # Approval is a latency-sensitive path. Startup normally starts this
    # worker first, but alternate entrypoints and very early commands must not
    # turn a release download into a synchronous approval stall.
    if background_only:
        ensure_installed(log_failures=False)
        return None

    # Check disk failure marker before attempting network download.
    # Preserve the marker's real reason so in-memory retry logic can
    # detect retryable causes (e.g. cosign_missing) without restart.
    disk_reason = _read_failure_reason()
    if disk_reason is not None and _is_install_failed_on_disk():
        state.resolved_path = _INSTALL_FAILED
        state.install_failure_reason = disk_reason
        return expanded

    installed, reason = _install_tirith()
    if installed:
        state.resolved_path = installed
        state.install_failure_reason = ""
        _clear_install_failed()
        _write_update_state("installed")
        return installed
    if reason == "lazy_installs_disabled":
        return expanded

    # Install failed — cache the miss and persist reason to disk
    state.resolved_path = _INSTALL_FAILED
    state.install_failure_reason = reason
    if reason != "managed_directory_untrusted":
        _mark_install_failed(reason)
    return expanded


def _background_install(
    *,
    log_failures: bool = True,
    state: _RuntimeState | None = None,
):
    """Background thread target: download and install tirith."""
    state = state or _runtime_state("tirith")
    with state.install_lock:
        # Double-check after acquiring lock (another thread may have resolved)
        if state.resolved_path is not None:
            return

        # Re-check local paths (may have been installed by another process)
        found = shutil.which("tirith")
        validated_path = _validated_tirith_path(found) if found else None
        if validated_path is not None:
            state.resolved_path = validated_path
            state.install_failure_reason = ""
            return
        if found and _is_managed_tirith_location(found):
            state.resolved_path = _INSTALL_FAILED
            state.install_failure_reason = "managed_cache_untrusted"
            return

        hermes_bin = _managed_tirith_path()
        if _is_managed_tirith(hermes_bin):
            state.resolved_path = hermes_bin
            state.install_failure_reason = ""
            return
        if os.path.lexists(hermes_bin):
            state.resolved_path = _INSTALL_FAILED
            state.install_failure_reason = "managed_cache_untrusted"
            return

        if not _tirith_auto_install_allowed():
            return

        installed, reason = _install_tirith(log_failures=log_failures)
        if installed:
            state.resolved_path = installed
            state.install_failure_reason = ""
            _clear_install_failed()
            _write_update_state("installed")
        elif reason == "lazy_installs_disabled":
            return
        else:
            state.resolved_path = _INSTALL_FAILED
            state.install_failure_reason = reason
            if reason != "managed_directory_untrusted":
                _mark_install_failed(reason)


def ensure_installed(*, log_failures: bool = True):
    """Ensure tirith is available, downloading in background if needed.

    Quick PATH/local checks are synchronous; network download runs in a
    daemon thread so startup never blocks. Safe to call multiple times.
    Returns the resolved path immediately if available, or None.
    """
    cfg = _load_security_config()
    if not cfg["tirith_enabled"]:
        return None
    configured_path = cfg["tirith_path"]
    state = _runtime_state(configured_path)

    # Already resolved from a previous call
    if isinstance(state.resolved_path, str):
        path = state.resolved_path
        validated_path = _validated_tirith_path(path)
        if validated_path is not None:
            state.resolved_path = validated_path
            _schedule_managed_update(
                validated_path,
                configured_path,
                log_failures=log_failures,
                state=state,
            )
            return validated_path
        if not _is_managed_tirith_location(path):
            return None
        state.resolved_path = _INSTALL_FAILED
        state.install_failure_reason = "managed_cache_untrusted"

    explicit = _is_explicit_path(configured_path)
    expanded = os.path.expanduser(configured_path)

    # Explicit path: synchronous check only, no download
    if explicit:
        validated_path = _validated_tirith_path(expanded)
        if validated_path is not None:
            state.resolved_path = validated_path
            return validated_path
        found = shutil.which(expanded)
        validated_path = _validated_tirith_path(found) if found else None
        if validated_path is not None:
            state.resolved_path = validated_path
            return validated_path
        state.resolved_path = _INSTALL_FAILED
        state.install_failure_reason = "explicit_path_missing"
        return None

    # Default "tirith" — quick local checks first (no network)
    found = shutil.which("tirith")
    validated_path = _validated_tirith_path(found) if found else None
    if validated_path is not None:
        state.resolved_path = validated_path
        state.install_failure_reason = ""
        _clear_install_failed()
        _schedule_managed_update(
            validated_path,
            configured_path,
            log_failures=log_failures,
            state=state,
        )
        return validated_path
    if found and _is_managed_tirith_location(found):
        state.resolved_path = _INSTALL_FAILED
        state.install_failure_reason = "managed_cache_untrusted"
        return None

    # Unsupported manager targets may still use explicit or PATH binaries,
    # but Hermes must not inspect its managed cache or start an installer for
    # an archive format it does not support.
    if not is_platform_supported():
        state.resolved_path = _INSTALL_FAILED
        state.install_failure_reason = "unsupported_platform"
        return None

    hermes_bin = _managed_tirith_path()
    if _is_managed_tirith(hermes_bin):
        state.resolved_path = hermes_bin
        state.install_failure_reason = ""
        _clear_install_failed()
        _schedule_managed_update(
            hermes_bin,
            configured_path,
            log_failures=log_failures,
            state=state,
        )
        return hermes_bin
    if os.path.lexists(hermes_bin):
        state.resolved_path = _INSTALL_FAILED
        state.install_failure_reason = "managed_cache_untrusted"
        return None

    # Preserve local discovery while honoring the global runtime-install
    # policy. Keep state unresolved so an in-process config change is enough
    # to enable a later attempt.
    if not _tirith_auto_install_allowed():
        return None

    # If previously failed in-memory, check if the cause is now resolved
    if state.resolved_path is _INSTALL_FAILED:
        if (
            state.install_failure_reason == "cosign_missing"
            and shutil.which("cosign")
        ):
            state.resolved_path = None
            state.install_failure_reason = ""
            _clear_install_failed()
        else:
            return None

    # Check disk failure marker (skip network attempt for 24h, unless
    # the cosign_missing reason was resolved — handled by _is_install_failed_on_disk).
    # Preserve the marker's real reason for in-memory retry logic.
    disk_reason = _read_failure_reason()
    if disk_reason is not None and _is_install_failed_on_disk():
        state.resolved_path = _INSTALL_FAILED
        state.install_failure_reason = disk_reason
        return None

    # Need to download — launch background thread so startup doesn't block
    if state.install_thread is None or not state.install_thread.is_alive():
        worker_context = copy_context()
        state.install_thread = threading.Thread(
            target=worker_context.run,
            args=(_background_install,),
            kwargs={"log_failures": log_failures, "state": state},
            daemon=True,
        )
        state.install_thread.start()

    return None  # Not available yet; commands will fail-open until ready


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

_MAX_FINDINGS = 50
_MAX_SUMMARY_LEN = 500


def check_command_security(command: str) -> dict:
    """Run tirith security scan on a command.

    Exit code determines action (0=allow, 1=block, 2=warn). JSON enriches
    findings/summary. Spawn failures and timeouts respect fail_open config.
    Programming errors propagate.

    Returns:
        {"action": "allow"|"warn"|"block", "findings": [...], "summary": str}
    """
    cfg = _load_security_config()

    if not cfg["tirith_enabled"]:
        return {"action": "allow", "findings": [], "summary": ""}
    state = _runtime_state(cfg["tirith_path"])

    # Circuit breaker: pause after repeated failures, then make a half-open
    # recovery attempt. Without this, a corrupted binary can make every tool
    # call hit the same slow failure; without the retry, a repaired or updated
    # binary stays disabled for the rest of a long-lived process.
    scan_allowed, claimed_probe = _circuit_scan_admission(state)
    if not scan_allowed:
        action = "allow" if cfg["tirith_fail_open"] else "block"
        return {
            "action": action,
            "findings": [],
            "summary": f"tirith unavailable (circuit breaker, fail-{'open' if action == 'allow' else 'closed'})",
        }

    try:
        return _check_command_security_with_state(command, cfg, state)
    finally:
        if claimed_probe:
            _finish_circuit_probe(state)


def _check_command_security_with_state(
    command: str,
    cfg: dict,
    state: _RuntimeState,
) -> dict:
    """Execute one scan after the caller applies circuit-breaker policy."""

    tirith_path = _resolve_tirith_path(
        cfg["tirith_path"], background_only=True, state=state
    )
    timeout = cfg["tirith_timeout"]
    fail_open = cfg["tirith_fail_open"]

    if tirith_path is None:
        unsupported_manager = state.install_failure_reason == "unsupported_platform"
        if not unsupported_manager:
            _warn_once(
                f"{state.key!r}:tirith_path_none",
                "tirith path resolved to None; scanning disabled",
            )
        if fail_open:
            summary = "" if unsupported_manager else "tirith path unavailable"
            return {"action": "allow", "findings": [], "summary": summary}
        return {"action": "block", "findings": [], "summary": "tirith path unavailable (fail-closed)"}

    # Managed cache ownership is re-proved immediately before execution. This
    # catches mode/ACL drift after path resolution without imposing Hermes'
    # private-cache policy on explicit or package-manager binaries.
    if _is_managed_tirith_location(tirith_path):
        if not _is_managed_tirith(tirith_path):
            if state.resolved_path == tirith_path:
                state.resolved_path = _INSTALL_FAILED
                state.install_failure_reason = "managed_cache_untrusted"
            action = "allow" if fail_open else "block"
            return {
                "action": action,
                "findings": [],
                "summary": "tirith managed cache is untrusted "
                f"(fail-{'open' if fail_open else 'closed'})",
            }
        # Never execute a PATH/config alias after using file identity to
        # classify it. The exact owned path has the validated parent chain and
        # cannot be swapped through an attacker-controlled alias directory.
        original_path = tirith_path
        tirith_path = os.path.abspath(_managed_tirith_path())
        if state.resolved_path == original_path:
            state.resolved_path = tirith_path

    try:
        result = subprocess.run(
            [tirith_path, "check", "--json", "--non-interactive",
             "--shell", "posix", "--", command],
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            env=_tirith_subprocess_env(),
        )
    except OSError as exc:
        # Covers FileNotFoundError, PermissionError, exec format error.
        # Invalidate only the path this call actually tried. A managed binary
        # may have been repaired concurrently, in which case its newer cached
        # path must win. Clearing a stale cache lets the next command re-run
        # local discovery (and, when allowed, start managed recovery).
        if state.resolved_path == tirith_path:
            state.resolved_path = None
        # Dedupe by ``(errno, exc class)`` so a transient failure mode
        # surfaces once but doesn't drown the log on every command —
        # commonly seen on Windows when the configured path "tirith"
        # isn't on PATH yet (background install still running, or
        # install marked failed for the day).
        spawn_key = f"tirith_spawn_failed:{type(exc).__name__}:{getattr(exc, 'errno', '')}"
        _warn_once(f"{state.key!r}:{spawn_key}", "tirith spawn failed: %s", exc)
        _record_tirith_crash(state)
        if fail_open:
            return {"action": "allow", "findings": [], "summary": f"tirith unavailable: {exc}"}
        return {"action": "block", "findings": [], "summary": f"tirith spawn failed (fail-closed): {exc}"}
    except subprocess.TimeoutExpired:
        _warn_once(
            f"{state.key!r}:tirith_timeout:{timeout}",
            "tirith timed out after %ds",
            timeout,
        )
        _record_tirith_crash(state)
        if fail_open:
            return {"action": "allow", "findings": [], "summary": f"tirith timed out ({timeout}s)"}
        return {"action": "block", "findings": [], "summary": "tirith timed out (fail-closed)"}

    # Map exit code to action
    exit_code = result.returncode
    if exit_code in (0, 1, 2):
        # A recognized verdict proves the scanner is responsive. This must
        # reset failures for warn/block verdicts too; otherwise unrelated
        # earlier failures accumulate and open a supposedly consecutive
        # failure breaker.
        _reset_tirith_crash_state(state)

    if exit_code == 0:
        action = "allow"
    elif exit_code == 1:
        action = "block"
    elif exit_code == 2:
        action = "warn"
    else:
        # Unknown exit code (includes signal-killed processes like -11/SIGSEGV)
        # — respect fail_open
        logger.warning("tirith returned unexpected exit code %d", exit_code)
        _record_tirith_crash(state)
        if fail_open:
            return {"action": "allow", "findings": [], "summary": f"tirith exit code {exit_code} (fail-open)"}
        return {"action": "block", "findings": [], "summary": f"tirith exit code {exit_code} (fail-closed)"}

    # Parse JSON for enrichment (never overrides the exit code verdict)
    findings = []
    raw_findings = []
    summary = ""
    try:
        data = json.loads(result.stdout) if result.stdout.strip() else {}
        raw_findings = data.get("findings", [])
        findings = raw_findings[:_MAX_FINDINGS]
        summary = (data.get("summary", "") or "")[:_MAX_SUMMARY_LEN]
    except (json.JSONDecodeError, AttributeError):
        # JSON parse failure degrades findings/summary, not the verdict
        logger.debug("tirith JSON parse failed, using exit code only")
        if action == "block":
            summary = "security issue detected (details unavailable)"
        elif action == "warn":
            summary = "security warning detected (details unavailable)"

    # Suppress warn verdicts that consist solely of a lookalike_tld finding for
    # the .app TLD.  .app is a legitimate gTLD used by many production services
    # and the "can be confused with file extensions" heuristic generates false
    # positives for normal API calls.  Any other finding (including other
    # lookalike_tld entries for non-.app TLDs) preserves the warn action.
    if action == "warn" and raw_findings:
        non_suppressible = [
            f for f in raw_findings if not _is_app_tld_finding(f)
        ]
        if not non_suppressible:
            action = "allow"
            findings = []
            summary = ""

    return {"action": action, "findings": findings, "summary": summary}


def _is_app_tld_finding(finding: dict) -> bool:
    """Return True if this finding is a lookalike_tld warning for the .app TLD only.

    Checks the rule_id and inspects common value/detail field names that
    Tirith may use to carry the TLD string.
    """
    if not isinstance(finding, dict):
        return False
    if finding.get("rule_id") != "lookalike_tld":
        return False
    for field in ("value", "tld", "detail"):
        val = finding.get(field)
        if val is not None and str(val).strip().casefold() in {"app", ".app"}:
            return True
    for field in ("description", "message"):
        val = finding.get(field)
        if val is not None and re.search(
            r"(?i)(?:^|[\s\"'])\.app[\"']?\s+(?:tld|top-level domain)\b",
            str(val),
        ):
            return True
    return False
