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
"""

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
from typing import Protocol, TypedDict, cast

from hermes_constants import get_hermes_home, is_termux

logger = logging.getLogger(__name__)

_REPO = "sheeki03/tirith"

# Cosign provenance verification — pinned to the specific release workflow
_COSIGN_IDENTITY_REGEXP = f"^https://github.com/{_REPO}/\\.github/workflows/release\\.yml@refs/tags/v"
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


# Cached path after first resolution (avoids repeated shutil.which per command).
# _INSTALL_FAILED means "we tried and failed" — prevents retry on every command.
_resolved_path: str | None | bool = None
_INSTALL_FAILED = False  # sentinel: distinct from "not yet tried"
_install_failure_reason: str = ""  # reason tag when _resolved_path is _INSTALL_FAILED

# Circuit breaker: after _CRASH_LIMIT consecutive spawn/execution failures,
# disable tirith for the rest of the process to prevent agent hangs (#41400).
# Reset on successful execution (see _record_tirith_crash / check_command_security).
#
# Thread safety: _crash_count and _circuit_open are module-level globals
# mutated without a lock. check_command_security can be called from
# concurrent agent threads (gateway multi-session). The race is benign —
# at worst two threads both increment past _CRASH_LIMIT and both set
# _circuit_open = True, opening the breaker one call early. No data
# corruption or security bypass is possible. This intentionally matches
# the lock-free style of error counters in mcp_tool.py rather than the
# locked _warn_once pattern, because the worst case is harmless.
_CRASH_LIMIT = 3
_crash_count: int = 0
_circuit_open: bool = False


def _record_tirith_crash() -> None:
    """Increment the crash counter and open the circuit breaker if needed."""
    global _crash_count, _circuit_open
    _crash_count += 1
    if _crash_count >= _CRASH_LIMIT:
        _circuit_open = True
        logger.warning(
            "tirith circuit breaker opened after %d consecutive failures; "
            "disabling for the rest of the process",
            _crash_count,
        )

# Background install thread coordination
_install_lock = threading.Lock()
_install_thread: threading.Thread | None = None

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

_update_schedule_lock = threading.Lock()
_update_thread: threading.Thread | None = None


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
    return True


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


def _is_managed_tirith(path: str) -> bool:
    """Return whether ``path`` is inside Hermes' real managed-bin boundary."""
    expected = os.path.normcase(os.path.abspath(_managed_tirith_path()))
    candidate = os.path.normcase(os.path.abspath(path))
    if candidate != expected:
        return False
    try:
        return _managed_install_directory_is_real() and stat.S_ISREG(
            os.lstat(path).st_mode
        )
    except OSError:
        return False


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


def _acquire_update_lock():
    """Acquire a process-bound advisory lock, or return ``None`` if busy.

    Tirith only ships on POSIX platforms. ``flock`` releases automatically on
    process death, so there is no stale-file reclamation race and a suspended
    updater cannot be mistaken for a dead one.
    """
    if os.name == "nt":
        return None

    import fcntl

    path = _update_process_lock_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except OSError:
        return None

    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
    except OSError:
        return None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            return None
        os.fchmod(fd, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, BlockingIOError):
        os.close(fd)
        return None
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
    """Remove the failure marker after successful install."""
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
        req.add_header("Authorization", f"token {token}")
    written = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as f:
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


def _verify_cosign(checksums_path: str, sig_path: str, cert_path: str) -> bool | None:
    """Verify cosign provenance signature on checksums.txt.

    Returns:
        True  — cosign verified successfully
        False — cosign found but verification failed
        None  — cosign not available (not on PATH, or execution failed)

    ``False`` is an explicit verification rejection. ``None`` lets the caller
    use Hermes' documented SHA-256-only fallback.
    """
    cosign = shutil.which("cosign")
    if not cosign:
        logger.info("cosign not found on PATH")
        return None

    try:
        result = subprocess.run(
            [cosign, "verify-blob",
             "--certificate", cert_path,
             "--signature", sig_path,
             "--certificate-identity-regexp", _COSIGN_IDENTITY_REGEXP,
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
            return True
        else:
            logger.warning("cosign verification failed (exit %d): %s",
                          result.returncode, result.stderr.strip())
            return False
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("cosign execution failed: %s", exc)
        return None


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
) -> None:
    """Stage ``source`` beside ``destination`` and atomically commit it.

    Download extraction happens in the system temporary directory, which may
    be on another filesystem. Copying into a sibling first makes the final
    ``os.replace`` atomic and preserves an existing working scanner if any
    staging or commit step fails.
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
) -> tuple[str | None, str, bool]:
    """Download, verify, and extract one Tirith release into ``workdir``."""
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
        return None, "download_failed", False

    cosign_verified = False
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
            logger.info(
                "cosign artifacts unavailable (%s), proceeding with SHA-256 only", exc
            )
        else:
            cosign_result = _verify_cosign(checksums_path, sig_path, cert_path)
            if cosign_result is True:
                cosign_verified = True
            elif cosign_result is False:
                log("tirith release rejected: cosign provenance verification failed")
                return None, "cosign_verification_failed", False
            else:
                logger.info("cosign execution failed, proceeding with SHA-256 only")
    else:
        logger.info(
            "cosign not on PATH — using SHA-256 verification only "
            "(install cosign for full supply chain verification)"
        )

    if not _verify_checksum(archive_path, checksums_path, archive_name):
        return None, "checksum_failed", cosign_verified

    src, reason = _extract_release_archive(archive_path, workdir, log)
    return src, reason, cosign_verified


def _install_tirith(
    *,
    log_failures: bool = True,
    expected_existing_sha256: str | None = None,
) -> tuple[str | None, str]:
    """Download and install Tirith to Hermes' private managed cache.

    Always verifies the SHA-256 checksum and verifies cosign provenance when
    cosign is available.
    Returns (installed_path, failure_reason).  On success failure_reason is "".
    failure_reason is a short tag used by the disk marker to decide if the
    failure is retryable (e.g. "cosign_missing" clears when cosign appears).
    """
    if not _tirith_auto_install_allowed():
        return None, "lazy_installs_disabled"

    log = logger.warning if log_failures else logger.debug

    target = _detect_target()
    if not target:
        logger.info("tirith auto-install: unsupported platform %s/%s",
                     platform.system(), platform.machine())
        return None, "unsupported_platform"

    base_url = f"https://github.com/{_REPO}/releases/latest/download"

    try:
        tmpdir = tempfile.mkdtemp(prefix="tirith-install-")
    except OSError as exc:
        log("tirith install failed: cannot create temp dir: %s", exc)
        return None, "no_space"
    try:
        logger.info("tirith not found — downloading latest release for %s...", target)
        src, reason, cosign_verified = _download_verified_tirith(
            base_url, target, tmpdir, log
        )
        if src is None:
            return None, reason

        # Config is live and the verified download may have taken seconds.
        # Re-check before creating or replacing anything under HERMES_HOME.
        if not _tirith_auto_install_allowed():
            return None, "lazy_installs_disabled"

        try:
            dest = os.path.join(_hermes_bin_dir(), "tirith")
        except OSError as exc:
            log("tirith install aborted: untrusted managed directory: %s", exc)
            return None, "managed_directory_untrusted"
        if not _managed_install_directory_is_real():
            log("tirith install aborted: Hermes managed-bin directory is redirected")
            return None, "managed_directory_untrusted"
        try:
            _atomic_replace_binary(
                src,
                dest,
                expected_existing_sha256=expected_existing_sha256,
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
    log_failures: bool = True,
) -> tuple[str | None, str]:
    """Prove a pre-Hermes-provenance binary matches published release bytes.

    Legacy Tirith cannot attest whether it is a release or local build. Hermes
    therefore verifies the matching tagged archive before replacing it. The
    returned digest binds that proof to the later atomic swap.
    """
    target = _detect_target()
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
        released, reason, _cosign_verified = _download_verified_tirith(
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
    # The user may disable runtime installs while the background worker is
    # probing version/provenance. Re-check at the mutating/network sink so the
    # live opt-out takes effect without waiting for the worker to finish.
    if not _tirith_auto_install_allowed():
        return "deferred"
    try:
        result = subprocess.run(
            [path, "update", "--yes", "--allow-unsigned", "--format", "json"],
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


def _maintain_managed_tirith(path: str, *, log_failures: bool = True) -> str:
    """Bootstrap or update an existing Hermes-managed Tirith binary."""
    if not _is_managed_tirith(path):
        return "skipped"

    version, reason = _probe_tirith_version(path)
    if version is None:
        if reason == "unparseable":
            logger.info("tirith background update skipped: unrecognized build version")
            return "skipped"
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
        )
        if installed and _is_managed_tirith(installed):
            return "bootstrapped"
        if install_reason == "lazy_installs_disabled":
            return "deferred"
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

    # Tirith 0.4.1 reports every AArch64 Linux build as the glibc release
    # target.  On Termux, delegating to its self-updater would therefore fetch
    # an unusable glibc binary over the working musl one.  Keep the same
    # provenance gate, then use Hermes' checksum-verified, preimage-bound
    # installer until Tirith can distinguish the musl target itself.
    if (
        _detect_target() == "aarch64-unknown-linux-musl"
        and provenance.get("target") != "aarch64-unknown-linux-musl"
    ):
        try:
            expected_sha256 = _sha256_file(path)
        except OSError:
            return "failed"
        installed, install_reason = _install_tirith(
            log_failures=log_failures,
            expected_existing_sha256=expected_sha256,
        )
        if installed and _is_managed_tirith(installed):
            return "updated"
        if install_reason == "lazy_installs_disabled":
            return "deferred"
        return "failed"

    return _run_tirith_update(path)


def _background_update(path: str, *, log_failures: bool = True) -> None:
    """Failure-isolated worker for managed Tirith maintenance."""
    if not _tirith_auto_install_allowed() or not _is_managed_tirith(path):
        return
    lock_fd = _acquire_update_lock()
    if lock_fd is None:
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
    path: str, configured_path: str, *, log_failures: bool = True
) -> None:
    """Launch at most one non-blocking managed update worker at a time."""
    global _update_thread
    if (
        _is_explicit_path(configured_path)
        or not _is_managed_tirith(path)
        or not _tirith_auto_install_allowed()
        or not _update_is_due()
    ):
        return
    with _update_schedule_lock:
        if _update_thread is not None and _update_thread.is_alive():
            return
        # Re-check after serializing schedulers. A worker in this process or
        # another Hermes process may have refreshed the shared state while we
        # were waiting for the lock.
        if not _update_is_due():
            return
        thread = threading.Thread(
            target=_background_update,
            args=(path,),
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
        _update_thread = thread


def _is_explicit_path(configured_path: str) -> bool:
    """Return True if the user explicitly configured a non-default tirith path."""
    return configured_path != "tirith"


def _resolve_tirith_path(configured_path: str) -> str:
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
    global _resolved_path, _install_failure_reason

    # Fast path: successfully resolved on a previous call.
    if isinstance(_resolved_path, str):
        path = _resolved_path
        # The resolver is exercised for every scan, including in long-lived
        # gateways. Reconsider completed workers here so a Tirith release made
        # after Hermes startup is discovered once the shared TTL expires.
        _schedule_managed_update(path, configured_path, log_failures=False)
        return path

    expanded = os.path.expanduser(configured_path)
    explicit = _is_explicit_path(configured_path)
    install_failed = _resolved_path is _INSTALL_FAILED

    # Platform is unsupported by Hermes' Tirith manager. Cache the verdict and
    # return the unexpanded configured path; the spawn loop will fail-open via
    # the dedupe'd OSError handler.
    if not explicit and not is_platform_supported():
        _resolved_path = _INSTALL_FAILED
        _install_failure_reason = "unsupported_platform"
        return expanded

    # Explicit path: check it and stop. Never auto-download a replacement.
    if explicit:
        if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
            _resolved_path = expanded
            return expanded
        # Also try shutil.which in case it's a bare name on PATH
        found = shutil.which(expanded)
        if found:
            _resolved_path = found
            return found
        logger.warning("Configured tirith path %r not found; scanning disabled", configured_path)
        _resolved_path = _INSTALL_FAILED
        _install_failure_reason = "explicit_path_missing"
        return expanded

    # Default "tirith" — always re-run cheap local checks so a manual
    # install is picked up even after a previous network failure (P2 fix:
    # long-lived gateway/CLI recovers without restart).
    found = shutil.which("tirith")
    if found:
        _resolved_path = found
        _install_failure_reason = ""
        _clear_install_failed()
        _schedule_managed_update(found, configured_path, log_failures=False)
        return found

    hermes_bin = _managed_tirith_path()
    if os.path.isfile(hermes_bin) and os.access(hermes_bin, os.X_OK):
        _resolved_path = hermes_bin
        _install_failure_reason = ""
        _clear_install_failed()
        _schedule_managed_update(hermes_bin, configured_path, log_failures=False)
        return hermes_bin

    # A policy opt-out is not an installation failure. Do not cache or persist
    # it, so changing the setting can take effect immediately in this process.
    if not _tirith_auto_install_allowed():
        return expanded

    # Local checks failed.  If a previous install attempt already failed,
    # skip the network retry — UNLESS the failure was "cosign_missing" and
    # cosign is now available (retryable cause resolved in-process).
    if install_failed:
        if _install_failure_reason == "cosign_missing" and shutil.which("cosign"):
            # Retryable cause resolved — clear sentinel and fall through to retry
            _resolved_path = None
            _install_failure_reason = ""
            _clear_install_failed()
            install_failed = False
        else:
            return expanded

    # If a background install thread is running, don't start a parallel one —
    # return the configured path; the OSError handler in check_command_security
    # will apply fail_open until the thread finishes.
    if _install_thread is not None and _install_thread.is_alive():
        return expanded

    # Check disk failure marker before attempting network download.
    # Preserve the marker's real reason so in-memory retry logic can
    # detect retryable causes (e.g. cosign_missing) without restart.
    disk_reason = _read_failure_reason()
    if disk_reason is not None and _is_install_failed_on_disk():
        _resolved_path = _INSTALL_FAILED
        _install_failure_reason = disk_reason
        return expanded

    installed, reason = _install_tirith()
    if installed:
        _resolved_path = installed
        _install_failure_reason = ""
        _clear_install_failed()
        _write_update_state("installed")
        return installed
    if reason == "lazy_installs_disabled":
        return expanded

    # Install failed — cache the miss and persist reason to disk
    _resolved_path = _INSTALL_FAILED
    _install_failure_reason = reason
    if reason != "managed_directory_untrusted":
        _mark_install_failed(reason)
    return expanded


def _background_install(*, log_failures: bool = True):
    """Background thread target: download and install tirith."""
    global _resolved_path, _install_failure_reason
    with _install_lock:
        # Double-check after acquiring lock (another thread may have resolved)
        if _resolved_path is not None:
            return

        # Re-check local paths (may have been installed by another process)
        found = shutil.which("tirith")
        if found:
            _resolved_path = found
            _install_failure_reason = ""
            return

        hermes_bin = _managed_tirith_path()
        if os.path.isfile(hermes_bin) and os.access(hermes_bin, os.X_OK):
            _resolved_path = hermes_bin
            _install_failure_reason = ""
            return

        if not _tirith_auto_install_allowed():
            return

        installed, reason = _install_tirith(log_failures=log_failures)
        if installed:
            _resolved_path = installed
            _install_failure_reason = ""
            _clear_install_failed()
            _write_update_state("installed")
        elif reason == "lazy_installs_disabled":
            return
        else:
            _resolved_path = _INSTALL_FAILED
            _install_failure_reason = reason
            if reason != "managed_directory_untrusted":
                _mark_install_failed(reason)


def ensure_installed(*, log_failures: bool = True):
    """Ensure tirith is available, downloading in background if needed.

    Quick PATH/local checks are synchronous; network download runs in a
    daemon thread so startup never blocks. Safe to call multiple times.
    Returns the resolved path immediately if available, or None.
    """
    global _resolved_path, _install_thread, _install_failure_reason

    cfg = _load_security_config()
    if not cfg["tirith_enabled"]:
        return None

    # Already resolved from a previous call
    if isinstance(_resolved_path, str):
        path = _resolved_path
        if os.path.isfile(path) and os.access(path, os.X_OK):
            _schedule_managed_update(
                path,
                cfg["tirith_path"],
                log_failures=log_failures,
            )
            return path
        return None

    # Platform is unsupported by Hermes' Tirith manager — don't probe PATH,
    # start a download thread, or write a disk failure marker. Pattern-matching
    # guards still run; this path stays silent.
    if not is_platform_supported():
        _resolved_path = _INSTALL_FAILED
        _install_failure_reason = "unsupported_platform"
        return None

    configured_path = cfg["tirith_path"]
    explicit = _is_explicit_path(configured_path)
    expanded = os.path.expanduser(configured_path)

    # Explicit path: synchronous check only, no download
    if explicit:
        if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
            _resolved_path = expanded
            return expanded
        found = shutil.which(expanded)
        if found:
            _resolved_path = found
            return found
        _resolved_path = _INSTALL_FAILED
        _install_failure_reason = "explicit_path_missing"
        return None

    # Default "tirith" — quick local checks first (no network)
    found = shutil.which("tirith")
    if found:
        _resolved_path = found
        _install_failure_reason = ""
        _clear_install_failed()
        _schedule_managed_update(
            found,
            configured_path,
            log_failures=log_failures,
        )
        return found

    hermes_bin = _managed_tirith_path()
    if os.path.isfile(hermes_bin) and os.access(hermes_bin, os.X_OK):
        _resolved_path = hermes_bin
        _install_failure_reason = ""
        _clear_install_failed()
        _schedule_managed_update(
            hermes_bin,
            configured_path,
            log_failures=log_failures,
        )
        return hermes_bin

    # Preserve local discovery while honoring the global runtime-install
    # policy. Keep state unresolved so an in-process config change is enough
    # to enable a later attempt.
    if not _tirith_auto_install_allowed():
        return None

    # If previously failed in-memory, check if the cause is now resolved
    if _resolved_path is _INSTALL_FAILED:
        if _install_failure_reason == "cosign_missing" and shutil.which("cosign"):
            _resolved_path = None
            _install_failure_reason = ""
            _clear_install_failed()
        else:
            return None

    # Check disk failure marker (skip network attempt for 24h, unless
    # the cosign_missing reason was resolved — handled by _is_install_failed_on_disk).
    # Preserve the marker's real reason for in-memory retry logic.
    disk_reason = _read_failure_reason()
    if disk_reason is not None and _is_install_failed_on_disk():
        _resolved_path = _INSTALL_FAILED
        _install_failure_reason = disk_reason
        return None

    # Need to download — launch background thread so startup doesn't block
    if _install_thread is None or not _install_thread.is_alive():
        _install_thread = threading.Thread(
            target=_background_install,
            kwargs={"log_failures": log_failures},
            daemon=True,
        )
        _install_thread.start()

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
    global _crash_count, _circuit_open

    cfg = _load_security_config()

    if not cfg["tirith_enabled"]:
        return {"action": "allow", "findings": [], "summary": ""}

    # Circuit breaker: if tirith has crashed _CRASH_LIMIT times in a row,
    # stop trying for the rest of the process.  Without this, a corrupted
    # or missing binary causes every tool call to hit the same spawn failure
    # → fail-open → agent retry loop, hanging the user for 20+ minutes
    # (issue #41400).
    if _circuit_open:
        return {"action": "allow", "findings": [], "summary": "tirith disabled (circuit breaker)"}

    # Unsupported manager platform (currently native Windows and unknown
    # architectures). Skip the resolver entirely; pattern-matching guards
    # still run via the rest of approval.py.
    if not is_platform_supported():
        return {"action": "allow", "findings": [], "summary": ""}

    tirith_path = _resolve_tirith_path(cfg["tirith_path"])
    timeout = cfg["tirith_timeout"]
    fail_open = cfg["tirith_fail_open"]

    if tirith_path is None:
        _warn_once(
            "tirith_path_none",
            "tirith path resolved to None; scanning disabled",
        )
        if fail_open:
            return {"action": "allow", "findings": [], "summary": "tirith path unavailable"}
        return {"action": "block", "findings": [], "summary": "tirith path unavailable (fail-closed)"}

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
        # Dedupe by ``(errno, exc class)`` so a transient failure mode
        # surfaces once but doesn't drown the log on every command —
        # commonly seen on Windows when the configured path "tirith"
        # isn't on PATH yet (background install still running, or
        # install marked failed for the day).
        spawn_key = f"tirith_spawn_failed:{type(exc).__name__}:{getattr(exc, 'errno', '')}"
        _warn_once(spawn_key, "tirith spawn failed: %s", exc)
        _record_tirith_crash()
        if fail_open:
            return {"action": "allow", "findings": [], "summary": f"tirith unavailable: {exc}"}
        return {"action": "block", "findings": [], "summary": f"tirith spawn failed (fail-closed): {exc}"}
    except subprocess.TimeoutExpired:
        _warn_once(
            f"tirith_timeout:{timeout}",
            "tirith timed out after %ds",
            timeout,
        )
        _record_tirith_crash()
        if fail_open:
            return {"action": "allow", "findings": [], "summary": f"tirith timed out ({timeout}s)"}
        return {"action": "block", "findings": [], "summary": "tirith timed out (fail-closed)"}

    # Map exit code to action
    exit_code = result.returncode
    if exit_code == 0:
        action = "allow"
        # Successful execution — reset circuit breaker
        _crash_count = 0
    elif exit_code == 1:
        action = "block"
    elif exit_code == 2:
        action = "warn"
    else:
        # Unknown exit code (includes signal-killed processes like -11/SIGSEGV)
        # — respect fail_open
        logger.warning("tirith returned unexpected exit code %d", exit_code)
        _record_tirith_crash()
        if fail_open:
            return {"action": "allow", "findings": [], "summary": f"tirith exit code {exit_code} (fail-open)"}
        return {"action": "block", "findings": [], "summary": f"tirith exit code {exit_code} (fail-closed)"}

    # Parse JSON for enrichment (never overrides the exit code verdict)
    findings = []
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
    if action == "warn" and findings:
        non_suppressible = [f for f in findings if not _is_app_tld_finding(f)]
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
    for field in ("value", "tld", "detail", "description", "message"):
        val = finding.get(field)
        if val is not None and ".app" in str(val).lower():
            return True
    return False
