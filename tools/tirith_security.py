"""Tirith pre-exec security scanning wrapper.

Runs the tirith binary as a subprocess to scan commands for content-level
threats (homograph URLs, pipe-to-interpreter, terminal injection, etc.).

Exit code is the verdict source of truth:
  0 = allow, 1 = block, 2 = warn

JSON stdout enriches findings/summary but never overrides the verdict.
Operational failures (spawn error, timeout, unknown exit code) respect
the fail_open config setting. Programming errors propagate.

Auto-install: if tirith is not found on PATH or at the configured path,
it is automatically downloaded from GitHub releases to $HERMES_HOME/bin/tirith.
The download always verifies SHA-256 checksums.  When cosign is available on
PATH, provenance verification (GitHub Actions workflow signature) is also
performed.  If cosign is not installed, the download proceeds with SHA-256
verification only — still secure via HTTPS + checksum, just without supply
chain provenance proof.  Installation runs in a background thread so startup
never blocks.
"""

import hashlib
import json
import logging
import os
import platform
import shutil
import stat
import subprocess
import tarfile
import tempfile
import threading
import time
import urllib.request

from hermes_constants import get_hermes_home

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
    return val.lower() in ("1", "true", "yes")


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
        from hermes_cli.config import load_config
        cfg = load_config().get("security", {}) or {}
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

# Cached path after first resolution (avoids repeated shutil.which per command).
# _INSTALL_FAILED means "we tried and failed" — prevents retry on every command.
_resolved_path: str | None | bool = None
_INSTALL_FAILED = False  # sentinel: distinct from "not yet tried"
_install_failure_reason: str = ""  # reason tag when _resolved_path is _INSTALL_FAILED

# Background install thread coordination
_install_lock = threading.Lock()
_install_thread: threading.Thread | None = None

# Disk-persistent failure marker — avoids retry across process restarts
_MARKER_TTL = 86400  # 24 hours


def _get_hermes_home() -> str:
    """Return the Hermes home directory, respecting HERMES_HOME env var."""
    return str(get_hermes_home())


def _failure_marker_path() -> str:
    """Return the path to the install-failure marker file."""
    return os.path.join(_get_hermes_home(), ".tirith-install-failed")


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
        with open(p, "r") as f:
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
        with open(p, "w") as f:
            f.write(reason)
    except OSError:
        pass


def _clear_install_failed():
    """Remove the failure marker after successful install."""
    try:
        os.unlink(_failure_marker_path())
    except OSError:
        pass


def _hermes_bin_dir() -> str:
    """Return $HERMES_HOME/bin, creating it if needed."""
    d = os.path.join(_get_hermes_home(), "bin")
    os.makedirs(d, exist_ok=True)
    return d


def _detect_target() -> str | None:
    """Return the Rust target triple for the current platform, or None."""
    system = platform.system()
    machine = platform.machine().lower()

    if system == "Darwin":
        plat = "apple-darwin"
    elif system == "Linux":
        plat = "unknown-linux-gnu"
    else:
        return None

    if machine in ("x86_64", "amd64"):
        arch = "x86_64"
    elif machine in ("aarch64", "arm64"):
        arch = "aarch64"
    else:
        return None

    return f"{arch}-{plat}"


def _download_file(url: str, dest: str, timeout: int = 10):
    """Download a URL to a local file."""
    req = urllib.request.Request(url)
    token = os.getenv("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"token {token}")
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)


def _verify_cosign(checksums_path: str, sig_path: str, cert_path: str) -> bool | None:
    """Verify cosign provenance signature on checksums.txt.

    Returns:
        True  — cosign verified successfully
        False — cosign found but verification failed
        None  — cosign not available (not on PATH, or execution failed)

    The caller treats both False and None as "abort auto-install" — only
    True allows the install to proceed.
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
            text=True,
            timeout=15,
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
    with open(checksums_path) as f:
        for line in f:
            # Format: "<hash>  <filename>"
            parts = line.strip().split("  ", 1)
            if len(parts) == 2 and parts[1] == archive_name:
                expected = parts[0]
                break
    if not expected:
        logger.warning("No checksum entry for %s", archive_name)
        return False

    sha = hashlib.sha256()
    with open(archive_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    actual = sha.hexdigest()
    if actual != expected:
        logger.warning("Checksum mismatch: expected %s, got %s", expected, actual)
        return False
    return True


def _install_tirith(*, log_failures: bool = True) -> tuple[str | None, str]:
    """Download and install tirith to $HERMES_HOME/bin/tirith.

    Verifies provenance via cosign and SHA-256 checksum.
    Returns (installed_path, failure_reason).  On success failure_reason is "".
    failure_reason is a short tag used by the disk marker to decide if the
    failure is retryable (e.g. "cosign_missing" clears when cosign appears).
    """
    log = logger.warning if log_failures else logger.debug

    target = _detect_target()
    if not target:
        logger.info("tirith auto-install: unsupported platform %s/%s",
                     platform.system(), platform.machine())
        return None, "unsupported_platform"

    archive_name = f"tirith-{target}.tar.gz"
    base_url = f"https://github.com/{_REPO}/releases/latest/download"

    tmpdir = tempfile.mkdtemp(prefix="tirith-install-")
    try:
        archive_path = os.path.join(tmpdir, archive_name)
        checksums_path = os.path.join(tmpdir, "checksums.txt")
        sig_path = os.path.join(tmpdir, "checksums.txt.sig")
        cert_path = os.path.join(tmpdir, "checksums.txt.pem")

        logger.info("tirith not found — downloading latest release for %s...", target)

        try:
            _download_file(f"{base_url}/{archive_name}", archive_path)
            _download_file(f"{base_url}/checksums.txt", checksums_path)
        except Exception as exc:
            log("tirith download failed: %s", exc)
            return None, "download_failed"

        # Cosign provenance verification — preferred but not mandatory.
        # When cosign is available, we verify that the release was produced
        # by the expected GitHub Actions workflow (full supply chain proof).
        # Without cosign, SHA-256 checksum + HTTPS still provides integrity
        # and transport-level authenticity.
        cosign_verified = False
        if shutil.which("cosign"):
            try:
                _download_file(f"{base_url}/checksums.txt.sig", sig_path)
                _download_file(f"{base_url}/checksums.txt.pem", cert_path)
            except Exception as exc:
                logger.info("cosign artifacts unavailable (%s), proceeding with SHA-256 only", exc)
            else:
                cosign_result = _verify_cosign(checksums_path, sig_path, cert_path)
                if cosign_result is True:
                    cosign_verified = True
                elif cosign_result is False:
                    # Verification explicitly rejected — abort, the release
                    # may have been tampered with.
                    log("tirith install aborted: cosign provenance verification failed")
                    return None, "cosign_verification_failed"
                else:
                    # None = execution failure (timeout/OSError) — proceed
                    # with SHA-256 only since cosign itself is broken.
                    logger.info("cosign execution failed, proceeding with SHA-256 only")
        else:
            logger.info("cosign not on PATH — installing tirith with SHA-256 verification only "
                        "(install cosign for full supply chain verification)")

        if not _verify_checksum(archive_path, checksums_path, archive_name):
            return None, "checksum_failed"

        with tarfile.open(archive_path, "r:gz") as tar:
            # Extract only the tirith binary (safety: reject paths with ..)
            for member in tar.getmembers():
                if member.name == "tirith" or member.name.endswith("/tirith"):
                    if ".." in member.name:
                        continue
                    member.name = "tirith"
                    tar.extract(member, tmpdir)
                    break
            else:
                log("tirith binary not found in archive")
                return None, "binary_not_in_archive"

        src = os.path.join(tmpdir, "tirith")
        dest = os.path.join(_hermes_bin_dir(), "tirith")
        shutil.move(src, dest)
        os.chmod(dest, os.stat(dest).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        verification = "cosign + SHA-256" if cosign_verified else "SHA-256 only"
        logger.info("tirith installed to %s (%s)", dest, verification)
        return dest, ""

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


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
    2. $HERMES_HOME/bin/tirith (previously auto-installed)
    3. Auto-install from GitHub releases → $HERMES_HOME/bin/tirith

    Failed installs are cached for the process lifetime (and persisted to
    disk for 24h) to avoid repeated network attempts.
    """
    global _resolved_path, _install_failure_reason

    # Fast path: successfully resolved on a previous call.
    if _resolved_path is not None and _resolved_path is not _INSTALL_FAILED:
        return _resolved_path

    expanded = os.path.expanduser(configured_path)
    explicit = _is_explicit_path(configured_path)
    install_failed = _resolved_path is _INSTALL_FAILED

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
        return found

    hermes_bin = os.path.join(_hermes_bin_dir(), "tirith")
    if os.path.isfile(hermes_bin) and os.access(hermes_bin, os.X_OK):
        _resolved_path = hermes_bin
        _install_failure_reason = ""
        _clear_install_failed()
        return hermes_bin

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
        return installed

    # Install failed — cache the miss and persist reason to disk
    _resolved_path = _INSTALL_FAILED
    _install_failure_reason = reason
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

        hermes_bin = os.path.join(_hermes_bin_dir(), "tirith")
        if os.path.isfile(hermes_bin) and os.access(hermes_bin, os.X_OK):
            _resolved_path = hermes_bin
            _install_failure_reason = ""
            return

        installed, reason = _install_tirith(log_failures=log_failures)
        if installed:
            _resolved_path = installed
            _install_failure_reason = ""
            _clear_install_failed()
        else:
            _resolved_path = _INSTALL_FAILED
            _install_failure_reason = reason
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
    if _resolved_path is not None and _resolved_path is not _INSTALL_FAILED:
        path = _resolved_path
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
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
        return found

    hermes_bin = os.path.join(_hermes_bin_dir(), "tirith")
    if os.path.isfile(hermes_bin) and os.access(hermes_bin, os.X_OK):
        _resolved_path = hermes_bin
        _install_failure_reason = ""
        _clear_install_failed()
        return hermes_bin

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
    cfg = _load_security_config()

    if not cfg["tirith_enabled"]:
        return {"action": "allow", "findings": [], "summary": ""}

    tirith_path = _resolve_tirith_path(cfg["tirith_path"])
    timeout = cfg["tirith_timeout"]
    fail_open = cfg["tirith_fail_open"]

    try:
        result = subprocess.run(
            [tirith_path, "check", "--json", "--non-interactive",
             "--shell", "posix", "--", command],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except OSError as exc:
        # Covers FileNotFoundError, PermissionError, exec format error
        logger.warning("tirith spawn failed: %s", exc)
        if fail_open:
            return {"action": "allow", "findings": [], "summary": f"tirith unavailable: {exc}"}
        return {"action": "block", "findings": [], "summary": f"tirith spawn failed (fail-closed): {exc}"}
    except subprocess.TimeoutExpired:
        logger.warning("tirith timed out after %ds", timeout)
        if fail_open:
            return {"action": "allow", "findings": [], "summary": f"tirith timed out ({timeout}s)"}
        return {"action": "block", "findings": [], "summary": f"tirith timed out (fail-closed)"}

    # Map exit code to action
    exit_code = result.returncode
    if exit_code == 0:
        action = "allow"
    elif exit_code == 1:
        action = "block"
    elif exit_code == 2:
        action = "warn"
    else:
        # Unknown exit code — respect fail_open
        logger.warning("tirith returned unexpected exit code %d", exit_code)
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

    return {"action": action, "findings": findings, "summary": summary}
