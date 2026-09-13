"""Tirith pre-exec scanning. PM owns its optional pinned binary; exit codes
remain the verdict authority and operational failures obey fail_open."""

import json
import logging
import os
import shutil
import subprocess
import threading
from contextvars import copy_context
from pathlib import Path

from hermes_constants import hermes_home_key

logger = logging.getLogger(__name__)
_REPO = "sheeki03/tirith"
# Only the release workflow may attest checksums, not arbitrary repo workflows.
_COSIGN_IDENTITY_REGEXP = f"^https://github.com/{_REPO}/\\.github/workflows/release\\.yml@refs/tags/v"
_COSIGN_ISSUER = "https://token.actions.githubusercontent.com"

# --- Config helpers ---
def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    return default if val is None else val.lower() in {"1", "true", "yes"}


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ[key])
    except (KeyError, ValueError):
        return default


def _load_security_config() -> dict:
    """Security settings from config.yaml, with env var overrides."""
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly().get("security", {}) or {}
    except Exception:
        cfg = {}
    return {
        "tirith_enabled": _env_bool("TIRITH_ENABLED", cfg.get("tirith_enabled", True)),
        "tirith_path": os.getenv("TIRITH_BIN", cfg.get("tirith_path", "tirith")),
        "tirith_timeout": _env_int("TIRITH_TIMEOUT", cfg.get("tirith_timeout", 5)),
        "tirith_fail_open": _env_bool("TIRITH_FAIL_OPEN", cfg.get("tirith_fail_open", True))}


# Circuit breaker: after _CRASH_LIMIT consecutive spawn/execution failures tirith is disabled
# for the rest of the process so a broken binary can't turn every tool call into a fail-open
# retry loop. Reset on success. Lock-free on purpose: a racing double-increment only opens the
# breaker one call early; no corruption or security bypass is possible.
_CRASH_LIMIT = 3
# Reset on successful execution (see _record_tirith_crash / check_command_security). Thread safety:
# _crash_count and _circuit_open are module-level globals mutated without a lock. check_command_security can
# be called from concurrent agent threads (gateway multi-session). The race is benign — at worst two threads
# both increment past _CRASH_LIMIT and both set _circuit_open = True, opening the breaker one call early.
# This intentionally matches the lock-free style of error counters in mcp_tool.py rather than the locked
# _warn_once pattern, because the worst case is harmless. See #41400.
_crash_count: int = 0
_circuit_open: bool = False

# Warn-once: spawn/path warnings sit in the hot path and would otherwise repeat once per
# terminal command while tirith is unavailable (e.g. install thread still running).
_warned_messages: set[str] = set()
_warned_lock = threading.Lock()

def _record_tirith_crash() -> None:
    global _crash_count, _circuit_open
    _crash_count += 1
    if _crash_count >= _CRASH_LIMIT:
        _circuit_open = True
        logger.warning("tirith circuit breaker opened after %d consecutive failures; "
                       "disabling for the rest of the process", _crash_count)


def _warn_once(key: str, message: str, *args) -> None:
    """``logger.warning`` at most once per ``key`` for the process lifetime."""
    with _warned_lock:
        if key in _warned_messages:
            return
        _warned_messages.add(key)
    logger.warning(message, *args)


def _verify_cosign(checksums_path: str, sig_path: str, cert_path: str) -> bool | None:
    """Cosign provenance of checksums.txt: True verified, False rejected, None if cosign absent/failed."""
    if not (cosign := shutil.which("cosign")):
        logger.info("cosign not found on PATH")
        return None
    try:
        result = subprocess.run(
            [cosign, "verify-blob", "--certificate", cert_path, "--signature", sig_path,
             "--certificate-identity-regexp", _COSIGN_IDENTITY_REGEXP,
             "--certificate-oidc-issuer", _COSIGN_ISSUER, checksums_path],
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=15, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("cosign execution failed: %s", exc)
        return None
    if result.returncode:
        logger.warning("cosign verification failed (exit %d): %s", result.returncode, result.stderr.strip())
        return False
    logger.info("cosign provenance verification passed")
    return True


def _verify_release_provenance(directory: Path, log) -> tuple[bool, str]:
    """Verify PM-acquired checksum provenance; this function never downloads.

    Missing/broken cosign is optional; an explicit rejection is fatal.
    """
    if not shutil.which("cosign"):
        logger.info("cosign not on PATH — installing tirith with SHA-256 verification only")
        return False, ""
    checksums = directory / "checksums.txt"
    signature, certificate = directory / "checksums.txt.sig", directory / "checksums.txt.pem"
    if not signature.is_file() or not certificate.is_file():
        logger.info("cosign artifacts unavailable, proceeding with SHA-256 only")
        return False, ""
    verified = _verify_cosign(str(checksums), str(signature), str(certificate))
    if verified is False:
        log("tirith install aborted: cosign provenance verification failed")
        return False, "cosign_verification_failed"
    return verified is True, ""


# One non-blocking startup attempt per routed home. Durable selection, failure
# recovery, download locks and publication belong to PM, not disk markers here.
_install_lock = threading.Lock()
_install_threads: dict[str, threading.Thread] = {}
_install_attempted: set[str] = set()


def _claim_install_attempt() -> bool:
    """Share the one-attempt budget between cold scans and startup threads."""
    with _install_lock:
        home = hermes_home_key()
        if home in _install_attempted:
            return False
        _install_attempted.add(home)
        return True


def is_platform_supported() -> bool:
    """Whether PM has a managed Tirith build for this host."""
    import pm

    try:
        return pm.get_package("tirith").missing_reason(pm.current_target()) is None
    except RuntimeError:
        return False


def _local_tirith(configured_path: str) -> str | None:
    expanded = os.path.expanduser(configured_path)
    if configured_path != "tirith":
        return (expanded if os.path.isfile(expanded) and os.access(expanded, os.X_OK)
                else shutil.which(expanded))
    if external := shutil.which("tirith"):
        return external
    import pm

    selected = pm.installed_package("tirith")
    return str(selected.binary) if selected and selected.binary else None


def _resolve_tirith_path(configured_path: str) -> str:
    """Resolve for a scan; do not wait on a startup download already in flight."""
    if found := _local_tirith(configured_path):
        return found
    if configured_path == "tirith":
        import pm

        if not pm.lazy_installs_allowed() or not _claim_install_attempt():
            return os.path.expanduser(configured_path)
        try:
            pm.ensure("tirith")
            selected = pm.installed_package("tirith")
            if selected and selected.binary:
                return str(selected.binary)
        except Exception as exc:
            _warn_once("tirith_install", "tirith install unavailable: %s", exc)
    return os.path.expanduser(configured_path)


def _background_install(*, log_failures: bool) -> None:
    import pm

    try:
        pm.ensure("tirith")
    except Exception as exc:
        log = logger.warning if log_failures else logger.debug
        log("tirith install failed: %s", exc)


def ensure_installed(*, log_failures: bool = True, explicit: bool = False):
    """Opt-in startup is non-blocking. Explicit setup waits and reports errors.

    Explicit executable configuration remains authoritative, including a miss.
    Lazy refusal never starts a thread; already installed tools remain usable.
    """
    import pm

    cfg = _load_security_config()
    if not cfg["tirith_enabled"]:
        return None
    configured = cfg["tirith_path"]
    if configured != "tirith":
        return _local_tirith(configured)
    if explicit:
        pm.ensure("tirith", explicit=True)
        selected = pm.installed_package("tirith")
        return str(selected.binary) if selected and selected.binary else None
    if found := _local_tirith(configured):
        return found
    if not is_platform_supported() or not pm.lazy_installs_allowed():
        return None
    if _claim_install_attempt():
        context = copy_context()
        thread = threading.Thread(
            target=context.run, args=(_background_install,),
            kwargs={"log_failures": log_failures}, daemon=True,
        )
        _install_threads[hermes_home_key()] = thread
        thread.start()
    return None


# --- Main API ---
_MAX_FINDINGS = 50
_MAX_SUMMARY_LEN = 500
_EXIT_ACTIONS = {0: "allow", 1: "block", 2: "warn"}
# Summary when tirith's JSON is unparseable and only the exit code is known.
_NO_DETAILS_SUMMARY = {
    "block": "security issue detected (details unavailable)",
    "warn": "security warning detected (details unavailable)"}


def _verdict(action: str, summary: str = "", findings: list | None = None) -> dict:
    return {"action": action, "findings": [] if findings is None else findings, "summary": summary}


def _fail(fail_open: bool, open_summary: str, closed_summary: str) -> dict:
    return _verdict("allow", open_summary) if fail_open else _verdict("block", closed_summary)


def _crash(fail_open: bool, open_summary: str, closed_summary: str) -> dict:
    """An operational failure: count it toward the circuit breaker, then fail open/closed."""
    _record_tirith_crash()
    return _fail(fail_open, open_summary, closed_summary)


def check_command_security(command: str) -> dict:
    """Run the tirith scan on a command -> ``{"action": allow|warn|block, "findings", "summary"}``.
    Exit code determines the action; JSON enriches. Spawn failures/timeouts respect fail_open."""
    global _crash_count
    cfg = _load_security_config()
    if not cfg["tirith_enabled"]:
        return _verdict("allow")
    # Circuit breaker: if tirith has crashed _CRASH_LIMIT times in a row, stop trying for the rest of the
    # process. Without this, a corrupted or missing binary causes every tool call to hit the same spawn
    # failure → fail-open → agent retry loop, hanging the user for 20+ minutes (issue #41400).
    if _circuit_open:
        return _verdict("allow", "tirith disabled (circuit breaker)")
    # No binary for this platform, ever: skip the resolver so we never spawn.
    if cfg["tirith_path"] == "tirith" and not is_platform_supported():
        return _verdict("allow")
    tirith_path = _resolve_tirith_path(cfg["tirith_path"])
    timeout, fail_open = cfg["tirith_timeout"], cfg["tirith_fail_open"]
    if tirith_path is None:
        _warn_once("tirith_path_none", "tirith path resolved to None; scanning disabled")
        return _fail(fail_open, "tirith path unavailable", "tirith path unavailable (fail-closed)")
    try:
        result = subprocess.run(
            [tirith_path, "check", "--json", "--non-interactive", "--shell", "posix", "--", command],
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout,
            stdin=subprocess.DEVNULL)
    except OSError as exc:
        # FileNotFoundError / PermissionError / exec format error: dedupe by (class, errno)
        # so each failure mode surfaces once, not per command.
        _warn_once(f"tirith_spawn_failed:{type(exc).__name__}:{getattr(exc, 'errno', '')}",
                   "tirith spawn failed: %s", exc)
        return _crash(fail_open, f"tirith unavailable: {exc}", f"tirith spawn failed (fail-closed): {exc}")
    except subprocess.TimeoutExpired:
        _warn_once(f"tirith_timeout:{timeout}", "tirith timed out after %ds", timeout)
        return _crash(fail_open, f"tirith timed out ({timeout}s)", "tirith timed out (fail-closed)")
    exit_code = result.returncode
    if (action := _EXIT_ACTIONS.get(exit_code)) is None:
        # Unknown exit code (includes signal-killed, e.g. -11): respect fail_open.
        logger.warning("tirith returned unexpected exit code %d", exit_code)
        return _crash(fail_open, f"tirith exit code {exit_code} (fail-open)",
                      f"tirith exit code {exit_code} (fail-closed)")
    if action == "allow":
        _crash_count = 0  # successful execution resets the circuit breaker
    # JSON enriches findings/summary; a parse failure never changes the verdict.
    findings, summary = [], ""
    try:
        data = json.loads(result.stdout) if result.stdout.strip() else {}
        findings = data.get("findings", [])[:_MAX_FINDINGS]
        summary = (data.get("summary", "") or "")[:_MAX_SUMMARY_LEN]
    except (json.JSONDecodeError, AttributeError):
        logger.debug("tirith JSON parse failed, using exit code only")
        summary = _NO_DETAILS_SUMMARY.get(action, "")
    # .app is a legitimate gTLD: a warn consisting solely of lookalike_tld findings for .app is a
    # known false positive and is downgraded to allow. Any other finding keeps the warn.
    if action == "warn" and findings and all(_is_app_tld_finding(f) for f in findings):
        return _verdict("allow")
    return _verdict(action, summary, findings)


def _is_app_tld_finding(finding: dict) -> bool:
    """True if this finding is a lookalike_tld warning for the .app TLD only."""
    if not isinstance(finding, dict) or finding.get("rule_id") != "lookalike_tld":
        return False
    return any(
        val is not None and ".app" in str(val).lower()
        for val in (finding.get(k) for k in ("value", "tld", "detail", "description", "message")))
