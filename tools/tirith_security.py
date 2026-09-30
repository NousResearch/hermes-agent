"""Tirith pre-exec scanning. PM owns its optional pinned binary; exit codes
remain the verdict authority and operational failures obey fail_open.

A candidate is a resolved scanner only when its header declares a format *this* process can execute
(ELF ``e_machine``, Mach-O magic + ``cputype``); a refused candidate is not found, so the next slot
and the install path are reached exactly as when a slot is empty. Every verdict the guard returns
carries ``scanner_state`` (``ran`` / ``unavailable`` / ``disabled``), so an allow that was never
scanned is a value a reader can test rather than prose to interpret."""

import json
import logging
import os
import shutil
import subprocess
import threading
import time
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


# Circuit breaker: after _CRASH_LIMIT consecutive spawn/execution failures tirith is disabled so a broken
# binary can't turn every tool call into a fail-open retry loop (#41400). The breaker HALF-OPENS after
# _CIRCUIT_RETRY_S: one caller re-probes tirith for real, and any completed scan (exit 0/1/2 — allow/block/warn
# all prove the binary is healthy) closes it, while a failed probe re-arms the timer. Without the TTL this was
# a one-way latch: once open, the reset branch below was unreachable for the rest of the process.
# Thread safety: crash counting stays lock-free — a racing double-increment only opens the breaker one call
# early, which is harmless, and matches the mcp_tool.py error counters rather than the locked _warn_once
# pattern. _breaker_lock guards ONLY the half-open claim (TTL check + timestamp re-arm, nanoseconds); it is
# never held across the subprocess probe, so it cannot reintroduce the #41400 hang. Claiming re-arms
# _circuit_open_at first, so concurrent callers see a fresh TTL and stay fail-open: one probe per TTL window.
_CRASH_LIMIT = 3
_CIRCUIT_RETRY_S = 300  # half-open probe interval (seconds)
_crash_count: int = 0
_circuit_open: bool = False
_circuit_open_at: float = 0.0
_breaker_lock = threading.Lock()

# Warn-once: spawn/path warnings sit in the hot path and would otherwise repeat once per
# terminal command while tirith is unavailable (e.g. install thread still running).
_warned_messages: set[str] = set()
_warned_lock = threading.Lock()

def _record_tirith_crash() -> None:
    global _crash_count, _circuit_open, _circuit_open_at
    _crash_count += 1
    if _crash_count >= _CRASH_LIMIT:
        _circuit_open, _circuit_open_at = True, time.monotonic()
        logger.warning("tirith circuit breaker opened after %d consecutive failures; "
                       "disabling for %ds", _crash_count, _CIRCUIT_RETRY_S)


def _warn_once(key: str, message: str, *args) -> None:
    """``logger.warning`` at most once per ``key`` for the process lifetime."""
    with _warned_lock:
        if key in _warned_messages:
            return
        _warned_messages.add(key)
    logger.warning(message, *args)


# The scanner's state, carried on every result the guard returns: `ran` = a scan completed and produced
# the verdict; `unavailable` = no scanner was usable, so nothing was scanned; `disabled` = the scanner
# is switched off by config, or the circuit breaker is open. The state is a value, never prose only.
_SCANNER_STATE_RAN = "ran"
_SCANNER_STATE_UNAVAILABLE = "unavailable"
_SCANNER_STATE_DISABLED = "disabled"
# One line per state for the turn's log: a state a reader can only find by interpreting a `summary`
# string is exactly what must not happen.
_SCANNER_STATE_DETAIL = {
    _SCANNER_STATE_UNAVAILABLE: "no scan ran — the scanner was not usable",
    _SCANNER_STATE_DISABLED: "no scan runs while the scanner is switched off",
}


def _note_scanner_state(state: str) -> None:
    """Name the scanner's state once per class on the turn's log (REQ-INS1-SAAS-066 Beh 4)."""
    if state in _SCANNER_STATE_DETAIL:
        _warn_once(f"scanner_state:{state}", "tirith scanner_state=%s: %s", state, _SCANNER_STATE_DETAIL[state])


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


def verify_release_provenance(directory: Path, log) -> tuple[bool, str]:
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


# --- Candidate architecture probe ---
# A candidate is a scanner only when the file it names is a binary *this* process can execute: the
# header is read (ELF `e_machine`, Mach-O `cputype`) and compared with the target PM names for this
# process. PM owns durable selection; this rule owns acceptance, and it is the resolution's own.
_ELF_MAGIC = b"\x7fELF"
_ELF_HEADER_LENGTHS = {1: 52, 2: 64}  # EI_CLASS 1 = 32-bit / 2 = 64-bit: the header the file claims
_ELF_MACHINES = {3: "x86", 40: "arm", 62: "x86_64", 183: "aarch64", 243: "riscv64"}  # e_machine
# Mach-O magic -> 64-bit header (a Mach-O header is 28 bytes, or 32 with the 64-bit magic). The measured
# foreign file's first octets were ``cffaedfe``: a little-endian 64-bit Mach-O.
_MACHO_MAGICS = {b"\xfe\xed\xfa\xce": False, b"\xce\xfa\xed\xfe": False,
                 b"\xfe\xed\xfa\xcf": True, b"\xcf\xfa\xed\xfe": True}
_MACHO_BIG_ENDIAN = frozenset({b"\xfe\xed\xfa\xce", b"\xfe\xed\xfa\xcf"})  # fields big-endian
_MACHO_CPUTYPES = {0x00000007: "x86", 0x0000000C: "arm", 0x01000007: "x86_64", 0x0100000C: "aarch64"}
# A format only runs on the platform that carries it: an ELF is a Linux/Android binary, a Mach-O a
# macOS one. The platform half of the target says which of the two is this process's own.
_FORMAT_PLATFORMS = {"elf": "unknown-linux-gnu", "mach-o": "apple-darwin"}
_HEADER_PROBE_BYTES = 64  # the longest header the probe reads (an ELF64 header)


def _is_executable(path: str) -> bool:
    """The cheap half of the acceptance rule: the file exists and the mode carries an execute bit.
    Necessary and — since REQ-INS1-SAAS-066 Beh 1 — insufficient on its own; `_runnable_candidate()`
    adds the fact the rule was missing."""
    return os.path.isfile(path) and os.access(path, os.X_OK)


def _process_target() -> str | None:
    """The target this process runs on, as PM names it (`<arch>-<platform>`). PM owns the mapping; a
    process PM cannot name has nothing to compare against, so the probe does not refuse."""
    import pm

    try:
        return pm.current_target()
    except RuntimeError:
        return None


def _read_elf_header(head: bytes) -> dict:
    """The format an ELF header declares: `e_machine` at the ELF header's own offset (18), read in the
    byte order `EI_DATA` declares. A file shorter than the header it claims is a refusal, not a crash."""
    declared_len = _ELF_HEADER_LENGTHS.get(head[4] if len(head) > 4 else 0, _ELF_HEADER_LENGTHS[2])
    if len(head) < declared_len:
        return {"format": "elf", "arch": None, "truncated": True,
                "declared": f"elf/unknown (header truncated: {len(head)} of {declared_len} bytes)"}
    raw_machine = bytes(head[18:20])
    if head[5] == 2:  # EI_DATA: 2 = the file's fields are big-endian
        machine = int.from_bytes(raw_machine, "big")
    else:
        machine = int.from_bytes(raw_machine, "little")
    arch = _ELF_MACHINES.get(machine)
    return {"format": "elf", "arch": arch, "truncated": False,
            "declared": f"elf/{arch or f'e_machine={machine}'}"}


def _read_macho_header(head: bytes, is64: bool, big_endian: bool) -> dict:
    """The format a Mach-O header declares: `cputype` at offset 4, in the file's own byte order. A file
    shorter than the header it claims is a refusal, not a crash."""
    declared_len = 32 if is64 else 28
    if len(head) < declared_len:
        return {"format": "mach-o", "arch": None, "truncated": True,
                "declared": f"mach-o/unknown (header truncated: {len(head)} of {declared_len} bytes)"}
    raw_cputype = bytes(head[4:8])
    if big_endian:
        cputype = int.from_bytes(raw_cputype, "big")
    else:
        cputype = int.from_bytes(raw_cputype, "little")
    arch = _MACHO_CPUTYPES.get(cputype)
    return {"format": "mach-o", "arch": arch, "truncated": False,
            "declared": f"mach-o/{arch or f'cputype=0x{cputype:08x}'}"}


def _read_candidate_header(path: str) -> dict | None:
    """What the file's own header declares -> ``{"format", "arch", "truncated", "declared"}``, or None
    when there is no header to read (a missing, unreadable or non-binary file). Those keep the module's
    existing states, unchanged (Beh 5) — the probe only refuses what it can read and cannot run."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(_HEADER_PROBE_BYTES)
    except OSError:
        return None
    if head.startswith(_ELF_MAGIC):
        return _read_elf_header(head)
    for magic, is64 in _MACHO_MAGICS.items():
        if head.startswith(magic):
            return _read_macho_header(head, is64, magic in _MACHO_BIG_ENDIAN)
    return None


def _header_runs_here(header: dict) -> bool:
    """True when the declared format and the declared architecture are this process's own (Beh 1). A
    process PM cannot name has nothing to compare against: it does not refuse."""
    target = _process_target()
    if not target:
        return True
    arch, platform_slot = target.split("-", 1)
    return arch == header["arch"] and platform_slot == _FORMAT_PLATFORMS.get(header["format"])


def _report_unrunnable(path: str, header: dict) -> None:
    """The typed refusal, produced **where the resolution refuses** — before any command is guarded —
    once per refused path through the module's own once-per-class channel. It names the path, the format
    and architecture found, and the target of this process, and it is not swallowed by a boot call
    made with ``log_failures=False`` (Beh 3): that flag only quiets the install path's own lines."""
    _warn_once(f"scanner_unrunnable:{path}",
               "tirith scanner_unrunnable: path=%s found=%s process=%s — candidate refused, "
               "treated as not found", path, header["declared"], _process_target() or "unknown")


def _candidate_refusal(path: str) -> dict | None:
    """The header a candidate declares when this process cannot run it, else None (Beh 1)."""
    if (header := _read_candidate_header(path)) is None or _header_runs_here(header):
        return None
    return header


def _runs_here(path: str) -> bool:
    """The fact the acceptance rule was missing, for a candidate whose *existence* is already
    established — `shutil.which` answered it, PM has just installed it, or the configured path passed
    `_is_executable()`. The file's own header decides: a binary this process cannot run is refused,
    reported typed, and therefore not found, so the next slot is asked exactly as when a slot were
    empty (Beh 1, Beh 2). A candidate with no header to read (missing, unreadable, not a binary) keeps
    the module's existing states, unchanged (Beh 5)."""
    if (header := _candidate_refusal(path)) is None:
        return True
    _report_unrunnable(path, header)
    return False


def _runnable_candidate(path: str) -> bool:
    """The whole rule, for a path the configuration names: it exists, it carries the execute bit, and
    its header declares a binary this process can run. The first two facts stay necessary and become
    insufficient on their own (Beh 1)."""
    return _is_executable(path) and _runs_here(path)


def _first_runnable(candidates) -> str | None:
    """The first candidate the resolution may accept: the slots answer in the order they are given, and
    a candidate the probe refuses simply does not answer (Beh 2)."""
    for candidate in candidates:
        if candidate and _runs_here(candidate):
            return candidate
    return None


def _local_tirith(configured_path: str) -> str | None:
    """The local slots, in the order they are asked: an explicit path (then PATH for that name), else
    PATH for `tirith` and then PM's own installed binary. Every one of them is a candidate, so each
    passes the acceptance rule — a refused one is not found (Beh 1, Beh 2)."""
    expanded = os.path.expanduser(configured_path)
    if configured_path != "tirith":
        if _runnable_candidate(expanded):
            return expanded
        return _first_runnable((shutil.which(expanded),))
    external = shutil.which("tirith")
    if external and _runs_here(external):
        return external
    import pm

    selected = pm.installed_package("tirith")
    return _first_runnable((str(selected.binary) if selected and selected.binary else None,))


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
                # The install path's own behaviour is unchanged; what it hands back is a candidate like
                # any other, so the acceptance rule is applied to it too (Beh 1, Beh 2). A refused one
                # falls through, exactly as when a slot were empty.
                return _first_runnable((str(selected.binary),)) or os.path.expanduser(configured_path)
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


def missing_is_expected() -> bool:
    """Whether an unresolved default tirith is by design rather than a fault.

    The first launch after a PM install starts the download in the background,
    and a lazy-install policy refusal is the operator's choice; neither is
    actionable. A missing explicit ``tirith_path`` always is.
    """
    import pm

    configured = _load_security_config()["tirith_path"]
    if configured != "tirith":
        return False
    thread = _install_threads.get(hermes_home_key())
    if thread is not None and thread.is_alive():
        return True
    return _local_tirith(configured) is not None or not pm.lazy_installs_allowed()


# --- Main API ---
_MAX_FINDINGS = 50
_MAX_SUMMARY_LEN = 500
_EXIT_ACTIONS = {0: "allow", 1: "block", 2: "warn"}
# Summary when tirith's JSON is unparseable and only the exit code is known.
_NO_DETAILS_SUMMARY = {
    "block": "security issue detected (details unavailable)",
    "warn": "security warning detected (details unavailable)"}
_VARIATION_SELECTOR_16 = "\ufe0f"
# Code points that carry the Unicode ``Emoji`` property and take VS16 for emoji presentation: the
# Miscellaneous Symbols / Dingbats blocks, the SMP emoji planes, and the BMP singletons outside them
# (©️ ®️ ‼️ ⁉️ ™️ ℹ️ arrows, ⌚ ⌨️ ⏏️ media keys, Ⓜ️ ▪️ ▶️ ◀️ ◻️ ⤴️ ⬅️ ⬛ ⭐ ⭕ 〰️ 〽️ ㊗️ ㊙️).
# Digits, ``#`` and ``*`` also carry the property (keycap bases) but are deliberately absent: VS16
# after a letter or digit is exactly the steganography signal the rule exists for.
_EMOJI_PRESENTATION_BASE_RANGES = (
    (0x00A9, 0x00A9), (0x00AE, 0x00AE), (0x203C, 0x203C), (0x2049, 0x2049), (0x2122, 0x2122),
    (0x2139, 0x2139), (0x2194, 0x2199), (0x21A9, 0x21AA), (0x231A, 0x231B), (0x2328, 0x2328),
    (0x23CF, 0x23CF), (0x23E9, 0x23F3), (0x23F8, 0x23FA), (0x24C2, 0x24C2), (0x25AA, 0x25AB),
    (0x25B6, 0x25B6), (0x25C0, 0x25C0), (0x25FB, 0x25FE), (0x2600, 0x27BF), (0x2934, 0x2935),
    (0x2B05, 0x2B07), (0x2B1B, 0x2B1C), (0x2B50, 0x2B50), (0x2B55, 0x2B55), (0x3030, 0x3030),
    (0x303D, 0x303D), (0x3297, 0x3297), (0x3299, 0x3299), (0x1F000, 0x1FAFF))


def _verdict(action: str, summary: str = "", findings: list | None = None, *,
             scanner_state: str = _SCANNER_STATE_RAN) -> dict:
    """The guard's result. ``scanner_state`` says what produced the verdict — ``ran`` for a completed
    scan, or the reason no scan ran (``unavailable`` / ``disabled``) — so an allow that was never
    scanned is a value a reader can test rather than prose to interpret (Beh 4)."""
    return {"action": action, "findings": [] if findings is None else findings,
            "summary": summary, "scanner_state": scanner_state}


def _fail(fail_open: bool, open_summary: str, closed_summary: str) -> dict:
    """No scan ran: the configured fail-open / fail-closed policy still decides allow versus block, and
    the verdict now carries the state that says why it was produced (Beh 4). The state is named once in
    the turn's log — no silent fail-open."""
    _note_scanner_state(_SCANNER_STATE_UNAVAILABLE)
    if fail_open:
        return _verdict("allow", open_summary, scanner_state=_SCANNER_STATE_UNAVAILABLE)
    return _verdict("block", closed_summary, scanner_state=_SCANNER_STATE_UNAVAILABLE)


def _crash(fail_open: bool, open_summary: str, closed_summary: str) -> dict:
    """An operational failure: count it toward the circuit breaker, then fail open/closed."""
    _record_tirith_crash()
    return _fail(fail_open, open_summary, closed_summary)


def check_command_security(command: str) -> dict:
    """Run the tirith scan on a command -> ``{"action": allow|warn|block, "findings", "summary"}``.
    Exit code determines the action; JSON enriches. Spawn failures/timeouts respect fail_open."""
    global _crash_count, _circuit_open, _circuit_open_at
    cfg = _load_security_config()
    if not cfg["tirith_enabled"]:
        _note_scanner_state(_SCANNER_STATE_DISABLED)
        return _verdict("allow", scanner_state=_SCANNER_STATE_DISABLED)
    # Circuit breaker: if tirith has crashed _CRASH_LIMIT times in a row, stop trying and fail open (issue
    # #41400). After _CIRCUIT_RETRY_S the breaker half-opens: exactly one caller claims the probe slot —
    # claiming re-arms _circuit_open_at under _breaker_lock, so concurrent callers see a fresh TTL and stay
    # fail-open — and falls through to a real scan below.
    if _circuit_open:
        with _breaker_lock:
            if _circuit_open and time.monotonic() - _circuit_open_at < _CIRCUIT_RETRY_S:
                held_open = True
            else:
                held_open = False
                if _circuit_open:  # TTL expired: claim the single-flight probe slot for this window
                    _circuit_open_at = time.monotonic()
                    logger.info("tirith circuit breaker half-open: probing after %ds", _CIRCUIT_RETRY_S)
        if held_open:
            # The summary the breaker has always returned, kept verbatim — the state rides beside it,
            # so an allow that was never scanned is testable, not prose (Beh 4).
            _note_scanner_state(_SCANNER_STATE_DISABLED)
            return _verdict("allow", "tirith disabled (circuit breaker)",
                            scanner_state=_SCANNER_STATE_DISABLED)
    # No binary for this platform, ever: skip the resolver so we never spawn. Silent in the log
    # (`unsupported_platform`), and the verdict says why it carries no scan (Beh 4, Beh 5).
    if cfg["tirith_path"] == "tirith" and not is_platform_supported():
        return _verdict("allow", scanner_state=_SCANNER_STATE_UNAVAILABLE)
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
    # Any completed scan (allow/block/warn) proves the binary is healthy: clear the streak and close the
    # breaker. This is the half-open probe's recovery path, and it also fixes the streak never resetting on
    # block/warn verdicts.
    _crash_count = 0
    if _circuit_open:
        _circuit_open, _circuit_open_at = False, 0.0
        logger.info("tirith circuit breaker closed after successful probe")
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
    # VS16 follows ordinary emoji-capable code points in standard emoji-presentation sequences.
    # Preserve warnings for every other selector, including VS16 after text, because those can
    # carry the steganographic payload that Tirith is intended to detect.
    if action == "warn" and findings and all(_is_emoji_variation_selector_finding(f) for f in findings) \
            and _has_only_emoji_presentation_selectors(command):
        return _verdict("allow")
    return _verdict(action, summary, findings)


def _is_app_tld_finding(finding: dict) -> bool:
    """True if this finding is a lookalike_tld warning for the .app TLD only."""
    if not isinstance(finding, dict) or finding.get("rule_id") != "lookalike_tld":
        return False
    return any(
        val is not None and ".app" in str(val).lower()
        for val in (finding.get(k) for k in ("value", "tld", "detail", "description", "message")))


def _is_emoji_variation_selector_finding(finding: dict) -> bool:
    """True only for the Tirith rule that reports variation selectors."""
    return isinstance(finding, dict) and finding.get("rule_id") == "variation_selector"


def _has_only_emoji_presentation_selectors(command: str) -> bool:
    """Whether every variation selector is VS16 immediately after an emoji-capable base."""
    selectors = ("\ufe00", "\U000e0100")
    saw_selector = False
    for idx, char in enumerate(command):
        if not selectors[0] <= char <= "\ufe0f" and not selectors[1] <= char <= "\U000e01ef":
            continue
        saw_selector = True
        if char != _VARIATION_SELECTOR_16 or idx == 0:
            return False
        base = ord(command[idx - 1])
        if not any(start <= base <= end for start, end in _EMOJI_PRESENTATION_BASE_RANGES):
            return False
    return saw_selector
