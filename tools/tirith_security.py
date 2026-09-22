"""Tirith pre-exec security scanning wrapper: runs the tirith binary as a subprocess to scan
commands for content-level threats (homograph URLs, pipe-to-interpreter, terminal injection).
The exit code is the verdict source of truth (0 allow, 1 block, 2 warn); JSON stdout only
enriches findings. Operational failures (spawn error, timeout, unknown exit) respect
``fail_open``; programming errors propagate. Auto-install: a missing tirith is downloaded from
GitHub releases to $HERMES_HOME/bin/tirith in a background thread -- SHA-256 always verified,
cosign provenance when cosign is on PATH.

A candidate is a resolved scanner only when its header declares a format *this* process can execute
(ELF ``e_machine``, Mach-O magic + ``cputype``); a refused candidate is not found, so the next slot
and the install path are reached exactly as when a slot is empty. Every verdict the guard returns
carries ``scanner_state`` (``ran`` / ``unavailable`` / ``disabled``), so an allow that was never
scanned is a value a reader can test rather than prose to interpret."""

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
from contextlib import suppress

from hermes_constants import get_hermes_home, get_hermes_home_override, hermes_home_key

logger = logging.getLogger(__name__)
_REPO = "sheeki03/tirith"
# Cosign provenance pinned to the release workflow, not the whole repo.
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


# --- Module state ---
# Cached path after first resolution. _INSTALL_FAILED means "tried and failed" (distinct
# from None = "not yet tried") so a failed install is not retried per command.
_resolved_path: str | None | bool = None
_INSTALL_FAILED = False
_install_failure_reason: str = ""  # reason tag when _resolved_path is _INSTALL_FAILED
# Routed profiles (multiplexed gateway) resolve their own binary: ``security.tirith_path`` and
# ``<home>/bin/tirith`` are per profile, so the launch profile's slot above must not answer for them.
_resolved_path_by_home: dict[str, str] = {}

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

_install_lock = threading.Lock()
_install_thread: threading.Thread | None = None

# Warn-once: spawn/path warnings sit in the hot path and would otherwise repeat once per
# terminal command while tirith is unavailable (e.g. install thread still running).
_warned_messages: set[str] = set()
_warned_lock = threading.Lock()

_MARKER_TTL = 86400  # disk failure marker validity (24h) -- avoids retry across restarts


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


def _cached_path() -> str | None:
    """The path resolved on a previous call, or None if unresolved (None) / failed (_INSTALL_FAILED)."""
    if get_hermes_home_override() is not None:
        return _resolved_path_by_home.get(hermes_home_key())
    return _resolved_path or None


def _store_resolved(path: str) -> None:
    global _resolved_path
    if get_hermes_home_override() is not None:
        _resolved_path_by_home[hermes_home_key()] = path
    else:
        _resolved_path = path


def _set_resolved(path: str) -> None:
    global _install_failure_reason
    _store_resolved(path)
    _install_failure_reason = ""


def _set_failed(reason: str) -> None:
    global _resolved_path, _install_failure_reason
    _resolved_path, _install_failure_reason = _INSTALL_FAILED, reason


def _forget_resolved() -> None:
    """Drop the cached resolution (and the failure tag) so the slots are asked again. Used when a cached
    path stops passing the resolution's own rule: it is not found from then on, not handed back."""
    global _resolved_path, _install_failure_reason
    if get_hermes_home_override() is not None:
        _resolved_path_by_home.pop(hermes_home_key(), None)
    else:
        _resolved_path = None
    _install_failure_reason = ""


# --- Disk failure marker ---
def _failure_marker_path() -> str:
    return os.path.join(str(get_hermes_home()), ".tirith-install-failed")


def _read_failure_reason() -> str | None:
    """The marker's reason, or None if absent or older than _MARKER_TTL."""
    try:
        p = _failure_marker_path()
        if (time.time() - os.path.getmtime(p)) >= _MARKER_TTL:
            return None
        with open(p, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def _is_install_failed_on_disk() -> bool:
    """True if a recent install failure was persisted and is still non-retryable.
    A 'cosign_missing' marker is auto-cleared once cosign appears on PATH."""
    reason = _read_failure_reason()
    if reason == "cosign_missing" and shutil.which("cosign"):
        _clear_install_failed()
        return False
    return reason is not None


def _mark_install_failed(reason: str = ""):
    """Persist install failure to disk; ``reason`` is a short retryability tag."""
    with suppress(OSError):
        p = _failure_marker_path()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(reason)


def _clear_install_failed():
    """Remove the failure marker and reset warn-once state (so a failure after a reinstall surfaces again)."""
    with _warned_lock:
        _warned_messages.clear()
    with suppress(OSError):
        os.unlink(_failure_marker_path())


def _disk_marker_blocks_install() -> bool:
    """Apply a still-valid disk marker to module state; True if install must be skipped.
    Keeps the marker's real reason so in-process retry can detect cosign_missing."""
    if (disk_reason := _read_failure_reason()) is None or not _is_install_failed_on_disk():
        return False
    _set_failed(disk_reason)
    return True


# --- Auto-install ---
def _hermes_bin_dir() -> str:
    """$HERMES_HOME/bin, created if needed."""
    os.makedirs(d := os.path.join(str(get_hermes_home()), "bin"), exist_ok=True)
    return d


# Rust target triple components. Android (Termux) is ABI-compatible with Linux. Windows is
# absent on purpose (no tirith build): None = "never available here", pattern guards still run.
_TARGET_PLATFORMS = {"Darwin": "apple-darwin", "Linux": "unknown-linux-gnu", "Android": "unknown-linux-gnu"}
_TARGET_ARCHES = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}


def _detect_target() -> str | None:
    """Rust target triple for this platform, or None if tirith has no build for it."""
    plat = _TARGET_PLATFORMS.get(platform.system())
    arch = _TARGET_ARCHES.get(platform.machine().lower())
    return f"{arch}-{plat}" if plat and arch else None


def is_platform_supported() -> bool:
    """True when tirith ships a prebuilt binary for this OS+arch (CLI banner uses this)."""
    return _detect_target() is not None


def _download_file(url: str, dest: str, timeout: int = 10):
    from agent.secret_scope import get_secret
    req = urllib.request.Request(url)
    if token := get_secret("GITHUB_TOKEN"):
        req.add_header("Authorization", f"token {token}")
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)


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


def _verify_release_provenance(base_url: str, tmpdir: str, checksums_path: str, log) -> tuple[bool, str]:
    """Cosign step of the install -> ``(cosign_verified, failure_reason)``. Only an explicit
    cosign rejection aborts; missing/broken cosign or artifacts fall back to SHA-256 only."""
    if not shutil.which("cosign"):
        logger.info("cosign not on PATH — installing tirith with SHA-256 verification only "
                    "(install cosign for full supply chain verification)")
        return False, ""
    sig_path, cert_path = os.path.join(tmpdir, "checksums.txt.sig"), os.path.join(tmpdir, "checksums.txt.pem")
    try:
        _download_file(f"{base_url}/checksums.txt.sig", sig_path)
        _download_file(f"{base_url}/checksums.txt.pem", cert_path)
    except Exception as exc:
        logger.info("cosign artifacts unavailable (%s), proceeding with SHA-256 only", exc)
        return False, ""
    verified = _verify_cosign(checksums_path, sig_path, cert_path)
    if verified is False:
        log("tirith install aborted: cosign provenance verification failed")
        return False, "cosign_verification_failed"
    if verified is None:
        logger.info("cosign execution failed, proceeding with SHA-256 only")
    return verified is True, ""


def _verify_checksum(archive_path: str, checksums_path: str, archive_name: str) -> bool:
    """Verify SHA-256 of the archive against checksums.txt ("<hash>  <filename>" lines)."""
    with open(checksums_path, encoding="utf-8") as f:
        parsed = (line.strip().split("  ", 1) for line in f)
        expected = next((h for h, *n in parsed if n == [archive_name]), None)
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
    return actual == expected


def _extract_tirith_binary(tar: tarfile.TarFile, dest_dir: str, log) -> tuple[str | None, str]:
    """Extract the tirith binary from a release archive into dest_dir -> ``(path, reason)``."""
    for member in tar.getmembers():
        if member.name.rsplit("/", 1)[-1] != "tirith" or ".." in member.name:
            continue
        if not member.isfile():
            log("tirith archive member is not a regular file: %s", member.name)
            return None, "binary_not_regular_file"
        if (src_file := tar.extractfile(member)) is None:
            log("tirith binary could not be read from archive")
            return None, "binary_extract_failed"
        dest_path = os.path.join(dest_dir, "tirith")
        with src_file, open(dest_path, "wb") as out:
            shutil.copyfileobj(src_file, out)
        return dest_path, ""
    log("tirith binary not found in archive")
    return None, "binary_not_in_archive"


def _install_tirith(*, log_failures: bool = True) -> tuple[str | None, str]:
    """Download and install tirith to $HERMES_HOME/bin/tirith -> ``(installed_path,
    failure_reason)``; the reason ("" on success) is the disk marker's retryability tag."""
    log = logger.warning if log_failures else logger.debug
    if not (target := _detect_target()):
        logger.info("tirith auto-install: unsupported platform %s/%s", platform.system(), platform.machine())
        return None, "unsupported_platform"
    archive_name = f"tirith-{target}.tar.gz"
    base_url = f"https://github.com/{_REPO}/releases/latest/download"
    try:
        tmpdir = tempfile.mkdtemp(prefix="tirith-install-")
    except OSError as exc:
        log("tirith install failed: cannot create temp dir: %s", exc)
        return None, "no_space"
    try:
        archive_path, checksums_path = os.path.join(tmpdir, archive_name), os.path.join(tmpdir, "checksums.txt")
        logger.info("tirith not found — downloading latest release for %s...", target)
        try:
            _download_file(f"{base_url}/{archive_name}", archive_path)
            _download_file(f"{base_url}/checksums.txt", checksums_path)
        except Exception as exc:
            log("tirith download failed: %s", exc)
            return None, "download_failed"
        cosign_verified, reason = _verify_release_provenance(base_url, tmpdir, checksums_path, log)
        if reason:
            return None, reason
        if not _verify_checksum(archive_path, checksums_path, archive_name):
            return None, "checksum_failed"
        with tarfile.open(archive_path, "r:gz") as tar:
            src, reason = _extract_tirith_binary(tar, tmpdir, log)
        if src is None:
            return None, reason
        dest = os.path.join(_hermes_bin_dir(), "tirith")
        try:
            shutil.move(src, dest)
        except OSError:
            # Cross-device move (Docker, NFS): copy2's metadata step can raise PermissionError,
            # so fall back to plain copy + chmod; a partial dest is removed to avoid a
            # non-executable retry loop.
            try:
                shutil.copy(src, dest)
            except OSError:
                with suppress(OSError):
                    os.unlink(dest)
                return None, "cross_device_copy_failed"
        os.chmod(dest, os.stat(dest).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        logger.info("tirith installed to %s (%s)", dest, "cosign + SHA-256" if cosign_verified else "SHA-256 only")
        return dest, ""
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# --- Path resolution ---
def _is_executable(path: str) -> bool:
    """The cheap half of the acceptance rule: the file exists and the mode carries an execute bit.
    Necessary and — since REQ-INS1-SAAS-066 Beh 1 — insufficient on its own; `_runnable_candidate()`
    adds the fact the rule was missing."""
    return os.path.isfile(path) and os.access(path, os.X_OK)


# --- Candidate architecture probe ---
# A candidate is a scanner only when the file it names is a binary *this* process can execute: the
# header is read (ELF `e_machine`, Mach-O `cputype`) and compared with the architecture of the running
# process through the mapping this module already owns (`_detect_target()`).
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
# macOS one. The platform half of `_detect_target()` says which of the two is this process's own.
_FORMAT_PLATFORMS = {"elf": "unknown-linux-gnu", "mach-o": "apple-darwin"}
_HEADER_PROBE_BYTES = 64  # the longest header the probe reads (an ELF64 header)


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
    process whose architecture the mapping does not name has nothing to compare against: it does not
    refuse (the module already answers `unsupported_platform` for it elsewhere)."""
    target = _detect_target()
    if not target:
        return True
    arch, platform_slot = target.split("-", 1)
    return arch == header["arch"] and platform_slot == _FORMAT_PLATFORMS.get(header["format"])


def _process_architecture() -> str:
    """This process's architecture, as the module's own mapping names it (the refusal names both sides)."""
    return _detect_target() or f"{platform.system()}/{platform.machine()}"


def _report_unrunnable(path: str, header: dict) -> None:
    """The typed refusal, produced **where the resolution refuses** — before any command is guarded —
    once per refused path through the module's own once-per-class channel. It names the path, the format
    and architecture found, and the architecture of this process, and it is not swallowed by a boot call
    made with ``log_failures=False`` (Beh 3): that flag only quiets the install path's own lines."""
    _warn_once(f"scanner_unrunnable:{path}",
               "tirith scanner_unrunnable: path=%s found=%s process=%s — candidate refused, "
               "treated as not found", path, header["declared"], _process_architecture())


def _candidate_refusal(path: str) -> dict | None:
    """The header a candidate declares when this process cannot run it, else None (Beh 1)."""
    if (header := _read_candidate_header(path)) is None or _header_runs_here(header):
        return None
    return header


def _runnable_candidate(path: str) -> bool:
    """The resolution's acceptance rule: the file is executable **and** its header declares a binary
    this process can run. A refused candidate is reported typed and is not found, so the next slot is
    asked and the install path is reached exactly as when a slot were empty (Beh 1, Beh 2)."""
    if not _is_executable(path):
        return False
    if (header := _candidate_refusal(path)) is None:
        return True
    _report_unrunnable(path, header)
    return False


def _first_runnable(candidates) -> str | None:
    """The first candidate the resolution may accept: the slots answer in the order they are given, and
    a candidate the probe refuses simply does not answer (Beh 2)."""
    for candidate in candidates:
        if candidate and _runnable_candidate(candidate):
            return candidate
    return None


def _cached_resolved_path() -> str | None:
    """The cached path, re-validated on the same rule as any other candidate (Beh 2). A path that is
    gone keeps the module's existing state, unchanged; a path that is *there* and no longer passes the
    rule (replaced by a mount, truncated, a foreign binary, its execute bit lost) is reported and
    dropped — not found for the rest of the process instead of being handed back."""
    cached = _cached_path()
    if cached is None or not os.path.isfile(cached):
        return cached
    if _runnable_candidate(cached):
        return cached
    _forget_resolved()
    return None


def _find_local_tirith() -> str | None:
    """Cheap local lookup for the default "tirith": PATH, then $HERMES_HOME/bin — in that order, and
    only for a candidate whose header declares a binary this process can run (Beh 1, Beh 2)."""
    hermes_bin = os.path.join(_hermes_bin_dir(), "tirith")
    return _first_runnable((shutil.which("tirith"), hermes_bin))


def _resolve_locally(configured_path: str, *, warn_missing: bool) -> tuple[str | None, bool]:
    """Network-free resolution -> ``(path, may_install)``: ``path`` set = resolved (module state
    updated); else ``may_install`` False = terminal miss (explicit path missing, cached non-retryable
    failure), True = the disk marker / install step may proceed."""
    global _resolved_path, _install_failure_reason
    expanded = os.path.expanduser(configured_path)
    # An explicit (non-"tirith") path is authoritative: never auto-download a replacement. It is still a
    # candidate, so it passes the same rule as any other — a refused one is not found (Beh 1, Beh 2).
    if configured_path != "tirith":
        if found := _first_runnable((expanded, shutil.which(expanded))):
            _store_resolved(found)
            return found, False
        if warn_missing:
            logger.warning("Configured tirith path %r not found; scanning disabled", configured_path)
        _set_failed("explicit_path_missing")
        return None, False
    # Always re-run the cheap local checks so a manual install is picked up even after a
    # previous network failure (a long-lived gateway recovers without restart).
    if found := _find_local_tirith():
        _set_resolved(found)
        _clear_install_failed()
        return found, False
    # Previous install failed: skip the network retry unless the retryable cosign_missing
    # cause has been resolved in-process.
    if _resolved_path is _INSTALL_FAILED:
        if _install_failure_reason != "cosign_missing" or not shutil.which("cosign"):
            return None, False
        _resolved_path, _install_failure_reason = None, ""
        _clear_install_failed()
    return None, True


def _record_install_result(installed: str | None, reason: str) -> str | None:
    """Cache an install outcome in module state + disk marker; returns *installed*."""
    if installed:
        _set_resolved(installed)
        _clear_install_failed()
    else:
        _set_failed(reason)
        _mark_install_failed(reason)
    return installed


def _resolve_tirith_path(configured_path: str) -> str:
    """Resolve the tirith path, auto-installing synchronously if needed (default "tirith": PATH →
    $HERMES_HOME/bin/tirith → install; failures cached in-process and on disk for 24h). The cached path
    is re-validated on the resolution's own rule before it answers (Beh 2). On a miss the expanded
    configured path is returned so the spawn fails open via the dedupe'd OSError."""
    if cached := _cached_resolved_path():
        return cached
    expanded = os.path.expanduser(configured_path)
    # No tirith build for this platform: cache the verdict; the spawn fails open once, then
    # the fast path above short-circuits.
    if configured_path == "tirith" and not is_platform_supported():
        _set_failed("unsupported_platform")
        return expanded
    found, may_install = _resolve_locally(configured_path, warn_missing=True)
    if found or not may_install:
        return found or expanded
    # A background install is running: don't start a parallel one; fail-open until it finishes.
    if _install_running() or _disk_marker_blocks_install():
        return expanded
    installed = _record_install_result(*_install_tirith())
    return installed or expanded


def _install_running() -> bool:
    return _install_thread is not None and _install_thread.is_alive()


def _background_install(*, log_failures: bool = True):
    """Background thread target: download and install tirith."""
    with _install_lock:
        if _resolved_path is not None:  # another thread resolved meanwhile
            return
        if found := _find_local_tirith():  # may have been installed by another process
            _set_resolved(found)
            return
        _record_install_result(*_install_tirith(log_failures=log_failures))


def ensure_installed(*, log_failures: bool = True):
    """Resolved path if available now, else None after kicking off a daemon-thread download (local
    checks are synchronous; the download never blocks startup). Safe to call repeatedly. The cached path
    is re-validated on the resolution's own rule: a scanner that stopped being runnable is not handed
    back but looked for again (Beh 2), and a refusal is reported even on the boot call made with
    ``log_failures=False`` (Beh 3)."""
    global _install_thread
    cfg = _load_security_config()
    if not cfg["tirith_enabled"]:
        return None
    cached = _cached_resolved_path()
    if cached is not None:
        return cached if _is_executable(cached) else None
    # No tirith build here (e.g. Windows): stay silent -- no PATH probe, no download thread,
    # no disk marker. Pattern-matching guards still run.
    if not is_platform_supported():
        _set_failed("unsupported_platform")
        return None
    found, may_install = _resolve_locally(cfg["tirith_path"], warn_missing=False)
    if found or not may_install or _disk_marker_blocks_install():
        return found
    if not _install_running():
        _install_thread = threading.Thread(target=_background_install, daemon=True,
                                           kwargs={"log_failures": log_failures})
        _install_thread.start()
    return None  # not available yet; commands fail-open until ready


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
    """Run the tirith scan on a command -> ``{"action": allow|warn|block, "findings", "summary",
    "scanner_state"}``. Exit code determines the action; JSON enriches. Spawn failures/timeouts respect
    ``fail_open``, and every verdict returned without a completed scan carries that state in
    ``scanner_state`` (Beh 4)."""
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
    # (`unsupported_platform`), and the verdict says why it carries no scan (Beh 4, Beh 8).
    if not is_platform_supported():
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
