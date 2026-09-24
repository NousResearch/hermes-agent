"""Git Bash health probe + Mandatory-ASLR diagnostics (Windows), used by
``local._find_bash`` to pick a bash.exe that can actually launch MSYS children
and to build the targeted remediation when none can."""

import logging
import ntpath
import platform
import shutil
import subprocess

from hermes_cli._subprocess_compat import bounded_probe_run, windows_hide_flags

_IS_WINDOWS = platform.system() == "Windows"

# Same logger as the origin module so log routing is unchanged.
logger = logging.getLogger("tools.environments.local")

_bash_starts_cache: dict[str, bool] = {}
_bash_probe_details_cache: dict[str, str] = {}
_mandatory_aslr_enabled_cache: "bool | None" = None
_BASH_PROBE_TIMEOUT = 15.0

# External ``true`` and ``cat`` are intentional: a builtin-only ``exit 0`` probe
# misses Git-for-Windows fork/spawn failures under system-wide Mandatory ASLR.
_BASH_EXTERNAL_PROGRAM_PROBE = "/usr/bin/true; /usr/bin/cat --version >/dev/null"

_MSYS_SPAWN_FAILURE_MARKERS = ("dofork:", "child_copy:", "0xc0000142", "0xc0000005")


def _looks_like_msys_spawn_failure(details: str) -> bool:
    """Match Git-for-Windows child-launch failures associated with ASLR."""
    lowered = details.lower()
    return any(marker in lowered for marker in _MSYS_SPAWN_FAILURE_MARKERS)


def _mandatory_aslr_enabled() -> "bool | None":
    """Return Windows' system-wide ForceRelocateImages state when available."""
    global _mandatory_aslr_enabled_cache
    if _mandatory_aslr_enabled_cache is not None:
        return _mandatory_aslr_enabled_cache
    cmd = [shutil.which("powershell.exe") or "powershell.exe", "-NoProfile", "-NonInteractive",
           "-Command", "(Get-ProcessMitigation -System).Aslr.ForceRelocateImages.ToString()"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                                errors="replace", timeout=10, creationflags=windows_hide_flags(), stdin=subprocess.DEVNULL)
    except Exception as exc:
        logger.debug("Could not query Windows Mandatory ASLR state: %s", exc)
        return None
    if result.returncode != 0:
        return None
    value = (result.stdout or "").strip().upper()
    _mandatory_aslr_enabled_cache = {"ON": True, "OFF": False, "NOTSET": False}.get(value)
    return _mandatory_aslr_enabled_cache


def _git_root_from_bash(bash: str) -> str:
    """Resolve Git's root from either <root>/bin or <root>/usr/bin bash."""
    bin_dir = ntpath.dirname(ntpath.normpath(bash))
    if ntpath.basename(bin_dir).lower() != "bin":
        return ntpath.dirname(bin_dir)
    parent = ntpath.dirname(bin_dir)
    return ntpath.dirname(parent) if ntpath.basename(parent).lower() == "usr" else parent


def _git_bash_aslr_help(bash: str, details: str = "") -> str:
    """Build the targeted per-program Mandatory-ASLR remediation."""
    escaped_root = _git_root_from_bash(bash).replace("'", "''")
    detail_line = f"\nGit Bash probe output: {details[:500]}" if details else ""
    return (
        f"Git Bash at {bash} cannot launch required MSYS child processes while "
        "Windows Mandatory ASLR (ForceRelocateImages) is enabled, or its output "
        f"matches that Git-for-Windows failure class.{detail_line}\n"
        "Reinstalling Git will not change the Windows mitigation policy. Open "
        "PowerShell as Administrator and run:\n"
        f"$gitRoot = '{escaped_root}'\n"
        'Get-Item "$gitRoot\\bin\\bash.exe", "$gitRoot\\usr\\bin\\*.exe" '
        "-ErrorAction SilentlyContinue | ForEach-Object { "
        "Set-ProcessMitigation -Name $_.FullName -Disable ForceRelocateImages }\n"
        "Then restart Hermes. If the override is blocked or later re-applied, "
        "ask your Windows administrator to allow this per-program exception."
    )


def _bash_starts(bash: str) -> bool:
    """True if *bash* can launch external MSYS programs (cached per path).
    ``--noprofile --norc`` so a broken login post-install (``Directory
    \\drivers\\etc``) does not falsely condemn an otherwise usable bash.

    On Windows, rejects WSL bash (``linux-gnu`` build) which triggers WSL
    infrastructure (wslrelay, vmmemWSL, wslhost) on every command.  Only
    MSYS/Cygwin builds (Git Bash) are accepted."""
    if bash in _bash_starts_cache:
        return _bash_starts_cache[bash]
    try:
        # bounded_probe_run, not subprocess.run: on Windows run()'s post-timeout cleanup is an
        # unbounded communicate(), and the MSYS children of the probe (true/cat) can outlive the
        # killed bash holding the pipe write ends — the ACP host then wedged for minutes (#73403).
        # The probe's tree is killed and the drain bounded, so a slow host fails the probe fast
        # and _find_bash falls through to its last-resort candidate instead of hanging.
        # stdin=DEVNULL (inside bounded_probe_run) is also what keeps the probe off the ACP
        # host's stdin pipe: cygwin init's handle_to_fn/NtQueryObject stalls ~22 s on a pipe
        # file object with a read pending on it, and the host's stdin reader always has one
        # (@Aaaarminn's strace on #73403). stderr stays captured: the Mandatory-ASLR
        # remediation keys off bash's dofork:/child_copy: text.
        result = bounded_probe_run(
            [bash, "--noprofile", "--norc", "-c", _BASH_EXTERNAL_PROGRAM_PROBE],
            timeout=_BASH_PROBE_TIMEOUT, raise_on_spawn_failure=True)
        ok = result is not None and result.returncode == 0
        if ok and _IS_WINDOWS:
            # Reject WSL bash: its --version reports "linux-gnu" (e.g.
            # "x86_64-pc-linux-gnu") while Git Bash reports "cygwin"/"msys".
            # WSL bash *can* run commands but spawns wslrelay/vmmemWSL/wslhost
            # on every invocation, polluting the process table.
            ver = bounded_probe_run([bash, "--version"], timeout=_BASH_PROBE_TIMEOUT)
            version_output = f"{ver.stdout or ''}{ver.stderr or ''}" if ver is not None else ""
            if "linux-gnu" in version_output.lower():
                ok = False
                _bash_probe_details_cache[bash] = (
                    f"Rejected: WSL bash detected (linux-gnu build). "
                    f"WSL bash triggers wslrelay/vmmemWSL on every command. "
                    f"Use Git Bash (cygwin build) instead.{version_output[:500]}")
                logger.warning("bash probe rejected WSL bash at %s (linux-gnu build)", bash)
        if not ok:
            combined = (f"{result.stdout or ''}{result.stderr or ''}".strip() if result is not None
                        else f"probe timed out after {_BASH_PROBE_TIMEOUT:g}s")
            _bash_probe_details_cache[bash] = combined[:2000]
            logger.debug("bash probe failed for %s: %s", bash, combined[:200])
    except Exception as exc:
        _bash_probe_details_cache[bash] = str(exc)[:2000]
        logger.debug("bash probe error for %s: %s", bash, exc)
        ok = False
    _bash_starts_cache[bash] = ok
    return ok
