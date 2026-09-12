"""macOS Seatbelt policy construction for the opt-in local terminal sandbox."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path


SANDBOX_EXEC = "/usr/bin/sandbox-exec"
_DISABLED_MODES = {"", "none", "off", "false"}
_NETWORK_POLICIES = {"allow", "deny"}
_SYSTEM_READ_PATHS = (
    "/System",
    "/usr",
    "/bin",
    "/sbin",
    "/Library/Apple",
    "/private/etc",
    "/private/var/db",
    "/dev",
)


def _normalized(value: object) -> str:
    return str(value or "").strip().lower()


def validate_seatbelt_config(
    sandbox_mode: object,
    network_policy: object,
    *,
    system_name: str | None = None,
    sandbox_exec: str = SANDBOX_EXEC,
) -> tuple[str, str]:
    """Validate local-sandbox policy before any child process can launch."""
    mode = _normalized(sandbox_mode)
    network = _normalized(network_policy)
    if mode not in {*_DISABLED_MODES, "seatbelt"}:
        raise ValueError(
            f"Invalid terminal.local_sandbox value {sandbox_mode!r}; use 'none' or 'seatbelt'."
        )
    if network not in _NETWORK_POLICIES:
        raise ValueError(
            f"Invalid terminal.local_sandbox_network value {network_policy!r}; "
            "use 'allow' or 'deny'."
        )
    if mode != "seatbelt":
        return "none", network
    if (system_name or platform.system()) != "Darwin":
        raise RuntimeError(
            "terminal.local_sandbox=seatbelt is available only on macOS; "
            "disable it or select an isolated backend supported by this host."
        )
    if not (os.path.isfile(sandbox_exec) and os.access(sandbox_exec, os.X_OK)):
        raise RuntimeError(
            f"terminal.local_sandbox=seatbelt was requested, but {sandbox_exec} is not "
            "available or executable. Restore the macOS system sandbox-exec binary or disable Seatbelt."
        )
    return mode, network


def _sbpl_string(path: str) -> str:
    """Encode a path as a Seatbelt string literal."""
    return json.dumps(str(Path(path).resolve()))


def _path_rule(actions: str, path: str) -> str:
    return f"(allow {actions} (subpath {_sbpl_string(path)}))"


def runtime_read_paths(shell: str, env: Mapping[str, str]) -> list[str]:
    """Return executable/runtime roots needed by the shell and child commands."""
    candidates: list[str] = [*_SYSTEM_READ_PATHS, shell, sys.executable, sys.prefix, sys.base_prefix]
    candidates.extend(filter(None, env.get("PATH", "").split(os.pathsep)))
    # Homebrew and locally installed runtimes commonly resolve through symlinks in bin/;
    # include both the link and target parents without making either writable.
    paths: list[str] = []
    for raw in candidates:
        try:
            candidate = Path(raw).resolve()
            if candidate.is_file():
                candidate = candidate.parent
            if candidate.is_absolute() and candidate.exists():
                paths.append(str(candidate))
        except OSError:
            continue
    return list(dict.fromkeys(paths))


def build_seatbelt_profile(
    *, cwd: str, temp_dir: str, read_paths: Iterable[str], network_policy: str,
    read_files: Iterable[str] = (), protected_write_paths: Iterable[str] = (),
) -> str:
    """Build a deny-by-default policy with writes limited to cwd and session temp."""
    network = _normalized(network_policy)
    if network not in _NETWORK_POLICIES:
        raise ValueError(
            f"Invalid terminal.local_sandbox_network value {network_policy!r}; "
            "use 'allow' or 'deny'."
        )
    writable_cwd = Path(cwd).resolve()
    user_home = Path.home().resolve()
    if writable_cwd == user_home or user_home.is_relative_to(writable_cwd):
        raise ValueError(
            "terminal.local_sandbox=seatbelt refuses a cwd that contains the OS-user home; "
            "configure terminal.cwd to a project subdirectory."
        )
    readable = list(dict.fromkeys([*read_paths, cwd, temp_dir]))
    lines = [
        "(version 1)",
        "(deny default)",
        "(allow process-exec)",
        "(allow process-fork)",
        "(allow signal (target same-sandbox))",
        "(allow process-info* (target same-sandbox))",
        """(allow sysctl-read
    (sysctl-name "hw.activecpu")
    (sysctl-name "hw.byteorder")
    (sysctl-name "hw.cacheconfig")
    (sysctl-name "hw.cachelinesize_compat")
    (sysctl-name "hw.cpufamily")
    (sysctl-name "hw.cputype")
    (sysctl-name "hw.logicalcpu")
    (sysctl-name "hw.logicalcpu_max")
    (sysctl-name "hw.machine")
    (sysctl-name "hw.memsize")
    (sysctl-name "hw.model")
    (sysctl-name "hw.ncpu")
    (sysctl-name "hw.pagesize")
    (sysctl-name "hw.physicalcpu")
    (sysctl-name "hw.physicalcpu_max")
    (sysctl-name-prefix "hw.optional.arm.")
    (sysctl-name-prefix "hw.optional.armv8_")
    (sysctl-name "kern.argmax")
    (sysctl-name "kern.hostname")
    (sysctl-name "kern.osproductversion")
    (sysctl-name "kern.osrelease")
    (sysctl-name "kern.ostype")
    (sysctl-name "kern.osversion")
    (sysctl-name "kern.version")
    (sysctl-name "vm.loadavg"))""",
        """(allow mach-lookup
    (global-name "com.apple.cfprefsd.agent")
    (global-name "com.apple.cfprefsd.daemon")
    (global-name "com.apple.logd")
    (global-name "com.apple.system.opendirectoryd.libinfo")
    (global-name "com.apple.system.opendirectoryd.membership")
    (global-name "com.apple.trustd")
    (global-name "com.apple.trustd.agent")
    (local-name "com.apple.cfprefsd.agent"))""",
        "(allow ipc-posix-sem)",
        "(allow pseudo-tty)",
        # getcwd/path traversal needs the filesystem root itself, without granting its subtree.
        '(allow file-read* file-test-existence (literal "/"))',
        '(allow file-read-metadata file-test-existence (literal "/etc") (literal "/tmp") (literal "/var"))',
        f"(allow file-read-metadata file-test-existence "
        f"(path-ancestors {_sbpl_string(cwd)}) (path-ancestors {_sbpl_string(temp_dir)}))",
        '(allow file-write-data (literal "/dev/null"))',
        *(_path_rule("file-read*", path) for path in readable),
        *(f"(allow file-read* (literal {_sbpl_string(path)}))" for path in read_files),
        _path_rule("file-write*", cwd),
        _path_rule("file-write*", temp_dir),
        *(f"(deny file-write* (literal {_sbpl_string(path)}))" for path in protected_write_paths),
        f"({'allow' if network == 'allow' else 'deny'} network*)",
    ]
    return "\n".join(lines) + "\n"


def prepare_seatbelt_profile(
    *, cwd: str, temp_root: str, shell: str, env: Mapping[str, str], network_policy: str,
    read_files: Iterable[str] = (),
) -> tuple[str, str]:
    """Create a private session directory and profile, or fail before spawning."""
    session_dir: str | None = None
    try:
        session_dir = tempfile.mkdtemp(prefix="hermes-seatbelt-", dir=temp_root)
        profile_path = Path(session_dir) / "profile.sb"
        profile = build_seatbelt_profile(
            cwd=cwd,
            temp_dir=session_dir,
            read_paths=runtime_read_paths(shell, env),
            network_policy=network_policy,
            read_files=read_files,
            protected_write_paths=[str(profile_path)],
        )
        profile_path.write_text(profile, encoding="utf-8")
        probe = subprocess.run(
            seatbelt_spawn_args(["/usr/bin/true"], str(profile_path)),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if probe.returncode != 0:
            detail = (probe.stderr or probe.stdout or f"exit code {probe.returncode}").strip()
            raise RuntimeError(f"sandbox-exec could not activate the generated profile: {detail}")
        return str(profile_path), session_dir
    except Exception as exc:
        if session_dir:
            shutil.rmtree(session_dir, ignore_errors=True)
        raise RuntimeError(
            "Failed to create or activate the macOS Seatbelt profile for the local terminal; "
            f"check terminal.temp_dir permissions or disable terminal.local_sandbox: {exc}"
        ) from exc


def seatbelt_spawn_args(args: Sequence[str], profile_path: str | None) -> list[str]:
    """Prefix one local child argv when Seatbelt is active; otherwise pass through."""
    original = list(args)
    if not profile_path:
        return original
    return [SANDBOX_EXEC, "-f", profile_path, *original]
