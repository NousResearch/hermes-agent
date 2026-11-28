"""Bubblewrap confinement for the local execute_code kernel child.

The kernel child is the one process that runs model-written code on the host. It
already inherits a scrubbed environment (``tools.code_execution_env``) and reaches
tools only through a 0600 Unix socket guarded by a token, an allow-list and a
per-cell call budget (``tools.code_execution_rpc``). What it does not have is a
filesystem scope or a network block: a cell can open the profile's ``config.yaml``,
walk ``$HOME`` and fetch any URL, and from there the env scrubbing is decoration.

This module builds the ``bwrap`` argv that adds both, and nothing else. It is
deliberately a pure planner: ``plan_kernel_sandbox`` either returns a wrapped argv
or the caller's argv untouched, plus a reason. It never spawns, never raises on a
missing dependency, and never changes behaviour when disabled.

Scope rules, in order of importance:

* no ``/home`` bind at all, so ``$HOME``, ``~/.hermes`` and dotfiles are absent;
* the kernel staging dir is the only writable host path, plus an explicit
  ``extra_rw_paths`` list and, in project mode, the project cwd;
* the RPC socket is bind-mounted as a file: the only channel out of the sandbox;
* ``--unshare-net`` unless ``allow_network`` is set;
* read-only system roots keep the interpreter and subprocess tooling working.

bwrap is optional on purpose. It is absent on macOS, on most CI images and on
Windows, and requiring a new dependency to run execute_code at all would be a
worse trade than running unconfined with the gap documented.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

# Read-only roots that make the interpreter and ordinary subprocess tooling work.
# Everything is --ro-bind-try: a distro without /lib32 or /sbin is not an error.
_SYSTEM_ROOTS = ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/lib32", "/libx64", "/libx32",
                 "/etc", "/sys", "/opt")

_TRUE = frozenset({"1", "true", "yes", "on"})

# ``code_execution.sandbox`` defaults. Kept here rather than in
# hermes_cli/config_defaults.py because that module is already past its line cap on
# main and may not grow; ``hermes config set code_execution.sandbox.enabled true``
# therefore prints the unknown-key notice once, and ``--force`` silences it.
# Keys: enabled, allow_network, extra_ro_paths, extra_rw_paths, bwrap_path.
DEFAULTS = {
    "enabled": False,
    "allow_network": False,
    "extra_ro_paths": [],
    "extra_rw_paths": [],
    "bwrap_path": "",
}


@dataclass(frozen=True)
class SandboxConfig:
    """``code_execution.sandbox`` as resolved from config.yaml."""

    enabled: bool = False
    allow_network: bool = False
    extra_ro_paths: tuple[str, ...] = ()
    extra_rw_paths: tuple[str, ...] = ()
    bwrap_path: str = ""


@dataclass(frozen=True)
class SandboxPlan:
    """The argv the kernel child should be spawned with, and why."""

    argv: tuple[str, ...]
    enabled: bool
    reason: str = ""
    # Host paths the sandbox exposes, for tests and for the log line.
    mounts: tuple[str, ...] = field(default_factory=tuple)


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in _TRUE
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _as_paths(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return ()
    out = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            continue
        expanded = os.path.abspath(os.path.expanduser(item.strip()))
        if os.path.exists(expanded):
            out.append(expanded)
        else:
            logger.warning("code_execution.sandbox: ignoring missing path %s", expanded)
    return tuple(out)


def load_sandbox_config(config: Optional[Mapping[str, Any]] = None) -> SandboxConfig:
    """Resolve ``code_execution.sandbox``. Unreadable config = disabled, with the reason logged."""
    if config is None:
        try:
            from hermes_cli.config import load_config_readonly

            config = (load_config_readonly() or {}).get("code_execution", {}) or {}
        except Exception:  # a config layer failure must not stop execute_code from running
            logger.debug("code_execution.sandbox: config unavailable, disabled", exc_info=True)
            return SandboxConfig()
    section = config.get("sandbox") if isinstance(config, Mapping) else None
    if not isinstance(section, Mapping):
        section = DEFAULTS
    return SandboxConfig(
        enabled=_as_bool(section.get("enabled"), False),
        allow_network=_as_bool(section.get("allow_network"), False),
        extra_ro_paths=_as_paths(section.get("extra_ro_paths")),
        extra_rw_paths=_as_paths(section.get("extra_rw_paths")),
        bwrap_path=(section.get("bwrap_path") or "").strip() if isinstance(section.get("bwrap_path"), str) else "",
    )


def sandbox_enabled(config: Optional[SandboxConfig] = None) -> bool:
    """True when a spawn would actually be wrapped: opt-in config plus a usable bwrap."""
    cfg = config or load_sandbox_config()
    return cfg.enabled and _resolve_bwrap(cfg) != ""


def _resolve_bwrap(cfg: SandboxConfig) -> str:
    if cfg.bwrap_path:
        return cfg.bwrap_path if os.path.isfile(cfg.bwrap_path) and os.access(cfg.bwrap_path, os.X_OK) else ""
    if sys.platform.startswith("linux"):
        return shutil.which("bwrap") or ""
    return ""


def interpreter_roots(python_path: str) -> tuple[str, ...]:
    """Read-only roots the interpreter needs: its own prefix and the base prefix behind a venv.

    A venv python is a symlink to the base interpreter but resolves its stdlib through
    ``pyvenv.cfg``'s ``home``, so binding only the venv leaves ``import sys`` working and
    ``import sqlite3`` failing. Both prefixes, plus whatever this process is running from,
    go in the mount list.
    """
    roots: list[str] = []
    try:
        resolved = Path(python_path).resolve()
        prefix = resolved.parent.parent
        roots.append(str(prefix))
        cfg = prefix / "pyvenv.cfg"
        if cfg.is_file():
            for line in cfg.read_text(encoding="utf-8-sig", errors="replace").splitlines():
                key, _, value = line.partition("=")
                if key.strip().lower() == "home" and value.strip():
                    roots.append(str(Path(value.strip()).resolve().parent))
    except OSError:
        logger.debug("code_execution.sandbox: could not resolve %s", python_path, exc_info=True)
    for prefix in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix):
        if prefix:
            roots.append(prefix)
    seen, unique = set(), []
    for root in roots:
        if root and root not in seen and os.path.isdir(root):
            seen.add(root)
            unique.append(root)
    return tuple(unique)


def build_sandbox_argv(
    *,
    bwrap: str,
    real_python: str,
    kernel_dir: str,
    rpc_socket: str,
    child_cwd: str,
    config: SandboxConfig,
) -> list[str]:
    """The bwrap prefix, ending with the interpreter the caller's argv should follow."""
    args = [
        bwrap,
        "--die-with-parent",   # host death takes the sandbox with it
        "--new-session",       # no controlling tty
        "--unshare-user",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-cgroup-try",
    ]
    if not config.allow_network:
        args.append("--unshare-net")   # no loopback either: the RPC socket is a file, not a port
    # The guest gets its own tmpfs; this is inside the sandbox, not a host scratch path.
    args += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]  # no-tmp: ok — sandbox-internal mount point

    for root in _SYSTEM_ROOTS:
        args += ["--ro-bind-try", root, root]
    for root in interpreter_roots(real_python):
        args += ["--ro-bind", root, root]
    for path in config.extra_ro_paths:
        args += ["--ro-bind", path, path]

    # The only writable host paths. /home is never bound wholesale.
    args += ["--bind", kernel_dir, kernel_dir]
    for path in config.extra_rw_paths:
        args += ["--bind", path, path]
    if child_cwd and os.path.abspath(child_cwd) != os.path.abspath(kernel_dir):
        args += ["--bind", child_cwd, child_cwd]

    if rpc_socket:
        args += ["--bind", rpc_socket, rpc_socket]
    if child_cwd:
        args += ["--chdir", child_cwd]
    args.append(real_python)
    return args


def plan_kernel_sandbox(
    argv: Sequence[str],
    *,
    kernel_dir: str,
    rpc_socket: str,
    child_cwd: str,
    config: Optional[SandboxConfig] = None,
) -> SandboxPlan:
    """Wrap *argv* in bwrap when configured and possible; otherwise return it untouched.

    ``argv[0]`` is the interpreter the kernel would have used. Fail-open is deliberate:
    a missing bwrap, a non-Linux host or a malformed path returns the caller's argv with a
    reason, so execute_code keeps working exactly as it did before this module existed.
    """
    cfg = config or load_sandbox_config()
    original = tuple(argv)
    if not cfg.enabled:
        return SandboxPlan(original, False, "disabled")
    bwrap = _resolve_bwrap(cfg)
    if not bwrap:
        return SandboxPlan(original, False, "bwrap not found" if sys.platform.startswith("linux") else "unsupported platform")
    if not original:
        return SandboxPlan(original, False, "empty argv")
    for label, path in (("kernel_dir", kernel_dir), ("child_cwd", child_cwd)):
        if not path or not os.path.isdir(path):
            return SandboxPlan(original, False, f"{label} missing")
    if rpc_socket and not os.path.exists(rpc_socket):
        return SandboxPlan(original, False, "rpc socket missing")

    prefix = build_sandbox_argv(
        bwrap=bwrap, real_python=original[0], kernel_dir=os.path.abspath(kernel_dir),
        rpc_socket=os.path.abspath(rpc_socket) if rpc_socket else "",
        child_cwd=os.path.abspath(child_cwd), config=cfg,
    )
    mounts = tuple(
        prefix[i + 1] for i, token in enumerate(prefix[:-1])
        if token in ("--bind", "--ro-bind", "--ro-bind-try")
    )
    return SandboxPlan(tuple(prefix) + tuple(original[1:]), True, "bwrap", mounts)
