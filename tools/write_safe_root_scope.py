"""Per-turn HERMES_WRITE_SAFE_ROOT scope for multiplexed surfaces (#70688).

Multiplexed hosts load the launch profile's dotenv into ``os.environ`` once; routed
profile turns must not inherit that vault. Like ``tools/terminal_scope``, a ContextVar
holds the active profile's value; while bound, ``agent.file_safety.get_safe_write_roots()``
resolves ONLY from it (never ambient env). An empty profile value means no profile vault,
but inherited launch-process roots stay blocked (#70688 M2).
"""

from __future__ import annotations

import os
from contextlib import contextmanager, suppress
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any, Iterator, Optional

_write_safe_root_scope_var: ContextVar[Optional[str]] = ContextVar(
    "hermes_write_safe_root_scope", default=None,
)


def set_write_safe_root_scope(value: Optional[str]) -> Token:
    return _write_safe_root_scope_var.set(value)


def reset_write_safe_root_scope(token: Token) -> None:
    _write_safe_root_scope_var.reset(token)


def get_write_safe_root_scope() -> Optional[str]:
    return _write_safe_root_scope_var.get()


def is_write_safe_root_scope_bound() -> bool:
    return get_write_safe_root_scope() is not None


def get_process_write_safe_roots() -> set[str]:
    """Resolved ``HERMES_WRITE_SAFE_ROOT`` from the launch process env only."""
    roots: set[str] = set()
    for path in filter(None, os.environ.get("HERMES_WRITE_SAFE_ROOT", "").split(os.pathsep)):
        with suppress(OSError, ValueError):
            roots.add(os.path.realpath(os.path.expanduser(path)))
    return roots


def write_safe_root_env() -> str:
    """Authoritative read of ``HERMES_WRITE_SAFE_ROOT`` for file write guards."""
    scope = _write_safe_root_scope_var.get()
    if scope is None:
        return os.environ.get("HERMES_WRITE_SAFE_ROOT", "")
    return scope


def build_profile_write_safe_root(hermes_home: Any) -> str:
    """Read ``HERMES_WRITE_SAFE_ROOT`` from a profile home's ``.env`` only."""
    env_path = Path(hermes_home) / ".env"
    if not env_path.is_file():
        return ""
    from agent.secret_scope import load_env_file

    value = load_env_file(env_path).get("HERMES_WRITE_SAFE_ROOT")
    return "" if value is None else str(value)


def install_profile_write_safe_root_scope(hermes_home: Any) -> Token:
    return set_write_safe_root_scope(build_profile_write_safe_root(hermes_home))


@contextmanager
def install_and_reset_profile_write_safe_root_scope(hermes_home: Any) -> Iterator[None]:
    token = install_profile_write_safe_root_scope(hermes_home)
    try:
        yield
    finally:
        reset_write_safe_root_scope(token)
