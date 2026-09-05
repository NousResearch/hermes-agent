"""Helpers shared by the non-local terminal backends (docker, ssh, singularity, cloud SDKs)."""

from __future__ import annotations

import os
import shlex
import subprocess
from typing import Callable, Iterable

from tools.environments.base_session_env import _SHELL_ENV_NAME_RE
from tools.environments.local_env_policy import _HERMES_PROVIDER_ENV_BLOCKLIST, _is_hermes_internal_secret


def load_hermes_env_vars() -> dict[str, str]:
    """``~/.hermes/.env`` values, or ``{}`` — a broken .env must not fail command execution."""
    try:
        from hermes_cli.config import load_env
        return load_env() or {}
    except Exception:
        return {}


def resolve_passthrough_env(explicit_forward: Iterable[str] = (),
                            hermes_env_loader: Callable[[], dict[str, str]] = load_hermes_env_vars,
                            ) -> tuple[dict[str, str], set[str]]:
    """Values to forward into a remote shell plus the scoped names that must be unset there.

    Implicit passthrough (skill ``required_environment_variables`` + ``terminal.env_passthrough``)
    is filtered through the Hermes provider-credential blocklist and the dynamic internal-secret
    check; ``explicit_forward`` entries (docker_forward_env) are an operator opt-in that bypasses
    both. Each value is the routed profile's secret when multiplex is active; a name the active
    scope lacks is returned in the unset set so a shared sandbox cannot leak another profile's
    value.
    """
    passthrough_keys: set[str] = set()
    resolve_passthrough_value = None
    multiplex_active = False
    is_global_env = lambda _name: False  # noqa: E731
    try:
        from tools.env_passthrough import get_all_passthrough, resolve_passthrough_value
        from agent.secret_scope import _is_global_env as is_global_env, is_multiplex_active
        multiplex_active = is_multiplex_active()
        passthrough_keys = set(get_all_passthrough())
    except Exception:
        pass
    implicit_forward = {k for k in passthrough_keys if not _is_hermes_internal_secret(k)}
    forward_keys = set(explicit_forward) | (implicit_forward - _HERMES_PROVIDER_ENV_BLOCKLIST)
    hermes_env = hermes_env_loader() if forward_keys else {}
    exec_env: dict[str, str] = {}
    unset_names: set[str] = set()
    for key in sorted(forward_keys):
        value = os.getenv(key) or hermes_env.get(key)
        if resolve_passthrough_value is not None:
            value = resolve_passthrough_value(key, value)
        if value is not None:
            exec_env[key] = value
        elif multiplex_active and not is_global_env(key) and _SHELL_ENV_NAME_RE.fullmatch(key):
            unset_names.add(key)
    return exec_env, unset_names


def prepend_unset(cmd_string: str, names: Iterable[str]) -> str:
    """Prefix ``cmd_string`` with ``unset`` of the profile-scoped names the remote shell must not see."""
    names = sorted(names)
    if not names:
        return cmd_string
    return f"unset {' '.join(shlex.quote(n) for n in names)} 2>/dev/null || true\n{cmd_string}"


def client_env_with(values: dict[str, str]) -> dict[str, str] | None:
    """Env for the docker/ssh CLIENT subprocess: forwarded values travel here (owner-readable
    /proc/*/environ) while the argv carries names only; ``None`` = inherit when nothing to add."""
    return {**os.environ, **values} if values else None


def run_capture(cmd: list[str], *, timeout: float, check: bool = False, env: dict | None = None,
                ) -> subprocess.CompletedProcess:
    """``subprocess.run`` with the backend-standard capture settings: text mode with utf-8/replace
    decoding and stdin closed (DEVNULL) so a CLI that unexpectedly prompts cannot hang the agent."""
    return subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout, check=check, stdin=subprocess.DEVNULL, env=env)


def bash_argv(cmd_string: str, login: bool = False) -> list[str]:
    """``bash [-l] -c <cmd>`` argv tail used by every spawn-per-call backend."""
    return ["bash", "-l", "-c", cmd_string] if login else ["bash", "-c", cmd_string]


def ensure_lazy_dep(feature: str) -> None:
    """Lazy-install an optional SDK via ``tools.lazy_deps`` (idempotent). Missing ``tools.lazy_deps``
    is tolerated (the SDK import that follows fails with its own message); any other failure
    surfaces as ``ImportError``."""
    try:
        from tools.lazy_deps import ensure as _lazy_ensure
        _lazy_ensure(feature, prompt=False)
    except ImportError:
        pass
    except Exception as e:
        raise ImportError(str(e))
