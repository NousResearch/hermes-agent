"""Helpers shared by the non-local terminal backends (docker, ssh, singularity, cloud SDKs)."""

from __future__ import annotations

import subprocess


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
