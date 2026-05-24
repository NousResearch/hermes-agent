"""Terminal sandbox policy enforcement.

Provides runtime checks that enforce sandbox mode configuration,
ensuring commands run in the configured isolation level (local vs
container-based sandbox).

Config
------
```yaml
terminal:
  sandbox_mode: "container"   # "local" | "container" (default: "local")
  sandbox_deny_native: true   # block native execution when container mode
```

When ``sandbox_mode`` is set to ``"container"``, the terminal tool
automatically routes all commands through the configured container backend
(Docker, Singularity, or Modal).  Commands that cannot be containerised
are blocked with a clear message.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_SANDBOXY_MODES = {"container", "docker", "singularity", "modal"}


def resolve_sandbox_mode(config_env_type: str, sandbox_mode: str = "") -> str:
    """Resolve the effective sandbox mode.

    Parameters
    ----------
    config_env_type:
        The ``env_type`` from terminal config (local, docker, singularity,
        modal, etc.).
    sandbox_mode:
        Explicit ``terminal.sandbox_mode`` override.  Empty string means
        use the env_type as-is.

    Returns
    -------
    str
        The effective environment type to use for execution.
    """
    if sandbox_mode and sandbox_mode.lower() in _SANDBOXY_MODES:
        # Override: force container mode
        return sandbox_mode.lower()

    # Default: use whatever env_type says
    return config_env_type


def check_sandbox_policy(command: str, env_type: str, sandbox_mode: str = "",
                          deny_native: bool = False) -> tuple[bool, Optional[str]]:
    """Check whether a command is allowed under the current sandbox policy.

    Parameters
    ----------
    command:
        The shell command to check.
    env_type:
        Current environment type.
    sandbox_mode:
        Explicit sandbox mode override.
    deny_native:
        If True and sandbox_mode is "container", block native execution.

    Returns
    -------
    (allowed, reason)
        ``allowed`` is True when the command passes the sandbox check.
    ``reason`` is a human-readable explanation when blocked.
    """
    effective_mode = resolve_sandbox_mode(env_type, sandbox_mode)

    # If native execution is denied and we're not in container mode
    if deny_native and effective_mode == "local":
        return False, (
            "Command blocked: native execution is disabled by "
            "terminal.sandbox_deny_native. Set terminal.sandbox_mode to "
            "'container' or use a container backend (docker, singularity, modal)."
        )

    # Warn (but don't block) when running dangerous commands in local mode
    if effective_mode == "local":
        dangerous_patterns = [
            "rm -rf", "mkfs", "dd if=", "dd of=/dev",
            "chmod -R 777", ">/dev/sd",
        ]
        cmd_lower = command.lower()
        for pat in dangerous_patterns:
            if pat in cmd_lower:
                logger.warning(
                    "Sandbox warning: dangerous command in local mode: %s",
                    command[:200],
                )
                return True, (
                    f"⚠️ Warning: Running '{pat}' in local (non-sandboxed) mode. "
                    "Consider setting terminal.sandbox_mode='container' for isolation."
                )

    return True, None
