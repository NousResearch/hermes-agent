"""Environment variable passthrough registry.

Skills can declare ``required_environment_variables`` in their frontmatter.
By default subprocess sandboxes strip secrets from the child process
environment for security. This module provides a session-scoped allowlist
so non-managed skill vars and user-configured overrides can pass through.
Hermes-managed secrets are never allowlisted into model-controlled children.

Two sources feed the allowlist:

1. **Skill declarations** — when a skill is loaded via ``skill_view``, its
   non-blocklisted ``required_environment_variables`` are registered here.
2. **User config** — ``terminal.env_passthrough`` in config.yaml lets users
   explicitly allowlist non-managed vars for non-skill use cases.

``code_execution_tool.py`` consults :func:`is_env_passthrough` before
stripping a variable; terminal backends use :func:`get_all_passthrough` only
for non-local sandbox adapters that require an explicit forwarded-var list.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Iterable
from hermes_cli.config import cfg_get

logger = logging.getLogger(__name__)

# Session-scoped set of env var names that should pass through to sandboxes.
# Backed by ContextVar to prevent cross-session data bleed in the gateway pipeline.
_allowed_env_vars_var: ContextVar[set[str]] = ContextVar("_allowed_env_vars")


def _get_allowed() -> set[str]:
    """Get or create the allowed env vars set for the current context/session."""
    try:
        return _allowed_env_vars_var.get()
    except LookupError:
        val: set[str] = set()
        _allowed_env_vars_var.set(val)
        return val


# Cache for the config-based allowlist (loaded once per process).
_config_passthrough: frozenset[str] | None = None


def _is_blocklisted_secret(name: str) -> bool:
    """True if ``name`` is a Hermes-managed secret that must not be
    forwarded into model-controlled subprocess environments.

    Skill-declared and config-declared passthrough must not be able to
    override this list; otherwise managed credentials from ``.env`` can be
    disclosed by terminal or execute_code subprocesses.
    """
    try:
        from tools.environments.local import _HERMES_PROVIDER_ENV_BLOCKLIST
    except Exception:
        return False
    return name in _HERMES_PROVIDER_ENV_BLOCKLIST


def register_env_passthrough(var_names: Iterable[str]) -> None:
    """Register environment variable names as allowed in sandboxed environments.

    Typically called when a skill declares ``required_environment_variables``.

    Hermes-managed secrets (from ``_HERMES_PROVIDER_ENV_BLOCKLIST``) are
    rejected here to preserve subprocess credential-scrubbing guarantees.
    This includes provider credentials and optional password variables saved
    by Hermes, such as bundled-skill API keys. Unmanaged, caller-owned
    variables can still be allowlisted explicitly.
    """
    for name in var_names:
        name = name.strip()
        if not name:
            continue
        if _is_blocklisted_secret(name):
            logger.warning(
                "env passthrough: refusing to register managed secret %r "
                "(blocked by _HERMES_PROVIDER_ENV_BLOCKLIST). Skills and "
                "terminal.env_passthrough must not override subprocess "
                "credential scrubbing.",
                name,
            )
            continue
        _get_allowed().add(name)
        logger.debug("env passthrough: registered %s", name)


def _load_config_passthrough() -> frozenset[str]:
    """Load ``tools.env_passthrough`` from config.yaml (cached)."""
    global _config_passthrough
    if _config_passthrough is not None:
        return _config_passthrough

    result: set[str] = set()
    try:
        from hermes_cli.config import read_raw_config
        cfg = read_raw_config()
        passthrough = cfg_get(cfg, "terminal", "env_passthrough")
        if isinstance(passthrough, list):
            for item in passthrough:
                if isinstance(item, str) and item.strip():
                    result.add(item.strip())
    except Exception as e:
        logger.debug("Could not read tools.env_passthrough from config: %s", e)

    _config_passthrough = frozenset(result)
    return _config_passthrough


def is_env_passthrough(var_name: str) -> bool:
    """Check whether *var_name* is allowed to pass through to sandboxes.

    Returns ``True`` if the variable was registered by a skill or listed in
    the user's ``tools.env_passthrough`` config, unless it is a managed
    secret that subprocess sandboxes must always scrub.
    """
    if _is_blocklisted_secret(var_name):
        return False
    if var_name in _get_allowed():
        return True
    return var_name in _load_config_passthrough()


def get_all_passthrough() -> frozenset[str]:
    """Return non-blocklisted skill-registered and config passthrough vars."""
    return frozenset(
        name
        for name in (frozenset(_get_allowed()) | _load_config_passthrough())
        if not _is_blocklisted_secret(name)
    )


def clear_env_passthrough() -> None:
    """Reset the skill-scoped allowlist (e.g. on session reset)."""
    _get_allowed().clear()


