"""Shared environment forwarding helpers for remote/container backends."""

from __future__ import annotations

import logging
import os
import re

from tools.environments.local import _HERMES_PROVIDER_ENV_BLOCKLIST

logger = logging.getLogger(__name__)

_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def normalize_forward_env_names(
    forward_env: list[str] | None,
    *,
    config_name: str = "forward_env",
) -> list[str]:
    """Return a deduplicated list of valid environment variable names."""
    normalized: list[str] = []
    seen: set[str] = set()

    for item in forward_env or []:
        if not isinstance(item, str):
            logger.warning("Ignoring non-string %s entry: %r", config_name, item)
            continue

        key = item.strip()
        if not key:
            continue
        if not _ENV_VAR_NAME_RE.match(key):
            logger.warning("Ignoring invalid %s entry: %r", config_name, item)
            continue
        if key in seen:
            continue

        seen.add(key)
        normalized.append(key)

    return normalized


def load_hermes_env_vars() -> dict[str, str]:
    """Load ~/.hermes/.env values without failing backend command execution."""
    try:
        from hermes_cli.config import load_env

        return load_env() or {}
    except Exception:
        return {}


def collect_forwarded_env_values(
    forward_env: list[str] | None,
    *,
    config_name: str = "forward_env",
    dotenv_loader=load_hermes_env_vars,
) -> dict[str, str]:
    """Resolve forwarded environment variables from process env or ~/.hermes/.env.

    Explicit forward_env entries are an intentional opt-in and bypass the
    provider-secret blocklist. Implicit env_passthrough entries keep using the
    blocklist so provider credentials are not copied unless explicitly named.
    """
    explicit_forward_keys = set(normalize_forward_env_names(forward_env, config_name=config_name))
    passthrough_keys: set[str] = set()
    try:
        from tools.env_passthrough import get_all_passthrough

        passthrough_keys = set(get_all_passthrough())
    except Exception:
        pass

    forward_keys = explicit_forward_keys | (passthrough_keys - _HERMES_PROVIDER_ENV_BLOCKLIST)
    hermes_env = dotenv_loader() if forward_keys else {}

    resolved: dict[str, str] = {}
    for key in sorted(forward_keys):
        value = os.getenv(key)
        if not value:
            value = hermes_env.get(key)
        if value:
            resolved[key] = value
    return resolved
