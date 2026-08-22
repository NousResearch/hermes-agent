"""Public environment-variable forwarding helpers for remote backends."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable, Iterable, Mapping

from tools.environments.local import (
    _HERMES_PROVIDER_ENV_BLOCKLIST,
    _is_hermes_internal_secret,
)

logger = logging.getLogger(__name__)

_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def normalize_forward_env_names(
    forward_env: Iterable[object] | None,
    *,
    config_name: str = "forward_env",
) -> list[str]:
    """Return deduplicated, valid environment-variable names."""
    normalized: list[str] = []
    seen: set[str] = set()

    entries = () if forward_env is None else forward_env
    for item in entries:
        if not isinstance(item, str):
            logger.warning("Ignoring non-string %s entry: %r", config_name, item)
            continue
        key = item.strip()
        if not key:
            continue
        if not _ENV_VAR_NAME_RE.fullmatch(key):
            logger.warning("Ignoring invalid %s entry: %r", config_name, item)
            continue
        if key in seen:
            continue
        seen.add(key)
        normalized.append(key)

    return normalized


def load_hermes_env_vars() -> dict[str, str]:
    """Load active Hermes dotenv values without breaking backend execution."""
    try:
        from hermes_cli.config import load_env

        return load_env() or {}
    except Exception:
        return {}


def collect_forwarded_env_values(
    forward_env: list[str] | None,
    *,
    config_name: str = "forward_env",
    dotenv_loader: Callable[[], Mapping[str, str]] = load_hermes_env_vars,
) -> dict[str, str]:
    """Resolve explicit and host-approved implicit environment passthrough.

    Explicit names are a deliberate user opt-in and may include provider
    credentials. Implicit passthrough keeps provider and Hermes-internal
    credentials blocked.
    """
    explicit = set(normalize_forward_env_names(forward_env, config_name=config_name))
    implicit: set[str] = set()
    try:
        from tools.env_passthrough import get_all_passthrough

        # Normalize implicit names independently. Building a set from raw
        # values first would let an unhashable entry crash collection before
        # invalid values can be rejected.
        implicit = set(
            normalize_forward_env_names(
                get_all_passthrough(),
                config_name="implicit environment passthrough",
            )
        )
    except Exception:
        pass

    filtered_implicit = {
        name
        for name in implicit
        if name not in _HERMES_PROVIDER_ENV_BLOCKLIST
        and not _is_hermes_internal_secret(name)
    }
    names = explicit | filtered_implicit
    dotenv_values = dotenv_loader() if names else {}

    resolved: dict[str, str] = {}
    for name in sorted(names):
        value = os.getenv(name) or dotenv_values.get(name)
        if value:
            resolved[name] = value
    return resolved
