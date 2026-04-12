"""Utility functions that are reusable among certain terminal backend implementations."""

import logging
import os
import re
import shutil

logger = logging.getLogger(__name__)

_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def normalize_forward_env_names(forward_env: list[str] | None) -> list[str]:
    """Return a deduplicated list of valid environment variable names."""
    normalized: list[str] = []
    seen: set[str] = set()

    for item in forward_env or []:
        if not isinstance(item, str):
            logger.warning("Ignoring non-string docker_forward_env entry: %r", item)
            continue

        key = item.strip()
        if not key:
            continue
        if not _ENV_VAR_NAME_RE.match(key):
            logger.warning("Ignoring invalid docker_forward_env entry: %r", item)
            continue
        if key in seen:
            continue

        seen.add(key)
        normalized.append(key)

    return normalized


def normalize_env_dict(env: dict | None) -> dict[str, str]:
    """Validate and normalize a docker_env dict to {str: str}.

    Filters out entries with invalid variable names or non-string values.
    """
    if not env:
        return {}
    if not isinstance(env, dict):
        logger.warning("docker_env is not a dict: %r", env)
        return {}

    normalized: dict[str, str] = {}
    for key, value in env.items():
        if not isinstance(key, str) or not _ENV_VAR_NAME_RE.match(key.strip()):
            logger.warning("Ignoring invalid docker_env key: %r", key)
            continue
        key = key.strip()
        if not isinstance(value, str):
            # Coerce simple scalar types (int, bool, float) to string;
            # reject complex types.
            if isinstance(value, (int, float, bool)):
                value = str(value)
            else:
                logger.warning("Ignoring non-string docker_env value for %r: %r", key, value)
                continue
        normalized[key] = value

    return normalized


def load_hermes_env_vars() -> dict[str, str]:
    """Load ~/.hermes/.env values without failing Docker/Podman command execution."""
    try:
        from hermes_cli.config import load_env

        return load_env() or {}
    except Exception:
        return {}


def find_container_cli_binary(exec_name: str, search_paths: list[str]):
    """Locate the container CLI binary.

    Checks ``shutil.which`` first (respects PATH), then probes well-known
    install locations where docker or podman may be found.

    Returns the absolute path, or ``None`` if docker or podman cannot be found.
    """
    found = shutil.which(exec_name)
    if found:
        return found

    for path in search_paths:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            logger.info("Found %s at non-PATH location: %s", exec_name, path)
            return path

    return None
