#!/usr/bin/env python3
"""Shared private TOML/env credential resolution for YallaPlay tool wrappers.

This module centralizes source precedence only. It never prints secret values and
leaves service-specific required-key validation in each caller.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python <3.11
    import tomli as tomllib  # type: ignore[no-redef]

PROJECT_DIR = Path(__file__).resolve().parents[1]
LOCAL_VARS = PROJECT_DIR / "vars.toml"
DEFAULT_SIBLING_VARS = PROJECT_DIR.parent / "yallaplay-analytics-agent-gpt" / "vars.toml"
COMMON_VARS_ENV_NAMES = ("HERMES_YALLAPLAY_VARS", "YALLAPLAY_VARS_TOML")


def load_toml(path: Path) -> dict[str, Any]:
    """Load a private TOML file as a dict."""
    with path.open("rb") as handle:
        return tomllib.load(handle)


def candidate_vars_paths(
    explicit: Path | None = None,
    *,
    env_var_names: Iterable[str] = COMMON_VARS_ENV_NAMES,
    include_local: bool = True,
    include_legacy: bool = True,
    environ: Mapping[str, str] | None = None,
) -> list[Path]:
    """Return existing private TOML candidates in repo-standard precedence order."""
    env = os.environ if environ is None else environ
    candidates: list[Path | None] = [explicit]
    candidates.extend(Path(env[name]) for name in env_var_names if env.get(name))
    if include_local:
        candidates.append(LOCAL_VARS if LOCAL_VARS.exists() else None)
    if include_legacy:
        candidates.append(DEFAULT_SIBLING_VARS if DEFAULT_SIBLING_VARS.exists() else None)
    return [path for path in candidates if path and path.exists()]


def first_value(mapping: Mapping[str, Any], keys: Iterable[str]) -> str | None:
    """Return the first non-empty mapping value for any key, coerced to str."""
    for key in keys:
        value = mapping.get(key)
        if value:
            return str(value)
    return None


def env_mapping_if_complete(
    required_keys: Iterable[str],
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str] | None:
    """Return environment mapping only when all required keys are present."""
    env = os.environ if environ is None else environ
    required = tuple(required_keys)
    if all(env.get(key) for key in required):
        return dict(env)
    return None


def pick_mapping_source(
    explicit: Path | None,
    *,
    env_required: Iterable[str],
    vars_env_names: Iterable[str] = COMMON_VARS_ENV_NAMES,
    missing_message: str,
    include_local: bool = True,
    include_legacy: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Pick env first when complete, otherwise the first existing private TOML.

    Callers that allow partial environment/CLI overrides should use
    candidate_vars_paths() directly instead.
    """
    env_mapping = env_mapping_if_complete(env_required)
    if env_mapping is not None:
        return "environment", env_mapping

    for path in candidate_vars_paths(
        explicit,
        env_var_names=vars_env_names,
        include_local=include_local,
        include_legacy=include_legacy,
    ):
        return str(path), load_toml(path)
    raise SystemExit(missing_message)
