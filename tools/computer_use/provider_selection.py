"""Fail-closed desktop selection, including pre-provider remote configuration.

Presence matters: defaults must not turn an absent provider into explicit local,
or an omitted remote.enabled into an explicit veto. Read raw files only to
validate them and retain that provenance; values still use the normal expanded,
managed-scope-aware config loader.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, cast

from agent.computer_use_registry import HOST_PROVIDER_NAME, _HOST_ALIASES

logger = logging.getLogger(__name__)


def _strict_block(path: Path) -> dict[str, Any]:
    from utils import fast_safe_load

    try:
        with path.open(encoding="utf-8") as stream:
            raw = fast_safe_load(stream)
    except FileNotFoundError:
        return {}
    except Exception as exc:
        # Do not echo YAML parser context: the file can contain credentials.
        raise RuntimeError(f"computer_use cannot read valid config at {path}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise RuntimeError("computer_use config root must be a mapping")
    block = raw.get("computer_use", {})
    if not isinstance(block, dict):
        raise RuntimeError("computer_use configuration must be a mapping")
    return block


def _same_config(left: Any, right: Any) -> bool:
    """Compare the complete block without bool/int equality hiding bad intent."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same_config(value, right[key]) for key, value in left.items()
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(_same_config(a, b) for a, b in zip(left, right))
    return left == right


def computer_use_config() -> dict[str, Any]:
    from hermes_cli import config

    raw = _strict_block(config.get_config_path())
    managed_dir = config.managed_scope.get_managed_dir()
    managed = _strict_block(managed_dir / "config.yaml") if managed_dir else {}
    provenance = {**raw, **managed}
    if isinstance(raw.get("remote"), dict) and isinstance(managed.get("remote"), dict):
        provenance["remote"] = {**raw["remote"], **managed["remote"]}
    try:
        effective = config.load_config().get("computer_use", {})
    except Exception as exc:
        raise RuntimeError("computer_use configuration could not be loaded") from exc
    if not isinstance(effective, dict):
        raise RuntimeError("computer_use configuration must be a mapping")
    # The general loader may silently return defaults or a cached last-good
    # config after an unrelated processing error. Neither may override current
    # desktop intent. Rebuild this block with the loader's defaults, expansion
    # and managed precedence: overlaying current leaves onto the loaded result
    # cannot detect deleted keys resurrected from last-good state. The loader's
    # canonicalization only changes agent/model, not computer_use.
    intended = config._deep_merge(
        cast(dict[str, Any], config._expand_env_vars(
            config._deep_merge(cast(dict[str, Any], config.DEFAULT_CONFIG.get("computer_use", {})), raw),
        )),
        cast(dict[str, Any], config._expand_env_vars(managed)),
    )
    if not _same_config(effective, intended):
        raise RuntimeError("computer_use configuration does not match current desktop intent; fix config.yaml")
    block = dict(effective)
    if "remote" not in provenance:
        block.pop("remote", None)  # only the disabled default block was present
    elif not isinstance(provenance["remote"], dict):
        block["remote"] = provenance["remote"]
    elif isinstance(block.get("remote"), dict):
        block["remote"] = dict(block["remote"])
        if "enabled" not in provenance["remote"]:
            block["remote"].pop("enabled", None)
    return block


def _name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError("computer_use.provider must be a non-empty string")
    name = value.strip().lower()
    return HOST_PROVIDER_NAME if name in _HOST_ALIASES else name


def configured_provider_name() -> str:
    block = computer_use_config()
    explicit = _name(block["provider"]) if "provider" in block else None
    legacy = os.environ.get("HERMES_COMPUTER_USE_BACKEND", "").strip()
    # An explicit provider owns transport choice, not any leftover remote block.
    remote = block.get("remote", {})
    inspect_remote = explicit is None or explicit == "remote" or legacy.lower() in {"cua", "cua-driver"}
    remote_intent = False
    if inspect_remote:
        if not isinstance(remote, dict):
            raise RuntimeError("remote computer use configuration must be a mapping")
        if "enabled" in remote and not isinstance(remote["enabled"], bool):
            raise RuntimeError("remote computer use configuration 'enabled' must be a boolean")
        remote_intent = remote.get("enabled") is True
        if explicit is None and remote and "enabled" not in remote:
            raise RuntimeError("orphaned computer_use.remote configuration: set computer_use.provider explicitly")
    if legacy:
        legacy_name = "remote" if legacy.lower() in {"cua", "cua-driver"} and remote_intent else _name(legacy)
        if explicit is not None and explicit != legacy_name:
            raise RuntimeError("computer_use.provider conflicts with deprecated HERMES_COMPUTER_USE_BACKEND; remove the env selector")
        logger.warning("HERMES_COMPUTER_USE_BACKEND is deprecated; set computer_use.provider in config.yaml instead.")
        explicit = explicit or legacy_name
    if remote_intent and "provider" not in block and (not legacy or legacy.lower() in {"cua", "cua-driver"}):
        logger.warning("computer_use.remote.enabled is deprecated as a selector; preserving the remote desktop. Set computer_use.provider: remote.")
        return "remote"
    return explicit or HOST_PROVIDER_NAME
