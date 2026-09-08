"""Validate current desktop intent before trusting the permissive config loader.

Only the computer_use block is reconstructed; the general loader retains its
last-known-good policy for unrelated settings. Values use its canonical defaults,
environment expansion and managed precedence, not a second config API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast


def _strict_block(path: Path) -> dict[str, Any]:
    from utils import fast_safe_load

    try:
        with path.open(encoding="utf-8") as stream:
            raw = fast_safe_load(stream)
    except FileNotFoundError:
        return {}
    except Exception as exc:
        # YAML parser context can contain credentials; never surface it to callers.
        raise RuntimeError("computer_use cannot read valid config.yaml") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise RuntimeError("computer_use config root must be a mapping")
    block = raw.get("computer_use", {})
    if not isinstance(block, dict):
        raise RuntimeError("computer_use configuration must be a mapping")
    return block


def _same_config(left: Any, right: Any) -> bool:
    """Compare complete blocks without Python's True == 1 masking invalid intent."""
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
    selection = {**raw, **managed}
    if isinstance(raw.get("remote"), dict) and isinstance(managed.get("remote"), dict):
        selection["remote"] = {**raw["remote"], **managed["remote"]}
    # This architecture intentionally has no provider registry/selector. A config
    # copied from the provider-based alternative must not silently choose a host.
    # Direct remote selection requires an authored remote.enabled choice.
    if "provider" in selection:
        raise RuntimeError(
            "computer_use.provider is unsupported by direct CUA; remove it and "
            "use computer_use.remote.enabled and computer_use.remote.url",
        )
    # The general deep-merge ignores None over a mapping. Preserve an explicit
    # malformed remote block instead of converting it to disabled defaults (or
    # retaining a user's remote transport under a managed null).
    if "remote" in selection and not isinstance(selection["remote"], dict):
        raise RuntimeError("remote computer use configuration must be a mapping")
    if "remote" in selection and "enabled" not in selection["remote"]:
        raise RuntimeError("remote computer use configuration requires an explicit enabled: true or false")
    try:
        effective = config.load_config().get("computer_use", {})
        # An unrelated processing error can silently yield defaults or stale LKG.
        # Rebuild the COMPLETE block: checking only present leaves misses deleted
        # URLs/blocks resurrected from the cache. Canonicalization changes only
        # agent/model, so this is the normal loader's computer_use result.
        intended = config._deep_merge(
            cast(dict[str, Any], config._expand_env_vars(
                config._deep_merge(cast(dict[str, Any], config.DEFAULT_CONFIG.get("computer_use", {})), raw),
            )),
            cast(dict[str, Any], config._expand_env_vars(managed)),
        )
    except Exception as exc:
        raise RuntimeError("computer_use configuration could not be loaded") from exc
    if not _same_config(effective, intended):
        raise RuntimeError("computer_use configuration does not match current desktop intent; fix config.yaml")
    return effective
