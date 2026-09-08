"""Fail-closed desktop selection, including pre-provider remote configuration.

Presence matters: defaults must not turn an absent provider into explicit local,
or an omitted remote.enabled into an explicit veto. Read raw files only to
validate them and retain that provenance; values still use the normal expanded,
managed-scope-aware config loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


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

