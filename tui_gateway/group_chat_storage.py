# ABOUTME: Store the group chat sync snapshot and revision outside profile.yaml.
# ABOUTME: Keep the existing UI metadata wire format and migrate legacy history without dropping it.

from copy import deepcopy
import logging
from pathlib import Path
import threading

import hermes_yaml as yaml


GROUP_CHAT_KEY = "hermes-bots-groups"
GROUP_CHAT_STATE = Path("state/group-chat.yaml")


def profile_ui_meta_lock(profile_dir: Path):
    """Serialize runtime writes and explicit migration across backend processes."""
    from hermes_cli.auth import _file_lock
    from hermes_constants import mkdir_under_hermes_home

    lock_path = profile_dir / "state" / "profile-ui-meta.lock"
    mkdir_under_hermes_home(lock_path.parent)
    return _file_lock(lock_path, threading.local(), 10.0, "Timed out waiting for profile UI metadata")


def _read_mapping(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path.name}")
    return data


def _read_state(profile_dir: Path) -> dict:
    try:
        state = yaml.safe_load((profile_dir / GROUP_CHAT_STATE).read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {}
    if (not isinstance(state, dict) or type(state.get("revision")) is not int
            or state["revision"] < 0 or "snapshot" not in state):
        raise ValueError("Invalid group chat state")
    return state


def with_group_chat_state(profile_dir: Path, metadata: dict, *, strict: bool = True) -> dict:
    """Project separate runtime state into the existing profiles.list/configure wire shape.

    The higher revision wins if an interrupted migration left both copies. Equal revisions
    prefer the separate file, including a deletion tombstone, so old logs cannot resurrect.
    A legacy process can still advance profile.yaml until all writers have restarted.
    Roster reads may omit unavailable group state; mutations must fail closed.
    """
    result = deepcopy(metadata)
    try:
        state = _read_state(profile_dir)
    except (OSError, ValueError, yaml.YAMLError):
        if strict:
            raise
        # Parser exceptions can quote conversation text. Log only the storage location.
        logging.getLogger(__name__).warning("profiles.list: unreadable group state in %s", profile_dir)
        for key in ("ui_meta", "_ui_meta_revisions"):
            if isinstance(result.get(key), dict):
                result[key].pop(GROUP_CHAT_KEY, None)
        return result
    if not state:
        return result
    revisions = result.get("_ui_meta_revisions")
    revisions = revisions if isinstance(revisions, dict) else {}
    legacy_revision = revisions.get(GROUP_CHAT_KEY, 0)
    legacy_revision = legacy_revision if type(legacy_revision) is int else 0
    if legacy_revision > state["revision"]:
        return result
    ui_meta = result.get("ui_meta")
    ui_meta = ui_meta if isinstance(ui_meta, dict) else {}
    if state["snapshot"] is None:
        ui_meta.pop(GROUP_CHAT_KEY, None)
    else:
        ui_meta[GROUP_CHAT_KEY] = state["snapshot"]
    revisions[GROUP_CHAT_KEY] = state["revision"]
    result["ui_meta"] = ui_meta
    result["_ui_meta_revisions"] = revisions
    return result


def write_profile_ui_meta(profile_dir: Path, metadata: dict) -> None:
    """Persist the snapshot first, then remove it and its revision from profile.yaml.

    A failed snapshot write leaves legacy history intact. If profile cleanup fails, the
    next read still sees the committed snapshot/revision and the next write retries cleanup.
    Group-only updates leave an already migrated profile.yaml byte-for-byte unchanged.
    """
    from utils import atomic_yaml_write

    path = profile_dir / "profile.yaml"
    original = _read_mapping(path)
    cleaned = deepcopy(metadata)
    ui_meta = cleaned.get("ui_meta")
    ui_meta = ui_meta if isinstance(ui_meta, dict) else {}
    revisions = cleaned.get("_ui_meta_revisions")
    revisions = revisions if isinstance(revisions, dict) else {}
    if GROUP_CHAT_KEY in ui_meta or GROUP_CHAT_KEY in revisions:
        state = {"revision": revisions.pop(GROUP_CHAT_KEY, 0),
                 "snapshot": ui_meta.pop(GROUP_CHAT_KEY, None)}
        if state != _read_state(profile_dir):
            atomic_yaml_write(profile_dir / GROUP_CHAT_STATE, state, create_mode=0o600)
    for key, values in (("ui_meta", ui_meta), ("_ui_meta_revisions", revisions)):
        if values:
            cleaned[key] = values
        else:
            cleaned.pop(key, None)
    if cleaned != original:
        atomic_yaml_write(path, cleaned, sort_keys=False)


def migrate_group_chat_state(profile_dir: Path) -> None:
    """Move logs without changing revisions, after all legacy writers have restarted."""
    with profile_ui_meta_lock(profile_dir):
        metadata = _read_mapping(profile_dir / "profile.yaml")
        write_profile_ui_meta(profile_dir, with_group_chat_state(profile_dir, metadata))
