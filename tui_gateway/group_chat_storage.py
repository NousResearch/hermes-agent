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
    if "pending_ui_meta" in state:
        _validate_pending(state["pending_ui_meta"])
    return state


def _validate_pending(pending: dict) -> None:
    if not isinstance(pending, dict) or not pending:
        raise ValueError("Invalid UI metadata journal")
    for key, change in pending.items():
        if not isinstance(key, str) or not isinstance(change, dict) or set(change) != {"before", "after"}:
            raise ValueError("Invalid UI metadata journal entry")
        for entry in change.values():
            if not isinstance(entry, dict) or set(entry) - {"value", "revision"}:
                raise ValueError("Invalid UI metadata journal value")
        after = change["after"]
        if "revision" in after and (type(after["revision"]) is not int or after["revision"] < 0):
            raise ValueError("Invalid UI metadata journal revision")
        if key == GROUP_CHAT_KEY and after:
            raise ValueError("Group history cannot be materialized into profile metadata")


def _namespace_entry(metadata: dict, key: str) -> dict:
    entry = {}
    for field, name in (("ui_meta", "value"), ("_ui_meta_revisions", "revision")):
        values = metadata.get(field)
        if isinstance(values, dict) and key in values:
            entry[name] = values[key]
    return entry


def _metadata_delta(original: dict, cleaned: dict) -> dict:
    keys = set()
    for metadata in (original, cleaned):
        for field in ("ui_meta", "_ui_meta_revisions"):
            values = metadata.get(field)
            if isinstance(values, dict):
                keys.update(values)
    changes = {}
    for key in keys:
        before, after = _namespace_entry(original, key), _namespace_entry(cleaned, key)
        if before != after:
            changes[key] = {"before": before, "after": after}
    return changes


def _apply_pending(metadata: dict, pending: dict) -> dict:
    result = deepcopy(metadata)
    for key, change in pending.items():
        current = _namespace_entry(metadata, key)
        if current != change["before"] and current != change["after"]:
            raise ValueError("UI metadata changed outside the pending transaction")
        for field, name in (("ui_meta", "value"), ("_ui_meta_revisions", "revision")):
            values = result.get(field)
            values = values if isinstance(values, dict) else {}
            if name in change["after"]:
                values[key] = deepcopy(change["after"][name])
            else:
                values.pop(key, None)
            if values:
                result[field] = values
            else:
                result.pop(field, None)
    return result


def _finish_pending(profile_dir: Path, state: dict) -> dict:
    """Replay under the profile lock; never overwrite unrelated metadata edits."""
    from utils import atomic_yaml_write

    if "pending_ui_meta" not in state:
        return state
    path = profile_dir / "profile.yaml"
    original = _read_mapping(path)
    cleaned = _apply_pending(original, state["pending_ui_meta"])
    if cleaned != original:
        atomic_yaml_write(path, cleaned, sort_keys=False)
    finished = {key: value for key, value in state.items() if key != "pending_ui_meta"}
    atomic_yaml_write(profile_dir / GROUP_CHAT_STATE, finished, create_mode=0o600)
    return finished


class GroupChatReadUnavailable(Exception):
    """A transient read failure with safe metadata that must not enter a file-signature cache."""

    def __init__(self, metadata: dict):
        super().__init__("Profile UI metadata snapshot unavailable")
        self.metadata = metadata


def read_profile_ui_meta(profile_dir: Path, load_metadata) -> dict:
    """Read a coherent projection without creating files or taking the writer lock.

    An atomic state replacement publishes every cross-file update. Seeing the same
    state before and after loading metadata means no such commit/retirement crossed
    that read. Bounded retries avoid blocking the roster behind a busy writer.
    """
    try:
        for _ in range(3):
            before = _read_state(profile_dir)
            metadata = load_metadata()
            after = _read_state(profile_dir)
            if before == after:
                return _project_group_state(metadata if isinstance(metadata, dict) else {}, after)
    except (OSError, ValueError, yaml.YAMLError):
        pass  # Return safe metadata, but never memoize a transiently unavailable snapshot.
    raise GroupChatReadUnavailable(_unavailable_group_state(profile_dir, load_metadata())) from None


def _unavailable_group_state(profile_dir: Path, metadata: dict) -> dict:
    # Parser exceptions can quote conversation text. Log only the storage location.
    logging.getLogger(__name__).warning("profiles.list: unreadable group state in %s", profile_dir)
    result = deepcopy(metadata) if isinstance(metadata, dict) else {}
    for key in ("ui_meta", "_ui_meta_revisions"):
        if isinstance(result.get(key), dict):
            result[key].pop(GROUP_CHAT_KEY, None)
    return result


def with_group_chat_state(profile_dir: Path, metadata: dict, *, strict: bool = True) -> dict:
    """Project separate runtime state into the existing profiles.list/configure wire shape.

    The higher revision wins if an interrupted migration left both copies. Equal revisions
    prefer the separate file, including a deletion tombstone, so old logs cannot resurrect.
    A legacy process can still advance profile.yaml until all writers have restarted.
    Roster reads may omit unavailable group state; mutations must fail closed.
    """
    try:
        return _project_group_state(metadata, _read_state(profile_dir))
    except (OSError, ValueError, yaml.YAMLError):
        if strict:
            raise
        return _unavailable_group_state(profile_dir, metadata)


def _project_group_state(metadata: dict, state: dict) -> dict:
    result = deepcopy(metadata)
    if "pending_ui_meta" in state:
        result = _apply_pending(result, state["pending_ui_meta"])
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
    """Commit a cross-file update with its recovery delta in the atomic state write.

    Once that write succeeds the entire UI update is committed, even if materializing
    profile.yaml fails. Roster reads project the durable delta until a later writer or
    migration can replay it. Single-file updates need no journal.
    """
    from utils import atomic_yaml_write

    previous = _finish_pending(profile_dir, _read_state(profile_dir))
    path = profile_dir / "profile.yaml"
    original = _read_mapping(path)
    # Only UI fields belong to this transaction; retain unrelated metadata edits.
    cleaned = deepcopy(original)
    for field in ("ui_meta", "_ui_meta_revisions"):
        cleaned[field] = deepcopy(metadata.get(field, {}))
    ui_meta = cleaned.get("ui_meta")
    ui_meta = ui_meta if isinstance(ui_meta, dict) else {}
    revisions = cleaned.get("_ui_meta_revisions")
    revisions = revisions if isinstance(revisions, dict) else {}
    state = previous
    if GROUP_CHAT_KEY in ui_meta or GROUP_CHAT_KEY in revisions:
        state = {"revision": revisions.pop(GROUP_CHAT_KEY, 0),
                 "snapshot": ui_meta.pop(GROUP_CHAT_KEY, None)}
    for key, values in (("ui_meta", ui_meta), ("_ui_meta_revisions", revisions)):
        if values:
            cleaned[key] = values
        else:
            cleaned.pop(key, None)
    if state == previous:
        if cleaned != original:
            atomic_yaml_write(path, cleaned, sort_keys=False)
        return
    pending = _metadata_delta(original, cleaned)
    if pending:
        _validate_pending(pending)
        state["pending_ui_meta"] = pending
    atomic_yaml_write(profile_dir / GROUP_CHAT_STATE, state, create_mode=0o600)
    if pending:
        try:
            _finish_pending(profile_dir, state)
        except (OSError, ValueError, yaml.YAMLError):
            # The complete update is already durable. Never misreport it as unapplied.
            logging.getLogger(__name__).warning("UI metadata materialization deferred in %s", profile_dir)


def migrate_group_chat_state(profile_dir: Path) -> None:
    """Move logs without changing revisions, after all legacy writers have restarted."""
    with profile_ui_meta_lock(profile_dir):
        metadata = _read_mapping(profile_dir / "profile.yaml")
        write_profile_ui_meta(profile_dir, with_group_chat_state(profile_dir, metadata))
