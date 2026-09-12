"""Profile-scoped live catalogs and durable per-session MCP skill pins."""
from __future__ import annotations

import hashlib
import json
import threading
import re
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from hermes_constants import get_hermes_home
from tools.mcp_skills_fs import (
    _descriptor_safety_available, _require_descriptor_safety,
    secure_atomic_json, secure_file_lock, secure_read_json,
)
from tools.mcp_skills_protocol import SkillEntry, manifest_fingerprint, validate_skill_entry

_SCHEMA_VERSION = 2
_lock = threading.RLock()
_live: dict[tuple[str, str], dict[str, Any]] = {}
_snapshots: dict[tuple[str, str], tuple[dict[str, Any], ...]] = {}
_active: dict[tuple[str, str], dict[tuple[str, str, str], dict[str, Any]]] = {}
_consented: set[tuple[str, str, str, str, str]] = set()
_materialized: dict[tuple[str, str], list[dict[str, Any]]] = {}


def _home_key(home: Path | str | None = None) -> str:
    # Preserve the configured root path so guarded I/O can reject a symlinked
    # profile home instead of resolving it into an unintended scope.
    return str(Path(home or get_hermes_home()).expanduser().absolute())


def _profile_key(home: Path | str | None = None) -> str:
    return hashlib.sha256(_home_key(home).encode("utf-8")).hexdigest()


def _session_token(session_id: str) -> str:
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()


def _snapshot_rel(session_id: str) -> Path:
    return Path("cache") / "mcp-skills" / "session-snapshots" / f"{_session_token(session_id)}.json"


def _snapshot_lock_rel(session_id: str) -> Path:
    return Path("cache") / "mcp-skills" / "locks" / f"snapshot-{_session_token(session_id)}.lock"


def _active_rel(session_id: str) -> Path:
    return Path("cache") / "mcp-skills" / "active-origins" / f"{_session_token(session_id)}.json"


def _active_lock_rel(session_id: str) -> Path:
    return Path("cache") / "mcp-skills" / "locks" / f"active-{_session_token(session_id)}.lock"


def _materialized_rel(session_id: str) -> Path:
    return Path("cache") / "mcp-skills" / "materialized-targets" / f"{_session_token(session_id)}.json"


def _materialized_lock_rel(session_id: str) -> Path:
    return Path("cache") / "mcp-skills" / "locks" / f"materialized-{_session_token(session_id)}.lock"


def server_config_fingerprint(config: dict[str, Any]) -> str:
    """Hash all config identity without persisting credential values."""
    raw = json.dumps(config or {}, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _entry_record(
    server: str, config_fingerprint: str, entry: SkillEntry | dict[str, Any], *,
    connected: bool, get_verified: bool = False,
) -> dict[str, Any]:
    item = validate_skill_entry(entry)
    from tools.mcp_skills_scan import RemoteSkillSecurityError, scan_catalog_metadata
    try:
        metadata_scan = scan_catalog_metadata(
            item, server=server, config_fingerprint=config_fingerprint)
        metadata_allowed = True
    except RemoteSkillSecurityError as exc:
        metadata_scan = exc.attestation
        metadata_allowed = False
    return {
        "server": server,
        "config_fingerprint": config_fingerprint,
        "uri": item.uri,
        "frontmatter": json.loads(json.dumps(item.frontmatter)),
        "resources": [r.model_dump(mode="json") for r in item.resources if not isinstance(r, str)],
        "manifest_fingerprint": manifest_fingerprint(item),
        "metadata_scan_allowed": metadata_allowed,
        "attestation_bound": True,
        "overlap_safe": True,
        "metadata_allowed": metadata_allowed,
        "metadata_scan": metadata_scan,
        "get_verified": get_verified,
        "connected": connected,
    }


def _root_key(record: dict[str, Any]) -> tuple[str, str, str, tuple[str, ...]]:
    parsed = urlsplit(record["uri"])
    decoded = unquote(parsed.path)
    segments = tuple(part for part in decoded.removesuffix("/SKILL.md").split("/") if part)
    return parsed.scheme, parsed.netloc, parsed.query, segments


def _strict_ancestor(left: tuple[str, str, str, tuple[str, ...]],
                     right: tuple[str, str, str, tuple[str, ...]]) -> bool:
    return left[:3] == right[:3] and len(left[3]) < len(right[3]) and right[3][:len(left[3])] == left[3]


def _classify_overlaps(records: tuple[dict[str, Any], ...] | list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Validate the complete same-origin ancestry graph and resource claims."""
    rows = tuple(records)
    for row in rows:
        row["overlap_safe"] = True
        row.pop("ownership_ambiguous", None)
    by_origin: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_origin.setdefault((row["server"], row["config_fingerprint"]), []).append(row)
    for origin_rows in by_origin.values():
        roots = [_root_key(row) for row in origin_rows]
        resources = [{
            item["uri"]: (item["digest"], item["size"])
            for item in row["resources"]
        } for row in origin_rows]
        unsafe: set[int] = set()
        # Structural ancestry is authoritative even when an invalid ancestor
        # omitted every descendant resource and therefore shares no claim.
        for left in range(len(origin_rows)):
            for right in range(left + 1, len(origin_rows)):
                if roots[left] == roots[right]:
                    unsafe.update((left, right))
                    continue
                if _strict_ancestor(roots[left], roots[right]):
                    ancestor, descendant = left, right
                elif _strict_ancestor(roots[right], roots[left]):
                    ancestor, descendant = right, left
                else:
                    ancestor = descendant = -1
                if ancestor >= 0:
                    if any(resources[ancestor].get(uri) != value
                           for uri, value in resources[descendant].items()):
                        unsafe.update((left, right))
                    continue
                if set(resources[left]) & set(resources[right]):
                    unsafe.update((left, right))

        claims: dict[str, list[int]] = {}
        for index, row in enumerate(origin_rows):
            for resource in row["resources"]:
                claims.setdefault(resource["uri"], []).append(index)
        for uri, owners in claims.items():
            if len(owners) < 2:
                continue
            tuples = {resources[index][uri] for index in owners}
            if len(tuples) != 1:
                unsafe.update(owners)
        for index, row in enumerate(origin_rows):
            row["overlap_safe"] = index not in unsafe
            if index in unsafe:
                row["ownership_ambiguous"] = True
    for row in rows:
        row["metadata_allowed"] = bool(
            row.get("metadata_scan_allowed") is True
            and row.get("attestation_bound") is True
            and row.get("overlap_safe") is True)
    return rows


def publish_live_catalog(home: Path | str, server: str, config_fingerprint: str,
                         entries: list[SkillEntry]) -> None:
    if not isinstance(server, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", server) is None:
        raise ValueError("MCP server label is not a safe qualified-name/path component")
    records_list = [_entry_record(server, config_fingerprint, entry, connected=True) for entry in entries]
    records = _classify_overlaps(records_list)
    with _lock:
        _live[(_home_key(home), server)] = {"entries": records}


def drop_live_catalog(home: Path | str, server: str) -> None:
    with _lock:
        _live.pop((_home_key(home), server), None)


def _validate_record(raw: Any, *, connected: bool) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("snapshot entry is not an object")
    entry = validate_skill_entry({"uri": raw.get("uri"), "frontmatter": raw.get("frontmatter"),
                                  "resources": raw.get("resources")})
    record = _entry_record(str(raw.get("server") or ""), str(raw.get("config_fingerprint") or ""),
                           entry, connected=connected, get_verified=bool(raw.get("get_verified")))
    if not record["server"] or not record["config_fingerprint"]:
        raise ValueError("snapshot entry lacks origin identity")
    if raw.get("manifest_fingerprint") != record["manifest_fingerprint"]:
        raise ValueError("snapshot manifest fingerprint mismatch")
    prior_scan = raw.get("metadata_scan")
    binding_keys = ("scanner_version", "source", "content_hash", "scope")
    if not isinstance(prior_scan, dict) or any(
            prior_scan.get(key) != record["metadata_scan"].get(key) for key in binding_keys):
        # The record remains privately held for exact resource ownership, but
        # a stale/unbound attestation can never restore public metadata.
        record["attestation_bound"] = False
        record["metadata_allowed"] = False
    return record


def _load_snapshot_locked(home: Path | str, session_id: str) -> tuple[dict[str, Any], ...] | None:
    try:
        raw = secure_read_json(home, _snapshot_rel(session_id))
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return ()
    try:
        if raw.get("schema_version") != _SCHEMA_VERSION or raw.get("profile_key") != _profile_key(home):
            return ()
        rows = raw.get("entries")
        if not isinstance(rows, list):
            return ()
        return _classify_overlaps(tuple(_validate_record(row, connected=False) for row in rows))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return ()


def _pin_session_all(home: Path | str | None, session_id: str | None) -> tuple[dict[str, Any], ...]:
    """Hold visible and quarantined ownership rows; never return directly to model-facing code."""
    if not session_id:
        return ()
    home_key = _home_key(home)
    key = (home_key, session_id)
    if not _descriptor_safety_available():
        with _lock:
            if key in _snapshots:
                return tuple(json.loads(json.dumps(row)) for row in _snapshots[key])
            if not any(live_home == home_key for live_home, _server in _live):
                return ()
        _require_descriptor_safety()
    with _lock, secure_file_lock(home_key, _snapshot_lock_rel(session_id)):
        # Disk is authoritative across processes.  Reload it under the kernel
        # lock before every security-sensitive snapshot read.
        held = _load_snapshot_locked(home_key, session_id)
        if held is None:
            held = tuple(row for (live_home, _), cat in sorted(_live.items()) if live_home == home_key
                         for row in cat["entries"])
            payload = {"schema_version": _SCHEMA_VERSION, "profile_key": _profile_key(home_key),
                       "session_id_hash": _session_token(session_id), "entries": list(held)}
            secure_atomic_json(home_key, _snapshot_rel(session_id), payload, mode=0o600)
        _snapshots[key] = held
        return tuple(json.loads(json.dumps(row)) for row in held)


def _write_snapshot_locked(home: str, session_id: str, rows: tuple[dict[str, Any], ...]) -> None:
    secure_atomic_json(home, _snapshot_rel(session_id), {
        "schema_version": _SCHEMA_VERSION, "profile_key": _profile_key(home),
        "session_id_hash": _session_token(session_id), "entries": list(rows)}, mode=0o600)


def _merge_snapshot_locked(home: str, session_id: str, additions: list[dict[str, Any]], *,
                           seed_live: bool = True, allow_new: bool = True) -> tuple[dict[str, Any], ...]:
    """Merge immutable manifest identities while the snapshot file lock is held."""
    current = _load_snapshot_locked(home, session_id)
    if current is None:
        current = tuple(row for (live_home, _), cat in sorted(_live.items())
                        if seed_live and live_home == home for row in cat["entries"])
    rows = [json.loads(json.dumps(row)) for row in current]
    by_uri = {(row["server"], row["uri"]): row for row in rows}
    for addition in additions:
        key = (addition["server"], addition["uri"])
        existing = by_uri.get(key)
        if existing is not None:
            if existing["manifest_fingerprint"] != addition["manifest_fingerprint"]:
                raise ValueError(
                    "remote skill manifest changed for this session-pinned server and URI; start a new session")
            if addition.get("get_verified"):
                existing["get_verified"] = True
            addition.update(json.loads(json.dumps(existing)))
            continue
        if not allow_new:
            raise KeyError("remote skill is not present in the pinned session manifest")
        copied = json.loads(json.dumps(addition))
        rows.append(copied)
        by_uri[key] = copied
    held = _classify_overlaps(rows)
    _snapshots[(home, session_id)] = held
    _write_snapshot_locked(home, session_id, held)
    return held


def register_lazy_entry(home: Path | str | None, session_id: str | None, server: str,
                        config_fingerprint: str, entry: SkillEntry | dict[str, Any]) -> dict[str, Any]:
    """Atomically append a point-fetched entry without replacing a held identity."""
    if not session_id:
        raise ValueError("URI-only remote skill registration requires a session id")
    home_key = _home_key(home)
    record = _entry_record(server, config_fingerprint, entry, connected=True, get_verified=True)
    with _lock, secure_file_lock(home_key, _snapshot_lock_rel(session_id)):
        held = _merge_snapshot_locked(home_key, session_id, [record])
        match = next(row for row in held if row["server"] == server and row["uri"] == record["uri"])
        record.update(json.loads(json.dumps(match)))
    return record


def pin_session(home: Path | str | None, session_id: str | None) -> tuple[dict[str, Any], ...]:
    """Return only Skills-Guard-approved metadata for a profile/session."""
    return tuple(row for row in _pin_session_all(home, session_id) if row.get("metadata_allowed") is True)


def resolve_remote_skill(identifier: str, home: Path | str | None, session_id: str | None) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(identifier, str) or not identifier.startswith("mcp:"):
        return None, "Remote skill identifiers must start with 'mcp:'."
    try:
        _, server, target = identifier.split(":", 2)
    except ValueError:
        return None, "Use mcp:<server-label>:<skill-uri-or-name>."
    if not server or not target:
        return None, "Use mcp:<server-label>:<skill-uri-or-name>."
    rows = [r for r in pin_session(home, session_id) if r["server"] == server]
    exact = [r for r in rows if r["uri"] == target]
    if exact:
        return exact[0], None
    named = [r for r in rows if r["frontmatter"].get("name") == target]
    if len(named) == 1:
        return named[0], None
    if len(named) > 1:
        matches = [qualified_name(r) for r in named]
        return None, f"Remote skill name {target!r} is ambiguous; use one of: {', '.join(matches)}"
    return None, f"Remote skill {identifier!r} is not in this session's pinned catalog."


def qualified_name(record: dict[str, Any]) -> str:
    return f"mcp:{record['server']}:{record['uri']}"


def relative_resource_path(record: dict[str, Any], resource_uri: str) -> str:
    skill_path = unquote(urlsplit(record["uri"]).path)
    resource_path = unquote(urlsplit(resource_uri).path)
    root = skill_path.rsplit("/", 1)[0] + "/"
    if not resource_path.startswith(root):
        raise ValueError("resource escapes held skill root")
    return resource_path[len(root):]


def resource_for_path(record: dict[str, Any], file_path: str | None) -> dict[str, Any] | None:
    wanted = "SKILL.md" if not file_path else str(file_path).replace("\\", "/").lstrip("/")
    matches = [r for r in record["resources"] if relative_resource_path(record, r["uri"]) == wanted]
    return matches[0] if len(matches) == 1 else None


def list_remote_skills(home: Path | str | None, session_id: str | None) -> list[dict[str, Any]]:
    return [{"name": r["frontmatter"]["name"], "description": r["frontmatter"]["description"],
             "category": f"mcp:{r['server']}", "source": "mcp", "server": r["server"],
             "uri": r["uri"], "qualified_name": qualified_name(r),
             "connected": bool(r.get("connected"))} for r in pin_session(home, session_id)]


def resolve_catalog_resource(server: str | None, uri: str, home: Path | str | None,
                             session_id: str | None) -> dict[str, Any] | None:
    """Owning held skill for an exact manifested resource, if any.

    This is intentionally an exact `(configured server label, URI)` lookup. It
    lets the generic generated `read_resource` route recognize catalog-owned
    skill bytes without treating arbitrary MCP resources as tainted.
    """
    if not server or not uri or not session_id:
        return None
    matches = [
        record for record in _pin_session_all(home, session_id)
        if record["server"] == server
        and any(resource.get("uri") == uri for resource in record["resources"])
    ]
    if not matches:
        return None
    if len(matches) > 1:
        if (all(record.get("overlap_safe") is True for record in matches)
                and all(record.get("metadata_allowed") is True for record in matches)):
            return max(matches, key=lambda record: len(_root_key(record)[3]))
        blocked = dict(matches[0])
        blocked["metadata_allowed"] = False
        blocked["ownership_ambiguous"] = True
        return blocked
    return matches[0]


def is_active(record: dict[str, Any], home: Path | str | None, session_id: str | None) -> bool:
    if not session_id:
        return False
    key = (record["server"], record["uri"], record["manifest_fingerprint"])
    home_key = _home_key(home)
    with _lock, secure_file_lock(home_key, _active_lock_rel(session_id)):
        _active.pop((home_key, session_id), None)
        return key in _load_active_locked(home_key, session_id)


def _load_active_locked(home: str, session_id: str) -> dict[tuple[str, str, str], dict[str, Any]]:
    key = (home, session_id)
    if key in _active:
        return _active[key]
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    try:
        raw = secure_read_json(home, _active_rel(session_id))
        if raw.get("schema_version") == _SCHEMA_VERSION and raw.get("profile_key") == _profile_key(home):
            for item in raw.get("active", []):
                record = _validate_record(item, connected=False)
                rows[(record["server"], record["uri"], record["manifest_fingerprint"])] = record
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    _active[key] = rows
    return rows


def _write_active_locked(home: str, session_id: str,
                         active: dict[tuple[str, str, str], dict[str, Any]]) -> None:
    secure_atomic_json(home, _active_rel(session_id), {
        "schema_version": _SCHEMA_VERSION, "profile_key": _profile_key(home),
        "session_id_hash": _session_token(session_id), "active": list(active.values())}, mode=0o600)


def mark_active(record: dict[str, Any], home: Path | str | None, session_id: str | None) -> None:
    if not session_id:
        raise ValueError("remote skill activation requires a session id")
    home_key = _home_key(home)
    with _lock, secure_file_lock(home_key, _active_lock_rel(session_id)):
        # A sibling process may have committed after our in-memory publication.
        # Reload under the kernel lock before every load/merge/write transaction.
        _active.pop((home_key, session_id), None)
        active = _load_active_locked(home_key, session_id)
        active[(record["server"], record["uri"], record["manifest_fingerprint"])] = dict(record)
        _write_active_locked(home_key, session_id, active)


def mark_get_verified(record: dict[str, Any], home: Path | str | None, session_id: str | None) -> None:
    """Persist successful skills/get verification in the pinned session snapshot."""
    if not session_id:
        raise ValueError("skills/get verification requires a session id")
    home_key = _home_key(home)
    update = json.loads(json.dumps(record))
    update["get_verified"] = True
    with _lock, secure_file_lock(home_key, _snapshot_lock_rel(session_id)):
        _merge_snapshot_locked(home_key, session_id, [update], allow_new=False)
    record["get_verified"] = True


def active_skills(home: Path | str | None, session_id: str | None) -> list[dict[str, Any]]:
    if not session_id:
        return []
    home_key = _home_key(home)
    key = (home_key, session_id)
    if not _descriptor_safety_available():
        with _lock:
            if key not in _active:
                return []
        _require_descriptor_safety()
    with _lock, secure_file_lock(home_key, _active_lock_rel(session_id)):
        _active.pop((home_key, session_id), None)
        return [json.loads(json.dumps(v)) for v in _load_active_locked(home_key, session_id).values()]


def has_execution_consent(record: dict[str, Any], home: Path | str | None, session_id: str) -> bool:
    with _lock:
        return (_home_key(home), session_id, record["server"], record["uri"], record["manifest_fingerprint"]) in _consented


def record_execution_consent(record: dict[str, Any], home: Path | str | None, session_id: str) -> None:
    with _lock:
        _consented.add((_home_key(home), session_id, record["server"], record["uri"], record["manifest_fingerprint"]))


def _load_materialized_locked(home: str, session_id: str) -> list[dict[str, Any]]:
    key = (home, session_id)
    if key in _materialized:
        return _materialized[key]
    rows: list[dict[str, Any]] = []
    try:
        raw = secure_read_json(home, _materialized_rel(session_id))
    except FileNotFoundError:
        raw = None
    if raw is not None:
        if (not isinstance(raw, dict) or raw.get("schema_version") != _SCHEMA_VERSION
                or raw.get("profile_key") != _profile_key(home)
                or not isinstance(raw.get("targets"), list)):
            raise ValueError("materialized target registry is invalid")
        for item in raw["targets"]:
            if not (isinstance(item, dict) and isinstance(item.get("workspace"), str)
                    and isinstance(item.get("target_rel"), str)
                    and isinstance(item.get("sidecar_rel"), str)
                    and isinstance(item.get("resource_uri"), str)):
                raise ValueError("materialized target registry row is invalid")
            rows.append(dict(item))
    _materialized[key] = rows
    return rows


def _write_materialized_locked(home: str, session_id: str, rows: list[dict[str, Any]]) -> None:
    secure_atomic_json(home, _materialized_rel(session_id), {
        "schema_version": _SCHEMA_VERSION, "profile_key": _profile_key(home),
        "session_id_hash": _session_token(session_id), "targets": rows}, mode=0o600)


def record_materialization(record: dict[str, Any], resource: dict[str, Any], *,
                           home: Path | str | None, session_id: str,
                           workspace: Path, target_rel: Path, sidecar_rel: Path,
                           is_text: bool) -> None:
    """Persist only an exact provenance-owned target; never crawl the workspace."""
    home_key = _home_key(home)
    row = {
        "server": record["server"], "skill_uri": record["uri"],
        "manifest_fingerprint": record["manifest_fingerprint"],
        "resource_uri": resource["uri"], "workspace": str(workspace.absolute()),
        "target_rel": target_rel.as_posix(), "sidecar_rel": sidecar_rel.as_posix(),
        "is_text": bool(is_text),
    }
    with _lock, secure_file_lock(home_key, _materialized_lock_rel(session_id)):
        _materialized.pop((home_key, session_id), None)
        rows = _load_materialized_locked(home_key, session_id)
        identity = (row["workspace"], row["target_rel"], row["sidecar_rel"])
        rows[:] = [item for item in rows if (
            item["workspace"], item["target_rel"], item["sidecar_rel"]) != identity]
        rows.append(row)
        _write_materialized_locked(home_key, session_id, rows)


def scan_materialized_targets(record: dict[str, Any], home: Path | str | None,
                              session_id: str) -> None:
    """Re-scan current bytes for exact persisted text targets of one active manifest."""
    from tools.mcp_skills_fs import secure_read_bytes, secure_read_json, validate_managed_root
    from tools.mcp_skills_scan import RemoteSkillSecurityError, scan_resource_bytes
    home_key = _home_key(home)
    with _lock, secure_file_lock(home_key, _materialized_lock_rel(session_id)):
        _materialized.pop((home_key, session_id), None)
        rows = list(_load_materialized_locked(home_key, session_id))
    resources = {item["uri"]: item for item in record["resources"]}
    for item in rows:
        if (item.get("server"), item.get("skill_uri"), item.get("manifest_fingerprint")) != (
                record["server"], record["uri"], record["manifest_fingerprint"]):
            continue
        resource = resources.get(item["resource_uri"])
        if resource is None:
            raise RemoteSkillSecurityError("materialized resource")
        workspace = validate_managed_root(Path(item["workspace"]).expanduser())
        provenance = secure_read_json(workspace, item["sidecar_rel"])
        expected = {
            "server": record["server"], "config_fingerprint": record["config_fingerprint"],
            "skill_uri": record["uri"], "resource_uri": resource["uri"],
            "digest": resource["digest"], "size": resource["size"],
            "manifest_fingerprint": record["manifest_fingerprint"],
        }
        if any(provenance.get(key) != value for key, value in expected.items()):
            raise RemoteSkillSecurityError("materialized resource")
        current = secure_read_bytes(workspace, item["target_rel"])
        scan_resource_bytes(current, record=record, resource=resource,
                            is_text=bool(item.get("is_text")))


def continue_session(home: Path | str | None, parent_session_id: str, child_session_id: str) -> None:
    """Atomically carry only an explicit compression continuation's pinned security state."""
    if not parent_session_id or not child_session_id or parent_session_id == child_session_id:
        return
    home_key = _home_key(home)
    parent_key = (home_key, parent_session_id)
    child_key = (home_key, child_session_id)
    if not _descriptor_safety_available():
        with _lock:
            has_runtime_state = (
                parent_key in _snapshots
                or parent_key in _active
                or any(origin_home == home_key and sid == parent_session_id
                       for origin_home, sid, *_rest in _consented)
            )
        if not has_runtime_state:
            return
        _require_descriptor_safety()
    lock_ids = sorted({parent_session_id, child_session_id}, key=_session_token)
    lock_factories = (_snapshot_lock_rel, _active_lock_rel, _materialized_lock_rel)
    with _lock, ExitStack() as stack:
        # One total order for every continuation transaction: session token,
        # then snapshot -> active -> materialized.
        for lock_id in lock_ids:
            for lock_factory in lock_factories:
                stack.enter_context(secure_file_lock(home_key, lock_factory(lock_id)))
        parent_snapshot = _load_snapshot_locked(home_key, parent_session_id)
        if parent_snapshot is None:
            parent_snapshot = _snapshots.get(parent_key)
        if parent_snapshot is not None:
            _merge_snapshot_locked(
                home_key, child_session_id,
                [json.loads(json.dumps(row)) for row in parent_snapshot], seed_live=False)
        _active.pop(parent_key, None)
        _active.pop(child_key, None)
        parent_active = _load_active_locked(home_key, parent_session_id)
        child_active = _load_active_locked(home_key, child_session_id)
        child_active.update({key: json.loads(json.dumps(value)) for key, value in parent_active.items()})
        if child_active:
            _write_active_locked(home_key, child_session_id, child_active)
        inherited = {
            (home_key, child_session_id, server, uri, fingerprint)
            for consent_home, consent_sid, server, uri, fingerprint in _consented
            if consent_home == home_key and consent_sid == parent_session_id
        }
        _consented.update(inherited)
        _materialized.pop(parent_key, None)
        _materialized.pop(child_key, None)
        parent_targets = _load_materialized_locked(home_key, parent_session_id)
        child_targets = _load_materialized_locked(home_key, child_session_id)
        existing = {(row["workspace"], row["target_rel"], row["sidecar_rel"]) for row in child_targets}
        child_targets.extend(json.loads(json.dumps(row)) for row in parent_targets
                             if (row["workspace"], row["target_rel"], row["sidecar_rel"]) not in existing)
        if child_targets:
            _write_materialized_locked(home_key, child_session_id, child_targets)


def clear_runtime_state() -> None:
    """Test/reset helper; durable snapshots remain on disk."""
    with _lock:
        _live.clear()
        _snapshots.clear()
        _active.clear()
        _consented.clear()
        _materialized.clear()
