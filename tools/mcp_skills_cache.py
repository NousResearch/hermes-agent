"""Verified lazy cache and editable materialization for manifested MCP skills."""
from __future__ import annotations

import hashlib
import os
import time
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Any

from agent.runtime_cwd import resolve_agent_cwd
from hermes_constants import get_hermes_home
from tools.mcp_skills_fs import (
    secure_atomic_bytes, secure_atomic_json, secure_read_bytes, secure_read_json, validate_managed_root,
)
from tools.mcp_skills_protocol import decode_read_resource_result, get_skill, manifest_fingerprint, validate_skill_entry
from tools.mcp_skills_registry import mark_get_verified, relative_resource_path
from tools.mcp_skills_scan import scan_resource_bytes


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _cache_rel_dir(record: dict[str, Any], resource: dict[str, Any]) -> Path:
    server_key = hashlib.sha256((record["server"] + ":" + record["config_fingerprint"]).encode()).hexdigest()
    uri_key = hashlib.sha256(resource["uri"].encode()).hexdigest()
    digest = resource["digest"].removeprefix("sha256:")
    return Path("cache") / "mcp-skills" / "content" / server_key / uri_key / digest


def _cache_dir(record: dict[str, Any], resource: dict[str, Any], home: Path | str | None = None) -> Path:
    return Path(home or get_hermes_home()) / _cache_rel_dir(record, resource)


def _verify(raw: bytes, resource: dict[str, Any]) -> None:
    if len(raw) != resource["size"]:
        raise ValueError(f"resource size mismatch: expected {resource['size']}, received {len(raw)}")
    actual = _sha256(raw)
    expected = resource["digest"].removeprefix("sha256:")
    if actual != expected:
        raise ValueError(f"resource digest mismatch: expected sha256:{expected}, received sha256:{actual}")


def _read_cached(record: dict[str, Any], resource: dict[str, Any], home=None) -> tuple[bytes, dict[str, Any]] | None:
    root = Path(home or get_hermes_home())
    relative = _cache_rel_dir(record, resource)
    try:
        raw = secure_read_bytes(root, relative / "content")
        meta = secure_read_json(root, relative / "metadata.json")
        _verify(raw, resource)
        expected = {
            "server": record["server"], "config_fingerprint": record["config_fingerprint"],
            "skill_uri": record["uri"], "uri": resource["uri"], "digest": resource["digest"],
            "size": resource["size"],
        }
        if any(meta.get(key) != value for key, value in expected.items()):
            raise ValueError("verified cache metadata mismatch")
        return raw, meta
    except FileNotFoundError:
        return None


async def _read_on_session(server: Any, uri: str):
    async with server._rpc_lock:
        return await server.session.read_resource(uri)


async def _get_on_session(server: Any, uri: str):
    async with server._rpc_lock:
        return await get_skill(server.session, uri)


def _ensure_get_verified(record: dict[str, Any], home=None, session_id: str | None = None) -> None:
    if record.get("get_verified"):
        return
    from tools import mcp_tool_discovery as discovery
    from tools import mcp_tool_loop
    server = discovery._get_connected_server_for_call(record["server"])
    if server is None or server.session is None:
        raise RuntimeError(
            f"MCP server {record['server']!r} is not connected and skills/get identity has not been verified")
    _require_live_origin(server, record, home)
    result = mcp_tool_loop._run_on_mcp_loop(
        lambda: _get_on_session(server, record["uri"]), timeout=float(getattr(server, "tool_timeout", 300)))
    current = validate_skill_entry(result)
    if manifest_fingerprint(current) != record["manifest_fingerprint"]:
        raise ValueError("skills/get manifest does not match the session-pinned skills/list manifest")
    mark_get_verified(record, home, session_id)


def _require_live_origin(server: Any, record: dict[str, Any], home=None) -> None:
    requested_home = str(validate_managed_root(Path(home or get_hermes_home()).expanduser()).absolute())
    live_home = str(validate_managed_root(
        Path(getattr(server, "_skills_home", "") or ".").expanduser()).absolute())
    if live_home != requested_home or getattr(server, "_skills_config_fingerprint", "") != record["config_fingerprint"]:
        raise RuntimeError(
            "MCP skill origin no longer matches the connected profile/configuration; start a new session after reconnect")


def _fetch(record: dict[str, Any], resource: dict[str, Any], home=None) -> tuple[bytes, str, bool]:
    from tools import mcp_tool_discovery as discovery
    from tools import mcp_tool_loop
    server = discovery._get_connected_server_for_call(record["server"])
    if server is None or server.session is None:
        raise RuntimeError(f"MCP server {record['server']!r} is not connected and the resource is not cached")
    _require_live_origin(server, record, home)
    result = mcp_tool_loop._run_on_mcp_loop(
        lambda: _read_on_session(server, resource["uri"]), timeout=float(getattr(server, "tool_timeout", 300)))
    return decode_read_resource_result(result, resource["uri"])


def get_verified_resource(
    record: dict[str, Any], resource: dict[str, Any], home=None, session_id: str | None = None,
) -> tuple[bytes, str, bool, Path, dict[str, Any]]:
    _ensure_get_verified(record, home, session_id)
    cached = _read_cached(record, resource, home)
    directory = _cache_dir(record, resource, home)
    if cached is not None:
        raw, meta = cached
        is_text = bool(meta.get("is_text"))
        scan = scan_resource_bytes(
            raw, record=record, resource=resource, is_text=is_text,
            mime_type=str(meta.get("mime_type") or ""))
        if meta.get("content_scan") != scan:
            secure_atomic_json(Path(home or get_hermes_home()), _cache_rel_dir(record, resource) / "metadata.json",
                               {**meta, "content_scan": scan}, mode=0o600)
        return raw, str(meta.get("mime_type") or "application/octet-stream"), is_text, directory / "content", scan
    raw, mime, is_text = _fetch(record, resource, home)
    _verify(raw, resource)
    scan = scan_resource_bytes(
        raw, record=record, resource=resource, is_text=is_text, mime_type=mime)
    root = Path(home or get_hermes_home())
    relative = _cache_rel_dir(record, resource)
    secure_atomic_bytes(root, relative / "content", raw, mode=0o600)
    secure_atomic_json(root, relative / "metadata.json", {
        "schema_version": 1, "server": record["server"], "config_fingerprint": record["config_fingerprint"],
        "skill_uri": record["uri"], "uri": resource["uri"], "digest": resource["digest"],
        "size": resource["size"], "mime_type": mime, "is_text": is_text, "fetched_at": time.time(),
        "content_scan": scan,
    }, mode=0o600)
    # Re-read and rehash: a successful write is not the trust decision.
    verified = _read_cached(record, resource, home)
    if verified is None:
        raise ValueError("verified cache write disappeared")
    return verified[0], mime, is_text, directory / "content", scan


def _ensure_local_workspace() -> Path:
    backend = (os.getenv("TERMINAL_ENV") or "local").strip().lower()
    try:
        from tools.terminal_scope import terminal_env
        backend = (terminal_env("TERMINAL_ENV", backend) or backend).strip().lower()
    except (ImportError, RuntimeError):
        pass
    if backend not in {"", "local"}:
        raise RuntimeError(
            f"MCP skill materialization is unavailable for terminal backend {backend!r}: controller-local paths "
            "would not be accessible in that environment. Configure a mounted workspace or materialize there explicitly.")
    workspace = resolve_agent_cwd().expanduser().resolve()
    if not workspace.is_dir():
        raise ValueError("authoritative agent workspace does not exist")
    return workspace


def _collision(parent: Path, name: str) -> bool:
    folded = unicodedata.normalize("NFC", name).casefold()
    try:
        return any(unicodedata.normalize("NFC", child.name).casefold() == folded and child.name != name
                   for child in parent.iterdir())
    except FileNotFoundError:
        return False


def materialize_resource(record: dict[str, Any], resource: dict[str, Any], raw: bytes,
                         destination: str | None = None, *, is_text: bool | None = None,
                         session_id: str | None = None, mime_type: str = "") -> dict[str, Any]:
    _verify(raw, resource)
    workspace = _ensure_local_workspace()
    server_component = str(record["server"])
    skill_component = str(record["frontmatter"]["name"])
    if any(value in {"", ".", ".."} or "/" in value or "\\" in value for value in (server_component, skill_component)):
        raise ValueError("server label and skill name must be safe single path components")
    rel = relative_resource_path(record, resource["uri"])
    origin_root = (
        Path(".hermes") / "mcp-skills" / "materialized"
        / hashlib.sha256((record["server"] + ":" + record["config_fingerprint"]).encode()).hexdigest()
        / hashlib.sha256(record["uri"].encode()).hexdigest()
        / record["manifest_fingerprint"]
    )
    if destination:
        candidate = Path(destination)
        if candidate.is_absolute() or "\\" in destination or any(p in {"", ".", ".."} for p in candidate.parts):
            raise ValueError("materialization destination must be a normalized path within the managed MCP workspace")
        target_rel = origin_root / candidate
    else:
        target_rel = origin_root / rel
    target = workspace / target_rel
    from agent import skill_utils
    discovery_roots = [Path(path).expanduser().resolve(strict=False) for path in skill_utils.get_all_skills_dirs()]
    create_dir = skill_utils.get_skill_create_dir()
    if create_dir is not None:
        discovery_roots.append(create_dir.expanduser().resolve(strict=False))
    skills_cfg = skill_utils._skills_cfg()
    for entry in skill_utils._config_str_list((skills_cfg or {}).get("external_dirs")):
        root = skill_utils._home_relative(skill_utils._expand_path(entry)).resolve(strict=False)
        if root not in discovery_roots:
            discovery_roots.append(root)
    discovery_roots.extend(
        Path(path).expanduser().resolve(strict=False) for path in skill_utils.get_project_skills_dirs())
    discovery_roots.extend((workspace / part).resolve(strict=False) for part in skill_utils.PROJECT_SKILLS_SUBDIRS)
    resolved_workspace = workspace.resolve(strict=False)
    resolved_target = target.resolve(strict=False)
    if any(resolved_workspace.is_relative_to(root) or resolved_target.is_relative_to(root)
           for root in discovery_roots):
        raise ValueError("materialized MCP resources cannot be placed in a local skill discovery root")
    if _collision(target.parent, target.name):
        raise ValueError("materialization path has a case/Unicode collision")
    sidecar_rel = target_rel.parent / f".{target_rel.name}.mcp-skill-origin.json"
    sidecar = workspace / sidecar_rel
    expected = {
        "schema_version": 1, "server": record["server"], "config_fingerprint": record["config_fingerprint"],
        "skill_uri": record["uri"], "resource_uri": resource["uri"], "digest": resource["digest"],
        "size": resource["size"], "manifest_fingerprint": record["manifest_fingerprint"],
    }
    if is_text is None:
        from tools.skills_guard import SCANNABLE_EXTENSIONS
        is_text = PurePosixPath(rel).suffix.lower() in SCANNABLE_EXTENSIONS or PurePosixPath(rel).name == "SKILL.md"
    # Direct callers must not be able to materialize unscanned source bytes;
    # native skill_view has already scanned them, but this boundary stands alone.
    current_scan = scan_resource_bytes(
        raw, record=record, resource=resource, is_text=is_text, mime_type=mime_type)
    try:
        existing = secure_read_bytes(workspace, target_rel)
        provenance = secure_read_json(workspace, sidecar_rel)
        if any(provenance.get(key) != value for key, value in expected.items()):
            raise ValueError("materialization provenance does not own this exact remote origin/version")
        status = "unchanged" if existing == raw else "user_modified"
        current_scan = scan_resource_bytes(
            existing, record=record, resource=resource, is_text=is_text,
            mime_type=mime_type)
    except FileNotFoundError:
        try:
            secure_read_bytes(workspace, target_rel)
        except FileNotFoundError:
            pass
        else:
            raise ValueError("existing materialization target has no provenance sidecar")
        try:
            secure_read_json(workspace, sidecar_rel)
        except FileNotFoundError:
            pass
        else:
            raise ValueError("orphaned materialization provenance sidecar already exists")
        provenance = {**expected, "materialized_at": time.time()}
        secure_atomic_bytes(workspace, target_rel, raw, mode=0o600)
        secure_atomic_json(workspace, sidecar_rel, provenance, mode=0o600)
        status = "created"
    if session_id:
        from tools.mcp_skills_registry import record_materialization
        record_materialization(record, resource, home=get_hermes_home(), session_id=session_id,
                               workspace=workspace, target_rel=target_rel, sidecar_rel=sidecar_rel,
                               is_text=current_scan["scope"] == "single_resource_text")
    return {"status": status, "path": str(target), "digest": resource["digest"], "size": len(raw),
            "provenance": provenance, "provenance_path": str(sidecar), "scan": current_scan}
