#!/usr/bin/env python3
"""Portable, opt-in structured access to bounded Hermes knowledge authorities."""

from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml

from agent.knowledge_router import resolve_information_owner
from tools.registry import registry, tool_error

INFORMATION_TYPE_ENUM = frozenset({
    "HISTORICAL_TRANSCRIPT",
    "PROCEDURE",
    "DOCUMENTARY_KNOWLEDGE",
})

PUBLIC_ARGUMENT_KEYS = frozenset({
    "information_type",
    "query",
    "session_id",
    "around_message_id",
    "window",
    "limit",
    "profile",
    "after",
    "before",
    "skill_identifier",
    "kb_id",
})

MAX_QUERY_CHARS = 4096
MAX_IDENTIFIER_CHARS = 512
MAX_KB_ID_CHARS = 256
MAX_TIME_BOUND_CHARS = 128
MAX_REGISTRY_BYTES = 1024 * 1024
MAX_CONFIG_BYTES = 1024 * 1024
MAX_REGISTRY_RECORDS = 4096
MAX_REGISTRY_PATH_CHARS = 4096
MAX_SKILL_CONTENT_CHARS = 50_000
MAX_SKILL_ROOTS = 64
MAX_SKILL_INDEX_FILES = 4096
MAX_FRONTMATTER_SCAN_CHARS = 65_536
MAX_SKILL_RELATIVE_PATH_CHARS = 4096

_MISSING = object()

KNOWLEDGE_ACCESS_SCHEMA = {
    "name": "knowledge_access",
    "description": (
        "Structured read-only access to past conversation history, procedure skills, "
        "and exact registered documentary knowledge."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "information_type": {
                "type": "string",
                "enum": sorted(INFORMATION_TYPE_ENUM),
            },
            "query": {"type": "string", "maxLength": MAX_QUERY_CHARS},
            "session_id": {"type": "string", "maxLength": MAX_IDENTIFIER_CHARS},
            "around_message_id": {"type": "integer"},
            "window": {"type": "integer", "minimum": 1, "maximum": 20},
            "limit": {"type": "integer", "minimum": 1, "maximum": 10},
            "profile": {"type": "string", "maxLength": MAX_IDENTIFIER_CHARS},
            "after": {"type": "string", "maxLength": MAX_TIME_BOUND_CHARS},
            "before": {"type": "string", "maxLength": MAX_TIME_BOUND_CHARS},
            "skill_identifier": {"type": "string", "maxLength": MAX_IDENTIFIER_CHARS},
            "kb_id": {"type": "string", "maxLength": MAX_KB_ID_CHARS},
        },
        "required": ["information_type"],
    },
}


def _fail(message: str) -> ValueError:
    return ValueError(message[:1000])


def _string(value: Any, name: str, max_chars: int, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise _fail(f"{name} must be a string")
    if len(value) > max_chars:
        raise _fail(f"{name} exceeds {max_chars} characters")
    text = value.strip()
    if not allow_empty and not text:
        raise _fail(f"{name} must be non-empty")
    return text


def _integer(value: Any, name: str, lo: int | None = None, hi: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail(f"{name} must be an integer")
    if lo is not None and value < lo:
        raise _fail(f"{name} must be >= {lo}")
    if hi is not None and value > hi:
        raise _fail(f"{name} must be <= {hi}")
    return value


def _present(value: Any) -> bool:
    return value is not _MISSING


def _reject_present(provided: dict[str, Any], *names: str) -> None:
    hits = [name for name in names if _present(provided[name])]
    if hits:
        raise _fail("Unexpected argument(s) for selected information_type: " + ", ".join(sorted(hits)))


def _parse_native_result(raw: Any, authority: str) -> dict[str, Any]:
    if not isinstance(raw, str):
        raise _fail(f"{authority} returned a non-string result")
    try:
        data = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise _fail(f"{authority} returned invalid JSON") from exc
    if not isinstance(data, dict):
        raise _fail(f"{authority} returned a non-object JSON result")
    if data.get("success") is not True:
        detail = data.get("error") or data.get("message") or "native authority reported failure"
        raise _fail(f"{authority} failed: {str(detail)[:500]}")
    return data


def _historical_adapter(provided: dict[str, Any], *, db: Any = None, current_session_id: str | None = None) -> dict[str, Any]:
    """Delegate to the already-registered native session_search tool without importing it.

    Built-in tool discovery registers session_search before model execution. If that authority is
    absent, registry.dispatch fails closed; knowledge_access never imports session_search_tool on demand.
    """
    query_p = _present(provided["query"])
    session_p = _present(provided["session_id"])
    around_p = _present(provided["around_message_id"])

    profile = None
    if _present(provided["profile"]):
        profile = _string(provided["profile"], "profile", MAX_IDENTIFIER_CHARS)

    native_args: dict[str, Any] = {}
    shape: str

    if around_p and not session_p:
        raise _fail("around_message_id requires session_id")

    if session_p and around_p:
        shape = "scroll"
        _reject_present(provided, "query", "limit", "after", "before", "skill_identifier", "kb_id")
        native_args["session_id"] = _string(provided["session_id"], "session_id", MAX_IDENTIFIER_CHARS)
        native_args["around_message_id"] = _integer(provided["around_message_id"], "around_message_id")
        native_args["window"] = 5 if not _present(provided["window"]) else _integer(provided["window"], "window", 1, 20)
        if profile is not None:
            native_args["profile"] = profile
    elif session_p:
        shape = "read"
        _reject_present(provided, "query", "around_message_id", "window", "limit", "after", "before", "skill_identifier", "kb_id")
        native_args["session_id"] = _string(provided["session_id"], "session_id", MAX_IDENTIFIER_CHARS)
        if profile is not None:
            native_args["profile"] = profile
    elif query_p:
        shape = "discovery"
        _reject_present(provided, "around_message_id", "window", "skill_identifier", "kb_id")
        native_args["query"] = _string(provided["query"], "query", MAX_QUERY_CHARS)
        native_args["limit"] = 3 if not _present(provided["limit"]) else _integer(provided["limit"], "limit", 1, 10)
        if profile is not None:
            native_args["profile"] = profile
        if _present(provided["after"]):
            native_args["after"] = _string(provided["after"], "after", MAX_TIME_BOUND_CHARS)
        if _present(provided["before"]):
            native_args["before"] = _string(provided["before"], "before", MAX_TIME_BOUND_CHARS)
    else:
        shape = "browse"
        _reject_present(provided, "around_message_id", "window", "after", "before", "skill_identifier", "kb_id")
        native_args["query"] = ""
        native_args["limit"] = 3 if not _present(provided["limit"]) else _integer(provided["limit"], "limit", 1, 10)
        if profile is not None:
            native_args["profile"] = profile

    raw = registry.dispatch(
        "session_search",
        native_args,
        db=db,
        current_session_id=current_session_id,
    )
    result = _parse_native_result(raw, "session_search")
    return {"shape": shape, "data": result}


def _validate_skill_identifier(value: Any) -> str:
    text = _string(value, "skill_identifier", MAX_IDENTIFIER_CHARS)
    if "\x00" in text or ":" in text:
        raise _fail("skill_identifier must be a relative non-plugin skill identifier")
    posix = PurePosixPath(text.replace("\\", "/"))
    win = PureWindowsPath(text)
    if posix.is_absolute() or win.is_absolute() or win.drive or ".." in posix.parts or ".." in win.parts:
        raise _fail("skill_identifier must remain within configured skill roots")
    normalized = posix.as_posix().strip("/")
    if not normalized or normalized == ".":
        raise _fail("skill_identifier must be non-empty")
    return normalized


def _read_prefix(path: Path, limit: int) -> str:
    with path.open("r", encoding="utf-8") as handle:
        return handle.read(limit)


def _config_name_set(raw: Any) -> set[str]:
    """Normalize Hermes scalar/list config spellings without importing gateway/CLI writers."""
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("["):
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                parsed = None
            raw = parsed if isinstance(parsed, (list, tuple, set, frozenset)) else [raw]
        else:
            raw = [raw]
    if not isinstance(raw, (list, tuple, set, frozenset)):
        return set()
    return {str(item).strip() for item in raw if str(item).strip()}


def _readonly_session_platform() -> str:
    """Current session platform without importing ``gateway`` as a read side effect.

    A live gateway has already loaded ``gateway.session_context``; consult that existing
    module for task-local ContextVar state. Fresh CLI/cron processes fall back to the same
    environment spellings without forcing the gateway package import graph.
    """
    explicit = (os.getenv("HERMES_PLATFORM") or "").strip()
    if explicit:
        return explicit
    module = sys.modules.get("gateway.session_context")
    if module is not None:
        getter = getattr(module, "get_session_env", None)
        if callable(getter):
            try:
                contextual = getter("HERMES_SESSION_PLATFORM")
            except Exception:
                contextual = None
            if contextual:
                return str(contextual).strip()
    return (os.getenv("HERMES_SESSION_PLATFORM") or "").strip()


def _readonly_disabled_skill_names() -> set[str]:
    """Read disabled-skill policy from config.yaml with no gateway/plugin discovery."""
    from hermes_constants import get_config_path

    path = get_config_path()
    if not path.exists():
        return set()
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise _fail(f"Unable to stat Hermes config for procedure policy: {exc}") from exc
    if size > MAX_CONFIG_BYTES:
        raise _fail(f"Hermes config exceeds {MAX_CONFIG_BYTES} bytes")
    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise _fail(f"Unable to read Hermes config for procedure policy: {exc}") from exc
    if parsed is None:
        parsed = {}
    if not isinstance(parsed, dict):
        raise _fail("Hermes config must be a mapping for procedure policy")
    skills_cfg = parsed.get("skills")
    if skills_cfg is None:
        return set()
    if not isinstance(skills_cfg, dict):
        raise _fail("Hermes skills config must be a mapping for procedure policy")

    disabled = _config_name_set(skills_cfg.get("disabled"))
    platform = _readonly_session_platform()
    platform_map = skills_cfg.get("platform_disabled")
    if platform_map is not None and not isinstance(platform_map, dict):
        raise _fail("skills.platform_disabled must be a mapping")
    if platform and isinstance(platform_map, dict):
        disabled |= _config_name_set(platform_map.get(platform))
    return disabled - {"hermes-agent"}


def _procedure_adapter(provided: dict[str, Any]) -> dict[str, Any]:
    _reject_present(
        provided,
        "query", "session_id", "around_message_id", "window", "limit", "profile", "after", "before", "kb_id",
    )
    if not _present(provided["skill_identifier"]):
        raise _fail("PROCEDURE requires skill_identifier")
    requested = _validate_skill_identifier(provided["skill_identifier"])

    from agent.skill_utils import (
        get_all_skills_dirs,
        iter_skill_index_files,
        parse_frontmatter,
        skill_matches_platform,
    )

    roots = []
    seen_roots = set()
    for raw_root in get_all_skills_dirs():
        root = Path(raw_root)
        if not root.is_dir():
            continue
        try:
            key = str(root.resolve())
        except OSError:
            key = str(root.absolute())
        if key in seen_roots:
            continue
        seen_roots.add(key)
        roots.append(root)
        if len(roots) > MAX_SKILL_ROOTS:
            raise _fail(f"Configured skill roots exceed {MAX_SKILL_ROOTS}")

    disabled = _readonly_disabled_skill_names()
    matches: list[tuple[str, Path, dict[str, Any], str]] = []
    seen_files = set()
    scanned = 0

    for root in roots:
        for skill_md in iter_skill_index_files(root, "SKILL.md"):
            scanned += 1
            if scanned > MAX_SKILL_INDEX_FILES:
                raise _fail(f"Skill index exceeds {MAX_SKILL_INDEX_FILES} files")
            try:
                rel_dir = skill_md.parent.relative_to(root).as_posix()
            except ValueError:
                continue
            prefix = _read_prefix(skill_md, MAX_FRONTMATTER_SCAN_CHARS)
            frontmatter, _ = parse_frontmatter(prefix)
            canonical_name = str(frontmatter.get("name") or skill_md.parent.name).strip()
            if requested != rel_dir and requested != canonical_name:
                continue
            if not canonical_name or len(canonical_name) > MAX_IDENTIFIER_CHARS or "\x00" in canonical_name:
                raise _fail("Matched procedure skill has an invalid canonical name")
            if len(rel_dir) > MAX_SKILL_RELATIVE_PATH_CHARS or "\x00" in rel_dir:
                raise _fail("Matched procedure skill has an invalid relative path")
            try:
                file_key = str(skill_md.resolve())
            except OSError:
                file_key = str(skill_md.absolute())
            if file_key in seen_files:
                continue
            seen_files.add(file_key)
            matches.append((rel_dir, skill_md, frontmatter, canonical_name))

    if not matches:
        raise _fail(f"Unknown procedure skill: {requested}")
    if len(matches) != 1:
        raise _fail(f"Ambiguous procedure skill: {requested}")

    rel_dir, skill_md, frontmatter, canonical_name = matches[0]
    if canonical_name in disabled:
        raise _fail(f"Procedure skill is disabled: {canonical_name}")
    if not skill_matches_platform(frontmatter):
        raise _fail(f"Procedure skill is not supported on this platform: {canonical_name}")

    with skill_md.open("r", encoding="utf-8") as handle:
        content = handle.read(MAX_SKILL_CONTENT_CHARS + 1)
    if len(content) > MAX_SKILL_CONTENT_CHARS:
        raise _fail(f"Procedure skill exceeds {MAX_SKILL_CONTENT_CHARS} characters")

    description = str(frontmatter.get("description") or "")
    if len(description) > 4096:
        description = description[:4096]

    return {
        "requested_identifier": requested,
        "name": canonical_name,
        "description": description,
        "content": content,
        "relative_skill_path": f"{rel_dir}/SKILL.md" if rel_dir not in {"", "."} else "SKILL.md",
    }


def _registry_path(override: Any = None) -> Path:
    if override is not None:
        return Path(override)
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "knowledge" / "KB_REGISTRY.md"


def _load_document_registry(override: Any = None) -> list[dict[str, str]]:
    path = _registry_path(override)
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise _fail("Knowledge registry unavailable") from exc
    if size > MAX_REGISTRY_BYTES:
        raise _fail(f"Knowledge registry exceeds {MAX_REGISTRY_BYTES} bytes")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise _fail("Knowledge registry unreadable") from exc

    blocks: list[str] = []
    active = False
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not active and stripped.lower() in {"```yaml", "```yml"}:
            active = True
            lines = []
            continue
        if active and stripped.startswith("```"):
            blocks.append("\n".join(lines))
            active = False
            lines = []
            continue
        if active:
            lines.append(line)
    if active or len(blocks) != 1:
        raise _fail("Knowledge registry must contain exactly one closed YAML fenced block")
    try:
        data = yaml.safe_load(blocks[0])
    except yaml.YAMLError as exc:
        raise _fail("Knowledge registry YAML is malformed") from exc
    if not isinstance(data, list):
        raise _fail("Knowledge registry YAML must be a list")
    if len(data) > MAX_REGISTRY_RECORDS:
        raise _fail(f"Knowledge registry exceeds {MAX_REGISTRY_RECORDS} records")

    records: list[dict[str, str]] = []
    seen_ids = set()
    for index, record in enumerate(data):
        if not isinstance(record, dict):
            raise _fail(f"Knowledge registry record {index} must be an object")
        kb_id = record.get("id")
        canonical_path = record.get("path")
        if not isinstance(kb_id, str) or not kb_id.strip() or len(kb_id) > MAX_KB_ID_CHARS or "\x00" in kb_id:
            raise _fail(f"Knowledge registry record {index} has invalid id")
        if not isinstance(canonical_path, str) or not canonical_path.strip() or len(canonical_path) > MAX_REGISTRY_PATH_CHARS or "\x00" in canonical_path:
            raise _fail(f"Knowledge registry record {index} has invalid path")
        kb_id = kb_id.strip()
        if kb_id in seen_ids:
            raise _fail(f"Duplicate knowledge registry id: {kb_id}")
        seen_ids.add(kb_id)
        records.append({"id": kb_id, "path": canonical_path.strip()})
    return records


def _documentary_adapter(provided: dict[str, Any], *, registry_path: Any = None) -> dict[str, Any]:
    _reject_present(
        provided,
        "query", "session_id", "around_message_id", "window", "limit", "profile", "after", "before", "skill_identifier",
    )
    if not _present(provided["kb_id"]):
        raise _fail("DOCUMENTARY_KNOWLEDGE requires kb_id")
    requested = _string(provided["kb_id"], "kb_id", MAX_KB_ID_CHARS)
    matches = [record for record in _load_document_registry(registry_path) if record["id"] == requested]
    if len(matches) != 1:
        raise _fail(f"Unknown knowledge registry id: {requested}")
    return {"kb_id": requested, "canonical_path": matches[0]["path"]}


def execute_knowledge_access(
    *,
    information_type: Any,
    query: Any = _MISSING,
    session_id: Any = _MISSING,
    around_message_id: Any = _MISSING,
    window: Any = _MISSING,
    limit: Any = _MISSING,
    profile: Any = _MISSING,
    after: Any = _MISSING,
    before: Any = _MISSING,
    skill_identifier: Any = _MISSING,
    kb_id: Any = _MISSING,
    db: Any = None,
    current_session_id: str | None = None,
    registry_path: Any = None,
) -> dict[str, Any]:
    if not isinstance(information_type, str) or information_type not in INFORMATION_TYPE_ENUM:
        raise _fail(f"Unsupported information_type: {information_type!r}")

    provided = {
        "query": query,
        "session_id": session_id,
        "around_message_id": around_message_id,
        "window": window,
        "limit": limit,
        "profile": profile,
        "after": after,
        "before": before,
        "skill_identifier": skill_identifier,
        "kb_id": kb_id,
    }

    if information_type == "DOCUMENTARY_KNOWLEDGE":
        route_kb = None if kb_id is _MISSING else kb_id
        owner = resolve_information_owner(information_type, kb_id=route_kb)
    else:
        owner = resolve_information_owner(information_type)

    try:
        if owner == "SESSION_SEARCH":
            result = _historical_adapter(provided, db=db, current_session_id=current_session_id)
            provenance = {"authority": "registered_tool:session_search", "mode": "registry_dispatch", "bounded": True}
        elif owner == "SKILLS":
            result = _procedure_adapter(provided)
            provenance = {"authority": "agent.skill_utils", "mode": "readonly_subset", "bounded": True}
        elif owner == "CANONICAL_KB":
            result = _documentary_adapter(provided, registry_path=registry_path)
            provenance = {"authority": "KB_REGISTRY.md", "exact_id": True, "bounded": True}
        else:
            raise _fail(f"Unsupported routed owner: {owner}")
    except ValueError:
        raise
    except Exception as exc:
        raise _fail(f"{owner} authority unavailable ({type(exc).__name__})") from exc

    return {
        "information_type": information_type,
        "owner": owner,
        "result": result,
        "provenance": provenance,
    }


def _knowledge_access_handler(args: Any, **kwargs: Any) -> str:
    if not isinstance(args, dict):
        return tool_error("knowledge_access arguments must be an object", success=False)
    unknown = sorted(set(args) - PUBLIC_ARGUMENT_KEYS)
    if unknown:
        return tool_error("Unknown knowledge_access argument(s): " + ", ".join(unknown), success=False)
    if "information_type" not in args:
        return tool_error("knowledge_access requires information_type", success=False)
    call_args = {key: value for key, value in args.items() if key != "information_type"}
    try:
        result = execute_knowledge_access(
            information_type=args["information_type"],
            db=kwargs.get("db"),
            current_session_id=kwargs.get("current_session_id"),
            **call_args,
        )
    except ValueError as exc:
        return tool_error(str(exc)[:1000], success=False)
    return json.dumps(result, ensure_ascii=False)


registry.register(
    name="knowledge_access",
    toolset="knowledge_access",
    schema=KNOWLEDGE_ACCESS_SCHEMA,
    handler=_knowledge_access_handler,
    emoji="🧭",
    max_result_size_chars=100_000,
)
