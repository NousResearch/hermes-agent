"""Typed, bounded SEP-2640 Skills extension client for MCP.

This module owns wire validation only.  Catalog pinning, cache paths and model
presentation live in sibling modules so extension SDK helpers can replace this
compatibility layer later.
"""
from __future__ import annotations

import base64
import hashlib
from typing import Any, Literal
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

SKILLS_EXTENSION = "io.modelcontextprotocol/skills"
MAX_LIST_PAGES = 50
MAX_SKILLS = 2048
MAX_CATALOG_BYTES = 4 * 1024 * 1024
MAX_RESOURCES = 512
MAX_SKILL_BYTES = 16 * 1024 * 1024
_DIGEST_PREFIX = "sha256:"


class SkillResource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    uri: str
    digest: str
    size: int = Field(ge=0)

    @field_validator("digest")
    @classmethod
    def valid_digest(cls, value: str) -> str:
        if not isinstance(value, str) or not value.startswith(_DIGEST_PREFIX):
            raise ValueError("digest must use sha256:<lowercase hex>")
        hexdigest = value[len(_DIGEST_PREFIX):]
        if len(hexdigest) != 64 or any(c not in "0123456789abcdef" for c in hexdigest):
            raise ValueError("digest must use sha256:<64 lowercase hex>")
        return value


class SkillEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    uri: str
    frontmatter: dict[str, Any]
    resources: list[SkillResource] | Literal["dynamic"]


class SkillsListResult(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    result_type: str = Field(default="complete", alias="resultType")
    skills: list[SkillEntry]
    next_cursor: str | None = Field(default=None, alias="nextCursor")
    ttl_ms: int | None = Field(default=None, alias="ttlMs", ge=0)
    cache_scope: str | None = Field(default=None, alias="cacheScope")


class SkillsGetResult(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    result_type: str = Field(default="complete", alias="resultType")
    skill: SkillEntry


class MCPResource(BaseModel):
    """Ordinary MCP Resource metadata returned by resources/directory/read."""
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    uri: str
    name: str
    title: str | None = None
    description: str | None = None
    mime_type: str | None = Field(default=None, alias="mimeType")
    size: int | None = Field(default=None, ge=0)


class DirectoryReadResult(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    resources: list[MCPResource]
    next_cursor: str | None = Field(default=None, alias="nextCursor")


_LIST_ADAPTER = TypeAdapter(SkillsListResult)
_GET_ADAPTER = TypeAdapter(SkillsGetResult)
_DIRECTORY_ADAPTER = TypeAdapter(DirectoryReadResult)


def skills_opted_in(config: dict[str, Any] | None) -> bool:
    value = ((config or {}).get("skills") or {}).get("enabled", False)
    return value is True or (isinstance(value, str) and value.strip().lower() in {"true", "1", "yes", "on"})


def advertised_skills_settings(initialize_result: Any) -> dict[str, Any] | None:
    capabilities = getattr(initialize_result, "capabilities", None)
    if capabilities is None and isinstance(initialize_result, dict):
        capabilities = initialize_result.get("capabilities")
    extensions = capabilities.get("extensions") if isinstance(capabilities, dict) else getattr(capabilities, "extensions", None)
    if not isinstance(extensions, dict) or SKILLS_EXTENSION not in extensions:
        return None
    settings = extensions[SKILLS_EXTENSION]
    return dict(settings) if isinstance(settings, dict) else None


def directory_read_advertised(initialize_result: Any) -> bool:
    settings = advertised_skills_settings(initialize_result)
    return settings is not None and settings.get("directoryRead") is True


def _uri_parts(uri: str):
    if (not isinstance(uri, str) or not uri or "\\" in uri or "<" in uri or ">" in uri
            or any(ord(ch) < 0x21 or ord(ch) == 0x7f for ch in uri)):
        raise ValueError("resource URI must be a nonempty display-safe URI without whitespace, controls, or backslashes")
    parsed = urlsplit(uri)
    if not parsed.scheme or parsed.fragment or parsed.username or parsed.password:
        raise ValueError("resource URI needs a scheme and cannot contain fragments or userinfo")
    decoded = unquote(parsed.path)
    if "\\" in decoded or any(part in {"", ".", ".."} for part in decoded.split("/")[1:]):
        raise ValueError("resource URI contains an empty, dot, or traversal path component")
    # A second decode catches doubly encoded traversal and separators.
    decoded_twice = unquote(decoded)
    if decoded_twice != decoded and ("\\" in decoded_twice or any(p in {".", ".."} for p in decoded_twice.split("/"))):
        raise ValueError("resource URI contains encoded traversal")
    return parsed, decoded


def validate_skill_entry(entry: SkillEntry | dict[str, Any]) -> SkillEntry:
    item = entry if isinstance(entry, SkillEntry) else SkillEntry.model_validate(entry)
    parsed, skill_path = _uri_parts(item.uri)
    parts = skill_path.rstrip("/").split("/")
    if len(parts) < 2 or parts[-1] != "SKILL.md":
        raise ValueError("skill uri must identify SKILL.md")
    name = item.frontmatter.get("name")
    description = item.frontmatter.get("description")
    if not isinstance(name, str) or not name.strip() or not isinstance(description, str) or not description.strip():
        raise ValueError("skill frontmatter requires nonempty name and description")
    if name in {".", ".."} or "/" in name or "\\" in name or any(ord(ch) < 0x20 or ord(ch) == 0x7f for ch in name):
        raise ValueError("skill name must be one safe path segment")
    # With no organizational prefix, the skill name occupies URI authority.
    # Authority stays an opaque identity component, never a network target.
    directory_name = parts[-2] if len(parts) > 2 else parsed.netloc
    if directory_name != name:
        raise ValueError("skill URI directory must equal frontmatter.name")
    if item.resources == "dynamic":
        raise ValueError("dynamic skill resources are not supported by this host")
    if not item.resources or len(item.resources) > MAX_RESOURCES:
        raise ValueError(f"skill manifest must contain 1..{MAX_RESOURCES} resources")
    root = "/".join(parts[:-1]) + "/"
    seen: set[str] = set()
    total = 0
    skill_md_count = 0
    for resource in item.resources:
        rp, decoded = _uri_parts(resource.uri)
        if (rp.scheme, rp.netloc, rp.query) != (parsed.scheme, parsed.netloc, parsed.query) or not decoded.startswith(root):
            raise ValueError("resource URI escapes the skill root or changes URI authority/query")
        if resource.uri in seen:
            raise ValueError("duplicate resource URI")
        seen.add(resource.uri)
        total += resource.size
        skill_md_count += int(resource.uri == item.uri)
    if skill_md_count != 1:
        raise ValueError("manifest must include its SKILL.md URI exactly once")
    if total > MAX_SKILL_BYTES:
        raise ValueError(f"skill manifest exceeds {MAX_SKILL_BYTES} bytes")
    return item


def manifest_fingerprint(entry: SkillEntry | dict[str, Any]) -> str:
    import json
    item = validate_skill_entry(entry)
    raw = json.dumps(item.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def list_skills(session: Any, server_name: str) -> tuple[list[SkillEntry], dict[str, Any]]:
    import mcp.types as types
    entries: list[SkillEntry] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()
    metadata: dict[str, Any] = {}
    catalog_bytes = 0
    for page_number in range(MAX_LIST_PAGES):
        params = {"cursor": cursor} if cursor else {}
        result = await session.send_request(types.Request[dict, str](method="skills/list", params=params), _LIST_ADAPTER)
        if result.result_type != "complete":
            raise ValueError("skills/list returned an incomplete result")
        if page_number == 0:
            metadata = {"ttl_ms": result.ttl_ms, "cache_scope": result.cache_scope}
        for raw in result.skills:
            entry = validate_skill_entry(raw)
            catalog_bytes += len(entry.model_dump_json().encode("utf-8"))
            entries.append(entry)
            if len(entries) > MAX_SKILLS or catalog_bytes > MAX_CATALOG_BYTES:
                raise ValueError("skills catalog exceeds host limits")
        cursor = result.next_cursor
        if not cursor:
            return entries, metadata
        if cursor in seen_cursors:
            raise ValueError(f"skills/list repeated cursor on server {server_name!r}")
        seen_cursors.add(cursor)
    raise ValueError(f"skills/list exceeded {MAX_LIST_PAGES} pages on server {server_name!r}")


async def get_skill(session: Any, uri: str) -> SkillEntry:
    import mcp.types as types
    result = await session.send_request(
        types.Request[dict, str](method="skills/get", params={"uri": uri}), _GET_ADAPTER)
    if result.result_type != "complete":
        raise ValueError("skills/get returned an incomplete result")
    return validate_skill_entry(result.skill)


async def read_directory(session: Any, uri: str, cursor: str | None = None) -> DirectoryReadResult:
    import mcp.types as types
    params = {"uri": uri, **({"cursor": cursor} if cursor else {})}
    return await session.send_request(
        types.Request[dict, str](method="resources/directory/read", params=params), _DIRECTORY_ADAPTER)


def decode_read_resource_result(result: Any, expected_uri: str) -> tuple[bytes, str, bool]:
    contents = getattr(result, "contents", None)
    if not isinstance(contents, list) or len(contents) != 1:
        raise ValueError("resources/read must return exactly one content block")
    block = contents[0]
    uri = str(getattr(block, "uri", "") or "")
    if uri != expected_uri:
        raise ValueError("resources/read returned a different URI")
    mime = str(getattr(block, "mime_type", None) or getattr(block, "mimeType", None) or "")
    text = getattr(block, "text", None)
    blob = getattr(block, "blob", None)
    if (text is None) == (blob is None):
        raise ValueError("resource content must contain exactly one of text or blob")
    if text is not None:
        return str(text).encode("utf-8"), mime or "text/plain", True
    try:
        return base64.b64decode(str(blob), validate=True), mime or "application/octet-stream", False
    except (ValueError, TypeError) as exc:
        raise ValueError("resource blob is not strict base64") from exc
