"""Versioned, opt-in identity metadata for the existing section-delimited memory files."""
from __future__ import annotations

import json
from uuid import UUID, uuid4

DELIMITER = "\n§\n"
HEADER = "<!-- hermes-memory-identities:v1 -->"
PREFIX = "<!-- hermes-memory-entry:"
SUFFIX = " -->"


class IdentityError(ValueError):
    """Invalid or missing identity metadata; management must not guess another entry."""


class EntryText(str):
    """Prose with identity attached in memory; string operations/rendering expose only prose."""

    def __new__(cls, content: str, entry_id: str, target: str, scope: str | None = None):
        obj = super().__new__(cls, content)
        obj.entry_id, obj.target, obj.scope = entry_id, target, scope
        return obj


def validate_id(value: str) -> str:
    try:
        if not isinstance(value, str) or str(UUID(value)) != value or UUID(value).version != 4:
            raise ValueError
    except (ValueError, AttributeError, TypeError) as exc:
        raise IdentityError("entry_id must be a canonical UUIDv4.") from exc
    return value


def validate_scope(scope: str | None) -> None:
    if scope is not None and (not isinstance(scope, str) or not scope or len(scope) > 256
                              or any(ord(char) < 32 or ord(char) == 127 for char in scope)):
        raise IdentityError("scope must be a nonempty string of at most 256 characters, without control characters.")


def validate_content(content: str) -> None:
    if DELIMITER in content or PREFIX in content or "hermes-memory-identities:" in content:
        raise IdentityError("Entry prose must not contain the section delimiter or reserved identity metadata.")


def enabled(raw: str) -> bool:
    return raw.strip().startswith(HEADER)


def new_entry(content: str, target: str, scope: str | None = None) -> EntryText:
    validate_scope(scope)
    validate_content(content)
    return EntryText(content, str(uuid4()), target, scope)


def inherit_identity(content: str, previous: str) -> str:
    if not isinstance(previous, EntryText):
        return content
    validate_content(content)
    return EntryText(content, previous.entry_id, previous.target, previous.scope)


def id_fields(entry: str) -> dict:
    return {"entry_id": entry.entry_id, "scope": entry.scope} if isinstance(entry, EntryText) else {}


def _unique_fields(pairs):
    fields = {}
    for key, value in pairs:
        if key in fields:
            raise IdentityError("Memory identity metadata contains duplicate fields.")
        fields[key] = value
    return fields


def _decode_chunk(chunk: str, target: str | None) -> EntryText:
    line, separator, content = chunk.partition("\n")
    if not separator or not line.startswith(PREFIX) or not line.endswith(SUFFIX) or len(line) > 4096:
        raise IdentityError("An identity-managed entry is missing its metadata header.")
    try:
        metadata = json.loads(line[len(PREFIX):-len(SUFFIX)], object_pairs_hook=_unique_fields)
    except (ValueError, TypeError) as exc:
        raise IdentityError("Memory identity metadata is malformed.") from exc
    if not isinstance(metadata, dict) or set(metadata) != {"id", "target", "scope"}:
        raise IdentityError("Memory identity metadata has unknown or missing fields.")
    if (not isinstance(metadata["target"], str) or metadata["target"] not in {"memory", "user"}
            or target is not None and metadata["target"] != target):
        raise IdentityError("Memory identity metadata belongs to a different target.")
    validate_id(metadata["id"])
    validate_scope(metadata["scope"])
    validate_content(content)
    if not content or content != content.strip():
        raise IdentityError("Identity-managed entry prose is empty or has noncanonical whitespace.")
    return EntryText(content, metadata["id"], metadata["target"], metadata["scope"])


def parse_entries(raw: str, target: str | None = None) -> list[str]:
    text = raw.strip()
    if not enabled(raw):
        if PREFIX in raw or "hermes-memory-identities:" in raw:
            raise IdentityError("Memory identity metadata is incomplete or uses an unsupported version.")
        return [entry for chunk in raw.split(DELIMITER) if (entry := chunk.strip())]
    body = text[len(HEADER):].removeprefix("\n")
    entries = [_decode_chunk(chunk, target) for chunk in body.split(DELIMITER)] if body else []
    if len({entry.entry_id for entry in entries}) != len(entries):
        raise IdentityError("Duplicate memory UUIDs make identity management ambiguous.")
    if encode_entries(entries, identity_enabled=True, target=target) != text:
        raise IdentityError("Memory identity metadata would not round-trip; the file was left unchanged.")
    return entries


def deduplicate_entries(entries: list[str]) -> list[str]:
    # Two equal texts can have different identities/scopes after explicit management writes.
    return entries if any(isinstance(entry, EntryText) for entry in entries) else list(dict.fromkeys(entries))


def encode_entries(entries: list[str], *, identity_enabled: bool, target: str | None = None) -> str:
    if not identity_enabled:
        return DELIMITER.join(entries)
    lines = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, EntryText):
            if target is None:
                raise IdentityError("Cannot assign memory identity without an explicit target.")
            entry = entries[index] = new_entry(entry, target)
        if target is not None and entry.target != target:
            raise IdentityError("Memory identity metadata belongs to a different target.")
        validate_id(entry.entry_id)
        validate_scope(entry.scope)
        validate_content(entry)
        metadata = json.dumps({"id": entry.entry_id, "target": entry.target, "scope": entry.scope},
                              ensure_ascii=False, separators=(",", ":"))
        lines.append(f"{PREFIX}{metadata}{SUFFIX}\n{entry}")
    if len({entry.entry_id for entry in entries}) != len(entries):
        raise IdentityError("Duplicate memory UUIDs make identity management ambiguous.")
    return HEADER + ("\n" + DELIMITER.join(lines) if lines else "")
