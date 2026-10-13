"""Wire-level models and validation for the SEP-2640 skills extension.

Split from the manager on purpose: everything here is pure (given a payload it
returns a verdict or a contract object) so it can be tested without a transport,
and so no validation rule can silently depend on which connection produced a
result. Field constraints come from the stable spec: a host MUST support up to
512 resources / 16 MiB per skill, entries carry ``sha256:<64 lowercase hex>``
digests, and a manifest must cover SKILL.md exactly once with every other file
inside the skill's own root.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, Sequence
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, ConfigDict
from mcp_types import Request

from agent.skills_contracts import (
    BackendFeatures,
    BACKEND_MCP,
    ManifestFile,
    OriginInfo,
    ResourceManifest,
    SkillEntry,
    SkillError,
    SkillErrorCode,
    SkillIdentity,
    VerificationResult,
)

logger = logging.getLogger(__name__)

SKILLS_EXTENSION_ID = "io.modelcontextprotocol/skills"
METHOD_LIST = "skills/list"
METHOD_GET = "skills/get"
METHOD_DIRECTORY_READ = "resources/directory/read"

MAIN_FILE_NAME = "SKILL.md"
MAX_RESOURCES = 512
MAX_TOTAL_BYTES = 16 * 1024 * 1024
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SHA256_PREFIX = "sha256:"


class _Loose(BaseModel):
    model_config = ConfigDict(extra="allow")


class SkillsListParams(_Loose):
    cursor: Optional[str] = None


class SkillsGetParams(_Loose):
    uri: str


class DirectoryReadParams(_Loose):
    uri: str
    cursor: Optional[str] = None


class SkillsListRequest(Request[SkillsListParams, Literal["skills/list"]]):
    method: Literal["skills/list"] = METHOD_LIST
    params: SkillsListParams = SkillsListParams()


class SkillsGetRequest(Request[SkillsGetParams, Literal["skills/get"]]):
    method: Literal["skills/get"] = METHOD_GET
    params: SkillsGetParams


class DirectoryReadRequest(Request[DirectoryReadParams, Literal["resources/directory/read"]]):
    method: Literal["resources/directory/read"] = METHOD_DIRECTORY_READ
    params: DirectoryReadParams


class SkillsListResult(_Loose):
    """``skills/list`` page. ``skills`` stays raw dicts: entry parsing is
    :func:`parse_skill_entry`'s job, and a malformed entry must be quarantined
    individually rather than failing the whole page."""

    skills: list[dict[str, Any]] = []
    nextCursor: Optional[str] = None


class SkillsGetResult(_Loose):
    skill: Optional[dict[str, Any]] = None


class ReadResourceParams(_Loose):
    uri: str


class ReadResourceRequest(Request[ReadResourceParams, Literal["resources/read"]]):
    method: Literal["resources/read"] = "resources/read"
    params: ReadResourceParams


class ResourceContents(_Loose):
    uri: str
    mimeType: Optional[str] = None
    text: Optional[str] = None
    blob: Optional[str] = None


class ReadResourceResult(_Loose):
    contents: list[ResourceContents] = []


class DirectoryEntry(_Loose):
    uri: str
    mimeType: Optional[str] = None
    name: Optional[str] = None


class DirectoryReadResult(_Loose):
    entries: list[DirectoryEntry] = []
    nextCursor: Optional[str] = None


@dataclass(frozen=True)
class CapabilityVerdict:
    """The gate a server must pass before Hermes sends any skills/* request."""

    support: str
    directory_read: bool = False
    protocol_version: str = ""
    reason: str = ""

    @property
    def supported(self) -> bool:
        return self.support == "supported"


def parse_capabilities(capabilities: Any, protocol_version: str) -> CapabilityVerdict:
    """Decide whether this instance's advertised capabilities admit the skills channel.

    The extension identifier is declared in ``initialize.capabilities.extensions``
    on both eras; modern revisions may additionally restate it per request, which
    does not change the verdict. ``resources`` is required by the spec alongside
    the extension — without it there is no way to read a skill's files, so the
    channel is unusable even though the extension is advertised.
    """
    from agent.skills_contracts import SkillsSupport

    if capabilities is None:
        return CapabilityVerdict(SkillsSupport.UNSUPPORTED.value, protocol_version=protocol_version,
                                 reason="server advertised no capabilities")
    extensions = _get(capabilities, "extensions")
    if not isinstance(extensions, Mapping):
        return CapabilityVerdict(SkillsSupport.UNSUPPORTED.value, protocol_version=protocol_version,
                                 reason="server does not advertise extensions")
    settings = extensions.get(SKILLS_EXTENSION_ID)
    if settings is None:
        return CapabilityVerdict(SkillsSupport.UNSUPPORTED.value, protocol_version=protocol_version,
                                 reason=f"server does not advertise {SKILLS_EXTENSION_ID}")
    if not isinstance(settings, Mapping):
        return CapabilityVerdict(SkillsSupport.PROTOCOL_INVALID.value, protocol_version=protocol_version,
                                 reason=f"{SKILLS_EXTENSION_ID} settings must be an object")
    if _get(capabilities, "resources") is None:
        return CapabilityVerdict(SkillsSupport.PROTOCOL_INVALID.value, protocol_version=protocol_version,
                                 reason="skills extension advertised without the required resources capability")
    directory_read = settings.get("directoryRead") is True
    unknown = sorted(set(settings) - {"directoryRead"})
    if unknown:
        # Extra settings are forward-compatible by design; log rather than refuse.
        logger.debug("skills extension advertised unknown settings: %s", ", ".join(unknown))
    return CapabilityVerdict(SkillsSupport.SUPPORTED.value, directory_read=directory_read,
                             protocol_version=protocol_version, reason=f"negotiated {protocol_version}")


def _get(obj: Any, name: str) -> Any:
    """Attribute-or-key read: SDK capability objects and raw dicts both flow through here."""
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def capabilities_mapping(capabilities: Any) -> Any:
    """Normalize an SDK capability object into a plain mapping (or return it unchanged)."""
    if capabilities is None or isinstance(capabilities, Mapping):
        return capabilities
    dump = getattr(capabilities, "model_dump", None)
    if callable(dump):
        try:
            return dump(by_alias=True, exclude_none=True)
        except Exception:  # pragma: no cover - defensive for foreign model shapes
            logger.debug("capability model_dump failed", exc_info=True)
    return capabilities


# ── URI handling ────────────────────────────────────────────────────────────

def split_skill_uri(uri: str) -> tuple[str, str, tuple[str, ...]]:
    """``(scheme, authority, path segments)`` for a skill resource URI.

    Segments are validated here, not by the caller: ``..``/``.``/empty components
    and percent-encoded traversal all make a URI unusable for containment checks,
    and a URI we cannot reason about must never reach the manifest trust path.
    """
    if not isinstance(uri, str) or not uri:
        raise SkillError(SkillErrorCode.PROTOCOL_INVALID, "skill resource URI must be a non-empty string")
    parts = urlsplit(uri)
    if not parts.scheme or not parts.netloc:
        raise SkillError(SkillErrorCode.PROTOCOL_INVALID,
                         f"skill resource URI needs a scheme and an authority: {uri!r}")
    raw_segments = parts.path.split("/")[1:] if parts.path else []
    if not raw_segments or any(not seg for seg in raw_segments):
        raise SkillError(SkillErrorCode.PROTOCOL_INVALID, f"skill resource URI has an empty path segment: {uri!r}")
    for seg in raw_segments:
        decoded = unquote(seg)
        if decoded in (".", "..") or "/" in decoded or "\\" in decoded or "\x00" in decoded:
            raise SkillError(SkillErrorCode.PROTOCOL_INVALID,
                             f"skill resource URI has a traversal segment: {uri!r}")
    return parts.scheme, parts.netloc, tuple(raw_segments)


def _root_origin(root_uri: str) -> tuple[str, str, tuple[str, ...]]:
    """``(scheme, authority, segments)`` of a skill *root*, which may carry no path.

    A skill published directly under an authority (``skill://team/SKILL.md``) has
    the root ``skill://team`` — parsing it with :func:`split_skill_uri` would fail
    on the empty path, so the root is resolved structurally instead of by
    re-appending ``/SKILL.md`` and re-parsing.
    """
    parts = urlsplit(root_uri)
    if not parts.scheme or not parts.netloc:
        raise SkillError(SkillErrorCode.PROTOCOL_INVALID, f"skill root URI needs a scheme and authority: {root_uri!r}")
    segments = tuple(seg for seg in parts.path.split("/")[1:] if seg) if parts.path else ()
    for seg in segments:
        decoded = unquote(seg)
        if decoded in (".", "..") or "/" in decoded or "\\" in decoded or "\x00" in decoded:
            raise SkillError(SkillErrorCode.PROTOCOL_INVALID, f"skill root URI has a traversal segment: {root_uri!r}")
    return parts.scheme, parts.netloc, segments


def skill_root_uri(uri: str) -> str:
    """``skill://acme/billing/refunds/SKILL.md`` → ``skill://acme/billing/refunds``."""
    scheme, authority, segments = split_skill_uri(uri)
    if segments[-1] != MAIN_FILE_NAME:
        raise SkillError(SkillErrorCode.PROTOCOL_INVALID,
                         f"skill URI must address {MAIN_FILE_NAME}: {uri!r}")
    return "/".join([f"{scheme}://{authority}", *segments[:-1]])


def relative_path_within(root_uri: str, uri: str) -> str:
    """Path of *uri* relative to *root_uri*, or raise when it escapes the root.

    Compared segment-wise on the raw spellings so ``skill://a/x`` can never be
    read as a prefix of ``skill://a/xy``.
    """
    scheme, authority, root_segments = _root_origin(root_uri)
    uri_scheme, uri_authority, uri_segments = split_skill_uri(uri)
    if (scheme, authority) != (uri_scheme, uri_authority):
        raise SkillError(SkillErrorCode.PROTOCOL_INVALID, f"{uri!r} is not on the same origin as {root_uri!r}")
    if uri_segments[: len(root_segments)] != root_segments or len(uri_segments) <= len(root_segments):
        raise SkillError(SkillErrorCode.PROTOCOL_INVALID, f"{uri!r} is outside the skill root {root_uri!r}")
    return "/".join(uri_segments[len(root_segments):])


# ── Entry parsing ───────────────────────────────────────────────────────────

def parse_skill_entry(raw: Any, *, origin_id: str, label: str) -> SkillEntry:
    """Validate one ``skills/list`` / ``skills/get`` entry into a :class:`SkillEntry`.

    Raises :class:`SkillError` with ``protocol_invalid`` /
    ``limit_exceeded`` / ``dynamic_disabled``. The manifest is checked for
    internal consistency only — whether the files exist and match their digests
    is decided when bytes actually arrive, never from metadata.
    """
    if not isinstance(raw, Mapping):
        raise SkillError(SkillErrorCode.PROTOCOL_INVALID, "skill entry must be an object")
    uri = raw.get("uri")
    root = skill_root_uri(uri)
    frontmatter = raw.get("frontmatter")
    if not isinstance(frontmatter, Mapping):
        raise SkillError(SkillErrorCode.PROTOCOL_INVALID, f"skill {uri!r} has no frontmatter object")
    name = _frontmatter_name(frontmatter, uri)
    manifest = _parse_manifest(raw.get("resources"), uri, root)
    identity = SkillIdentity(backend_id=BACKEND_MCP, origin_id=origin_id, locator=uri)
    return SkillEntry(
        identity=identity,
        name=name,
        description=_description_of(frontmatter),
        frontmatter=dict(frontmatter),
        manifest=manifest,
        content_revision=manifest.fingerprint() if not manifest.dynamic else None,
        features=BackendFeatures(attachments=bool(manifest.files), directory=True, remote=True),
        provenance=OriginInfo(backend_id=BACKEND_MCP, origin_id=origin_id, label=label, detail=root),
    )


def _frontmatter_name(frontmatter: Mapping[str, Any], uri: str) -> str:
    name = frontmatter.get("name")
    if not isinstance(name, str) or not name.strip():
        raise SkillError(SkillErrorCode.PROTOCOL_INVALID, f"skill {uri!r} frontmatter has no name")
    return name.strip()


def _description_of(frontmatter: Mapping[str, Any]) -> str:
    description = frontmatter.get("description")
    return description.strip() if isinstance(description, str) else ""


def _parse_manifest(resources: Any, uri: str, root: str) -> ResourceManifest:
    if resources == "dynamic":
        # Legal per spec, but a dynamic manifest carries no content binding, so a
        # host cannot ever prove that what it reads is what the author meant.
        return ResourceManifest(files=(), dynamic=True)
    if not isinstance(resources, Sequence) or isinstance(resources, (str, bytes)) or not resources:
        raise SkillError(SkillErrorCode.PROTOCOL_INVALID, f"skill {uri!r} manifest must be a non-empty array or 'dynamic'")
    if len(resources) > MAX_RESOURCES:
        raise SkillError(SkillErrorCode.LIMIT_EXCEEDED,
                         f"skill {uri!r} lists {len(resources)} resources; the host accepts at most {MAX_RESOURCES}")
    files: list[ManifestFile] = []
    seen_uris: set[str] = set()
    total = 0
    main_seen = False
    for item in resources:
        if not isinstance(item, Mapping):
            raise SkillError(SkillErrorCode.PROTOCOL_INVALID, f"skill {uri!r} has a non-object manifest entry")
        file_uri = item.get("uri")
        relative = relative_path_within(root, file_uri)
        if file_uri in seen_uris:
            raise SkillError(SkillErrorCode.PROTOCOL_INVALID, f"skill {uri!r} lists {file_uri!r} twice")
        seen_uris.add(file_uri)
        digest = item.get("digest")
        if not isinstance(digest, str) or not DIGEST_RE.match(digest):
            raise SkillError(SkillErrorCode.PROTOCOL_INVALID,
                             f"skill {uri!r} resource {file_uri!r} has an invalid sha256 digest")
        size = item.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise SkillError(SkillErrorCode.PROTOCOL_INVALID,
                             f"skill {uri!r} resource {file_uri!r} has an invalid size")
        total += size
        main_seen = main_seen or relative == MAIN_FILE_NAME
        files.append(ManifestFile(uri=file_uri, digest=digest, size=size, relative_path=relative))
    if not main_seen:
        raise SkillError(SkillErrorCode.PROTOCOL_INVALID, f"skill {uri!r} manifest does not list its own {MAIN_FILE_NAME}")
    if total > MAX_TOTAL_BYTES:
        raise SkillError(SkillErrorCode.LIMIT_EXCEEDED,
                         f"skill {uri!r} declares {total} bytes; the host accepts at most {MAX_TOTAL_BYTES}")
    return ResourceManifest(files=tuple(files))


def verify_bytes(raw: bytes, expected: ManifestFile, *, expect_text: bool) -> VerificationResult:
    """Check raw resource bytes against the held manifest entry.

    Runs on the untouched bytes — no unicode normalization, newline rewriting or
    template expansion may happen before the digest is computed, or the check
    would be measuring the host's edit rather than the server's bytes.
    """
    errors: list[str] = []
    size_ok = len(raw) == expected.size
    if not size_ok:
        errors.append(f"size {len(raw)} != manifest {expected.size}")
    digest_ok = _SHA256_PREFIX + hashlib.sha256(raw).hexdigest() == expected.digest
    if not digest_ok:
        errors.append("sha256 digest does not match the manifest")
    frontmatter_ok = True
    if expect_text:
        frontmatter_ok = _has_frontmatter(raw)
        if not frontmatter_ok:
            errors.append(f"{MAIN_FILE_NAME} has no frontmatter block")
    return VerificationResult(size_ok=size_ok, digest_ok=digest_ok, frontmatter_ok=frontmatter_ok, errors=tuple(errors))


def _has_frontmatter(raw: bytes) -> bool:
    """Whether the bytes open with a YAML frontmatter fence (parsed elsewhere)."""
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return False
    return text.lstrip().startswith("---")


def decode_text(raw: bytes) -> Optional[str]:
    """UTF-8 text of *raw*, or None for binary (spec: text and blob are exclusive)."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def linked_files_of(entry: SkillEntry) -> dict[str, list[str]]:
    """``{group: [relative path, …]}`` for a remote entry, mirroring the local
    ``linked_files`` shape so the model sees one vocabulary across backends."""
    if entry.manifest is None or entry.manifest.dynamic:
        return {}
    groups: dict[str, list[str]] = {}
    for item in entry.manifest.files:
        if item.relative_path == MAIN_FILE_NAME:
            continue
        top = item.relative_path.split("/", 1)[0] if "/" in item.relative_path else "other"
        groups.setdefault(top, []).append(item.relative_path)
    return {k: sorted(v) for k, v in groups.items()}
