"""Skill-set catalog client — prototype for the agentskills #254 + AI Catalog layering.

Implements the design discussed with agentskills.io:

1. **AI Catalog** (https://ai-catalog.io) — a typed discovery index served at
   ``/.well-known/ai-catalog.json``. Entries with
   ``type: "application/agent-skills+json"`` point at an agent-skills
   discovery index and represent an installable *skill set*.

2. **agentskills PR #254 discovery index** — the index the entry points at:
   ``{"$schema": ..., "skills": [{name, description, type, url, digest}]}``
   where ``type`` is ``"skill-md"`` (single file) or ``"archive"``
   (``.tar.gz`` / ``.zip`` with SKILL.md at the archive root).

3. **``io.hermes.skill-set`` extension** — optional namespaced metadata on
   the AI Catalog entry carrying set-level usage intent::

       "extensions": {
         "io.hermes.skill-set": {
           "command": "backend-dev",
           "instruction": "Prefer TDD. Run the linter before opening a PR."
         }
       }

   ``command`` is the suggested local load-alias (Hermes creates a skill
   bundle so ``/backend-dev`` loads every member in one turn);
   ``instruction`` is a shared preamble injected above the member skills.
   Clients that don't recognize the extension still install the correct
   set — they only miss the alias/preamble sugar.

Catalog entries may deliver their artifact via ``url`` OR inline ``data``
(per the AI Catalog spec); both are supported for skill-set entries and
sub-catalogs. Relative URLs are resolved per RFC 3986 against the URL the
referencing document was *actually retrieved from* (i.e. after redirects),
and archive formats are detected from the ``Content-Type`` header first,
falling back to the URL file extension when the header is absent or
generic — both per agentskills #254.

Security posture matches the rest of the skills hub: all HTTP goes through
the SSRF-guarded fetcher, member artifacts are digest-verified (SHA-256,
required by #254), archive extraction rejects path traversal / absolute
paths / symlinks / hardlinks and caps decompressed size, and every fetched
skill flows through the existing quarantine -> scan -> install pipeline.

This module only *fetches and validates*; installation is orchestrated by
``hermes_cli.skills_hub.do_install_set``.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import posixpath
import re
import tarfile
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: MIME-ish AI Catalog entry type identifying an agent-skills discovery index.
SKILL_SET_ENTRY_TYPE = "application/agent-skills+json"

#: Nested AI Catalog entry type (sub-catalogs) — followed one level deep.
CATALOG_ENTRY_TYPE = "application/ai-catalog+json"

#: Namespaced AI Catalog extension carrying set-level usage intent.
HERMES_SET_EXTENSION = "io.hermes.skill-set"

#: ``$schema`` URIs this client knows how to process (agentskills #254).
KNOWN_INDEX_SCHEMAS = frozenset({
    "https://schemas.agentskills.io/discovery/0.2.0/schema.json",
})

#: Well-known path for AI Catalog discovery.
AI_CATALOG_WELL_KNOWN = "/.well-known/ai-catalog.json"

_DIGEST_RE = re.compile(r"^sha256:([0-9a-f]{64})$")

# Archive safety caps (decompression-bomb guard per #254 "Archive safety").
MAX_ARCHIVE_BYTES = 20 * 1024 * 1024        # compressed artifact cap
MAX_UNPACKED_BYTES = 50 * 1024 * 1024       # total decompressed cap
MAX_MEMBER_BYTES = 5 * 1024 * 1024          # single decompressed file cap
MAX_ARCHIVE_MEMBERS = 500

#: Max skills a single set may enumerate (sanity cap for the prototype).
MAX_SET_MEMBERS = 50


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class SkillSetError(Exception):
    """Base error for skill-set catalog operations."""


class SchemaError(SkillSetError):
    """Unrecognized or missing ``$schema`` — per #254, warn and stop."""


class DigestError(SkillSetError):
    """Artifact digest missing, malformed, or mismatched."""


class ArchiveSafetyError(SkillSetError):
    """Archive violated safety rules (traversal, links, bomb, structure)."""


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SkillSetInfo:
    """A skill set discovered in an AI Catalog (or a bare #254 index)."""
    name: str                       # display name (catalog displayName or derived)
    description: str
    index_url: str                  # resolved URL of the #254 index.json (or a
                                    #   synthetic "#inline" marker for data entries)
    identifier: str = ""            # catalog entry identifier (urn:...), if any
    command: str = ""               # io.hermes.skill-set suggested alias
    instruction: str = ""           # io.hermes.skill-set shared preamble
    catalog_url: str = ""           # catalog this was discovered from, if any
    inline_index: Optional[Dict[str, Any]] = None   # AI Catalog `data` payload
    base_url: str = ""              # RFC 3986 base for inline-index member URLs


@dataclass
class SkillSetMember:
    """One skill entry from a #254 discovery index."""
    name: str
    description: str
    type: str                       # "skill-md" | "archive"
    url: str                        # resolved absolute URL
    digest: str                     # "sha256:<hex>"


@dataclass
class ResolvedSkillSet:
    """A fully parsed set: info + members, ready to fetch."""
    info: SkillSetInfo
    members: List[SkillSetMember] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)   # entries skipped w/ reason


# ---------------------------------------------------------------------------
# HTTP (indirection point — tests monkeypatch _http_fetch)
# ---------------------------------------------------------------------------

@dataclass
class FetchResult:
    """Raw fetch result carrying what URL resolution + format detection need."""
    url: str            # FINAL URL after redirects (RFC 3986 base for children)
    content: bytes
    content_type: str   # Content-Type header value ("" when absent)


def _http_fetch(url: str, *, timeout: int = 30) -> Optional[FetchResult]:
    """SSRF-guarded GET returning content + final URL + Content-Type.

    The final (post-redirect) URL matters: #254 resolves relative member
    URLs against the URL the index was *actually retrieved from*, so an
    index that redirects to a CDN must have its members resolved against
    the CDN location, not the original well-known path.
    """
    from tools.skills_hub import _guarded_http_get
    resp = _guarded_http_get(url, timeout=timeout)
    if resp is None or resp.status_code != 200:
        return None
    # _guarded_http_get follows redirects manually (follow_redirects=False),
    # so the response's request URL IS the final hop.
    try:
        final_url = str(resp.url) or url
    except Exception:
        final_url = url
    content_type = ""
    try:
        content_type = resp.headers.get("content-type", "") or ""
    except Exception:
        pass
    return FetchResult(url=final_url, content=resp.content,
                       content_type=content_type)


def _http_get_bytes(url: str, *, timeout: int = 30) -> Optional[bytes]:
    """Bytes-only convenience wrapper over :func:`_http_fetch`."""
    result = _http_fetch(url, timeout=timeout)
    return None if result is None else result.content


def _fetch_json(url: str, *, timeout: int = 20) -> Optional[tuple]:
    """Fetch and parse JSON. Returns ``(data, final_url)`` or None."""
    result = _http_fetch(url, timeout=timeout)
    if result is None:
        return None
    try:
        return json.loads(result.content.decode("utf-8")), result.url
    except (json.JSONDecodeError, UnicodeDecodeError):
        logger.warning("Skill-set fetch: invalid JSON at %s", url)
        return None


# ---------------------------------------------------------------------------
# Digest verification
# ---------------------------------------------------------------------------

def verify_digest(content: bytes, digest: str, *, what: str = "artifact") -> None:
    """Verify ``content`` against a ``sha256:<hex>`` digest string.

    Raises :class:`DigestError` on missing/malformed/mismatched digests —
    #254 makes the digest required and forbids using unverified content.
    """
    if not isinstance(digest, str) or not digest:
        raise DigestError(f"{what}: missing digest (required by the discovery spec)")
    m = _DIGEST_RE.match(digest.strip())
    if not m:
        raise DigestError(
            f"{what}: malformed digest {digest!r} (expected sha256:<64 lowercase hex>)"
        )
    actual = hashlib.sha256(content).hexdigest()
    if actual != m.group(1):
        raise DigestError(
            f"{what}: digest mismatch — index says sha256:{m.group(1)}, "
            f"downloaded content is sha256:{actual}. Refusing to use it."
        )


def compute_digest(content: bytes) -> str:
    """Format a #254 digest string for raw bytes (publisher-side helper)."""
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


# ---------------------------------------------------------------------------
# AI Catalog parsing
# ---------------------------------------------------------------------------

def catalog_url_for(origin_or_url: str) -> str:
    """Normalize user input to an AI Catalog URL.

    ``https://example.com`` -> ``https://example.com/.well-known/ai-catalog.json``;
    anything already ending in ``.json`` is returned as-is.
    """
    url = origin_or_url.strip().rstrip("/")
    if url.endswith(".json"):
        return origin_or_url.strip()
    return url + AI_CATALOG_WELL_KNOWN


def _entry_extensions(entry: Dict[str, Any]) -> Dict[str, Any]:
    ext = entry.get("extensions")
    return ext if isinstance(ext, dict) else {}


def discover_skill_sets(catalog_url: str, *, _depth: int = 0,
                        _inline_catalog: Optional[Dict[str, Any]] = None,
                        ) -> List[SkillSetInfo]:
    """Fetch an AI Catalog and return every skill-set entry in it.

    Follows ``application/ai-catalog+json`` sub-catalog entries one level
    deep (AI Catalog is nestable; the prototype bounds recursion at 1).
    Entries may carry their payload via ``url`` or inline ``data`` — the
    AI Catalog spec allows either (exactly one per entry).
    """
    if _inline_catalog is not None:
        data: Any = _inline_catalog
        base_url = catalog_url
    else:
        fetched = _fetch_json(catalog_url)
        if fetched is None:
            raise SkillSetError(f"Could not fetch AI Catalog at {catalog_url}")
        data, base_url = fetched
    if not isinstance(data, dict):
        raise SkillSetError(f"Could not fetch AI Catalog at {catalog_url}")

    entries = data.get("entries")
    if not isinstance(entries, list):
        raise SkillSetError(
            f"{catalog_url} is not an AI Catalog (no 'entries' array)"
        )

    sets: List[SkillSetInfo] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        etype = entry.get("type")
        url = entry.get("url")
        inline = entry.get("data")
        has_url = isinstance(url, str) and bool(url)
        has_data = isinstance(inline, dict)
        if not has_url and not has_data:
            continue
        # RFC 3986: resolve against the URL the catalog was actually
        # retrieved from (post-redirect), not the URL we asked for.
        resolved = urljoin(base_url, url) if has_url else ""

        if etype == SKILL_SET_ENTRY_TYPE:
            hermes_ext = _entry_extensions(entry).get(HERMES_SET_EXTENSION)
            hermes_ext = hermes_ext if isinstance(hermes_ext, dict) else {}
            display = str(
                entry.get("displayName") or entry.get("identifier")
                or resolved or "inline skill set"
            )
            sets.append(SkillSetInfo(
                name=display,
                description=str(entry.get("description") or ""),
                index_url=resolved or f"{base_url}#inline",
                identifier=str(entry.get("identifier") or ""),
                command=str(hermes_ext.get("command") or ""),
                instruction=str(hermes_ext.get("instruction") or ""),
                catalog_url=catalog_url,
                inline_index=inline if has_data and not has_url else None,
                base_url=base_url,
            ))
        elif etype == CATALOG_ENTRY_TYPE and _depth < 1:
            try:
                if has_url:
                    sets.extend(discover_skill_sets(resolved, _depth=_depth + 1))
                else:
                    # Inline sub-catalog: relative URLs inside it resolve
                    # against the parent catalog's retrieved location.
                    sets.extend(discover_skill_sets(
                        base_url, _depth=_depth + 1, _inline_catalog=inline,
                    ))
            except SkillSetError as exc:
                logger.warning("Skipping unreadable sub-catalog %s: %s",
                               resolved or "(inline)", exc)

    return sets


# ---------------------------------------------------------------------------
# #254 discovery index parsing
# ---------------------------------------------------------------------------

def resolve_skill_set(info: SkillSetInfo) -> ResolvedSkillSet:
    """Fetch and validate the #254 index a :class:`SkillSetInfo` points at.

    Member URLs are resolved per RFC 3986 against the URL the index was
    *actually retrieved from* — if the index URL redirects (e.g. to a CDN),
    relative member URLs resolve against the CDN location, per #254.
    """
    if info.inline_index is not None:
        data: Any = info.inline_index
        # Inline data was transported inside the catalog document, so its
        # relative URLs resolve against the catalog's retrieved location.
        member_base = info.base_url or info.catalog_url or info.index_url
    else:
        fetched = _fetch_json(info.index_url)
        if fetched is None:
            raise SkillSetError(f"Could not fetch skill index at {info.index_url}")
        data, member_base = fetched
    if not isinstance(data, dict):
        raise SkillSetError(f"Could not fetch skill index at {info.index_url}")

    schema = data.get("$schema")
    if not isinstance(schema, str) or schema not in KNOWN_INDEX_SCHEMAS:
        # #254: "Clients encountering an unrecognized or absent $schema
        # should warn the user and should not process the index."
        raise SchemaError(
            f"Unrecognized index $schema {schema!r} at {info.index_url}. "
            f"Known: {', '.join(sorted(KNOWN_INDEX_SCHEMAS))}"
        )

    raw_skills = data.get("skills")
    if not isinstance(raw_skills, list):
        raise SkillSetError(f"Index at {info.index_url} has no 'skills' array")
    if len(raw_skills) > MAX_SET_MEMBERS:
        raise SkillSetError(
            f"Index enumerates {len(raw_skills)} skills (cap: {MAX_SET_MEMBERS})"
        )

    members: List[SkillSetMember] = []
    skipped: List[str] = []
    for entry in raw_skills:
        if not isinstance(entry, dict):
            skipped.append("(non-object entry)")
            continue
        name = entry.get("name")
        etype = entry.get("type")
        url = entry.get("url")
        digest = entry.get("digest")
        if not isinstance(name, str) or not name:
            skipped.append("(entry without a name)")
            continue
        if etype not in ("skill-md", "archive"):
            # #254: skip entries with an unrecognized type and warn.
            skipped.append(f"{name} (unrecognized type {etype!r})")
            continue
        if not isinstance(url, str) or not url:
            skipped.append(f"{name} (missing url)")
            continue
        members.append(SkillSetMember(
            name=name,
            description=str(entry.get("description") or ""),
            type=etype,
            url=urljoin(member_base, url),
            digest=str(digest or ""),
        ))

    return ResolvedSkillSet(info=info, members=members, skipped=skipped)


def resolve_bare_index(index_url: str, *, name: str = "") -> ResolvedSkillSet:
    """Treat a raw #254 index URL as a set (no AI Catalog wrapper).

    Used when the user hands us an index.json directly. The set name falls
    back to the host name; no extension metadata is available on this path.
    """
    host = urlparse(index_url).netloc or index_url
    info = SkillSetInfo(
        name=name or host,
        description=f"Skill set from {host}",
        index_url=index_url,
    )
    return resolve_skill_set(info)


# ---------------------------------------------------------------------------
# Member fetching (skill-md + archive) -> SkillBundle
# ---------------------------------------------------------------------------

def _safe_member_path(raw_name: str) -> str:
    """Validate an archive member path per #254 archive-safety rules."""
    name = raw_name.replace("\\", "/")
    if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        raise ArchiveSafetyError(f"absolute path in archive: {raw_name!r}")
    normalized = posixpath.normpath(name)
    if normalized.startswith("..") or "/../" in f"/{normalized}/":
        raise ArchiveSafetyError(f"path traversal in archive: {raw_name!r}")
    if normalized in (".", ""):
        raise ArchiveSafetyError(f"empty archive member path: {raw_name!r}")
    # Reuse the hub-wide validator for charset / depth constraints.
    from tools.skills_hub import _validate_bundle_rel_path
    return _validate_bundle_rel_path(normalized)


def _decode_member(raw: bytes) -> Union[str, bytes]:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw


def _extract_tar_gz(content: bytes) -> Dict[str, Union[str, bytes]]:
    files: Dict[str, Union[str, bytes]] = {}
    total = 0
    try:
        tf = tarfile.open(fileobj=io.BytesIO(content), mode="r:gz")
    except tarfile.TarError as exc:
        raise ArchiveSafetyError(f"invalid .tar.gz archive: {exc}") from exc
    with tf:
        members = tf.getmembers()
        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise ArchiveSafetyError(
                f"archive has {len(members)} members (cap: {MAX_ARCHIVE_MEMBERS})"
            )
        for m in members:
            if m.issym() or m.islnk():
                # #254: reject symlinks/hardlinks outright (resolving them
                # safely is not worth it for skill payloads).
                raise ArchiveSafetyError(f"link member in archive: {m.name!r}")
            if not m.isfile():
                continue
            safe = _safe_member_path(m.name)
            if m.size > MAX_MEMBER_BYTES:
                raise ArchiveSafetyError(
                    f"archive member {safe} is {m.size} bytes (cap: {MAX_MEMBER_BYTES})"
                )
            total += m.size
            if total > MAX_UNPACKED_BYTES:
                raise ArchiveSafetyError("archive exceeds decompressed size cap")
            fh = tf.extractfile(m)
            if fh is None:
                continue
            raw = fh.read(MAX_MEMBER_BYTES + 1)
            if len(raw) > MAX_MEMBER_BYTES:
                raise ArchiveSafetyError(f"archive member {safe} exceeded size cap")
            files[safe] = _decode_member(raw)
    return files


def _extract_zip(content: bytes) -> Dict[str, Union[str, bytes]]:
    files: Dict[str, Union[str, bytes]] = {}
    total = 0
    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise ArchiveSafetyError(f"invalid .zip archive: {exc}") from exc
    with zf:
        infos = zf.infolist()
        if len(infos) > MAX_ARCHIVE_MEMBERS:
            raise ArchiveSafetyError(
                f"archive has {len(infos)} members (cap: {MAX_ARCHIVE_MEMBERS})"
            )
        for zinfo in infos:
            if zinfo.is_dir():
                continue
            # Zip symlinks encode the link mode in external_attr's high bits.
            if (zinfo.external_attr >> 16) & 0o170000 == 0o120000:
                raise ArchiveSafetyError(f"symlink member in archive: {zinfo.filename!r}")
            safe = _safe_member_path(zinfo.filename)
            if zinfo.file_size > MAX_MEMBER_BYTES:
                raise ArchiveSafetyError(
                    f"archive member {safe} is {zinfo.file_size} bytes "
                    f"(cap: {MAX_MEMBER_BYTES})"
                )
            total += zinfo.file_size
            if total > MAX_UNPACKED_BYTES:
                raise ArchiveSafetyError("archive exceeds decompressed size cap")
            with zf.open(zinfo) as fh:
                raw = fh.read(MAX_MEMBER_BYTES + 1)
            if len(raw) > MAX_MEMBER_BYTES:
                raise ArchiveSafetyError(f"archive member {safe} exceeded size cap")
            files[safe] = _decode_member(raw)
    return files


#: Content-Type values mapped to archive formats (#254: header wins; the
#: URL extension is only a fallback for absent/generic headers).
_ARCHIVE_CONTENT_TYPES = {
    "application/gzip": "tar.gz",
    "application/x-gzip": "tar.gz",
    "application/x-tar+gzip": "tar.gz",
    "application/zip": "zip",
    "application/x-zip-compressed": "zip",
}
_GENERIC_CONTENT_TYPES = frozenset({
    "", "application/octet-stream", "binary/octet-stream",
    "application/binary", "text/plain",
})


def _archive_format(url: str, content_type: str = "") -> str:
    """Determine archive format: Content-Type header first, then extension.

    Per #254: "Clients should determine the archive format from the
    server's Content-Type header, falling back to the URL file extension
    if the header is absent or generic."
    """
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type not in _GENERIC_CONTENT_TYPES:
        fmt = _ARCHIVE_CONTENT_TYPES.get(media_type)
        if fmt:
            return fmt
        # Specific but unknown media type — fall through to the extension
        # rather than hard-failing on an exotic-but-honest server config.
        logger.debug("Unrecognized archive Content-Type %r for %s; "
                     "falling back to URL extension", content_type, url)
    path = urlparse(url).path.lower()
    if path.endswith((".tar.gz", ".tgz")):
        return "tar.gz"
    if path.endswith(".zip"):
        return "zip"
    raise ArchiveSafetyError(
        f"unsupported archive format for {url} "
        f"(Content-Type {content_type or '(none)'!r}; expected .tar.gz or .zip)"
    )


def fetch_member(member: SkillSetMember, *, set_info: SkillSetInfo):
    """Download, digest-verify, and unpack one set member.

    Returns a ``tools.skills_hub.SkillBundle`` ready for the standard
    quarantine -> scan -> install pipeline.
    """
    from tools.skills_hub import SkillBundle, _validate_skill_name

    skill_name = _validate_skill_name(member.name)

    fetched = _http_fetch(member.url)
    if fetched is None:
        raise SkillSetError(f"{member.name}: could not download {member.url}")
    content = fetched.content
    if len(content) > MAX_ARCHIVE_BYTES:
        raise ArchiveSafetyError(
            f"{member.name}: artifact is {len(content)} bytes (cap: {MAX_ARCHIVE_BYTES})"
        )

    verify_digest(content, member.digest, what=member.name)

    if member.type == "skill-md":
        try:
            files: Dict[str, Union[str, bytes]] = {
                "SKILL.md": content.decode("utf-8")
            }
        except UnicodeDecodeError as exc:
            raise SkillSetError(f"{member.name}: SKILL.md is not valid UTF-8") from exc
    else:
        # Format detection uses the FINAL (post-redirect) URL for the
        # extension fallback, and the response Content-Type first.
        fmt = _archive_format(fetched.url, fetched.content_type)
        files = _extract_tar_gz(content) if fmt == "tar.gz" else _extract_zip(content)
        if "SKILL.md" not in files:
            # #254: "The archive must contain SKILL.md at the root."
            raise ArchiveSafetyError(
                f"{member.name}: archive has no SKILL.md at its root"
            )

    return SkillBundle(
        name=skill_name,
        files=files,
        source="skill-set",
        identifier=f"skill-set:{set_info.index_url}#{skill_name}",
        trust_level="community",
        metadata={
            "set_name": set_info.name,
            "index_url": set_info.index_url,
            "catalog_url": set_info.catalog_url,
            "artifact_url": member.url,
            "artifact_type": member.type,
            "digest": member.digest,
            "source_url": member.url,
        },
    )
