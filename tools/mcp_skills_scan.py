"""Skills Guard policy at remote MCP skill exposure boundaries."""
from __future__ import annotations

import hashlib
import io
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlsplit


class RemoteSkillSecurityError(ValueError):
    """A fixed, non-content-bearing remote skill quarantine error."""

    def __init__(self, boundary: str, *, attestation: dict[str, Any] | None = None):
        super().__init__(f"Remote MCP skill {boundary} was blocked by Skills Guard")
        self.boundary = boundary
        self.attestation = attestation or {}


def _validate_docx_package(raw: bytes, *, source: str, digest: str) -> None:
    """Reject malformed or active OOXML without claiming content safety."""
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            infos = archive.infolist()
            if not infos or len(infos) > 2048:
                raise ValueError("invalid member count")
            names: set[str] = set()
            total_size = 0
            for info in infos:
                name = info.filename.replace("\\", "/")
                parts = PurePosixPath(name).parts
                if (name.startswith("/") or any(part in {"", ".", ".."} for part in parts)
                        or name in names or info.flag_bits & 0x1):
                    raise ValueError("unsafe package member")
                names.add(name)
                total_size += info.file_size
                if info.file_size > 64 * 1024 * 1024 or total_size > 256 * 1024 * 1024:
                    raise ValueError("package expansion limit")
                if info.file_size > 1024 * 1024 and info.compress_size > 0:
                    if info.file_size / info.compress_size > 1000:
                        raise ValueError("package compression ratio")
                lowered = name.casefold()
                if (lowered.endswith("vbaproject.bin") or "/embeddings/" in lowered
                        or "/oleobjects/" in lowered or lowered.endswith("activex.bin")):
                    raise ValueError("active package content")
            if "[Content_Types].xml" not in names or "word/document.xml" not in names:
                raise ValueError("missing required OOXML members")
            content_types = ET.fromstring(archive.read("[Content_Types].xml"))
            document = ET.fromstring(archive.read("word/document.xml"))
            package_ns = "http://schemas.openxmlformats.org/package/2006/content-types"
            word_ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            if content_types.tag != f"{{{package_ns}}}Types" or document.tag != f"{{{word_ns}}}document":
                raise ValueError("invalid OOXML roots")
            overrides = {
                (node.get("PartName"), node.get("ContentType"))
                for node in content_types.findall(f"{{{package_ns}}}Override")
            }
            expected_document_type = (
                "/word/document.xml",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml",
            )
            if expected_document_type not in overrides:
                raise ValueError("missing Word document content type")
            # Force CRC/decompression validation within the same aggregate cap.
            for info in infos:
                if not info.is_dir():
                    archive.read(info)
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError) as exc:
        try:
            from tools.skills_guard import SCANNER_VERSION
        except Exception:
            SCANNER_VERSION = "unavailable"
        raise RemoteSkillSecurityError("resource", attestation={
            "scanner_version": SCANNER_VERSION, "source": source,
            "content_hash": digest, "scope": "docx_package_structure",
            "verdict": "invalid_binary_package", "allowed": False,
            "finding_count": 0, "rules": [],
        }) from exc


def _identity(server: str, config_fingerprint: str, skill_uri: str,
              resource_uri: str = "") -> str:
    raw = f"mcp\0{server}\0{config_fingerprint}\0{skill_uri}\0{resource_uri}"
    return "mcp:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _scan_text(raw: bytes, *, filename: str, source: str, scope: str,
               boundary: str) -> dict[str, Any]:
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8", errors="strict")
        from tools.skills_guard import SCANNER_VERSION, scan_skill, should_allow_install
        from tools.skills_guard import SCANNABLE_EXTENSIONS
        suffix = PurePosixPath(filename).suffix.lower()
        scan_name = ("SKILL.md" if PurePosixPath(filename).name == "SKILL.md"
                     else f"resource{suffix if suffix in SCANNABLE_EXTENSIONS else '.txt'}")
        with tempfile.TemporaryDirectory(prefix="hermes-mcp-skill-scan-") as tmp:
            path = Path(tmp) / scan_name
            path.write_text(text, encoding="utf-8")
            result = scan_skill(path, source=source)
        allowed, _reason = should_allow_install(result)
        attestation = {
            "scanner_version": SCANNER_VERSION,
            "source": source,
            "content_hash": digest,
            "scope": scope,
            "verdict": result.verdict,
            "trust_level": result.trust_level,
            "allowed": allowed is True,
            "finding_count": len(result.findings),
            "rules": sorted({finding.pattern_id for finding in result.findings}),
        }
        if allowed is not True:
            raise RemoteSkillSecurityError(boundary, attestation=attestation)
        return attestation
    except RemoteSkillSecurityError:
        raise
    except Exception as exc:
        # Scanner/import/decode/temp-I/O failures are security failures. Do not
        # carry exception text: it may contain remote bytes or scanner matches.
        try:
            from tools.skills_guard import SCANNER_VERSION
        except Exception:
            SCANNER_VERSION = "unavailable"
        raise RemoteSkillSecurityError(boundary, attestation={
            "scanner_version": SCANNER_VERSION,
            "source": source,
            "content_hash": digest,
            "scope": scope,
            "verdict": "scanner_error",
            "allowed": False,
            "finding_count": 0,
            "rules": [],
        }) from exc


def _metadata_lines(value: Any, prefix: str) -> list[str]:
    """Render scalar keys/values without JSON newline escaping."""
    if isinstance(value, dict):
        lines: list[str] = []
        for key in sorted(value, key=lambda item: str(item)):
            key_text = str(key)
            lines.extend((f"{prefix}.key", key_text))
            lines.extend(_metadata_lines(value[key], f"{prefix}.{key_text}"))
        return lines
    if isinstance(value, (list, tuple)):
        lines = []
        for index, item in enumerate(value):
            lines.extend(_metadata_lines(item, f"{prefix}[{index}]"))
        return lines
    if value is None:
        return []
    # Prefix and value are separate lines. Embedded newlines remain real lines,
    # so prompt-injection phrases cannot disappear behind JSON escaping.
    return [prefix, str(value)]


def scan_catalog_metadata(entry: Any, *, server: str,
                          config_fingerprint: str) -> dict[str, Any]:
    item = entry.model_dump(mode="json") if hasattr(entry, "model_dump") else dict(entry)
    skill_uri = str(item.get("uri") or "")
    rendered: list[str] = ["skill.uri", skill_uri, "skill.path", unquote(urlsplit(skill_uri).path)]
    rendered.extend(_metadata_lines(item.get("frontmatter") or {}, "frontmatter"))
    for index, resource in enumerate(item.get("resources") or []):
        if isinstance(resource, dict):
            resource_uri = str(resource.get("uri") or "")
            rendered.extend((f"resources[{index}].uri", resource_uri,
                             f"resources[{index}].path", unquote(urlsplit(resource_uri).path)))
    raw = ("\n".join(rendered) + "\n").encode("utf-8")
    return _scan_text(
        raw, filename="catalog-metadata.txt",
        source=_identity(server, config_fingerprint, skill_uri),
        scope="catalog_model_visible_metadata", boundary="metadata")


def scan_resource_bytes(raw: bytes, *, record: dict[str, Any],
                        resource: dict[str, Any], is_text: bool,
                        mime_type: str = "") -> dict[str, Any]:
    source = _identity(record["server"], record["config_fingerprint"],
                       record["uri"], resource["uri"])
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    filename = PurePosixPath(unquote(urlsplit(resource["uri"]).path)).name or "resource.bin"
    suffix = PurePosixPath(filename).suffix.lower()
    from tools.skills_guard import SCANNABLE_EXTENSIONS
    base_mime = str(mime_type or "").split(";", 1)[0].strip().lower()
    text_mimes = {"application/json", "application/xml", "application/javascript",
                  "application/yaml", "application/x-yaml", "application/toml"}
    valid_utf8 = False
    try:
        raw.decode("utf-8", errors="strict")
        valid_utf8 = True
    except UnicodeDecodeError:
        pass
    if suffix == ".docx":
        _validate_docx_package(raw, source=source, digest=digest)
        effective_text = False
    else:
        effective_text = (is_text or filename == "SKILL.md" or suffix in SCANNABLE_EXTENSIONS
                          or base_mime.startswith("text/") or base_mime in text_mimes
                          or valid_utf8)
    try:
        from tools.skills_guard import SCANNER_VERSION, scan_skill, should_allow_install
        with tempfile.TemporaryDirectory(prefix="hermes-mcp-skill-structure-") as tmp:
            root = Path(tmp) / "resource"
            root.mkdir()
            (root / filename).write_bytes(raw)
            structure = scan_skill(root, source=source)
        allowed, _reason = should_allow_install(structure)
        structure_rules = sorted({finding.pattern_id for finding in structure.findings})
        if allowed is not True:
            raise RemoteSkillSecurityError("resource", attestation={
                "scanner_version": SCANNER_VERSION, "source": source,
                "content_hash": digest, "scope": "single_resource_structure",
                "verdict": structure.verdict, "allowed": False,
                "finding_count": len(structure.findings), "rules": structure_rules,
            })
    except RemoteSkillSecurityError:
        raise
    except Exception as exc:
        try:
            from tools.skills_guard import SCANNER_VERSION
        except Exception:
            SCANNER_VERSION = "unavailable"
        raise RemoteSkillSecurityError("resource", attestation={
            "scanner_version": SCANNER_VERSION, "source": source,
            "content_hash": digest, "scope": "single_resource_structure",
            "verdict": "scanner_error", "allowed": False,
            "finding_count": 0, "rules": [],
        }) from exc
    if effective_text:
        attestation = _scan_text(raw, filename=filename, source=source,
                                 scope="single_resource_text", boundary="resource")
        attestation["structural_rules"] = structure_rules
        return attestation
    return {
        "scanner_version": SCANNER_VERSION, "source": source,
        "content_hash": digest, "scope": "binary_structure_only",
        "verdict": "not_scanned_binary", "structural_verdict": structure.verdict,
        "allowed": None, "finding_count": len(structure.findings),
        "rules": structure_rules,
    }


def scan_resource_listing_metadata(values: dict[str, Any], *, record: dict[str, Any],
                                   resource: dict[str, Any]) -> dict[str, Any]:
    rendered = _metadata_lines(values, "resource_listing")
    raw = ("\n".join(rendered) + "\n").encode("utf-8")
    return _scan_text(
        raw, filename="resource-listing.txt",
        source=_identity(record["server"], record["config_fingerprint"],
                         record["uri"], resource["uri"]),
        scope="catalog_resource_listing_metadata", boundary="resource metadata")
