"""Skills extension metadata and byte verification; never preprocess remote instructions."""

import base64
import hashlib
import json
import re
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit

import hermes_yaml as yaml

from tools.mcp_tool_common import mcp_field

EXTENSION = "io.modelcontextprotocol/skills"
MAX_RESOURCES = 512
MAX_BYTES = 16 * 1024 * 1024


class SkillVerificationError(ValueError):
    """Remote content cannot be used under the supplied entry."""


def supports_skills(server) -> bool:
    caps = getattr(getattr(server, "initialize_result", None), "capabilities", None)
    extensions = getattr(caps, "extensions", None)
    return (getattr(caps, "resources", None) is not None and isinstance(extensions, dict)
            and isinstance(extensions.get(EXTENSION), dict))


def _valid_uri(uri: str) -> bool:
    if not isinstance(uri, str) or not uri or re.search(r"[\x00-\x20\\]", uri):
        return False
    parsed = urlsplit(uri)
    decoded = unquote(uri)
    return (bool(parsed.scheme) and not parsed.query and not parsed.fragment
            and not re.search(r"%(?![a-fA-F0-9]{2})", uri)
            and not re.search(r"[\x00-\x1f\x7f\\]", decoded)
            and all(p not in (".", "..") for p in decoded.split("/")))


@dataclass(frozen=True)
class SkillEntry:
    """Canonical JSON keeps held entries independent of mutable server/SDK objects."""

    data: str

    @property
    def value(self) -> dict:
        return json.loads(self.data)

    @property
    def uri(self) -> str:
        return self.value["uri"]

    @property
    def root(self) -> str:
        return self.uri.rsplit("/", 1)[0] + "/"

    @property
    def revision(self) -> str:
        return hashlib.sha256(self.data.encode("utf-8")).hexdigest()

    def resource(self, uri: str) -> dict:
        for resource in self.value["resources"]:
            if resource["uri"] == uri:
                return resource
        raise SkillVerificationError("File is outside the retained skill manifest; reload the skill to refresh it.")

    def resolve(self, relative: str) -> str:
        if not isinstance(relative, str) or relative.startswith(("/", "\\")) or ":" in relative:
            raise SkillVerificationError("Use a relative path within this skill.")
        uri = self.root + relative
        if not _valid_uri(uri):
            raise SkillVerificationError("Invalid skill file path.")
        return uri


def parse_entry(raw: dict) -> SkillEntry:
    if not isinstance(raw, dict) or not _valid_uri(raw.get("uri")):
        raise SkillVerificationError("Invalid skill URI.")
    uri, fm = raw["uri"], raw.get("frontmatter")
    if not isinstance(fm, dict) or not all(isinstance(fm.get(k), str) and fm[k] for k in ("name", "description")):
        raise SkillVerificationError("Skill frontmatter needs a name and description.")
    root = uri.rsplit("/", 1)[0] + "/"
    if not uri.endswith("/SKILL.md") or unquote(root.rstrip("/").rsplit("/", 1)[-1]) != fm["name"]:
        raise SkillVerificationError("Skill name must match its parent directory.")
    resources = raw.get("resources")
    if resources == "dynamic":
        raise SkillVerificationError("Dynamic MCP skills are not supported: a stable manifest is required.")
    if not isinstance(resources, list) or not resources or len(resources) > MAX_RESOURCES:
        raise SkillVerificationError(f"Skill must have a manifest with at most {MAX_RESOURCES} files.")
    seen, size = set(), 0
    for resource in resources:
        size += _validate_resource(resource, root, seen)
    if uri not in seen or size > MAX_BYTES:
        raise SkillVerificationError("Manifest must include SKILL.md and total at most 16 MiB.")
    # Resource ordering and optional transport metadata do not change a content-bound grant.
    entry = {"uri": uri, "frontmatter": fm, "resources": sorted(resources, key=lambda r: r["uri"])}
    return SkillEntry(json.dumps(entry, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def _validate_resource(resource, root: str, seen: set) -> int:
    if not isinstance(resource, dict):
        raise SkillVerificationError("Invalid manifest resource.")
    uri, digest, size = (resource.get(k) for k in ("uri", "digest", "size"))
    if not _valid_uri(uri) or not uri.startswith(root) or uri in seen or uri.endswith("/"):
        raise SkillVerificationError("Manifest files must be unique and within the skill directory.")
    if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise SkillVerificationError("Manifest requires SHA-256 digests.")
    if type(size) is not int or size < 0:
        raise SkillVerificationError("Manifest requires nonnegative byte sizes.")
    seen.add(uri)
    return size


async def send(session, method: str, params: dict) -> dict:
    """The SDK adds the negotiated per-request metadata, including on extension requests."""
    from mcp import types
    from pydantic import TypeAdapter
    return await session.send_request(types.Request[dict, str](method=method, params=params), TypeAdapter(dict))


async def read_verified(session, entry: SkillEntry, uri: str) -> dict:
    resource = entry.resource(uri)  # no I/O for unlisted files
    result = await session.read_resource(uri)
    contents = getattr(result, "contents", [])
    if len(contents) != 1 or str(contents[0].uri) != uri:
        raise SkillVerificationError("Skill read must return exactly the requested file.")
    block = contents[0]
    text, blob = getattr(block, "text", None), getattr(block, "blob", None)
    if isinstance(text, str):
        data = text.encode("utf-8")
    elif isinstance(blob, str):
        data = base64.b64decode(blob, validate=True)
    else:
        raise SkillVerificationError("Skill resource contains neither text nor binary content.")
    if len(data) != resource["size"] or "sha256:" + hashlib.sha256(data).hexdigest() != resource["digest"]:
        raise SkillVerificationError("Skill size or digest mismatch; reload the skill to refresh its manifest.")
    if uri == entry.uri:
        text = data.decode("utf-8")
        _verify_frontmatter(text, entry.value["frontmatter"])
    return ({"content": text} if isinstance(text, str) else
            {"blob": blob, "mime_type": mcp_field(block, "mime_type", "mimeType"), "is_binary": True})


def _verify_frontmatter(text: str, expected: dict) -> None:
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        raise SkillVerificationError("SKILL.md must begin with YAML frontmatter.")
    try:
        end = lines.index("---", 1)
        actual = yaml.safe_load("\n".join(lines[1:end]))
        canonical = json.dumps(actual, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except (ValueError, TypeError, yaml.YAMLError) as exc:
        raise SkillVerificationError("Invalid SKILL.md frontmatter.") from exc
    if canonical != json.dumps(expected, sort_keys=True, ensure_ascii=False, separators=(",", ":")):
        raise SkillVerificationError("SKILL.md frontmatter differs from the skill entry.")
