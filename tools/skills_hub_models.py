"""Skills Hub data models, path validators, and SKILL.md helpers. Leaf module (no imports from
tools.skills_hub) so every source adapter module can import it at top level without cycles."""

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Callable, Dict, Iterable, List, Optional, Union
from urllib.parse import unquote, urlsplit

import httpx
import yaml

logger = logging.getLogger("tools.skills_hub")


def hub():
    """``tools.skills_hub`` resolved at call time: its cache / HTTP / index helpers are the test-patch targets
    (``patch("tools.skills_hub._read_index_cache")`` ...), so adapters look them up on every call, not at import."""
    import tools.skills_hub as mod
    return mod


@dataclass
class SkillMeta:
    """Minimal metadata returned by search results."""
    name: str
    description: str
    source: str           # "official", "github", "clawhub", "lobehub"
    identifier: str       # source-specific ID (e.g. "openai/skills/skill-creator")
    trust_level: str      # "builtin" | "trusted" | "community"
    repo: Optional[str] = None
    path: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillBundle:
    """A downloaded skill ready for quarantine/scanning/installation."""
    name: str
    files: Dict[str, Union[str, bytes]]   # relative_path -> file content
    source: str
    identifier: str
    trust_level: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def _skill_meta_to_dict(meta: SkillMeta) -> dict:
    return dict(vars(meta))


def _cached_metas(key: str) -> Optional[List[SkillMeta]]:
    """SkillMeta list from the shared index cache, or None on miss/expiry."""
    cached = hub()._read_index_cache(key)
    return None if cached is None else [SkillMeta(**item) for item in cached]


def _cache_metas(key: str, metas: List[SkillMeta]) -> None:
    hub()._write_index_cache(key, [_skill_meta_to_dict(m) for m in metas])


def _memo_json(key: str, compute: Callable[[], Any], valid: Callable[[Any], bool] = lambda c: c is not None) -> Any:
    """Shared-index-cache memo: a cached value passing ``valid`` is returned as-is; otherwise ``compute()``
    runs and a non-None result is written back."""
    cached = hub()._read_index_cache(key)
    if valid(cached):
        return cached
    data = compute()
    if data is not None:
        hub()._write_index_cache(key, data)
    return data


def _get_json(url: str, *, timeout: int = 20, **kwargs) -> Optional[Any]:
    """Plain (unguarded) GET + JSON decode; None on non-200 or transport/decode error."""
    try:
        resp = httpx.get(url, timeout=timeout, **kwargs)
        return resp.json() if resp.status_code == 200 else None
    except (httpx.HTTPError, json.JSONDecodeError):
        return None


def _get_text(url: str, *, timeout: int = 20, **kwargs) -> Optional[str]:
    """Plain (unguarded) GET; body text on 200, None on any other status or transport error."""
    try:
        resp = httpx.get(url, timeout=timeout, **kwargs)
    except httpx.HTTPError:
        return None
    return resp.text if resp.status_code == 200 else None


def _matches_query(query_lower: str, *fields: Any) -> bool:
    """Case-insensitive substring match against joined fields (lists space-joined; empty query matches all)."""
    parts = [" ".join(str(t) for t in f) if isinstance(f, list) else str(f) for f in fields]
    return query_lower in " ".join(parts).lower()


def _first_matching(query_lower: str, items: Iterable[Any], fields_of: Callable[[Any], tuple],
                    to_meta: Callable[[Any], Optional[SkillMeta]], limit: int) -> List[SkillMeta]:
    """Substring-search ``items`` in order, converting hits with ``to_meta`` until ``limit``."""
    results: List[SkillMeta] = []
    for item in items:
        if _matches_query(query_lower, *fields_of(item)) and (meta := to_meta(item)):
            results.append(meta)
        if len(results) >= limit:
            break
    return results


TRUST_RANK = {"builtin": 2, "trusted": 1, "community": 0}


def _dedupe_by_trust(results: Iterable[SkillMeta]) -> List[SkillMeta]:
    """Dedupe by identifier, keeping the higher-trust copy (first wins on ties). identifier is unique per
    skill; name is not — two taps can publish same-named skills, and browse-sh reuses task names across sites."""
    seen: Dict[str, SkillMeta] = {}
    for r in results:
        kept = seen.get(r.identifier)
        if kept is None or TRUST_RANK.get(r.trust_level, 0) > TRUST_RANK.get(kept.trust_level, 0):
            seen[r.identifier] = r
    return list(seen.values())


class SkillSource(ABC):
    """Abstract base for all skill registry adapters. ``SOURCE_ID`` is the unique source id (e.g. 'github',
    'clawhub'); ``TRUST_LEVEL`` the trust every identifier gets unless ``trust_level_for`` is overridden."""

    SOURCE_ID: str = ""
    TRUST_LEVEL: str = "community"

    @abstractmethod
    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        """Search for skills matching a query string."""

    @abstractmethod
    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        """Download a skill bundle by identifier."""

    @abstractmethod
    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        """Fetch metadata for a skill without downloading all files."""

    def source_id(self) -> str:
        return self.SOURCE_ID

    def trust_level_for(self, identifier: str) -> str:
        return self.TRUST_LEVEL


class GuardedFetchMixin:
    """SSRF/policy-guarded GETs, routed through ``tools.skills_hub`` (test-patchable)."""

    @staticmethod
    def _fetch_text(url: str) -> Optional[str]:
        resp = hub()._guarded_http_get(url, timeout=20)
        return resp.text if resp is not None and resp.status_code == 200 else None

    @staticmethod
    def _fetch_bytes(url: str) -> Optional[bytes]:
        resp = hub()._guarded_http_get(url, timeout=20)
        return resp.content if resp is not None and resp.status_code == 200 else None


# --- SKILL.md frontmatter ---------------------------------------------------
def _parse_frontmatter(content: str) -> dict:
    """Parse YAML frontmatter from SKILL.md content ({} when absent/invalid)."""
    content = content.lstrip("\ufeff")  # tolerate UTF-8 BOM (Windows editors)
    match = re.search(r'\n---\s*\n', content[3:]) if content.startswith("---") else None
    if not match:
        return {}
    try:
        parsed = yaml.safe_load(content[3:match.start() + 3])
        return parsed if isinstance(parsed, dict) else {}
    except yaml.YAMLError:
        return {}


def _hermes_tags(fm: dict) -> Any:
    """``metadata.hermes.tags`` from parsed frontmatter, or ``[]`` (unvalidated type)."""
    metadata = fm.get("metadata", {})
    hermes_meta = metadata.get("hermes", {}) if isinstance(metadata, dict) else None
    return hermes_meta.get("tags", []) if isinstance(hermes_meta, dict) else []


def source_url_for_bundle(bundle: SkillBundle) -> str:
    """Best available human-facing immutable-source provenance URL."""
    explicit = bundle.metadata.get("source_url") or bundle.metadata.get("url")
    if explicit:
        return str(explicit)
    if bundle.source == "github":
        parts = bundle.identifier.split("/", 2)
        if len(parts) >= 2:
            suffix = f"/tree/main/{parts[2]}" if len(parts) == 3 else ""
            return f"https://github.com/{parts[0]}/{parts[1]}{suffix}"
    return bundle.identifier


# --- Bundle path validation -------------------------------------------------
def _normalize_bundle_path(path_value: str, *, field_name: str, allow_nested: bool) -> str:
    """Normalize and validate bundle-controlled paths before touching disk."""
    if not isinstance(path_value, str):
        raise ValueError(f"Unsafe {field_name}: expected a string")
    raw = path_value.strip()
    if not raw:
        raise ValueError(f"Unsafe {field_name}: empty path")
    normalized = raw.replace("\\", "/")
    path = PurePosixPath(normalized)
    parts = [part for part in path.parts if part not in {"", "."}]
    # A colon in any component is rejected: on Windows it marks a drive (``C:foo``) or an NTFS Alternate Data
    # Stream (``file.py:payload`` writes scanner-invisible bytes); ``/`` is the only legal separator once normalized.
    if (normalized.startswith("/") or path.is_absolute() or not parts or any(part == ".." for part in parts)
            or any(":" in part for part in parts) or (not allow_nested and len(parts) != 1)):
        raise ValueError(f"Unsafe {field_name}: {path_value}")
    return "/".join(parts)


def _validate_skill_name(name: str) -> str:
    return _normalize_bundle_path(name, field_name="skill name", allow_nested=False)


def _validate_install_parent_path(category: str) -> str:
    return _normalize_bundle_path(category, field_name="install parent path", allow_nested=True)


def _validate_bundle_rel_path(rel_path: str) -> str:
    return _normalize_bundle_path(rel_path, field_name="bundle file path", allow_nested=True)


def _normalize_lock_install_path(install_path: str, skill_name: str) -> str:
    """Validate a lock-file ``install_path`` (the ``uninstall_skill`` rmtree target).

    Must be relative, traversal-free, and end with ``<skill_name>`` — nested official skills legitimately
    live at ``mlops/training/<skill_name>``; an empty/``"."``/absolute/mismatched entry could point rmtree
    at the whole ``skills/`` tree or outside it.
    """
    safe_skill_name = _validate_skill_name(skill_name)
    normalized = _normalize_bundle_path(install_path, field_name="install path", allow_nested=True)
    if normalized.split("/")[-1] != safe_skill_name:
        raise ValueError(f"Unsafe install path: {install_path}")
    return normalized


# --- Referenced support-file extraction from SKILL.md -----------------------
_ALLOWED_SUPPORT_DIRS = frozenset({"references", "templates", "scripts", "assets", "examples"})
_LOCAL_LINK_RE = re.compile(
    r"(?:\]\(|`|(?:^|[\s\"']))((?:references|templates|scripts|assets|examples)/[^\s)`\"'<>]+)", re.MULTILINE)
_SUSPICIOUS_LOCAL_REF_RE = re.compile(
    r"(?:references|templates|scripts|assets|examples)/(?:[^\s)`\"'<>]*/)?\.\.(?:/|$)")
_VALUELESS_QUERY_FLAG_RE = re.compile(r"(?:[A-Za-z0-9_~-]|%[0-9A-Fa-f]{2})+\Z")
# Same-directory links (``](./FILE.ext)`` / ``](FILE.ext)``): siblings of SKILL.md the document links
# explicitly (e.g. ./CONTEXT-FORMAT.md). Dropping them made the install "succeed" with unresolved links.
# The extension requirement keeps prose words out; support-dir links stay on _LOCAL_LINK_RE.
# Skills legitimately ship supporting docs next to SKILL.md instead of under a support directory (e.g.
# mattpocock/skills' domain-modeling links ./CONTEXT-FORMAT.md); dropping them made the install "succeed"
# while the bundle came out with unresolved links (#96310).
_SAMEDIR_LINK_RE = re.compile(r"\]\(([^)\s\"'<>]+)")
_SAMEDIR_NAME_RE = re.compile(r"^(?:\./)?[A-Za-z0-9][A-Za-z0-9._-]*$")


def _query_is_concrete(query: str) -> bool:
    """Whether a URL query is real URL syntax rather than glob prose.

    A non-empty ``key=value`` part is always concrete. Valueless flags are accepted only when RFC 3986
    unreserved-token shaped (percent escapes ok); ``.``, brackets and extra ``?`` are excluded because
    ``?x.md`` / ``?.md`` is indistinguishable from a single-char glob finishing a filename in prose.
    """
    return all(("=" in part and bool(part.split("=", 1)[0])) or bool(_VALUELESS_QUERY_FLAG_RE.fullmatch(part))
               for part in query.split("&"))


def _referenced_support_paths(skill_md: str) -> Optional[set[str]]:
    """Extract safe referenced paths; return None on a traversal attempt (fail closed)."""
    normalized = skill_md.replace("\\", "/")
    if _SUSPICIOUS_LOCAL_REF_RE.search(normalized):
        return None
    paths: set[str] = set()
    for match in _LOCAL_LINK_RE.finditer(normalized):
        candidate = match.group(1).rstrip(".,;:")
        parsed = urlsplit(candidate)
        raw = unquote(parsed.path)
        if (candidate.endswith("?") or any(char in raw for char in "*?[]")
                or (parsed.query and not _query_is_concrete(parsed.query))):
            continue
        try:
            safe = _validate_bundle_rel_path(raw)
        except ValueError:
            return None
        if safe.split("/", 1)[0] in _ALLOWED_SUPPORT_DIRS:
            # Prose placeholders (``references/type-<name>.md``, truncated at ``<`` to
            # ``references/type-``) are instructions, not files: a basename ending in a
            # separator is skipped. No extension requirement — ``references/LICENSE`` is legitimate.
            base = safe.rsplit("/", 1)[-1]
            if re.search(r"[*?<>]", safe) or not re.search(r"[A-Za-z0-9]$", base):
                continue
            paths.add(safe)
    for match in _SAMEDIR_LINK_RE.finditer(normalized):
        raw = match.group(1).rstrip(".,;:")
        # Canonicalize like the support-dir branch (drop query/fragment, percent-decode), strip leading ``./``.
        name = unquote(urlsplit(raw).path)
        name = name[2:] if name.startswith("./") else name
        # External URLs, anchors, mailto and site-absolute targets are not same-directory file links.
        if not name or "://" in raw or raw.startswith(("mailto:", "#", "/")):
            continue
        if name.startswith(".."):
            return None
        # Only unambiguous file links: an extension, no internal slash, never SKILL.md itself (casefolded —
        # a ``skill.md`` entry would collide with the bundle root on macOS/Windows; skipped, not merged).
        if ("/" in name or name.casefold() == "skill.md" or "." not in name.lstrip(".")
                or not _SAMEDIR_NAME_RE.match(name)):
            continue
        try:
            safe = _validate_bundle_rel_path(name)
        except ValueError:
            return None
        paths.add(safe)
    # Case-folded collisions among accepted same-dir names (``A.md`` + ``a.md``)
    # would collide on install — drop the pair rather than guess.
    folded: dict[str, list[str]] = {}
    for p in sorted(paths):
        folded.setdefault(p.casefold(), []).append(p)
    for group in folded.values():
        if len(group) > 1:
            paths.difference_update(group)
    return paths
