#!/usr/bin/env python3
"""
Skills Hub — hub state management and the public facade for the source adapters.

Library module (not an agent tool). Owns the hub paths, guarded HTTP, index
cache, lock file, taps and audit log. Install/uninstall/update live in
``skills_hub_install``, the index fetch/source router/search in
``skills_hub_search``, and the adapters in the other ``tools.skills_hub_*``
siblings; import each name from its defining module.

Used by hermes_cli/skills_hub.py for CLI commands and the /skills slash command.
"""

import json
import logging
import time
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import urljoin

import httpx

from hermes_constants import get_hermes_home
from tools.url_safety import is_safe_url
from tools.website_policy import check_website_access
from tools.skills_hub_models import _normalize_lock_install_path, _validate_skill_name

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Resolved per-call (not frozen at import) so the profile override is honored;
# import-time constants leaked across profiles in single-process multi-profile
# runtimes. The path names (SKILLS_DIR, ...) resolve live via __getattr__ below.

INDEX_CACHE_TTL = 3600  # 1 hour


def _path_resolver(name: str, parent: str, leaf: str):
    """Resolver for hub path ``<parent>/<leaf>``: a test-injected real module attribute
    (patch.object/monkeypatch on SKILLS_DIR etc.) wins over live resolution."""
    def resolve() -> Path:
        forced = globals().get(name)
        return Path(forced) if forced is not None else _DYNAMIC_PATH_RESOLVERS[parent]() / leaf
    resolve.__name__ = f"_{name.lower()}"
    return resolve


def _hermes_home() -> Path:
    return get_hermes_home()


_skills_dir = _path_resolver("SKILLS_DIR", "HERMES_HOME", "skills")
_hub_dir = _path_resolver("HUB_DIR", "SKILLS_DIR", ".hub")
_lock_file = _path_resolver("LOCK_FILE", "HUB_DIR", "lock.json")
_quarantine_dir = _path_resolver("QUARANTINE_DIR", "HUB_DIR", "quarantine")
_audit_log = _path_resolver("AUDIT_LOG", "HUB_DIR", "audit.log")
_taps_file = _path_resolver("TAPS_FILE", "HUB_DIR", "taps.json")
_index_cache_dir = _path_resolver("INDEX_CACHE_DIR", "HUB_DIR", "index-cache")
_DYNAMIC_PATH_RESOLVERS = {"HERMES_HOME": _hermes_home, **{
    r.__name__[1:].upper(): r
    for r in (_skills_dir, _hub_dir, _lock_file, _quarantine_dir, _audit_log, _taps_file, _index_cache_dir)
}}


def __getattr__(name: str):
    """Resolve legacy path constants dynamically (PEP 562) so they reflect the
    active profile override; a test's patch.object-set real attribute shadows it."""
    resolver = _DYNAMIC_PATH_RESOLVERS.get(name)
    if resolver is not None:
        return resolver()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Install-path safety + guarded HTTP
# ---------------------------------------------------------------------------

_REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}
_MAX_SKILL_FETCH_REDIRECTS = 5


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

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


_ALLOWED_SUPPORT_DIRS = frozenset({"references", "templates", "scripts", "assets", "examples"})
_LOCAL_LINK_RE = re.compile(
    r"(?:\]\(|`|(?:^|[\s\"']))((?:references|templates|scripts|assets|examples)/[^\s)`\"'<>]+)",
    re.MULTILINE,
)
_SUSPICIOUS_LOCAL_REF_RE = re.compile(
    r"(?:references|templates|scripts|assets|examples)/(?:[^\s)`\"'<>]*/)?\.\.(?:/|$)"
)
_VALUELESS_QUERY_FLAG_RE = re.compile(
    r"(?:[A-Za-z0-9_~-]|%[0-9A-Fa-f]{2})+\Z"
)


def _query_is_concrete(query: str) -> bool:
    """Whether ``query`` is unambiguously URL syntax rather than glob prose.

    A non-empty ``key=value`` part is concrete URL syntax regardless of key
    spelling, preserving the established behavior.  Valueless flags are also
    accepted when they are RFC 3986 unreserved-token shaped (including valid
    percent escapes), rather than being restricted to a fixed allowlist.

    Deliberately exclude ``.`` from valueless flags.  A suffix such as
    ``?x.md`` or ``?.md`` is indistinguishable from a single-character glob
    completing a filename in inline prose.  Brackets and additional question
    marks are excluded for the same reason.  This syntactic ambiguity policy
    preserves ordinary flags such as ``?view`` and ``?preview-mode`` while
    rejecting glob-shaped references without guessing flag names.
    """
    parts = query.split("&")
    return all(
        ("=" in part and bool(part.split("=", 1)[0]))
        or bool(_VALUELESS_QUERY_FLAG_RE.fullmatch(part))
        for part in parts
    )

# Same-directory links (``](./FILE.ext)`` / ``](FILE.ext)``) — siblings of
# SKILL.md that the document explicitly links. Skills legitimately ship
# supporting docs next to SKILL.md instead of under a support directory
# (e.g. mattpocock/skills' domain-modeling links ./CONTEXT-FORMAT.md);
# dropping them made the install "succeed" while the bundle came out with
# unresolved links (#96310). The trailing extension requirement keeps prose
# words out; the code-side checks keep this strictly to the skill's own
# directory (support-dir links stay on _LOCAL_LINK_RE).
_SAMEDIR_LINK_RE = re.compile(r"\]\(([^)\s\"'<>]+)")
_SAMEDIR_NAME_RE = re.compile(r"^(?:\./)?[A-Za-z0-9][A-Za-z0-9._-]*$")


def _referenced_support_paths(skill_md: str) -> Optional[set[str]]:
    """Extract safe referenced paths; return None on a traversal attempt."""
    normalized = skill_md.replace("\\", "/")
    if _SUSPICIOUS_LOCAL_REF_RE.search(normalized):
        return None
    paths: set[str] = set()
    for match in _LOCAL_LINK_RE.finditer(normalized):
        candidate = match.group(1).rstrip(".,;:")
        if candidate.endswith("?"):
            continue
        parsed = urlsplit(candidate)
        raw = unquote(parsed.path)
        if any(char in raw for char in "*?[]"):
            continue
        if parsed.query and not _query_is_concrete(parsed.query):
            continue
        try:
            safe = _validate_bundle_rel_path(raw)
        except ValueError:
            return None
        if safe.split("/", 1)[0] in _ALLOWED_SUPPORT_DIRS:
            # Prose placeholders — e.g. ``references/type-<name>.md`` (which
            # the link regex truncates at ``<`` to the bare prefix
            # ``references/type-``) — are agent instructions, not files.
            # Glob shapes (*, ?, []) were already rejected on the raw
            # candidate above; a truncated placeholder leaves a basename
            # ending in a separator, which no real file uses. No extension
            # requirement: extensionless support files
            # (``references/LICENSE``) are legitimate.
            base = safe.rsplit("/", 1)[-1]
            if re.search(r"[*?<>]", safe) or not re.search(r"[A-Za-z0-9]$", base):
                continue
            paths.add(safe)
    for match in _SAMEDIR_LINK_RE.finditer(normalized):
        raw = match.group(1).rstrip(".,;:")
        # Canonicalize first: drop query/fragment components (``?raw=1``,
        # ``#section``) and percent-decode — the same normalization the
        # support-dir branch applies via urlsplit+unquote — then strip a
        # leading ``./``. The set below deduplicates case-SENSITIVE repeats;
        # case-VARIANT collisions are rejected rather than merged (below).
        name = unquote(urlsplit(raw).path)
        name = name[2:] if name.startswith("./") else name
        # External URLs, anchors, mailto and site-absolute targets are not
        # same-directory file links — leave them to their own resolution.
        if not name or "://" in raw or raw.startswith(("mailto:", "#", "/")):
            continue
        # A ``..`` prefix is a traversal attempt — same fail-closed contract
        # as the support-dir branch above, before any shape-based skipping.
        if name.startswith(".."):
            return None
        # Only unambiguous file links: an extension, no internal slash, and
        # never SKILL.md itself (that IS the bundle root). The casefold
        # check keeps a ``skill.md`` link from shipping as a bundle entry
        # that collides with SKILL.md on case-insensitive filesystems
        # (macOS/Windows) — skipped, not merged, so the bundle root is
        # never overwritten (#96310 review).
        if (
            "/" in name
            or name.casefold() == "skill.md"
            or "." not in name.lstrip(".")
        ):
            continue
        if not _SAMEDIR_NAME_RE.match(name):
            continue
        try:
            safe = _validate_bundle_rel_path(name)
        except ValueError:
            return None
        paths.add(safe)
    # Case-folded collision among the accepted same-dir names themselves
    # (``A.md`` + ``a.md``) would also collide on install — drop the pair
    # rather than guess which variant the author meant.
    folded: dict[str, str] = {}
    for p in sorted(paths):
        key = p.casefold()
        if key in folded:
            paths.discard(folded[key])
            paths.discard(p)
        else:
            folded[key] = p
    return paths


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

    if normalized.startswith("/") or path.is_absolute():
        raise ValueError(f"Unsafe {field_name}: {path_value}")
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"Unsafe {field_name}: {path_value}")
    # Reject a colon in any component. On Windows a colon marks either a drive
    # (``C:`` / ``C:foo``) or an NTFS Alternate Data Stream: a bundle member
    # named ``file.py:payload`` writes hidden, scanner-invisible bytes into the
    # visible ``file.py`` (rglob-based review never enumerates the stream).
    # ``/`` is the only legal separator once normalized, so no portable bundle
    # path needs a colon in a component.
    if any(":" in part for part in parts):
        raise ValueError(f"Unsafe {field_name}: {path_value}")
    if not allow_nested and len(parts) != 1:
        raise ValueError(f"Unsafe {field_name}: {path_value}")

    return "/".join(parts)


def _validate_skill_name(name: str) -> str:
    return _normalize_bundle_path(name, field_name="skill name", allow_nested=False)


def _validate_install_parent_path(category: str) -> str:
    return _normalize_bundle_path(category, field_name="install parent path", allow_nested=True)


def _normalize_lock_install_path(install_path: str, skill_name: str) -> str:
    """Validate a skill install path before it touches the lock file or disk.

    Lock-file ``install_path`` entries are the source-of-truth for where
    ``uninstall_skill`` will call ``shutil.rmtree``. A poisoned or buggy
    entry — empty string, ``"."``, an absolute path, ``../..`` traversal,
    or anything whose final component doesn't match the skill name — would
    let ``rmtree`` wipe either the entire ``skills/`` tree or content
    outside it.

    Enforce that ``install_path`` ends with ``<skill_name>``. Nested
    official optional skills may legitimately install below paths such as
    ``mlops/training/<skill_name>``; traversal, absolute paths, empty paths,
    and mismatched final components are still rejected.
    """
    safe_skill_name = _validate_skill_name(skill_name)
    normalized = _normalize_bundle_path(
        install_path,
        field_name="install path",
        allow_nested=True,
    )
    parts = normalized.split("/")
    if not parts or parts[-1] != safe_skill_name:
        raise ValueError(f"Unsafe install path: {install_path}")
    return normalized


def _is_path_redirect(path: Path) -> bool:
    """True when ``path`` is a symlink or (on Windows) a directory junction.

    Either form lets an attacker who can write into the ``skills/`` tree
    redirect a subsequent ``rmtree`` to content outside it. ``is_junction``
    only exists on Python 3.12+ Windows; gate with ``hasattr``.
    """
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def _resolve_lock_install_path(install_path: str, skill_name: str) -> Path:
    """Resolve a lock-file install path without allowing escapes from ``SKILLS_DIR``.

    Two layers of defence on top of the existing ``is_relative_to`` check
    that's been on main:

    1. Walk the path component-by-component and refuse if any intermediate
       component is a symlink/junction (a path resolution that follows a
       symlink to outside skills/ would otherwise be hidden by Path.resolve).
    2. After resolve(), reject not just escape-out but also ``resolved == SKILLS_DIR``
       — an empty/``"."``/``""`` install_path resolves to the skills root itself,
       and ``rmtree(SKILLS_DIR)`` would wipe every installed skill.
    """
    normalized = _normalize_lock_install_path(install_path, skill_name)
    skills_dir = _skills_dir()
    skills_root = skills_dir.resolve()

    target = skills_dir
    for part in normalized.split("/"):
        target = target / part
        if _is_path_redirect(target):
            raise ValueError(f"Unsafe install path: {install_path}")

    target = target.resolve()
    if target == skills_root or not target.is_relative_to(skills_root):
        raise ValueError(f"Unsafe install path: {install_path}")
    return target


def _ssrf_safe_http_get(url: str, *, timeout: int = 20) -> httpx.Response:
    """Fetch one URL with connect-time SSRF validation and no automatic redirects."""
    from tools.url_safety import create_ssrf_safe_client

    with create_ssrf_safe_client(timeout=timeout, follow_redirects=False) as client:
        return client.get(url)


def _guarded_http_get(url: str, *, timeout: int = 20) -> Optional[httpx.Response]:
    """Fetch a URL with SSRF and redirect-target validation (each hop re-checked)."""
    from tools.url_safety import SSRFConnectionBlocked

    current_url = url

    for _ in range(_MAX_SKILL_FETCH_REDIRECTS + 1):
        if not is_safe_url(current_url):
            logger.warning("Blocked unsafe Skills Hub URL: %s", current_url)
            return None

        blocked = check_website_access(current_url)
        if blocked:
            logger.info(
                "Blocked Skills Hub fetch for %s by rule %s",
                blocked["host"],
                blocked["rule"],
            )
            return None

        try:
            resp = _ssrf_safe_http_get(current_url, timeout=timeout)
        except (SSRFConnectionBlocked, httpx.HTTPError) as exc:
            logger.debug("Skills Hub fetch failed for %s: %s", current_url, exc)
            return None

        if resp.status_code in _REDIRECT_STATUS_CODES:
            location = getattr(resp, "headers", {}).get("location")
            if not location:
                return None
            current_url = urljoin(current_url, location)
            continue

        return resp

    logger.warning("Skills Hub fetch exceeded redirect limit for %s", url)
    return None


@contextmanager
def _guarded_http_stream(
    url: str,
    *,
    params: Optional[Dict[str, str]] = None,
    timeout: int = 20,
) -> Iterator[Optional[httpx.Response]]:
    """Stream one response with bounded, policy-checked redirects."""
    from tools.url_safety import SSRFConnectionBlocked, create_ssrf_safe_client

    current_url = url
    current_params = params
    response: Optional[httpx.Response] = None
    stack = ExitStack()

    try:
        for _ in range(_MAX_SKILL_FETCH_REDIRECTS + 1):
            if not is_safe_url(current_url):
                logger.warning("Blocked unsafe Skills Hub URL: %s", current_url)
                response = None
                break

            blocked = check_website_access(current_url)
            if blocked:
                logger.info(
                    "Blocked Skills Hub fetch for %s by rule %s",
                    blocked["host"],
                    blocked["rule"],
                )
                response = None
                break

            stack.close()
            stack = ExitStack()
            try:
                client = stack.enter_context(
                    create_ssrf_safe_client(timeout=timeout, follow_redirects=False)
                )
                response = stack.enter_context(
                    client.stream("GET", current_url, params=current_params)
                )
            except (SSRFConnectionBlocked, httpx.HTTPError) as exc:
                logger.debug("Skills Hub stream failed for %s: %s", current_url, exc)
                response = None
                break

            if response.status_code not in _REDIRECT_STATUS_CODES:
                break

            location = response.headers.get("location")
            if not location:
                response = None
                break
            current_url = urljoin(current_url, location)
            current_params = None
        else:
            logger.warning("Skills Hub fetch exceeded redirect limit for %s", url)
            response = None

        yield response
    finally:
        stack.close()


# ---------------------------------------------------------------------------
# Shared index cache (used by every adapter)
# ---------------------------------------------------------------------------

def _read_json_if_fresh(path: Path, ttl: float) -> Optional[Any]:
    """Parsed JSON from ``path`` when it exists and is younger than ``ttl`` seconds."""
    try:
        if time.time() - path.stat().st_mtime > ttl:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_index_cache(key: str) -> Optional[Any]:
    return _read_json_if_fresh(_index_cache_dir() / f"{key}.json", INDEX_CACHE_TTL)


def _write_index_cache(key: str, data: Any) -> None:
    index_cache_dir = _index_cache_dir()
    index_cache_dir.mkdir(parents=True, exist_ok=True)
    # Cache files hold unvetted community text (possible prompt injection);
    # a .ignore keeps ripgrep and .ignore-aware tools out of the hub dir.
    ignore_file = _hub_dir() / ".ignore"
    if not ignore_file.exists():
        try:
            ignore_file.write_text("# Exclude hub internals from search tools\n*\n", encoding="utf-8")
        except OSError:
            pass
    try:
        (index_cache_dir / f"{key}.json").write_text(
            json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8"
        )
    except OSError as e:
        logger.debug("Could not write cache: %s", e)


# ---------------------------------------------------------------------------
# Hub state files: lock.json, taps.json, audit.log
# ---------------------------------------------------------------------------

class _JsonStateFile:
    """A JSON file under the hub dir with a fixed empty shape (``EMPTY``, deep-copied
    on every miss/corrupt read); ``DEFAULT_PATH`` is the hub path resolver."""

    EMPTY: dict = {}
    DEFAULT_PATH: Any = None

    def __init__(self, path: Optional[Path] = None):
        self.path = path if path is not None else type(self).DEFAULT_PATH()

    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return json.loads(json.dumps(self.EMPTY))

    def _write(self, data: dict, **dumps_kwargs) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, **dumps_kwargs) + "\n", encoding="utf-8")


class HubLockFile(_JsonStateFile):
    """skills/.hub/lock.json — provenance of installed hub skills."""

    EMPTY = {"version": 1, "installed": {}}
    DEFAULT_PATH = staticmethod(_lock_file)

    def load(self) -> dict:
        return self._read()

    def save(self, data: dict) -> None:
        self._write(data, ensure_ascii=False)

    def record_install(
        self,
        name: str,
        source: str,
        identifier: str,
        trust_level: str,
        scan_verdict: str,
        skill_hash: str,
        install_path: str,
        files: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        scan_provenance: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Validate name and install-path SHAPE at write time: a poisoned lock
        # entry is the precondition for the uninstall_skill rmtree-escape.
        safe_name = _validate_skill_name(name)
        safe_install_path = _normalize_lock_install_path(install_path, safe_name)
        data = self.load()
        now = datetime.now(timezone.utc).isoformat()
        data["installed"][safe_name] = {
            "source": source,
            "identifier": identifier,
            "trust_level": trust_level,
            "scan_verdict": scan_verdict,
            "content_hash": skill_hash,
            "install_path": safe_install_path,
            "files": files,
            "metadata": metadata or {},
            "scan_provenance": scan_provenance or {},
            "installed_at": now,
            "updated_at": now,
        }
        self.save(data)

    def record_uninstall(self, name: str) -> None:
        data = self.load()
        data["installed"].pop(name, None)
        self.save(data)

    def get_installed(self, name: str) -> Optional[dict]:
        return self.load()["installed"].get(name)

    def list_installed(self) -> List[dict]:
        return [{"name": name, **entry} for name, entry in self.load()["installed"].items()]


class TapsManager(_JsonStateFile):
    """skills/.hub/taps.json — custom GitHub repo sources."""

    EMPTY = {"taps": []}
    DEFAULT_PATH = staticmethod(_taps_file)

    def load(self) -> List[dict]:
        return self._read().get("taps", [])

    def save(self, taps: List[dict]) -> None:
        self._write({"taps": taps})

    def add(self, repo: str, path: str = "skills/") -> bool:
        """Add a tap. Returns False if already exists."""
        taps = self.load()
        if any(t["repo"] == repo for t in taps):
            return False
        taps.append({"repo": repo, "path": path})
        self.save(taps)
        return True

    def remove(self, repo: str) -> bool:
        """Remove a tap by repo name. Returns False if not found."""
        taps = self.load()
        new_taps = [t for t in taps if t["repo"] != repo]
        if len(new_taps) == len(taps):
            return False
        self.save(new_taps)
        return True

    list_taps = load


def append_audit_log(action: str, skill_name: str, source: str,
                     trust_level: str, verdict: str, extra: str = "") -> None:
    """Append one space-separated line to the audit log (best-effort)."""
    audit_log = _audit_log()
    audit_log.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    parts = [timestamp, action, skill_name, f"{source}:{trust_level}", verdict]
    if extra:
        parts.append(extra)
    try:
        with open(audit_log, "a", encoding="utf-8") as f:
            f.write(" ".join(parts) + "\n")
    except OSError as e:
        logger.debug("Could not write audit log: %s", e)


# ---------------------------------------------------------------------------
# Hub operations (high-level)
# ---------------------------------------------------------------------------

def ensure_hub_dirs() -> None:
    """Create the .hub directory structure if it doesn't exist."""
    _hub_dir().mkdir(parents=True, exist_ok=True)
    _quarantine_dir().mkdir(exist_ok=True)
    _index_cache_dir().mkdir(exist_ok=True)
    if not lock_file.exists():
        lock_file.write_text('{"version": 1, "installed": {}}\n', encoding="utf-8")
    if not audit_log.exists():
        audit_log.touch()
    if not taps_file.exists():
        taps_file.write_text('{"taps": []}\n', encoding="utf-8")


def quarantine_bundle(bundle: SkillBundle) -> Path:
    """Write a skill bundle to the quarantine directory for scanning."""
    ensure_hub_dirs()
    skill_name = _validate_skill_name(bundle.name)
    validated_files: List[Tuple[str, Union[str, bytes]]] = []
    for rel_path, file_content in bundle.files.items():
        safe_rel_path = _validate_bundle_rel_path(rel_path)
        validated_files.append((safe_rel_path, file_content))

    dest = _quarantine_dir() / skill_name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    for rel_path, file_content in validated_files:
        file_dest = dest.joinpath(*rel_path.split("/"))
        file_dest.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(file_content, bytes):
            file_dest.write_bytes(file_content)
        else:
            file_dest.write_text(file_content, encoding="utf-8")

    return dest


def _category_skill_dirs(directory: Path) -> List[str]:
    """Names of direct children of *directory* that contain skills.

    A child counts when it is a non-hidden directory holding at least one
    active ``SKILL.md`` anywhere below it (recursive, so nested category
    layouts like ``mlops/training/<skill>`` are detected). Vendored,
    cache, and progressive-disclosure support paths are pruned via
    :func:`is_excluded_skill_path` so a lone ``node_modules`` or
    ``references/pkg/SKILL.md`` hit does not misclassify the directory as
    a category. Shared by the install-time category guard here and
    ``hermes_cli.skills_hub._existing_categories``.
    """
    skill_dirs: List[str] = []
    for entry in directory.iterdir():
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        for skill_md in entry.rglob("SKILL.md"):
            if is_excluded_skill_path(
                skill_md.relative_to(directory), root=directory
            ):
                continue
            skill_dirs.append(entry.name)
            break
    return skill_dirs


def install_from_quarantine(
    quarantine_path: Path,
    skill_name: str,
    category: str,
    bundle: SkillBundle,
    scan_result: ScanResult,
    scan_provenance: Optional[Dict[str, Any]] = None,
) -> Path:
    """Move a scanned skill from quarantine into the skills directory."""
    safe_skill_name = _validate_skill_name(skill_name)
    safe_category = _validate_install_parent_path(category) if category else ""
    quarantine_resolved = quarantine_path.resolve()
    quarantine_root = _quarantine_dir().resolve()
    if not quarantine_resolved.is_relative_to(quarantine_root):
        raise ValueError(f"Unsafe quarantine path: {quarantine_path}")

    if safe_category:
        install_rel_path = f"{safe_category}/{safe_skill_name}"
    else:
        install_rel_path = safe_skill_name

    # Resolve via the same lock-path validator the uninstaller uses. Catches
    # symlink-in-skills-tree redirects at install time so the lock entry's
    # path can never refer to a redirected target.
    install_dir = _resolve_lock_install_path(install_rel_path, safe_skill_name)

    # Refuse to nest a skill inside an existing skill directory. Installing
    # with ``--category <name-of-an-existing-skill>`` would create a hybrid
    # skill-plus-category directory; a later update or uninstall of the outer
    # skill would then rmtree the inner one — the sibling case of the
    # category-bucket wipe reported in issue #75983.
    skills_root = _skills_dir().resolve()
    ancestor = install_dir.parent
    while ancestor != skills_root and ancestor.is_relative_to(skills_root):
        if (ancestor / "SKILL.md").is_file():
            raise ValueError(
                f"Refusing to install into '{ancestor.name}': it is an "
                f"existing skill directory, not a category. Choose a "
                f"different category."
            )
        ancestor = ancestor.parent

    if install_dir.exists():
        if not install_dir.is_dir():
            # A stray regular file at the install path. rmtree() on a file
            # raises NotADirectoryError (an uncaught traceback at the CLI);
            # refuse with the same actionable ValueError contract instead.
            raise ValueError(
                f"Refusing to install: '{install_dir.name}' already exists "
                f"and is not a directory. Remove it or choose a different "
                f"skill name."
            )
        # Guard against silent data loss when the install target collides with
        # an existing category bucket (a directory that holds other skills).
        # This was reported as GitHub issue #75983: installing a skill with
        # --name matching an existing category directory caused rmtree to wipe
        # all sibling skills.  A directory that directly contains SKILL.md is
        # an existing skill installation and stays overwritable (hub-installed
        # skills are additionally guarded by the lock-file check in
        # do_install()).  But a directory that contains *other* skill
        # directories is a category bucket and must NOT be silently deleted.
        if not (install_dir / "SKILL.md").exists():
            skill_dirs_in = _category_skill_dirs(install_dir)
            if skill_dirs_in:
                raise ValueError(
                    f"Refusing to overwrite category directory '{install_dir}' "
                    f"which contains {len(skill_dirs_in)} skill(s): "
                    f"{', '.join(sorted(skill_dirs_in))}. "
                    f"Use a different --name or install into a subcategory."
                )
        shutil.rmtree(install_dir)

    # Warn (but don't block) if SKILL.md is very large
    skill_md = quarantine_path / "SKILL.md"
    if skill_md.exists():
        try:
            skill_size = skill_md.stat().st_size
            if skill_size > 100_000:
                logger.warning(
                    "Skill '%s' has a large SKILL.md (%s chars). "
                    "Large skills consume significant context when loaded. "
                    "Consider asking the author to split it into smaller files.",
                    safe_skill_name,
                    f"{skill_size:,}",
                )
        except OSError:
            pass

    # Reject symlinks inside the quarantined skill before moving it.
    # A malicious skill bundle could include a symlink pointing outside the
    # skills tree; its target contents would then be copied into skills/ and
    # leaked to the agent on the next skill_view call.
    for entry in quarantine_path.rglob("*"):
        if not _is_path_redirect(entry):
            continue
        try:
            rel = entry.relative_to(quarantine_resolved)
        except ValueError:
            rel = entry
        raise ValueError(
            f"Installed skill contains symlinks, which is not allowed: {rel}"
        )

    install_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(quarantine_path), str(install_dir))

    # Record in lock file
    lock = HubLockFile()
    lock.record_install(
        name=safe_skill_name,
        source=bundle.source,
        identifier=bundle.identifier,
        trust_level=bundle.trust_level,
        scan_verdict=scan_result.verdict,
        skill_hash=content_hash(install_dir),
        install_path=install_dir.resolve().relative_to(_skills_dir().resolve()).as_posix(),
        files=list(bundle.files.keys()),
        metadata=bundle.metadata,
        scan_provenance=scan_provenance or getattr(scan_result, "scan_provenance", None),
    )

    append_audit_log(
        "INSTALL", safe_skill_name, bundle.source,
        bundle.trust_level, scan_result.verdict,
        content_hash(install_dir),
    )

    try:
        from tools.skill_usage import record_installed

        record_installed(safe_skill_name)
    except Exception:
        logger.debug(
            "Unable to record skill install lifecycle for %s",
            safe_skill_name,
            exc_info=True,
        )

    return install_dir


def uninstall_skill(skill_name: str) -> Tuple[bool, str]:
    """Remove a hub-installed skill. Refuses to remove builtins."""
    lock = HubLockFile()
    entry = lock.get_installed(skill_name)
    if not entry:
        return False, f"'{skill_name}' is not a hub-installed skill (may be a builtin)"

    # Validate the lock entry's install_path against the skill name. This is
    # the destructive boundary — anything that falls through to the rmtree
    # below MUST be inside SKILLS_DIR and MUST NOT be SKILLS_DIR itself
    # (an empty/"."/"/" install_path would otherwise wipe the entire tree).
    # _resolve_lock_install_path enforces a relative path ending in
    # <skill_name>, rejects absolute/traversal paths, and walks the path
    # component-by-component refusing symlink/junction redirects.
    try:
        install_path = _resolve_lock_install_path(
            entry.get("install_path", ""), skill_name
        )
    except ValueError as exc:
        return False, f"Refusing to uninstall '{skill_name}': {exc}"

    if install_path.exists():
        shutil.rmtree(install_path)

    lock.record_uninstall(skill_name)
    append_audit_log("UNINSTALL", skill_name, entry["source"], entry["trust_level"], "n/a", "user_request")

    return True, f"Uninstalled '{skill_name}' from {entry['install_path']}"


def bundle_content_hash(bundle: SkillBundle) -> str:
    """Compute a deterministic hash for an in-memory skill bundle.

    MUST stay symmetric with ``tools.skills_guard.content_hash`` (which
    hashes the same skill from disk). That function keys files by
    ``relative_to(...).as_posix()`` — forward slashes on every OS. Bundle
    keys built on Windows carry backslashes (``str(f.relative_to(dir))``),
    which changed both the hashed bytes AND the sort order, so every
    installed skill reported ``update_available`` forever on Windows
    (#62310). Normalize to POSIX separators before sorting/hashing.
    """
    h = hashlib.sha256()
    normalized = {
        rel_path.replace("\\", "/"): content
        for rel_path, content in bundle.files.items()
    }
    for rel_path in sorted(normalized):
        # Include the path so swapping file contents between two paths
        # changes the hash (avoids filename-swap evading update detection).
        h.update(rel_path.encode("utf-8"))
        h.update(b"\x00")
        content = normalized[rel_path]
        if isinstance(content, bytes):
            h.update(content)
        else:
            h.update(content.encode("utf-8"))
    return f"sha256:{h.hexdigest()[:16]}"


def _source_matches(source: SkillSource, source_name: str) -> bool:
    aliases = {
        "skills.sh": "skills-sh",
    }
    normalized = aliases.get(source_name, source_name)
    return source.source_id() == normalized


def check_for_skill_updates(
    name: Optional[str] = None,
    *,
    lock: Optional[HubLockFile] = None,
    sources: Optional[List[SkillSource]] = None,
    auth: Optional[GitHubAuth] = None,
) -> List[dict]:
    """Check installed hub skills for upstream changes."""
    lock = lock or HubLockFile()
    installed = lock.list_installed()
    if name:
        installed = [entry for entry in installed if entry.get("name") == name]

    if sources is None:
        sources = create_source_router(auth=auth)

    results: List[dict] = []
    for entry in installed:
        identifier = entry.get("identifier", "")
        source_name = entry.get("source", "")
        candidate_sources = [src for src in sources if _source_matches(src, source_name)]
        if not candidate_sources:
            # No adapter for the recorded source (e.g. a tap was removed, or the
            # source was renamed upstream). Previously this fell back to *all*
            # sources, which meant a same-named skill in a DIFFERENT registry
            # could satisfy the fetch and be reported as an update for this
            # entry -- silently reassigning provenance. Skill names are not
            # namespaced across registries, so that fallback is unsafe by
            # construction. Report unavailable instead and let the user decide.
            results.append({
                "name": entry.get("name", ""),
                "identifier": identifier,
                "source": source_name,
                "status": "unavailable",
            })
            continue

        bundle = None
        for src in candidate_sources:
            try:
                bundle = src.fetch(identifier)
            except Exception:
                bundle = None
            if bundle:
                break

        if not bundle:
            results.append({
                "name": entry.get("name", ""),
                "identifier": identifier,
                "source": source_name,
                "status": "unavailable",
            })
            continue

        current_hash = entry.get("content_hash", "")
        latest_hash = bundle_content_hash(bundle)
        status = "up_to_date" if current_hash == latest_hash else "update_available"
        results.append({
            "name": entry.get("name", ""),
            "identifier": identifier,
            "source": source_name,
            "status": status,
            "current_hash": current_hash,
            "latest_hash": latest_hash,
            "bundle": bundle,
        })

    return results


# ---------------------------------------------------------------------------
# Hermes centralized index source
# ---------------------------------------------------------------------------

HERMES_INDEX_URL = "https://hermes-agent.nousresearch.com/docs/api/skills-index.json"
HERMES_INDEX_TTL = 6 * 3600  # 6 hours


def _hermes_index_cache_file() -> Path:
    return _index_cache_dir() / "hermes-index.json"


def _load_hermes_index() -> Optional[dict]:
    """Fetch the centralized skills index, with local cache.

    The index is a JSON file hosted on the docs site, rebuilt daily by CI.
    We cache it locally for HERMES_INDEX_TTL seconds to avoid repeated
    downloads within a session.
    """
    # Check local cache
    hermes_index_cache_file = _hermes_index_cache_file()
    if hermes_index_cache_file.exists():
        try:
            age = time.time() - hermes_index_cache_file.stat().st_mtime
            if age < HERMES_INDEX_TTL:
                return json.loads(hermes_index_cache_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass

    # Fetch from docs site.
    #
    # We deliberately DON'T let httpx negotiate Brotli here.  The index is a
    # large body (tens of MB); httpx's streaming Brotli decoder, backed by
    # brotlicffi 1.2.0.1 (pinned for Discord attachment decoding), trips over
    # its own output_buffer_limit on payloads this size and raises
    # DecodingError("brotli: decoder process called with data when
    # 'can_accept_more_data()' is False").  That surfaces as an empty Skills
    # Hub (blank Browse-hub landing, index contributes 0 search hits) because
    # the error is caught below and we silently fall back to a (often absent)
    # stale cache.  Requesting gzip/deflate sidesteps the broken decoder while
    # still compressing the transfer.  The identity retry is belt-and-braces
    # for any future proxy that ignores the header and returns Brotli anyway.
    data = None
    for accept_encoding in ("gzip, deflate", "identity"):
        try:
            resp = httpx.get(
                HERMES_INDEX_URL,
                timeout=15,
                follow_redirects=True,
                headers={"Accept-Encoding": accept_encoding},
            )
            if resp.status_code != 200:
                logger.debug("Hermes index fetch returned %d", resp.status_code)
                return _load_stale_index_cache()
            data = resp.json()
            break
        except httpx.DecodingError as e:
            # Content-Encoding decode failed — retry once uncompressed before
            # giving up on the network path entirely.
            logger.debug(
                "Hermes index decode failed (Accept-Encoding=%s): %s",
                accept_encoding,
                e,
            )
            continue
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            logger.debug("Hermes index fetch failed: %s", e)
            return _load_stale_index_cache()

    if data is None:
        return _load_stale_index_cache()

    # Validate structure
    if not isinstance(data, dict) or "skills" not in data:
        return _load_stale_index_cache()

    # Cache locally
    try:
        hermes_index_cache_file.parent.mkdir(parents=True, exist_ok=True)
        hermes_index_cache_file.write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        pass

    return data


def _load_stale_index_cache() -> Optional[dict]:
    """Fall back to stale cache when the network fetch fails."""
    hermes_index_cache_file = _hermes_index_cache_file()
    if hermes_index_cache_file.exists():
        try:
            return json.loads(hermes_index_cache_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return None


class HermesIndexSource(SkillSource):
    """Skill source backed by the centralized Hermes Skills Index.

    The index is a JSON catalog published to the docs site and rebuilt
    daily by CI.  It contains metadata + resolved GitHub paths for every
    skill, eliminating the need for users to hit the GitHub API for
    search or path discovery.

    When the index is unavailable, all methods return empty / None so
    downstream sources take over transparently.
    """

    def __init__(self, auth: GitHubAuth):
        self._index: Optional[dict] = None
        self._loaded = False
        self.auth = auth
        # Lazily create GitHubSource for fetch — only used when actually
        # downloading files, which requires real GitHub API calls.
        self._github: Optional[GitHubSource] = None

    def _ensure_loaded(self) -> dict:
        if not self._loaded:
            self._index = _load_hermes_index()
            self._loaded = True
        return self._index or {}

    def _get_github(self) -> GitHubSource:
        if self._github is None:
            self._github = GitHubSource(auth=self.auth)
        return self._github

    def source_id(self) -> str:
        return "hermes-index"

    @property
    def is_available(self) -> bool:
        """Whether the index is loaded and has skills."""
        index = self._ensure_loaded()
        return bool(index.get("skills"))

    def trust_level_for(self, identifier: str) -> str:
        index = self._ensure_loaded()
        for skill in index.get("skills", []):
            if skill.get("identifier") == identifier:
                return skill.get("trust_level", "community")
        return "community"

    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        """Search the cached index.  Zero API calls.

        Matches against name, description, tags, identifier, and the per-tap
        ``extra.provider`` label (so a query like ``nvidia`` surfaces the
        ``NVIDIA/skills/...`` entries even though their ``source`` is the bare
        ``github``).  Results are scored and ranked (exact name > name prefix >
        whole-word > substring) rather than returned in raw index order and
        truncated at the first ``limit`` hits — that earlier break-at-limit
        behaviour returned an arbitrary file-order slice and buried the most
        relevant skills.
        """
        index = self._ensure_loaded()
        skills = index.get("skills", [])
        if not skills:
            return []

        if not query.strip():
            # No query — return featured/popular (index order)
            return [self._to_meta(s) for s in skills[:limit]]

        query_lower = query.lower()
        scored: List[Tuple[int, int, dict]] = []
        for i, s in enumerate(skills):
            name = str(s.get("name", "")).lower()
            provider = str((s.get("extra") or {}).get("provider", "")).lower()
            haystack = " ".join([
                name,
                str(s.get("description", "")).lower(),
                " ".join(str(t).lower() for t in s.get("tags", [])),
                str(s.get("identifier", "")).lower(),
                provider,
            ])
            if query_lower not in haystack:
                continue
            # Lower score sorts first.
            if name == query_lower:
                score = 0
            elif name.startswith(query_lower):
                score = 1
            elif provider == query_lower:
                score = 2
            elif query_lower in name.split() or query_lower in provider.split():
                score = 3
            elif query_lower in name:
                score = 4
            else:
                score = 5
            # i (original index order) is the stable tiebreaker.
            scored.append((score, i, s))

        scored.sort(key=lambda x: (x[0], x[1]))
        return [self._to_meta(s) for _, _, s in scored[:limit]]

    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        """Fetch a skill using the resolved path from the index.

        If the index has a ``resolved_github_id`` for this skill, we skip
        the entire candidate/discovery chain and go directly to GitHub
        with the exact path.  This reduces install from ~31 API calls to
        just the file content downloads (~5-22 depending on skill size).
        """
        index = self._ensure_loaded()
        entry = self._find_entry(identifier, index)
        if not entry:
            return None

        # Use resolved path if available
        resolved = entry.get("resolved_github_id")
        if resolved:
            bundle = self._get_github().fetch(resolved)
            if bundle:
                bundle.source = entry.get("source", "hermes-index")
                bundle.identifier = identifier
                return bundle

        # Fall back to identifier-based fetch via repo/path
        repo = entry.get("repo", "")
        path = entry.get("path", "")
        if repo and path:
            github_id = f"{repo}/{path}"
            bundle = self._get_github().fetch(github_id)
            if bundle:
                bundle.source = entry.get("source", "hermes-index")
                bundle.identifier = identifier
                return bundle

        return None

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        """Return metadata from the index.  Zero API calls."""
        index = self._ensure_loaded()
        entry = self._find_entry(identifier, index)
        if entry:
            return self._to_meta(entry)
        return None

    def _find_entry(self, identifier: str, index: dict) -> Optional[dict]:
        """Look up a skill in the index by identifier or name."""
        skills = index.get("skills", [])

        # Exact identifier match
        for s in skills:
            if s.get("identifier") == identifier:
                return s

        # Try without source prefix (e.g. "skills-sh/" stripped)
        normalized = identifier
        for prefix in ("skills-sh/", "skills.sh/", "official/", "github/", "clawhub/"):
            if identifier.startswith(prefix):
                normalized = identifier[len(prefix):]
                break

        # Match on normalized identifier or name
        for s in skills:
            sid = s.get("identifier", "")
            # Strip prefix from stored identifier too
            stored_normalized = sid
            for prefix in ("skills-sh/", "skills.sh/", "official/", "github/", "clawhub/"):
                if sid.startswith(prefix):
                    stored_normalized = sid[len(prefix):]
                    break
            if stored_normalized == normalized:
                return s

        return None

    @staticmethod
    def _to_meta(entry: dict) -> SkillMeta:
        return SkillMeta(
            name=entry.get("name", ""),
            description=entry.get("description", ""),
            source=entry.get("source", "hermes-index"),
            identifier=entry.get("identifier", ""),
            trust_level=entry.get("trust_level", "community"),
            repo=entry.get("repo"),
            path=entry.get("path"),
            tags=entry.get("tags", []),
            extra=entry.get("extra", {}),
        )


def create_source_router(auth: Optional[GitHubAuth] = None) -> List[SkillSource]:
    """
    Create all configured source adapters.
    Returns a list of active sources for search/fetch operations.
    """
    if auth is None:
        auth = GitHubAuth()

    taps_mgr = TapsManager()
    extra_taps = taps_mgr.list_taps()

    sources: List[SkillSource] = [
        OptionalSkillSource(auth=auth),  # Official optional skills (highest priority)
        HermesIndexSource(auth=auth), # Centralized index (search + resolved install paths)
        SkillsShSource(auth=auth),
        WellKnownSkillSource(),
        UrlSource(),                  # Direct HTTP(S) URL to a SKILL.md file
        GitHubSource(auth=auth, extra_taps=extra_taps),
        ClawHubSource(),
        LobeHubSource(),
        BrowseShSource(),   # browse.sh: 169+ site-specific browser automation skills
    ]

    return sources


def _search_one_source(
    src: SkillSource, query: str, limit: int
) -> Tuple[str, List[SkillMeta]]:
    """Search a single source.  Runs in a thread for parallelism."""
    try:
        return src.source_id(), src.search(query, limit=limit)
    except Exception as e:
        logger.debug("Search failed for %s: %s", src.source_id(), e)
        return src.source_id(), []


def parallel_search_sources(
    sources: List[SkillSource],
    query: str = "",
    per_source_limits: Optional[Dict[str, int]] = None,
    source_filter: str = "all",
    overall_timeout: float = 30,
    on_source_done: Optional[Any] = None,
) -> Tuple[List[SkillMeta], Dict[str, int], List[str]]:
    """Search all sources in parallel with per-source timeout.

    Returns ``(all_results, source_counts, timed_out_ids)``.

    *on_source_done* is an optional callback ``(source_id, count) -> None``
    invoked as each source completes — useful for progress indicators.
    """
    from concurrent.futures import as_completed

    per_source_limits = per_source_limits or {}

    # A provider filter (e.g. "nvidia", "openai") targets GitHub-tap skills
    # that the runtime index stores under source="github" with an
    # ``extra.provider`` label. It is NOT a real source id, so source-level
    # selection must treat it like "all" (the index / github source carries
    # the data); the per-provider narrowing happens downstream on the merged
    # results (see ``_filter_results_by_provider``).
    _provider_filter = source_filter.strip().lower() in _PROVIDER_FILTER_VALUES
    _effective_filter = "all" if _provider_filter else source_filter

    active: List[SkillSource] = []
    # When the centralized index is available and the user hasn't filtered
    # to a specific source, skip external API sources (github, skills-sh,
    # clawhub, etc.) — the index already has their data.  This avoids
    # ~70 GitHub API calls per search for unauthenticated users.
    _index_available = False
    _api_source_ids = frozenset({"github", "skills-sh", "clawhub",
                                  "lobehub", "well-known"})
    if _effective_filter == "all":
        for src in sources:
            if (src.source_id() == "hermes-index"
                    and getattr(src, "is_available", False)):
                _index_available = True
                break

    for src in sources:
        sid = src.source_id()
        if _effective_filter != "all" and sid != _effective_filter and sid != "official":
            continue
        # Skip external API sources when the index covers them
        if _index_available and sid in _api_source_ids:
            continue
        active.append(src)

    all_results: List[SkillMeta] = []
    source_counts: Dict[str, int] = {}
    timed_out_ids: List[str] = []

    if not active:
        return all_results, source_counts, timed_out_ids

    # NOTE: a `with ThreadPoolExecutor(...) as pool` block calls
    # ``shutdown(wait=True)`` on exit, which blocks until every submitted
    # worker finishes — so a single slow source (e.g. ClawHub) keeps the
    # caller blocked for minutes and renders ``overall_timeout`` a no-op.
    # Manage the executor manually and shut it down with ``wait=False`` so
    # the timeout is actually honoured.  Daemon workers (tools.daemon_pool):
    # an abandoned slow source must not block interpreter exit either —
    # stdlib workers are joined unconditionally by the atexit hook.
    from tools.daemon_pool import DaemonThreadPoolExecutor
    pool = DaemonThreadPoolExecutor(max_workers=min(len(active), 8))
    futures = {}
    for src in active:
        lim = per_source_limits.get(src.source_id(), 50)
        fut = pool.submit(_search_one_source, src, query, lim)
        futures[fut] = src.source_id()

    try:
        try:
            for fut in as_completed(futures, timeout=overall_timeout):
                try:
                    sid, results = fut.result(timeout=0)
                    source_counts[sid] = len(results)
                    all_results.extend(results)
                    if on_source_done:
                        on_source_done(sid, len(results))
                except Exception:
                    pass
        except TimeoutError:
            timed_out_ids = [
                futures[f] for f in futures if not f.done()
            ]
            if timed_out_ids:
                logger.debug(
                    "Skills browse timed out waiting for: %s",
                    ", ".join(timed_out_ids),
                )
    finally:
        # wait=False so a slow source cannot block the caller's return;
        # cancel_futures drops not-yet-started work.
        pool.shutdown(wait=False, cancel_futures=True)

    return all_results, source_counts, timed_out_ids


def unified_search(query: str, sources: List[SkillSource],
                   source_filter: str = "all", limit: int = 10) -> List[SkillMeta]:
    """Search all sources (in parallel) and merge results."""
    all_results, _, _ = parallel_search_sources(
        sources,
        query=query,
        source_filter=source_filter,
        overall_timeout=30,
    )

    # A provider filter (nvidia/openai/...) is applied here, on the merged set,
    # because it targets the per-tap ``extra.provider`` label rather than a real
    # source id (the runtime index stores every GitHub tap as source="github").
    if source_filter.strip().lower() in _PROVIDER_FILTER_VALUES:
        all_results = _filter_results_by_provider(all_results, source_filter)

    # Deduplicate by identifier, preferring higher trust levels.
    # identifier is always unique per skill (e.g. "browse-sh/airbnb.com/search-listings-ddgioa").
    # Using name would incorrectly collapse browse-sh skills from different sites that share
    # the same task name (e.g. "search-listings" from Airbnb and Booking.com).
    _TRUST_RANK = {"builtin": 2, "trusted": 1, "community": 0}
    seen: Dict[str, SkillMeta] = {}
    for r in all_results:
        if r.identifier not in seen:
            seen[r.identifier] = r
        elif _TRUST_RANK.get(r.trust_level, 0) > _TRUST_RANK.get(seen[r.identifier].trust_level, 0):
            seen[r.identifier] = r
    deduped = list(seen.values())

    # Stable-sort by trust rank before truncating: the limit cut must not
    # drop a builtin/official catalog entry because a high-volume community
    # source (skills.sh mirrors every repo) happened to finish first and
    # flood the merged list. Insertion order is preserved within each rank.
    deduped.sort(key=lambda r: -_TRUST_RANK.get(r.trust_level, 0))

    return deduped[:limit]
