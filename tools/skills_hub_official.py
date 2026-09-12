"""Skills Hub official sources: repo-shipped optional skills and the centralized Hermes index."""

import logging
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple, Union

from agent.skill_utils import is_excluded_skill_path
from tools.skills_hub_github import GitHubAuth, GitHubSource, _skip_bundle_file, _tree_members
from tools.skills_hub_models import (
    SkillBundle, SkillMeta, SkillSource, _hermes_tags, _matches_query, _memo_json, _parse_frontmatter, hub,
)

logger = logging.getLogger("tools.skills_hub")

# Identifier prefixes stripped when matching index entries loosely.
_INDEX_ID_PREFIXES = ("skills-sh/", "skills.sh/", "official/", "github/", "clawhub/")


def _strip_prefix(value: str, prefixes) -> str:
    return next((value[len(p):] for p in prefixes if value.startswith(p)), value)


def _clean_rel_parts(path: str) -> Optional[List[str]]:
    """Split a relative path, dropping ``.``/empty parts; None on traversal or empty."""
    parts = [p for p in path.split("/") if p not in ("", ".")]
    return None if not parts or ".." in parts else parts


class OptionalSkillSource(SkillSource):
    """Skills from the repo's ``optional-skills/`` directory: official (Nous-maintained) but not
    activated by default — absent from the system prompt and not copied to ~/.hermes/skills/ at
    setup. Discoverable via the Skills Hub as source "official" with "builtin" trust."""

    SOURCE_ID = "official"
    TRUST_LEVEL = "builtin"
    OFFICIAL_REPO = "NousResearch/hermes-agent"
    OPTIONAL_SKILLS_PREFIX = "optional-skills"

    _parse_frontmatter = staticmethod(_parse_frontmatter)

    def __init__(self, auth: Optional[GitHubAuth] = None):
        from hermes_constants import get_optional_skills_dir

        self._optional_dir = get_optional_skills_dir(Path(__file__).parent.parent / "optional-skills")
        self._auth = auth
        # GitHubSource for the live-repo fallback, created only when a skill is missing locally.
        self._github: Optional[GitHubSource] = None
        # "category/skill" -> True from the live repo tree; None = not fetched yet.
        self._remote_dirs: Optional[Dict[str, bool]] = None

    @staticmethod
    def _rel(identifier: str) -> str:
        return identifier.split("/", 1)[-1] if identifier.startswith("official/") else identifier

    def _meta(self, rel_dir: str, name: str, description: str, tags: list) -> SkillMeta:
        return SkillMeta(
            name=name, description=description, source="official", identifier=f"official/{rel_dir}",
            trust_level="builtin", repo=self.OFFICIAL_REPO,
            # The centralized skills index consumes repo-root-relative paths.
            path=f"{self.OPTIONAL_SKILLS_PREFIX}/{rel_dir}", tags=tags,
        )

    def _remote_meta(self, rel_dir: str) -> SkillMeta:
        """Placeholder meta for a skill that exists on live main but not locally."""
        desc = "Official optional skill (from live repo; run install to fetch)"
        return self._meta(rel_dir, rel_dir.rsplit("/", 1)[-1], desc, [])

    @staticmethod
    def _bundle(rel_id: str, files: Dict[str, Union[str, bytes]], **kwargs) -> SkillBundle:
        return SkillBundle(name=rel_id.rsplit("/", 1)[-1], files=files, source="official",
                           identifier=f"official/{rel_id}", trust_level="builtin", **kwargs)

    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        results: List[SkillMeta] = []
        query_lower = query.lower()
        local_rels: set = set()
        for meta in self._scan_all():
            local_rels.add(meta.identifier.split("/", 1)[-1] if meta.identifier else "")
            if _matches_query(query_lower, meta.name, meta.description, meta.tags):
                results.append(meta)
            if len(results) >= limit:
                break
        # Also surface skills that landed on live main after this install was cut.
        if len(results) < limit:
            for rel_dir in sorted(self._list_remote_skill_dirs()):
                if rel_dir in local_rels or (query_lower and query_lower not in rel_dir.lower()):
                    continue
                results.append(self._remote_meta(rel_dir))
                if len(results) >= limit:
                    break
        return results

    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        # identifier format: "official/category/skill" or "official/skill"
        rel = self._rel(identifier)
        # Guard against path traversal (e.g. "official/../../etc")
        try:
            resolved = (self._optional_dir / rel).resolve()
            optional_root = self._optional_dir.resolve()
            if not resolved.is_relative_to(optional_root):
                return None
        except (OSError, ValueError):
            return None

        # Else try by bare skill name; if still absent, the skill may have landed on main
        # after this install was cut — use the live repo.
        skill_dir = resolved if resolved.is_dir() else self._find_skill_dir(rel.rsplit("/", 1)[-1])
        if not skill_dir:
            return self._fetch_from_live_repo(rel)
        rel_id = skill_dir.resolve().relative_to(optional_root).as_posix()

        # Catalog stubs point at the real skill in an upstream-maintained repo
        # (metadata.hermes.upstream); install pulls the live content from there.
        try:
            skill_md = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            skill_md = None
        upstream = None if skill_md is None else self._upstream_pointer_from_content(skill_md)
        if upstream is not None:
            return self._fetch_from_upstream(upstream, rel_id)
        files: Dict[str, Union[str, bytes]] = {}
        for f in skill_dir.rglob("*"):
            if f.is_file() and not _skip_bundle_file(f.relative_to(skill_dir).as_posix()):
                try:
                    files[str(f.relative_to(skill_dir))] = f.read_bytes()
                except OSError:
                    continue
        return self._bundle(rel_id, files) if files else None

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        skill_name = self._rel(identifier).rsplit("/", 1)[-1]
        for meta in self._scan_all():
            if meta.name == skill_name:
                return meta
        matches = self._remote_matches(skill_name)  # not in the local checkout — check live main
        return self._remote_meta(matches[0]) if len(matches) == 1 else None

    def list_local(self) -> List[SkillMeta]:
        """Every optional skill in the local checkout, with frontmatter metadata
        (backs the dashboard/desktop "built-in optional skills" catalog)."""
        return self._scan_all()

    def _get_github(self) -> GitHubSource:
        if self._github is None:
            self._github = GitHubSource(auth=self._auth or GitHubAuth())
        return self._github

    def _remote_matches(self, name: str) -> List[str]:
        return [d for d in self._list_remote_skill_dirs() if d.rsplit("/", 1)[-1] == name]

    def _fetch_from_live_repo(self, rel: str) -> Optional[SkillBundle]:
        """Fetch an optional skill straight from the live default branch. Local installs lag
        ``main``; rather than demanding ``hermes update`` first, resolve against the live repo.
        ``rel`` is ``category/skill`` (used verbatim) or a bare skill name (located via the repo tree)."""
        parts = _clean_rel_parts(rel.strip("/"))
        if parts is None:
            return None
        rel = "/".join(parts)
        github = self._get_github()
        if rel not in self._list_remote_skill_dirs():
            # Bare name (or stale category) — locate by final path segment.
            matches = self._remote_matches(parts[-1])
            if len(matches) != 1:
                return None
            rel = matches[0]

        # Download the FULL directory byte-exact (root-level install scripts, LICENSE, tests/).
        # GitHubSource.fetch() would only pull SKILL.md + referenced support dirs.
        tree = github._get_repo_tree(self.OFFICIAL_REPO)
        if tree is None:
            return None
        files: Dict[str, Union[str, bytes]] = {}
        for rel_file, item_path, regular in _tree_members(tree[1], f"{self.OPTIONAL_SKILLS_PREFIX}/{rel}/"):
            if not regular or _skip_bundle_file(rel_file):
                continue
            content = github._fetch_file_bytes(self.OFFICIAL_REPO, item_path)
            if content is None:
                logger.warning("Live-repo optional skill fetch failed for %s", item_path)
                return None
            files[rel_file] = content
        if "SKILL.md" not in files:
            return None
        # Live-fetched catalog stubs redirect the same way local ones do.
        upstream = self._upstream_pointer_from_content(files["SKILL.md"])
        if upstream is not None:
            return self._fetch_from_upstream(upstream, rel)
        logger.info("Optional skill '%s' fetched from live repo (not in local checkout)", rel)
        return self._bundle(rel, files)

    def _list_remote_skill_dirs(self) -> Dict[str, bool]:
        """``category/skill`` dirs under optional-skills/ on live main. One repo-tree call (cached
        per-process by GitHubSource + the on-disk index cache). {} when the network/API is
        unavailable — callers degrade to local-only."""
        if self._remote_dirs is not None:
            return self._remote_dirs

        def compute():
            dirs: Dict[str, bool] = {}
            if (tree := self._get_github()._get_repo_tree(self.OFFICIAL_REPO)) is None:
                return None
            prefix, suffix = f"{self.OPTIONAL_SKILLS_PREFIX}/", "/SKILL.md"
            for item in tree[1]:
                path = item.get("path", "")
                if item.get("type") == "blob" and path.startswith(prefix) and path.endswith(suffix):
                    rel_dir = path[len(prefix):-len(suffix)]
                    if rel_dir and not is_excluded_skill_path(PurePosixPath(rel_dir + suffix)):
                        dirs[rel_dir] = True
            return dirs or None

        self._remote_dirs = _memo_json("official_optional_dirs", compute,
                                       valid=lambda c: isinstance(c, dict) and bool(c)) or {}
        return self._remote_dirs

    def _upstream_pointer_from_content(self, content: Union[str, bytes]) -> Optional[Dict[str, str]]:
        """Parse ``metadata.hermes.upstream: {repo: owner/name, path: ...}`` out of SKILL.md content
        (a catalog stub); None for vendored skills."""
        if isinstance(content, bytes):
            try:
                content = content.decode("utf-8")
            except UnicodeDecodeError:
                return None
        meta_block = _parse_frontmatter(content).get("metadata")
        hermes_meta = meta_block.get("hermes") if isinstance(meta_block, dict) else None
        upstream = hermes_meta.get("upstream") if isinstance(hermes_meta, dict) else None
        if not isinstance(upstream, dict):
            return None
        repo = str(upstream.get("repo", "")).strip().strip("/")
        path = str(upstream.get("path", "")).strip().strip("/")
        # repo must be exactly owner/name; path must be a clean relative path.
        if not repo or repo.count("/") != 1 or not path:
            return None
        parts = _clean_rel_parts(path)
        return None if parts is None else {"repo": repo, "path": "/".join(parts)}

    def _fetch_from_upstream(self, upstream: Dict[str, str], rel_id: str) -> Optional[SkillBundle]:
        """Fetch an upstream-maintained optional skill via GitHubSource.fetch() (full-tree download,
        symlink/unsafe-path rejection, quarantine + scan downstream) and re-label it as an official
        catalog entry."""
        bundle = self._get_github().fetch(f"{upstream['repo']}/{upstream['path']}")
        if bundle is None:
            logger.warning("Upstream fetch failed for optional skill %s (%s:%s)",
                           rel_id, upstream["repo"], upstream["path"])
            return None
        return SkillBundle(
            name=bundle.name, files=bundle.files, source="official", identifier=f"official/{rel_id}",
            # Curated endorsement, but the content is live third-party:
            # "trusted", not "builtin", so a dangerous scan verdict still blocks.
            trust_level="trusted",
            metadata={**bundle.metadata, "upstream_repo": upstream["repo"], "upstream_path": upstream["path"]},
        )

    def _local_skill_mds(self):
        root = self._optional_dir
        return (md for md in (sorted(root.rglob("SKILL.md")) if root.is_dir() else [])
                if not is_excluded_skill_path(md.relative_to(root), root=root))

    def _find_skill_dir(self, name: str) -> Optional[Path]:
        """Find a skill directory by name anywhere in optional-skills/."""
        return next((md.parent for md in self._local_skill_mds() if md.parent.name == name), None)

    def _scan_all(self) -> List[SkillMeta]:
        """Enumerate all optional skills with metadata."""
        results: List[SkillMeta] = []
        for skill_md in self._local_skill_mds():
            parent = skill_md.parent
            try:
                content = skill_md.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            fm = _parse_frontmatter(content)
            tags = _hermes_tags(fm)
            results.append(self._meta(parent.relative_to(self._optional_dir).as_posix(), fm.get("name", parent.name),
                                      fm.get("description", "")[:200], tags if isinstance(tags, list) else []))
        return results


class HermesIndexSource(SkillSource):
    """Skill source backed by the centralized Hermes Skills Index: a JSON catalog on the docs site,
    rebuilt daily by CI, with metadata + resolved GitHub paths for every skill — search and path
    discovery cost zero GitHub API calls. When unavailable every method returns empty/None so
    downstream sources take over transparently."""

    SOURCE_ID = "hermes-index"

    def __init__(self, auth: GitHubAuth):
        self._index: Optional[dict] = None
        self._loaded = False
        self.auth = auth
        self._github: Optional[GitHubSource] = None  # only needed for fetch

    def _ensure_loaded(self) -> dict:
        if not self._loaded:
            from tools.skills_hub_search import _load_hermes_index
            self._index, self._loaded = _load_hermes_index(), True
        return self._index or {}

    def _skills(self) -> list:
        return self._ensure_loaded().get("skills", [])

    def _get_github(self) -> GitHubSource:
        if self._github is None:
            self._github = GitHubSource(auth=self.auth)
        return self._github

    @property
    def is_available(self) -> bool:
        """Whether the index is loaded and has skills."""
        return bool(self._skills())

    def trust_level_for(self, identifier: str) -> str:
        entry = next((s for s in self._skills() if s.get("identifier") == identifier), None)
        return entry.get("trust_level", "community") if entry else "community"

    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        """Search the cached index (zero API calls). Matches name, description, tags, identifier and
        ``extra.provider`` (so ``nvidia`` finds ``NVIDIA/skills/...`` entries stored as source
        "github"). Ranked exact name > name prefix > provider > whole-word > name substring > other,
        index order as tiebreaker — a raw break-at-limit slice buried the most relevant skills."""
        skills = self._skills()
        if not skills:
            return []
        if not query.strip():
            return [self._to_meta(s) for s in skills[:limit]]  # featured / index order
        query_lower = query.lower()
        scored: List[Tuple[int, int, dict]] = []
        for i, s in enumerate(skills):
            name = str(s.get("name", "")).lower()
            provider = str((s.get("extra") or {}).get("provider", "")).lower()
            haystack = " ".join([
                name, str(s.get("description", "")).lower(), " ".join(str(t).lower() for t in s.get("tags", [])),
                str(s.get("identifier", "")).lower(), provider,
            ])
            if query_lower not in haystack:
                continue
            ranks = (
                name == query_lower, name.startswith(query_lower), provider == query_lower,
                query_lower in name.split() or query_lower in provider.split(), query_lower in name, True,
            )
            scored.append((ranks.index(True), i, s))
        scored.sort(key=lambda x: (x[0], x[1]))
        return [self._to_meta(s) for _, _, s in scored[:limit]]

    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        """Fetch via the index's ``resolved_github_id`` (skipping the whole
        candidate/discovery chain), falling back to ``repo/path``."""
        entry = self._find_entry(identifier)
        if not entry:
            return None
        repo, path = entry.get("repo", ""), entry.get("path", "")
        candidates = [entry.get("resolved_github_id")] + ([f"{repo}/{path}"] if repo and path else [])
        for github_id in filter(None, candidates):
            bundle = self._get_github().fetch(github_id)
            if bundle:
                bundle.source = entry.get("source", "hermes-index")
                bundle.identifier = identifier
                return bundle
        return None

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        """Return metadata from the index (zero API calls)."""
        entry = self._find_entry(identifier)
        return self._to_meta(entry) if entry else None

    def _find_entry(self, identifier: str) -> Optional[dict]:
        """Exact identifier match first, then match with source prefixes stripped."""
        skills = self._skills()
        normalized = _strip_prefix(identifier, _INDEX_ID_PREFIXES)
        return next((s for s in skills if s.get("identifier") == identifier), None) or next(
            (s for s in skills if _strip_prefix(s.get("identifier", ""), _INDEX_ID_PREFIXES) == normalized), None,
        )

    @staticmethod
    def _to_meta(entry: dict) -> SkillMeta:
        return SkillMeta(
            name=entry.get("name", ""), description=entry.get("description", ""),
            source=entry.get("source", "hermes-index"), identifier=entry.get("identifier", ""),
            trust_level=entry.get("trust_level", "community"), repo=entry.get("repo"), path=entry.get("path"),
            tags=entry.get("tags", []), extra=entry.get("extra", {}),
        )
