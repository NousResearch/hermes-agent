"""Skills Hub adapters for well-known endpoints, direct URLs, LobeHub, and browse.sh."""

import hashlib
import json
import logging
import re
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote, urljoin, urlparse, urlunparse

from tools.skills_hub_models import (
    GuardedFetchMixin, SkillBundle, SkillMeta, SkillSource, _first_matching, _get_json, _get_text,
    _hermes_tags, _memo_json, _parse_frontmatter, _referenced_support_paths,
    _validate_bundle_rel_path, _validate_skill_name, hub,
)

logger = logging.getLogger("tools.skills_hub")


# --- Well-known Agent Skills endpoint source adapter ------------------------

class WellKnownSkillSource(GuardedFetchMixin, SkillSource):
    """Read skills from a domain exposing /.well-known/skills/index.json."""

    SOURCE_ID = "well-known"
    BASE_PATH = "/.well-known/skills"

    def _meta(self, parsed: dict, skill_name: str, description: str, files: Any, **extra) -> SkillMeta:
        return SkillMeta(
            name=skill_name, description=description, source="well-known",
            identifier=self._wrap_identifier(parsed["base_url"], skill_name), trust_level="community",
            path=skill_name,
            extra={"index_url": parsed["index_url"], "base_url": parsed["base_url"], "files": files, **extra},
        )

    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        index_url = self._query_to_index_url(query)
        parsed = self._parse_index(index_url) if index_url else None
        if not parsed:
            return []
        results: List[SkillMeta] = []
        for entry in parsed["skills"][:limit]:
            name = entry.get("name")
            if not isinstance(name, str) or not name:
                continue
            files = entry.get("files", ["SKILL.md"])
            results.append(self._meta(parsed, name, str(entry.get("description", "")),
                                      files if isinstance(files, list) else ["SKILL.md"]))
        return results

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        parsed = self._parse_identifier(identifier)
        entry = self._index_entry(parsed["index_url"], parsed["skill_name"]) if parsed else None
        skill_md = self._fetch_text(f"{parsed['skill_url']}/SKILL.md") if entry else None
        if skill_md is None:
            return None
        fm = _parse_frontmatter(skill_md)
        return self._meta(parsed, str(fm.get("name") or parsed["skill_name"]),
                          str(fm.get("description") or entry.get("description") or ""),
                          entry.get("files", ["SKILL.md"]), endpoint=parsed["skill_url"])

    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        parsed = self._parse_identifier(identifier)
        if not parsed:
            return None
        try:
            skill_name = _validate_skill_name(parsed["skill_name"])
        except ValueError:
            logger.warning("Well-known skill identifier contained unsafe skill name: %s", identifier)
            return None
        entry = self._index_entry(parsed["index_url"], parsed["skill_name"])
        if not entry:
            return None
        files = entry.get("files", ["SKILL.md"])
        if not isinstance(files, list) or not files:
            files = ["SKILL.md"]
        downloaded: Dict[str, str] = {}
        for rel_path in files:
            if not isinstance(rel_path, str) or not rel_path:
                continue
            try:
                safe_rel_path = _validate_bundle_rel_path(rel_path)
            except ValueError:
                logger.warning("Well-known skill %s advertised unsafe file path: %r", identifier, rel_path)
                return None
            text = self._fetch_text(f"{parsed['skill_url']}/{safe_rel_path}")
            if text is None:
                return None
            downloaded[safe_rel_path] = text
        if "SKILL.md" not in downloaded:
            return None
        return SkillBundle(
            name=skill_name, files=downloaded, source="well-known",
            identifier=self._wrap_identifier(parsed["base_url"], skill_name), trust_level="community",
            metadata={"index_url": parsed["index_url"], "base_url": parsed["base_url"],
                      "endpoint": parsed["skill_url"], "files": files},
        )

    def _query_to_index_url(self, query: str) -> Optional[str]:
        query = query.strip()
        if not query.startswith(("http://", "https://")):
            return None
        if query.endswith("/index.json"):
            return query
        if f"{self.BASE_PATH}/" in query:
            return query.split(f"{self.BASE_PATH}/", 1)[0] + f"{self.BASE_PATH}/index.json"
        return query.rstrip("/") + f"{self.BASE_PATH}/index.json"

    def _parse_identifier(self, identifier: str) -> Optional[dict]:
        raw = identifier[len("well-known:"):] if identifier.startswith("well-known:") else identifier
        if not raw.startswith(("http://", "https://")):
            return None
        parsed_url = urlparse(raw)
        clean_url = urlunparse(parsed_url._replace(fragment=""))
        if clean_url.endswith("/index.json"):
            if not parsed_url.fragment:
                return None
            base_url, skill_name = clean_url[:-len("/index.json")], parsed_url.fragment
            skill_url = f"{base_url}/{skill_name}"
        else:
            skill_url = clean_url[:-len("/SKILL.md")] if clean_url.endswith("/SKILL.md") else clean_url.rstrip("/")
            if f"{self.BASE_PATH}/" not in skill_url:
                return None
            base_url, skill_name = skill_url.rsplit("/", 1)
        return {"index_url": f"{base_url}/index.json", "base_url": base_url,
                "skill_name": skill_name, "skill_url": skill_url}

    def _parse_index(self, index_url: str) -> Optional[dict]:
        def compute():
            resp = hub()._guarded_http_get(index_url, timeout=20)
            if resp is None or resp.status_code != 200:
                return None
            try:
                data = resp.json()
            except json.JSONDecodeError:
                return None
            skills = data.get("skills", []) if isinstance(data, dict) else []
            if not isinstance(skills, list):
                return None
            return {"index_url": index_url, "base_url": index_url[:-len("/index.json")], "skills": skills}

        return _memo_json(f"well_known_index_{hashlib.md5(index_url.encode()).hexdigest()}", compute,
                          valid=lambda c: isinstance(c, dict) and isinstance(c.get("skills"), list))

    def _index_entry(self, index_url: str, skill_name: str) -> Optional[dict]:
        parsed = self._parse_index(index_url)
        skills = parsed["skills"] if parsed else []
        return next((e for e in skills if isinstance(e, dict) and e.get("name") == skill_name), None)

    @staticmethod
    def _wrap_identifier(base_url: str, skill_name: str) -> str:
        return f"well-known:{base_url.rstrip('/')}/{skill_name}"


# --- Direct URL source adapter ----------------------------------------------

class UrlSource(GuardedFetchMixin, SkillSource):
    """Fetch SKILL.md plus explicitly referenced, allowlisted support files.

    The identifier IS the URL (``https://example.com/path/SKILL.md``). Bare URLs cannot enumerate a
    repository, so only exact references below references/templates/scripts/assets are fetched. The
    skill name comes from frontmatter ``name:`` (URL-slug fallback); trust is always ``community``.
    """

    SOURCE_ID = "url"
    # Skill names must look like identifiers: lowercase letters/digits with optional hyphens/underscores.
    # Blocks dangerous (``../evil``) AND useless (``SKILL``, ``README``, empty) candidates before they hit the disk.
    _VALID_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]*$")

    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        return []  # search is meaningless for a direct URL

    def _matches(self, identifier: str) -> bool:
        """Claim bare HTTP(S) URLs ending in ``.md``; leave wrapped identifiers
        and ``/.well-known/skills/`` URLs to their own adapters."""
        if not isinstance(identifier, str):
            return False
        ident = identifier.strip()
        if (not ident.lower().startswith(("http://", "https://")) or "/.well-known/skills/" in ident
                or ident.rstrip("/").endswith("/index.json")):
            return False
        try:
            return urlparse(ident).path.lower().endswith(".md")
        except ValueError:
            return False

    def _load(self, identifier: str):
        """``(url, text, frontmatter, resolved name)`` for a claimed identifier, else None."""
        if not self._matches(identifier):
            return None
        url = identifier.strip()
        text = self._fetch_text(url)
        if text is None:
            return None
        fm = _parse_frontmatter(text)
        return url, text, fm, self._resolve_skill_name(fm, url)

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        loaded = self._load(identifier)
        if loaded is None:
            return None
        url, _text, fm, name = loaded
        raw_tags = _hermes_tags(fm)
        return SkillMeta(
            name=name or "", description=str(fm.get("description") or ""), source="url", identifier=url,
            trust_level="community", path=name or "",
            tags=[str(t) for t in raw_tags] if isinstance(raw_tags, list) else [],
            extra={"url": url, "awaiting_name": name is None},
        )

    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        loaded = self._load(identifier)
        if loaded is None:
            return None
        url, text, _fm, name = loaded
        referenced = _referenced_support_paths(text)
        if referenced is None:
            return None
        files: Dict[str, Union[str, bytes]] = {"SKILL.md": text}
        base_url = url.rsplit("/", 1)[0] + "/"
        for rel_path in sorted(referenced):
            support_url = urljoin(base_url, quote(rel_path, safe="/"))
            if urlparse(support_url).netloc != urlparse(url).netloc:
                return None
            content = self._fetch_bytes(support_url)
            if content is None:  # A 404ing support file shouldn't sink the whole install.
                logger.warning("URL skill %s: referenced support file %r could not be fetched from %s; skipping it",
                               url, rel_path, support_url)
                continue
            files[rel_path] = content
        # When no name resolves, return the bundle with an empty name and ``awaiting_name=True``: ``do_install``
        # prompts on a TTY or refuses non-interactively, without re-downloading after the user picks a name.
        skill_name = ""
        if name is not None:
            try:
                skill_name = _validate_skill_name(name)
            except ValueError:
                logger.warning("URL skill %s produced unsafe skill name: %r", url, name)
                return None
        return SkillBundle(name=skill_name, files=files, source="url", identifier=url, trust_level="community",
                           metadata={"url": url, "source_url": url, "awaiting_name": not skill_name})

    @classmethod
    def _is_valid_skill_name(cls, name: Optional[str]) -> bool:
        if not isinstance(name, str):
            return False
        candidate = name.strip().lower()
        return bool(candidate) and candidate not in {"skill", "readme", "index", "unnamed-skill"} and bool(
            cls._VALID_NAME_RE.match(candidate))

    @classmethod
    def _resolve_skill_name(cls, fm: dict, url: str) -> Optional[str]:
        """Frontmatter ``name:`` when valid, else a URL-slug candidate (``.../<name>/SKILL.md`` -> ``<name>``,
        ``.../<name>.md`` -> ``<name>``). None when nothing usable — the CLI then prompts or refuses rather
        than auto-naming something like ``SKILL``."""
        fm_name = fm.get("name") if isinstance(fm, dict) else None
        if isinstance(fm_name, str) and cls._is_valid_skill_name(fm_name):
            return fm_name.strip()
        try:
            path = urlparse(url).path
        except ValueError:
            return None
        parts = [p for p in path.split("/") if p]
        if len(parts) >= 2 and parts[-1].lower() == "skill.md" and cls._is_valid_skill_name(parts[-2]):
            return parts[-2]
        candidate = re.sub(r"\.md$", "", parts[-1], flags=re.IGNORECASE) if parts else ""
        return candidate if cls._is_valid_skill_name(candidate) else None


# --- LobeHub source adapter -------------------------------------------------

class LobeHubSource(SkillSource):
    """LobeHub agent marketplace (14,500+ system-prompt agents, converted to
    SKILL.md on fetch). Data lives in GitHub: lobehub/lobe-chat-agents."""

    SOURCE_ID = "lobehub"
    INDEX_URL = "https://chat-agents.lobehub.com/index.json"

    def _agents(self) -> Optional[list]:
        index = self._fetch_index()
        agents = (index.get("agents", index) if isinstance(index, dict) else index) if index else None
        return agents if isinstance(agents, list) else None

    @staticmethod
    def _agent_id(identifier: str) -> str:
        return identifier.split("/", 1)[-1] if identifier.startswith("lobehub/") else identifier

    @staticmethod
    def _agent_meta(agent: dict, name: str, description: str) -> SkillMeta:
        tags = agent.get("meta", agent).get("tags", [])
        return SkillMeta(name=name, description=description, source="lobehub", identifier=f"lobehub/{name}",
                         trust_level="community", tags=tags if isinstance(tags, list) else [])

    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        agents = self._agents()
        if agents is None:
            return []

        def fields(agent):
            meta = agent.get("meta", agent)
            tags = meta.get("tags", [])
            return (meta.get("title", agent.get("identifier", "")), meta.get("description", ""),
                    tags if isinstance(tags, list) else "")

        def to_meta(agent):
            meta = agent.get("meta", agent)
            title = meta.get("title", agent.get("identifier", ""))
            identifier = agent.get("identifier", title.lower().replace(" ", "-"))
            return self._agent_meta(agent, identifier, meta.get("description", "")[:200])

        return _first_matching(query.lower(), agents, fields, to_meta, limit)

    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        agent_id = self._agent_id(identifier)
        agent_data = self._fetch_agent(agent_id)
        if not agent_data:
            return None
        return SkillBundle(name=agent_id, files={"SKILL.md": self._convert_to_skill_md(agent_data)}, source="lobehub",
                           identifier=f"lobehub/{agent_id}", trust_level="community")

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        agent_id = self._agent_id(identifier)
        agent = next((a for a in self._agents() or [] if a.get("identifier") == agent_id), None)
        return self._agent_meta(agent, agent_id, agent.get("meta", agent).get("description", "")) if agent else None

    def _fetch_index(self) -> Optional[Any]:
        return _memo_json("lobehub_index", lambda: _get_json(self.INDEX_URL, timeout=30))

    def _fetch_agent(self, agent_id: str) -> Optional[dict]:
        return _get_json(f"https://chat-agents.lobehub.com/{agent_id}.json", timeout=15)

    @staticmethod
    def _convert_to_skill_md(agent_data: dict) -> str:
        """Convert a LobeHub agent JSON into SKILL.md format."""
        meta = agent_data.get("meta", agent_data)
        identifier = agent_data.get("identifier", "lobehub-agent")
        title = meta.get("title", identifier)
        description = meta.get("description", "")
        tags = meta.get("tags", [])
        tag_list = tags if isinstance(tags, list) else []
        system_role = agent_data.get("config", {}).get("systemRole", "")
        fm_lines = ["---", f"name: {identifier}", f"description: {description[:500]}", "metadata:", "  hermes:",
                    f"    tags: [{', '.join(str(t) for t in tag_list)}]", "  lobehub:", "    source: lobehub", "---"]
        body_lines = [f"# {title}", "", description, "", "## Instructions", "",
                      system_role if system_role else "(No system role defined)"]
        return "\n".join(fm_lines) + "\n\n" + "\n".join(body_lines) + "\n"


# --- browse.sh source adapter -----------------------------------------------

class BrowseShSource(SkillSource):
    """Browserbase's browse.sh catalog of site-specific browser-automation SKILL.md files.

    The catalog is ``/api/skills``; content comes from ``/api/skills/{slug}``'s ``skillMdUrl`` (CDN blob).
    The catalog's ``sourceUrl`` is a GitHub HTML URL whose repo is not always public, so it is not used for content.
    """

    SOURCE_ID = "browse-sh"
    CATALOG_URL = "https://browse.sh/api/skills"
    SKILL_DETAIL_URL = "https://browse.sh/api/skills/{slug}"
    _CACHE_KEY = "browse_sh_catalog"

    def _fetch_catalog(self) -> List[Dict]:
        def compute():
            data = _get_json(self.CATALOG_URL)
            skills = data.get("skills", []) if isinstance(data, dict) else []
            return skills if isinstance(skills, list) else None

        return _memo_json(self._CACHE_KEY, compute) or []

    def _item_to_meta(self, item: Dict) -> Optional[SkillMeta]:
        slug = item.get("slug", "")
        name = item.get("name", "")
        description = item.get("description", item.get("title", name))
        if not slug or not name:
            return None
        if len(description) > 1024:
            description = description[:1021] + "..."
        return SkillMeta(
            name=name, description=description, source="browse-sh", identifier=f"browse-sh/{slug}",
            trust_level="community", tags=item.get("tags", []),
            extra={"slug": slug, "hostname": item.get("hostname", ""), "category": item.get("category", ""),
                   "source_url": item.get("sourceUrl", ""), "recommended_method": item.get("recommendedMethod", ""),
                   "proxies": item.get("proxies", False), "install_count": item.get("installCount", 0)},
        )

    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        def fields(item):
            return (item.get("name", ""), item.get("title", ""), item.get("description", ""),
                    item.get("hostname", ""), item.get("category", ""), item.get("tags", []))

        return _first_matching(query.lower(), self._fetch_catalog(), fields, self._item_to_meta, limit)

    def _catalog_item(self, identifier: str) -> Optional[Dict]:
        slug = self._slug_from_identifier(identifier)
        return next((i for i in self._fetch_catalog() if i.get("slug") == slug), None) if slug else None

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        item = self._catalog_item(identifier)
        return self._item_to_meta(item) if item else None

    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        item = self._catalog_item(identifier)
        if not item:
            return None
        slug = item["slug"]
        md_url = self._resolve_skill_md_url(slug, item)
        content = _get_text(md_url, follow_redirects=True) if md_url else None
        if content is None:
            return None
        meta = self._item_to_meta(item)
        return SkillBundle(
            name=meta.name if meta else slug.split("/")[-1], files={"SKILL.md": content}, source="browse-sh",
            identifier=identifier, trust_level="community",
            metadata={"slug": slug, "hostname": item.get("hostname", ""), "source_url": item.get("sourceUrl", ""),
                      "skill_md_url": md_url},
        )

    def _resolve_skill_md_url(self, slug: str, item: Dict) -> Optional[str]:
        """``skillMdUrl`` from ``/api/skills/{slug}``; fallback to a ``raw.githubusercontent.com`` ``sourceUrl``."""
        data = _get_json(self.SKILL_DETAIL_URL.format(slug=slug), follow_redirects=True)
        md_url = data.get("skillMdUrl") if isinstance(data, dict) else None
        if isinstance(md_url, str) and md_url.startswith("http"):
            return md_url
        source_url = item.get("sourceUrl", "") if isinstance(item, dict) else ""
        from utils import base_url_host_matches
        return source_url if source_url and base_url_host_matches(source_url, "raw.githubusercontent.com") else None

    def _slug_from_identifier(self, identifier: str) -> str:
        """'browse-sh/airbnb.com/search-listings-abc' -> 'airbnb.com/search-listings-abc'."""
        return identifier[len("browse-sh/"):] if identifier.startswith("browse-sh/") else identifier
