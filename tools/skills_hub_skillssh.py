"""Skills Hub skills.sh adapter: catalog discovery via skills.sh, content via GitHub."""

import hashlib
import logging
import re
from typing import Any, Dict, List, Optional

from tools.skills_hub_github import GitHubAuth, GitHubSource, _split_repo_id
from tools.skills_hub_models import (
    SkillBundle, SkillMeta, SkillSource, _cache_metas, _cached_metas, _get_json, _get_text, _memo_json,
)

logger = logging.getLogger("tools.skills_hub")


def _strip_html(value: str) -> str:
    return re.sub(r'<[^>]+>', '', value)


class SkillsShSource(SkillSource):
    """Discover skills via skills.sh and fetch content from the underlying GitHub repo."""

    BASE_URL = "https://skills.sh"
    SEARCH_URL = f"{BASE_URL}/api/search"
    # The sitemap is the real catalog source: the homepage only exposes a
    # ~200-entry featured strip; sitemap.xml points at sitemap-skills-N.xml
    # files (10k URLs each) covering the full ~20k+ catalog.
    SITEMAP_INDEX_URL = "https://www.skills.sh/sitemap.xml"
    # skills.sh serves per-skill sitemaps brotli-compressed and httpx's optional
    # brotlicffi backend has a streaming-decode bug on them; asking for gzip
    # only makes the server fall back to gzip/identity on every httpx install.
    _SITEMAP_HEADERS = {"Accept-Encoding": "gzip"}
    _SITEMAP_LOC_RE = re.compile(r"<loc>([^<]+)</loc>", re.IGNORECASE)
    _SITEMAP_SKILL_RE = re.compile(
        r"^https?://(?:www\.)?skills\.sh/(?P<owner>[^/]+)/(?P<repo>[^/]+)/(?P<skill>[^/]+)/?$", re.IGNORECASE,
    )
    _SKILL_LINK_RE = re.compile(r'href=["\']/(?P<id>(?!agents/|_next/|api/)[^"\'/]+/[^"\'/]+/[^"\'/]+)["\']')
    _INSTALL_CMD_RE = re.compile(
        r'npx\s+skills\s+add\s+(?P<repo>https?://github\.com/[^\s<]+|[^\s<]+)'
        r'(?:\s+--skill\s+(?P<skill>[^\s<]+))?',
        re.IGNORECASE,
    )
    _PAGE_H1_RE = re.compile(r'<h1[^>]*>(?P<title>.*?)</h1>', re.IGNORECASE | re.DOTALL)
    _PROSE_H1_RE = re.compile(
        r'<div[^>]*class=["\'][^"\']*prose[^"\']*["\'][^>]*>.*?<h1[^>]*>(?P<title>.*?)</h1>',
        re.IGNORECASE | re.DOTALL,
    )
    _PROSE_P_RE = re.compile(
        r'<div[^>]*class=["\'][^"\']*prose[^"\']*["\'][^>]*>.*?<p[^>]*>(?P<body>.*?)</p>',
        re.IGNORECASE | re.DOTALL,
    )
    _WEEKLY_INSTALLS_RE = re.compile(r'Weekly Installs.*?children\\":\\"(?P<count>[0-9.,Kk]+)\\"', re.DOTALL)
    _ID_PREFIX_ALIASES = ("skills-sh/", "skills.sh/", "skils-sh/", "skils.sh/")
    # Standard skill sub-paths tried before any tree/discovery walk.
    _STANDARD_BASE_PATHS = ("skills/", ".agents/skills/", ".claude/skills/")

    _strip_html = staticmethod(_strip_html)
    SOURCE_ID = "skills-sh"

    def __init__(self, auth: GitHubAuth):
        self.auth, self.github = auth, GitHubSource(auth=auth)

    def trust_level_for(self, identifier: str) -> str:
        return self.github.trust_level_for(self._normalize_identifier(identifier))

    def _meta(self, canonical: str, *, name: str, description: str, path: str,
              extra: Optional[Dict[str, Any]] = None) -> SkillMeta:
        return SkillMeta(
            name=name, description=description, source="skills.sh", identifier=self._wrap_identifier(canonical),
            trust_level=self.github.trust_level_for(canonical), repo="/".join(canonical.split("/", 2)[:2]),
            path=path, extra=extra if extra is not None else {},
        )

    def _urls_for(self, canonical: str, repo: str) -> Dict[str, str]:
        return {"detail_url": f"{self.BASE_URL}/{canonical}", "repo_url": f"https://github.com/{repo}"}

    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        if not query.strip():
            # Empty query = bulk catalog dump (build_skills_index.py) — walk the sitemap.
            return self._sitemap_catalog(limit)
        cache_key = f"skills_sh_search_{hashlib.md5(f'{query}|{limit}'.encode()).hexdigest()}"
        cached = _cached_metas(cache_key)
        if cached is not None:
            return cached[:limit]
        data = _get_json(self.SEARCH_URL, params={"q": query, "limit": limit})
        if data is None:
            return []
        items = data.get("skills", []) if isinstance(data, dict) else []
        if not isinstance(items, list):
            return []
        results = [m for m in map(self._meta_from_search_item, items[:limit]) if m]
        _cache_metas(cache_key, results)
        return results

    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        canonical = self._normalize_identifier(identifier)
        detail = self._fetch_detail_page(canonical)

        def _relabel(github_id: Optional[str]) -> Optional[SkillBundle]:
            bundle = self.github.fetch(github_id) if github_id else None
            if bundle:
                bundle.source, bundle.identifier = "skills.sh", self._wrap_identifier(canonical)
                bundle.metadata.update(self._detail_to_metadata(canonical, detail))
            return bundle or None

        for candidate in self._candidate_identifiers(canonical):
            bundle = _relabel(candidate)
            if bundle:
                return bundle
        return _relabel(self._discover_identifier(canonical, detail=detail))

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        canonical = self._normalize_identifier(identifier)
        detail = self._fetch_detail_page(canonical)
        meta = self._resolve_github_meta(canonical, detail=detail)
        return self._finalize_inspect_meta(meta, canonical, detail) if meta else None

    def _sitemap_catalog(self, limit: int) -> List[SkillMeta]:
        """Enumerate the full catalog via the sitemap (cached for the index TTL —
        ~2 MB of XML). Falls back to ``_featured_skills`` when unreachable/empty."""
        cache_key = "skills_sh_sitemap_v1"
        cached = _cached_metas(cache_key)
        if cached is not None:
            return cached[:limit] if limit > 0 else cached

        # Step 1: sitemap index -> per-skill sitemap URLs.
        index_xml = _get_text(self.SITEMAP_INDEX_URL, follow_redirects=True, headers=self._SITEMAP_HEADERS)
        skill_sitemap_urls = [m.group(1).strip() for m in self._SITEMAP_LOC_RE.finditer(index_xml or "")
                              if "sitemap-skills" in m.group(1)]
        if not skill_sitemap_urls:
            return self._featured_skills(limit)

        # Step 2: collect canonical "owner/repo/skill" IDs from each sitemap.
        seen, results = set(), []
        for sitemap_url in skill_sitemap_urls:
            xml = _get_text(sitemap_url, timeout=30, follow_redirects=True, headers=self._SITEMAP_HEADERS)
            for loc_match in self._SITEMAP_LOC_RE.finditer(xml or ""):
                m = self._SITEMAP_SKILL_RE.match(loc_match.group(1).strip())
                if not m:
                    continue
                repo, skill = f"{m.group('owner')}/{m.group('repo')}", m.group("skill")
                canonical = f"{repo}/{skill}"
                if canonical not in seen:
                    seen.add(canonical)
                    results.append(self._meta(canonical, name=skill, description=f"Indexed by skills.sh from {repo}",
                                              path=skill, extra=self._urls_for(canonical, repo)))
        if not results:
            return self._featured_skills(limit)
        _cache_metas(cache_key, results)
        return results[:limit] if limit > 0 else results

    def _featured_skills(self, limit: int) -> List[SkillMeta]:
        cache_key = "skills_sh_featured"
        cached = _cached_metas(cache_key)
        if cached is not None:
            return cached[:limit]
        html = _get_text(self.BASE_URL)
        if html is None:
            return []
        seen, results = set(), []
        for match in self._SKILL_LINK_RE.finditer(html):
            canonical = match.group("id")
            split = None if canonical in seen else _split_repo_id(canonical)
            seen.add(canonical)
            if split is None:
                continue
            repo, skill_path = split
            results.append(self._meta(canonical, name=skill_path.split("/")[-1],
                                      description=f"Featured on skills.sh from {repo}", path=skill_path))
            if len(results) >= limit:
                break
        _cache_metas(cache_key, results)
        return results

    def _meta_from_search_item(self, item: dict) -> Optional[SkillMeta]:
        if not isinstance(item, dict):
            return None
        canonical, repo, skill_path = item.get("id"), item.get("source"), item.get("skillId")
        if not isinstance(canonical, str) or canonical.count("/") < 2:
            if not (isinstance(repo, str) and isinstance(skill_path, str)):
                return None
            canonical = f"{repo}/{skill_path}"
        split = _split_repo_id(canonical)
        if split is None:
            return None
        repo, skill_path = split
        installs = item.get("installs")
        installs_label = f" · {int(installs):,} installs" if isinstance(installs, int) else ""
        return self._meta(
            canonical, name=str(item.get("name") or skill_path.split("/")[-1]),
            description=f"Indexed by skills.sh from {repo}{installs_label}", path=skill_path,
            extra={"installs": installs, **self._urls_for(canonical, repo)},
        )

    def _fetch_detail_page(self, identifier: str) -> Optional[dict]:
        def compute():
            html = _get_text(f"{self.BASE_URL}/{identifier}")
            return None if html is None else self._parse_detail_page(identifier, html) or None
        key = f"skills_sh_detail_{hashlib.md5(identifier.encode()).hexdigest()}"
        return _memo_json(key, compute, valid=lambda c: isinstance(c, dict))

    def _parse_detail_page(self, identifier: str, html: str) -> Optional[dict]:
        split = _split_repo_id(identifier)
        if split is None:
            return None
        repo, install_skill = split
        install_command, install_match = None, self._INSTALL_CMD_RE.search(html)
        if install_match:
            install_command = install_match.group(0).strip()
            install_skill = (install_match.group("skill") or install_skill).strip()
            repo = self._extract_repo_slug((install_match.group("repo") or "").strip()) or repo
        return {
            "repo": repo, "install_skill": install_skill,
            "page_title": self._extract_first_match(self._PAGE_H1_RE, html),
            "body_title": self._extract_first_match(self._PROSE_H1_RE, html),
            "body_summary": self._extract_first_match(self._PROSE_P_RE, html),
            "weekly_installs": self._extract_weekly_installs(html),
            "install_command": install_command,
            **self._urls_for(identifier, repo),
            "security_audits": self._extract_security_audits(html, identifier),
        }

    def _discover_identifier(self, identifier: str, detail: Optional[dict] = None) -> Optional[str]:
        split = _split_repo_id(identifier)
        if split is None:
            return None
        default_repo, skill_path = split
        repo = detail.get("repo", default_repo) if isinstance(detail, dict) else default_repo
        skill_token = skill_path.split("/")[-1]
        tokens = [skill_token]
        if isinstance(detail, dict):
            tokens.extend(detail.get(k, "") for k in ("install_skill", "page_title", "body_title"))

        def _match_in(base_path: str) -> Optional[str]:
            try:
                skills = self.github._list_skills_in_repo(repo, base_path)
            except Exception:
                return None
            return next((m.identifier for m in skills if self._matches_skill_tokens(m, tokens)), None)

        # One recursive tree lookup before brute-forcing every top-level dir
        # (avoids request bursts on categorized repos like borghei/claude-skills).
        found = (next((f for f in map(_match_in, self._STANDARD_BASE_PATHS) if f), None)
                 or self.github._find_skill_in_repo_tree(repo, skill_token))
        if found:
            return found

        # Fallback: scan repo root for directories that might contain skills.
        try:
            entries = _get_json(f"https://api.github.com/repos/{repo}/contents/",
                                headers=self.github.auth.get_headers(), timeout=15, follow_redirects=True)
            for entry in entries if isinstance(entries, list) else []:
                if entry.get("type") != "dir":
                    continue
                dir_name = entry["name"]
                if dir_name.startswith((".", "_")) or dir_name in {"skills", ".agents", ".claude"}:
                    continue
                meta = self.github.inspect(f"{repo}/{dir_name}/{skill_token}")
                if meta:
                    return meta.identifier
                found = _match_in(dir_name + "/")
                if found:
                    return found
        except Exception:
            pass
        return None

    def _resolve_github_meta(self, identifier: str, detail: Optional[dict] = None) -> Optional[SkillMeta]:
        for candidate in self._candidate_identifiers(identifier):
            meta = self.github.inspect(candidate)
            if meta:
                return meta
        resolved = self._discover_identifier(identifier, detail=detail)
        return self.github.inspect(resolved) if resolved else None

    def _finalize_inspect_meta(self, meta: SkillMeta, canonical: str, detail: Optional[dict]) -> SkillMeta:
        meta.source, meta.identifier = "skills.sh", self._wrap_identifier(canonical)
        meta.trust_level = self.trust_level_for(canonical)
        meta.extra = {**meta.extra, **self._detail_to_metadata(canonical, detail)}
        if isinstance(detail, dict):
            body_summary, weekly_installs = detail.get("body_summary"), detail.get("weekly_installs")
            if body_summary:
                meta.description = body_summary
            elif meta.description and weekly_installs:
                meta.description = f"{meta.description} · {weekly_installs} weekly installs on skills.sh"
        return meta

    @classmethod
    def _matches_skill_tokens(cls, meta: SkillMeta, skill_tokens: List[str]) -> bool:
        candidates = (cls._token_variants(meta.name) | cls._token_variants(meta.path)
                      | cls._token_variants(meta.identifier.split("/", 2)[-1] if meta.identifier else None))
        return any(cls._token_variants(token) & candidates for token in skill_tokens)

    @staticmethod
    def _token_variants(value: Optional[str]) -> set[str]:
        if not value:
            return set()
        plain = _strip_html(str(value)).strip().strip("/").lower()
        if not plain:
            return set()
        base, sanitized = plain.split("/")[-1], re.sub(r'[^a-z0-9/_-]+', '-', plain).strip('-')
        tail = base.lstrip('@')
        variants = {
            plain, plain.replace("_", "-"), plain.replace("/", "-"), base, base.replace("_", "-"),
            sanitized, sanitized.replace("/", "-"), sanitized.split("/")[-1], tail, tail.replace("_", "-"),
        }
        return {v for v in variants if v}

    @staticmethod
    def _extract_repo_slug(repo_value: str) -> Optional[str]:
        parts = repo_value.strip().removeprefix("https://github.com/").strip("/").split("/")
        return f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else None

    @staticmethod
    def _extract_first_match(pattern: re.Pattern, text: str) -> Optional[str]:
        match = pattern.search(text)
        value = next((group for group in match.groups() if group), None) if match else None
        return (_strip_html(value).strip() or None) if value is not None else None

    def _detail_to_metadata(self, canonical: str, detail: Optional[dict]) -> Dict[str, Any]:
        parts = canonical.split("/", 2)
        metadata = {"detail_url": f"{self.BASE_URL}/{canonical}"}
        if len(parts) >= 2:
            metadata["repo_url"] = f"https://github.com/{parts[0]}/{parts[1]}"
        if isinstance(detail, dict):
            for key in ("weekly_installs", "install_command", "repo_url", "detail_url", "security_audits"):
                if detail.get(key):
                    metadata[key] = detail[key]
        return metadata

    @classmethod
    def _extract_weekly_installs(cls, html: str) -> Optional[str]:
        match = cls._WEEKLY_INSTALLS_RE.search(html)
        return match.group("count") if match else None

    @staticmethod
    def _extract_security_audits(html: str, identifier: str) -> Dict[str, str]:
        audits: Dict[str, str] = {}
        for audit in ("agent-trust-hub", "socket", "snyk"):
            idx = html.find(f"/security/{audit}")
            match = re.search(r'(Pass|Warn|Fail)', html[idx:idx + 500], re.IGNORECASE) if idx != -1 else None
            if match:
                audits[audit] = match.group(1).title()
        return audits

    @classmethod
    def _normalize_identifier(cls, identifier: str) -> str:
        prefix = next((p for p in cls._ID_PREFIX_ALIASES if identifier.startswith(p)), "")
        return identifier[len(prefix):]

    @classmethod
    def _candidate_identifiers(cls, identifier: str) -> List[str]:
        split = _split_repo_id(identifier)
        if split is None:
            return [identifier]
        repo, path = split[0], split[1].lstrip("/")
        return list(dict.fromkeys([f"{repo}/{path}"] + [f"{repo}/{b}{path}" for b in cls._STANDARD_BASE_PATHS]))

    @staticmethod
    def _wrap_identifier(identifier: str) -> str:
        return f"skills-sh/{identifier}"
