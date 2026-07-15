from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urljoin

import httpx

from hermes_constants import get_hermes_dir

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_MS = 8000
DEFAULT_REGISTRY_TTL_SECONDS = 86400
DEFAULT_MAX_PAGES_PER_QUERY = 6
DEFAULT_MIN_SELECT_SCORE = 70
DEFAULT_FRESHNESS_TRIGGER_TERMS = [
    "latest",
    "current",
    "recently",
    "recent",
    "today",
    "this week",
    "this month",
    "new",
    "just released",
    "roadmap",
    "upcoming",
]
DEFAULT_REGISTRY_INDEX_URL = "https://agentwikis.com/index.json"
DEFAULT_REGISTRY_LLMSTXT_URL = "https://agentwikis.com/llms.txt"

_MUTABLE_TOPIC_TERMS = {
    "pricing",
    "price",
    "billing",
    "cost",
    "policy",
    "policies",
    "release",
    "releases",
    "roadmap",
    "status",
    "availability",
    "plan",
    "plans",
}
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+_.-]*")
_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2}|20\d{2}-\d{2}|20\d{2})\b")
_RAW_LINK_RE = re.compile(r"\((/raw/[^)]+)\)")


@dataclass
class AgentWikiDecision:
    selected_source: str
    reason: str
    selected_wiki_slug: Optional[str] = None
    current_as_used: Optional[str] = None
    pages_fetched_count: int = 0
    fallback_occurred: bool = False
    metadata: Optional[Dict[str, Any]] = None

    def log_record(self, query: str) -> Dict[str, Any]:
        record = {
            "query_hash": hashlib.sha256(_normalize_query(query).encode("utf-8")).hexdigest()[:12],
            "selected_source": self.selected_source,
            "selected_wiki_slug": self.selected_wiki_slug,
            "selection_reason": self.reason,
            "current_as_used": self.current_as_used,
            "pages_fetched_count": self.pages_fetched_count,
            "fallback_occurred": self.fallback_occurred,
        }
        if self.metadata:
            record.update(self.metadata)
        logger.info("agentwiki_routing_decision %s", json.dumps(record, sort_keys=True, ensure_ascii=False))
        return record


@dataclass
class AgentWikiResult:
    response_data: Optional[Dict[str, Any]]
    decision: AgentWikiDecision


class AgentWikiError(RuntimeError):
    pass


class AgentWikiRouter:
    def __init__(self, web_config: Dict[str, Any], *, client: Optional[httpx.Client] = None):
        self.web_config = web_config or {}
        self.cfg = dict((self.web_config.get("agentwikis") or {}))
        self.client = client
        self.cache_dir = get_hermes_dir("cache/agentwikis", "agentwikis_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def enabled(self) -> bool:
        return bool(self.cfg.get("enabled", False))

    def shadow_mode(self) -> bool:
        return bool(self.cfg.get("shadow_mode", False))

    def maybe_route_search(self, query: str, limit: int = 5) -> AgentWikiResult:
        if not self.enabled() or not query or not query.strip():
            return AgentWikiResult(None, AgentWikiDecision(selected_source="web_search", reason="disabled"))

        try:
            registry = self._load_registry()
            selection = self._select_wiki(query, registry)
            if selection["decision"].reason != "success":
                return AgentWikiResult(None, selection["decision"])

            wiki = selection["wiki"]
            retrieval = self._retrieve_pages(query, wiki, limit=limit)
            decision = selection["decision"]
            decision.pages_fetched_count = len(retrieval["results"])
            decision.metadata = {
                **(decision.metadata or {}),
                "retrieval_urls": [r.get("url") for r in retrieval["results"]],
            }
            if not retrieval["results"]:
                decision.selected_source = "web_search"
                decision.reason = retrieval.get("fallback_reason", "raw_unavailable")
                decision.fallback_occurred = True
                return AgentWikiResult(None, decision)

            if self.shadow_mode():
                decision.selected_source = "web_search"
                decision.reason = "shadow_mode"
                decision.fallback_occurred = True
                return AgentWikiResult(None, decision)

            response = {
                "success": True,
                "data": {"web": retrieval["results"]},
                "source_routing": {
                    "selected_source": "agentwikis",
                    "selected_wiki_slug": wiki["slug"],
                    "selection_reason": "success",
                    "current_as_used": decision.current_as_used,
                    "pages_fetched_count": len(retrieval["results"]),
                    "fallback_occurred": False,
                    "registry_metadata": {
                        "title": wiki.get("title"),
                        "raw_base": wiki.get("raw_base"),
                        "html_base": wiki.get("html_base"),
                    },
                    "retrieval_plan": retrieval.get("retrieval_plan", []),
                },
            }
            return AgentWikiResult(response, decision)
        except AgentWikiError as exc:
            return AgentWikiResult(None, AgentWikiDecision(selected_source="web_search", reason=str(exc), fallback_occurred=True))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Agentwiki routing failed open: %s", exc)
            return AgentWikiResult(None, AgentWikiDecision(selected_source="web_search", reason="router_error", fallback_occurred=True))

    def _timeout_s(self) -> float:
        timeout_ms = ((self.cfg.get("retrieval") or {}).get("timeout_ms") or DEFAULT_TIMEOUT_MS)
        try:
            timeout_ms = int(timeout_ms)
        except (TypeError, ValueError):
            timeout_ms = DEFAULT_TIMEOUT_MS
        return max(timeout_ms, 1000) / 1000.0

    def _registry_urls(self) -> tuple[str, str]:
        registry = self.cfg.get("registry") or {}
        return (
            registry.get("index_json_url") or DEFAULT_REGISTRY_INDEX_URL,
            registry.get("llms_txt_url") or DEFAULT_REGISTRY_LLMSTXT_URL,
        )

    def _registry_ttl(self) -> int:
        registry = self.cfg.get("registry") or {}
        try:
            return max(int(registry.get("refresh_ttl_seconds", DEFAULT_REGISTRY_TTL_SECONDS)), 0)
        except (TypeError, ValueError):
            return DEFAULT_REGISTRY_TTL_SECONDS

    def _conditional_get_enabled(self) -> bool:
        return bool((self.cfg.get("registry") or {}).get("conditional_get", True))

    def _max_pages_per_query(self) -> int:
        retrieval = self.cfg.get("retrieval") or {}
        try:
            return max(1, min(int(retrieval.get("max_pages_per_query", DEFAULT_MAX_PAGES_PER_QUERY)), 20))
        except (TypeError, ValueError):
            return DEFAULT_MAX_PAGES_PER_QUERY

    def _retry_html_markdown(self) -> bool:
        return bool((self.cfg.get("retrieval") or {}).get("retry_html_accept_markdown", True))

    def _min_select_score(self) -> int:
        routing = self.cfg.get("routing") or {}
        try:
            return max(0, int(routing.get("min_select_score", DEFAULT_MIN_SELECT_SCORE)))
        except (TypeError, ValueError):
            return DEFAULT_MIN_SELECT_SCORE

    def _freshness_trigger_terms(self) -> List[str]:
        routing = self.cfg.get("routing") or {}
        terms = routing.get("freshness_trigger_terms")
        if isinstance(terms, list) and terms:
            return [str(t).strip().lower() for t in terms if str(t).strip()]
        return list(DEFAULT_FRESHNESS_TRIGGER_TERMS)

    def _domains_cfg(self) -> Dict[str, Dict[str, Any]]:
        return dict((self.cfg.get("domains") or {}))

    def _cache_paths(self) -> tuple[Path, Path]:
        return self.cache_dir / "registry.json", self.cache_dir / "registry.meta.json"

    def _load_registry(self) -> Dict[str, Any]:
        body_path, meta_path = self._cache_paths()
        cached_body = None
        cached_meta: Dict[str, Any] = {}
        if body_path.exists():
            try:
                cached_body = json.loads(body_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                cached_body = None
        if meta_path.exists():
            try:
                cached_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                cached_meta = {}

        now_ts = datetime.utcnow().timestamp()
        is_fresh = bool(cached_body) and cached_meta.get("fetched_at") and (now_ts - float(cached_meta["fetched_at"]) < self._registry_ttl())

        if is_fresh:
            registry = cached_body
        else:
            registry = self._refresh_registry(cached_body, cached_meta)

        if not registry:
            raise AgentWikiError("registry_unavailable")

        self._warn_invalid_configured_slugs(registry)
        return registry

    def _refresh_registry(self, cached_body: Optional[Dict[str, Any]], cached_meta: Dict[str, Any]) -> Dict[str, Any]:
        index_url, llms_url = self._registry_urls()
        errors: List[str] = []
        for kind, url in (("index_json", index_url), ("llms_txt", llms_url)):
            try:
                data, response_meta = self._fetch_registry_url(url, kind, cached_meta if kind == "index_json" else {})
                if data:
                    body_path, meta_path = self._cache_paths()
                    body_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                    response_meta["fetched_at"] = datetime.utcnow().timestamp()
                    meta_path.write_text(json.dumps(response_meta, ensure_ascii=False, indent=2), encoding="utf-8")
                    return data
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{kind}:{exc}")

        if cached_body:
            logger.warning("Agentwiki registry refresh failed; using cached registry (%s)", "; ".join(errors))
            return cached_body
        raise AgentWikiError("registry_unavailable")

    def _fetch_registry_url(self, url: str, kind: str, cached_meta: Dict[str, Any]) -> tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        headers: Dict[str, str] = {}
        if kind == "index_json":
            headers["Accept"] = "application/json"
        else:
            headers["Accept"] = "text/plain, text/markdown;q=0.9"
        if self._conditional_get_enabled():
            if cached_meta.get("etag"):
                headers["If-None-Match"] = str(cached_meta["etag"])
            if cached_meta.get("last_modified"):
                headers["If-Modified-Since"] = str(cached_meta["last_modified"])

        response = self._http_get(url, headers=headers)
        if response.status_code == 304:
            body_path, _ = self._cache_paths()
            if body_path.exists():
                return json.loads(body_path.read_text(encoding="utf-8")), cached_meta
            raise AgentWikiError("registry_unavailable")
        response.raise_for_status()
        if kind == "index_json":
            data = response.json()
        else:
            data = self._parse_global_llms_txt(response.text, base_url=url)
        meta = {
            "source_url": url,
            "etag": response.headers.get("etag"),
            "last_modified": response.headers.get("last-modified"),
        }
        return data, meta

    def _select_wiki(self, query: str, registry: Dict[str, Any]) -> Dict[str, Any]:
        domains = self._domains_cfg()
        if not domains:
            return {"decision": AgentWikiDecision(selected_source="web_search", reason="no_candidate")}

        registry_wikis = {w.get("slug"): w for w in registry.get("wikis", []) if w.get("slug")}
        query_norm = _normalize_query(query)
        query_tokens = set(_tokenize(query_norm))
        explicit_matches: List[str] = []
        scores: Dict[str, Dict[str, Any]] = {}

        for domain_name, domain_cfg in domains.items():
            slug = str(domain_cfg.get("wiki_slug") or "").strip()
            wiki = registry_wikis.get(slug)
            if not wiki:
                continue
            aliases = [str(a).strip().lower() for a in domain_cfg.get("aliases", []) if str(a).strip()]
            title = str(wiki.get("title") or "").strip().lower()
            wiki_slug = slug.lower()
            tags = [str(t).strip().lower() for t in wiki.get("tags", []) if str(t).strip()]

            score = 0
            matched_on: List[str] = []
            explicit = False
            explicit_targets = aliases + [wiki_slug, title]
            for phrase in explicit_targets:
                if phrase and phrase in query_norm:
                    score = max(score, 100 if phrase in aliases or phrase == wiki_slug else 70)
                    matched_on.append(phrase)
                    explicit = True
            for tag in tags:
                if tag and tag in query_norm:
                    score = max(score, 50)
                    matched_on.append(tag)
            if score == 0:
                title_tokens = set(_tokenize(title))
                if title_tokens and (title_tokens & query_tokens):
                    score = max(score, 70)
                    matched_on.append("title_token_overlap")
            if score == 0:
                alias_tokens = set()
                for alias in aliases:
                    alias_tokens.update(_tokenize(alias))
                if alias_tokens and (alias_tokens & query_tokens):
                    score = max(score, 30)
                    matched_on.append("keyword_overlap")

            if score > 0:
                scores[str(slug)] = {
                    "domain_name": domain_name,
                    "wiki": wiki,
                    "score": score,
                    "matched_on": matched_on,
                    "explicit": explicit,
                }
                if explicit:
                    explicit_matches.append(str(slug))

        if not scores:
            return {"decision": AgentWikiDecision(selected_source="web_search", reason="no_candidate")}

        ranked = sorted(scores.values(), key=lambda item: item["score"], reverse=True)
        top = ranked[0]
        top_score = top["score"]
        min_score = self._min_select_score()
        if top_score < min_score:
            return {"decision": AgentWikiDecision(selected_source="web_search", reason="below_threshold", metadata={"top_score": top_score})}

        tied = [item for item in ranked if item["score"] == top_score]
        if len(tied) > 1 and len(set(explicit_matches)) != 1 and bool((self.cfg.get("routing") or {}).get("abstain_on_tie", True)):
            return {"decision": AgentWikiDecision(selected_source="web_search", reason="tie", metadata={"tied_wikis": [item["wiki"]["slug"] for item in tied]})}

        wiki = top["wiki"]
        scope_ok, scope_reason = self._passes_scope_gate(query_norm, wiki)
        if not scope_ok:
            return {
                "decision": AgentWikiDecision(
                    selected_source="web_search",
                    reason="out_of_scope",
                    selected_wiki_slug=wiki.get("slug"),
                    metadata={"scope_reason": scope_reason, "matched_on": top["matched_on"]},
                )
            }

        current_as = str(((wiki.get("scope") or {}).get("currentAs") or "")).strip() or None
        freshness_ok, freshness_reason = self._passes_freshness_gate(query_norm, wiki)
        if not freshness_ok:
            return {
                "decision": AgentWikiDecision(
                    selected_source="web_search",
                    reason="stale_for_query",
                    selected_wiki_slug=wiki.get("slug"),
                    current_as_used=current_as,
                    metadata={"freshness_reason": freshness_reason, "matched_on": top["matched_on"]},
                )
            }

        return {
            "wiki": wiki,
            "decision": AgentWikiDecision(
                selected_source="agentwikis",
                reason="success",
                selected_wiki_slug=wiki.get("slug"),
                current_as_used=current_as,
                metadata={"matched_on": top["matched_on"], "top_score": top_score},
            ),
        }

    def _passes_scope_gate(self, query_norm: str, wiki: Dict[str, Any]) -> tuple[bool, str]:
        scope = wiki.get("scope") or {}
        covers = str(scope.get("covers") or "")
        not_covered = str(scope.get("notCovered") or "")
        query_tokens = set(_tokenize(query_norm))
        not_covered_phrases = _split_scope_phrases(not_covered)
        for phrase in not_covered_phrases:
            phrase_tokens = set(_tokenize(phrase))
            if phrase and phrase in query_norm:
                return False, f"notCovered phrase match: {phrase}"
            if phrase_tokens and len(phrase_tokens) >= 2 and phrase_tokens.issubset(query_tokens):
                return False, f"notCovered token match: {phrase}"

        # Heuristic for neighboring products: if the query explicitly mentions
        # terms known to be excluded from scope, reject.
        if any(term in query_tokens for term in {"pricing", "billing", "cost", "enterprise", "sdk", "gateway", "api"}):
            lowered_not_covered = not_covered.lower()
            if "pricing" in query_tokens and any(t in lowered_not_covered for t in ("pricing", "billing", "cost")):
                return False, "pricing/billing excluded by notCovered"
            if "enterprise" in query_tokens and "enterprise" in lowered_not_covered:
                return False, "enterprise excluded by notCovered"
            if "sdk" in query_tokens and "sdk" in lowered_not_covered:
                return False, "sdk excluded by notCovered"
            if "api" in query_tokens and "api" in lowered_not_covered:
                return False, "api excluded by notCovered"
            if "gateway" in query_tokens and "gateway" in lowered_not_covered:
                return False, "gateway excluded by notCovered"

        if covers:
            cover_tokens = set(_tokenize(covers))
            if cover_tokens & query_tokens:
                return True, "covers overlap"

        # Explicit domain selection is good enough once exclusions did not fire.
        return True, "selected_domain"

    def _passes_freshness_gate(self, query_norm: str, wiki: Dict[str, Any]) -> tuple[bool, str]:
        scope = wiki.get("scope") or {}
        current_as = str(scope.get("currentAs") or "")
        current_date = _extract_date(current_as)
        not_covered = str(scope.get("notCovered") or "").lower()

        for trigger in self._freshness_trigger_terms():
            if trigger and trigger in query_norm:
                return False, f"trigger term: {trigger}"

        query_dates = [_extract_date(match.group(1)) for match in _DATE_RE.finditer(query_norm)]
        query_dates = [d for d in query_dates if d]
        if current_date and query_dates and any(qd > current_date for qd in query_dates):
            return False, f"query date newer than currentAs ({current_date.isoformat()})"

        query_tokens = set(_tokenize(query_norm))
        if query_tokens & _MUTABLE_TOPIC_TERMS and (query_tokens & set(_tokenize(not_covered)) or any(term in not_covered for term in _MUTABLE_TOPIC_TERMS)):
            return False, "mutable topic guarded by notCovered/currentAs"

        return True, "stable query"

    def _retrieve_pages(self, query: str, wiki: Dict[str, Any], *, limit: int) -> Dict[str, Any]:
        raw_base = _absolute_url(self._registry_urls()[0], wiki.get("raw_base") or "")
        html_base = _absolute_url(self._registry_urls()[0], wiki.get("html_base") or "")
        page_urls = []
        llms_txt_url = urljoin(html_base.rstrip("/") + "/", "llms.txt") if html_base else None
        if llms_txt_url:
            try:
                resp = self._http_get(llms_txt_url, headers={"Accept": "text/plain, text/markdown;q=0.9"})
                resp.raise_for_status()
                page_urls = self._parse_wiki_llms_txt(resp.text, base_url=llms_txt_url)
            except Exception as exc:  # noqa: BLE001
                logger.info("Agentwiki llms.txt fetch failed for %s: %s", wiki.get("slug"), exc)

        candidates = []
        for fallback in (urljoin(raw_base.rstrip("/") + "/", "README.md"), urljoin(raw_base.rstrip("/") + "/", "wiki/index.md")):
            if fallback and fallback not in page_urls:
                candidates.append(fallback)
        page_urls = candidates + [u for u in page_urls if u not in candidates]
        page_urls = page_urls[: self._max_pages_per_query()]
        if not page_urls:
            return {"results": [], "fallback_reason": "raw_unavailable", "retrieval_plan": []}

        query_tokens = set(_tokenize(query))
        pages: List[Dict[str, Any]] = []
        for page_url in page_urls:
            page = self._fetch_page_markdown(page_url)
            if page:
                pages.append(page)

        if not pages:
            return {"results": [], "fallback_reason": "raw_unavailable", "retrieval_plan": page_urls}

        ranked_pages = sorted(pages, key=lambda page: self._score_page(page, query_tokens), reverse=True)
        max_results = max(1, min(int(limit or 5), self._max_pages_per_query()))
        selected_pages = ranked_pages[:max_results]
        results = []
        for idx, page in enumerate(selected_pages, start=1):
            results.append(
                {
                    "title": page.get("title") or page.get("url"),
                    "url": page.get("url"),
                    "description": _summarize_page(page, wiki.get("slug") or "", wiki.get("title") or ""),
                    "position": idx,
                    "source": "agentwikis",
                }
            )
        return {"results": results, "retrieval_plan": page_urls}

    def _fetch_page_markdown(self, url: str) -> Optional[Dict[str, Any]]:
        try:
            response = self._http_get(url, headers={"Accept": "text/markdown, text/plain;q=0.9"})
            if response.status_code == 402:
                logger.info("Agentwiki page %s gated behind payment/auth", url)
                return None
            response.raise_for_status()
            return _parse_markdown_page(url, response.text)
        except Exception:
            if self._retry_html_markdown() and "/raw/" in url:
                html_url = url.replace("/raw/", "/wiki/", 1)
                try:
                    response = self._http_get(html_url, headers={"Accept": "text/markdown"})
                    if response.status_code == 402:
                        return None
                    response.raise_for_status()
                    return _parse_markdown_page(str(response.url), response.text)
                except Exception:
                    return None
            return None

    def _score_page(self, page: Dict[str, Any], query_tokens: set[str]) -> int:
        haystacks = [
            str(page.get("title") or "").lower(),
            str(page.get("content") or "").lower(),
            str(page.get("url") or "").lower(),
        ]
        score = 0
        for token in query_tokens:
            if len(token) < 3:
                continue
            for hay in haystacks:
                if token in hay:
                    score += 5 if hay is haystacks[0] else 1
                    break
        if str(page.get("url") or "").endswith("README.md"):
            score += 1
        if str(page.get("url") or "").endswith("wiki/index.md"):
            score += 2
        return score

    def _warn_invalid_configured_slugs(self, registry: Dict[str, Any]) -> None:
        registry_slugs = {w.get("slug") for w in registry.get("wikis", [])}
        invalid = []
        for domain_name, domain_cfg in self._domains_cfg().items():
            slug = domain_cfg.get("wiki_slug")
            if slug and slug not in registry_slugs:
                invalid.append({"domain": domain_name, "wiki_slug": slug})
        if invalid:
            logger.warning("Agentwiki config has unknown wiki slugs: %s", json.dumps(invalid, ensure_ascii=False))

    def _http_get(self, url: str, *, headers: Optional[Dict[str, str]] = None) -> httpx.Response:
        timeout = self._timeout_s()
        if self.client is not None:
            return self.client.get(url, headers=headers or {}, timeout=timeout, follow_redirects=True)
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            return client.get(url, headers=headers or {})

    @staticmethod
    def _parse_global_llms_txt(text: str, *, base_url: str) -> Dict[str, Any]:
        wikis: List[Dict[str, Any]] = []
        sections = re.split(r"^## ", text, flags=re.MULTILINE)
        for section in sections[1:]:
            lines = [line.rstrip() for line in section.splitlines()]
            if not lines:
                continue
            title = lines[0].strip()
            body = lines[1:]
            covers = _strip_quote_prefix(_find_line_value(body, "> Covers:"))
            not_covered = _strip_quote_prefix(_find_line_value(body, "> Not covered:"))
            current_as = _strip_quote_prefix(_find_line_value(body, "> Current as of:"))
            links = _RAW_LINK_RE.findall("\n".join(body))
            if not links:
                continue
            first_link = links[0]
            slug_match = re.search(r"/raw/([^/]+)/", first_link)
            if not slug_match:
                continue
            slug = slug_match.group(1)
            wikis.append(
                {
                    "slug": slug,
                    "title": title,
                    "description": title,
                    "tags": _tokenize(title),
                    "category": "unknown",
                    "scope": {
                        "covers": covers,
                        "notCovered": not_covered,
                        "currentAs": current_as,
                    },
                    "lastUpdated": None,
                    "documentCount": len(links),
                    "llms_txt": "/llms.txt",
                    "raw_base": f"/raw/{slug}/",
                    "html_base": f"/wiki/{slug}/",
                }
            )
        return {"name": "Agent Wikis", "wikis": wikis}

    @staticmethod
    def _parse_wiki_llms_txt(text: str, *, base_url: str) -> List[str]:
        urls = []
        for path in _RAW_LINK_RE.findall(text):
            urls.append(_absolute_url(base_url, path))
        return urls


def maybe_agentwiki_route_search(query: str, web_config: Dict[str, Any], *, limit: int = 5, client: Optional[httpx.Client] = None) -> AgentWikiResult:
    router = AgentWikiRouter(web_config, client=client)
    return router.maybe_route_search(query, limit=limit)


def _parse_markdown_page(url: str, text: str) -> Dict[str, Any]:
    frontmatter: Dict[str, str] = {}
    body = text
    if text.startswith("---\n"):
        parts = text.split("\n---\n", 1)
        if len(parts) == 2:
            _, fm_body = parts
            fm_text = text[4 : len(text) - len(fm_body) - 5]
            body = fm_body
            for line in fm_text.splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                frontmatter[key.strip()] = value.strip().strip('"')
    title = frontmatter.get("title")
    if not title:
        for line in body.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
    return {
        "url": url,
        "title": title or url,
        "content": body,
        "frontmatter": frontmatter,
    }


def _summarize_page(page: Dict[str, Any], slug: str, title: str) -> str:
    body = str(page.get("content") or "")
    lines = [line.strip() for line in body.splitlines() if line.strip() and not line.strip().startswith("#")]
    excerpt = " ".join(lines[:3])
    excerpt = re.sub(r"\s+", " ", excerpt).strip()
    if len(excerpt) > 260:
        excerpt = excerpt[:257].rstrip() + "..."
    return f"Agentwiki ({slug or title}): {excerpt}" if excerpt else f"Agentwiki ({slug or title}) raw markdown page"


def _normalize_query(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _find_line_value(lines: Iterable[str], prefix: str) -> str:
    prefix_lower = prefix.lower()
    for line in lines:
        if line.lower().startswith(prefix_lower):
            return line.split(":", 1)[-1].strip()
    return ""


def _strip_quote_prefix(value: str) -> str:
    return value.lstrip("> ").strip()


def _split_scope_phrases(text: str) -> List[str]:
    raw_parts = re.split(r"[;,]", (text or ""))
    parts: List[str] = []
    for raw in raw_parts:
        cleaned = raw.strip().lower()
        if not cleaned:
            continue
        if cleaned.startswith("and "):
            cleaned = cleaned[4:]
        parts.append(cleaned)
    return parts


def _extract_date(text: Optional[str]) -> Optional[date]:
    if not text:
        return None
    match = _DATE_RE.search(text)
    if not match:
        return None
    raw = match.group(1)
    try:
        if len(raw) == 10:
            return datetime.strptime(raw, "%Y-%m-%d").date()
        if len(raw) == 7:
            return datetime.strptime(raw + "-01", "%Y-%m-%d").date()
        if len(raw) == 4:
            return datetime.strptime(raw + "-01-01", "%Y-%m-%d").date()
    except ValueError:
        return None
    return None


def _absolute_url(base_url: str, path: str) -> str:
    if not path:
        return ""
    if path.startswith("http://") or path.startswith("https://"):
        return path
    origin_match = re.match(r"^(https?://[^/]+)", base_url)
    origin = origin_match.group(1) if origin_match else base_url
    return urljoin(origin.rstrip("/") + "/", path.lstrip("/"))
