"""Skills Hub ClawHub adapter (clawhub.ai HTTP API)."""

import hashlib
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from tools.skills_hub_models import (
    GuardedFetchMixin, SkillBundle, SkillMeta, SkillSource, _cache_metas, _cached_metas, _get_json,
    _validate_bundle_rel_path,
)

logger = logging.getLogger("tools.skills_hub")


def _query_terms(query: str) -> List[str]:
    return [term for term in re.split(r"[^a-z0-9]+", query.lower()) if term]


def _dedupe_results(results: List[SkillMeta]) -> List[SkillMeta]:
    """Dedupe by lowercased identifier (name fallback), first wins."""
    seen: Dict[str, SkillMeta] = {}
    for result in results:
        seen.setdefault((result.identifier or result.name).lower(), result)
    return list(seen.values())


def _search_score(query: str, meta: SkillMeta) -> int:
    query_norm = query.strip().lower()
    if not query_norm:
        return 1
    identifier, name = (meta.identifier or "").lower(), (meta.name or "").lower()
    description = (meta.description or "").lower()
    query_terms, identifier_terms, name_terms = _query_terms(query_norm), _query_terms(identifier), _query_terms(name)
    normalized_identifier, normalized_name = " ".join(identifier_terms), " ".join(name_terms)
    checks = (
        (140, query_norm == identifier), (130, query_norm == name),
        (125, normalized_identifier == query_norm), (120, normalized_name == query_norm),
        (95, normalized_identifier.startswith(query_norm)), (90, normalized_name.startswith(query_norm)),
        (70, bool(query_terms) and identifier_terms[: len(query_terms)] == query_terms),
        (65, bool(query_terms) and name_terms[: len(query_terms)] == query_terms),
        (40, query_norm in identifier), (35, query_norm in name), (10, query_norm in description),
    )
    score = sum(points for points, hit in checks if hit)
    for term in query_terms:
        score += 15 * (term in identifier_terms) + 12 * (term in name_terms) + 3 * (term in description)
    return score


def _first_str(*values: Any) -> Optional[str]:
    """First non-empty ``str`` among ``values`` (dicts/None are skipped)."""
    return next((v for v in values if isinstance(v, str) and v), None)


class ClawHubSource(GuardedFetchMixin, SkillSource):
    """ClawHub (clawhub.ai) HTTP API. Every skill is community trust — the ClawHavoc
    incident (341 malicious skills, Feb 2026) showed their vetting is insufficient."""

    SOURCE_ID = "clawhub"
    BASE_URL = "https://clawhub.ai/api/v1"
    # Wall-clock budget for a full catalog walk: 50k+ skills, sequential
    # (~250 requests each under timeout=30), so unbounded it blocks for minutes.
    CATALOG_WALK_BUDGET_SECONDS = 12
    _SLUG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*$")

    _query_terms = staticmethod(_query_terms)
    _dedupe_results = staticmethod(_dedupe_results)
    _search_score = staticmethod(_search_score)
    _get_json = staticmethod(_get_json)

    @staticmethod
    def _normalize_tags(tags: Any) -> List[str]:
        if isinstance(tags, list):
            return [str(t) for t in tags]
        if isinstance(tags, dict):
            return [str(k) for k in tags if str(k) != "latest"]
        return []

    @staticmethod
    def _coerce_skill_payload(data: Any) -> Optional[Dict[str, Any]]:
        """Flatten ``{"skill": {...}, "latestVersion", "owner"}`` listing shapes."""
        if not isinstance(data, dict):
            return None
        nested = data.get("skill")
        if not isinstance(nested, dict):
            return data
        merged = dict(nested)
        # latestVersion and owner (needed for valid detail URLs) live beside the skill.
        if data.get("latestVersion") is not None and "latestVersion" not in merged:
            merged["latestVersion"] = data["latestVersion"]
        if "owner" in data and "owner" not in merged:
            merged["owner"] = data["owner"]
        return merged

    @staticmethod
    def _owner_from_payload(data: Optional[Dict[str, Any]]) -> Optional[str]:
        owner = data.get("owner") if isinstance(data, dict) else None
        if isinstance(owner, dict):
            owner = owner.get("handle")
        return owner.strip() if isinstance(owner, str) and owner.strip() else None

    @classmethod
    def _owner_matches(cls, expected_owner: Optional[str], data: Optional[Dict[str, Any]]) -> bool:
        if not expected_owner:
            return True
        actual = cls._owner_from_payload(data)
        return not actual or actual.lower() == expected_owner.lower()

    @classmethod
    def _item_to_meta(cls, item: Dict[str, Any]) -> Optional[SkillMeta]:
        """Listing item -> SkillMeta (None without a slug)."""
        slug = item.get("slug")
        if not isinstance(slug, str) or not slug:
            return None
        owner = cls._owner_from_payload(item)
        return SkillMeta(
            name=item.get("displayName") or item.get("name") or slug,
            description=item.get("summary") or item.get("description") or "",
            source="clawhub", identifier=slug, trust_level="community",
            tags=cls._normalize_tags(item.get("tags", [])), extra={"owner": owner} if owner else {},
        )

    def _skill_detail(self, identifier: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """``(slug, payload)`` for an identifier, or None when unparsable,
        missing, or owned by someone other than the ``@owner`` requested."""
        parsed = self._parse_identifier(identifier)
        if parsed is None:
            return None
        slug, expected_owner = parsed
        data = self._coerce_skill_payload(self._get_json(f"{self.BASE_URL}/skills/{slug}"))
        if not isinstance(data, dict) or not self._owner_matches(expected_owner, data):
            return None
        return slug, data

    def _exact_slug_meta(self, query: str) -> Optional[SkillMeta]:
        query = query.strip()
        parsed, query_terms = self._parse_identifier(query), _query_terms(query)
        candidates = [parsed[0]] if parsed else [query] if "/" not in query and self._SLUG_RE.fullmatch(query) else []
        if query_terms:
            base_slug = "-".join(query_terms)
            if len(query_terms) >= 2:
                candidates.extend(f"{base_slug}-{s}" for s in ("agent", "skill", "tool", "assistant", "playbook"))
            candidates.append(base_slug)
        return next((m for m in map(self.inspect, dict.fromkeys(candidates)) if m), None)

    def _finalize_search_results(self, query: str, results: List[SkillMeta], limit: int) -> List[SkillMeta]:
        query_norm = query.strip()
        if not query_norm:
            return _dedupe_results(results)[:limit]
        filtered = [meta for meta in results if _search_score(query_norm, meta) > 0]
        filtered.sort(key=lambda meta: (-_search_score(query_norm, meta), meta.name.lower(), meta.identifier.lower()))
        filtered = _dedupe_results(filtered)
        exact = self._exact_slug_meta(query_norm)
        if exact:
            filtered = _dedupe_results([exact] + [m for m in filtered if _search_score(query_norm, m) >= 20])
        if filtered or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", query_norm):
            return filtered[:limit]
        return _dedupe_results(results)[:limit]

    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        query = query.strip()
        if query:
            if len(_query_terms(query)) >= 2:
                direct = self._exact_slug_meta(query)
                if direct:
                    return [direct]
            results = self._search_catalog(query, limit=limit)
            if results:
                return results
        else:
            # Empty query: paginating catalog walker. A disk-cached full catalog
            # is returned whole (caller paginates); on a cold cache the walk is
            # bounded to `limit` so browse renders page one without walking
            # 50k+ skills (max_items=0 = unbounded, offline index builder only).
            catalog = self._load_catalog_index(max_items=max(limit, 0))
            if catalog:
                deduped = _dedupe_results(catalog)
                return deduped[:limit] if limit > 0 else deduped

        # Catalog miss / walker failure: best-effort lightweight listing API.
        cache_key = f"clawhub_search_listing_v1_{hashlib.md5(query.encode()).hexdigest()}_{limit}"
        cached = _cached_metas(cache_key)
        if cached is not None:
            return self._finalize_search_results(query, cached, limit)
        data = self._get_json(f"{self.BASE_URL}/skills", timeout=15, params={"search": query, "limit": limit})
        skills_data = data.get("items", data) if isinstance(data, dict) else data
        if not isinstance(skills_data, list):
            return []
        results = [m for m in map(self._item_to_meta, skills_data[:limit]) if m]
        final_results = self._finalize_search_results(query, results, limit)
        _cache_metas(cache_key, final_results)
        return final_results

    @classmethod
    def _parse_identifier(cls, identifier: str) -> Optional[Tuple[str, Optional[str]]]:
        """``(slug, expected_owner)`` for a bare slug, ``clawhub/<slug>``,
        ``@owner/slug``, or the URL path ``owner/skills/slug``.

        GitHub-style ``owner/repo/skill`` identifiers are NOT ClawHub's —
        claiming them by last segment would install a same-named skill from a
        different author.
        """
        raw = (identifier or "").strip()
        if not raw:
            return None
        had_at = raw.startswith("@")
        parts = [part for part in raw.removeprefix("@").removeprefix("clawhub/").split("/") if part]
        if len(parts) == 1:
            owner, slug = None, parts[0]
        elif (len(parts) == 2 and had_at) or (len(parts) == 3 and parts[1].lower() == "skills"):
            owner, slug = parts[0], parts[-1]
        else:
            return None
        if not cls._SLUG_RE.fullmatch(slug) or (owner is not None and not cls._SLUG_RE.fullmatch(owner)):
            return None
        return slug, owner

    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        detail = self._skill_detail(identifier)
        if detail is None:
            return None
        slug, skill_data = detail
        latest_version = self._resolve_latest_version(slug, skill_data)
        if not latest_version:
            logger.warning("ClawHub fetch failed for %s: could not resolve latest version", slug)
            return None

        # Primary: ZIP bundle from /download. Fallback: version metadata with
        # inline/raw content (files may sit under version_data["version"]).
        files = self._download_zip(slug, latest_version)
        if "SKILL.md" not in files:
            version_data = self._get_json(f"{self.BASE_URL}/skills/{slug}/versions/{latest_version}")
            if isinstance(version_data, dict):
                files = self._extract_files(version_data) or files
                nested = version_data.get("version", {})
                if "SKILL.md" not in files and isinstance(nested, dict):
                    files = self._extract_files(nested) or files
        if "SKILL.md" not in files:
            logger.warning(
                "ClawHub fetch for %s resolved version %s but could not retrieve file content", slug, latest_version,
            )
            return None
        return SkillBundle(name=slug, files=files, source="clawhub", identifier=slug, trust_level="community")

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        detail = self._skill_detail(identifier)
        if detail is None:
            return None
        slug, data = detail
        return self._item_to_meta({**data, "slug": data.get("slug") or slug})

    def _search_catalog(self, query: str, limit: int = 10) -> List[SkillMeta]:
        cache_key = f"clawhub_search_catalog_v1_{hashlib.md5(f'{query}|{limit}'.encode()).hexdigest()}"
        cached = _cached_metas(cache_key)
        if cached is not None:
            return cached[:limit]
        catalog = self._load_catalog_index()
        if not catalog:
            return []
        results = self._finalize_search_results(query, catalog, limit)
        _cache_metas(cache_key, results)
        return results

    def _load_catalog_index(self, max_items: int = 0) -> List[SkillMeta]:
        """Walk the ClawHub catalog via cursor pagination.

        ``max_items`` stops the walk early once that many distinct skills are
        gathered (browse's cold-start fallback renders one page); ``0`` walks
        to exhaustion (offline index builder). Only a COMPLETE walk (cursor
        exhausted or page cap) is written to the shared ``clawhub_catalog_v1``
        cache — a walk cut by ``max_items`` or the wall-clock budget would
        poison it with a partial slice.
        """
        cache_key = "clawhub_catalog_v1"
        cached = _cached_metas(cache_key)
        if cached is not None:
            return cached
        cursor: Optional[str] = None
        results: List[SkillMeta] = []
        seen: set[str] = set()
        # 750 pages * 200/page = 150k ceiling over the ~50k catalog; a safety
        # rail against an infinite-cursor loop, normally ended by nextCursor=None.
        # Wall-clock budget applies to interactive browse only: the index builder
        # (max_items=0) must walk everything or it trips the deploy health floor.
        deadline = time.monotonic() + self.CATALOG_WALK_BUDGET_SECONDS if max_items > 0 else None
        partial = False
        for _ in range(750):
            if deadline is not None and time.monotonic() > deadline:
                partial = True
                break
            params: Dict[str, Any] = {"limit": 200, "cursor": cursor} if cursor else {"limit": 200}
            data = self._get_json(f"{self.BASE_URL}/skills", timeout=30, params=params)
            items = data.get("items", []) if isinstance(data, dict) else []
            if not isinstance(items, list) or not items:
                break
            for item in items:
                slug = item.get("slug")
                if isinstance(slug, str) and slug and slug not in seen:
                    seen.add(slug)
                    meta = self._item_to_meta(item)
                    if meta:
                        results.append(meta)
            cursor = data.get("nextCursor") if isinstance(data, dict) else None
            if not isinstance(cursor, str) or not cursor:
                break
            if max_items > 0 and len(results) >= max_items:
                partial = True
                break
        if not partial:
            _cache_metas(cache_key, results)
        return results

    def _resolve_latest_version(self, slug: str, skill_data: Dict[str, Any]) -> Optional[str]:
        latest, tags = skill_data.get("latestVersion"), skill_data.get("tags")
        version = _first_str(
            latest.get("version") if isinstance(latest, dict) else None,
            tags.get("latest") if isinstance(tags, dict) else None,
        )
        if version:
            return version
        vd = self._get_json(f"{self.BASE_URL}/skills/{slug}/versions")
        return _first_str(vd[0].get("version")) if isinstance(vd, list) and vd and isinstance(vd[0], dict) else None

    def _fetch_owner_handle(self, slug: str) -> Optional[str]:
        """Owner handle from the detail API (the listing API lacks it), or None.

        Bounded retry: 3 attempts total. 429 honours ``Retry-After`` else
        exponential backoff (2s -> 4s); 5xx and transport errors back off;
        other 4xx means the resource doesn't exist — no retry.
        """
        url = f"{self.BASE_URL}/skills/{slug}"
        max_attempts = 3
        for attempt in range(max_attempts):
            delay = 2.0 * (2 ** attempt)
            try:
                resp = httpx.get(url, timeout=20)
            except (httpx.HTTPError, OSError):
                reason = "transport error"
            else:
                if resp.status_code == 200:
                    try:
                        raw = resp.json()
                    except (json.JSONDecodeError, ValueError):
                        return None
                    return self._owner_from_payload(self._coerce_skill_payload(raw))
                if resp.status_code == 429:
                    try:
                        delay = float(resp.headers.get("Retry-After") or delay)
                    except (TypeError, ValueError):
                        pass
                    reason = "HTTP 429"
                elif 500 <= resp.status_code < 600:
                    reason = f"HTTP {resp.status_code}"
                else:
                    return None  # 4xx (non-429): doesn't exist / bad request
            if attempt >= max_attempts - 1:
                return None
            logger.debug("_fetch_owner_handle(%s): %s on attempt %d/%d, retrying in %.1fs",
                         slug, reason, attempt + 1, max_attempts, delay)
            time.sleep(delay)
        return None

    def enrich_owners(self, skills: List[SkillMeta], max_workers: int = 30) -> int:
        """Batch-fetch owner handles for ClawHub skills missing ``extra["owner"]``
        (in-place; returns the number enriched). For the offline index builder:
        the full 50k catalog takes ~5–10 min at 30 workers.

        Safety rails: aborts after 50 consecutive failures (systemic outage),
        per-request 429 backoff, progress log every 1000 skills.
        """
        needs_enrichment = [s for s in skills if s.source == "clawhub" and not (s.extra or {}).get("owner")]
        if not needs_enrichment:
            return 0
        enriched = consecutive_failures = processed = 0
        max_consecutive_failures = 50

        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(self._fetch_owner_handle, s.identifier): s for s in needs_enrichment}
            for future in as_completed(futures):
                meta = futures[future]
                processed += 1
                try:
                    handle = future.result()
                except Exception:
                    handle = None
                consecutive_failures = 0 if handle else consecutive_failures + 1
                if handle:
                    meta.extra = meta.extra or {}
                    meta.extra["owner"] = handle
                    enriched += 1
                if processed % 1000 == 0:
                    logger.info("ClawHub owner enrichment: %d/%d processed, %d enriched",
                                processed, len(needs_enrichment), enriched)
                if consecutive_failures >= max_consecutive_failures:
                    logger.warning(
                        "ClawHub owner enrichment: %d consecutive failures — "
                        "aborting early (%d/%d processed, %d enriched). "
                        "The ClawHub API may be down or rate-limited.",
                        max_consecutive_failures, processed, len(needs_enrichment), enriched,
                    )
                    for f in futures:
                        f.cancel()
                    break
        return enriched

    def _extract_files(self, version_data: Dict[str, Any]) -> Dict[str, str]:
        files: Dict[str, str] = {}
        file_list = version_data.get("files")
        if isinstance(file_list, dict):
            return {k: v for k, v in file_list.items() if isinstance(v, str)}
        for file_meta in file_list if isinstance(file_list, list) else []:
            if not isinstance(file_meta, dict):
                continue
            fname, inline = file_meta.get("path") or file_meta.get("name"), file_meta.get("content")
            raw_url = file_meta.get("rawUrl") or file_meta.get("downloadUrl") or file_meta.get("url")
            if not fname or not isinstance(fname, str):
                continue
            if isinstance(inline, str):
                files[fname] = inline
            elif isinstance(raw_url, str) and raw_url.startswith("http"):
                content = self._fetch_text(raw_url)
                if content is not None:
                    files[fname] = content
        return files

    def _download_zip(self, slug: str, version: str) -> Dict[str, str]:
        """Download the skill ZIP from /download and extract its text files."""
        import io
        import zipfile

        files: Dict[str, str] = {}
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = httpx.get(f"{self.BASE_URL}/download", params={"slug": slug, "version": version},
                                 timeout=30, follow_redirects=True)
                if resp.status_code == 429:
                    try:
                        retry_after = min(int(resp.headers.get("retry-after", "5")), 15)  # Cap wait time
                    except (ValueError, TypeError):
                        retry_after = 5
                    logger.debug("ClawHub download rate-limited for %s, retrying in %ds (attempt %d/%d)",
                                 slug, retry_after, attempt + 1, max_retries)
                    time.sleep(retry_after)
                    continue
                if resp.status_code != 200:
                    logger.debug("ClawHub ZIP download for %s v%s returned %s", slug, version, resp.status_code)
                    return files
                with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        try:
                            name = _validate_bundle_rel_path(info.filename)
                        except ValueError:
                            logger.debug("Skipping unsafe ZIP member path: %s", info.filename)
                            continue
                        if info.file_size > 500_000:  # skip large binaries
                            logger.debug("Skipping large file in ZIP: %s (%d bytes)", name, info.file_size)
                            continue
                        try:
                            files[name] = zf.read(info.filename).decode("utf-8")
                        except (UnicodeDecodeError, KeyError):
                            logger.debug("Skipping non-text file in ZIP: %s", name)
                return files
            except zipfile.BadZipFile:
                logger.warning("ClawHub returned invalid ZIP for %s v%s", slug, version)
                return files
            except httpx.HTTPError as exc:
                logger.debug("ClawHub ZIP download failed for %s v%s: %s", slug, version, exc)
                return files
        logger.debug("ClawHub ZIP download exhausted retries for %s v%s", slug, version)
        return files
