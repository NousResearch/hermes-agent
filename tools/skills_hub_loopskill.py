"""Skills Hub LoopSkill adapter: catalog discovery + install via app.loopskill.io.

Mirrors ``tools/skills_hub_skillssh.py`` (the skills.sh adapter): a third-party
catalog, ``"community"`` trust, own identifier namespace (``loopskill/<slug>``),
and the shared ``_memo_json``/``_cached_metas`` cache helpers (never a bespoke
cache — that breaks the seam ``tools.skills_hub_models.hub()`` tests patch).
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional

from tools.skills_hub_models import (
    SkillBundle, SkillMeta, SkillSource, _cache_metas, _cached_metas, _get_json, _memo_json,
)

logger = logging.getLogger("tools.skills_hub")

# Tiers an anonymous adapter may fetch/install content for. Anything else
# (pro/pro_plus) is listed (so search stays honest about what exists) but
# fetch() refuses with a clear message rather than silently 200ing empty
# content or requiring an undocumented env var/API key.
_FREE_TIER = "free"


class LoopSkillSource(SkillSource):
    """Discover and install skills from LoopSkill (``app.loopskill.io``)."""

    BASE_URL = "https://app.loopskill.io"
    SEARCH_URL = f"{BASE_URL}/api/skills/search"
    DETAIL_URL = f"{BASE_URL}/api/skills/{{slug}}"

    SOURCE_ID = "loopskill"
    TRUST_LEVEL = "community"

    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        query = query.strip()
        if not query:
            # No practical bulk-catalog endpoint is confirmed yet (unlike
            # skills.sh's sitemap walk) — behave like UrlSource.search() for
            # an unsupported bulk case rather than guessing an endpoint.
            return []
        cache_key = f"loopskill_search_{hashlib.md5(f'{query}|{limit}'.encode()).hexdigest()}"
        cached = _cached_metas(cache_key)
        if cached is not None:
            return cached[:limit]
        data = _get_json(self.SEARCH_URL, params={"q": query, "limit": limit})
        items = data.get("results", []) if isinstance(data, dict) else None
        if not isinstance(items, list):
            return []
        results = [m for m in map(self._item_to_meta, items[:limit]) if m]
        _cache_metas(cache_key, results)
        return results

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        slug = self._slug_from_identifier(identifier)
        if not slug:
            return None
        detail = self._fetch_detail(slug)
        return self._detail_to_meta(slug, detail) if detail else None

    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        slug = self._slug_from_identifier(identifier)
        if not slug:
            return None
        detail = self._fetch_detail(slug)
        if not detail:
            return None
        tier = detail.get("tier") or _FREE_TIER
        if tier != _FREE_TIER:
            logger.warning(
                "LoopSkill skill %r is tier=%r: locked content is never fetched anonymously; "
                "get it at %s/skills/%s", slug, tier, self.BASE_URL, slug,
            )
            return None
        readme = detail.get("readme")
        if not isinstance(readme, str) or not readme.strip():
            return None
        return SkillBundle(
            name=slug, files={"SKILL.md": readme}, source="loopskill",
            identifier=self._wrap_identifier(slug), trust_level=self.TRUST_LEVEL,
            metadata=self._detail_metadata(slug, detail),
        )

    def _item_to_meta(self, item: dict) -> Optional[SkillMeta]:
        if not isinstance(item, dict):
            return None
        slug = item.get("slug")
        if not isinstance(slug, str) or not slug:
            return None
        if item.get("is_public") is False:
            return None
        tier = item.get("tier") or _FREE_TIER
        name = str(item.get("title") or slug)
        description = str(item.get("description") or "")
        return SkillMeta(
            name=name, description=description, source="loopskill",
            identifier=self._wrap_identifier(slug), trust_level=self.TRUST_LEVEL, path=slug,
            extra={
                "slug": slug, "tier": tier, "locked": tier != _FREE_TIER,
                "category": item.get("category"), "latest_version": item.get("latest_version"),
                "detail_url": f"{self.BASE_URL}/skills/{slug}",
            },
        )

    def _detail_to_meta(self, slug: str, detail: dict) -> SkillMeta:
        tier = detail.get("tier") or _FREE_TIER
        return SkillMeta(
            name=str(detail.get("title") or slug), description=str(detail.get("description") or ""),
            source="loopskill", identifier=self._wrap_identifier(slug), trust_level=self.TRUST_LEVEL, path=slug,
            extra=self._detail_metadata(slug, detail) | {"tier": tier, "locked": tier != _FREE_TIER},
        )

    def _detail_metadata(self, slug: str, detail: dict) -> Dict[str, Any]:
        return {
            "slug": slug, "license": detail.get("license"), "latest_version": detail.get("latest_version"),
            "detail_url": f"{self.BASE_URL}/skills/{slug}",
        }

    def _fetch_detail(self, slug: str) -> Optional[dict]:
        cache_key = f"loopskill_detail_{hashlib.md5(slug.encode()).hexdigest()}"
        return _memo_json(cache_key, lambda: _get_json(self.DETAIL_URL.format(slug=slug)),
                          valid=lambda c: isinstance(c, dict))

    @staticmethod
    def _slug_from_identifier(identifier: str) -> Optional[str]:
        if not isinstance(identifier, str) or not identifier:
            return None
        slug = identifier[len("loopskill/"):] if identifier.startswith("loopskill/") else identifier
        return slug or None

    @staticmethod
    def _wrap_identifier(slug: str) -> str:
        return f"loopskill/{slug}"
