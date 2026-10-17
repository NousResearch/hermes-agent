"""Per-task model routing for ``delegate_task`` pins.

A pin is matched against the model picker's listing in three steps, each tried only when the one before found
nothing, and every rewrite is reported back to the caller (``route_note``):

1. exact: the listed id, ignoring case and a ``vendor/`` prefix (``model:variant`` keeps its suffix);
2. fixed rules: the same id spelled differently (``4-5`` / ``4.5``, a trailing date), a version-less family name
   (``astra``, ``sol``) meaning that family's newest standard id, or the start of exactly one id (``step-5``);
3. typo correction: a near-miss of one id or family name, only when it is clearly closest. Different version
   numbers are never a typo (``v4.1`` is a new release, not ``v4``).

A model named without a provider runs on the first provider that matches at the earliest step: provider plugins,
then first-party / subscription providers, then aggregators; the inherited provider wins ties within a tier.
Anything else fails the spawn with similar ids and the logged-in providers' families. Listed means available:
auth/permission errors are reported by the child's own call. Catalogs come from the provider listing the model
picker reads, loaded once per call from cached catalogs; with no cached provider registry the check is skipped.
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# A typo is corrected only when the closest candidate clears this similarity and leads the runner-up by the margin.
_MIN_RATIO = 0.8
_MIN_LEAD = 0.1
_PLUGIN, _FIRST_PARTY, _AGGREGATOR = range(3)
# Size/tier words many vendors share: never a model family on their own (``flash`` names no one model).
_TIER_WORDS = frozenset({
    "flash", "flashx", "mini", "nano", "lite", "pro", "max", "plus", "ultra", "turbo", "fast", "preview", "free",
    "latest", "small", "medium", "large", "air", "instruct", "chat", "high", "low",
})
_DATE_SUFFIX = re.compile(r"(?:-\d{8}|-\d{4}-\d{2}-\d{2})$")


class TaskRouteCatalog:
    """Logged-in providers and their model lists for one ``delegate_task`` call. ``configured`` is the inherited
    provider first (it wins ties within a tier and is kept unpinned when chosen)."""

    def __init__(self, configured: tuple[str, ...], *, direct_endpoint: bool = False):
        self.configured = tuple(p for p in configured if p and p.strip().lower() != "auto")
        # The inherited route is a direct endpoint (delegation.base_url): a chosen provider is always pinned.
        self.direct_endpoint = direct_endpoint
        self._rows: Optional[list[dict[str, Any]]] = None

    def rows(self) -> list[dict[str, Any]]:
        if self._rows is None:
            self._rows = _load_rows(self.configured[0] if self.configured else "")
        return self._rows


def _load_rows(current_provider: str) -> list[dict[str, Any]]:
    from agent.models_dev import fetch_models_dev
    from hermes_cli.config import get_compatible_custom_providers, load_config_readonly
    from hermes_cli.model_switch_providers import list_authenticated_providers
    # Cache only: a profile that never fetched the provider registry skips the check rather than waiting on a
    # download before any child is built. A cached copy (even a stale one) is served without blocking.
    if not fetch_models_dev(allow_network=False):
        return []
    cfg = load_config_readonly() or {}
    user_providers = cfg.get("providers")
    try:
        return list_authenticated_providers(
            current_provider=current_provider, user_providers=user_providers if isinstance(user_providers, dict) else None,
            custom_providers=get_compatible_custom_providers(cfg),
            max_models=None, non_blocking_catalogs=True, probe_custom_providers=False,
            # A provider whose credentials are all in a (per-model) cooldown is rate-limited, not signed out.
            for_picker=True, fast_custom_probe=False)
    except Exception:
        logger.debug("Could not list providers for delegate_task model routing", exc_info=True)
        return []


@lru_cache(maxsize=256)
def _provider_id(name: str) -> str:
    """Canonical provider id, honoring provider-profile aliases (``codex`` -> ``openai-codex``) that
    ``normalize_provider`` does not apply. Registry lookup only, never a network call."""
    from hermes_cli.providers import normalize_provider
    key = normalize_provider(name)
    try:
        from providers import get_provider_profile
        profile = get_provider_profile(key)
    except Exception:
        logger.debug("Could not look up provider profile %s for delegate_task model routing", key, exc_info=True)
        profile = None
    # Only an alias the profile itself declares maps to it: every saved custom endpoint shares the generic
    # ``custom`` profile, and two endpoints are never the same route.
    names = {str(getattr(profile, "name", "") or "").lower(), *(str(a).lower() for a in getattr(profile, "aliases", ()) or ())}
    return str(profile.name) if profile is not None and key.lower() in names else key


def _same_provider(a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b:
        return False
    return _provider_id(a) == _provider_id(b)


def _is_plugin(slug: str) -> bool:
    from hermes_cli.providers import resolve_provider_full
    try:
        pdef = resolve_provider_full(slug, None, None)
    except Exception:
        logger.debug("Could not resolve provider %s for delegate_task model routing", slug, exc_info=True)
        return False
    return pdef is not None and pdef.source == "plugin-profile"


def _provider_tier(slug: str) -> int:
    from hermes_cli.models_catalog_static import _AGGREGATOR_PROVIDERS
    from hermes_cli.providers import normalize_provider
    if _is_plugin(slug):
        return _PLUGIN
    if slug.lower() in _AGGREGATOR_PROVIDERS or normalize_provider(slug) in _AGGREGATOR_PROVIDERS:
        return _AGGREGATOR
    return _FIRST_PARTY


def _bare(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1].lower()


def _undated(model_id: str) -> str:
    """Bare id without a ``:variant`` or a trailing release date."""
    return _DATE_SUFFIX.sub("", _bare(model_id).split(":", 1)[0])


def _canon(model_id: str) -> str:
    return re.sub(r"[._]", "-", _undated(model_id))


def _version(model_id: str) -> tuple[int, ...]:
    return tuple(int(n) for n in re.findall(r"\d+", _undated(model_id)))


def _family_of(model_id: str) -> Optional[str]:
    """Family word of a standard id: its last name part, followed only by version numbers (``gpt-6.1-sol`` -> ``sol``,
    ``claude-haiku-5-5`` -> ``haiku``). Variants (``-900k``, ``-pro``) and tier words have no family."""
    words = [t for t in re.split(r"[-_.]", _undated(model_id)) if t and not t.isdigit()]
    last = words[-1] if words else ""
    return last if re.fullmatch(r"[a-z]+", last) and last not in _TIER_WORDS else None


def exact_model_id(requested: str, catalog: list[str]) -> Optional[str]:
    """Catalog id for *requested*: case-insensitive equality, else the single ``vendor/<requested>`` entry."""
    wanted = requested.strip().lower()
    hit = next((m for m in catalog if m.lower() == wanted), None)
    if hit is not None or "/" in wanted:
        return hit
    hits = [m for m in catalog if _bare(m) == wanted]
    return hits[0] if len(hits) == 1 else None


def canonical_model_id(requested: str, catalog: list[str]) -> Optional[str]:
    """The one id that is *requested* spelled differently: ``.``/``-``/``_`` separators and a trailing release date
    are ignored (``claude-haiku-4-5`` -> ``claude-haiku-4-5-20251001``). Prefers the undated id when both exist."""
    wanted = _canon(requested)
    hits = [m for m in catalog if ":" not in _bare(m) and _canon(m) == wanted]
    if len(hits) > 1:
        hits = [m for m in hits if _undated(m) == _bare(m)]
    return hits[0] if len(hits) == 1 else None


def _newest(ids: list[str]) -> Optional[str]:
    ids = [m for m in ids if ":" not in _bare(m)] or ids
    ranked = sorted(ids, key=_version, reverse=True)
    if len(ranked) > 1 and _version(ranked[0]) == _version(ranked[1]):
        return None
    return ranked[0] if ranked else None


def family_model_id(requested: str, catalog: list[str]) -> Optional[str]:
    """Newest standard id of the family *requested* names exactly (``astra``, ``sol``), else None. Tier words
    (``flash``) are never families and a bare vendor word (``gemini``) is no id's family, so neither picks a model."""
    wanted = requested.strip().lower()
    if wanted in _TIER_WORDS or not re.fullmatch(r"[a-z]+", wanted):
        return None
    return _newest([m for m in catalog if _family_of(m) == wanted])


def prefix_model_id(requested: str, catalog: list[str]) -> Optional[str]:
    """The one catalog id *requested* is the leading part of, cut at a ``-`` or ``:`` boundary (``step-5`` ->
    ``stepfun/step-5-preview:free``), else None. Version numbers must match, and ids that differ only by a
    ``:variant`` suffix count as one model (the plain id wins)."""
    wanted = requested.strip().lower()
    if not wanted or ":" in wanted:
        return None
    numbers = re.findall(r"\d+", wanted)
    by_base: dict[str, list[str]] = {}
    for model_id in catalog:
        key = model_id.lower() if "/" in wanted else _bare(model_id)
        if (len(key) > len(wanted) and key.startswith(wanted) and key[len(wanted)] in "-:"
                and re.findall(r"\d+", key.split(":", 1)[0]) == numbers):
            by_base.setdefault(key.split(":", 1)[0], []).append(model_id)
    if len(by_base) != 1:
        return None
    ids = next(iter(by_base.values()))
    plain = [m for m in ids if ":" not in _bare(m)]
    if plain:
        return plain[0] if len(plain) == 1 else None
    return ids[0] if len(ids) == 1 else None


def _clear_winner(wanted: str, candidates: list[tuple[str, str]]) -> Optional[str]:
    """Value of the ``(key, value)`` whose key is clearly closest to *wanted*, else None."""
    scored = sorted(((SequenceMatcher(None, wanted, key).ratio(), value) for key, value in candidates), reverse=True)
    if not scored or scored[0][0] < _MIN_RATIO:
        return None
    if len(scored) > 1 and scored[0][0] - scored[1][0] < _MIN_LEAD:
        return None
    return scored[0][1]


def closest_model_id(requested: str, catalog: list[str]) -> Optional[str]:
    """The catalog id a near-miss name clearly means, else None. Candidates must carry the same version numbers
    and ``:variant`` requests are never rewritten."""
    wanted = requested.strip().lower()
    if not wanted or ":" in wanted:
        return None
    numbers = re.findall(r"\d+", wanted)
    keyed = [(m.lower() if "/" in wanted else _bare(m), m) for m in catalog]
    return _clear_winner(wanted, [(k, m) for k, m in keyed if re.findall(r"\d+", k) == numbers])


def typo_family_model_id(requested: str, catalog: list[str]) -> Optional[str]:
    """Newest standard id of the family a near-miss family name clearly means (``atra`` -> ``astra``), else None."""
    wanted = requested.strip().lower()
    if wanted in _TIER_WORDS or not re.fullmatch(r"[a-z]+", wanted):
        return None
    families = sorted({f for f in map(_family_of, catalog) if f})
    family = _clear_winner(wanted, [(f, f) for f in families])
    return family_model_id(family, catalog) if family else None


# ``(model, listed ids) -> (catalog id, how)``; ``how`` is None when the id is the model as written.
_Match = tuple[Optional[str], Optional[str]]


def _exact_step(model: str, models: list[str]) -> _Match:
    hit = exact_model_id(model, models)
    base, sep, variant = model.partition(":")
    if hit is None and sep and variant:
        # ``model:variant`` (e.g. OpenRouter ``:nitro``) is a request modifier the picker never lists.
        base_hit = exact_model_id(base, models)
        hit = f"{base_hit}:{variant}" if base_hit is not None else None
    return hit, (None if hit is None or hit == model else "catalog id")


def _first(model: str, models: list[str], rules: tuple[tuple[Callable[[str, list[str]], Optional[str]], str], ...]
           ) -> _Match:
    if ":" in model:
        return None, None
    for rule, how in rules:
        hit = rule(model, models)
        if hit is not None:
            return hit, how
    return None, None


def _rule_step(model: str, models: list[str]) -> _Match:
    return _first(model, models, (
        (canonical_model_id, "same model, different spelling"),
        (family_model_id, "newest model in that family"),
        (prefix_model_id, "only listed id starting with it"),
    ))


def _typo_step(model: str, models: list[str]) -> _Match:
    return _first(model, models, (
        (closest_model_id, "typo correction"),
        (typo_family_model_id, "typo correction, newest model in that family"),
    ))


_STEPS = (_exact_step, _rule_step, _typo_step)


def _match_on(model: str, models: list[str]) -> _Match:
    """First step's ``(catalog id, how)`` for *model* in one provider's list; ``(None, None)`` on a miss."""
    for step in _STEPS:
        hit, how = step(model, models)
        if hit is not None:
            return hit, how
    return None, None


def _row_for(provider: str, rows: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    return next((r for r in rows if _same_provider(r.get("slug"), provider)), None)


def _families_hint(rows: list[dict[str, Any]], limit: int = 8) -> str:
    parts = []
    for row in rows[:limit]:
        families = sorted({f for f in map(_family_of, row.get("models") or []) if f})[:limit]
        parts.append(f"{row['slug']} ({', '.join(families)})" if families else str(row["slug"]))
    return f" Logged-in providers and model families: {'; '.join(parts)}." if parts else ""


def _not_listed(model: str, rows: list[dict[str, Any]], where: str) -> str:
    from difflib import get_close_matches
    listed = list(dict.fromkeys(m for r in rows for m in (r.get("models") or [])))
    close = get_close_matches(model, listed, n=3, cutoff=0.5)
    return (f"Model '{model}' is not in the model picker {where}." + (f" Similar: {', '.join(close)}." if close else "")
            + _families_hint(rows))


def _provider_missing(provider: str, rows: list[dict[str, Any]]) -> str:
    return (f"Provider '{provider}' is not in the model picker (not logged in or not configured)."
            + _families_hint(rows))


def _not_a_model(model: str, rows: list[dict[str, Any]]) -> str:
    row = _row_for(model, rows) or {}
    return (f"'{model}' is a provider, not a model. Set provider='{row.get('slug')}' and name a model."
            + _families_hint([row]))


def _deterministic_hit(name: str, rows: list[dict[str, Any]]) -> bool:
    return any(step(name, list(r.get("models") or []))[0] for step in _STEPS[:2] for r in rows)


def _join(notes: list[str]) -> Optional[str]:
    return " ".join(notes) or None


def _resolve_on_provider(provider: str, model: Optional[str], rows: list[dict[str, Any]],
                         notes: list[str]) -> tuple[str, Optional[str], Optional[str]]:
    row = _row_for(provider, rows)
    if row is None:
        raise ValueError(_provider_missing(provider, rows))
    slug = str(row["slug"])
    if provider.strip().lower() != slug.lower():
        notes.append(f"Provider '{provider}' -> '{slug}'.")
    models = list(row.get("models") or [])
    if not model or not models:
        return slug, model, _join(notes)
    hit, how = _match_on(model, models)
    if hit is None:
        raise ValueError(_not_listed(model, [row], f"for provider '{slug}'"))
    if hit != model:
        notes.append(f"Model '{model}' -> '{hit}' on provider '{slug}' ({how}).")
    return slug, hit, _join(notes)


def resolve_task_model(
    provider: Optional[str], model: Optional[str], catalog: TaskRouteCatalog,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """``(provider, model, note)`` for a task's pins. ``provider`` stays None when the inherited route serves the
    model; ``note`` reports every rewrite. The model picker's listing is the availability check: a model or provider
    it does not list raises ValueError with suggestions. An unreadable listing lets the spawn proceed unchanged."""
    if not provider and not model:
        return provider, model, None
    rows = catalog.rows()
    if not rows:
        return provider, model, None
    notes: list[str] = []
    if provider and not model and _row_for(provider, rows) is None and _deterministic_hit(provider, rows):
        # Weak callers put the model name in ``provider`` (``provider: "astra"``).
        notes.append(f"'{provider}' names a model, not a provider; used as the model.")
        provider, model = None, provider
    if provider:
        return _resolve_on_provider(provider, model, rows, notes)
    assert model is not None

    inherited = catalog.configured[0] if catalog.configured else None
    ranked = sorted(
        enumerate(rows),
        key=lambda ir: (_provider_tier(str(ir[1].get("slug") or "")),
                        not any(_same_provider(ir[1].get("slug"), c) for c in catalog.configured), ir[0]))
    for step in _STEPS:
        if step is not _exact_step and _row_for(model, rows) is not None:
            # A provider sent as the model must never become a look-alike model (``codex`` -> ``gpt-5.3-codex``).
            raise ValueError(_not_a_model(model, rows))
        for _, row in ranked:
            hit, how = step(model, list(row.get("models") or []))
            if hit is None:
                continue
            slug = str(row["slug"])
            if _same_provider(slug, inherited) and not catalog.direct_endpoint:
                if hit != model:
                    notes.append(f"Model '{model}' -> '{hit}' on provider '{slug}' ({how}).")
                return None, hit, _join(notes)
            notes.append(f"Model '{model}' runs on provider '{slug}'" + (f" as '{hit}' ({how})" if hit != model else "") + ".")
            return slug, hit, _join(notes)
    raise ValueError(_not_listed(model, rows, "for any logged-in provider"))
