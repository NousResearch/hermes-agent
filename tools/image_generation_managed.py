"""Managed ("Nous Subscription") image generation: one picker row, one model catalog.

Three gateways sit behind the single stored selection ``image_gen.provider: nous``; the stored
``image_gen.model`` decides which one serves a request:

* ``fal``    — the FAL managed gateway, every id in the in-tree ``FAL_MODELS`` catalog;
* ``krea``   — the Krea managed gateway, the ``plugins/image_gen/krea`` model ids;
* ``portal`` — Nous Portal chat-completions image models (``plugins/image_gen/openrouter``'s
  ``nous`` provider), anything else.

Before this module the second and third gateways each had their own picker row that also
wrote ``provider: nous`` — every managed row read "active" at once and picking the Portal row
silently generated on FAL.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

FAL, KREA, PORTAL = "fal", "krea", "portal"

# Portal ids that are the same model as a FAL or Krea catalog entry. FAL wins (it is the
# gateway the free tool pool funds); Krea 2 goes through its native gateway, never FAL or the
# Portal. Only cross-id duplicates live here — identical ids collapse by key.
_PORTAL_DUPLICATE_OF = {
    "openai/gpt-image-2": "fal-ai/gpt-image-2",
    "google/gemini-3-pro-image": "fal-ai/nano-banana-pro",
    "google/gemini-3.1-flash-image": "fal-ai/nano-banana-2",
    "google/gemini-3.1-flash-lite-image": "google/nano-banana-2-lite",
    "krea/krea-2-medium": "krea-2-medium",
    "krea/krea-2-medium-turbo": "krea-2-medium-turbo",
    "krea/krea-2-large": "krea-2-large",
}
_FAL_KREA_PREFIX = "fal-ai/krea/"


def managed_backend_for_model(model_id: Optional[str]) -> str:
    """Gateway that serves ``model_id`` under the managed selection (unset → FAL default)."""
    from plugins.image_gen.krea import KREA_MODEL_IDS
    from tools.image_generation_catalog import FAL_MODELS

    candidate = model_id.strip() if isinstance(model_id, str) else ""
    if not candidate or candidate in FAL_MODELS:
        return FAL
    if candidate in KREA_MODEL_IDS:
        return KREA
    return PORTAL


def _plugin_rows(name: str) -> list:
    """``list_models()`` of a registered image gen plugin; ``[]`` when unavailable."""
    from tools.image_generation_tool import _get_plugin_provider

    try:
        provider = _get_plugin_provider(name)
        return list(provider.list_models() or []) if provider is not None else []
    except Exception:  # noqa: BLE001 - a broken plugin must not empty the whole picker
        return []


def managed_image_catalog(
    *, include_krea: bool = True, include_portal: bool = True,
) -> Tuple[Dict[str, Dict[str, Any]], str]:
    """``({model_id: metadata}, default_model)`` for the managed row's model picker.

    FAL catalog first (minus the Krea-on-FAL entries), then native Krea, then Portal models
    that are not another entry in disguise. Every row carries ``backend`` so callers can hide
    the gateways an account is not entitled to.
    """
    from tools.image_generation_catalog import DEFAULT_MODEL, FAL_MODELS

    catalog: Dict[str, Dict[str, Any]] = {
        mid: {**meta, "backend": FAL} for mid, meta in FAL_MODELS.items()
        if not mid.startswith(_FAL_KREA_PREFIX)}
    if include_krea:
        for row in _plugin_rows("krea"):
            catalog[row["id"]] = {**row, "backend": KREA}
    if include_portal:
        for row in _plugin_rows("nous"):
            mid = row["id"]
            if mid in catalog or _PORTAL_DUPLICATE_OF.get(mid) in catalog:
                continue
            catalog[mid] = {**row, "backend": PORTAL}
    return catalog, DEFAULT_MODEL
