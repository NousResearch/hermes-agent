"""Vision-routing decisions for ``computer_use`` capture results. ``capture`` (mode som|vision) returns a
``_multimodal`` screenshot envelope as the tool result. A text-only main model, or a provider that rejects multimodal
tool results, turns that into a hard 400/404 — even with a working ``auxiliary.vision`` model in config. This module
decides: multimodal envelope, or pre-analyse via aux vision so the main model only ever sees text?

Decision order (mirrors ``vision_analyze``):
1. ``auxiliary.vision`` explicitly configured (provider not ""/"auto", or model / base_url set) → aux routing; users
   who pay for a vision model want it used.
2. User-declared ``supports_vision`` for the active route (escape hatch for custom/local VLMs absent from models.dev)
   → honour it (True → multimodal).
3. Provider+model carries images inside tool-result messages AND models.dev says ``supports_vision=True`` → multimodal.
4. Everything else (non-vision model, provider rejecting multimodal tool results, lookup failure) → aux routing.

Fails *closed* toward aux routing when metadata is missing or ambiguous: a screenshot sent to a model that cannot read
it is a hard failure, while aux routing costs one extra LLM call and yields a usable description.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

def _explicit_aux_vision_override(cfg: Optional[Dict[str, Any]]) -> bool:
    """True when ``auxiliary.vision`` carries a non-default user override; mirrors ``agent.image_routing`` so the capture
    and user-attached-image paths agree. ``provider: "auto"``, blanks or a missing block are *not* explicit."""
    aux = cfg.get("auxiliary") if isinstance(cfg, dict) else None
    vision = aux.get("vision") if isinstance(aux, dict) else None
    if not isinstance(vision, dict):
        return False
    provider = str(vision.get("provider") or "").strip().lower()
    return provider not in ("", "auto") or any(str(vision.get(k) or "").strip() for k in ("model", "base_url"))

def _lookup_user_declared_supports_vision(provider: str, model: str, cfg: Optional[Dict[str, Any]]) -> Optional[bool]:
    """Config-declared ``supports_vision`` for the active route (None on failure)."""
    try:
        from agent.image_routing import _supports_vision_override
        return _supports_vision_override(cfg, provider, model)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("computer_use vision_routing: config override lookup failed: %s", exc)
        return None

def _models_dev_supports_vision(provider: str, model: str, cfg: Optional[Dict[str, Any]]) -> Optional[bool]:
    """Raw models.dev capability lookup — fallback when ``agent.image_routing`` is unavailable."""
    from agent.models_dev import get_model_capabilities
    caps = get_model_capabilities(provider, model)
    return None if caps is None else bool(getattr(caps, "supports_vision", False))

def _lookup_supports_vision(provider: str, model: str, cfg: Optional[Dict[str, Any]] = None) -> Optional[bool]:
    """Config/models.dev ``supports_vision`` for *(provider, model)*; prefers
    ``agent.image_routing._lookup_supports_vision``. Any lookup error → None (caller fails closed to aux)."""
    if not provider or not model:
        return None
    try:
        from agent.image_routing import _lookup_supports_vision as lookup
    except Exception:
        lookup = _models_dev_supports_vision
    try:
        return lookup(provider, model, cfg)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("computer_use vision_routing: caps lookup failed for %s:%s — %s", provider, model, exc)
        return None

def _provider_accepts_multimodal_tool_result(provider: str, model: str) -> Optional[bool]:
    """Whether *provider*+*model* carries images inside tool-result messages; reuses ``tools.vision_tools`` to stay in
    lockstep with the ``vision_analyze`` fast path. None on import failure so callers fall back to aux, not guess."""
    if not provider:
        return None
    try:
        from tools.vision_tools import _supports_media_in_tool_results
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("computer_use vision_routing: tool-result support lookup failed: %s", exc)
        return None
    return bool(_supports_media_in_tool_results(provider, model))

def should_route_capture_to_aux_vision(provider: str, model: str, cfg: Optional[Dict[str, Any]]) -> bool:
    """True iff the screenshot should be pre-analysed via aux vision; False keeps the multimodal envelope. *provider* is
    the lower-case canonical id, *model* the slug sent to the provider, *cfg* the loaded ``config.yaml`` dict (or None).
    Steps follow the module docstring's decision order."""
    # auto: an explicitly configured auxiliary.vision backend is the DE-FACTO choice — the user named a
    # dedicated vision model, so that's what they want images to go through, even when the main model has
    # native vision (maintainer decision, 2026-08-28, reversing #29135's fallback-only posture: config that
    # only takes effect when the main model gets worse is a trap, not a setting). Native vision remains the
    # default for unconfigured installs, and the fallback when the aux backend is unset.
    if _explicit_aux_vision_override(cfg):
        return True
    user_declared = _lookup_user_declared_supports_vision(provider, model, cfg)
    if isinstance(user_declared, bool):  # True → multimodal, False → aux
        return not user_declared
    if not _provider_accepts_multimodal_tool_result(provider, model):
        return True
    return _lookup_supports_vision(provider, model, cfg) is not True

__all__ = ["should_route_capture_to_aux_vision"]
