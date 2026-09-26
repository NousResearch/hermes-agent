"""Vision-routing decisions for ``computer_use`` capture results. ``capture`` (mode som|vision) returns a
``_multimodal`` screenshot envelope as the tool result. A text-only main model, or a provider that rejects multimodal
tool results, turns that into a hard 400/404 — even with a working ``auxiliary.vision`` model in config. This module
decides: multimodal envelope, or pre-analyse via aux vision so the main model only ever sees text?

Decision order (mirrors inbound image routing where the tool-result provider permits media):
1. With `agent.vision_capability_first`, a model advertised as vision-capable uses native media
   when the provider accepts image tool results; a text-only model uses aux. Unknown falls through.
2. User-declared `supports_vision: true` for the active route keeps multimodal (the custom/local
   escape hatch), even with an explicitly configured auxiliary backend.
3. An explicitly configured `auxiliary.vision` routes to aux by default.
4. The shared `vision_analyze` tool-result gate determines the remaining cases; missing or
   ambiguous capability routes to aux.

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

def _provider_accepts_multimodal_tool_result(provider: str, model: str, cfg: Optional[Dict[str, Any]] = None) -> Optional[bool]:
    """Whether *provider*+*model* may carry images inside tool-result messages — the SAME predicate the
    ``vision_analyze`` fast path uses (#115248: the two gates disagreed for deepseek/deepseek-flash, so the route
    depended on which tool asked). None on import failure so callers fall back to aux, not guess."""
    if not provider:
        return None
    try:
        from tools.vision_tools import _accepts_tool_result_images
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("computer_use vision_routing: tool-result support lookup failed: %s", exc)
        return None
    return bool(_accepts_tool_result_images(provider, model, cfg))

def should_route_capture_to_aux_vision(provider: str, model: str, cfg: Optional[Dict[str, Any]]) -> bool:
    """True iff the screenshot should be pre-analysed via aux vision; False keeps the multimodal envelope. *provider* is
    the lower-case canonical id, *model* the slug sent to the provider, *cfg* the loaded ``config.yaml`` dict (or None).
    Steps follow the module docstring's decision order."""
    # auto: an explicitly configured auxiliary.vision backend is the DE-FACTO choice — the user named a
    # dedicated vision model, so that's what they want images to go through, even when the main model has
    # native vision (maintainer decision, 2026-08-28, reversing #29135's fallback-only posture: config that
    # only takes effect when the main model gets worse is a trap, not a setting). Native vision remains the
    # default for unconfigured installs, and the fallback when the aux backend is unset.
    #
    # Exception (parity with agent.image_routing): a per-model ``supports_vision: true`` DECLARED in config
    # is a direct user statement that this model takes images, and it beats the aux-de-facto rule — same
    # precedence the declaration already has on the no-aux path below (``test_user_declared_vision_support_
    # keeps_custom_provider_native``), so the declaration means one thing regardless of aux config.
    user_declared = _lookup_user_declared_supports_vision(provider, model, cfg)
    agent_cfg = cfg.get("agent") if isinstance(cfg, dict) else None
    if isinstance(agent_cfg, dict) and agent_cfg.get("vision_capability_first") is True:
        from agent.image_routing import _lookup_supports_vision

        capability = _lookup_supports_vision(provider, model, cfg)
        if capability is True and _provider_accepts_multimodal_tool_result(provider, model, cfg):
            return False
        if capability is False:
            return True
    if user_declared is True:
        return False
    if _explicit_aux_vision_override(cfg):
        return True
    if isinstance(user_declared, bool):  # False → aux (declaration says text-only)
        return not user_declared
    # The shared gate already folds the capability lookup in; demanding a second `is True` here made
    # the two lanes disagree for whitelisted providers whose model the catalog does not know.
    return not _provider_accepts_multimodal_tool_result(provider, model, cfg)

__all__ = ["should_route_capture_to_aux_vision"]
