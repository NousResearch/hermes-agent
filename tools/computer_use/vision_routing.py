"""Vision-routing decisions for ``computer_use`` capture results.

Background
----------
``computer_use(action='capture', mode='som'|'vision')`` returns a
``_multimodal`` envelope containing the captured screenshot. That envelope
is delivered back to the **active session model** as the tool result. When
the active main model has no vision capability (e.g. text-only or
text+code-only models), or when the active provider rejects multimodal
content inside tool-result messages, the screenshot trips a 404 / 400 at
the provider boundary and the agent loop reports a hard tool failure.

Issue #24015 reports this regression for the ``cua-driver`` backend:
configuring ``auxiliary.vision`` (a dedicated vision-capable model) in
``config.yaml`` was silently ignored — the screenshot was still routed at
the *main* model and failed with HTTP 404 ``No endpoints found that
support image input`` even though a perfectly good vision backend was
sitting in config waiting to be used.

This module centralises the small policy decision: should a captured
screenshot be returned as multimodal content (main model handles vision
natively) or pre-analysed via the auxiliary vision pipeline so the main
model only ever sees text?

Behaviour (mirrors ``vision_analyze`` for consistency)
------------------------------------------------------
* If the user explicitly configured ``auxiliary.vision`` (any of
  ``provider``, ``model``, or ``base_url`` non-empty / not ``"auto"``),
  the screenshot is routed through the aux vision pipeline. Users who
  pay for a dedicated vision model usually want it used.
* Otherwise, if the user explicitly declared the active model vision-capable
  via ``model.supports_vision`` / provider model config, return ``False``.
  This is the escape hatch for custom/local OpenAI-compatible VLM routes that
  are absent from models.dev and provider allowlists.
* Otherwise, if the active main model+provider can carry an image inside
  a tool-result message AND the model reports ``supports_vision=True``
  in models.dev metadata, return ``False`` (use the multimodal path).
* In every other case (non-vision main model, provider that does not
  accept multimodal tool results, lookup failure), route through aux
  vision so the main model receives a text description it can act on.

The decision intentionally fails *closed* (i.e. towards aux routing) when
metadata is missing or ambiguous: returning a screenshot to a model that
cannot read it is a hard tool failure, while routing it through aux costs
one extra LLM call and yields a usable description.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _explicit_aux_vision_override(cfg: Optional[Dict[str, Any]]) -> bool:
    """True when ``auxiliary.vision`` carries a non-default user override.

    Mirrors ``agent.image_routing._explicit_aux_vision_override`` so the
    capture path and the user-attached-image path agree on what counts as
    an explicit user request for the aux vision pipeline. ``provider:
    "auto"``, blank values, or a missing block all count as *not*
    explicit.
    """
    if not isinstance(cfg, dict):
        return False
    aux = cfg.get("auxiliary") or {}
    if not isinstance(aux, dict):
        return False
    vision = aux.get("vision") or {}
    if not isinstance(vision, dict):
        return False

    provider = str(vision.get("provider") or "").strip().lower()
    model = str(vision.get("model") or "").strip()
    base_url = str(vision.get("base_url") or "").strip()

    if provider in ("", "auto") and not model and not base_url:
        return False
    return True


def _lookup_user_declared_supports_vision(
    provider: str,
    model: str,
    cfg: Optional[Dict[str, Any]],
) -> Optional[bool]:
    """Return config-declared ``supports_vision`` for the active route."""
    try:
        from agent.image_routing import _supports_vision_override
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(
            "computer_use vision_routing: config override lookup import failed: %s",
            exc,
        )
        return None
    try:
        return _supports_vision_override(cfg, provider, model)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(
            "computer_use vision_routing: config override lookup failed: %s",
            exc,
        )
        return None


def _lookup_supports_vision(
    provider: str,
    model: str,
    cfg: Optional[Dict[str, Any]] = None,
) -> Optional[bool]:
    """Return config/models.dev ``supports_vision`` for *(provider, model)*."""
    if not provider or not model:
        return None
    try:
        from agent.image_routing import _lookup_supports_vision as _lookup_image_supports
    except Exception:
        _lookup_image_supports = None
    if _lookup_image_supports is not None:
        try:
            return _lookup_image_supports(provider, model, cfg)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(
                "computer_use vision_routing: image-routing caps lookup failed "
                "for %s:%s — %s",
                provider, model, exc,
            )
            return None
    try:
        from agent.models_dev import get_model_capabilities
        caps = get_model_capabilities(provider, model)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(
            "computer_use vision_routing: caps lookup failed for %s:%s — %s",
            provider, model, exc,
        )
        return None
    if caps is None:
        return None
    return bool(getattr(caps, "supports_vision", False))


def _provider_accepts_multimodal_tool_result(provider: str, model: str) -> Optional[bool]:
    """Return whether *provider*+*model* carries images inside tool-result messages.

    Reuses ``tools.vision_tools._supports_media_in_tool_results`` so the
    capture-routing decision stays in lockstep with the
    ``vision_analyze`` native fast path. Returns None on import failure
    so callers fall back to aux routing rather than guessing.
    """
    if not provider:
        return None
    try:
        from tools.vision_tools import _supports_media_in_tool_results
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(
            "computer_use vision_routing: tool-result support lookup failed: %s",
            exc,
        )
        return None
    return bool(_supports_media_in_tool_results(provider, model))


def should_route_capture_to_aux_vision(
    provider: str,
    model: str,
    cfg: Optional[Dict[str, Any]],
) -> bool:
    """Return True iff the captured screenshot should be pre-analysed via aux vision.

    Args:
      provider: active inference provider id (e.g. ``"openrouter"``,
        ``"anthropic"``, ``"openai-codex"``). Lower-case canonical id.
      model:    active main model slug as it would be sent to the provider.
      cfg:      loaded ``config.yaml`` dict (or None).

    Returns:
      ``True`` when the caller should hand the screenshot to the aux vision
      pipeline (and surface a text-only tool result). ``False`` when the
      caller should keep the existing multimodal envelope (main model
      handles vision natively).
    """
    if _explicit_aux_vision_override(cfg):
        return True

    user_declared = _lookup_user_declared_supports_vision(provider, model, cfg)
    if user_declared is True:
        return False
    if user_declared is False:
        return True

    accepts_tool_image = _provider_accepts_multimodal_tool_result(provider, model)
    if accepts_tool_image is None or accepts_tool_image is False:
        return True

    supports_vision = _lookup_supports_vision(provider, model, cfg)
    if supports_vision is True:
        return False
    return True


__all__ = [
    "should_route_capture_to_aux_vision",
]
