"""Vision routing for ``computer_use`` capture results: return the ``_multimodal`` screenshot envelope, or
pre-analyse it via ``auxiliary.vision`` so the main model only sees text. A text-only main model, or a provider
that rejects multimodal tool results, turns the envelope into a hard 400/404.

The decision IS the shared native-tool-result gate (``tools.vision_tools._native_tool_result_images``) that
``vision_analyze``, browser screenshots and MCP image results use: ``agent.image_input_mode``, an explicit
``auxiliary.vision`` backend (the de-facto route in ``auto``) or the capability lookup (config ``supports_vision`` →
catalog → local probes) picks ``native``, AND the route accepts images inside tool results (profile veto first).
One predicate, so a screenshot takes the same lane whichever tool produced it (#115248). Anything the gate cannot
vouch for goes to aux: one extra LLM call beats a screenshot the model cannot read.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def should_route_capture_to_aux_vision(provider: str, model: str, cfg: Optional[Dict[str, Any]]) -> bool:
    """True iff the screenshot should be pre-analysed via aux vision; False keeps the multimodal envelope. *provider* is
    the lower-case canonical id, *model* the slug sent to the provider, *cfg* the loaded ``config.yaml`` dict (or None)."""
    from tools.vision_tools import _native_tool_result_images
    return not _native_tool_result_images(provider, model, cfg)


__all__ = ["should_route_capture_to_aux_vision"]
