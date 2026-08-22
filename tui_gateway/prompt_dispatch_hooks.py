"""Stable plugin contract for pre-agent TUI prompt dispatch decisions.

The resolver is intentionally independent of TUI server internals. Plugins see
only immutable public values, while the gateway keeps ownership of session
state, history persistence, and event delivery.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Sequence


logger = logging.getLogger(__name__)

REQUIRED_HANDLER_UNAVAILABLE_TEXT = (
    "Das erforderliche OCR-Freigabe-Gate ist momentan nicht verfügbar. "
    "Das Bild wurde nicht an einen Agenten weitergegeben. Bitte versuche es "
    "nach der technischen Prüfung erneut."
)


@dataclass(frozen=True)
class PromptDispatchDecision:
    """Validated host-side outcome of all pre-prompt plugin callbacks."""

    action: Literal["allow", "respond", "block"]
    handler: str | None = None
    text: str = ""
    reason: str = ""


def normalize_required_prompt_handler(value: Any) -> str | None:
    """Normalize the optional opaque handler id carried by a live session."""

    handler = str(value or "").strip()
    return handler or None


def resolve_prompt_dispatch_results(
    results: Iterable[Any],
    *,
    required_prompt_handler: str | None,
    has_images: bool,
) -> PromptDispatchDecision:
    """Validate plugin results and enforce a required image handler.

    Required image turns accept only a directive from the exact handler named
    by the client. Optional turns prefer the first valid response so a generic
    handlerless ``allow`` cannot mask a later image consumer.
    """

    required_handler = normalize_required_prompt_handler(required_prompt_handler)
    required_image_turn = bool(required_handler and has_images)
    optional_allow: PromptDispatchDecision | None = None

    for result in results:
        if not isinstance(result, dict):
            continue
        action = str(result.get("action") or "").strip().lower()
        if action not in {"allow", "respond"}:
            continue
        handler = normalize_required_prompt_handler(result.get("handler"))
        if required_image_turn and handler != required_handler:
            continue
        reason = str(result.get("reason") or "").strip()

        if action == "allow":
            decision = PromptDispatchDecision(
                action="allow",
                handler=handler,
                reason=reason,
            )
            if required_image_turn:
                return decision
            if optional_allow is None:
                optional_allow = decision
            continue

        text = str(result.get("text") or "").strip()
        if not text:
            continue
        return PromptDispatchDecision(
            action="respond",
            handler=handler,
            text=text,
            reason=reason,
        )

    if required_image_turn:
        return PromptDispatchDecision(
            action="block",
            handler=required_handler,
            text=REQUIRED_HANDLER_UNAVAILABLE_TEXT,
            reason="required_prompt_handler_unavailable",
        )
    return optional_allow or PromptDispatchDecision(action="allow")


def invoke_pre_prompt_dispatch(
    *,
    session_id: str,
    session_key: str,
    source: str,
    text: Any,
    attached_images: Sequence[str],
    required_prompt_handler: str | None,
) -> PromptDispatchDecision:
    """Invoke plugins with public immutable data and resolve their directives."""

    images = tuple(str(path) for path in attached_images)
    required_handler = normalize_required_prompt_handler(required_prompt_handler)
    try:
        from hermes_cli.lifecycle import invoke_hook

        results = invoke_hook(
            "pre_prompt_dispatch",
            session_id=str(session_id),
            session_key=str(session_key),
            surface="tui",
            source=str(source or "tui"),
            text=text if isinstance(text, str) else str(text),
            attached_images=images,
            required_prompt_handler=required_handler,
        )
    except Exception:
        logger.warning("pre_prompt_dispatch hook invocation failed", exc_info=True)
        results = []

    return resolve_prompt_dispatch_results(
        results,
        required_prompt_handler=required_handler,
        has_images=bool(images),
    )
