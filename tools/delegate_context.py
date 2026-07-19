"""Deterministic, operator-controlled context projection for delegated agents.

The recent window counts eligible textual message entries, not semantic turns.
Nonempty user/assistant string content is eligible, along with runtime-authenticated
OOB steer sidecars; tool content itself is never projected. Bounds keep malformed
operator configuration small and deterministic: 1..100 entries and
256..1,000,000 characters.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Literal, Mapping

_MIN_RECENT_TURNS = 1
_MAX_RECENT_TURNS = 100
_MIN_MAX_CHARS = 256
_MAX_MAX_CHARS = 1_000_000
_TRUNCATION_MARKER = "\n...[deterministic head/tail truncation]...\n"
AUTHENTICATED_OOB_KEY = "_hermes_authenticated_oob"


@dataclass(frozen=True)
class DelegationContextPolicy:
    """Context compilation settings; explicit mode preserves legacy behavior."""

    mode: Literal["explicit", "recent_projection"] = "explicit"
    recent_turns: int = 3
    max_chars: int = 12000

    def __post_init__(self) -> None:
        if self.mode not in ("explicit", "recent_projection"):
            raise ValueError(f"unknown delegation context mode: {self.mode!r}")
        if not isinstance(self.recent_turns, int) or isinstance(self.recent_turns, bool):
            raise TypeError("recent_turns must be an integer")
        if not isinstance(self.max_chars, int) or isinstance(self.max_chars, bool):
            raise TypeError("max_chars must be an integer")
        object.__setattr__(
            self,
            "recent_turns",
            min(_MAX_RECENT_TURNS, max(_MIN_RECENT_TURNS, self.recent_turns)),
        )
        object.__setattr__(
            self,
            "max_chars",
            min(_MAX_MAX_CHARS, max(_MIN_MAX_CHARS, self.max_chars)),
        )


def _strict_config_int(value: Any) -> int:
    if type(value) is int:
        return value
    if type(value) is str:
        digits = value[1:] if value[:1] in ("+", "-") else value
        if digits and all(character in "0123456789" for character in digits):
            return int(value, 10)
    raise ValueError("delegation context bounds must be base-10 integers")


def load_delegation_context_policy(config: Any) -> DelegationContextPolicy:
    """Load policy from a delegation mapping, failing safely to explicit mode."""
    if not isinstance(config, Mapping):
        return DelegationContextPolicy()
    mode = config.get("context_mode", "explicit")
    if mode not in ("explicit", "recent_projection"):
        return DelegationContextPolicy()
    try:
        recent_turns = _strict_config_int(config.get("context_recent_turns", 3))
        max_chars = _strict_config_int(config.get("context_max_chars", 12000))
        return DelegationContextPolicy(
            mode=mode,
            recent_turns=recent_turns,
            max_chars=max_chars,
        )
    except (TypeError, ValueError, OverflowError):
        return DelegationContextPolicy()


def _fit_head_tail(text: str, budget: int) -> str:
    if len(text) <= budget:
        return text
    if budget <= len(_TRUNCATION_MARKER):
        return text[:budget]
    payload = budget - len(_TRUNCATION_MARKER)
    head = (payload + 1) // 2
    tail = payload - head
    return text[:head] + _TRUNCATION_MARKER + (text[-tail:] if tail else "")


def attach_authenticated_oob(message: dict[str, Any], text: str) -> None:
    """Attach runtime-authenticated steer text outside untrusted tool content."""
    if not isinstance(text, str) or not text.strip():
        return
    existing = message.get(AUTHENTICATED_OOB_KEY)
    entries = list(existing) if isinstance(existing, list) else []
    entries.append(text)
    message[AUTHENTICATED_OOB_KEY] = entries


def _project_recent(parent_messages: Any, recent_turns: int) -> str:
    if not isinstance(parent_messages, (list, tuple)):
        return ""
    eligible: deque[str] = deque(maxlen=recent_turns)
    for message in parent_messages:
        if not isinstance(message, Mapping):
            continue
        role = message.get("role")
        content = message.get("content")
        if role == "tool":
            authenticated = message.get(AUTHENTICATED_OOB_KEY)
            if isinstance(authenticated, list):
                for text in authenticated:
                    if isinstance(text, str) and text.strip():
                        eligible.append(f"[OOB user]\n{text}")
            continue
        if (
            role not in ("user", "assistant")
            or not isinstance(content, str)
            or not content.strip()
        ):
            continue
        eligible.append(f"[{role}]\n{content}")
    return "\n\n".join(eligible)


def compile_delegation_context(
    *,
    explicit_context: str | None,
    parent_messages: Any,
    policy: DelegationContextPolicy,
) -> str | None:
    """Compile context only, with explicit text authoritative and budget-first."""
    if policy.mode == "explicit":
        return explicit_context

    explicit = explicit_context if isinstance(explicit_context, str) else None
    if explicit is not None and len(explicit) >= policy.max_chars:
        return _fit_head_tail(explicit, policy.max_chars)

    projection = _project_recent(parent_messages, policy.recent_turns)
    if not projection:
        return explicit_context

    if explicit:
        remaining = policy.max_chars - len(explicit) - 2
        if remaining <= 0:
            return explicit
        return f"{explicit}\n\n{_fit_head_tail(projection, remaining)}"
    return _fit_head_tail(projection, policy.max_chars)
