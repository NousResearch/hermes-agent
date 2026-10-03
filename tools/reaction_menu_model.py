"""Bounded choices for a Matrix reaction menu."""

from __future__ import annotations

import json
from dataclasses import dataclass

MIN_OPTIONS = 1
MAX_OPTIONS = 5


class MenuValidationError(ValueError):
    """A menu cannot be rendered within the supported limits."""


def _text(value: object, field: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise MenuValidationError(f"{field} must be non-empty text of at most {limit} characters")
    return value.strip()


@dataclass(frozen=True)
class MenuOption:
    emoji: str
    label: str
    payload: str


@dataclass(frozen=True)
class ReactionMenu:
    prompt: str
    options: tuple[MenuOption, ...]
    context_id: str | None = None

    @classmethod
    def from_arguments(cls, prompt: object, options: object, context_id: object = None) -> ReactionMenu:
        prompt = _text(prompt, "prompt", 500)
        if context_id is not None:
            context_id = _text(context_id, "context_id", 128)
        if not isinstance(options, list) or not MIN_OPTIONS <= len(options) <= MAX_OPTIONS:
            raise MenuValidationError(f"a menu needs {MIN_OPTIONS} to {MAX_OPTIONS} options")

        normalized = []
        seen_emoji: set[str] = set()
        for index, option in enumerate(options):
            if not isinstance(option, dict) or set(option) != {"emoji", "label", "payload"}:
                raise MenuValidationError(f"option {index} must contain emoji, label and payload")
            emoji = _text(option["emoji"], "emoji", 32)
            label = _text(option["label"], "label", 120)
            payload = _text(option["payload"], "payload", 2000)
            if emoji in seen_emoji:
                raise MenuValidationError(f"duplicate emoji {emoji!r} in menu")
            seen_emoji.add(emoji)
            normalized.append(MenuOption(emoji, label, payload))
        return cls(prompt, tuple(normalized), context_id)

    def choice_body(self, option: MenuOption) -> str:
        return "[menu-choice]\n" + json.dumps({
            "prompt": self.prompt, "context_id": self.context_id,
            "emoji": option.emoji, "label": option.label, "payload": option.payload,
        }, ensure_ascii=False)
