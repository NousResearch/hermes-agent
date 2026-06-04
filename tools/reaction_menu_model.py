"""Source option validation from Dan Montgomery's reaction menu contribution."""

from typing import Any, Dict, List

MIN_OPTIONS = 1
MAX_OPTIONS = 5

# Reaction reserved by the menu choreography itself (reload / regenerate).
# An option may NOT claim it.
RELOAD_EMOJI = "♻️"  # ♻️


class MenuValidationError(ValueError):
    """Raised when an option list fails validation."""


def validate_options(options: Any) -> List[Dict[str, Any]]:
    """Validate and normalise a menu's option list.

    Each option is a dict with:
      * ``emoji``    (str, required) — the reaction key / number anchor.
      * ``label``    (str, required) — human-readable choice text.
      * ``payload``  (str, required) — injected as the synthetic turn body.
      * ``terminal`` (bool, optional, default False) — when True the chosen
        path ends the menu lifecycle: no ``♻️`` reload reaction is seeded.

    Rules: 1–5 options; ``emoji``/``label``/``payload`` non-empty; emoji unique
    within the menu and never the reserved ``♻️``; ``silent: true`` is reserved
    and rejected in v1 (spec §8).

    Returns the normalised list.  Raises :class:`MenuValidationError` otherwise.
    """
    if not isinstance(options, list):
        raise MenuValidationError("options must be a list")
    if not (MIN_OPTIONS <= len(options) <= MAX_OPTIONS):
        raise MenuValidationError(
            f"a menu needs {MIN_OPTIONS}–{MAX_OPTIONS} options, got {len(options)}"
        )

    normalized: List[Dict[str, Any]] = []
    seen_emoji: set[str] = set()
    for idx, opt in enumerate(options):
        if not isinstance(opt, dict):
            raise MenuValidationError(f"option {idx} must be an object")
        if opt.get("silent"):
            raise MenuValidationError(
                "silent menus are reserved and not supported in this version"
            )
        emoji = str(opt.get("emoji", "")).strip()
        label = str(opt.get("label", "")).strip()
        payload = str(opt.get("payload", "")).strip()
        if not emoji:
            raise MenuValidationError(f"option {idx} is missing 'emoji'")
        if not label:
            raise MenuValidationError(f"option {idx} is missing 'label'")
        if not payload:
            raise MenuValidationError(f"option {idx} is missing 'payload'")
        if emoji == RELOAD_EMOJI:
            raise MenuValidationError(
                f"option {idx} uses the reserved reload reaction {RELOAD_EMOJI}"
            )
        if emoji in seen_emoji:
            raise MenuValidationError(f"duplicate emoji {emoji!r} in menu")
        seen_emoji.add(emoji)
        normalized.append({
            "emoji": emoji,
            "label": label,
            "payload": payload,
            "terminal": bool(opt.get("terminal", False)),
        })
    return normalized
