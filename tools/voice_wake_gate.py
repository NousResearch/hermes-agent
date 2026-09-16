"""Wake-name gate for shared voice channels.

When two or more humans are in the bot's voice channel, an utterance must
start or end with a wake name or it is dropped. One human talks naturally.
Pure functions — no Discord, no websocket.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence, Tuple

_DEFAULT_WAKE = "Hermes"
_LEADING_SOFT = r"(?:hey|ok|okay|yo)\s+"


def normalize_wake_names(names: Optional[Iterable[str]]) -> List[str]:
    """Keep unique 1–2 word names, longest-first for matching."""
    out: List[str] = []
    seen = set()
    for raw in names or []:
        parts = str(raw or "").split()
        if not parts:
            continue
        label = " ".join(parts[:2])
        key = label.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(label)
    out.sort(key=lambda s: (-len(s.split()), -len(s), s.casefold()))
    return out


def default_wake_names(*extra: str) -> List[str]:
    """Bot display name (and its first token) plus Hermes."""
    names: List[str] = []
    for item in extra:
        if item:
            names.append(item)
    names.append(_DEFAULT_WAKE)
    return normalize_wake_names(names)


def wake_gate_active(policy: str, human_count: int) -> bool:
    """True when utterances must carry a wake name."""
    p = str(policy or "auto").strip().lower()
    if p in ("false", "never", "off", "0", "no"):
        return False
    if p in ("true", "always", "on", "1", "yes"):
        return True
    try:
        n = int(human_count)
    except (TypeError, ValueError):
        n = 0
    return n >= 2


def apply_wake_gate(
    text: str,
    *,
    policy: str = "auto",
    human_count: int = 1,
    names: Optional[Sequence[str]] = None,
) -> Tuple[bool, str]:
    """Return ``(accepted, text_for_brain)``.

    When the gate is inactive, ``accepted`` is True and the text is only
    stripped. When active, a wake name must start or end the utterance;
    the name (and a leading hey/ok/yo) is removed before dispatch.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return False, ""
    if not wake_gate_active(policy, human_count):
        return True, cleaned
    labels = normalize_wake_names(names)
    if not labels:
        labels = [_DEFAULT_WAKE]
    alt = "|".join(re.escape(n) for n in labels)
    start = re.compile(
        rf"^(?:{_LEADING_SOFT})?(?:{alt})\b[\s,.:;!\-]*",
        re.IGNORECASE,
    )
    m = start.match(cleaned)
    if m:
        return True, cleaned[m.end():].strip()
    end = re.compile(
        rf"[\s,.:;!\-]+(?:{alt})[\s,.:;!?]*$",
        re.IGNORECASE,
    )
    m = end.search(cleaned)
    if m:
        return True, cleaned[: m.start()].strip()
    return False, cleaned
