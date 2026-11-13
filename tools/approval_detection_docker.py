"""Docker lifecycle prose exemption: strict single-command shell subset only.

Anything other than a known text carrier, with the hit entirely within a quoted
argument, retains the base scanner. Redirection, operators, expansions and
unrecognized shell syntax never qualify.
"""

_DOCKER_LIFECYCLE_DESCRIPTIONS = frozenset({
    "docker compose restart/stop/kill/down (container lifecycle)",
    "docker restart/stop/kill (container lifecycle)",
})


def _simple_words(command: str) -> list[tuple[str, str]] | None:
    """Tokenize an intentionally narrow shell subset, preserving quote masks."""
    if not command.isascii():
        return None
    words = []
    word = []
    mask = []
    quote = None
    active = False
    for ch in command:
        if ch in "\\$`\n\r\x00":
            return None
        if quote:
            if ch == quote:
                quote = None
            else:
                word.append(ch)
                mask.append(" ")
        elif ch in "'\"":
            quote = ch
            active = True
        elif ch in " \t":
            if active:
                words.append(("".join(word), "".join(mask)))
                word, mask, active = [], [], False
        elif ch in ";&|(){}<>#*?[]~":
            return None
        else:
            word.append(ch)
            mask.append(ch)
            active = True
    if quote:
        return None
    if active:
        words.append(("".join(word), "".join(mask)))
    return words


def _inert_carrier(words: list[str]) -> bool:
    if not words:
        return False
    if words[0] == "echo":
        return True
    if words[0] == "printf":
        return not any(word.startswith("-v") for word in words[1:])
    if len(words) == 4 and words[:3] == ["git", "commit", "-m"]:
        return True
    if len(words) == 5 and words[:3] == ["hermes", "kanban", "comment"]:
        return True
    return len(words) == 6 and words[:3] == ["hermes", "kanban", "create"] and words[4] == "--body"


def _docker_lifecycle_prose_exempt(command: str, patterns) -> bool:
    words = _simple_words(command)
    if not words or not _inert_carrier([word for word, _ in words]):
        return False
    # Require each lifecycle match to occur fully within one quoted argument;
    # no unquoted lifecycle match may remain anywhere in the simple command.
    all_text = " ".join(word for word, _ in words).lower()
    all_mask = " ".join(mask for _, mask in words)
    matches = [match for pattern in patterns for match in pattern.finditer(all_text)]
    quoted_matches = sum(
        1 for word, mask in words[1:] for pattern in patterns
        for match in pattern.finditer(word.lower())
        if not mask[match.start():match.end()].strip()
    )
    return bool(matches) and len(matches) == quoted_matches and all(
        not all_mask[m.start():m.end()].strip() for m in matches
    )
