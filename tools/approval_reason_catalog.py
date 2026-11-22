"""Display-only catalog that localizes command-approval reasons.

The detectors in ``tools/approval_detection.py`` name each finding by its English description, and
that description is also the approval/consent key a session or always grant is stored under, so it
must never be translated or renamed. This module maps a fixed set of those keys to stable slugs under
``gateway.exec_approval.reasons.*`` so a chat surface can show the reason in the routed profile's
language. Only keys listed here are ever looked up: a description is never parsed and a locale key
is never built from free text. Anything unmapped, or a slug the active language's own catalog (packs,
user overlay, bundled) lacks, keeps the redacted English text it arrived with; the English catalog is
never consulted for a reason, so an English overlay cannot replace that text in another language.
"""
from __future__ import annotations

from typing import Callable, Iterable, Mapping

from agent.i18n import DEFAULT_LANGUAGE, get_language, lookup

REASON_KEY_PREFIX = "gateway.exec_approval.reasons."
_JOINER_KEY = "gateway.exec_approval.reason_joiner"

# Detector key (the English description, verbatim) -> catalog slug. Literal strings keep this module
# free of the detector import; a test pins every key to a finding the detectors really produce. A
# vertical slice: extend by adding a row plus ``gateway.exec_approval.reasons.<slug>`` in every locale.
REASON_SLUGS: Mapping[str, str] = {
    "delete in root path": "delete_in_root_path",
    "recursive delete": "recursive_delete",
    "world/other-writable permissions": "world_writable_permissions",
    "SQL DROP": "sql_drop",
    "pipe remote content to shell": "pipe_remote_to_shell",
    "git reset --hard (destroys uncommitted changes)": "git_reset_hard",
    "upload a secret or credential file via curl/wget (possible exfiltration)": "credential_upload",
    "invisible or bidirectional Unicode control character (possible obfuscation)": "invisible_unicode",
    "command too complex to scan fully; review before running": "scan_incomplete",
}
_KNOWN_SLUGS = frozenset(REASON_SLUGS.values())


def reason_items(findings: Iterable[tuple[str, str]], redact: Callable[[str], str]) -> list[dict]:
    """``[{"slug", "text"}]`` for ordered ``(pattern_key, description)`` findings: the catalog slug
    (``None`` when unmapped) and the redacted English text that is shown when no translation applies."""
    return [{"slug": REASON_SLUGS.get(key), "text": redact(text)} for key, text in findings]


def render_reasons(items: Iterable[Mapping], redact: Callable[[str], str], *,
                   lang: str | None = None, home: str | None = None) -> str:
    """One display string for *items* in the active profile's language, in the given order.

    ``lang``/``home`` name the chat's profile when the caller is not running in it (a delegated
    child under another profile raising an approval in its parent's chat).

    ``home`` without ``lang`` is refused (see :func:`agent.i18n.lookup`).

    The language is resolved once so every item and the joiner agree. An unknown slug, or a slug the
    active language does not translate itself, falls back to that item's own text. The joined string
    goes through the surface's display redaction as a whole, like the legacy ``description``: catalog
    text (a user overlay, a plugin pack) is display input from outside the core too."""
    if home is not None and not lang:
        raise ValueError("render_reasons(home=...) needs lang=: the active language belongs to the bound profile")
    lang = lang or get_language()
    parts = []
    for item in items:
        slug = item.get("slug")
        key = REASON_KEY_PREFIX + slug if slug in _KNOWN_SLUGS else None
        localized = lookup(key, lang, home) if key else None
        parts.append(localized or str(item.get("text") or ""))
    # Not t(): on a catalog that predates the key (an older HERMES_BUNDLED_LOCALES) it returns the raw key.
    joiner = lookup(_JOINER_KEY, lang, home) or lookup(_JOINER_KEY, DEFAULT_LANGUAGE, home) or "; "
    return redact(joiner.join(p for p in parts if p))


__all__ = ["REASON_KEY_PREFIX", "REASON_SLUGS", "reason_items", "render_reasons"]
