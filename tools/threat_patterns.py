"""Shared threat-pattern library for context window security scanning.

This module is the single source of truth for prompt-injection / promptware /
exfiltration patterns used across the context-assembly scanners
(``agent/prompt_builder.py``, ``tools/memory_tool.py``) and the tool-result
delimiter system in ``agent/tool_dispatch_helpers.py``.

Pattern philosophy
------------------
Patterns are organized by ATTACK CLASS, not by source file.  Each pattern
is a ``(regex, pattern_id, scope)`` tuple, where ``scope`` controls which
scanners use it:

- ``"all"``  — applied everywhere (classic prompt injection, exfiltration)
- ``"context"`` — applied to context files + memory + tool results
  (promptware / C2 / behavioral hijack; broader detection)
- ``"strict"`` — applied to memory writes + skill installs only
  (aggressive checks acceptable for user-curated content but too noisy
  for tool results)

The split exists because tool results contain web pages, GitHub issues,
and MCP responses — content the user did not author — and we want broad
detection there, but blocking is reserved for paths where the user can
intervene (memory writes, skill installs).

Pattern anchoring
-----------------
New patterns anchor on **C2-specific vocabulary or unambiguous attack
behavior**, NOT on bossy English.  Phrases like "you are obligated to"
or "you must" alone are too common in legitimate instruction-writing
(see AGENTS.md, CLAUDE.md, etc.) to flag.  See the pattern comments for
the rationale on borderline cases.

Multi-word bypass
-----------------
Patterns use bounded ``(?:\\w+\\s+){0,8}`` filler between key tokens to prevent
attackers from inserting a handful of words (e.g. "ignore all prior
instructions" instead of "ignore all instructions") without allowing unbounded
regex backtracking. This mirrors the fix applied to ``skills_guard.py`` in
commit 4ea29978.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Tuple

# Hard cap on text scanned with regexes.  Context/tool-result strings can be
# arbitrarily large, and the scanners are advisory guards rather than archival
# search; bounding input keeps worst-case runtime predictable while preserving
# detections near the beginning of injected content.
MAX_SCAN_CHARS = 65_536

# Bounded filler used between key attack words.  Earlier patterns used
# ``(?:\w+\s+)*`` which is ambiguous and can backtrack heavily on adversarial
# near-misses.  Eight filler words is enough for the intended obfuscation
# bypasses without introducing unbounded repetition.
_FILLER = r"(?:\w+\s+){0,8}"

# Negation prefixes that flip a would-be injection pattern from a positive
# instruction ("pretend to be X") into a benign negative guidance ("don't
# pretend to be X").  When one of these precedes a match within a small
# window, the match is treated as advisory-only and suppressed at the
# context-file layer (see ``agent/prompt_builder.py``).
#
# The set is intentionally narrow: it covers the contractions and long forms
# that occur in natural prose.  Imperative negation ("do not") and contraction
# ("don't") are the common case; "never" / "avoid" / "without" cover stylistic
# variants.  We do not attempt to handle languages other than English or
# roundabout phrasings — the failure mode of missing one is a false positive,
# not a false negative (the underlying pattern still applies to the rest of
# the file).
_NEGATION_PREFIXES = (
    "don't",
    "do not",
    "dont",        # strip-apostrophe typo / plain-ASCII authoring
    "never",
    "avoid",
    "without",
    "not",         # "you are not to pretend …"
)

# Compiled once: a single regex that matches any negation prefix at a word
# boundary, used by the negation-aware scanner.  Case-insensitive.
_NEGATION_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in _NEGATION_PREFIXES) + r")\b",
    re.IGNORECASE,
)

# Maximum number of filler tokens allowed between a negation prefix and the
# pattern match.  Five tokens covers "don't ever pretend to be" or
# "never under any circumstances pretend to be" without allowing an attacker
# to push the negation arbitrarily far away from the verb they're trying to
# cloak.  Slightly tighter than ``_FILLER``'s 8 because a real negated threat
# almost always binds within a single clause, while obfuscation attempts to
# bury it.
_NEGATION_WINDOW_TOKENS = 5

# Patterns whose context-scope matches are subject to negation suppression.
# These are the role / identity / restriction patterns that read as natural
# guidance when negated ("don't pretend to be", "never answer without
# filters") and as attacks when asserted positively.  C2 / exfil / classic
# injection patterns are NOT in this set — "never ignore previous
# instructions" does not turn an injection benign.
_NEGATION_AWARE_PIDS = frozenset({
    "role_pretend",
    "role_hijack",
    "remove_filters",
    "fake_update",
    "bypass_restrictions",
    "deception_hide",
})

# Each entry: (regex, pattern_id, scope)
# scope ∈ {"all", "context", "strict"}
_PATTERNS: List[Tuple[str, str, str]] = [
    # ── Classic prompt injection (applies everywhere) ────────────────
    (rf'ignore\s+{_FILLER}(previous|all|above|prior)\s+{_FILLER}instructions', "prompt_injection", "all"),
    (r'system\s+prompt\s+override', "sys_prompt_override", "all"),
    (rf'disregard\s+{_FILLER}(your|all|any)\s+{_FILLER}(instructions|rules|guidelines)', "disregard_rules", "all"),
    (rf'act\s+as\s+(if|though)\s+{_FILLER}you\s+{_FILLER}(have\s+no|don\'t\s+have)\s+{_FILLER}(restrictions|limits|rules)', "bypass_restrictions", "all"),
    (r'<!--[^>]{0,512}(?:ignore|override|system|secret|hidden)[^>]{0,512}-->', "html_comment_injection", "all"),
    (r'<\s*div\s+style\s*=\s*["\'][^>]{0,2048}display\s*:\s*none', "hidden_div", "all"),
    (r'translate\s+[^\n]{0,512}\s+into\s+[^\n]{0,512}\s+and\s+(execute|run|eval)', "translate_execute", "all"),
    (rf'do\s+not\s+{_FILLER}tell\s+{_FILLER}the\s+user', "deception_hide", "all"),

    # ── Role-play / identity hijack (context + strict; common attack
    #    surface in scraped web content and poisoned context files) ──
    (rf'you\s+are\s+{_FILLER}now\s+(?:a|an|the)\s+', "role_hijack", "context"),
    (rf'pretend\s+{_FILLER}(you\s+are|to\s+be)\s+', "role_pretend", "context"),
    (rf'output\s+{_FILLER}(system|initial)\s+prompt', "leak_system_prompt", "context"),
    (rf'(respond|answer|reply)\s+without\s+{_FILLER}(restrictions|limitations|filters|safety)', "remove_filters", "context"),
    (rf'you\s+have\s+been\s+{_FILLER}(updated|upgraded|patched)\s+to', "fake_update", "context"),
    # "name yourself X" is a Brainworm-specific tell — identity override
    # via spec instead of jailbreak.  Anchored on the verb pair so it
    # doesn't match "name your variables" etc.
    (r'\bname\s+yourself\s+\w+', "identity_override", "context"),

    # ── C2 / Brainworm-style promptware (context scope) ──────────────
    # These anchor on C2-specific vocabulary.  "register as a node" appears
    # in legitimate distributed-systems docs, but in combination with the
    # other patterns the signal is strong; we WARN, not block, so a security
    # researcher reading the Brainworm post in a webpage doesn't break their
    # session.
    (r'register\s+(as\s+)?a?\s*node', "c2_node_registration", "context"),
    (r'(heartbeat|beacon|check[\s\-]?in)\s+(to|with)\s+', "c2_heartbeat", "context"),
    (r'pull\s+(down\s+)?(?:new\s+)?task(?:ing|s)?\b', "c2_task_pull", "context"),
    (r'connect\s+to\s+the\s+network\b', "c2_network_connect", "context"),
    # Verb-anchored "you must register/connect/report/beacon" — the verbs
    # are C2-specific so this avoids the broader "you must X" false positive.
    (r'you\s+must\s+(?:\w+\s+){0,3}(register|connect|report|beacon)\b', "forced_action", "context"),
    # Anti-forensic instructions ("never write to disk", "one-liners only")
    # — extremely unusual in legitimate content; near-zero false positive.
    (r'only\s+use\s+one[\s\-]?liners?\b', "anti_forensic_oneliner", "context"),
    (rf'never\s+{_FILLER}(?:create|write)\s+{_FILLER}(?:script|file)\s+{_FILLER}disk', "anti_forensic_disk", "context"),
    # Environment-variable unsetting targeting known agent runtimes —
    # this is pure attack behavior (Brainworm sub-session bypass).
    (r'unset\s+\w*(?:CLAUDE|CODEX|HERMES|AGENT|OPENAI|ANTHROPIC)\w*', "env_var_unset_agent", "context"),

    # ── Known C2 / red-team framework names (near-zero false positive
    #    outside security research; warn-only by default) ─────────────
    # NOTE: do not add common English words here. Every token must be a
    # distinctive offensive-security tool brand, otherwise legitimate
    # AGENTS.md / SOUL.md content false-positives and the whole file is
    # blocked. "praxis" was removed for exactly this reason — it's a common
    # word and a legitimate agent name (Greek for practice/action), not a
    # C2-specific tell like the brands below.
    (r'\b(?:cobalt\s*strike|sliver|havoc|mythic|metasploit|brainworm)\b', "known_c2_framework", "context"),
    (r'\bc2\s+(?:server|channel|infrastructure|beacon)\b', "c2_explicit", "context"),
    (r'\bcommand\s+and\s+control\b', "c2_explicit_long", "context"),

    # ── Exfiltration via curl/wget/cat with secrets (applies everywhere) ──
    (r'curl\s+[^\n]{0,2048}\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "exfil_curl", "all"),
    (r'wget\s+[^\n]{0,2048}\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "exfil_wget", "all"),
    (r'cat\s+[^\n]{0,2048}(\.env|credentials|\.netrc|\.pgpass|\.npmrc|\.pypirc)', "read_secrets", "all"),
    (r'(send|post|upload|transmit)\s+[^\n]{0,2048}\s+(to|at)\s+https?://', "send_to_url", "strict"),
    (rf'(include|output|print|share)\s+{_FILLER}(conversation|chat\s+history|previous\s+messages|full\s+context|entire\s+context)', "context_exfil", "strict"),

    # ── Persistence / SSH backdoor (strict scope — memory + skills) ──
    (r'authorized_keys', "ssh_backdoor", "strict"),
    (r'\$HOME/\.ssh|\~/\.ssh', "ssh_access", "strict"),
    (r'\$HOME/\.hermes/\.env|\~/\.hermes/\.env', "hermes_env", "strict"),
    (r'(update|modify|edit|write|change|append|add\s+to)\s+[^\n]{0,2048}(?:AGENTS\.md|CLAUDE\.md|\.cursorrules|\.clinerules)', "agent_config_mod", "strict"),
    (r'(update|modify|edit|write|change|append|add\s+to)\s+[^\n]{0,2048}\.hermes/(config\.yaml|SOUL\.md)', "hermes_config_mod", "strict"),

    # ── Hardcoded secrets ────────────────────────────────────────────
    (r'(?:api[_-]?key|token|secret|password)\s*[=:]\s*["\'][A-Za-z0-9+/=_-]{20,}', "hardcoded_secret", "strict"),
]

# Invisible / bidirectional unicode characters used in injection attacks.
# Aligned with skills_guard.py INVISIBLE_CHARS — directional isolates
# (U+2066-U+2069) and invisible math operators (U+2062-U+2064) are real
# attack tools.
INVISIBLE_CHARS = frozenset({
    '\u200b',  # zero-width space
    '\u200c',  # zero-width non-joiner
    '\u200d',  # zero-width joiner
    '\u2060',  # word joiner
    '\u2062',  # invisible times
    '\u2063',  # invisible separator
    '\u2064',  # invisible plus
    '\ufeff',  # zero-width no-break space (BOM)
    '\u202a',  # left-to-right embedding
    '\u202b',  # right-to-left embedding
    '\u202c',  # pop directional formatting
    '\u202d',  # left-to-right override
    '\u202e',  # right-to-left override
    '\u2066',  # left-to-right isolate
    '\u2067',  # right-to-left isolate
    '\u2068',  # first strong isolate
    '\u2069',  # pop directional isolate
})


# Compiled pattern sets, indexed by scope.  Compiled once at import time;
# scan_for_threats() looks them up.
_COMPILED: dict[str, List[Tuple[re.Pattern, str]]] = {}


def _compile() -> None:
    """Compile pattern sets for each scope (all / context / strict).

    A pattern with scope="all" lands in every set.  A pattern with
    scope="context" lands in context + strict (context implies the
    strict scanners want it too).  Scope="strict" lands in strict only.
    """
    global _COMPILED
    if _COMPILED:
        return

    all_patterns: List[Tuple[re.Pattern, str]] = []
    context_patterns: List[Tuple[re.Pattern, str]] = []
    strict_patterns: List[Tuple[re.Pattern, str]] = []

    for pattern, pid, scope in _PATTERNS:
        compiled = re.compile(pattern, re.IGNORECASE)
        entry = (compiled, pid)
        if scope == "all":
            all_patterns.append(entry)
            context_patterns.append(entry)
            strict_patterns.append(entry)
        elif scope == "context":
            context_patterns.append(entry)
            strict_patterns.append(entry)
        elif scope == "strict":
            strict_patterns.append(entry)
        else:
            raise ValueError(f"threat_patterns: unknown scope {scope!r} for pattern {pid!r}")

    _COMPILED = {
        "all": all_patterns,
        "context": context_patterns,
        "strict": strict_patterns,
    }


_compile()


def scan_for_threats(content: str, scope: str = "context") -> List[str]:
    """Return a list of matched pattern IDs in ``content`` at the given scope.

    ``scope`` selects which pattern set to apply:

    - ``"all"`` (narrow): classic injection + exfil only — minimal false
      positives, suitable for any text.
    - ``"context"`` (default): adds promptware / C2 / role-play patterns —
      suitable for context files, memory entries, and tool results.
    - ``"strict"`` (broad): adds persistence / SSH backdoor / exfil-URL
      patterns — appropriate for user-mediated writes (memory tool,
      skills install) where false positives can be resolved interactively.

    Also checks for invisible unicode characters (returned as
    ``"invisible_unicode_U+XXXX"`` so the caller can surface the offending
    codepoint in a log line).

    Negation awareness: this entry point reports ALL pattern matches,
    including ones that read as benign because a negation prefix
    ("don't pretend to be") precedes them.  The strict-scope callers
    (memory writes, skill installs) want those hits surfaced so an
    operator can confirm the content.  The context-file pipeline uses
    :func:`scan_for_threats_with_negation` instead, which suppresses
    negated matches at the line level.
    """
    if not content:
        return []

    findings: List[str] = []

    content = content[:MAX_SCAN_CHARS]

    # Invisible unicode — single pass through the content set, not 17
    # ``in`` lookups.  Run this on the RAW content before NFKC normalisation,
    # since normalisation can strip some of these codepoints.
    char_set = set(content)
    invisible_hits = char_set & INVISIBLE_CHARS
    for ch in invisible_hits:
        findings.append(f"invisible_unicode_U+{ord(ch):04X}")

    # Normalise to NFKC so full-width / compatibility Unicode variants
    # (e.g. ｃａｔ → cat, Ａ → A) are folded to their ASCII counterparts before
    # the regex engine sees them.  This prevents homograph substitution from
    # bypassing keyword checks (e.g. ``ｃａｔ ~/.hermes/.env``).  NOTE: this
    # does NOT defend against cross-script confusables (Cyrillic ``а`` U+0430),
    # which NFKC leaves untouched — that needs a TR#39 confusable database.
    normalised = unicodedata.normalize("NFKC", content)

    # Threat patterns
    patterns = _COMPILED.get(scope)
    if patterns is None:
        raise ValueError(f"scan_for_threats: unknown scope {scope!r}")
    for compiled, pid in patterns:
        if compiled.search(normalised):
            findings.append(pid)

    return findings


# Result tuple for the negation-aware scanner.  ``span`` is the half-open
# [start, end) character offsets of the pattern match in the NORMALISED text
# (offsets in the raw text would shift after NFKC folding).  ``negated`` is
# True if a negation prefix precedes the match within the configured window,
# in which case the caller may downgrade the hit from "block" to "warn" or
# redact only the offending line.
ThreatMatch = Tuple[str, Tuple[int, int], bool]


def scan_for_threats_with_negation(
    content: str, scope: str = "context"
) -> List[ThreatMatch]:
    """Like :func:`scan_for_threats` but returns per-match spans and
    whether each match is preceded by a negation prefix.

    Returns one ``ThreatMatch`` per pattern that fires (i.e. a pattern
    fires at most once even if it matches multiple times — the span points
    at the first match).  Invisible-unicode hits are reported with a span
    of ``(-1, -1)`` and ``negated=False``; they are never subject to
    negation suppression.

    The negation check looks for one of :data:`_NEGATION_PREFIXES` ending
    within ``_NEGATION_WINDOW_TOKENS`` tokens before the match start.  Only
    patterns in :data:`_NEGATION_AWARE_PIDS` are subject to suppression;
    classic injection, C2, and exfil patterns always report ``negated=False``
    so callers can keep blocking on them.
    """
    if not content:
        return []

    matches: List[ThreatMatch] = []

    content = content[:MAX_SCAN_CHARS]
    char_set = set(content)
    invisible_hits = char_set & INVISIBLE_CHARS
    for ch in invisible_hits:
        matches.append((
            f"invisible_unicode_U+{ord(ch):04X}",
            (-1, -1),
            False,
        ))

    normalised = unicodedata.normalize("NFKC", content)
    patterns = _COMPILED.get(scope)
    if patterns is None:
        raise ValueError(
            f"scan_for_threats_with_negation: unknown scope {scope!r}"
        )
    for compiled, pid in patterns:
        m = compiled.search(normalised)
        if m is None:
            continue
        negated = False
        if pid in _NEGATION_AWARE_PIDS:
            negated = _is_negated(normalised, m.start())
        matches.append((pid, (m.start(), m.end()), negated))

    return matches


def _is_negated(text: str, match_start: int) -> bool:
    """Return True if a negation prefix precedes ``match_start`` within
    ``_NEGATION_WINDOW_TOKENS`` tokens.

    The lookbehind is bounded by scanning at most ``_NEGATION_WINDOW_TOKENS
    + 1`` whitespace-separated tokens backwards from ``match_start`` and
    checking whether any of them is a negation prefix.  Bounded scanning
    keeps worst-case runtime constant per match regardless of input size.
    """
    # Walk backwards from the match start collecting tokens.  We split the
    # window into roughly token-sized chunks using ``split()`` on a small
    # slice that ends at match_start; this is cheaper than a regex finditer
    # over the whole prefix.  The slice is bounded by
    # ``_NEGATION_WINDOW_TOKENS`` * (max reasonable token length + space)
    # ≈ a few hundred chars at most, which stays well within input-size
    # bounds even for very long files.
    max_window_chars = (_NEGATION_WINDOW_TOKENS + 1) * 32
    window_start = max(0, match_start - max_window_chars)
    window = text[window_start:match_start]
    tokens = window.split()
    # Keep at most the last N tokens leading up to the match.
    tail = tokens[-_NEGATION_WINDOW_TOKENS:] if tokens else []
    for tok in tail:
        # ``split()`` strips punctuation attached to tokens, so "don't" comes
        # through cleanly.  But authoring like "Don't," or "do-not" can land
        # as "Don't" or "do-not" — both handled below.
        lower = tok.lower().strip(".,;:!?")
        if lower in _NEGATION_PREFIXES:
            return True
        # Handle hyphenated authoring ("do-not pretend", "don-t pretend").
        # Normalize hyphens to spaces and re-check the whole token sequence:
        # we only need to know if ANY token is a bare negation prefix.
        if "-" in tok:
            for piece in tok.lower().split("-"):
                if piece.strip(".,;:!?") in _NEGATION_PREFIXES:
                    return True
    return False


def first_threat_message(content: str, scope: str = "strict") -> Optional[str]:
    """Return a human-readable error string for the first threat found, or None.

    Convenience wrapper used by paths that block on the first hit
    (memory tool writes, skills install) where the caller just needs a
    yes/no + a message.
    """
    findings = scan_for_threats(content, scope=scope)
    if not findings:
        return None
    pid = findings[0]
    if pid.startswith("invisible_unicode_"):
        codepoint = pid.replace("invisible_unicode_", "")
        return f"Blocked: content contains invisible unicode character {codepoint} (possible injection)."
    return (
        f"Blocked: content matches threat pattern '{pid}'. "
        f"Content is injected into the system prompt and must not contain "
        f"injection or exfiltration payloads."
    )


__all__ = [
    "INVISIBLE_CHARS",
    "MAX_SCAN_CHARS",
    "ThreatMatch",
    "first_threat_message",
    "scan_for_threats",
    "scan_for_threats_with_negation",
]
