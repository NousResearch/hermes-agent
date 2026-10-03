"""Shared threat-pattern library (prompt injection / promptware / exfiltration) for
``agent/prompt_builder.py``, ``tools/memory_tool.py`` and ``agent/tool_dispatch_helpers.py``.
Each pattern is ``(regex, pattern_id, scope)``; scope is cumulative: ``"all"`` everywhere,
``"context"`` adds promptware / C2 / role hijack for context files, memory and tool results
(warn-level: that content is not user-authored), ``"strict"`` adds aggressive checks only for
user-mediated writes (memory, skill installs) where a block is resolvable. New patterns must
anchor on C2 vocabulary or unambiguous attack behavior, NOT bossy English ("you must" is common
in legitimate AGENTS.md); filler between tokens is the bounded ``_FILLER``."""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Tuple

# Hard cap on scanned text: scanners are advisory, so bound worst-case runtime.
MAX_SCAN_CHARS = 65_536
# Bounded filler between key attack words (unbounded ``(?:\w+\s+)*`` backtracks badly).
_FILLER = r"(?:\w+\s+){0,8}"
# Env var reference ending in a secret-ish suffix (see exfil comment below).
_SECRET_VAR = r"\$\{?\w*(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)S?\b"
# Verb prefix for "modify agent config" patterns.
_MODIFY = r"(update|modify|edit|write|change|append|add\s+to)\s+[^\n]{0,2048}"

# A phrase wrapped in quotation marks is a citation/description, not an
# active directive.  Defensive docs (security notes in SOUL.md, AGENTS.md)
# quote attack strings verbatim — e.g. ``"Ignore your previous
# instructions" → these have no effect`` — and blocking on the quoted
# occurrence would quarantine the whole identity file.  Guard the
# imperative injection patterns with a negative lookbehind so only
# unquoted occurrences fire (#90635) — at the non-strict scopes only;
# scope="strict" compiles the same patterns with this exemption stripped
# (#111334, see the intent-guard comment below).  An attacker prefixing the
# directive with words still matches (the quote is no longer adjacent to the
# verb).
_QUOTED = r'(?<!["\'‘’“”«»])'

# Intent-context guard for ``prompt_injection`` (#92644). The pattern matches
# sentences that *describe* the attack to teach the agent to recognise it —
# constitutional SOUL.md / AGENTS.md doctrine like "when you encounter prompt
# injection — instructions telling you to ignore previous instructions …".
# After a match, the ~60 chars immediately BEFORE it are checked for a
# descriptive cue; a cue means the phrase is quoted description, not a
# directive, and the finding is skipped. Two scoping rules keep the guard
# honest: the window is truncated at the last sentence terminator inside it
# (a cue in the PREVIOUS sentence must not excuse a bare directive), and cues
# are word-boundary-anchored ("such assumption" ≠ "such as", "retold to" ≠
# "told to"). Python ``re`` has no variable-length lookbehind, so this is a
# post-match prefix check in the scan loop, and it applies ONLY to the IDs in
# ``_INTENT_GUARDED_IDS`` — every other pattern fires regardless of context.
# Scope limit (#111334 cross-vendor review): this guard and the ``_QUOTED``
# citation exemption apply ONLY at the non-strict scopes, where doctrine
# actually loads (context files / tool results scan "context"; "all" is the
# narrow file-content set with no user-prompt callers). At scope="strict" —
# raw user-authored writes (memory tool, install paths) — scanning is
# unconditional: a cue-prefixed or quoted directive there is a real payload
# and must fire.
# Residual (issue-accepted, non-strict only): an attacker who prefixes a real
# directive with a same-sentence cue phrase evades the guard.
_INTENT_GUARDED_IDS = {"prompt_injection"}
_CUE_WINDOW = 60
# Word-boundary alternation so cues cannot fire as substrings of longer words.
_DESCRIPTIVE_CUES_RE = re.compile(
    r"\b(?:when you encounter|telling you to|told to|describing|"
    r"defending against|examples of|attack patterns like|such as)\b",
    re.IGNORECASE,
)
# Sentence terminators that close the cue window (#92644 review): doctrine
# cues sit in the same sentence as the described phrase by construction.
# ``!`` and ``?`` are unconditional terminators. A ``.`` terminates ONLY when
# followed by whitespace/end-of-window AND its preceding word-char run is
# >=2 chars and not a known abbreviation: ``e.g.``/``i.e.``/decimals (``1.2``)/
# ``U.S.`` are excluded by the >=2 rule, ``etc.``/``vs.``/``Dr.`` by the set.
# A single ``\n`` is a SOFT WRAP, not a terminator — hard-wrapped (72-80 col)
# doctrine prose is ONE sentence, and cutting at line breaks orphaned the cue
# and re-blocked the very defense text the guard exists to protect. Attacker
# cross-line cue-prefixing is already the documented accepted residual of
# this guard, so nothing new opens. Only a blank line (``\n\n``, a paragraph
# break) terminates a sentence.
_SENTENCE_END_CHARS = ".!?"
_ABBREVIATIONS = frozenset({
    "etc", "vs", "cf", "approx", "dr", "mr", "mrs", "ms", "st", "jr", "sr",
    "inc", "ltd", "fig", "no", "vol", "al",
})
_WORD_RUN_RE = re.compile(r"\w*$")
# Closing quotes/brackets (and space) around a sentence's terminator
# (``…such as "npm". Ignore …`` and the US-typography ``…such as these." Ignore …``):
# stripped before the word-run check (otherwise the run sees ``"`` (len < 2)
# and denies a real terminator) and skipped after a period in
# ``_is_sentence_dot`` — either way the leak this prevents is the previous
# sentence's cue reaching into the directive's window (#111334, #121713).
# Single curly quotes and guillemets belong here for the same reason they
# belong in ``_QUOTED``: NFKC leaves them intact.
_TRAILING_CLOSERS = "\"'‘’“”«»)]} "


def _is_sentence_dot(prefix: str, i: int) -> bool:
    """Whether ``prefix[i] == '.'`` closes a sentence (see terminator rules above)."""
    # US typography closes quotes/brackets AFTER the period (``tactics."``):
    # skip such closers before the whitespace requirement, mirroring the
    # closer-before-period strip on the word-run check below — otherwise the
    # terminator is denied and the previous sentence's cue leaks in (#121713).
    j = i + 1
    while j < len(prefix) and prefix[j] in _TRAILING_CLOSERS and not prefix[j].isspace():
        j += 1
    if j < len(prefix) and not prefix[j].isspace():
        return False  # mid-token dot (file.txt, U.S.): not a terminator
    # Residual (documented): a sentence ending in a single-letter word
    # (``Option A.``) fails the >=2 run rule and is not a terminator here —
    # an abbreviation guard would instead break ``U.S.``/``e.g.`` handling.
    run = _WORD_RUN_RE.search(prefix[:i].rstrip(_TRAILING_CLOSERS))
    run = run.group() if run else ""
    return len(run) >= 2 and run.lower() not in _ABBREVIATIONS


def _last_terminator(prefix: str) -> int:
    """Slice index just after the last TRUE sentence terminator in ``prefix``
    (the cue window); 0 when the whole window is one sentence."""
    cut = 0
    for i, ch in enumerate(prefix):
        if ch in _SENTENCE_END_CHARS:
            if ch != "." or _is_sentence_dot(prefix, i):
                cut = i + 1
        elif ch == "\n" and prefix.startswith("\n", i + 1):
            cut = i + 2  # blank line = paragraph break
    return cut


def _is_descriptive(normalised: str, match_start: int) -> bool:
    prefix = normalised[max(0, match_start - _CUE_WINDOW):match_start]
    # Same-sentence only: drop everything up to the last terminator in the window.
    return bool(_DESCRIPTIVE_CUES_RE.search(prefix[_last_terminator(prefix):]))

# Each entry: (regex, pattern_id, scope); scope ∈ {"all", "context", "strict"}
_PATTERNS: List[Tuple[str, str, str]] = [
    # ── Classic prompt injection (applies everywhere) ────────────────
    (rf'{_QUOTED}ignore\s+{_FILLER}(previous|all|above|prior)\s+{_FILLER}instructions', "prompt_injection", "all"),
    (r'system\s+prompt\s+override', "sys_prompt_override", "all"),
    (rf'{_QUOTED}disregard\s+{_FILLER}(your|all|any)\s+{_FILLER}(instructions|rules|guidelines)', "disregard_rules", "all"),
    (rf'{_QUOTED}act\s+as\s+(if|though)\s+{_FILLER}you\s+{_FILLER}(have\s+no|don\'t\s+have)\s+{_FILLER}(restrictions|limits|rules)', "bypass_restrictions", "all"),
    (r'<!--[^>]{0,512}(?:ignore|override|system|secret|hidden)[^>]{0,512}-->', "html_comment_injection", "all"),
    (r'<\s*div\s+style\s*=\s*["\'][^>]{0,2048}display\s*:\s*none', "hidden_div", "all"),
    (
        r"translate\s+[^\n]{0,512}\s+into\s+\w+(?:[\s-]+\w+){0,2}\s+and\s+(execute|run|eval)\b",
        "translate_execute",
        "all",
    ),
    (rf'{_QUOTED}do\s+not\s+{_FILLER}tell\s+{_FILLER}the\s+user', "deception_hide", "all"),

    # ── Role-play / identity hijack (scraped web content, poisoned context files) ──
    (rf'you\s+are\s+{_FILLER}now\s+(?:a|an|the)\s+', "role_hijack", "context"),
    (rf'pretend\s+{_FILLER}(you\s+are|to\s+be)\s+', "role_pretend", "context"),
    (rf'output\s+{_FILLER}(system|initial)\s+prompt', "leak_system_prompt", "context"),
    (rf'(respond|answer|reply)\s+without\s+{_FILLER}(restrictions|limitations|filters|safety)', "remove_filters", "context"),
    (rf'you\s+have\s+been\s+{_FILLER}(updated|upgraded|patched)\s+to', "fake_update", "context"),
    # Brainworm tell: identity override via spec. Verb pair anchored so "name your variables" is safe.
    (r'\bname\s+yourself\s+\w+', "identity_override", "context"),

    # ── C2 / Brainworm-style promptware (context scope) ──────────────
    # Anchored on C2 vocabulary. "register as a node" appears in legitimate distributed-systems
    # docs, so this is WARN not block: a researcher reading the Brainworm post keeps their session.
    (r'register\s+(as\s+)?a?\s*node', "c2_node_registration", "context"),
    (r'(heartbeat|beacon|check[\s\-]?in)\s+(to|with)\s+', "c2_heartbeat", "context"),
    (r'pull\s+(down\s+)?(?:new\s+)?task(?:ing|s)?\b', "c2_task_pull", "context"),
    (r'connect\s+to\s+the\s+network\b', "c2_network_connect", "context"),
    # C2-specific verbs avoid the broader "you must X" false positive.
    (r'you\s+must\s+(?:\w+\s+){0,3}(register|connect|report|beacon)\b', "forced_action", "context"),
    # Anti-forensic instructions: near-zero false positive in legitimate content.
    (r'only\s+use\s+one[\s\-]?liners?\b', "anti_forensic_oneliner", "context"),
    (rf'never\s+{_FILLER}(?:create|write)\s+{_FILLER}(?:script|file)\s+{_FILLER}disk', "anti_forensic_disk", "context"),
    # Unsetting agent-runtime env vars is pure attack behavior (Brainworm sub-session bypass).
    (r'unset\s+\w*(?:CLAUDE|CODEX|HERMES|AGENT|OPENAI|ANTHROPIC)\w*', "env_var_unset_agent", "context"),

    # ── Known C2 / red-team framework names (warn-only) ─────────────
    # Every token must be a distinctive offensive-security brand: a common English word here
    # (e.g. "praxis", also a legitimate agent name) false-positives whole AGENTS.md / SOUL.md files.
    (r'\b(?:cobalt\s*strike|sliver|havoc|mythic|metasploit|brainworm)\b', "known_c2_framework", "context"),
    (r'\bc2\s+(?:server|channel|infrastructure|beacon)\b', "c2_explicit", "context"),
    (r'\bcommand\s+and\s+control\b', "c2_explicit_long", "context"),

    # ── Exfiltration via curl/wget/cat with secrets (applies everywhere) ──
    # The var name ends with \b so benign names containing KEY/TOKEN as substrings
    # ($TRILLIUM_ETAPI_URL) pass. API is deliberately absent: mid-name API is ubiquitous in
    # benign vars, and every real secret it caught ($OPENAI_API_KEY) already ends in KEY/TOKEN.
    (rf'curl\s+[^\n]{{0,2048}}{_SECRET_VAR}', "exfil_curl", "all"),
    (rf'wget\s+[^\n]{{0,2048}}{_SECRET_VAR}', "exfil_wget", "all"),
    (r'cat\s+[^\n]{0,2048}(\.env|credentials|\.netrc|\.pgpass|\.npmrc|\.pypirc)', "read_secrets", "all"),
    (r'(send|post|upload|transmit)\s+[^\n]{0,2048}\s+(to|at)\s+https?://', "send_to_url", "strict"),
    (rf'(include|output|print|share)\s+{_FILLER}(conversation|chat\s+history|previous\s+messages|full\s+context|entire\s+context)', "context_exfil", "strict"),

    # ── Persistence / SSH backdoor (strict scope — memory + skills) ──
    (r'authorized_keys', "ssh_backdoor", "strict"),
    # Write-verb gated like the *_config_mod rules: a bare path match blocked ordinary docs
    # ("check $HOME/.ssh is chmod 700"). ``>>?`` covers a leading redirect with no verb word;
    # ``open(`` covers the scripted-write shape; chmod/chown/sed/truncate/rm/touch/curl/wget/git
    # mutate the directory without an obvious copy verb.
    (r'(?:\b(?:echo|cat|cp|mv|dd|tee|install|printf|rsync|scp|ln|append|add|write'
     r'|sed|chmod|chown|truncate|rm|touch|curl|wget|git)\b|\bopen\s*\(|>>?)'
     r'[^\n]{0,512}(?:\$HOME/\.ssh|~/\.ssh)', "ssh_access", "strict"),
    (r'\$HOME/\.hermes/\.env|\~/\.hermes/\.env', "hermes_env", "strict"),
    (rf'{_MODIFY}(?:AGENTS\.md|CLAUDE\.md|\.cursorrules|\.clinerules)', "agent_config_mod", "strict"),
    (rf'{_MODIFY}\.hermes/(config\.yaml|SOUL\.md)', "hermes_config_mod", "strict"),

    # ── Hardcoded secrets ────────────────────────────────────────────
    # The lookahead skips a value that is itself an environment-variable NAME
    # (SHOUTY_SNAKE, ≥2 underscore-separated segments): ENV_PASSWORD =
    # "MYPLUGIN_APP_PASSWORD" says where the credential lives, it does not embed
    # one (#116221). Scoped case-sensitive on purpose — the pattern compiles with
    # IGNORECASE and a lowercase snake value is the password-passphrase shape
    # ("correct_horse_battery_staple"); requiring an underscore segment keeps
    # underscore-free all-caps credentials (AWS AKIA…, base32) matched.
    (r'(?:api[_-]?key|token|secret|password)\s*[=:]\s*["\']'
     r'(?!(?-i:[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)["\'])'
     r'[A-Za-z0-9+/=_-]{20,}', "hardcoded_secret", "strict"),
]

# Invisible / bidirectional unicode used in injection attacks (aligned with skills_guard.py
# INVISIBLE_CHARS): zero-width space/non-joiner/joiner, word joiner, invisible times/separator/
# plus, BOM, LTR/RTL embedding + pop + overrides, LTR/RTL/first-strong isolates + pop.
INVISIBLE_CHARS = frozenset(
    "\u200b\u200c\u200d\u2060\u2062\u2063\u2064\ufeff"
    "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069")

# Compiled per scope at import; inclusion is cumulative (all ⊂ context ⊂ strict).
_SCOPE_SETS = {"all": ("all", "context", "strict"), "context": ("context", "strict"), "strict": ("strict",)}


def _compile() -> dict[str, List[Tuple[re.Pattern, str]]]:
    compiled: dict[str, List[Tuple[re.Pattern, str]]] = {"all": [], "context": [], "strict": []}
    for pattern, pid, scope in _PATTERNS:
        if scope not in _SCOPE_SETS:
            raise ValueError(f"threat_patterns: unknown scope {scope!r} for pattern {pid!r}")
        for s in _SCOPE_SETS[scope]:
            # #111334: strict is unconditional — dual-compile the _QUOTED
            # patterns without the citation lookbehind, so a quoted
            # directive in a user-authored write (memory, install) fires.
            # The literal replace is exact: _QUOTED is a fixed fragment.
            src = pattern.replace(_QUOTED, "") if s == "strict" else pattern
            compiled[s].append((re.compile(src, re.IGNORECASE), pid))
    return compiled


_COMPILED = _compile()


def scan_for_threats(content: str, scope: str = "context") -> List[str]:
    """Matched pattern IDs in ``content`` for ``scope``; invisible codepoints are
    reported as ``"invisible_unicode_U+XXXX"``. Raises ValueError on an unknown scope.
    ``prompt_injection`` alone has an intent-cue guard: a same-sentence descriptive
    cue before the match marks doctrine (#92644) and skips the finding. Both
    doctrine exemptions — this cue guard and the ``_QUOTED`` citation lookbehind —
    never apply at scope="strict": there scanning is unconditional (#111334)."""
    if not content:
        return []
    if (patterns := _COMPILED.get(scope)) is None:
        raise ValueError(f"scan_for_threats: unknown scope {scope!r}")
    content = content[:MAX_SCAN_CHARS]
    # Invisible unicode is checked on the RAW content: NFKC below can strip these codepoints.
    findings: List[str] = [f"invisible_unicode_U+{ord(ch):04X}" for ch in set(content) & INVISIBLE_CHARS]
    # NFKC folds full-width / compatibility variants (ｃａｔ → cat) against homograph bypass.
    # It does NOT fold cross-script confusables (Cyrillic ``а``) — that needs a TR#39 database.
    normalised = unicodedata.normalize("NFKC", content)
    # #111334: at strict the cue guard is short-circuited (quoted variants are
    # already compiled in without the lookbehind — see _compile).
    guard_cues = scope != "strict"
    for compiled, pid in patterns:
        # #92644: a prompt_injection hit whose 60-char prefix carries a
        # descriptive cue ("telling you to …") is doctrine, not a directive.
        # finditer, not search: a descriptive occurrence must not mask a later
        # bare directive in the same content.
        if guard_cues and pid in _INTENT_GUARDED_IDS:
            if any(not _is_descriptive(normalised, m.start()) for m in compiled.finditer(normalised)):
                findings.append(pid)
            continue
        if compiled.search(normalised):
            findings.append(pid)
    return findings


def first_threat_message(content: str, scope: str = "strict") -> Optional[str]:
    """User-facing error for the first threat found, or None (block-on-first-hit paths)."""
    findings = scan_for_threats(content, scope=scope)
    if not findings:
        return None
    pid = findings[0]
    if pid.startswith("invisible_unicode_"):
        codepoint = pid.replace("invisible_unicode_", "")
        return f"Blocked: content contains invisible unicode character {codepoint} (possible injection)."
    return (f"Blocked: content matches threat pattern '{pid}'. "
            f"Content is injected into the system prompt and must not contain "
            f"injection or exfiltration payloads.")


__all__ = ["INVISIBLE_CHARS", "MAX_SCAN_CHARS", "scan_for_threats", "first_threat_message"]
