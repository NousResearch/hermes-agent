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
# The whole input is still scanned — just in bounded chunks of this size — so a
# payload past the cap (which the renderer may still emit as a tail) is not
# silently dropped, while worst-case per-chunk runtime stays predictable.
MAX_SCAN_CHARS = 65_536
# Overlap between adjacent scan chunks.  A match no wider than the overlap that
# straddles a chunk edge lies whole inside the next chunk, so every quantifier
# below is bounded (``\s{1,64}``, not ``\s+``) and ``_compile`` rejects any
# pattern whose maximum match width exceeds this.
_CHUNK_OVERLAP = 4096
# Bounded filler between key attack words (unbounded ``(?:\w+\s+)*`` backtracks badly).
_FILLER = r"(?:\w{1,64}\s{1,64}){0,8}"
# Env var reference ending in a secret-ish suffix (see exfil comment below).
_SECRET_VAR = r"\$\{?\w{0,256}(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)S?\b"
# Verb prefix for "modify agent config" patterns.
_MODIFY = r"(update|modify|edit|write|change|append|add\s{1,64}to)\s{1,64}[^\n]{0,2048}"
# (regex, pattern_id, scope); scope ∈ {"all", "context", "strict"}
_PATTERNS: list[tuple[str, str, str]] = [
    # ── Classic prompt injection (applies everywhere) ────────────────
    (rf'ignore\s{{1,64}}{_FILLER}(previous|all|above|prior)\s{{1,64}}{_FILLER}instructions', "prompt_injection", "all"),
    (r'system\s{1,64}prompt\s{1,64}override', "sys_prompt_override", "all"),
    (rf'disregard\s{{1,64}}{_FILLER}(your|all|any)\s{{1,64}}{_FILLER}(instructions|rules|guidelines)', "disregard_rules", "all"),
    (rf'act\s{{1,64}}as\s{{1,64}}(if|though)\s{{1,64}}{_FILLER}you\s{{1,64}}{_FILLER}(have\s{{1,64}}no|don\'t\s{{1,64}}have)\s{{1,64}}{_FILLER}(restrictions|limits|rules)', "bypass_restrictions", "all"),
    (r'<!--[^>]{0,512}(?:ignore|override|system|secret|hidden)[^>]{0,512}-->', "html_comment_injection", "all"),
    (r'<\s{0,64}div\s{1,64}style\s{0,64}=\s{0,64}["\'][^>]{0,2048}display\s{0,64}:\s{0,64}none', "hidden_div", "all"),
    (
        r"translate\s{1,64}[^\n]{0,512}\s{1,64}into\s{1,64}\w{1,64}(?:[\s-]{1,64}\w{1,64}){0,2}\s{1,64}and\s{1,64}(execute|run|eval)\b",
        "translate_execute",
        "all",
    ),
    (rf'do\s{{1,64}}not\s{{1,64}}{_FILLER}tell\s{{1,64}}{_FILLER}the\s{{1,64}}user', "deception_hide", "all"),

    # ── Role-play / identity hijack (scraped web content, poisoned context files) ──
    (rf'you\s{{1,64}}are\s{{1,64}}{_FILLER}now\s{{1,64}}(?:a|an|the)\s{{1,64}}', "role_hijack", "context"),
    (rf'pretend\s{{1,64}}{_FILLER}(you\s{{1,64}}are|to\s{{1,64}}be)\s{{1,64}}', "role_pretend", "context"),
    (rf'output\s{{1,64}}{_FILLER}(system|initial)\s{{1,64}}prompt', "leak_system_prompt", "context"),
    (rf'(respond|answer|reply)\s{{1,64}}without\s{{1,64}}{_FILLER}(restrictions|limitations|filters|safety)', "remove_filters", "context"),
    (rf'you\s{{1,64}}have\s{{1,64}}been\s{{1,64}}{_FILLER}(updated|upgraded|patched)\s{{1,64}}to', "fake_update", "context"),
    # Brainworm tell: identity override via spec. Verb pair anchored so "name your variables" is safe.
    (r'\bname\s{1,64}yourself\s{1,64}\w{1,64}', "identity_override", "context"),

    # ── C2 / Brainworm-style promptware (context scope) ──────────────
    # Anchored on C2 vocabulary. "register as a node" appears in legitimate distributed-systems
    # docs, so this is WARN not block: a researcher reading the Brainworm post keeps their session.
    (r'register\s{1,64}(as\s{1,64})?a?\s{0,64}node', "c2_node_registration", "context"),
    (r'(heartbeat|beacon|check[\s\-]?in)\s{1,64}(to|with)\s{1,64}', "c2_heartbeat", "context"),
    (r'pull\s{1,64}(down\s{1,64})?(?:new\s{1,64})?task(?:ing|s)?\b', "c2_task_pull", "context"),
    (r'connect\s{1,64}to\s{1,64}the\s{1,64}network\b', "c2_network_connect", "context"),
    # C2-specific verbs avoid the broader "you must X" false positive.
    (r'you\s{1,64}must\s{1,64}(?:\w{1,64}\s{1,64}){0,3}(register|connect|report|beacon)\b', "forced_action", "context"),
    # Anti-forensic instructions: near-zero false positive in legitimate content.
    (r'only\s{1,64}use\s{1,64}one[\s\-]?liners?\b', "anti_forensic_oneliner", "context"),
    (rf'never\s{{1,64}}{_FILLER}(?:create|write)\s{{1,64}}{_FILLER}(?:script|file)\s{{1,64}}{_FILLER}disk', "anti_forensic_disk", "context"),
    # Unsetting agent-runtime env vars is pure attack behavior (Brainworm sub-session bypass).
    (r'unset\s{1,64}\w{0,256}(?:CLAUDE|CODEX|HERMES|AGENT|OPENAI|ANTHROPIC)\w{0,64}', "env_var_unset_agent", "context"),

    # ── Known C2 / red-team framework names (warn-only) ─────────────
    # Every token must be a distinctive offensive-security brand: a common English word here
    # (e.g. "praxis", also a legitimate agent name) false-positives whole AGENTS.md / SOUL.md files.
    (r'\b(?:cobalt\s{0,64}strike|sliver|havoc|mythic|metasploit|brainworm)\b', "known_c2_framework", "context"),
    (r'\bc2\s{1,64}(?:server|channel|infrastructure|beacon)\b', "c2_explicit", "context"),
    (r'\bcommand\s{1,64}and\s{1,64}control\b', "c2_explicit_long", "context"),

    # ── Exfiltration via curl/wget/cat with secrets (applies everywhere) ──
    # The var name ends with \b so benign names containing KEY/TOKEN as substrings
    # ($TRILLIUM_ETAPI_URL) pass. API is deliberately absent: mid-name API is ubiquitous in
    # benign vars, and every real secret it caught ($OPENAI_API_KEY) already ends in KEY/TOKEN.
    (rf'curl\s{{1,64}}[^\n]{{0,2048}}{_SECRET_VAR}', "exfil_curl", "all"),
    (rf'wget\s{{1,64}}[^\n]{{0,2048}}{_SECRET_VAR}', "exfil_wget", "all"),
    (r'cat\s{1,64}[^\n]{0,2048}(\.env|credentials|\.netrc|\.pgpass|\.npmrc|\.pypirc)', "read_secrets", "all"),
    (r'(send|post|upload|transmit)\s{1,64}[^\n]{0,2048}\s{1,64}(to|at)\s{1,64}https?://', "send_to_url", "strict"),
    (rf'(include|output|print|share)\s{{1,64}}{_FILLER}(conversation|chat\s{{1,64}}history|previous\s{{1,64}}messages|full\s{{1,64}}context|entire\s{{1,64}}context)', "context_exfil", "strict"),

    # ── Persistence / SSH backdoor (strict scope — memory + skills) ──
    (r'authorized_keys', "ssh_backdoor", "strict"),
    # Write-verb gated like the *_config_mod rules: a bare path match blocked ordinary docs
    # ("check $HOME/.ssh is chmod 700"). ``>>?`` covers a leading redirect with no verb word;
    # ``open(`` covers the scripted-write shape; chmod/chown/sed/truncate/rm/touch/curl/wget/git
    # mutate the directory without an obvious copy verb.
    (r'(?:\b(?:echo|cat|cp|mv|dd|tee|install|printf|rsync|scp|ln|append|add|write'
     r'|sed|chmod|chown|truncate|rm|touch|curl|wget|git)\b|\bopen\s{0,64}\(|>>?)'
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
    (r'(?:api[_-]?key|token|secret|password)\s{0,64}[=:]\s{0,64}["\']'
     r'(?!(?-i:[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)["\'])'
     r'[A-Za-z0-9+/=_-]{20}', "hardcoded_secret", "strict"),
]

# Invisible / bidirectional unicode used in injection attacks (aligned with skills_guard.py
# INVISIBLE_CHARS): zero-width space/non-joiner/joiner, word joiner, invisible times/separator/
# plus, BOM, LTR/RTL embedding + pop + overrides, LTR/RTL/first-strong isolates + pop.
INVISIBLE_CHARS = frozenset(
    "\u200b\u200c\u200d\u2060\u2062\u2063\u2064\ufeff"
    "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069")

# Compiled per scope at import; inclusion is cumulative (all ⊂ context ⊂ strict).
_SCOPE_SETS = {"all": ("all", "context", "strict"), "context": ("context", "strict"), "strict": ("strict",)}


def _compile() -> dict[str, list[tuple[re.Pattern, str]]]:
    compiled: dict[str, list[tuple[re.Pattern, str]]] = {"all": [], "context": [], "strict": []}
    for pattern, pid, scope in _PATTERNS:
        if scope not in _SCOPE_SETS:
            raise ValueError(f"threat_patterns: unknown scope {scope!r} for pattern {pid!r}")
        if (width := re._parser.parse(pattern).getwidth()[1]) > _CHUNK_OVERLAP:
            raise ValueError(f"threat_patterns: pattern {pid!r} can match {width} chars, "
                             f"wider than the {_CHUNK_OVERLAP}-char chunk overlap")
        for s in _SCOPE_SETS[scope]:
            compiled[s].append((re.compile(pattern, re.IGNORECASE), pid))
    return compiled


_COMPILED = _compile()


_WS_RUN = re.compile(r"\s{2,}")


def _collapse_ws(m: re.Match) -> str:
    return "\n" if "\n" in m.group() else " "


def scan_for_threats(content: str, scope: str = "context") -> list[str]:
    """Matched pattern IDs in ``content`` for ``scope``; invisible codepoints are
    reported as ``"invisible_unicode_U+XXXX"``. Raises ValueError on an unknown scope."""
    if not content:
        return []
    if (patterns := _COMPILED.get(scope)) is None:
        raise ValueError(f"scan_for_threats: unknown scope {scope!r}")
    # Invisible unicode is checked on the RAW content: NFKC below can strip these
    # codepoints.  Checked over the COMPLETE input, not just the first scan chunk.
    findings: list[str] = [f"invisible_unicode_U+{ord(ch):04X}" for ch in set(content) & INVISIBLE_CHARS]
    # Threat patterns — scan the COMPLETE input in bounded chunks with overlap so a
    # payload past ``MAX_SCAN_CHARS`` (which the renderer may still emit as a tail)
    # is not silently skipped.  Findings are deduplicated: a pattern can legitimately
    # fire in more than one overlapping chunk.  The overlap is at least every
    # pattern's maximum match width (checked in ``_compile``), so a match straddling
    # a chunk edge is seen whole by the next chunk.
    # NFKC folds full-width / compatibility variants (ｃａｔ → cat) against
    # homograph bypass.  It does NOT fold cross-script confusables (Cyrillic
    # ``а``) — that needs a TR#39 database.  Each whitespace run then collapses to
    # one char (newline-preserving, for the ``[^\n]`` spans) so the bounded ``\s``
    # quantifiers still see arbitrarily long whitespace as a single separator.
    normalised = _WS_RUN.sub(_collapse_ws, unicodedata.normalize("NFKC", content))
    found_ids: set[str] = set()
    step = MAX_SCAN_CHARS - _CHUNK_OVERLAP
    for start in range(0, len(normalised), step):
        chunk = normalised[start:start + MAX_SCAN_CHARS]
        for compiled, pid in patterns:
            if pid in found_ids:
                continue
            if compiled.search(chunk):
                found_ids.add(pid)
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


__all__ = ["INVISIBLE_CHARS", "MAX_SCAN_CHARS", "first_threat_message", "scan_for_threats"]
