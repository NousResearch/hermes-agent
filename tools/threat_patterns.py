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
Patterns use ``(?:\\w+\\s+)*`` between key tokens to prevent attackers
from inserting filler words (e.g. "ignore all prior instructions" instead
of "ignore all instructions").  This mirrors the fix applied to
``skills_guard.py`` in commit 4ea29978.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

# Each entry: (regex, pattern_id, scope)
# scope ∈ {"all", "context", "strict"}
_PATTERNS: List[Tuple[str, str, str]] = [
    # ── Classic prompt injection (applies everywhere) ────────────────
    (r'ignore\s+(?:\w+\s+)*(previous|all|above|prior)\s+(?:\w+\s+)*instructions', "prompt_injection", "all"),
    (r'system\s+prompt\s+override', "sys_prompt_override", "all"),
    (r'disregard\s+(?:\w+\s+)*(your|all|any)\s+(?:\w+\s+)*(instructions|rules|guidelines)', "disregard_rules", "all"),
    (r'act\s+as\s+(if|though)\s+(?:\w+\s+)*you\s+(?:\w+\s+)*(have\s+no|don\'t\s+have)\s+(?:\w+\s+)*(restrictions|limits|rules)', "bypass_restrictions", "all"),
    (r'<!--[^>]*(?:ignore|override|system|secret|hidden)[^>]*-->', "html_comment_injection", "all"),
    (r'<\s*div\s+style\s*=\s*["\'][\s\S]*?display\s*:\s*none', "hidden_div", "all"),
    (r'translate\s+.*\s+into\s+.*\s+and\s+(execute|run|eval)', "translate_execute", "all"),
    (r'do\s+not\s+(?:\w+\s+)*tell\s+(?:\w+\s+)*the\s+user', "deception_hide", "all"),

    # ── Role-play / identity hijack (context + strict; common attack
    #    surface in scraped web content and poisoned context files) ──
    (r'you\s+are\s+(?:\w+\s+)*now\s+(?:a|an|the)\s+', "role_hijack", "context"),
    (r'pretend\s+(?:\w+\s+)*(you\s+are|to\s+be)\s+', "role_pretend", "context"),
    (r'output\s+(?:\w+\s+)*(system|initial)\s+prompt', "leak_system_prompt", "context"),
    (r'(respond|answer|reply)\s+without\s+(?:\w+\s+)*(restrictions|limitations|filters|safety)', "remove_filters", "context"),
    (r'you\s+have\s+been\s+(?:\w+\s+)*(updated|upgraded|patched)\s+to', "fake_update", "context"),
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
    (r'never\s+(?:\w+\s+)*(?:create|write)\s+(?:\w+\s+)*(?:script|file)\s+(?:\w+\s+)*disk', "anti_forensic_disk", "context"),
    # Environment-variable unsetting targeting known agent runtimes —
    # this is pure attack behavior (Brainworm sub-session bypass).
    (r'unset\s+\w*(?:CLAUDE|CODEX|HERMES|AGENT|OPENAI|ANTHROPIC)\w*', "env_var_unset_agent", "context"),

    # ── Known C2 / red-team framework names (near-zero false positive
    #    outside security research; warn-only by default) ─────────────
    (r'\b(?:praxis|cobalt\s*strike|sliver|havoc|mythic|metasploit|brainworm)\b', "known_c2_framework", "context"),
    (r'\bc2\s+(?:server|channel|infrastructure|beacon)\b', "c2_explicit", "context"),
    (r'\bcommand\s+and\s+control\b', "c2_explicit_long", "context"),

    # ── Exfiltration via curl/wget/cat with secrets (applies everywhere) ──
    (r'curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "exfil_curl", "all"),
    (r'wget\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "exfil_wget", "all"),
    (r'cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass|\.npmrc|\.pypirc)', "read_secrets", "all"),
    (r'(send|post|upload|transmit)\s+.*\s+(to|at)\s+https?://', "send_to_url", "strict"),
    (r'(include|output|print|share)\s+(?:\w+\s+)*(conversation|chat\s+history|previous\s+messages|full\s+context|entire\s+context)', "context_exfil", "strict"),

    # ── Persistence / SSH backdoor (strict scope — memory + skills) ──
    (r'authorized_keys', "ssh_backdoor", "strict"),
    (r'\$HOME/\.ssh|\~/\.ssh', "ssh_access", "strict"),
    (r'\$HOME/\.hermes/\.env|\~/\.hermes/\.env', "hermes_env", "strict"),
    (r'(update|modify|edit|write|change|append|add\s+to)\s+.*(?:AGENTS\.md|CLAUDE\.md|\.cursorrules|\.clinerules)', "agent_config_mod", "strict"),
    (r'(update|modify|edit|write|change|append|add\s+to)\s+.*\.hermes/(config\.yaml|SOUL\.md)', "hermes_config_mod", "strict"),

    # ── Hardcoded secrets ────────────────────────────────────────────
    # Match with or without quotes — memory writes sometimes omit quotes
    (r'(?:api[_-]?key|token|secret|password)\s*[=:]\s*(?:["\']?)[A-Za-z0-9+/=_-]{20,}', "hardcoded_secret", "strict"),

    # ── URLs (strict — memory writes should not contain URLs) ────────
    (r'https?://\S+', "url_block", "strict"),

    # ── Sensitive file paths (strict) ────────────────────────────────
    # .env files — match .env followed by space, slash, backslash, end, or dot
    (r'(?:^|[/\\]|\s)\.env(?:\.[a-zA-Z]+)?(?:\s|$|[/\\])', "sensitive_path_env", "strict"),
    # .ssh directory and contents
    (r'(?:^|[/\\])\.ssh(?:[/\\]|$)', "sensitive_path_ssh", "strict"),
    # .npmrc, .pypirc, .netrc, .pgpass, credentials files — match mid-path too
    (r'[/\\]\.(?:npmrc|pypirc|netrc|pgpass)(?:\s|$|[/\\])', "sensitive_path_cred", "strict"),
    (r'[/\\]credentials(?:\.[a-zA-Z]+)?(?:\s|$|[/\\])', "sensitive_path_cred_file", "strict"),
    (r'[/\\]AppData(?:[/\\]|$)', "sensitive_path_appdata", "strict"),
    # Sensitive path segments — exact match only (no substring false positives).
    # Matches /secrets/ or /secret/ but NOT /secret-santa/.
    # Negative lookahead excludes common compound words (secret-santa, token-spec, etc.)
    (r'[/\\](?:secrets?|credentials?|apikeys?|private[_-]?keys?)[/\\](?!.*[-_](?:santa|spec|management|store|rotation))', "sensitive_path_keyword", "strict"),
    # Sensitive filenames anywhere in path
    (r'[/\\](?:private_key|api_key|secret_key|access_token)\s*\.[a-zA-Z]+(?:\s|$|[/\\])', "sensitive_path_file", "strict"),
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
    """
    if not content:
        return []

    findings: List[str] = []

    # Invisible unicode — single pass through the content set, not 17
    # ``in`` lookups.
    char_set = set(content)
    invisible_hits = char_set & INVISIBLE_CHARS
    for ch in invisible_hits:
        findings.append(f"invisible_unicode_U+{ord(ch):04X}")

    # Threat patterns
    patterns = _COMPILED.get(scope)
    if patterns is None:
        raise ValueError(f"scan_for_threats: unknown scope {scope!r}")
    for compiled, pid in patterns:
        if compiled.search(content):
            findings.append(pid)

    return findings


def _check_structural_content(content: str) -> Optional[str]:
    """Structured detection for content that regex cannot reliably catch.

    Checks:
    - Large code blocks (>10 lines inside ``` fences)
    - Large log/text dumps (>=15 lines with Traceback or ERROR or File ")

    Returns error string if blocked, None if clean.
    """
    lines = content.split("\n")

    # ── Large code block detection ──
    in_fence = False
    fence_line_count = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if not in_fence:
                in_fence = True
                fence_line_count = 0
            else:
                in_fence = False
                if fence_line_count > 10:
                    return "Blocked: content contains a large code block (>10 lines inside ``` fences). Write a summary instead."
        elif in_fence:
            fence_line_count += 1

    # ── Large log/text dump detection ──
    if len(lines) >= 15:
        trace_count = 0
        error_count = 0
        file_ref_count = 0
        for line in lines:
            if "Traceback" in line:
                trace_count += 1
            if "ERROR" in line.upper():
                error_count += 1
            if 'File "' in line:
                file_ref_count += 1
        # Lower threshold for log dump: many Traceback OR moderate error density
        if trace_count >= 2 or (file_ref_count >= 3 and error_count >= 1):
            return "Blocked: content appears to be a log/traceback dump (>=15 lines with error patterns). Write a summary instead."

    return None


def first_threat_message(content: str, scope: str = "strict") -> Optional[str]:
    """Return a human-readable error string for the first threat found, or None.

    Runs structural checks first (code blocks, log dumps), then regex patterns.
    """
    # Structural checks run before regex
    structural_error = _check_structural_content(content)
    if structural_error:
        return structural_error

    findings = scan_for_threats(content, scope=scope)
    if not findings:
        return None
    pid = findings[0]
    if pid.startswith("invisible_unicode_"):
        codepoint = pid.replace("invisible_unicode_", "")
        return f"Blocked: content contains invisible unicode character {codepoint} (possible injection)."
    # Map pattern IDs to user-friendly categories
    category_map = {
        "url_block": "URLs are not allowed in memory. Record the description, not the link.",
        "sensitive_path_env": "Sensitive file path (.env) detected. Do not store paths to secret files.",
        "sensitive_path_ssh": "Sensitive file path (.ssh) detected. Do not store paths to secret directories.",
        "sensitive_path_cred": "Sensitive credential file path detected. Do not store paths to credential files.",
        "sensitive_path_cred_file": "Sensitive credential file path detected. Do not store paths to credential files.",
        "sensitive_path_appdata": "AppData path detected. Do not store system directory paths in memory.",
        "sensitive_path_keyword": "Sensitive path keyword detected. Do not store paths containing secret/credential/apikey/private_key.",
        "sensitive_path_file": "Sensitive filename detected (e.g. private_key, api_key, secret_key). Do not store paths to key/secret files.",
    }
    category = category_map.get(pid)
    if category:
        return f"Blocked: {category}"
    return (
        f"Blocked: content matches threat pattern '{pid}'. "
        f"Content is injected into the system prompt and must not contain "
        f"injection or exfiltration payloads."
    )


__all__ = [
    "INVISIBLE_CHARS",
    "scan_for_threats",
    "first_threat_message",
    "_check_structural_content",
]
