"""Shared threat-pattern library (prompt injection / promptware / exfiltration) for
``agent/prompt_builder.py``, ``tools/memory_tool.py`` and ``agent/tool_dispatch_helpers.py``.
Each pattern is ``(regex, pattern_id, scope)``; scope is cumulative: ``"all"`` everywhere,
``"context"`` adds promptware / C2 / role hijack for context files, memory and tool results
(warn-level: that content is not user-authored), ``"strict"`` adds aggressive checks only for
user-mediated writes (memory, skill installs) where a block is resolvable. New patterns must
anchor on C2 vocabulary or unambiguous attack behavior, NOT bossy English ("you must" is common
in legitimate AGENTS.md); filler between tokens is the bounded ``_FILLER``."""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
import unicodedata
from pathlib import Path
from typing import List, Optional, Tuple

# Hard cap on scanned text: scanners are advisory, so bound worst-case runtime.
MAX_SCAN_CHARS = 65_536
# Bounded filler between key attack words (unbounded ``(?:\w+\s+)*`` backtracks badly).
_FILLER = r"(?:\w+\s+){0,8}"
# Env var reference ending in a secret-ish suffix (see exfil comment below).
_SECRET_VAR = r"\$\{?\w*(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)S?\b"
# Verb prefix for "modify agent config" patterns.
_MODIFY = r"(update|modify|edit|write|change|append|add\s+to)\s+[^\n]{0,2048}"
# (regex, pattern_id, scope); scope ∈ {"all", "context", "strict"}
_PATTERNS: List[Tuple[str, str, str]] = [
    # ── Classic prompt injection (applies everywhere) ────────────────
    (rf'ignore\s+{_FILLER}(previous|all|above|prior)\s+{_FILLER}instructions', "prompt_injection", "all"),
    (r'system\s+prompt\s+override', "sys_prompt_override", "all"),
    (rf'disregard\s+{_FILLER}(your|all|any)\s+{_FILLER}(instructions|rules|guidelines)', "disregard_rules", "all"),
    (rf'act\s+as\s+(if|though)\s+{_FILLER}you\s+{_FILLER}(have\s+no|don\'t\s+have)\s+{_FILLER}(restrictions|limits|rules)', "bypass_restrictions", "all"),
    (r'<!--[^>]{0,512}(?:ignore|override|system|secret|hidden)[^>]{0,512}-->', "html_comment_injection", "all"),
    (r'<\s*div\s+style\s*=\s*["\'][^>]{0,2048}display\s*:\s*none', "hidden_div", "all"),
    (
        r"translate\s+[^\n]{0,512}\s+into\s+\w+(?:[\s-]+\w+){0,2}\s+and\s+(execute|run|eval)\b",
        "translate_execute",
        "all",
    ),
    (rf'do\s+not\s+{_FILLER}tell\s+{_FILLER}the\s+user', "deception_hide", "all"),

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
    (r'\$HOME/\.ssh|\~/\.ssh', "ssh_access", "strict"),
    (r'\$HOME/\.hermes/\.env|\~/\.hermes/\.env', "hermes_env", "strict"),
    (rf'{_MODIFY}(?:AGENTS\.md|CLAUDE\.md|\.cursorrules|\.clinerules)', "agent_config_mod", "strict"),
    (rf'{_MODIFY}\.hermes/(config\.yaml|SOUL\.md)', "hermes_config_mod", "strict"),

    # ── Hardcoded secrets ────────────────────────────────────────────
    (r'(?:api[_-]?key|token|secret|password)\s*[=:]\s*["\'][A-Za-z0-9+/=_-]{20,}', "hardcoded_secret", "strict"),
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
            compiled[s].append((re.compile(pattern, re.IGNORECASE), pid))
    return compiled


_COMPILED = _compile()


def scan_for_threats(content: str, scope: str = "context") -> List[str]:
    """Matched pattern IDs in ``content`` for ``scope``; invisible codepoints are
    reported as ``"invisible_unicode_U+XXXX"``. Raises ValueError on an unknown scope."""
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
    findings.extend(pid for compiled, pid in patterns if compiled.search(normalised))
    return findings


# Shadow-mode comparison log for `jev screen` (TypeSafe CLI) — a second, purely informational signal
# used only where callers opt in (see _jev_screen_shadow). Never wired into scan_for_threats itself:
# these deterministic regex patterns remain the only real authority everywhere.
_JEV_SCREEN_SHADOW_LOG = Path(__file__).resolve().parents[1] / "cron" / "logs" / "jev-shadow-screen-threats.jsonl"
_JEV_SCREEN_TIMEOUT_SECONDS = 8.0


def _append_jev_screen_shadow_log(record: dict) -> None:
    """Append one JSON line. Never raises: a log-write failure must not affect the caller, which is
    itself already inside a best-effort shadow path."""
    try:
        _JEV_SCREEN_SHADOW_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(_JEV_SCREEN_SHADOW_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _jev_screen_shadow(content: str, source: str, regex_findings: List[str]) -> None:
    """Best-effort second opinion via `jev screen` (TypeSafe CLI), for callers that want it as a
    complementary, non-blocking signal alongside the deterministic regex scan above.

    This NEVER blocks the caller (runs on a daemon thread), NEVER modifies anything, and NEVER
    changes what the regex scanners already decided — those patterns stay the sole real authority.
    Only a discrepancy where jev flags something the regex missed (``regex_findings`` empty but jev
    recommends "review" or "block") is written to ``_JEV_SCREEN_SHADOW_LOG``, for later human review —
    not for any automated action. Any failure (binary missing, timeout, malformed output) is
    swallowed silently; this must never raise into or slow down a security-path caller.
    """
    if not content or not content.strip():
        return

    def _run() -> None:
        try:
            proc = subprocess.run(
                ["jev", "screen", "--json", "--timeout", str(int(_JEV_SCREEN_TIMEOUT_SECONDS * 1000))],
                input=content[:MAX_SCAN_CHARS], capture_output=True, text=True,
                timeout=_JEV_SCREEN_TIMEOUT_SECONDS + 2,
            )
            if proc.returncode != 0:
                return
            data = json.loads(proc.stdout)
            action = (data.get("recommendation") or {}).get("action")
            if not regex_findings and action in ("block", "review"):
                _append_jev_screen_shadow_log({
                    "timestamp": time.time(), "source": source, "jev_action": action,
                    "jev_injection_probability": (data.get("probabilities") or {}).get("injection"),
                    "content_chars": len(content),
                })
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True, name="jev-shadow-screen-threats").start()


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
