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
# U+200D is the documented exception — see _EMOJI_ZWJ_NEIGHBOUR_RANGES / zwj_is_emoji_only().
INVISIBLE_CHARS = frozenset(
    "\u200b\u200c\u200d\u2060\u2062\u2063\u2064\ufeff"
    "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069")

# --- U+200D (zero-width joiner) is the one invisible codepoint with a legitimate
# orthographic use: it joins emoji into a single grapheme (surfer, family, heart-on-fire).
# A ZWJ flanked by emoji on BOTH sides is part of the emoji, not hidden text, and dropping
# a whole SOUL.md / SKILL.md over one is a false positive: two context files were rejected
# on every session start for 30 days before anyone noticed the log line.
#
# The neighbour test is derived from the Unicode emoji properties — accept a side when it is
# Extended_Pictographic or Emoji_Modifier — rather than a hand-rolled block list. The
# Emoji_Modifier half matters: skin-tone modifiers (U+1F3FB-U+1F3FF) are NOT
# Extended_Pictographic, so a pictographic-only test still rejects every skin-toned
# family/role emoji. The block-list form this replaces missed 4 of the 1614 RGI ZWJ
# sequences (black cat, black bird, and the two head-shaking emoji, whose partners live in
# Arrows and Misc Symbols & Arrows). Ranges generated from emoji-data.txt, Unicode 18.0.0
# (emoji 17.0). Bare ZWJs glued between non-emoji characters (``pay`` + U+200D + ``load``)
# are still flagged.
_EXTENDED_PICTOGRAPHIC_RANGES: Tuple[Tuple[int, int], ...] = (
    (0x00A9, 0x00A9), (0x00AE, 0x00AE), (0x203C, 0x203C), (0x2049, 0x2049), (0x2122, 0x2122),
    (0x2139, 0x2139), (0x2194, 0x2199), (0x21A9, 0x21AA), (0x231A, 0x231B), (0x2328, 0x2328),
    (0x23CF, 0x23CF), (0x23E9, 0x23EC), (0x23ED, 0x23EE), (0x23EF, 0x23EF), (0x23F0, 0x23F0),
    (0x23F1, 0x23F2), (0x23F3, 0x23F3), (0x23F8, 0x23FA), (0x24C2, 0x24C2), (0x25AA, 0x25AB),
    (0x25B6, 0x25B6), (0x25C0, 0x25C0), (0x25FB, 0x25FE), (0x2600, 0x2601), (0x2602, 0x2603),
    (0x2604, 0x2604), (0x260E, 0x260E), (0x2611, 0x2611), (0x2614, 0x2615), (0x2618, 0x2618),
    (0x261D, 0x261D), (0x2620, 0x2620), (0x2622, 0x2623), (0x2626, 0x2626), (0x262A, 0x262A),
    (0x262E, 0x262E), (0x262F, 0x262F), (0x2638, 0x2639), (0x263A, 0x263A), (0x2640, 0x2640),
    (0x2642, 0x2642), (0x2648, 0x2653), (0x265F, 0x265F), (0x2660, 0x2660), (0x2663, 0x2663),
    (0x2665, 0x2666), (0x2668, 0x2668), (0x267B, 0x267B), (0x267E, 0x267E), (0x267F, 0x267F),
    (0x2692, 0x2692), (0x2693, 0x2693), (0x2694, 0x2694), (0x2695, 0x2695), (0x2696, 0x2697),
    (0x2699, 0x2699), (0x269B, 0x269C), (0x26A0, 0x26A1), (0x26A7, 0x26A7), (0x26AA, 0x26AB),
    (0x26B0, 0x26B1), (0x26BD, 0x26BE), (0x26C4, 0x26C5), (0x26C8, 0x26C8), (0x26CE, 0x26CE),
    (0x26CF, 0x26CF), (0x26D1, 0x26D1), (0x26D3, 0x26D3), (0x26D4, 0x26D4), (0x26E9, 0x26E9),
    (0x26EA, 0x26EA), (0x26F0, 0x26F1), (0x26F2, 0x26F3), (0x26F4, 0x26F4), (0x26F5, 0x26F5),
    (0x26F7, 0x26F9), (0x26FA, 0x26FA), (0x26FD, 0x26FD), (0x2702, 0x2702), (0x2705, 0x2705),
    (0x2708, 0x270C), (0x270D, 0x270D), (0x270F, 0x270F), (0x2712, 0x2712), (0x2714, 0x2714),
    (0x2716, 0x2716), (0x271D, 0x271D), (0x2721, 0x2721), (0x2728, 0x2728), (0x2733, 0x2734),
    (0x2744, 0x2744), (0x2747, 0x2747), (0x274C, 0x274C), (0x274E, 0x274E), (0x2753, 0x2755),
    (0x2757, 0x2757), (0x2763, 0x2763), (0x2764, 0x2764), (0x2795, 0x2797), (0x27A1, 0x27A1),
    (0x27B0, 0x27B0), (0x27BF, 0x27BF), (0x2934, 0x2935), (0x2B05, 0x2B07), (0x2B1B, 0x2B1C),
    (0x2B50, 0x2B50), (0x2B55, 0x2B55), (0x3030, 0x3030), (0x303D, 0x303D), (0x3297, 0x3297),
    (0x3299, 0x3299), (0x1F004, 0x1F004), (0x1F02C, 0x1F02F), (0x1F094, 0x1F09F), (0x1F0AF, 0x1F0B0),
    (0x1F0C0, 0x1F0C0), (0x1F0CF, 0x1F0CF), (0x1F0D0, 0x1F0D0), (0x1F0F6, 0x1F0FF), (0x1F170, 0x1F171),
    (0x1F17E, 0x1F17F), (0x1F18E, 0x1F18E), (0x1F191, 0x1F19A), (0x1F1AF, 0x1F1E5), (0x1F201, 0x1F202),
    (0x1F203, 0x1F20F), (0x1F21A, 0x1F21A), (0x1F22F, 0x1F22F), (0x1F232, 0x1F23A), (0x1F23C, 0x1F23F),
    (0x1F249, 0x1F24F), (0x1F250, 0x1F251), (0x1F252, 0x1F25F), (0x1F266, 0x1F2FF), (0x1F300, 0x1F30C),
    (0x1F30D, 0x1F30E), (0x1F30F, 0x1F30F), (0x1F310, 0x1F310), (0x1F311, 0x1F311), (0x1F312, 0x1F312),
    (0x1F313, 0x1F315), (0x1F316, 0x1F318), (0x1F319, 0x1F319), (0x1F31A, 0x1F31A), (0x1F31B, 0x1F31B),
    (0x1F31C, 0x1F31C), (0x1F31D, 0x1F31E), (0x1F31F, 0x1F320), (0x1F321, 0x1F321), (0x1F324, 0x1F32C),
    (0x1F32D, 0x1F32F), (0x1F330, 0x1F331), (0x1F332, 0x1F333), (0x1F334, 0x1F335), (0x1F336, 0x1F336),
    (0x1F337, 0x1F34A), (0x1F34B, 0x1F34B), (0x1F34C, 0x1F34F), (0x1F350, 0x1F350), (0x1F351, 0x1F37B),
    (0x1F37C, 0x1F37C), (0x1F37D, 0x1F37D), (0x1F37E, 0x1F37F), (0x1F380, 0x1F393), (0x1F396, 0x1F397),
    (0x1F399, 0x1F39B), (0x1F39E, 0x1F39F), (0x1F3A0, 0x1F3C4), (0x1F3C5, 0x1F3C5), (0x1F3C6, 0x1F3C6),
    (0x1F3C7, 0x1F3C7), (0x1F3C8, 0x1F3C8), (0x1F3C9, 0x1F3C9), (0x1F3CA, 0x1F3CA), (0x1F3CB, 0x1F3CE),
    (0x1F3CF, 0x1F3D3), (0x1F3D4, 0x1F3DF), (0x1F3E0, 0x1F3E3), (0x1F3E4, 0x1F3E4), (0x1F3E5, 0x1F3F0),
    (0x1F3F3, 0x1F3F3), (0x1F3F4, 0x1F3F4), (0x1F3F5, 0x1F3F5), (0x1F3F7, 0x1F3F7), (0x1F3F8, 0x1F3FA),
    (0x1F400, 0x1F407), (0x1F408, 0x1F408), (0x1F409, 0x1F40B), (0x1F40C, 0x1F40E), (0x1F40F, 0x1F410),
    (0x1F411, 0x1F412), (0x1F413, 0x1F413), (0x1F414, 0x1F414), (0x1F415, 0x1F415), (0x1F416, 0x1F416),
    (0x1F417, 0x1F429), (0x1F42A, 0x1F42A), (0x1F42B, 0x1F43E), (0x1F43F, 0x1F43F), (0x1F440, 0x1F440),
    (0x1F441, 0x1F441), (0x1F442, 0x1F464), (0x1F465, 0x1F465), (0x1F466, 0x1F46B), (0x1F46C, 0x1F46D),
    (0x1F46E, 0x1F4AC), (0x1F4AD, 0x1F4AD), (0x1F4AE, 0x1F4B5), (0x1F4B6, 0x1F4B7), (0x1F4B8, 0x1F4EB),
    (0x1F4EC, 0x1F4ED), (0x1F4EE, 0x1F4EE), (0x1F4EF, 0x1F4EF), (0x1F4F0, 0x1F4F4), (0x1F4F5, 0x1F4F5),
    (0x1F4F6, 0x1F4F7), (0x1F4F8, 0x1F4F8), (0x1F4F9, 0x1F4FC), (0x1F4FD, 0x1F4FD), (0x1F4FF, 0x1F502),
    (0x1F503, 0x1F503), (0x1F504, 0x1F507), (0x1F508, 0x1F508), (0x1F509, 0x1F509), (0x1F50A, 0x1F514),
    (0x1F515, 0x1F515), (0x1F516, 0x1F52B), (0x1F52C, 0x1F52D), (0x1F52E, 0x1F53D), (0x1F549, 0x1F54A),
    (0x1F54B, 0x1F54E), (0x1F550, 0x1F55B), (0x1F55C, 0x1F567), (0x1F56F, 0x1F570), (0x1F573, 0x1F579),
    (0x1F57A, 0x1F57A), (0x1F587, 0x1F587), (0x1F58A, 0x1F58D), (0x1F590, 0x1F590), (0x1F595, 0x1F596),
    (0x1F5A4, 0x1F5A4), (0x1F5A5, 0x1F5A5), (0x1F5A8, 0x1F5A8), (0x1F5B1, 0x1F5B2), (0x1F5BC, 0x1F5BC),
    (0x1F5C2, 0x1F5C4), (0x1F5D1, 0x1F5D3), (0x1F5DC, 0x1F5DE), (0x1F5E1, 0x1F5E1), (0x1F5E3, 0x1F5E3),
    (0x1F5E8, 0x1F5E8), (0x1F5EF, 0x1F5EF), (0x1F5F3, 0x1F5F3), (0x1F5FA, 0x1F5FA), (0x1F5FB, 0x1F5FF),
    (0x1F600, 0x1F600), (0x1F601, 0x1F606), (0x1F607, 0x1F608), (0x1F609, 0x1F60D), (0x1F60E, 0x1F60E),
    (0x1F60F, 0x1F60F), (0x1F610, 0x1F610), (0x1F611, 0x1F611), (0x1F612, 0x1F614), (0x1F615, 0x1F615),
    (0x1F616, 0x1F616), (0x1F617, 0x1F617), (0x1F618, 0x1F618), (0x1F619, 0x1F619), (0x1F61A, 0x1F61A),
    (0x1F61B, 0x1F61B), (0x1F61C, 0x1F61E), (0x1F61F, 0x1F61F), (0x1F620, 0x1F625), (0x1F626, 0x1F627),
    (0x1F628, 0x1F62B), (0x1F62C, 0x1F62C), (0x1F62D, 0x1F62D), (0x1F62E, 0x1F62F), (0x1F630, 0x1F633),
    (0x1F634, 0x1F634), (0x1F635, 0x1F635), (0x1F636, 0x1F636), (0x1F637, 0x1F640), (0x1F641, 0x1F644),
    (0x1F645, 0x1F64F), (0x1F680, 0x1F680), (0x1F681, 0x1F682), (0x1F683, 0x1F685), (0x1F686, 0x1F686),
    (0x1F687, 0x1F687), (0x1F688, 0x1F688), (0x1F689, 0x1F689), (0x1F68A, 0x1F68B), (0x1F68C, 0x1F68C),
    (0x1F68D, 0x1F68D), (0x1F68E, 0x1F68E), (0x1F68F, 0x1F68F), (0x1F690, 0x1F690), (0x1F691, 0x1F693),
    (0x1F694, 0x1F694), (0x1F695, 0x1F695), (0x1F696, 0x1F696), (0x1F697, 0x1F697), (0x1F698, 0x1F698),
    (0x1F699, 0x1F69A), (0x1F69B, 0x1F6A1), (0x1F6A2, 0x1F6A2), (0x1F6A3, 0x1F6A3), (0x1F6A4, 0x1F6A5),
    (0x1F6A6, 0x1F6A6), (0x1F6A7, 0x1F6AD), (0x1F6AE, 0x1F6B1), (0x1F6B2, 0x1F6B2), (0x1F6B3, 0x1F6B5),
    (0x1F6B6, 0x1F6B6), (0x1F6B7, 0x1F6B8), (0x1F6B9, 0x1F6BE), (0x1F6BF, 0x1F6BF), (0x1F6C0, 0x1F6C0),
    (0x1F6C1, 0x1F6C5), (0x1F6CB, 0x1F6CB), (0x1F6CC, 0x1F6CC), (0x1F6CD, 0x1F6CF), (0x1F6D0, 0x1F6D0),
    (0x1F6D1, 0x1F6D2), (0x1F6D5, 0x1F6D5), (0x1F6D6, 0x1F6D7), (0x1F6D8, 0x1F6D8), (0x1F6D9, 0x1F6D9),
    (0x1F6DA, 0x1F6DB), (0x1F6DC, 0x1F6DC), (0x1F6DD, 0x1F6DF), (0x1F6E0, 0x1F6E5), (0x1F6E9, 0x1F6E9),
    (0x1F6EB, 0x1F6EC), (0x1F6ED, 0x1F6EF), (0x1F6F0, 0x1F6F0), (0x1F6F3, 0x1F6F3), (0x1F6F4, 0x1F6F6),
    (0x1F6F7, 0x1F6F8), (0x1F6F9, 0x1F6F9), (0x1F6FA, 0x1F6FA), (0x1F6FB, 0x1F6FC), (0x1F6FD, 0x1F6FF),
    (0x1F7DC, 0x1F7DF), (0x1F7E0, 0x1F7EB), (0x1F7EC, 0x1F7EF), (0x1F7F0, 0x1F7F0), (0x1F80C, 0x1F80F),
    (0x1F848, 0x1F84F), (0x1F85A, 0x1F85F), (0x1F888, 0x1F88F), (0x1F8AE, 0x1F8AF), (0x1F8BC, 0x1F8BF),
    (0x1F8C2, 0x1F8CF), (0x1F8D9, 0x1F8FF), (0x1F90C, 0x1F90C), (0x1F90D, 0x1F90F), (0x1F910, 0x1F918),
    (0x1F919, 0x1F91E), (0x1F91F, 0x1F91F), (0x1F920, 0x1F927), (0x1F928, 0x1F92F), (0x1F930, 0x1F930),
    (0x1F931, 0x1F932), (0x1F933, 0x1F93A), (0x1F93C, 0x1F93E), (0x1F93F, 0x1F93F), (0x1F940, 0x1F945),
    (0x1F947, 0x1F94B), (0x1F94C, 0x1F94C), (0x1F94D, 0x1F94F), (0x1F950, 0x1F95E), (0x1F95F, 0x1F96B),
    (0x1F96C, 0x1F970), (0x1F971, 0x1F971), (0x1F972, 0x1F972), (0x1F973, 0x1F976), (0x1F977, 0x1F978),
    (0x1F979, 0x1F979), (0x1F97A, 0x1F97A), (0x1F97B, 0x1F97B), (0x1F97C, 0x1F97F), (0x1F980, 0x1F984),
    (0x1F985, 0x1F991), (0x1F992, 0x1F997), (0x1F998, 0x1F9A2), (0x1F9A3, 0x1F9A4), (0x1F9A5, 0x1F9AA),
    (0x1F9AB, 0x1F9AD), (0x1F9AE, 0x1F9AF), (0x1F9B0, 0x1F9B9), (0x1F9BA, 0x1F9BF), (0x1F9C0, 0x1F9C0),
    (0x1F9C1, 0x1F9C2), (0x1F9C3, 0x1F9CA), (0x1F9CB, 0x1F9CB), (0x1F9CC, 0x1F9CC), (0x1F9CD, 0x1F9CF),
    (0x1F9D0, 0x1F9E6), (0x1F9E7, 0x1F9FF), (0x1FA58, 0x1FA5F), (0x1FA6E, 0x1FA6F), (0x1FA70, 0x1FA73),
    (0x1FA74, 0x1FA74), (0x1FA75, 0x1FA77), (0x1FA78, 0x1FA7A), (0x1FA7B, 0x1FA7C), (0x1FA7D, 0x1FA7F),
    (0x1FA80, 0x1FA82), (0x1FA83, 0x1FA86), (0x1FA87, 0x1FA88), (0x1FA89, 0x1FA89), (0x1FA8A, 0x1FA8A),
    (0x1FA8B, 0x1FA8D), (0x1FA8E, 0x1FA8E), (0x1FA8F, 0x1FA8F), (0x1FA90, 0x1FA95), (0x1FA96, 0x1FAA8),
    (0x1FAA9, 0x1FAAC), (0x1FAAD, 0x1FAAF), (0x1FAB0, 0x1FAB6), (0x1FAB7, 0x1FABA), (0x1FABB, 0x1FABD),
    (0x1FABE, 0x1FABE), (0x1FABF, 0x1FABF), (0x1FAC0, 0x1FAC2), (0x1FAC3, 0x1FAC5), (0x1FAC6, 0x1FAC6),
    (0x1FAC7, 0x1FAC7), (0x1FAC8, 0x1FAC8), (0x1FAC9, 0x1FACB), (0x1FACC, 0x1FACC), (0x1FACD, 0x1FACD),
    (0x1FACE, 0x1FACF), (0x1FAD0, 0x1FAD6), (0x1FAD7, 0x1FAD9), (0x1FADA, 0x1FADB), (0x1FADC, 0x1FADC),
    (0x1FADD, 0x1FADD), (0x1FADE, 0x1FADE), (0x1FADF, 0x1FADF), (0x1FAE0, 0x1FAE7), (0x1FAE8, 0x1FAE8),
    (0x1FAE9, 0x1FAE9), (0x1FAEA, 0x1FAEA), (0x1FAEB, 0x1FAEB), (0x1FAEC, 0x1FAEE), (0x1FAEF, 0x1FAEF),
    (0x1FAF0, 0x1FAF6), (0x1FAF7, 0x1FAF8), (0x1FAF9, 0x1FAFA), (0x1FAFB, 0x1FAFF), (0x1FC00, 0x1FFFD),
)
_EMOJI_MODIFIER_RANGES: Tuple[Tuple[int, int], ...] = (
    (0x1F3FB, 0x1F3FF),
)
_EMOJI_ZWJ_NEIGHBOUR_RANGES: Tuple[Tuple[int, int], ...] = (
    _EXTENDED_PICTOGRAPHIC_RANGES + _EMOJI_MODIFIER_RANGES
)
# VS15 (U+FE0E, text presentation) and VS16 (U+FE0F, emoji presentation) may sit between an
# emoji base and the joiner, so both are skipped when looking for the flanking emoji.
_EMOJI_VARIATION_SELECTOR_CPS = frozenset({0xFE0E, 0xFE0F})


def _is_emoji_zwj_neighbour(char: str) -> bool:
    """True when *char* can legitimately flank a U+200D: Extended_Pictographic or Emoji_Modifier."""
    codepoint = ord(char)
    return any(lo <= codepoint <= hi for lo, hi in _EMOJI_ZWJ_NEIGHBOUR_RANGES)


def is_zwj_in_emoji_sequence(content: str, idx: int) -> bool:
    r"""True when the U+200D at *idx* joins emoji bases on both sides.

    Variation selectors are skipped on either side, so ``\U0001F3C4\u200D\u2642\uFE0F``
    (surfer) and ``\u2764\uFE0F\u200D\U0001F525`` (heart-on-fire) both qualify. One-sided
    shapes such as ``A\u200D<emoji>`` or ``<emoji>\u200DA`` are NOT sequences — they glue a
    joiner next to real text, which is the hiding trick this scanner exists to catch.
    """
    left, right = idx - 1, idx + 1
    while left >= 0 and ord(content[left]) in _EMOJI_VARIATION_SELECTOR_CPS:
        left -= 1
    while right < len(content) and ord(content[right]) in _EMOJI_VARIATION_SELECTOR_CPS:
        right += 1
    return (
        left >= 0
        and right < len(content)
        and _is_emoji_zwj_neighbour(content[left])
        and _is_emoji_zwj_neighbour(content[right])
    )


def zwj_is_emoji_only(content: str) -> bool:
    r"""True when EVERY U+200D in *content* sits inside an emoji sequence (True when there are none).

    The scanners report U+200D once per payload, so the exemption is all-or-nothing: a single
    bare joiner anywhere keeps the finding, which is the injection shape worth reporting.
    """
    return all(
        is_zwj_in_emoji_sequence(content, idx)
        for idx, char in enumerate(content)
        if char == "\u200d"
    )


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
    findings: List[str] = [
        f"invisible_unicode_U+{ord(ch):04X}"
        for ch in set(content) & INVISIBLE_CHARS
        # A U+200D that only ever joins emoji is part of the emoji, not hidden text.
        if not (ch == "\u200d" and zwj_is_emoji_only(content))
    ]
    # NFKC folds full-width / compatibility variants (ｃａｔ → cat) against homograph bypass.
    # It does NOT fold cross-script confusables (Cyrillic ``а``) — that needs a TR#39 database.
    normalised = unicodedata.normalize("NFKC", content)
    findings.extend(pid for compiled, pid in patterns if compiled.search(normalised))
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


__all__ = ["INVISIBLE_CHARS", "MAX_SCAN_CHARS", "scan_for_threats", "first_threat_message",
           "is_zwj_in_emoji_sequence", "zwj_is_emoji_only"]
