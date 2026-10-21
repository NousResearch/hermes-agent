"""Cron prompt threat scanning: the small user-authored prompt gets the strict pattern set;
the assembled prompt (with skill bodies) gets only prose-proof directives."""

import logging
import re

# Single source of truth shared with the install-time scanner (skills_guard): a narrower
# cron-local copy once let obfuscated directives slip past this runtime tripwire.
from tools.threat_patterns import INVISIBLE_CHARS as _CRON_INVISIBLE_CHARS

# Logger parity with the origin module (these functions used to log there).
logger = logging.getLogger("tools.cronjob_tools")

# Strict patterns — user prompt only. A directive-shaped cron prompt has no business
# containing `cat ~/.hermes/.env` or `rm -rf /`; there it is a smoking gun, not prose.
# Two threat surfaces, two scanners: 1. `_scan_cron_prompt()` runs against this at create/update time and as
# a runtime defense-in-depth. 2. Assembled prompt that includes loaded skill content (large markdown bodies,
# often security docs, postmortems, runbooks discussing attack patterns in PROSE). Reusing the strict
# patterns here false-positives every time a skill *describes* a command — see #3968 follow-up: the
# `hermes-agent-dev` skill contains a security postmortem mentioning `cat ~/.hermes/.env`, which tripped
# `read_secrets` and silently killed all PR-scout jobs. Skill bodies are user-curated and scanned at install
# time by `skills_guard.py`. The runtime cron scan only needs to catch the patterns whose phrasing does NOT
# survive normal English prose: classic prompt-injection directives ("ignore previous instructions",
# "disregard your rules"), deception directives, and invisible unicode. `_scan_cron_skill_assembled()` runs
# against the assembled prompt with this tighter pattern set. Both scanners share the invisible-unicode
# check and the GitHub Authorization header exemption.
_CRON_THREAT_PATTERNS = [
    (r'ignore\s+(?:\w+\s+)*(?:previous|all|above|prior)\s+(?:\w+\s+)*instructions', "prompt_injection"),
    (r'do\s+not\s+tell\s+the\s+user', "deception_hide"),
    (r'system\s+prompt\s+override', "sys_prompt_override"),
    (r'disregard\s+(your|all|any)\s+(instructions|rules|guidelines)', "disregard_rules"),
    (r'cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass|id_rsa|id_ed25519|id_ecdsa)', "read_secrets"),
    (r'authorized_keys', "ssh_backdoor"), (r'/etc/sudoers|visudo', "sudoers_mod"),
    (r'rm\s+-rf\s+/', "destructive_root_rm"),
]

# Looser set for the assembled prompt: command-shape patterns are dropped because skill
# markdown (postmortems, runbooks) legitimately *describes* those commands and skill bodies
# are vetted at install time — only unambiguous injection directives remain.
_CRON_SKILL_ASSEMBLED_PATTERNS = _CRON_THREAT_PATTERNS[:4]

_CRON_SECRET_VAR_RE = r'\$\{?\w*(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)\w*\}?'
# Obvious leak paths only: secret in the destination URL, in a POST/form body, or in an
# Authorization header to an arbitrary host.
_CRON_EXFIL_COMMAND_PATTERNS = [
    (rf'curl\s+[^\n]*https?://[^\s"\'`]*{_CRON_SECRET_VAR_RE}', "exfil_curl_url"),
    (rf'wget\s+[^\n]*https?://[^\s"\'`]*{_CRON_SECRET_VAR_RE}', "exfil_wget_url"),
    (rf'curl\s+[^\n]*(?:--data(?:-raw|-binary|-urlencode)?|-d|--form|-F)\s+[^\n]*{_CRON_SECRET_VAR_RE}', "exfil_curl_data"),
    (rf'wget\s+[^\n]*--post-(?:data|file)=[^\n]*{_CRON_SECRET_VAR_RE}', "exfil_wget_post"),
    (rf'curl\s+[^\n]*(?:-H|--header)\s+["\']Authorization:\s*(?:Bearer|token)\s+{_CRON_SECRET_VAR_RE}["\']', "exfil_curl_auth_header"),
]

# U+200D (ZWJ) is a required part of many emoji sequences (👨‍👩‍👧, 🏳️‍🌈): block it
# between plain text, allow it inside an emoji grapheme cluster.
_EMOJI_NEIGHBOUR_CP_RANGES = ((0x1F000, 0x1FFFF), (0x2600, 0x27BF), (0x2300, 0x23FF), (0x1F1E6, 0x1F1FF), (0x20E3, 0x20E3))
_VARIATION_SELECTOR_CP = 0xFE0F


def _zwj_has_emoji_neighbour(text: str, idx: int) -> bool:
    """True when the ZWJ at text[idx] sits between emoji codepoints (skipping VS16)."""
    left = idx - 1
    while left >= 0 and ord(text[left]) == _VARIATION_SELECTOR_CP:
        left -= 1
    right = idx + 1
    while right < len(text) and ord(text[right]) == _VARIATION_SELECTOR_CP:
        right += 1
    if left < 0 or right >= len(text):
        return False
    return all(
        any(lo <= ord(text[pos]) <= hi for lo, hi in _EMOJI_NEIGHBOUR_CP_RANGES) for pos in (left, right)
    )


def _strip_cron_safe_constructs(prompt: str) -> str:
    """Scrub the bundled GitHub skill's `Authorization: token $GITHUB_TOKEN` + api.github.com
    curl so it doesn't trip the auth-header exfil rule.

    re.sub scrubs EVERY occurrence. The trailing ``[^\\s;&|$`]*`` consumes only the URL path —
    never separators or subshell openers — so a payload smuggled onto the same line still gets
    scanned. Host must be exactly api.github.com followed by ``/``, whitespace, quote, or end:
    lookalike authorities (api.github.com.evil.com, api.github.com@evil.com) fall through.
    """
    return re.sub(
        rf'curl\s+[^\n;&|$`]*(?:-H|--header)\s+["\']Authorization:\s*token\s+{_CRON_SECRET_VAR_RE}["\']'
        r'\s+["\']?https://api\.github\.com(?::\d+)?(?:/|\s|$|["\'])[^\s;&|$`]*',
        'curl https://api.github.com/user',
        prompt,
        flags=re.IGNORECASE,
    )


def _strip_invisible_unicode(prompt: str) -> tuple[str, list[str]]:
    """Strip invisible-unicode chars, keeping ZWJ inside legitimate emoji.

    Returns ``(cleaned, sorted U+XXXX labels removed)``. The skills-attached path sanitizes
    (a stray zero-width space in vetted skill content must not permanently kill the job).
    """
    if not prompt:
        return prompt, []
    removed: set[str] = set()
    cleaned: list[str] = []
    for idx, ch in enumerate(prompt):
        if ch in _CRON_INVISIBLE_CHARS and not (ch == '\u200d' and _zwj_has_emoji_neighbour(prompt, idx)):
            removed.add(f"U+{ord(ch):04X}")
            continue
        cleaned.append(ch)
    return ''.join(cleaned), sorted(removed)


def _first_pattern_error(text: str, *pattern_sets) -> str:
    for patterns in pattern_sets:
        for pattern, pid in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return (
                    f"Blocked: prompt matches threat pattern '{pid}'. Cron prompts must not "
                    "contain injection or exfiltration payloads."
                )
    return ""


def _scan_cron_prompt(prompt: str) -> str:
    """Strict scan of the USER-SUPPLIED prompt (create/update + runtime defense-in-depth).
    Returns an error string when blocked, else "". Invisible unicode is reported first, in
    ``_CRON_INVISIBLE_CHARS`` order (emoji ZWJ allowed)."""
    prompt_to_scan = _strip_cron_safe_constructs(prompt)
    removed = set(_strip_invisible_unicode(prompt_to_scan)[1])
    for char in _CRON_INVISIBLE_CHARS:
        if f"U+{ord(char):04X}" in removed:
            return f"Blocked: prompt contains invisible unicode U+{ord(char):04X} (possible injection)."
    return _first_pattern_error(prompt_to_scan, _CRON_THREAT_PATTERNS, _CRON_EXFIL_COMMAND_PATTERNS)


def _scan_cron_skill_assembled(assembled: str) -> tuple[str, str]:
    """Loose scan of the ASSEMBLED prompt (skill content included). Invisible unicode is
    SANITIZED (stripped + logged), not blocked — the hard block stays on raw user prompts,
    the actual injection surface. Returns ``(cleaned_prompt, error)``, error "" when passed."""
    cleaned, removed = _strip_invisible_unicode(assembled)
    if removed:
        logger.warning(
            "Cron skill-assembled prompt: stripped %d invisible-unicode "
            "char(s) (%s) from vetted skill content",
            len(removed), ", ".join(removed),
        )
    return cleaned, _first_pattern_error(_strip_cron_safe_constructs(cleaned), _CRON_SKILL_ASSEMBLED_PATTERNS)
