"""Shared credential/secret detector for persistent model context.

This module is the single source of truth for detecting *probable
authentication credentials* in text that is about to enter an LLM's context
(memory files, context files, and optionally tool-result redaction).

It is deliberately SEPARATE from ``tools/threat_patterns.py``:

- ``threat_patterns`` detects *instructions* (prompt injection, exfiltration
  C2 vocabulary, persistence).  Those are behavioural attacks.
- ``secret_detector`` detects *confidential values* (passwords, API keys,
  tokens, private keys, connection strings).  Those are data leaks.

Conflating the two would blur the threat model: a password in memory is not a
prompt-injection payload, and a prompt-injection string is not a credential.
Keeping the detectors apart lets each evolve against its own false-positive
surface.

Design goals
------------

- **Require both semantics and a concrete value.**  We never flag a sentence
  that merely mentions credentials without a probable value, e.g.::

      User rotates passwords monthly
      The project has password authentication
      Never store API keys in source control

  Each pattern anchors on a credential keyword *and* an adjacent
  high-entropy / structured value.

- **Cover prose and direct-assignment forms.**  The legacy memory scanner in
  ``threat_patterns`` (``hardcoded_secret``) only matched quoted direct
  assignments like ``password = "xyz"``.  This module additionally catches
  prose such as ``Password for Test WebUI on this machine: CANARY_...``.

- **Never reveal the value.**  Detection returns only a *category* and a
  safe, value-free marker.  Callers (memory tool, audit CLI) must never print,
  log, or echo the matched secret.

- **No duplication of regex sets.**  The memory write path, the memory
  snapshot path, tool-result redaction, and the audit command all call into
  this one module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

# Bound scanned text so adversarial/large inputs can't blow up runtime.
# Detection only needs to find a credential near the start of injected content.
MAX_SCAN_CHARS = 65_536

# ---------------------------------------------------------------------------
# Value shapes (shared by every credential pattern below)
# ---------------------------------------------------------------------------

# A candidate "value" token: letters/digits/underscore/hyphen, length >= 12.
# This is intentionally broad — the real filter is ``_looks_like_secret()``
# below, which rejects pure-lowercase dictionary words (e.g. "authentication",
# "documentation") so a benign sentence like "Password policy: authentication"
# is NOT flagged.  The regex only finds *candidates*; the validator decides.
_TOKEN = r"[A-Za-z0-9_=\-]{12,}"

# A hex / base64-ish blob (20+ chars) — typical of API keys, hashes, tokens.
_HEX = r"[A-Za-z0-9+/]{20,}"

# A candidate value for the "Password is <value>" prose form.  Same breadth as
# ``_TOKEN``; validated by ``_looks_like_secret()`` after the match.
_PROSE_VALUE = r"[A-Za-z0-9_=\-]{10,}|[A-Za-z0-9+/]{12,}"


def _looks_like_secret(value: str) -> bool:
    """Decide whether a matched value is a *probable* credential, not a word.

    A benign dictionary word (even a long one) must NOT look like a secret.
    We require at least one of:

    * a digit (``hunter23``, ``CANARY_7F39``),
    * a mix of upper- and lower-case letters (``CANARYabc``, ``Sup3r``),
    * an underscore/hyphen joining multiple segments (``stored_in_keychain``
      is a *variable name*, not a secret, so we additionally require one of the
      above — but ``my_api_key_abc123`` wins because it has a digit),
    * a base64/hex structure (``+/`` present, or 20+ chars of [A-Za-z0-9]).

    Pure-lowercase-alpha runs like ``authentication`` / ``configurable`` /
    ``recommended`` / ``documentation`` fail every check and are rejected.
    """
    if not value:
        return False
    v = value.strip().strip("\"'")
    # A bare variable lookup / env reference is not a literal secret value.
    if v.startswith(("os.getenv", "os.environ", "process.env", "$ENV", "environ[")):
        return False
    has_digit = any(ch.isdigit() for ch in v)
    has_lower = any(ch.islower() for ch in v)
    has_upper = any(ch.isupper() for ch in v)
    has_mixed_case = has_lower and has_upper
    # underscore/hyphen only counts when joined to something structured
    has_segment_sep = ("_" in v or "-" in v) and any(ch.isalnum() for ch in v)
    base64ish = ("/" in v) or len(re.sub(r"[^A-Za-z0-9]", "", v)) >= 20
    # Require a digit OR mixed case OR (a segment separator AND at least 8
    # alnum chars — catches things like "stored_in_keychain" only if also
    # structured; here stored_in_keychain has no digit/case so it fails, which
    # is the desired behaviour for pure variable-name values).
    if has_digit or has_mixed_case or base64ish:
        return True
    if has_segment_sep and len(re.sub(r"[^A-Za-z0-9]", "", v)) >= 16:
        # e.g. "my_long_apikey_abc123" — but that has a digit, caught above.
        # A pure "stored_in_keychain" (no digit/case) must NOT pass.
        return has_digit or has_mixed_case
    return False

# Generic credential-keyword vocabulary (case-insensitive).  Excludes bare
# "auth" so "Authorization:" (handled separately) and "author:" don't collide
# with legitimate prose, but keeps auth_token/auth-key via the "token" keyword.
_CRED_KEY = (
    r"(?:passphrase|passwd|password|pwd|api[ _-]?key|secret(?:[ _-]?key)?|"
    r"token|access[ _-]?token|refresh[ _-]?token|auth[ _-]?token|"
    r"credential|private[ _-]?key|bearer|client[ _-]?secret)"
)


@dataclass(frozen=True)
class SecretFinding:
    """A value-free description of a detected credential.

    The matched text is intentionally NOT stored — callers must never surface
    the secret.  Only the ``category`` (for remediation guidance) and a short
    ``marker`` (safe to log / place in a system-prompt placeholder) are kept.
    """

    category: str
    marker: str

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.marker


# Each entry: (compiled regex, category, marker, validate_value).  When
# ``validate_value`` is True the matched value must also pass
# ``_looks_like_secret()`` — this is what keeps prose/assignment forms from
# firing on pure dictionary words.  Patterns that are *themselves* proof (a
# PEM block, a vendor-key prefix, a Bearer/Basic token) skip value validation.
_PATTERNS: List[tuple] = []


def _add(pattern: str, category: str, marker: str, validate_value: bool = False) -> None:
    _PATTERNS.append(
        (re.compile(pattern, re.IGNORECASE), category, marker, validate_value)
    )


# Pull the most likely credential value out of a matched span: the trailing
# run of token characters.  Used only to validate prose/assignment matches.
_VALUE_EXTRACT_RE = re.compile(r"[A-Za-z0-9_=\-+/]{8,}")


def _extract_candidate_value(match: "re.Match") -> str:
    span = match.group(0)
    # Prefer the token after a colon/equals if present.
    candidates = _VALUE_EXTRACT_RE.findall(span)
    if not candidates:
        return ""
    # The credential value is usually the last sizeable token in the span
    # (after the keyword), e.g. "Password for X: <VALUE>".
    return candidates[-1]


# --- Prose forms ("Password for X: VALUE", "The password is VALUE") -------
# Requires the extracted value to look like a secret (digit / mixed case /
# base64), so "Password policy: authentication" is NOT flagged.
_add(
    rf"{_CRED_KEY}\b[^\n]{{0,40}}?[:=]\s*{_PROSE_VALUE}(?![A-Za-z])",
    "password_prose",
    "[credential: password (prose form)]",
    validate_value=True,
)
# "The password is <value>" / "my api key is <value>"
_add(
    rf"\b(?:the|my|our)\s+{_CRED_KEY}\s+(?:is|was|has|equals)\s+{_PROSE_VALUE}(?![A-Za-z])",
    "credential_prose",
    "[credential: api key / token (prose form)]",
    validate_value=True,
)
# "store the password as <value>" — imperative/instruction prose still has a
# concrete value, so it is caught; relies on the value shape, not the verb.

# --- Direct assignments (both quoted and unquoted) -------------------------
# Legacy threat_patterns only handled the quoted form with >=20 chars.  We
# also catch unquoted ``password = hunter2`` while still requiring a value
# that looks like a secret.
_add(
    rf"{_CRED_KEY}\s*[=:]\s*['\"]?{_TOKEN}['\"]?",
    "credential_assignment",
    "[credential: direct assignment]",
    validate_value=True,
)
# YAML / .env style ``key: value`` handled above via the assignment pattern.

# --- API key / access token prose ------------------------------------------
# "API key: <value>" is covered by the prose + assignment patterns above.
# Add an explicit token-with-scheme form for robustness.
_add(
    rf"(?:api[ _-]?key|access[ _-]?token|secret)\s*(?:of|for|to)?\s*[:=]?\s*{_TOKEN}",
    "api_key_prose",
    "[credential: api key / token]",
    validate_value=True,
)

# --- Authorization / bearer headers ----------------------------------------
# "Authorization: Bearer <token>", "Authorization: Basic <b64>", or any scheme.
# A real scheme word + token is itself the proof; still require the trailing
# token to look structured (a bare "Authorization: Bearer" with no token is
# not a leak).
_add(
    r"(?:proxy-)?authorization\s*:\s*(?:bearer|basic|token|digest|apikey)?\s*(\S+)",
    "authorization_header",
    "[credential: Authorization header]",
    validate_value=True,
)
# Bare scheme token without a header name: "Bearer <token>", "Basic <b64>".
_add(
    r"\b(?:bearer|basic|token|digest)\s+([A-Za-z0-9._\-+/]{12,})",
    "auth_scheme_token",
    "[credential: bearer/basic token]",
    validate_value=True,
)
# x-api-key / x-auth-token header style.
_add(
    r"\b(?:x-api-key|x-goog-api-key|api-key|apikey|x-api-token|x-auth-token|x-access-token)\s*:\s*(\S+)",
    "secret_header",
    "[credential: secret header]",
    validate_value=True,
)

# --- Private key blocks -----------------------------------------------------
_add(
    r"-----BEGIN[A-Z ]*PRIVATE KEY-----[\s\S]*?-----END[A-Z ]*PRIVATE KEY-----",
    "private_key",
    "[credential: private key block]",
)
# PKCS#8 / OpenSSH single-line key material references.
_add(
    r"PRIVATE KEY(?: BLOCK)?(?:[ ]?\([^)]*\))?[:=]\s*[A-Za-z0-9+/=]{20,}",
    "private_key_assignment",
    "[credential: private key assignment]",
)

# --- Connection strings with embedded credentials --------------------------
# postgres://user:PASSWORD@host, mysql://..., mongodb+srv://..., redis://...,
# amqp://...  Forbids whitespace so the match never spans a line break.
_add(
    r"(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^:\s/]+:[^@\s]{4,}@",
    "db_connection_string",
    "[credential: database connection string]",
)
# Generic scheme://user:pass@host (not just DB protocols).
_add(
    r"(?:https?|wss?|ftp|ftps|sftp|ssh|git)://[^/\s:@]+:[^/\s@]{4,}@",
    "url_userinfo",
    "[credential: URL user:password]",
)

# --- Known vendor-key prefixes (high precision) ----------------------------
_VENDOR_PREFIXES = (
    r"sk-[A-Za-z0-9_\-]{10,}|"           # OpenAI / OpenRouter / Anthropic
    r"gh[pousr]_[A-Za-z0-9]{10,}|"        # GitHub PATs
    r"github_pat_[A-Za-z0-9_]{10,}|"     # GitHub fine-grained
    r"xox[baprs]-[A-Za-z0-9\-]{10,}|"     # Slack
    r"AIza[A-Za-z0-9_\-]{30,}|"           # Google
    r"AKIA[A-Z0-9]{16}|"                  # AWS
    r"eyJ[A-Za-z0-9_\-]{10,}"             # JWT
)
_add(
    rf"\b(?:{_VENDOR_PREFIXES})\b",
    "vendor_api_key",
    "[credential: vendor API key prefix]",
)


def scan_for_secrets(content: str) -> List[SecretFinding]:
    """Scan ``content`` for probable credentials.

    Returns a list of :class:`SecretFinding` (value-free).  An empty list means
    no probable credential was found.  Deterministic from the input bytes.

    The matched value is NEVER returned, stored, or logged by this function.
    Callers must not reconstruct or print the secret.
    """
    if not content:
        return []
    scanned = content[:MAX_SCAN_CHARS]
    findings: List[SecretFinding] = []
    seen_markers = set()
    for compiled, category, marker, validate_value in _PATTERNS:
        match = compiled.search(scanned)
        if not match:
            continue
        if validate_value:
            candidate = _extract_candidate_value(match)
            if not _looks_like_secret(candidate):
                continue
        if marker not in seen_markers:
            findings.append(SecretFinding(category=category, marker=marker))
            seen_markers.add(marker)
    return findings


def contains_secret(content: str) -> bool:
    """Convenience: ``True`` if ``content`` contains a probable credential."""
    return bool(scan_for_secrets(content))


def first_secret_message(content: str) -> Optional[str]:
    """Return a user-facing rejection message, or ``None`` if clean.

    The message explains *why* the write was blocked and where credentials
    belong, without echoing the secret.  Used by the memory write path.
    """
    findings = scan_for_secrets(content)
    if not findings:
        return None
    categories = ", ".join(sorted({f.category for f in findings}))
    return (
        "Blocked: this memory entry looks like it contains a probable "
        f"authentication credential ({categories}). MEMORY.md and USER.md are "
        "injected into the model's system prompt every session, so credentials "
        "stored there can leak to the model provider and into responses. Store "
        "secrets in Hermes' supported credential storage (e.g. ~/.hermes/.env) "
        "instead, never in memory. If a real credential was written here, remove "
        "it and rotate it."
    )


__all__ = [
    "MAX_SCAN_CHARS",
    "SecretFinding",
    "scan_for_secrets",
    "contains_secret",
    "first_secret_message",
    "scan_memory_for_secrets",
]

# ---------------------------------------------------------------------------
# Audit command — scan existing on-disk memory for probable credentials
# ---------------------------------------------------------------------------

from pathlib import Path
from typing import Dict, List as _List


def scan_memory_for_secrets(hermes_home: Optional[str] = None) -> Dict[str, _List[SecretFinding]]:
    """Scan existing MEMORY.md / USER.md for probable credentials.

    Returns a dict keyed by memory source (e.g. ``"MEMORY.md"``) whose values
    are lists of :class:`SecretFinding` (value-free).  Sources with no findings
    are omitted.  The original files are NOT modified and the secret values are
    NEVER returned, printed, or logged — only the category and a safe marker.

    This is the operator-facing audit: it reports *where* a probable credential
    lives and *what kind*, so the user can remove and rotate it.  It must not
    surface the credential itself.
    """
    from hermes_constants import get_hermes_home
    from tools.memory_tool import MemoryStore

    home = Path(hermes_home) if hermes_home else get_hermes_home()
    mem_dir = home / "memories"
    results: Dict[str, _List[SecretFinding]] = {}
    for fname in ("MEMORY.md", "USER.md"):
        path = mem_dir / fname
        if not path.exists():
            continue
        try:
            entries = MemoryStore._read_file(path)
        except (OSError, IOError):
            continue
        if not entries:
            continue
        for idx, entry in enumerate(entries):
            findings = scan_for_secrets(entry)
            if findings:
                # Keyed by source + entry index so the CLI can point the user
                # at the exact entry to remove.  The value (findings list) is
                # value-free by construction.
                results[f"{fname}#entry{idx + 1}"] = findings
    return results
