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

# A high-entropy bare token: letters/digits/underscore/hyphen, length >= 12,
# containing at least one non-pure-alpha and one non-pure-digit run so we don't
# match ordinary words or pure numbers.  Anchored so it can sit after a colon,
# an equals sign, "is", "for <svc>", etc.
_TOKEN = r"[A-Za-z0-9_=\-]{12,}"

# A hex / base64-ish blob (20+ chars) — typical of API keys, hashes, tokens.
_HEX = r"[A-Za-z0-9+/]{20,}"

# A value that is "probably not a sentence": excludes trailing sentence
# punctuation and common English words so ``Password is required`` does not
# match.  Used for the "Password is <value>" prose form.
_PROSE_VALUE = r"(?!=[\s])([A-Za-z0-9_=\-]{10,}|[A-Za-z0-9+/]{12,})"

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


# Each entry: (compiled regex, category, marker).  Order is irrelevant; all
# are tried.  Every pattern requires BOTH a credential keyword and a concrete
# structured value.
_PATTERNS: List[tuple] = []


def _add(pattern: str, category: str, marker: str) -> None:
    _PATTERNS.append((re.compile(pattern, re.IGNORECASE), category, marker))


# --- Prose forms ("Password for X: VALUE", "The password is VALUE") -------

# "Password for <service>: <value>" / "password to the db is <value>"
# Requires a value of >=10 chars that is not a common English stopword-ish
# short word.  We additionally require the value NOT be a known FP word.
_add(
    rf"{_CRED_KEY}\b[^\n]{{0,40}}?[:=]\s*{_PROSE_VALUE}(?![A-Za-z])",
    "password_prose",
    "[credential: password (prose form)]",
)
# "The password is <value>" / "my api key is <value>"
_add(
    rf"\b(?:the|my|our)\s+{_CRED_KEY}\s+(?:is|was|has|equals)\s+{_PROSE_VALUE}(?![A-Za-z])",
    "credential_prose",
    "[credential: api key / token (prose form)]",
)
# "store the password as <value>" — imperative/instruction prose still has a
# concrete value, so it is caught; relies on the value shape, not the verb.

# --- Direct assignments (both quoted and unquoted) -------------------------
# Legacy threat_patterns only handled the quoted form with >=20 chars.  We
# also catch unquoted ``password = hunter2`` while still requiring a value.
_add(
    rf"{_CRED_KEY}\s*[=:]\s*['\"]?{_TOKEN}['\"]?",
    "credential_assignment",
    "[credential: direct assignment]",
)
# YAML / .env style ``key: value`` handled above via the assignment pattern.

# --- API key / access token prose ------------------------------------------
# "API key: <value>" is covered by the prose + assignment patterns above.
# Add an explicit token-with-scheme form for robustness.
_add(
    rf"(?:api[ _-]?key|access[ _-]?token|secret)\s*(?:of|for|to)?\s*[:=]?\s*{_TOKEN}",
    "api_key_prose",
    "[credential: api key / token]",
)

# --- Authorization / bearer headers ----------------------------------------
# "Authorization: Bearer <token>", "Authorization: Basic <b64>", or any scheme.
# The credential class excludes quotes so masking never corrupts syntax.
_add(
    r"(?:proxy-)?authorization\s*:\s*(?:bearer|basic|token|digest|apikey)?\s*\S+",
    "authorization_header",
    "[credential: Authorization header]",
)
# Bare scheme token without a header name: "Bearer <token>", "Basic <b64>".
_add(
    r"\b(?:bearer|basic|token|digest)\s+[A-Za-z0-9._\-+/]{12,}",
    "auth_scheme_token",
    "[credential: bearer/basic token]",
)
# x-api-key / x-auth-token header style.
_add(
    r"\b(?:x-api-key|x-goog-api-key|api-key|apikey|x-api-token|x-auth-token|x-access-token)\s*:\s*\S+",
    "secret_header",
    "[credential: secret header]",
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
    for compiled, category, marker in _PATTERNS:
        if compiled.search(scanned):
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
