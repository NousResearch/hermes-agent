"""Output redaction for every Rob read-only operator tool.

Applies to command output, ``docker inspect`` dumps, file content, HTTP
diagnostics, DB diagnostics and logs alike — anything that could reach a
transcript. Name-pattern redaction alone is proven insufficient: a real
credential leak happened this engagement when a ``docker inspect`` env
dump's ``DATABASE_URL=postgresql://user:PASSWORD@host/db`` line was missed
because the filter only matched *variable names*, never the embedded
credential inside a URI *value*. This module redacts by both name and
value pattern for exactly that reason.
"""

from __future__ import annotations

import re

_NAME_PATTERN = re.compile(
    # A key is a secret-shaped name either bare (`PASSWORD=x`) or with any
    # number of underscore-separated segments before the suffix
    # (`MCP_TOKEN_SIGNING_SECRET=x`) — the naive `_SUFFIX` form used
    # earlier only matched when a literal underscore preceded the suffix,
    # missing the common bare-word case entirely (caught by
    # tests/tools/test_secret_redaction.py's own regression suite).
    r"^(?P<key>(?:[A-Za-z0-9.-]+_)*(?:TOKEN|PASSWORD|SECRET|KEY|PASSWD|CREDENTIAL))"
    r"(?P<sep>\s*[:=]\s*)"
    r"(?P<value>.*)$",
    re.IGNORECASE,
)

# scheme://user:password@host or scheme://:password@host (Redis-style,
# empty username) — redact only the credential portion, keep the
# scheme/host visible since that's the useful diagnostic part.
_URI_CREDENTIAL_PATTERN = re.compile(
    r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*://)"
    r"(?P<user>[^:@/\s]*):(?P<pass>[^@/\s]+)"
    r"(?P<at>@)"
)

_AUTH_HEADER_PATTERN = re.compile(
    r"(?i)(Authorization\s*:\s*)(Bearer|Basic|Digest|Token)\s+([A-Za-z0-9._~+/=-]+)"
)

# Well-known token-shaped prefixes.
_TOKEN_PREFIX_PATTERN = re.compile(
    r"\b("
    r"gh[pousr]_[A-Za-z0-9]{20,}"  # GitHub PAT/OAuth/user/server/refresh tokens
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|sk-[A-Za-z0-9]{20,}"  # OpenAI-style secret keys
    r"|sk-ant-[A-Za-z0-9-]{20,}"  # Anthropic-style secret keys
    r"|xox[baprs]-[A-Za-z0-9-]{10,}"  # Slack tokens
    r"|AKIA[0-9A-Z]{16}"  # AWS access key id
    r"|glpat-[A-Za-z0-9_-]{20,}"  # GitLab PAT
    r")\b"
)

# JWT-shaped: three base64url segments separated by dots. Deliberately
# requires all three segments to be reasonably long to avoid false
# positives on ordinary dotted version-ish strings.
_JWT_PATTERN = re.compile(
    r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\b"
)

# Cookie / session-id style: `key=<long opaque token>` inside a Cookie
# header or Set-Cookie line.
_COOKIE_PATTERN = re.compile(
    r"(?i)((?:Cookie|Set-Cookie)\s*:\s*[^=;\s]+=)([^;\s]{16,})"
)

REDACTED = "[REDACTED]"


def redact_text(text: str) -> str:
    """Redact secret-shaped substrings anywhere in ``text``. Never raises —
    a redaction bug must never crash the caller and risk the *unredacted*
    text being surfaced by a fallback path instead."""
    if not text:
        return text
    try:
        out_lines = []
        for line in text.splitlines(keepends=True):
            out_lines.append(_redact_line(line))
        return "".join(out_lines)
    except Exception:
        # Fail closed: if redaction itself errors, never return the
        # original text — better a loud, useless placeholder than a
        # silent leak.
        return "[REDACTION_ERROR — output withheld]"


def _redact_line(line: str) -> str:
    m = _NAME_PATTERN.match(line.strip("\n").lstrip())
    if m and m.group("value").strip():
        prefix_len = len(line) - len(line.lstrip())
        newline = "\n" if line.endswith("\n") else ""
        return line[:prefix_len] + m.group("key") + m.group("sep") + REDACTED + newline

    line = _URI_CREDENTIAL_PATTERN.sub(
        lambda mo: f"{mo.group('scheme')}{mo.group('user')}:{REDACTED}{mo.group('at')}", line
    )
    line = _AUTH_HEADER_PATTERN.sub(lambda mo: f"{mo.group(1)}{mo.group(2)} {REDACTED}", line)
    line = _TOKEN_PREFIX_PATTERN.sub(REDACTED, line)
    line = _JWT_PATTERN.sub(REDACTED, line)
    line = _COOKIE_PATTERN.sub(lambda mo: f"{mo.group(1)}{REDACTED}", line)
    return line


def redact_mapping(data: dict) -> dict:
    """Redact a shallow or nested dict/list structure in place-equivalent
    fashion (returns a new structure; never mutates the input)."""
    return _redact_value(data)


def _redact_value(value):
    if isinstance(value, dict):
        return {k: (REDACTED if _is_secret_key(k) else _redact_value(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_value(v) for v in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


_SECRET_KEY_BARE_WORDS = ("TOKEN", "PASSWORD", "SECRET", "KEY", "PASSWD", "CREDENTIAL")


def _is_secret_key(key: str) -> bool:
    # Same rule as _NAME_PATTERN: bare ("PASSWORD") or underscore-suffixed
    # ("MCP_TOKEN_SIGNING_SECRET") both count — a dict key that IS exactly
    # one of the bare words, or ends with "_" + one of them.
    upper = str(key).upper()
    return any(upper == word or upper.endswith("_" + word) for word in _SECRET_KEY_BARE_WORDS)
