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

_SECRET_WORD = r"(?:TOKEN|PASSWORD|PASSWD|SECRET|KEY|CREDENTIAL)"

_NAME_KEY_SEP_PATTERN = re.compile(
    # A key is any identifier CONTAINING a secret-shaped word segment,
    # anywhere in the line — not just at the start of the (stripped) line.
    # An earlier `^`-anchored version only matched when the secret-shaped
    # name was the first token on the line, which misses the exact shape a
    # `docker inspect` env dump actually produces (`    "MCP_TOKEN=x",` —
    # the name is preceded by indentation AND a quote, never at line start)
    # as well as any secret embedded mid-line (a JSON body, a log line with
    # a prefix). A one-character negative lookbehind keeps this from
    # matching in the middle of a longer identifier at a word boundary,
    # while intentionally still matching substrings like "PGPASSWORD" or
    # "APIKEY" that have no separating underscore ("SOMETOKENISH" is NOT
    # exempted by this — it still matches, deliberately: over-redacting an
    # identifier that merely contains a secret word is far safer than
    # missing a real bare-word secret name).
    #
    # This pattern locates the key+separator only — NOT the value. How far
    # the value extends is decided procedurally in
    # `_redact_name_value_pairs`, based on how the value is quoted, rather
    # than by matching up to a fixed delimiter set (space/comma/etc) here:
    # a real secret value can legitimately contain any of those characters
    # (a generated passphrase, a base64/URL-safe token used in a query
    # string), and a delimiter-bounded value group truncates the redaction
    # right there and leaks the rest of the secret in plaintext — a real
    # regression an earlier version of this exact fix introduced.
    r"(?P<lead_quote>[\"'])?"
    r"(?<![A-Za-z0-9_])"
    r"(?P<key>[A-Za-z0-9_.-]*" + _SECRET_WORD + r"[A-Za-z0-9_.-]*)"
    r"(?:[\"'])?"  # the key's OWN closing quote in `"NAME": "value"` — not captured, just skipped
    r"(?P<sep>\s*[:=]\s*)"
    r"(?P<value_quote>[\"'])?",
    re.IGNORECASE,
)


def _find_unescaped_char(line: str, char: str, start: int) -> int:
    """Find the first occurrence of ``char`` at/after ``start`` that is NOT
    escaped by a preceding backslash (an odd number of consecutive
    preceding backslashes means it IS escaped, matching standard JSON/shell
    escaping — ``\\"`` is an escaped quote, ``\\\\"`` is an escaped
    backslash followed by a real quote). A naive ``str.find`` would treat
    the first literal quote character as the closing one even when it's
    escaped inside a JSON string, cutting the value short and leaving
    everything after it — including the rest of the secret — in plaintext.
    Returns -1 if no unescaped occurrence exists."""
    idx = start
    while True:
        idx = line.find(char, idx)
        if idx == -1:
            return -1
        backslashes = 0
        j = idx - 1
        while j >= 0 and line[j] == "\\":
            backslashes += 1
            j -= 1
        if backslashes % 2 == 0:
            return idx
        idx += 1


def _redact_name_value_pairs(line: str) -> str:
    """Redact every secret-shaped ``NAME=value`` / ``"NAME": "value"``
    occurrence in ``line``, choosing how far the value extends based on
    how it's quoted rather than stopping at the first punctuation
    character:

    - ``"NAME": "value"`` / ``NAME: 'value'`` — value ends at the matching
      (unescaped) quote that opened right after the separator.
    - ``"NAME=value"`` (the docker-inspect env-array shape: the whole
      ``KEY=value`` pair sits inside one JSON string) — value ends at the
      same quote that opened right before the key, i.e. the redaction is
      bounded to the JSON string the pair is embedded in, even if that
      swallows harmless trailing text inside the same string.
    - a bare ``NAME=value`` with no quoting context at all — value runs to
      the end of the line, since there is no other reliable terminator and
      a truncated redaction that leaks the value's tail is worse than
      over-redacting trailing text on the same line.
    """
    out: list[str] = []
    pos = 0
    for m in _NAME_KEY_SEP_PATTERN.finditer(line):
        if m.start() < pos:
            continue  # already consumed by a previous match's value span
        value_start = m.end()
        terminator = m.group("value_quote") or m.group("lead_quote")
        if terminator:
            close_idx = _find_unescaped_char(line, terminator, value_start)
            value_end = close_idx if close_idx != -1 else len(line.rstrip("\n"))
        else:
            value_end = len(line.rstrip("\n"))
        if value_end <= value_start:
            continue  # empty value (e.g. NAME="") — nothing to redact
        out.append(line[pos:value_start])
        out.append(REDACTED)
        pos = value_end
    out.append(line[pos:])
    return "".join(out)

# scheme://user:password@host or scheme://:password@host (Redis-style,
# empty username) — redact only the credential portion, keep the
# scheme/host visible since that's the useful diagnostic part.
_URI_CREDENTIAL_PATTERN = re.compile(
    # The scheme's repeating group was unbounded (`*`), which is
    # catastrophic on a long run of scheme-shaped characters with no
    # `://` ever following (e.g. a long alnum blob in an HTTP response
    # body or a log line) — the greedy match consumes to end-of-line and
    # then backtracks one character at a time from every one of n start
    # positions, an O(n^2) blowup confirmed live: ~30s of CPU on a 200,000
    # -char adversarial line through the real, registered `rob_http_probe`
    # tool. No real URI scheme is anywhere near this long (the longest in
    # common use, e.g. "postgresql"/"mongodb+srv", is under 16 chars) —
    # bounding the repetition removes the pathological case entirely
    # without narrowing what actually gets redacted.
    r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]{0,15}://)"
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
    line = _redact_name_value_pairs(line)
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
    # Same rule as _NAME_PATTERN: any key CONTAINING a secret-shaped word
    # counts, not just one ending with "_" + the word — a bare, unseparated
    # name like "PGPASSWORD" or "APIKEY" is exactly as sensitive as
    # "MCP_TOKEN_SIGNING_SECRET" and must not slip through for lack of an
    # underscore.
    upper = str(key).upper()
    return any(word in upper for word in _SECRET_KEY_BARE_WORDS)
