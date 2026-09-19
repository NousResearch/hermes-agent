"""Output redaction for every Rob read-only operator tool.

Applies to command output, ``docker inspect`` dumps, file content, HTTP
diagnostics, DB diagnostics and logs alike — anything that could reach a
transcript. Name-pattern redaction alone is proven insufficient: a real
credential leak happened this engagement when a ``docker inspect`` env
dump's ``DATABASE_URL=postgresql://user:PASSWORD@host/db`` line was missed
because the filter only matched *variable names*, never the embedded
credential inside a URI *value*. This module redacts by both name and
value pattern for exactly that reason.

Every pattern here is applied to attacker-influenced text (HTTP bodies,
log output, DB output), so each one is bounded/linear — no nested
unbounded quantifiers, no per-start-position backtracking over the rest
of the line. See the comments on ``_redact_name_value_pairs`` and
``_COOKIE_PATTERN`` for the closed O(n^2) shapes, and on ``_JWT_PATTERN``
for the backtracking-elimination change, in the consolidated security
pass.
"""

from __future__ import annotations

import re

# Name/key detection is a linear SCAN, not a regex, on purpose. The
# previous `_NAME_KEY_SEP_PATTERN` was effectively
# `[A-Za-z0-9_.-]*SECRET_WORD[A-Za-z0-9_.-]*`: an unbounded greedy prefix
# tried from every one of n start positions, each attempt backtracing the
# whole remaining line when no `[:=]` separator ever followed — O(n^2) on
# a hostile 200 KB text body (measured ~63s through the registered
# `rob_http_probe` tool). No real identifier needs an unbounded prefix,
# so the scan below finds each secret-word occurrence once, extends the
# key left/right over identifier characters, and moves on — linear in
# input size with identical redaction semantics (see
# `_redact_name_value_pairs`).
_SECRET_WORD_ALTERNATION = re.compile(
    r"TOKEN|PASSWORD|PASSWD|SECRET|KEY|CREDENTIAL", re.IGNORECASE
)

_ASCII_IDENT_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_"
)
_KEY_CHARS = _ASCII_IDENT_CHARS | frozenset(".-")
_QUOTES = frozenset("\"'")


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

    This is the same match semantics as the previous
    ``_NAME_KEY_SEP_PATTERN`` (leftmost match, greedy key extension over
    ``[A-Za-z0-9_.-]``, optional surrounding quotes) implemented as a
    linear scan: the left-extension loop is the same greedy prefix, but
    each input position is visited once instead of rescanned from every
    possible start. A key with no following ``[:=]`` separator makes every
    later secret word in the SAME identifier run fail identically, so the
    scan skips the whole run at once — this keeps dense word runs
    (``"TOKEN" * 50_000``) linear instead of quadratic.
    """
    out: list[str] = []
    pos = 0
    skip_until = 0
    n = len(line)
    rstrip_len = len(line.rstrip("\n"))
    for m in _SECRET_WORD_ALTERNATION.finditer(line):
        if m.start() < skip_until:
            continue  # already consumed by a previous match's span/skipped run

        # Greedy key extension leftward: identical to the regex's
        # `[A-Za-z0-9_.-]*` prefix. Because this consumes every alnum/
        # underscore char, the char before the key can never be one, which
        # is exactly the regex's `(?<![A-Za-z0-9_])` guard.
        key_start = m.start()
        while key_start > 0 and line[key_start - 1] in _KEY_CHARS:
            key_start -= 1
        lead_quote = (
            line[key_start - 1]
            if key_start > 0 and line[key_start - 1] in _QUOTES
            else None
        )

        key_end = m.end()
        while key_end < n and line[key_end] in _KEY_CHARS:
            key_end += 1
        if key_end < n and line[key_end] in _QUOTES:
            key_end += 1  # the key's own closing quote in `"NAME": "value"`

        sep_idx = key_end
        while sep_idx < n and line[sep_idx] in " \t":
            sep_idx += 1
        if sep_idx >= n or line[sep_idx] not in ":=":
            # No separator after this key — not a name=value pair. Every
            # later secret-word occurrence inside this same identifier run
            # reaches the same run end and fails the same way, so skip the
            # whole run rather than re-extending left across it. (skip_until
            # is separate from `pos` on purpose: nothing has been emitted
            # for the skipped region yet.)
            skip_until = max(skip_until, key_end)
            continue
        sep_idx += 1
        while sep_idx < n and line[sep_idx] in " \t":
            sep_idx += 1

        value_quote = line[sep_idx] if sep_idx < n and line[sep_idx] in _QUOTES else None
        value_start = sep_idx + (1 if value_quote else 0)
        terminator = value_quote or lead_quote
        if terminator:
            close_idx = _find_unescaped_char(line, terminator, value_start)
            value_end = close_idx if close_idx != -1 else rstrip_len
        else:
            value_end = rstrip_len
        if value_end <= value_start:
            continue  # empty value (e.g. NAME="") — nothing to redact
        out.append(line[pos:value_start])
        out.append(REDACTED)
        pos = value_end
        skip_until = max(skip_until, value_end)
    out.append(line[pos:])
    return "".join(out)


# scheme://user:password@host or scheme://:password@host (Redis-style,
# empty username) — redact only the credential portion, keep the
# scheme/host visible since that's the useful diagnostic part. The
# scheme's repetition is bounded ({0,15}: no real URI scheme is anywhere
# near 16 chars) so a long scheme-shaped blob with no `://` can never
# trigger per-position backtracking.
_URI_CREDENTIAL_PATTERN = re.compile(
    r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]{0,15}://)"
    r"(?P<user>[^:@/\s]*):(?P<pass>[^@/\s]+)"
    r"(?P<at>@)"
)

_AUTH_HEADER_PATTERN = re.compile(
    r"(?i)(Authorization\s*:\s*)(Bearer|Basic|Digest|Token)\s+([A-Za-z0-9._~+/=-]+)"
)

# Well-known token-shaped prefixes. Each alternative is anchored on a
# distinct literal prefix and ends in a greedy quantifier with no trailing
# constraint beyond `\b`, which is always satisfied at the end of the
# character class / line — no backtracking path.
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
# positives on ordinary dotted version-ish strings. The quantifiers are
# LAZY: each segment expands forward one character at a time until the
# next `.` (or, for the last segment, a word boundary) matches, so there
# is no backtracking at all — expansion is forward-only, each position
# visited a constant number of times regardless of how dotted runs
# overlap. Lazy is also the correct span: exactly three segments, not the
# longest possible run.
_JWT_PATTERN = re.compile(
    r"\beyJ[A-Za-z0-9_-]{5,}?\.[A-Za-z0-9_-]{5,}?\.[A-Za-z0-9_-]{5,}?\b"
)

# Cookie / session-id style: `key=<long opaque token>` inside a Cookie
# header or Set-Cookie line. The attribute-name span between the header
# name and `=` is bounded (lazy, {1,64}): the previous unbounded
# `[^=;\s]+` made repeated `Cookie:` prefixes with no `=` ever following
# rescan the rest of the line from every occurrence — O(n^2). Real
# cookie/attribute names are short; the value span is greedy with no
# trailing constraint, so it never backtracks.
_COOKIE_PATTERN = re.compile(
    r"(?i)((?:Cookie|Set-Cookie)\s*:\s*[^=;\s]{1,64}?=)([^;\s]{16,})"
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
    # Same rule as the name scan: any key CONTAINING a secret-shaped word
    # counts, not just one ending with "_" + the word — a bare, unseparated
    # name like "PGPASSWORD" or "APIKEY" is exactly as sensitive as
    # "MCP_TOKEN_SIGNING_SECRET" and must not slip through for lack of an
    # underscore.
    upper = str(key).upper()
    return any(word in upper for word in _SECRET_KEY_BARE_WORDS)
