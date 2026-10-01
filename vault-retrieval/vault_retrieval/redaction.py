"""Secret and PII redaction.

Used to scrub extracted evidence text and to count redactions for the
metadata-only query log. The redactor replaces matches with a stable
generic token (``[REDACTED:secret]`` / ``[REDACTED:pii]``) — it does NOT
emit the original value. Pattern coverage is intentionally conservative:
we redact what we recognise, and let humans review the rest.

The redactor is deliberately NOT deterministic in the sense of being
plugin-hash-stable across versions; what matters is that no original
secret ever appears in output. The agent.redact module in core already
maintains a richer registry; this one is intentionally a smaller,
plugin-private copy so the plugin does not depend on agent.redact at
load time (the plugin may be loaded before agent.redact has registered
patterns). See acceptance test #13.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Tuple


# Common credential shapes we always redact. Each entry: (label, pattern).
# Patterns are deliberately conservative — false positives are fine, false
# negatives are not.
SECRET_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("openai_key", re.compile(r"sk-[A-Za-z0-9]{20,}")),
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("github_pat", re.compile(r"ghp_[A-Za-z0-9]{20,}")),
    ("github_oauth", re.compile(r"gho_[A-Za-z0-9]{20,}")),
    ("slack_token", re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}")),
    ("stripe_live", re.compile(r"sk_live_[A-Za-z0-9]{16,}")),
    ("stripe_test", re.compile(r"sk_test_[A-Za-z0-9]{16,}")),
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("google_api_key", re.compile(r"AIza[0-9A-Za-z\-_]{35}")),
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}")),
    ("bearer", re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{16,}")),
    ("password_in_url", re.compile(r"(?i)(?:password|passwd|pwd)=[^\s&'\"<>]{4,}")),
    ("api_key_in_url", re.compile(r"(?i)(?:api[_-]?key|apikey)=[^\s&'\"<>]{8,}")),
    ("private_key_block", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
)


PII_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")),
    # Thai mobile: 0[6-9]X-XXX-XXXX or 0[6-9]XXXXXXXX (10 digits starting 06-09)
    ("thai_mobile", re.compile(r"(?:\+66|0)[6-9](?:[\- ]?\d){8}")),
    ("thai_national_id", re.compile(r"\b\d{1}[\s\-]?\d{4}[\s\-]?\d{5}[\s\-]?\d{2}[\s\-]?\d{1}\b")),
    ("credit_card_like", re.compile(r"\b(?:\d[ \-]?){13,16}\d\b")),
    ("us_ssn_like", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
)


@dataclass(frozen=True)
class RedactionCounts:
    secrets: int = 0
    pii: int = 0

    def total(self) -> int:
        return self.secrets + self.pii


SECRET_TOKEN = "[REDACTED:secret]"
PII_TOKEN = "[REDACTED:pii]"


class Redactor:
    """A stateless redactor instance. Safe to share across threads."""

    def redact(self, text: str) -> str:
        out = text
        for _label, pat in SECRET_PATTERNS:
            out = pat.sub(SECRET_TOKEN, out)
        for _label, pat in PII_PATTERNS:
            out = pat.sub(PII_TOKEN, out)
        return out

    def redact_with_counts(self, text: str) -> Tuple[str, RedactionCounts]:
        s_count = 0
        p_count = 0
        out = text
        for _label, pat in SECRET_PATTERNS:
            new_out, n = pat.subn(SECRET_TOKEN, out)
            s_count += n
            out = new_out
        for _label, pat in PII_PATTERNS:
            new_out, n = pat.subn(PII_TOKEN, out)
            p_count += n
            out = new_out
        return out, RedactionCounts(secrets=s_count, pii=p_count)


_default_redactor = Redactor()


def redact(text: str) -> str:
    """Module-level convenience: redact with the default redactor."""
    return _default_redactor.redact(text)


def redact_with_counts(text: str) -> Tuple[str, RedactionCounts]:
    """Module-level convenience: redact + counts."""
    return _default_redactor.redact_with_counts(text)
