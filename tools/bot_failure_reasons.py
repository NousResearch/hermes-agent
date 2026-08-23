"""Typed failure-reason codes for bot turns and relay replies (#93091).

A closed vocabulary of machine-readable reason codes carried ALONGSIDE the
existing free-text ``error`` fields (additive schema — old consumers keep
working). Platform-side codes are assigned by the transport/relay layer;
agent-side codes are derived from raw agent/provider error text via
``classify_agent_error``.

Classifier precedence (deterministic, documented, tested):
    1. auth — an explicit ``authentication_error`` type, a 401/403 status,
       or "invalid api key" wins over everything else. Rationale: real
       provider 401 bodies (e.g. Anthropic) say "invalid, blocked or out of
       funds" — quota words inside an auth error must not misclassify it.
    2. quota   — 402 / out of funds / quota / balance.
    3. rate    — 429 / rate limit.
    4. server  — 5xx / server error / overloaded.
    5. context — context length / context_overflow / maximum context.
    6. config  — No LLM provider configured / missing config / No access token.
    7. model   — model not found / does not exist.
    8. unknown — anything else (including empty text).
"""

from __future__ import annotations

import re

# ── platform-side reason codes ───────────────────────────────────────────────
RUNTIME_OFFLINE = "runtime_offline"
QUEUED_EXPIRED = "queued_expired"
DELIVERY_TIMEOUT = "delivery_timeout"
AGENT_BLOCKED = "agent_blocked"
CANCELLED = "cancelled"

# ── agent-side reason codes ──────────────────────────────────────────────────
PROVIDER_AUTH_OR_ACCESS = "provider_auth_or_access"
PROVIDER_QUOTA_LIMIT = "provider_quota_limit"
PROVIDER_RATE_LIMIT = "provider_rate_limit"
PROVIDER_SERVER_ERROR = "provider_server_error"
CONTEXT_OVERFLOW = "context_overflow"
MISSING_CONFIG = "missing_config"
MODEL_UNAVAILABLE = "model_unavailable"
UNKNOWN = "unknown"

ALL_REASONS = frozenset(
    {
        RUNTIME_OFFLINE,
        QUEUED_EXPIRED,
        DELIVERY_TIMEOUT,
        AGENT_BLOCKED,
        CANCELLED,
        PROVIDER_AUTH_OR_ACCESS,
        PROVIDER_QUOTA_LIMIT,
        PROVIDER_RATE_LIMIT,
        PROVIDER_SERVER_ERROR,
        CONTEXT_OVERFLOW,
        MISSING_CONFIG,
        MODEL_UNAVAILABLE,
        UNKNOWN,
    }
)

#: Reasons a supervisor may retry automatically without human intervention.
AUTO_RETRYABLE = frozenset(
    {RUNTIME_OFFLINE, DELIVERY_TIMEOUT, PROVIDER_RATE_LIMIT, PROVIDER_SERVER_ERROR}
)


def is_auto_retryable(reason: str) -> bool:
    """True when ``reason`` is safe to retry automatically."""
    return reason in AUTO_RETRYABLE


# Ordered (pattern, code) rules — first match wins. See module docstring for
# the precedence rationale (auth beats quota by design).
_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"authentication_error|invalid api key"
            r"|(?:error code:?\s*|status(?:\s*code)?:?\s*|http\s*)(?:401|403)\b",
            re.IGNORECASE,
        ),
        PROVIDER_AUTH_OR_ACCESS,
    ),
    (
        re.compile(
            r"(?:error code:?\s*|status(?:\s*code)?:?\s*|http\s*)402\b"
            r"|out of funds|quota|balance",
            re.IGNORECASE,
        ),
        PROVIDER_QUOTA_LIMIT,
    ),
    (
        re.compile(
            r"(?:error code:?\s*|status(?:\s*code)?:?\s*|http\s*)429\b|rate.?limit",
            re.IGNORECASE,
        ),
        PROVIDER_RATE_LIMIT,
    ),
    (
        re.compile(
            r"(?:error code:?\s*|status(?:\s*code)?:?\s*|http\s*)5\d{2}\b"
            r"|server error|overloaded",
            re.IGNORECASE,
        ),
        PROVIDER_SERVER_ERROR,
    ),
    (
        re.compile(r"context length|context_overflow|maximum context", re.IGNORECASE),
        CONTEXT_OVERFLOW,
    ),
    (
        re.compile(
            r"no llm provider configured|missing config|no access token",
            re.IGNORECASE,
        ),
        MISSING_CONFIG,
    ),
    (
        re.compile(r"model .*(not found|does not exist)|model_not_found", re.IGNORECASE),
        MODEL_UNAVAILABLE,
    ),
)


def classify_agent_error(text: str) -> str:
    """Map raw agent/provider error text to a closed reason code.

    First matching rule in ``_RULES`` wins; anything unmatched (or empty)
    is ``unknown``. Auth intentionally outranks quota: a 401 body that also
    mentions "out of funds" is still an auth/access failure.
    """
    raw = str(text or "")
    if not raw.strip():
        return UNKNOWN
    for pattern, code in _RULES:
        if pattern.search(raw):
            return code
    return UNKNOWN
