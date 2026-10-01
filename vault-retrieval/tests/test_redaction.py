"""Tests for secret / PII redaction (redaction.py).

Covers acceptance test #13 (redaction from output AND from JSONL log).
The redactor must:
  - recognise credential-shaped strings (api_key, token, password)
  - recognise PII-shaped strings (email, phone-like, Thai national ID)
  - count redactions by class
  - replace secrets/PII with a stable token (so output is unambiguous)
  - be case-insensitive on common patterns
  - never emit the original secret in its replacement
"""
from __future__ import annotations

import pytest


from vault_retrieval.redaction import (
    Redactor,
    redact,
    RedactionCounts,
    SECRET_PATTERNS,
    PII_PATTERNS,
)


class TestRedactSecrets:
    def test_openai_key_redacted(self):
        text = "key is sk-abcdef0123456789abcdef0123456789abcd"
        out = redact(text)
        assert "sk-abcdef" not in out
        assert "[REDACTED:secret]" in out

    def test_github_pat_redacted(self):
        text = "token ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        out = redact(text)
        assert "ghp_aaaaaaaa" not in out
        assert "[REDACTED:secret]" in out

    def test_bearer_token_redacted(self):
        text = "Authorization: Bearer abc123.def456-ghi789"
        out = redact(text)
        assert "abc123.def456" not in out
        assert "[REDACTED:secret]" in out


class TestRedactPII:
    def test_email_redacted(self):
        text = "contact alice@example.com please"
        out = redact(text)
        assert "alice@example.com" not in out
        assert "[REDACTED:pii]" in out

    def test_thai_phone_redacted(self):
        text = "call 081-234-5678"
        out = redact(text)
        assert "081-234-5678" not in out

    def test_credit_card_like_redacted(self):
        text = "card 4111 1111 1111 1111"
        out = redact(text)
        assert "4111 1111 1111 1111" not in out


class TestRedactionCounts:
    def test_secrets_and_pii_counted_separately(self):
        text = "key sk-abcdef0123456789abcdef0123456789abcd and alice@example.com"
        _, counts = Redactor().redact_with_counts(text)
        assert counts.secrets >= 1
        assert counts.pii >= 1
        assert counts.total() >= 2

    def test_no_matches_yields_zero_counts(self):
        _, counts = Redactor().redact_with_counts("just some ordinary text")
        assert counts.secrets == 0
        assert counts.pii == 0
        assert counts.total() == 0


class TestStability:
    def test_replacement_is_stable_token(self):
        text1 = "sk-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        text2 = "sk-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        out1 = redact(text1)
        out2 = redact(text2)
        # Both should be replaced with the SAME generic token (not the secret).
        assert out1 == out2
        assert "[REDACTED:secret]" in out1
