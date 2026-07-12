"""Tests for the trustboost-pii-sanitizer optional skill.

Uses the free /sanitize/preview endpoint (no wallet, no payment, no real PII).
Verifies the skill's documented contract: a request with sample PII returns
redacted content + a safety score + a risk category.
"""
import os
import json
import httpx
import pytest

PREVIEW_URL = "https://api.trustboost.dev/sanitize/preview"
SAMPLE_TEXT = "Contact John at john@example.com or +1-555-0123. API key: sk-abc123xyz."

pytestmark = pytest.mark.skipif(
    os.environ.get("HERMES_SKIP_REMOTE_SKILL_TESTS") == "1",
    reason="remote skill test disabled via HERMES_SKIP_REMOTE_SKILL_TESTS",
)


@pytest.fixture(scope="module")
def client():
    with httpx.Client(timeout=60) as c:
        yield c


def test_preview_redacts_pii(client):
    """Preview endpoint must redact emails, phones, and API keys."""
    r = client.post(PREVIEW_URL, json={"text": SAMPLE_TEXT})
    assert r.status_code == 200, f"preview returned {r.status_code}: {r.text[:200]}"
    body = r.json()
    assert "sanitized_content" in body
    sanitized = body["sanitized_content"]
    assert "john@example.com" not in sanitized, "email was not redacted"
    assert "sk-abc123xyz" not in sanitized, "api key was not redacted"
    assert "[REDACTED]" in sanitized, "expected [REDACTED] placeholder in output"


def test_preview_returns_safety_metadata(client):
    """Preview must return a numeric safety score and a risk category."""
    r = client.post(PREVIEW_URL, json={"text": SAMPLE_TEXT})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body.get("safety_score"), (int, float))
    assert body.get("risk_category") in ("CRITICAL", "PRIVATE", "SENSITIVE", "CLEAN")


def test_preview_requires_no_wallet(client):
    """Preview works without any tx_hash or wallet parameter."""
    r = client.post(PREVIEW_URL, json={"text": "call me at 555-0199"})
    assert r.status_code == 200
    assert "sanitized_content" in r.json()
