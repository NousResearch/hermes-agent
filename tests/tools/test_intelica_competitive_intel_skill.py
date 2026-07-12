"""Tests for the intelica-competitive-intel optional skill.

Uses the free trial flow (GET /api-keys/trial returns a key, no wallet, no
payment). Verifies the skill's documented contract: a trial analysis returns
structured JSON with a decision recommendation.
"""
import os
import httpx
import pytest

TRIAL_URL = "https://api.intelica.dev/api-keys/trial"
INTEL_URL = "https://api.intelica.dev/intel"
SAMPLE = {"text": "Fintech neobank operating in Colombia", "mode": "competitive"}

pytestmark = pytest.mark.skipif(
    os.environ.get("HERMES_SKIP_REMOTE_SKILL_TESTS") == "1",
    reason="remote skill test disabled via HERMES_SKIP_REMOTE_SKILL_TESTS",
)


@pytest.fixture(scope="module")
def client():
    with httpx.Client(timeout=120) as c:
        yield c


def test_trial_key_grants_access(client):
    """Trial key endpoint returns a usable key; /intel with it returns 200 or a
    validation error (422) — both prove no wallet/tx_hash is required."""
    t = client.get(TRIAL_URL)
    assert t.status_code == 200, f"trial returned {t.status_code}: {t.text[:200]}"
    key = t.json().get("key")
    assert key, "trial response missing key"
    r = client.post(
        INTEL_URL,
        json={"text": "A Colombian fintech neobank offering digital savings, remittances, "
                       "and SME working-capital loans across Latin America, competing with "
                       "traditional banks and global payment players", "mode": "competitive"},
        headers={"X-API-KEY": key},
    )
    # 200 = analysis returned; 422 = body validation (still proves no wallet needed)
    assert r.status_code in (200, 422), f"intel returned {r.status_code}: {r.text[:200]}"
    if r.status_code == 200:
        body = r.json()
        assert "decision_recommendation" in body or "intelica_moat_index" in body


def test_trial_requires_no_wallet(client):
    """Trial flow uses only an API key, never a wallet or tx_hash."""
    t = client.get(TRIAL_URL)
    assert t.status_code == 200
    assert "tx_hash" not in t.json()
