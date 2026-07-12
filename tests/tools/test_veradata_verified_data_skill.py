"""Tests for the veradata-verified-data optional skill.

Uses the free trial endpoint (X-TRIAL header, no wallet, no payment, no real
PII). Verifies the skill's documented contract: a trial rates query returns
structured JSON with a numeric rate.
"""
import os
import httpx
import pytest

RATES_URL = "https://api.veradata.dev/rates"
SAMPLE = {"country": "CO", "signals": ["usd_cop"]}

pytestmark = pytest.mark.skipif(
    os.environ.get("HERMES_SKIP_REMOTE_SKILL_TESTS") == "1",
    reason="remote skill test disabled via HERMES_SKIP_REMOTE_SKILL_TESTS",
)


@pytest.fixture(scope="module")
def client():
    with httpx.Client(timeout=60) as c:
        yield c


def test_rates_trial_returns_json(client):
    """Trial rates query must return 200 + structured JSON with a rate."""
    r = client.post(RATES_URL, json=SAMPLE, headers={"X-TRIAL": "true"})
    assert r.status_code == 200, f"rates returned {r.status_code}: {r.text[:200]}"
    body = r.json()
    assert "usd_cop" in body or "trm_official" in body
    assert isinstance(body.get("usd_cop", body.get("trm_official", 0)), (int, float))


def test_rates_trial_requires_no_wallet(client):
    """Trial works with only the X-TRIAL header, no tx_hash or wallet."""
    r = client.post(RATES_URL, json=SAMPLE, headers={"X-TRIAL": "true"})
    assert r.status_code == 200
    assert "sanitized_content" not in r.json()  # not a PII skill; sanity check
