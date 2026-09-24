import json
from types import SimpleNamespace

import pytest

from hermes_wisdom.local_security import prepared_security_check
from hermes_wisdom.review_presentation import full_review_text
from hermes_wisdom.consent import WisdomConsent
from hermes_wisdom.mediation_view import advice_view


@pytest.mark.parametrize("in_description", [False, True])
@pytest.mark.parametrize("content,key", [
    ("-----BEGIN PRIVATE KEY-----", "private_keys"),
    ("ghp_" + "a" * 36, "live_credentials"),
    ("password=" + "a" * 24, "secret_assignments"),
])
def test_local_scan_blocks_credentials_without_echoing_them(content, key, in_description):
    result = prepared_security_check(
        [{"path": "SKILL.md", "content_utf8": "Safe" if in_description else content}],
        content if in_description else "Safe",
        {"guard": {"allowed": True, "findings": []}},
    )
    assert result["status"] == "blocked" and result["upload_allowed"] is False
    row = next(row for row in result["checks"] if row["key"] == key)
    assert row["status"] == "blocked" and row["finding_count"] == 1
    assert content not in json.dumps(result)


def test_local_guard_failure_is_not_a_pass():
    for allowed in (False, None):
        result = prepared_security_check([], "Safe", {"guard": {"allowed": allowed}})
        assert result["upload_allowed"] is False
        assert result["status"] == ("blocked" if allowed is False else "unavailable")


def test_local_checklist_does_not_claim_gateway_checks_have_run():
    result = prepared_security_check([], "Safe", {"guard": {"allowed": True}})
    text = full_review_text(result, {"status": "pass"})
    assert "✅ Private keys" in text
    assert "⏳ Organization policy: Pending" in text
    assert "⏳ Personal information: Pending" in text
    assert "Required Gateway check after you authorize upload; before publication." in text
    assert "No issues detected by local security checks." in text


@pytest.mark.parametrize("allowed", [True, False])
def test_approval_card_surfaces_local_results_and_preserves_blocking(allowed, tmp_path):
    from hermes_wisdom.store import WisdomStore

    check = prepared_security_check([], "Safe", {"guard": {"allowed": allowed}})
    consent = WisdomConsent(SimpleNamespace(store=WisdomStore(tmp_path / "wisdom")))
    interaction = consent.project({
        "id": "consent", "assessment_id": "assessment", "operation": "publish",
        "state": "pending", "expires_at": 9999999999,
        "plan": {"allowed": check["upload_allowed"], "security_check": check},
    })
    item = {
        "assessment": {"reference": {"kind": "candidate"}},
        "advice": {"title": "Test skill", "explanation": "Review before sharing.",
                   "relevance": "recommend"},
        "interaction": interaction,
    }
    compact = advice_view([item]).items[0]
    assert check["summary"] in compact.detail
    assert "Private keys" not in compact.detail
    expanded = advice_view([item], checks_expanded=True).items[0]
    assert "✅ Private keys" in expanded.detail
    assert "⏳ Organization policy: Pending" in expanded.detail
    assert ("confirm" in interaction["actions"]) is allowed
