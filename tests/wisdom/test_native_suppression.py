"""Native Not Now keeps the reviewed organisation policy across offline restarts."""

from unittest.mock import Mock

import pytest

from hermes_wisdom.client import AgentLedPolicyResponse, WisdomError
from hermes_wisdom.consent import WisdomConsent
from hermes_wisdom.preferences import WisdomPreferences
from tests.wisdom.test_native_install_policy import native_install as native_install, request


def policy(days, org="org-1"):
    return AgentLedPolicyResponse(
        org_id=org, usage_evidence_window_days=7, min_aggregate_invocations=3,
        consecutive_day_usage_counts=True, repeated_edits_count=True,
        max_recommendations_per_user_per_week=3, publication_mode="moderated",
        install_popularity_threshold=10,
        notification_defaults={"skill_ready_to_share": True, "teammate_published": True,
                               "update_available": True},
        manager_review_email_cadence="daily", not_now_suppression_days=days,
        version=1, updated_by_user_id=None,
    )


@pytest.mark.parametrize("days", [3, 60])
def test_offline_deferral_uses_saved_policy_and_never_replays_after_its_expiry(native_install, days):
    service, actor, now = native_install
    service.client.agent_led_policy = Mock(return_value=policy(days))
    consent = WisdomConsent(service, clock=lambda: now[0])
    shown = request(consent, actor, "MANUAL")
    service.client.agent_led_policy.side_effect = WisdomError("offline")
    service.client.suppress_recommendation = Mock(side_effect=WisdomError("offline"))
    now[0] += 60
    restarted = WisdomConsent(service, clock=lambda: now[0])
    deferred = restarted.resolve("org-1", shown["id"], actor, "defer")
    assert deferred["suppressed_until"] == now[0] + days * 86400
    assert service.client.agent_led_policy.call_count == 1
    assert service.client.records == []
    preferences = WisdomPreferences(service, clock=lambda: now[0])
    preferences.flush("org-1")
    service.client.suppress_recommendation.assert_called_once()
    now[0] = deferred["suppressed_until"] + 1
    preferences.flush("org-1")
    service.client.suppress_recommendation.assert_called_once()
    with service.store.transaction() as db:
        assert db.execute("SELECT state FROM wisdom_preference_outbox").fetchone()[0] == "expired"


@pytest.mark.parametrize("response", [policy(3, "another-org"), {"not_now_suppression_days": 3}])
def test_unverified_policy_cannot_override_legacy_offline_fallback(native_install, response):
    service, actor, now = native_install
    service.client.agent_led_policy = Mock(return_value=response)
    consent = WisdomConsent(service, clock=lambda: now[0])
    shown = request(consent, actor, "MANUAL")
    deferred = consent.resolve("org-1", shown["id"], actor, "defer")
    assert deferred["suppressed_until"] == now[0] + 30 * 86400
    assert service.client.records == []
