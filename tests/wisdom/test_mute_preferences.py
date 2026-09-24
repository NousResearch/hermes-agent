from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_wisdom.client import WisdomError, WisdomMuteResponse
from hermes_wisdom.preferences import WisdomPreferences
from hermes_wisdom.store import WisdomStore
from tests.wisdom.test_preferences import REF, preferences  # noqa: F401


def acknowledgement(org="org", **kwargs):
    duration = kwargs.get("duration")
    days = {"1_day": 1, "1_week": 7, "30_days": 30}.get(duration)
    until = (
        (
            datetime.fromisoformat(kwargs["requested_at"].replace("Z", "+00:00"))
            + timedelta(days=days)
        ).isoformat()
        if days
        else None
    )
    return WisdomMuteResponse(
        org_id=org,
        muted=duration is not None,
        duration=duration,
        muted_until=until,
        forever=duration == "forever",
        revision=kwargs["expected_revision"] + 1,
        mutation_id=kwargs["mutation_id"],
        updated_at=kwargs["requested_at"],
    )


def configure(service):
    service.client.set_recommendation_mute.side_effect = lambda duration, **kwargs: (
        acknowledgement(duration=duration, **kwargs)
    )


def test_mutation_sync_is_leased_and_idempotent_across_connections(preferences):
    p, service, now = preferences
    configure(service)
    pending = p.stage_mute("org", "1_week", expected_revision=7)
    assert pending["preference_sync"] == "pending"
    service.client.set_recommendation_mute.assert_not_called()
    other = WisdomPreferences(
        SimpleNamespace(store=WisdomStore(p.store.path.parent), client=service.client),
        clock=lambda: now[0],
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda pref: pref.flush_mute("org"), [p, other]))
    service.client.set_recommendation_mute.assert_called_once_with(
        "1_week",
        expected_revision=7,
        mutation_id=pending["mutation_id"],
        requested_at="1970-01-01T00:16:40.000Z",
    )
    assert p.mute_status("org")["preference_sync"] == "synced"


def test_lost_response_retries_identical_choice_and_expiry(preferences):
    p, service, now = preferences
    p.stage_mute("org", "1_day", expected_revision=0)
    service.client.set_recommendation_mute.side_effect = TimeoutError
    p.flush_mute("org")
    first = service.client.set_recommendation_mute.call_args
    now[0] += 61
    configure(service)
    p.flush_mute("org")
    assert service.client.set_recommendation_mute.call_args == first
    assert p.mute_status("org")["preference_sync"] == "synced"


def test_fractional_clock_uses_the_wire_timestamp_for_expiry(preferences):
    p, service, now = preferences
    now[0] = 1000.9999998
    configure(service)
    p.stage_mute("org", "1_day", expected_revision=0)
    p.flush_mute("org")
    assert p.mute_status("org")["preference_sync"] == "synced"


def test_failed_mute_suppresses_locally_but_failed_unmute_cannot_override_server(
    preferences,
):
    p, service, _ = preferences
    p.stage_mute("org", "forever", expected_revision=0)
    service.client.set_recommendation_mute.side_effect = TimeoutError
    assert p.check("org", [REF])["muted"]
    p.stage_mute("org", None, expected_revision=0)
    service.client.recommendation_mute.return_value = WisdomMuteResponse(
        org_id="org", muted=True, duration="forever", muted_until=None, forever=True
    )
    result = p.check("org", [REF])
    assert result["muted"]
    assert result["mute_sync"]["requested_duration"] is None
    assert result["mute_sync"]["preference_sync"] == "pending"


@pytest.mark.parametrize("status,state", [(409, "conflict"), (410, "expired")])
def test_rejected_choice_is_not_rebased_automatically(preferences, status, state):
    p, service, now = preferences
    p.stage_mute("org", "forever", expected_revision=0)
    service.client.set_recommendation_mute.side_effect = WisdomError(
        "refused", status=status
    )
    p.flush_mute("org")
    now[0] += 600
    p.flush_mute("org")
    assert service.client.set_recommendation_mute.call_count == 1
    assert p.mute_status("org")["preference_sync"] == state
    assert not p.check("org", [REF])["muted"]


def test_expired_choice_is_never_sent(preferences):
    p, service, now = preferences
    p.stage_mute("org", "forever", expected_revision=0)
    now[0] += 86401
    p.flush_mute("org")
    service.client.set_recommendation_mute.assert_not_called()
    assert p.mute_status("org")["preference_sync"] == "expired"


def test_crash_on_last_attempt_leaves_terminal_status_after_lease_expiry(preferences):
    p, service, now = preferences
    p.stage_mute("org", "forever", expected_revision=0)
    with p.store.transaction() as db:
        db.execute(
            "UPDATE wisdom_mute_outbox SET state='syncing',attempts=3,lease_until=?",
            (now[0] + 60,),
        )
    now[0] += 61
    p.flush_mute("org")
    service.client.set_recommendation_mute.assert_not_called()
    assert p.mute_status("org")["preference_sync"] == "failed"


def test_response_to_superseded_choice_cannot_overwrite_new_local_intent(preferences):
    p, service, _ = preferences
    first = p.stage_mute("org", "forever", expected_revision=0)

    def finish_old(duration, **kwargs):
        p.stage_mute("org", None, expected_revision=1)
        return acknowledgement(duration=duration, **kwargs)

    service.client.set_recommendation_mute.side_effect = finish_old
    p.flush_mute("org")
    current = p.mute_status("org")
    assert current["mutation_id"] != first["mutation_id"]
    assert current["requested_duration"] is None
    assert current["preference_sync"] == "pending"


def test_preference_owner_change_cannot_sync_prior_accounts_intent(preferences):
    p, service, _ = preferences
    p.stage_mute("org", "forever", expected_revision=0)
    service.client.identity = {"owner": "another-user"}
    p.flush_mute("org")
    service.client.set_recommendation_mute.assert_not_called()
    assert p.mute_status("org") is None


def test_mutation_response_must_match_exact_request(preferences):
    p, service, _ = preferences
    p.stage_mute("org", "forever", expected_revision=0)
    service.client.set_recommendation_mute.side_effect = lambda duration, **kwargs: (
        acknowledgement(org="other", duration=duration, **kwargs)
    )
    p.flush_mute("org")
    assert p.mute_status("org")["preference_sync"] == "pending"


def test_native_command_uses_current_revision_and_keeps_pending_state_visible(
    preferences,
):
    p, service, _ = preferences
    service.require_setup = Mock()
    service.client.recommendation_mute.return_value = WisdomMuteResponse(
        org_id="org",
        muted=False,
        duration=None,
        muted_until=None,
        forever=False,
        revision=9,
    )
    service.client.set_recommendation_mute.side_effect = TimeoutError
    result = p.native_mute_command("1w")
    assert result["gateway_available"]
    assert result["sync"]["preference_sync"] == "pending"
    assert result["sync"]["requested_duration"] == "1_week"
    assert not result["mute"]["muted"]
    assert (
        service.client.set_recommendation_mute.call_args.kwargs["expected_revision"]
        == 9
    )
    assert p.native_mute_command("status")["sync"] == result["sync"]


def test_native_command_refuses_old_gateway_without_revision_support(preferences):
    p, service, _ = preferences
    service.require_setup = Mock()
    with pytest.raises(WisdomError, match="updated"):
        p.native_mute_command("forever")
    assert p.mute_status("org") is None
    service.client.set_recommendation_mute.assert_not_called()


@pytest.mark.parametrize(
    "patch",
    [{"muted": False}, {"forever": False}, {"muted_until": "1970-01-01T01:00:00Z"}],
)
def test_false_success_acknowledgement_stays_pending(preferences, patch):
    p, service, _ = preferences
    p.stage_mute("org", "forever", expected_revision=0)
    service.client.set_recommendation_mute.side_effect = lambda duration, **kwargs: (
        acknowledgement(duration=duration, **kwargs).model_copy(update=patch)
    )
    p.flush_mute("org")
    assert p.mute_status("org")["preference_sync"] == "pending"
