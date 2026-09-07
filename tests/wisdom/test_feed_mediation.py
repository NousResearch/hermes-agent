import json
from unittest.mock import Mock

import pytest

from hermes_wisdom.client import WisdomNotFound
from hermes_wisdom.consent import ConsentActor
from hermes_wisdom.mediation import WisdomMediation
from hermes_wisdom.mediation_store import LEASE_SECONDS
from hermes_wisdom.store import WisdomStore


@pytest.fixture
def mediation(tmp_path, monkeypatch):
    store = WisdomStore(tmp_path / "wisdom")
    store.activate_installation_identity("installation", "org")
    service = Mock(store=store)
    service.client.identity = {"owner": "owner"}
    service.client.display_org_id = "org"
    service.local_candidate_events.return_value = []
    service.notifications.return_value = {"events": []}
    service.version_detail.return_value = {
        "skill": {"id": "skill", "state": "active"},
        "version": {
            "version": 2,
            "content_hash": "sha256:" + "a" * 64,
            "author_description": "Exact release description",
            "security_check": {"status": "advisory"},
            "professionalism_check": {"status": "unavailable"},
            "system_spec": {"runtime": "sandbox"},
            "scan": {"unnecessary_raw_scan": "not for model"},
            "unknown_future_field": "not for model",
        },
        "local_installation": {"target_path": "/private/path"},
        "local_compatibility": {"outcome": "compatible"},
    }
    now = [1000.0]
    instance = WisdomMediation(service, clock=lambda: now[0])
    actor = ConsentActor("session", "telegram", "user", "chat")

    def register():
        instance.queue.register_session(
            "org",
            session_key="session",
            session_id="session",
            platform="telegram",
            actor_id="user",
            private=True,
            available=True,
            user_activity=True,
            address=actor.address,
        )

    register()
    monkeypatch.setattr(
        "hermes_wisdom.weekly_queue.enqueue_weekly_review", lambda _: None
    )
    monkeypatch.setattr("hermes_wisdom.mediation.delivery_mode", lambda: "agent")
    # Preference gates have separate integration tests; this fixture isolates
    # feed classification and the real SQLite ownership/retirement boundary.
    monkeypatch.setattr(instance, "_eligible_jobs", lambda org, jobs: jobs)
    return instance, actor, now, register


def event(category="new_skill", version=2):
    return {
        "event_id": "event",
        "skill_id": "skill",
        "version": version,
        "category": category,
        "kind": "updated",
        "skill_name": "helpful",
        "source_event_ids": ["event"],
        "security_check": {"status": "pass", "summary": "LATEST NOT EXACT"},
        "professionalism_check": {"status": "pass"},
    }


def enqueue(instance, category="new_skill"):
    instance.service.notifications.return_value = {"events": [event(category)]}
    assert instance.ingest() == "org"
    return instance.queue.assessments("org")[0]


def ready(instance):
    job = instance.queue.claim("org", "session")[0]
    advice = {
        "title": "Arrival",
        "explanation": "Worth reviewing",
        "relevance": "digest",
    }
    assert instance.queue.save_advice("org", job["id"], job["lease_token"], advice)
    return {
        "assessment": {**job, "state": "ready"},
        "advice": advice,
        "interaction": None,
    }


@pytest.mark.parametrize(
    "category,kind",
    [
        ("new_skill", "skill"),
        ("update_available", "skill"),
        ("installed", "notice"),
        ("updated", "notice"),
        ("publication_decision", "notice"),
        ("unavailable", "notice"),
    ],
)
def test_only_actionable_arrivals_become_recommendations(mediation, category, kind):
    instance, actor, *_ = mediation
    job = enqueue(instance, category)
    assert job["reference"]["kind"] == kind
    instance.ingest()
    assert len(instance.queue.assessments("org")) == 1
    if kind == "notice":
        instance.consent.present = Mock(side_effect=AssertionError("not an action"))
        results = instance.prepare(
            "org",
            actor,
            runtime={},
            history=[],
            assessor=lambda evidence, **kw: {
                item["assessment_id"]: {
                    "title": "Outcome",
                    "explanation": "Recorded",
                    "relevance": "recommend",
                }
                for item in evidence
            },
        )
        assert len(results) == 1 and results[0]["interaction"] is None
        instance.service.version_detail.assert_not_called()


@pytest.mark.parametrize("version", [None, 0, -1, True, "2"])
def test_malformed_or_absent_version_never_prepares_install(mediation, version):
    instance, *_ = mediation
    instance.service.notifications.return_value = {"events": [event(version=version)]}
    instance.ingest()
    assert instance.queue.assessments("org")[0]["reference"]["kind"] == "notice"


@pytest.mark.parametrize(
    "state",
    [
        "pending",
        "assessing",
        "ready",
        "delivering",
        "delivered",
        "delivery_uncertain",
        "retired",
    ],
)
def test_legacy_classification_repair_fences_workers_without_replaying_delivery(
    mediation, state
):
    instance, *_ = mediation
    row = enqueue(instance)
    with instance.service.store.transaction() as db:
        db.execute(
            "UPDATE wisdom_assessment SET state=?,lease_token='old',lease_until=2000,advice_json=? WHERE id=?",
            (state, json.dumps({"title": "old"}), row["id"]),
        )
    instance.service.notifications.return_value = {"events": [event("installed")]}
    instance.ingest()
    updated = instance.queue.assessments("org")[0]
    assert updated["id"] == row["id"] and updated["reference"]["kind"] == "notice"
    if state in {"delivering", "delivered", "delivery_uncertain", "retired"}:
        assert updated["state"] == state
    else:
        assert updated["state"] == "pending" and updated.get("advice") is None
        assert not instance.queue.save_advice(
            "org", row["id"], "old", {"title": "late"}
        )


def test_exact_version_projection_excludes_latest_checks_raw_scan_and_local_paths(
    mediation,
):
    instance, *_ = mediation
    job = enqueue(instance)
    evidence = instance.inspect("org", job)
    facts = evidence["facts"]
    assert facts["version"]["security_check"]["status"] == "advisory"
    assert facts["version"]["professionalism_check"]["status"] == "unavailable"
    assert facts["version"]["author_description"] == "Exact release description"
    assert facts["version"]["system_spec"] == {"runtime": "sandbox"}
    assert "LATEST NOT EXACT" not in json.dumps(evidence)
    assert "/private/path" not in json.dumps(evidence)
    assert "not for model" not in json.dumps(evidence)


@pytest.mark.parametrize(
    "failure,retired",
    [
        ("archived", True),
        ("taken_down", True),
        ("missing", True),
        ("network", False),
        ("wrong_skill", False),
        ("wrong_version", False),
        ("unknown_state", False),
    ],
)
def test_delivery_revalidates_current_authority_without_discarding_advice_on_errors(
    mediation, failure, retired
):
    instance, *_ = mediation
    enqueue(instance)
    item = ready(instance)
    detail = instance.service.version_detail.return_value
    if failure in {"archived", "taken_down", "unknown_state"}:
        detail["skill"]["state"] = failure
    elif failure == "missing":
        instance.service.version_detail.side_effect = WisdomNotFound("gone")
    elif failure == "network":
        instance.service.version_detail.side_effect = TimeoutError
    elif failure == "wrong_skill":
        detail["skill"]["id"] = "other-org-skill"
    else:
        detail["version"]["version"] = 3
    assert instance.begin_delivery("org", [item]) == []
    row = instance.queue.assessments("org")[0]
    assert row["state"] == ("retired" if retired else "ready")
    assert row["advice"] == item["advice"] and row["delivered_at"] is None
    assert row["lease_token"] is None


@pytest.mark.parametrize("installed_version", [2, 3])
def test_completed_install_retires_arrival_before_any_model_call(
    mediation, tmp_path, installed_version
):
    instance, actor, *_ = mediation
    enqueue(instance, "update_available")
    instance.service.store.record_install({
        "skill_id": "skill",
        "org_id": "org",
        "slug": "helpful",
        "version": installed_version,
        "content_hash": "content",
        "baseline": {},
        "target_path": str(tmp_path / "managed"),
        "update_mode": "MANUAL",
    })
    assessor = Mock(side_effect=AssertionError("already installed"))
    assert (
        instance.prepare("org", actor, runtime={}, history=[], assessor=assessor) == []
    )
    assert instance.queue.assessments("org")[0]["state"] == "retired"
    instance.service.version_detail.assert_not_called()


def test_stale_owner_cannot_retire_reclaimed_arrival(mediation):
    instance, _, now, register = mediation
    enqueue(instance)
    old = instance.queue.claim("org", "session")[0]
    now[0] += LEASE_SECONDS + 1
    register()
    current = instance.queue.claim("org", "session")[0]
    assert not instance.queue.retire("org", old)
    assert instance.queue.assessments("org")[0]["lease_token"] == current["lease_token"]


def test_withdrawn_arrival_invalidates_pending_consent_but_does_not_mark_feed_read(
    mediation,
):
    instance, actor, *_ = mediation
    enqueue(instance)
    item = ready(instance)
    instance.service.store.persist_feed_page(
        [event()], next_cursor="cursor", cadences={}, now="2026-09-07T00:00:00Z"
    )
    instance.service.install_plan.return_value = {
        "skill_id": "skill",
        "slug": "helpful",
        "version": 2,
        "allowed": True,
        "receipt": "wip_one",
        "content_hash": "content",
        "manifest_hash": "manifest",
        "compatibility": {"outcome": "compatible"},
    }
    interaction = instance.consent.present("org", item["assessment"]["id"], actor)
    assert interaction["state"] == "pending"
    instance.service.version_detail.return_value["skill"]["state"] = "archived"
    assert instance.begin_delivery("org", [item]) == []
    assert (
        instance.consent.resolve("org", interaction["id"], actor, "confirm")["state"]
        == "stale"
    )
    instance.service.install_apply.assert_not_called()
    assert len(instance.service.store.feed_events(unseen_only=True)) == 1


def test_old_informational_prompt_is_repaired_even_when_feed_refresh_was_skipped(
    mediation,
):
    instance, *_ = mediation
    notification = event("installed")
    instance.queue.enqueue(
        "org",
        "feed:event",
        {
            "kind": "skill",
            "event_id": "event",
            "skill_id": "skill",
            "version": 2,
            "notification": notification,
        },
    )
    item = ready(instance)
    assert instance.begin_delivery("org", [item]) == []
    assert instance.queue.assessments("org")[0]["reference"]["kind"] == "notice"
    instance.service.version_detail.assert_not_called()


def test_transient_lookup_failure_retries_without_spending_model_attempt(mediation):
    instance, actor, now, register = mediation
    enqueue(instance)
    instance.service.version_detail.side_effect = TimeoutError
    assessor = Mock(return_value={})
    assert (
        instance.prepare("org", actor, runtime={}, history=[], assessor=assessor) == []
    )
    assert instance.queue.assessments("org")[0]["attempts"] == 0
    assessor.assert_not_called()
    now[0] += 61
    register()
    instance.service.version_detail.side_effect = None
    assessor.side_effect = lambda evidence, **kw: {
        item["assessment_id"]: {
            "title": "Restored",
            "explanation": "Useful",
            "relevance": "digest",
        }
        for item in evidence
    }
    result = instance.prepare("org", actor, runtime={}, history=[], assessor=assessor)
    assert len(result) == 1
    assert result[0]["advice"]["title"] == "Restored"
    assessor.assert_called_once()


def test_inspection_through_real_service_and_typed_version_response(
    mediation, monkeypatch
):
    from hermes_wisdom.client import VersionDetail
    from hermes_wisdom.service import WisdomService

    instance, *_ = mediation
    job = enqueue(instance)
    client = Mock()
    client.version.return_value = VersionDetail.model_validate({
        "skill": {"id": "skill", "state": "active", "takedown_generation": 0},
        "version": {
            "version": 2,
            "content_hash": "sha256:" + "a" * 64,
            "author_description": "This exact version",
            "system_spec": None,
            "security_check": {"status": "advisory"},
            "professionalism_check": {"status": "unavailable"},
        },
    })
    monkeypatch.setattr("hermes_wisdom.service.portal_base_url", lambda: None)
    service = WisdomService(store=instance.service.store, client=client)
    evidence = WisdomMediation(service).inspect("org", job)
    client.version.assert_called_once_with("skill", 2)
    assert evidence["facts"]["version"]["author_description"] == "This exact version"
    assert evidence["facts"]["version"]["security_check"]["status"] == "advisory"
    assert "local_installation" not in evidence["facts"]


@pytest.mark.parametrize("active", [True, False])
def test_policy_uses_active_installation_not_a_retired_ledger_entry(
    mediation, monkeypatch, tmp_path, active
):
    from hermes_wisdom.agent_led.policy import AgentLedPolicy

    instance, *_ = mediation
    enqueue(instance)
    item = ready(instance)
    instance.service.store.record_install({
        "skill_id": "skill",
        "org_id": "org",
        "slug": "helpful",
        "version": 1,
        "content_hash": "content",
        "baseline": {},
        "target_path": str(tmp_path / "managed"),
        "update_mode": "MANUAL",
    })
    if not active:
        instance.service.store.deactivate_install("skill")
    monkeypatch.setattr(
        "hermes_wisdom.agent_led.policy.load_policy",
        lambda **kw: AgentLedPolicy(
            enabled=True,
            notification_defaults={
                "skill_ready_to_share": False,
                "teammate_published": True,
                "update_available": False,
            },
        ),
    )
    monkeypatch.setattr(
        "hermes_wisdom.preferences.WisdomPreferences.check",
        lambda *args: {
            "available": True,
            "muted": False,
            "suppressed": {},
        },
    )
    result = WisdomMediation._eligible_jobs(instance, "org", [item["assessment"]])
    assert bool(result) is not active
