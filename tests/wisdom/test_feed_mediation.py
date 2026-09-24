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
    from hermes_cli.config import save_config
    from tests.wisdom.local_auth import authorize_local

    authorize_local(monkeypatch, "org")
    save_config({"wisdom": {"enabled": True, "disclosure_acknowledged_at": "fixture"}})
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


@pytest.mark.parametrize("review_status", ["pass", "advisory", "unavailable"])
def test_qualified_candidate_uses_professionalism_not_installation_assessor(
    mediation, monkeypatch, review_status
):
    from hermes_wisdom.mediation_view import advice_view, delivery_groups

    instance, actor, _, _ = mediation
    instance.service.local_candidate_events.return_value = [
        {
            "id": "qualified",
            "organization_id": "org",
            "session_id": "session",
            "skill_id": "local",
            "content_hash": "hash",
            "qualification": "high_usage",
        }
    ]
    instance.ingest()
    instance.service._candidate_event_context.return_value = (
        instance.service.local_candidate_events.return_value[0],
        "local",
        "hash",
        "My skill",
    )
    instance.service.finish_candidate_professionalism_review.return_value = {
        "status": review_status
    }
    security = {"source": "local_preflight", "status": "pass", "local_status": "pass"}
    instance.service.candidate_security_check.return_value = security
    present = Mock(
        return_value={
            "id": "consent",
            "operation": "share",
            "facts": {
                "slug": "local",
                "source_hash": "hash",
                "professionalism_check": {"status": review_status},
                "security_check": security,
            },
            "actions": ["defer", "inspect", "confirm"],
        }
    )
    monkeypatch.setattr(instance.consent, "present", present)
    assessor = Mock(
        side_effect=AssertionError("Publishing must not use the installation assessor")
    )
    items = instance.prepare("org", actor, runtime={"model": "test-model", "provider": "test-provider"}, history=[], assessor=assessor)
    assert len(items) == 1
    assert items[0]["advice"]["assessment_kind"] == "qualification"
    assessor.assert_not_called()
    instance.service.finish_candidate_professionalism_review.assert_called_once_with(
        skill_id="local", content_hash="hash"
    )
    present.assert_called_once()
    groups = delivery_groups(items)
    assert groups == [items]
    view = advice_view(groups[0])
    assert [a.label for a in view.items[0].actions] == [
        "Show checks",
        "Not Now",
        "Review first",
        "Share",
    ]
    assert view.items[0].actions[-1].primary
    assert "consistently across many days" in view.to_text()
    assert (
        "Safe to share at work" if review_status == "pass"
        else "Needs a look before sharing at work" if review_status == "advisory"
        else review_status.capitalize()
    ) in view.to_text()
    assert "Hermes recommendation:" not in view.to_text()
    assert "✅ Security check (local preflight)" in view.to_text()
    assert "will be scanned" not in view.to_text()
    instance.service.candidate_security_check.assert_called_once_with(
        skill_id="local", content_hash="hash"
    )
    instance.service.approve_candidate.assert_not_called()


def test_qualification_security_and_professionalism_run_together(mediation):
    from threading import Event

    instance, _, _, _ = mediation
    instance.service._candidate_event_context.return_value = (
        {"organization_id": "org", "qualification": "high_usage"}, "local", "hash", "Skill"
    )
    scan_started, review_started = Event(), Event()

    def scan(**_):
        scan_started.set()
        assert review_started.wait(5)
        return {"status": "pass"}

    def review(**_):
        review_started.set()
        assert scan_started.wait(5)
        return {"status": "pass"}

    instance.service.candidate_security_check.side_effect = scan
    instance.service.finish_candidate_professionalism_review.side_effect = review
    result = instance.qualification_advice(
        "org", {"reference": {"event_id": "event", "content_hash": "hash"}}
    )
    assert result["assessment_kind"] == "qualification"


def test_source_changed_during_professionalism_review_never_reaches_consent(
    mediation, monkeypatch
):
    instance, actor, _, _ = mediation
    candidate = {
        "id": "qualified",
        "organization_id": "org",
        "session_id": "session",
        "skill_id": "local",
        "content_hash": "hash",
    }
    instance.service.local_candidate_events.return_value = [candidate]
    instance.service._candidate_event_context.side_effect = [
        (candidate, "local", "hash", "My skill"),
        (candidate, "local", "changed", "My skill"),
    ]
    instance.ingest()
    present = Mock()
    monkeypatch.setattr(instance.consent, "present", present)
    assessor = Mock()
    assert (
        instance.prepare("org", actor, runtime={"model": "test-model", "provider": "test-provider"}, history=[], assessor=assessor) == []
    )
    present.assert_not_called()
    assessor.assert_not_called()


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
            runtime={"model": "test-model", "provider": "test-provider"},
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
        instance.prepare("org", actor, runtime={"model": "test-model", "provider": "test-provider"}, history=[], assessor=assessor) == []
    )
    assert instance.queue.assessments("org")[0]["state"] == "retired"
    instance.service.version_detail.assert_not_called()


@pytest.mark.parametrize("category", ["new_skill", "update_available"])
def test_own_publication_skips_assessment_but_preserves_activity(mediation, category):
    instance, actor, *_ = mediation
    enqueue(instance, category)
    instance.service.version_detail.return_value["version"]["published_by_user_id"] = "owner"
    assessor = Mock(side_effect=AssertionError("must not assess our own publication"))
    assert instance.prepare("org", actor, runtime={"model": "test-model", "provider": "test-provider"}, history=[], assessor=assessor) == []
    assessor.assert_not_called()
    assert instance.queue.assessments("org")[0]["state"] == "retired"
    assert instance.service.notifications(mark_seen=False)["events"] == [event(category)]
    instance.ingest()
    assert instance.prepare("org", actor, runtime={"model": "test-model", "provider": "test-provider"}, history=[], assessor=assessor) == []
    assert len(instance.queue.assessments("org")) == 1


@pytest.mark.parametrize("category", ["new_skill", "update_available"])
def test_own_publication_already_assessed_is_not_delivered(mediation, category):
    instance, _, *_ = mediation
    enqueue(instance, category)
    item = ready(instance)
    instance.service.version_detail.return_value["version"]["published_by_user_id"] = "owner"
    assert instance.begin_delivery("org", [item]) == []
    assert instance.queue.assessments("org")[0]["state"] == "retired"


@pytest.mark.parametrize("category", ["new_skill", "update_available"])
def test_teammate_version_stays_eligible_when_recipient_created_skill(mediation, category):
    instance, _, *_ = mediation
    enqueue(instance, category)
    item = ready(instance)
    detail = instance.service.version_detail.return_value
    detail["skill"]["created_by_user_id"] = "owner"
    detail["version"]["published_by_user_id"] = "teammate"
    assert instance._current_feed_jobs("org", [item["assessment"]]) == [item["assessment"]]
    assert instance.queue.assessments("org")[0]["state"] == "ready"


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
        instance.prepare("org", actor, runtime={"model": "test-model", "provider": "test-provider"}, history=[], assessor=assessor) == []
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
    result = instance.prepare("org", actor, runtime={"model": "test-model", "provider": "test-provider"}, history=[], assessor=assessor)
    assert len(result) == 1
    assert result[0]["advice"]["title"] == "Restored"
    assessor.assert_called_once()


@pytest.mark.parametrize("category", ["update_available", "installed", "updated"])
def test_existing_consent_prevents_duplicate_feed_assessment(mediation, category):
    instance, actor, _, _ = mediation
    request = instance.queue.enqueue("org", "request:skill:skill:2", {
        "kind": "skill", "skill_id": "skill", "version": 2, "user_requested": True,
    }, origin_session=actor.session_key)
    ready(instance)
    instance.service.install_plan.return_value = {
        "skill_id": "skill", "version": 2, "allowed": True, "receipt": "wip_one",
    }
    shown = instance.consent.present("org", request, actor)
    if category in {"installed", "updated"}:
        instance.service.install_apply.return_value = {"state": "active"}
        assert instance.consent.resolve("org", shown["id"], actor, "confirm")["state"] == "completed"
    # End the current claim so the next poll can acquire the feed job.
    with instance.service.store.transaction() as db:
        db.execute("UPDATE wisdom_assessment SET state='delivered',lease_token=NULL,lease_until=NULL WHERE id=?", (request,))
    enqueue(instance, category)
    assessor = Mock(side_effect=AssertionError("already has a consent card"))
    assert instance.prepare("org", actor, runtime={"model": "test-model", "provider": "test-provider"}, history=[], assessor=assessor) == []
    assert instance.consent.resolve("org", shown["id"], actor, "inspect")["state"] == (
        "pending" if category == "update_available" else "completed"
    )
    assert next(j for j in instance.queue.assessments("org") if j["event_key"] == "feed:event")["state"] == "retired"
    assessor.assert_not_called()


def test_legacy_operation_outcome_is_not_reassessed(mediation):
    instance, actor, _, _ = mediation
    instance.queue.enqueue("org", "outcome:old", {"kind": "notice", "notification": {"state": "stale"}})
    assessor = Mock(side_effect=AssertionError("outcome is already on the native card"))
    assert instance.prepare("org", actor, runtime={"model": "test-model", "provider": "test-provider"}, history=[], assessor=assessor) == []
    assert instance.queue.assessments("org")[0]["state"] == "retired"
    assessor.assert_not_called()


@pytest.mark.parametrize("category", ["installed", "updated"])
def test_unlinked_completion_uses_compact_receipt_without_model(mediation, category):
    from hermes_wisdom.mediation_view import advice_view

    instance, actor, _, _ = mediation
    enqueue(instance, category)
    assessor = Mock(side_effect=AssertionError("no assessment for a completed operation"))
    items = instance.prepare("org", actor, runtime={"model": "test-model", "provider": "test-provider"}, history=[], assessor=assessor)
    view = advice_view(items)
    assert view.summary == category.capitalize()
    assert view.items[0].title == "helpful"
    assert not view.items[0].detail
    assessor.assert_not_called()


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
    monkeypatch.setattr("hermes_wisdom.service.portal_base_url", lambda: "https://portal.test")
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


def test_prepare_does_not_claim_queued_work_without_current_entitlement(mediation, monkeypatch):
    instance, actor, _, _ = mediation
    instance.service.notifications.return_value = {"events": [event()]}
    instance.ingest()
    from tests.wisdom.local_auth import authorize_local

    authorize_local(monkeypatch, "org", scopes=[])

    assert instance.prepare(
        "org", actor, runtime={"model": "test", "provider": "test"}, history=[]
    ) == []
    assert instance.queue.assessments("org")[0]["state"] == "pending"
