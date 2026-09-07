import json
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_wisdom.agent_led.policy import AgentLedPolicy
from hermes_wisdom.agent_led.schemas import CandidateReviewResult
from hermes_wisdom.client import WisdomMuteResponse
from hermes_wisdom.consent import ConsentActor
from hermes_wisdom.mediation import WisdomMediation
from hermes_wisdom.mediation_store import MediationStore
from hermes_wisdom.weekly_queue import (
    enqueue_weekly_review,
    process_weekly_review,
    _session_call,
)
from tests.wisdom.test_agent_led import NOW, _store, _skill, _make_eligible, _use, _rec


@pytest.fixture
def weekly(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    _make_eligible(monkeypatch, root)
    store = _store(tmp_path)
    _skill(root, "release-notes")
    _use(store, "release-notes", [0, 1, 2])
    policy = AgentLedPolicy()
    monkeypatch.setattr("hermes_wisdom.weekly_queue.load_policy", lambda **kw: policy)
    client = Mock(display_org_id="org-1", identity={"owner": "owner"})
    client.recommendation_mute.return_value = WisdomMuteResponse(
        org_id="org-1", muted=False, duration=None, muted_until=None, forever=False
    )
    client.recommendation_suppressions.return_value = []
    service = Mock(store=store, client=client)
    now = [NOW.timestamp()]
    mediation = WisdomMediation(service, clock=lambda: now[0])
    actor = ConsentActor("session", "local", "owner", "local:session")
    return service, mediation, actor, root, now


def register(mediation, actor):
    mediation.queue.register_session(
        "org-1",
        session_key=actor.session_key,
        session_id=actor.session_key,
        platform=actor.platform,
        actor_id=actor.actor_id,
        private=True,
        available=True,
        user_activity=True,
        address=actor.address,
    )


def claim(weekly):
    service, mediation, actor, root, _ = weekly
    result = enqueue_weekly_review(service, now=NOW, skills_root=root)
    register(mediation, actor)
    return result, mediation.queue.claim("org-1", actor.session_key)[0]


def reviewer(payload, **kwargs):
    item = payload["candidates"][0]
    return CandidateReviewResult.model_validate({
        "window_days": 7,
        "recommendations": [
            _rec(item["skill_name"], item["content_hash"], count=99999)
        ],
    })


def test_producer_is_idempotent_across_connections_and_waits_for_active_session(weekly):
    from hermes_wisdom.store import WisdomStore

    service, mediation, actor, root, _ = weekly
    other = Mock(store=WisdomStore(service.store.root), client=service.client)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda s: enqueue_weekly_review(s, now=NOW, skills_root=root),
                [service, other],
            )
        )
    assert results[0]["assessment_id"] == results[1]["assessment_id"]
    assert len(mediation.queue.assessments("org-1")) == 1
    assert mediation.queue.claim("org-1", actor.session_key) == []
    service.client.suggest.assert_not_called()
    assert not (service.store.root / "agent_led_review_state.json").exists()


def test_dry_run_is_evidence_only_and_does_not_write(weekly):
    service, mediation, _, root, _ = weekly
    result = enqueue_weekly_review(service, now=NOW, skills_root=root, dry_run=True)
    assert result["dry_run"] and len(result["candidates"]) == 1
    assert mediation.queue.assessments("org-1") == []


def test_usage_window_excludes_future_and_old_days(weekly):
    service, _, _, root, _ = weekly
    _use(service.store, "release-notes", [-1, 8], per_day=100)
    result = enqueue_weekly_review(service, now=NOW, skills_root=root, dry_run=True)
    evidence = result["candidates"][0]["evidence"]
    assert evidence["invocation_count"] == 3
    assert evidence["days_used"] == 3
    assert "path" not in evidence and "frontmatter" not in evidence


def test_selected_advice_and_exact_native_candidate_commit_without_second_model_call(
    weekly,
):
    service, mediation, actor, _, _ = weekly
    _, job = claim(weekly)
    process_weekly_review(
        mediation,
        "org-1",
        job,
        runtime={"provider": "session-provider", "model": "session-model"},
        history=[],
        reviewer=reviewer,
    )
    rows = mediation.queue.assessments("org-1")
    parent = next(row for row in rows if row["id"] == job["id"])
    child = next(row for row in rows if row["id"] != job["id"])
    assert parent["state"] == "reviewed"
    assert child["state"] == "ready" and child["attempts"] == 0
    assert child["origin_session"] == actor.session_key
    assert "Used 3 times" in child["advice"]["explanation"]
    assert "99999" not in json.dumps(child)
    assert child["advice"]["provenance"]["model"] == "session-model"
    event = service.store.local_event(child["reference"]["event_id"])
    assert event["payload"]["agent_led_weekly"]
    assert event["content_hash"] == child["reference"]["content_hash"]
    assert child["reference"]["weekly_rank"] == 1
    assert mediation.queue.claim("org-1", actor.session_key)[0]["state"] == "ready"
    service.approve_candidate.assert_not_called()
    service.suggest.assert_not_called()


def test_quiet_week_commits_without_any_proactive_card(weekly):
    _, mediation, _, _, _ = weekly
    _, job = claim(weekly)
    quiet = Mock(
        return_value=CandidateReviewResult(
            window_days=7,
            recommendations=[],
            nothing_to_recommend_reason="Nothing useful to share this week.",
        )
    )
    process_weekly_review(
        mediation, "org-1", job, runtime={}, history=[], reviewer=quiet
    )
    rows = mediation.queue.assessments("org-1")
    assert len(rows) == 1 and rows[0]["state"] == "reviewed"
    assert rows[0]["advice"]["selected"] == []


def test_suppression_and_mute_prevent_review_work(weekly):
    service, mediation, _, _, now = weekly
    _, job = claim(weekly)
    service.client.recommendation_mute.return_value = WisdomMuteResponse(
        org_id="org-1", muted=True, duration="forever", muted_until=None, forever=True
    )
    review = Mock(side_effect=AssertionError("must not invoke model"))
    process_weekly_review(
        mediation, "org-1", job, runtime={}, history=[], reviewer=review
    )
    row = mediation.queue.assessments("org-1")[0]
    assert row["state"] == "pending" and row["attempts"] == 0
    assert row["available_at"] > now[0]


def test_stale_owner_cannot_commit_selection(weekly):
    _, mediation, _, _, now = weekly
    _, job = claim(weekly)
    now[0] += 181
    process_weekly_review(
        mediation, "org-1", job, runtime={}, history=[], reviewer=reviewer
    )
    assert len(mediation.queue.assessments("org-1")) == 1
    assert mediation.queue.assessments("org-1")[0].get("advice") is None


def test_changed_skill_is_not_presented_from_old_usage_snapshot(weekly):
    _, mediation, _, root, _ = weekly
    _, job = claim(weekly)
    (root / "release-notes" / "SKILL.md").write_text("changed", encoding="utf-8")
    review = Mock(side_effect=AssertionError("must not review stale bytes"))
    process_weekly_review(
        mediation, "org-1", job, runtime={}, history=[], reviewer=review
    )
    assert len(mediation.queue.assessments("org-1")) == 1
    assert mediation.queue.assessments("org-1")[0]["state"] == "reviewed"


def test_model_cannot_invent_a_candidate(weekly):
    _, mediation, _, _, _ = weekly
    _, job = claim(weekly)
    wrong = Mock(
        return_value=CandidateReviewResult.model_validate({
            "window_days": 7,
            "recommendations": [_rec("invented", "sha256:" + "a" * 64)],
        })
    )
    with pytest.raises(ValueError, match="invented"):
        process_weekly_review(
            mediation, "org-1", job, runtime={}, history=[], reviewer=wrong
        )
    assert len(mediation.queue.assessments("org-1")) == 1


def test_later_week_does_not_repeat_an_unchanged_recommendation(weekly):
    service, mediation, _, root, _ = weekly
    _, job = claim(weekly)
    process_weekly_review(
        mediation, "org-1", job, runtime={}, history=[], reviewer=reviewer
    )
    _use(service.store, "release-notes", [-7], per_day=3)
    next_week = enqueue_weekly_review(
        service, now=NOW + timedelta(days=7), skills_root=root, dry_run=True
    )
    assert next_week["candidates"] == []
    assert "release-notes" in next_week["excluded"]["previously_handled"]


def test_weekly_selection_does_not_duplicate_an_immediate_qualification(weekly):
    service, mediation, _, _, _ = weekly
    _, job = claim(weekly)
    source = job["reference"]["candidates"][0]
    identity = mediation.queue.enqueue(
        "org-1",
        "candidate:immediate",
        {
            "kind": "candidate",
            "event_id": "immediate",
            "local_skill_id": source["local_skill_id"],
            "content_hash": source["content_hash"],
        },
        origin_session="session",
    )
    process_weekly_review(
        mediation, "org-1", job, runtime={}, history=[], reviewer=reviewer
    )
    rows = mediation.queue.assessments("org-1")
    assert len(rows) == 2
    assert (
        next(row for row in rows if row["id"] == job["id"])["advice"]["selected"] == []
    )
    assert next(row for row in rows if row["id"] == identity)["state"] == "pending"


def test_weekly_selected_session_model_has_no_tools_or_auxiliary_fallback(monkeypatch):
    call = Mock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None))]
        )
    )
    monkeypatch.setattr("agent.auxiliary_client.call_llm", call)
    monkeypatch.setattr(
        "agent.auxiliary_client.extract_content_or_reasoning", lambda _: "{}"
    )
    runtime = {"model": "selected", "provider": "chosen", "api_key": "credential"}
    assert _session_call(runtime)([], {}) == "{}"
    assert call.call_args.kwargs["tools"] == []
    assert call.call_args.kwargs["main_runtime"] is runtime
    assert "task" not in call.call_args.kwargs
    call.return_value.choices[0].message.tool_calls = [Mock()]
    with pytest.raises(ValueError, match="disallowed"):
        _session_call(runtime)([], {})


def test_prepare_reuses_weekly_advice_and_native_exact_plan(weekly, monkeypatch):
    service, mediation, actor, root, _ = weekly
    monkeypatch.setattr("hermes_wisdom.agent_led.agent.review_candidates", reviewer)
    monkeypatch.setattr(
        "hermes_wisdom.agent_led.policy.load_policy", lambda **kw: AgentLedPolicy()
    )
    enqueue_weekly_review(service, now=NOW, skills_root=root)
    register(mediation, actor)
    no_second_model = Mock(side_effect=AssertionError("selection must be reused"))
    assert (
        mediation.prepare(
            "org-1", actor, runtime={}, history=[], assessor=no_second_model
        )
        == []
    )

    def context(event_id):
        event = service.store.local_event(event_id)
        return event, event["skill_id"], event["content_hash"], "release-notes"

    service._candidate_event_context.side_effect = context
    service.prepare_candidate.return_value = {
        "stage": "review",
        "review": {
            "hashes": {
                "content": "exact-package",
                "author_description": "exact-description",
                "package_manifest": "exact-manifest",
            },
            "draft": {},
        },
    }
    items = mediation.prepare(
        "org-1", actor, runtime={}, history=[], assessor=no_second_model
    )
    assert len(items) == 1 and items[0]["interaction"]["operation"] == "share"
    assert items[0]["interaction"]["actions"][-1] == "confirm"
    no_second_model.assert_not_called()
    service.approve_candidate.assert_not_called()
    service.prepare_candidate.assert_not_called()


def test_failed_model_uses_bounded_retries_then_one_manual_fallback(
    weekly, monkeypatch
):
    service, mediation, actor, root, now = weekly
    failed = Mock(side_effect=TimeoutError)
    monkeypatch.setattr("hermes_wisdom.agent_led.agent.review_candidates", failed)
    enqueue_weekly_review(service, now=NOW, skills_root=root)
    for _ in range(3):
        register(mediation, actor)
        assert mediation.prepare("org-1", actor, runtime={}, history=[]) == []
        now[0] += 61
    register(mediation, actor)
    assert mediation.prepare("org-1", actor, runtime={}, history=[]) == []
    items = mediation.prepare("org-1", actor, runtime={}, history=[])
    assert failed.call_count == 3
    assert len(items) == 1 and items[0]["interaction"] is None
    assert "/wisdom candidates" in items[0]["advice"]["explanation"]
    assert not service.store.local_events(kind="wisdom.candidate")


def test_selection_transaction_rolls_back_all_cards_on_partial_failure(
    weekly, monkeypatch
):
    service, mediation, _, root, _ = weekly
    _skill(root, "second-skill")
    _use(service.store, "second-skill", [0, 1, 2])
    _, job = claim(weekly)
    emit = service.store.emit_local_event
    calls = []

    def fail_second(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise RuntimeError("simulated crash")
        return emit(**kwargs)

    monkeypatch.setattr(service.store, "emit_local_event", fail_second)
    both = lambda payload, **kw: CandidateReviewResult.model_validate({
        "window_days": 7,
        "recommendations": [
            _rec(c["skill_name"], c["content_hash"]) for c in payload["candidates"]
        ],
    })
    with pytest.raises(RuntimeError, match="simulated crash"):
        process_weekly_review(
            mediation, "org-1", job, runtime={}, history=[], reviewer=both
        )
    assert len(mediation.queue.assessments("org-1")) == 1
    assert not service.store.local_events(kind="wisdom.candidate")
    monkeypatch.setattr(service.store, "emit_local_event", emit)
    process_weekly_review(
        mediation, "org-1", job, runtime={}, history=[], reviewer=both
    )
    assert len(mediation.queue.assessments("org-1")) == 3


def test_edit_during_model_review_does_not_commit_stale_selection(weekly):
    _, mediation, _, root, _ = weekly
    _, job = claim(weekly)

    def changed(payload, **kwargs):
        (root / "release-notes" / "SKILL.md").write_text(
            "changed during review", encoding="utf-8"
        )
        return reviewer(payload, **kwargs)

    process_weekly_review(
        mediation, "org-1", job, runtime={}, history=[], reviewer=changed
    )
    rows = mediation.queue.assessments("org-1")
    assert len(rows) == 1 and rows[0]["state"] == "reviewed"
    assert rows[0]["advice"]["excluded"]["changed_during_review"]


def test_server_cap_cannot_be_exceeded_by_model(weekly, monkeypatch):
    service, mediation, _, root, _ = weekly
    _skill(root, "second-skill")
    _use(service.store, "second-skill", [0, 1, 2])
    monkeypatch.setattr(
        "hermes_wisdom.weekly_queue.load_policy",
        lambda **kw: AgentLedPolicy(max_candidates=1),
    )
    _, job = claim(weekly)
    both = lambda payload, **kw: CandidateReviewResult.model_validate({
        "window_days": 7,
        "recommendations": [
            _rec(c["skill_name"], c["content_hash"]) for c in payload["candidates"]
        ],
    })
    with pytest.raises(ValueError, match="cap"):
        process_weekly_review(
            mediation, "org-1", job, runtime={}, history=[], reviewer=both
        )
    assert len(mediation.queue.assessments("org-1")) == 1


def test_new_org_has_its_own_weekly_identity_and_old_owner_cannot_commit(weekly):
    service, mediation, _, root, _ = weekly
    result, job = claim(weekly)
    service.store.verify_installation_identity("org-2")
    service.client.display_org_id = "org-2"
    other = enqueue_weekly_review(service, now=NOW, skills_root=root)
    assert result["assessment_id"] != other["assessment_id"]
    with pytest.raises(ValueError):
        process_weekly_review(
            mediation, "org-1", job, runtime={}, history=[], reviewer=reviewer
        )
    assert len(mediation.queue.assessments("org-2")) == 1


@pytest.mark.parametrize("qualification", ["high_usage", "refinement", "weekly_usage"])
def test_qualification_queues_once_in_its_originating_session(
    weekly, monkeypatch, qualification
):
    service, mediation, _, _, _ = weekly
    monkeypatch.setattr(
        "hermes_wisdom.weekly_queue.enqueue_weekly_review", lambda _: {}
    )
    service.local_candidate_events.return_value = [
        {
            "id": "old-signal",
            "organization_id": "org-1",
            "payload": {"skill_name": "release-notes"},
            "content_hash": "hash",
            "skill_id": "local",
            "session_id": "originating-session",
            "qualification": qualification,
        }
    ]
    service.notifications.return_value = {"events": []}
    assert mediation.ingest() == "org-1"
    assert mediation.ingest() == "org-1"
    jobs = mediation.queue.assessments("org-1")
    assert len(jobs) == 1
    assert jobs[0]["event_key"] == "candidate:old-signal"
    assert jobs[0]["origin_session"] == "originating-session"
    assert jobs[0]["reference"] == {
        "kind": "candidate",
        "event_id": "old-signal",
        "content_hash": "hash",
        "local_skill_id": "local",
    }
    service.local_candidate_events.return_value = []
    mediation.ingest()
    assert mediation.queue.assessments("org-1")[0]["state"] == "retired"


@pytest.mark.parametrize(
    "policy",
    [
        AgentLedPolicy(enabled=False),
        AgentLedPolicy(max_candidates=0),
        AgentLedPolicy(notification_defaults={"skill_ready_to_share": False}),
    ],
)
def test_disabled_weekly_policy_does_not_authenticate_build_or_queue(
    weekly, monkeypatch, policy
):
    service, mediation, _, root, _ = weekly
    monkeypatch.setattr("hermes_wisdom.weekly_queue.load_policy", lambda **kw: policy)
    service.require_setup.side_effect = AssertionError(
        "disabled producer must be inert"
    )
    assert enqueue_weekly_review(service, now=NOW, skills_root=root)["queued"] is False
    assert mediation.queue.assessments("org-1") == []


def test_signed_out_profile_cannot_queue_usage_for_later_replay(weekly):
    service, _, _, root, _ = weekly
    with service.store.transaction() as db:
        db.execute("UPDATE installation_identity SET verified_org_id=NULL")
    with pytest.raises(ValueError):
        enqueue_weekly_review(service, now=NOW, skills_root=root)
    with service.store.transaction() as db:
        assert db.execute("SELECT COUNT(*) FROM wisdom_assessment").fetchone()[0] == 0


def test_malformed_model_output_never_creates_native_candidates(weekly, monkeypatch):
    from hermes_wisdom.agent_led.schemas import SchemaRejected

    _, mediation, _, _, _ = weekly
    _, job = claim(weekly)
    call = Mock(return_value="not JSON")
    monkeypatch.setattr("hermes_wisdom.weekly_queue._session_call", lambda _: call)
    with pytest.raises(SchemaRejected):
        process_weekly_review(mediation, "org-1", job, runtime={}, history=[])
    assert len(mediation.queue.assessments("org-1")) == 1
    assert not mediation.service.store.local_events(kind="wisdom.candidate")
