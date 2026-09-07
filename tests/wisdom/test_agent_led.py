"""Tests for the agent-led Collective Wisdom sharing layer."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from hermes_wisdom.agent_led import templates
from hermes_wisdom.agent_led.evidence import build_evidence, provenance_exclusion
from hermes_wisdom.agent_led.history import SuggestionHistory
from hermes_wisdom.agent_led.install_flow import InstallFlow, detect_prerequisites
from hermes_wisdom.agent_led.notify import (
    STALE_ACTION_MESSAGE,
    DeliveryLedger,
    deliver,
    share_candidate_event,
    teammate_event,
)
from hermes_wisdom.agent_led.policy import AgentLedPolicy, load_policy
from hermes_wisdom.agent_led.schemas import (
    SchemaRejected,
    parse_candidate_review,
    parse_recipient_recommendation,
    parse_share_package,
)
from hermes_wisdom.agent_led.share_flow import ShareFlow, scan_credentials
from hermes_wisdom.agent_led.weekly import run_weekly_review
from hermes_wisdom.qualification import record_successful_use
from hermes_wisdom.store import WisdomStore

NOW = datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _store(tmp_path: Path, *, signed_in: bool = True) -> WisdomStore:
    store = WisdomStore(tmp_path / "state")
    store.installation_identity()
    if signed_in:
        store.verify_installation_identity("org-1")
    return store


def _skill(root: Path, name: str, *, env: list[str] | None = None) -> Path:
    path = root / name
    path.mkdir(parents=True)
    env_block = ""
    if env:
        env_block = "required_environment_variables: [" + ", ".join(env) + "]\n"
    (path / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Use when doing {name} work.\n{env_block}---\n# Body\n",
        encoding="utf-8",
    )
    return path


def _make_eligible(monkeypatch, skills_root: Path) -> None:
    monkeypatch.setattr("hermes_wisdom.qualification.get_skills_dir", lambda: skills_root)
    monkeypatch.setattr(
        "hermes_wisdom.qualification._find_skill_dir", lambda name: skills_root / name
    )
    monkeypatch.setattr("hermes_wisdom.qualification.is_bundled", lambda _n: False)
    monkeypatch.setattr("hermes_wisdom.qualification.is_hub_installed", lambda _n: False)
    monkeypatch.setattr("hermes_wisdom.qualification._profile_timezone", lambda: (timezone.utc, "UTC"))
    monkeypatch.setattr("hermes_wisdom.qualification.ensure_skill_editorial_metadata", lambda p: {"editorial_name": p.name, "editorial_description": "", "changed": False})
    monkeypatch.setattr("hermes_wisdom.agent_led.evidence.get_skills_dir", lambda: skills_root)
    monkeypatch.setattr("hermes_wisdom.agent_led.evidence.is_bundled", lambda _n: False)
    monkeypatch.setattr("hermes_wisdom.agent_led.evidence.is_hub_installed", lambda _n: False)


def _use(store: WisdomStore, name: str, days_ago: list[int], per_day: int = 1) -> None:
    for offset in days_ago:
        for _ in range(per_day):
            record_successful_use(name, store=store, at=NOW - timedelta(days=offset))


def _rec(name: str, content_hash: str, count: int = 5) -> dict:
    return {
        "skill_name": name,
        "content_hash": content_hash,
        "editorial_name": "Release Notes Drafter",
        "one_line_description": "Drafts release notes from merged PRs.",
        "what_it_does": "Collects merged PRs and drafts notes.",
        "how_user_relied_on_it": "weekly release announcements",
        "evidence": {"window_days": 7, "invocation_count": count, "days_used": 3, "last_used_at": None, "examples": []},
        "why_coworkers_benefit": "every team ships weekly",
        "audience": "the platform team",
        "portability": [],
        "confidence": 0.8,
        "allowed_actions": ["share", "review", "not_now"],
    }


def _review_json(name: str, content_hash: str, count: int = 5) -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "window_days": 7,
            "considered": [name],
            "recommendations": [_rec(name, content_hash, count)],
            "nothing_to_recommend_reason": None,
            "requires_confirmation": True,
        }
    )


# ---------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------

def test_policy_defaults_and_local_override(monkeypatch):
    monkeypatch.setattr("hermes_wisdom.mediation.delivery_mode", lambda: "agent")
    policy = load_policy(local={})
    assert policy.enabled is False  # no authoritative organization policy
    assert policy.window_days == 7
    assert policy.min_aggregate_count == 3
    assert policy.dismiss_suppression_days == 30
    assert policy.popular_install_threshold == 10
    custom = load_policy(local={"enabled": False, "window_days": 14, "min_aggregate_count": 0})
    assert custom.enabled is False
    assert custom.window_days == 14
    assert custom.min_aggregate_count == 1  # floor
    assert custom.source == "local_config"


@pytest.mark.parametrize("policy", [
    AgentLedPolicy(max_candidates=0),
    AgentLedPolicy(notification_defaults={"skill_ready_to_share": False}),
])
def test_weekly_org_policy_gate_runs_before_model_or_delivery(tmp_path, policy):
    def unexpected(*args, **kwargs):
        raise AssertionError("disabled recommendations must not call the model or sender")

    result = run_weekly_review(
        store=_store(tmp_path), policy=policy, model_call=unexpected,
        sender=unexpected, force=True, state_path=tmp_path / "weekly.json",
    )
    assert result["ran"] is False
    assert result["skipped_reason"] == "organization_notifications_disabled"
    assert not (tmp_path / "weekly.json").exists()


def test_policy_server_block_wins(monkeypatch):
    from hermes_wisdom.client import AgentLedPolicyResponse

    monkeypatch.setattr("hermes_wisdom.mediation.delivery_mode", lambda: "agent")
    class Client:
        display_org_id = "org-1"

        def agent_led_policy(self):
            return AgentLedPolicyResponse(
                org_id="org-1", usage_evidence_window_days=10,
                min_aggregate_invocations=8, consecutive_day_usage_counts=False,
                repeated_edits_count=False, max_recommendations_per_user_per_week=0,
                publication_mode="moderated", install_popularity_threshold=20,
                notification_defaults={"skill_ready_to_share": False, "teammate_published": True, "update_available": False},
                manager_review_email_cadence="daily", not_now_suppression_days=60,
                version=1, updated_by_user_id=None,
            )

    policy = load_policy(client=Client(), local={"window_days": 5})
    assert policy.window_days == 10
    assert policy.source == "server_policy"
    assert policy.enabled is True
    assert policy.min_aggregate_count == 8
    assert policy.max_candidates == 0
    assert policy.dismiss_suppression_days == 60
    assert policy.notification_defaults["skill_ready_to_share"] is False
    assert policy.consecutive_day_usage_counts is False


# ---------------------------------------------------------------------------
# evidence: built-in exclusion + 7-day aggregate counting
# ---------------------------------------------------------------------------

def test_builtin_skills_excluded(monkeypatch, tmp_path):
    skills = tmp_path / "skills"
    _skill(skills, "own-skill")
    _skill(skills, "bundled-skill")
    _make_eligible(monkeypatch, skills)
    store = _store(tmp_path)
    _use(store, "own-skill", [0, 1, 2], per_day=2)
    _use(store, "bundled-skill", [0, 1, 2], per_day=2)
    monkeypatch.setattr(
        "hermes_wisdom.agent_led.evidence.is_bundled", lambda n: n == "bundled-skill"
    )
    payload = build_evidence(
        store=store, policy=AgentLedPolicy(), history=SuggestionHistory(tmp_path / "h.json"), at=NOW, skills_root=skills
    )
    names = [c.skill_name for c in payload.candidates]
    assert names == ["own-skill"]
    assert payload.excluded["bundled"] == ["bundled-skill"]
    assert provenance_exclusion("x", skills / "_wisdom" / "org" / "x", skills_root=skills) == "managed_or_archived"


def test_seven_day_aggregate_counting(monkeypatch, tmp_path):
    skills = tmp_path / "skills"
    _skill(skills, "counted", env=["MY_TOKEN"])
    _make_eligible(monkeypatch, skills)
    store = _store(tmp_path)
    # 2 uses/day on 3 distinct days inside the window, plus 4 uses outside it.
    _use(store, "counted", [0, 3, 6], per_day=2)
    _use(store, "counted", [7, 8], per_day=2)
    payload = build_evidence(
        store=store, policy=AgentLedPolicy(), history=SuggestionHistory(tmp_path / "h.json"), at=NOW, skills_root=skills
    )
    assert len(payload.candidates) == 1
    candidate = payload.candidates[0]
    assert candidate.invocation_count == 6
    assert candidate.days_used == 3
    assert candidate.last_used_day == NOW.date().isoformat()
    assert candidate.required_environment_variables == ["MY_TOKEN"]
    assert payload.window_start == (NOW.date() - timedelta(days=6)).isoformat()


def test_below_min_count_excluded(monkeypatch, tmp_path):
    skills = tmp_path / "skills"
    _skill(skills, "rare")
    _make_eligible(monkeypatch, skills)
    store = _store(tmp_path)
    _use(store, "rare", [0])
    payload = build_evidence(
        store=store, policy=AgentLedPolicy(min_aggregate_count=3), history=SuggestionHistory(tmp_path / "h.json"), at=NOW, skills_root=skills
    )
    assert payload.candidates == []
    assert payload.excluded["below_min_count"] == ["rare"]


# ---------------------------------------------------------------------------
# dismissal suppression by hash
# ---------------------------------------------------------------------------

def test_dismissal_suppresses_same_hash_only(tmp_path):
    history = SuggestionHistory(tmp_path / "h.json")
    result = history.record_dismissal("s", "hash-a", suppression_days=30, at=NOW)
    assert result["gateway"] == "pending_gateway"
    assert history.is_suppressed("s", "hash-a", at=NOW + timedelta(days=29))
    assert not history.is_suppressed("s", "hash-a", at=NOW + timedelta(days=31))
    assert not history.is_suppressed("s", "hash-b", at=NOW)  # content changed
    reloaded = SuggestionHistory(tmp_path / "h.json")
    assert reloaded.is_suppressed("s", "hash-a", at=NOW)
    assert reloaded.pending_gateway()[0]["kind"] == "dismissal"


def test_dismissed_skill_excluded_from_evidence(monkeypatch, tmp_path):
    skills = tmp_path / "skills"
    _skill(skills, "dismissed")
    _make_eligible(monkeypatch, skills)
    store = _store(tmp_path)
    _use(store, "dismissed", [0, 1, 2], per_day=2)
    history = SuggestionHistory(tmp_path / "h.json")
    first = build_evidence(store=store, policy=AgentLedPolicy(), history=history, at=NOW, skills_root=skills)
    history.record_dismissal("dismissed", first.candidates[0].content_hash, at=NOW)
    second = build_evidence(store=store, policy=AgentLedPolicy(), history=history, at=NOW, skills_root=skills)
    assert second.candidates == []
    assert second.excluded["dismissed"] == ["dismissed"]


def test_mute_uses_fixed_options(tmp_path):
    history = SuggestionHistory(tmp_path / "h.json")
    assert [label for _k, label, _d in templates.MUTE_OPTIONS] == ["1 day", "1 week", "30 days", "Forever"]
    history.record_mute("*", days=templates.mute_duration_days("1w"), at=NOW)
    assert history.is_muted(at=NOW + timedelta(days=6))
    assert not history.is_muted(at=NOW + timedelta(days=8))
    history.record_mute("skill-x", days=templates.mute_duration_days("forever"), at=NOW)
    assert history.is_muted("skill-x", at=NOW + timedelta(days=4000))


# ---------------------------------------------------------------------------
# schema validation
# ---------------------------------------------------------------------------

def test_candidate_review_schema_accepts_valid_and_repairs_bad_item():
    good = parse_candidate_review(_review_json("s", "abcdef1234"))
    assert good.recommendations[0].skill_name == "s"
    assert good.requires_confirmation is True
    payload = json.loads(_review_json("s", "abcdef1234"))
    payload["recommendations"].append({"skill_name": "broken"})  # invalid item
    repaired = parse_candidate_review(json.dumps(payload))
    assert [r.skill_name for r in repaired.recommendations] == ["s"]


def test_candidate_review_schema_rejects_secrets_and_garbage():
    payload = json.loads(_review_json("s", "abcdef1234"))
    payload["recommendations"][0]["what_it_does"] = "uses token ghp_" + "a" * 36
    # A credential-shaped recommendation is dropped by the repair pass, never surfaced.
    assert parse_candidate_review(json.dumps(payload)).recommendations == []
    with pytest.raises(SchemaRejected):
        parse_candidate_review(json.dumps({**payload, "window_days": 0}))
    with pytest.raises(SchemaRejected):
        parse_candidate_review("not json at all")
    with pytest.raises(SchemaRejected):
        parse_candidate_review(json.dumps({**json.loads(_review_json("s", "abcdef1234")), "requires_confirmation": False}))


def test_candidate_review_never_removes_not_now():
    payload = json.loads(_review_json("s", "abcdef1234"))
    payload["recommendations"][0]["allowed_actions"] = ["share"]
    parsed = parse_candidate_review(json.dumps(payload))
    assert "not_now" in parsed.recommendations[0].allowed_actions


def test_recipient_recommendation_schema():
    irrelevant = parse_recipient_recommendation(
        json.dumps({"schema_version": 1, "skill_id": "sk", "version": 1, "relevant": False, "not_relevant_reason": "no overlap", "confidence": 0.9})
    )
    assert irrelevant.relevant is False and "mute" in irrelevant.allowed_actions
    with pytest.raises(SchemaRejected):
        parse_recipient_recommendation(
            json.dumps({"schema_version": 1, "skill_id": "sk", "version": 1, "relevant": True, "confidence": 0.9})
        )
    with pytest.raises(SchemaRejected):
        parse_recipient_recommendation(
            json.dumps({"schema_version": 1, "skill_id": "sk", "version": 1, "relevant": False, "confidence": 0.9, "installed": True})
        )


def test_share_package_schema_requires_skill_md_and_safe_paths():
    base = {
        "schema_version": 1, "skill_name": "s", "source_content_hash": "abcdef1234", "editorial_name": "S",
        "plain_description": "Does S.", "files": [{"path": "SKILL.md", "content": "# S"}],
        "requirements": [], "setup_instructions": [], "credential_handoff": [], "compatibility_limits": [],
        "verification_step": "hermes skills list | grep S", "removed_or_generalized": [], "related_skills": [],
    }
    assert parse_share_package(json.dumps(base)).skill_name == "s"
    with pytest.raises(SchemaRejected):
        parse_share_package(json.dumps({**base, "files": [{"path": "README.md", "content": "x"}]}))
    with pytest.raises(SchemaRejected):
        parse_share_package(json.dumps({**base, "files": [{"path": "../SKILL.md", "content": "x"}]}))


# ---------------------------------------------------------------------------
# templates: exact headers, no 'Pass'
# ---------------------------------------------------------------------------

def test_share_template_exact_strings():
    notice = templates.render_share(
        editorial_name="Release Notes Drafter",
        description="Drafts release notes from merged PRs.",
        count_7d=9,
        specific_work="weekly release announcements",
        audience="the platform team",
        reason="every team ships weekly",
        checks_passed=True,
    )
    text = notice.text
    assert notice.title == "Reusable skill ready to review"
    assert notice.lines[0] == "Release Notes Drafter"
    assert notice.lines[1] == "    Drafts release notes from merged PRs."
    assert "Why we ask" in notice.lines
    assert (
        "You used this skill 9 times in the last 7 days. It has helped with weekly release "
        "announcements. I think it could help the platform team because every team ships weekly."
    ) in notice.lines
    assert "Checks complete: no known matches detected." in notice.lines
    for check in (
        "No profanity or abusive language",
        "No hate or harassment",
        "No sexual or graphic content",
        "No detected credentials or private keys",
    ):
        assert f"✓ {check}" in notice.lines
    assert notice.lines[-1] == "Would you like to share it?"
    assert [a.label for a in notice.actions] == ["Not now", "Review", "Share"]
    assert "Pass" not in text.split()
    assert "Pass" not in text


def test_share_template_requires_real_count():
    with pytest.raises(ValueError):
        templates.render_share(
            editorial_name="X", description="d", count_7d=None,  # type: ignore[arg-type]
            specific_work="w", audience="a", reason="r",
        )


def test_teammate_template_exact_strings():
    notice = templates.render_teammate(
        editorial_name="Incident Timeline Builder",
        outcome_one_liner="Turn a noisy incident channel into a clean timeline.",
        publisher_name="Dana",
        specific_work="post-incident reviews",
        count_7d=12,
        recipient_reason="your on-call rotation",
        installation_count=14,
    )
    assert notice.title == "New skill from your team"
    assert notice.lines[0] == "Incident Timeline Builder"
    assert notice.lines[2] == "Published by Dana"
    assert (
        "Dana uses this for post-incident reviews. It was used 12 times in the last 7 days. "
        "I think it could help you with your on-call rotation."
    ) in notice.lines
    assert "✓ Security check complete (credentials, private keys, organization policy)" in notice.lines
    assert "Popular with your team: 10+ installations" in notice.lines
    assert [a.label for a in notice.actions] == ["Mute", "View", "Install"]
    assert "New notification" not in notice.text
    assert "Pass" not in notice.text
    quiet = templates.render_teammate(
        editorial_name="X", outcome_one_liner="o", publisher_name="P", specific_work="w",
        count_7d=1, recipient_reason="r", installation_count=3,
    )
    assert not any(line.startswith("Popular with your team") for line in quiet.lines)


def test_published_and_update_templates():
    open_notice = templates.render_published(organization_name="Acme", moderated=False)
    assert open_notice.title == "Published to Acme."
    assert open_notice.lines == ["Your teammates can now find and install this skill."]
    assert [a.label for a in open_notice.actions] == ["View in Portal"]
    moderated = templates.render_published(organization_name="Acme", moderated=True)
    assert moderated.title == "Sent for review."
    assert moderated.lines == [
        "It is not available to your organization yet. You will be notified when the review is complete."
    ]
    update = templates.render_update(skill_name="Release Notes Drafter", summary="Adds changelog links.", reason="you ship weekly", setup_impact=None)
    assert update.title == "Update available for Release Notes Drafter"
    assert update.lines == [
        "Adds changelog links.",
        "This matters for your work because you ship weekly.",
        "Setup impact: None",
    ]
    assert [a.label for a in update.actions] == ["Mute", "View changes", "Update"]
    for notice in (open_notice, moderated, update):
        assert "Pass" not in notice.text


def test_templates_reject_forbidden_words_in_prose():
    with pytest.raises(ValueError):
        templates.render_update(skill_name="S", summary="Pass", reason="r")


# ---------------------------------------------------------------------------
# idempotent delivery + stale buttons
# ---------------------------------------------------------------------------

def test_delivery_is_idempotent_and_retries(tmp_path):
    from hermes_wisdom.agent_led.schemas import CandidateRecommendation

    rec = CandidateRecommendation.model_validate(_rec("s", "abcdef1234"))
    event = share_candidate_event(rec, organization_id="org-1", recipient_id="u1", at=NOW)
    sent: list[str] = []
    calls = {"n": 0}

    def flaky(evt):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("network")
        sent.append(evt.dedup_key)

    ledger = DeliveryLedger(tmp_path / "ledger.json")
    first = deliver(event, sender=flaky, ledger=ledger, retries=2, sleep=lambda _s: None, now=NOW)
    assert first["delivered"] and first["attempts"] == 2
    second = deliver(event, sender=flaky, ledger=ledger, retries=2, sleep=lambda _s: None, now=NOW)
    assert second["delivered"] and second["duplicate"] is True
    assert sent == [event.dedup_key]
    # Same inputs -> same dedup key across processes.
    again = share_candidate_event(rec, organization_id="org-1", recipient_id="u1", at=NOW)
    assert again.dedup_key == event.dedup_key


def test_delivery_failure_reports_and_does_not_mark_delivered(tmp_path):
    from hermes_wisdom.agent_led.schemas import CandidateRecommendation

    rec = CandidateRecommendation.model_validate(_rec("s", "abcdef1234"))
    event = share_candidate_event(rec, organization_id="org-1", recipient_id="u1", at=NOW)
    ledger = DeliveryLedger(tmp_path / "ledger.json")

    def broken(_evt):
        raise RuntimeError("down")

    result = deliver(event, sender=broken, ledger=ledger, retries=1, sleep=lambda _s: None, now=NOW)
    assert result["delivered"] is False and result["attempts"] == 2
    assert not ledger.delivered(event.dedup_key)


def test_stale_and_duplicate_actions(tmp_path):
    from hermes_wisdom.agent_led.schemas import CandidateRecommendation

    rec = CandidateRecommendation.model_validate(_rec("s", "abcdef1234"))
    event = share_candidate_event(rec, organization_id="org-1", recipient_id="u1", at=NOW, ttl_hours=1)
    ledger = DeliveryLedger(tmp_path / "ledger.json")
    deliver(event, sender=lambda _e: None, ledger=ledger, now=NOW)
    share_target = next(a.target for a in event.allowed_actions if a.id == "share")
    resolved = ledger.resolve_action(share_target, at=NOW)
    assert resolved["ok"] and resolved["action"] == "share"
    assert ledger.mark_acted(event.dedup_key, "share") is True
    assert ledger.mark_acted(event.dedup_key, "share") is False  # no duplicate action
    assert ledger.resolve_action(share_target, at=NOW)["message"] == STALE_ACTION_MESSAGE
    fresh = share_candidate_event(rec, organization_id="org-2", recipient_id="u1", at=NOW, ttl_hours=1)
    deliver(fresh, sender=lambda _e: None, ledger=ledger, now=NOW)
    expired = ledger.resolve_action(fresh.allowed_actions[0].target, at=NOW + timedelta(hours=2))
    assert expired["stale"] and expired["message"] == STALE_ACTION_MESSAGE
    assert ledger.resolve_action("garbage", at=NOW)["stale"]


def test_teammate_event_requires_relevance():
    from hermes_wisdom.agent_led.schemas import RecipientRecommendation

    rec = RecipientRecommendation.model_validate(
        {"schema_version": 1, "skill_id": "sk", "version": 1, "relevant": False, "confidence": 0.5}
    )
    with pytest.raises(ValueError):
        teammate_event(rec, organization_id="o", recipient_id="r")


# ---------------------------------------------------------------------------
# weekly job: end to end with a fake agent
# ---------------------------------------------------------------------------

def test_weekly_review_end_to_end(monkeypatch, tmp_path):
    skills = tmp_path / "skills"
    _skill(skills, "notes")
    _make_eligible(monkeypatch, skills)
    store = _store(tmp_path)
    _use(store, "notes", [0, 1, 2], per_day=3)
    history = SuggestionHistory(tmp_path / "h.json")
    ledger = DeliveryLedger(tmp_path / "ledger.json")
    seen: list[dict] = []

    def model_call(messages, schema):
        payload = json.loads(messages[1]["content"])
        seen.append(payload)
        cand = payload["candidates"][0]
        # Agent lies about the count; the job must overwrite it with ledger truth.
        return _review_json(cand["skill_name"], cand["content_hash"], count=999)

    delivered: list = []
    log = run_weekly_review(
        store=store, policy=AgentLedPolicy(), history=history, ledger=ledger,
        sender=delivered.append, model_call=model_call, now=NOW, skills_root=skills,
        state_path=tmp_path / "state.json", organization={"id": "org-1", "name": "Acme"},
    )
    assert log["ran"] and log["considered"] == ["notes"] and log["selected"] == ["notes"]
    assert log["delivery"][0]["delivered"] is True
    assert seen[0]["organization"]["name"] == "Acme"
    assert "You used this skill 9 times in the last 7 days." in delivered[0].notice.lines[4]
    # Suggested only after successful delivery; a second run is not due and stays quiet.
    assert history.recently_suggested("notes", delivered[0].rendering_hints["content_hash"], cooldown_days=14, at=NOW)
    second = run_weekly_review(
        store=store, policy=AgentLedPolicy(), history=history, ledger=ledger,
        sender=delivered.append, model_call=model_call, now=NOW + timedelta(hours=1), skills_root=skills,
        state_path=tmp_path / "state.json", organization={"id": "org-1"},
    )
    assert second["skipped_reason"] == "not_due" and len(delivered) == 1


def test_weekly_review_failed_delivery_does_not_mark_suggested(monkeypatch, tmp_path):
    skills = tmp_path / "skills"
    _skill(skills, "notes")
    _make_eligible(monkeypatch, skills)
    store = _store(tmp_path)
    _use(store, "notes", [0, 1, 2], per_day=3)
    history = SuggestionHistory(tmp_path / "h.json")

    def model_call(messages, schema):
        cand = json.loads(messages[1]["content"])["candidates"][0]
        return _review_json(cand["skill_name"], cand["content_hash"])

    def broken(_e):
        raise RuntimeError("offline")

    log = run_weekly_review(
        store=store, policy=AgentLedPolicy(delivery_retries=1), history=history,
        ledger=DeliveryLedger(tmp_path / "l.json"), sender=broken, model_call=model_call,
        now=NOW, skills_root=skills, state_path=tmp_path / "s.json", organization={"id": "org-1"},
        sleep=lambda _s: None,
    )
    assert log["delivery"][0]["delivered"] is False
    assert history.snapshot()["suggested"] == {}


def test_signed_out_user_gets_nothing_and_no_replay(monkeypatch, tmp_path):
    skills = tmp_path / "skills"
    _skill(skills, "notes")
    _make_eligible(monkeypatch, skills)
    store = _store(tmp_path, signed_in=False)
    # Usage while signed out is not even recorded by qualification.
    assert record_successful_use("notes", store=store, at=NOW) is None
    called = {"agent": 0, "sender": 0}

    def model_call(messages, schema):
        called["agent"] += 1
        return "{}"

    def sender(_e):
        called["sender"] += 1

    log = run_weekly_review(
        store=store, policy=AgentLedPolicy(), history=SuggestionHistory(tmp_path / "h.json"),
        ledger=DeliveryLedger(tmp_path / "l.json"), sender=sender, model_call=model_call,
        now=NOW, skills_root=skills, state_path=tmp_path / "s.json", force=True,
    )
    assert log["skipped_reason"] == "signed_out" and log["ran"] is False
    assert called == {"agent": 0, "sender": 0}
    assert not (tmp_path / "s.json").exists()  # nothing queued for later replay
    assert not (tmp_path / "l.json").exists()


def test_weekly_review_disabled_flag(monkeypatch, tmp_path):
    store = _store(tmp_path)
    log = run_weekly_review(
        store=store, policy=AgentLedPolicy(enabled=False), history=SuggestionHistory(tmp_path / "h.json"),
        ledger=DeliveryLedger(tmp_path / "l.json"), sender=lambda _e: None, model_call=lambda m, s: "{}",
        now=NOW, state_path=tmp_path / "s.json", force=True,
    )
    assert log["skipped_reason"] == "agent_led_disabled"


def test_weekly_review_rejects_agent_garbage(monkeypatch, tmp_path):
    skills = tmp_path / "skills"
    _skill(skills, "notes")
    _make_eligible(monkeypatch, skills)
    store = _store(tmp_path)
    _use(store, "notes", [0, 1, 2], per_day=3)
    sent: list = []
    log = run_weekly_review(
        store=store, policy=AgentLedPolicy(), history=SuggestionHistory(tmp_path / "h.json"),
        ledger=DeliveryLedger(tmp_path / "l.json"), sender=sent.append, model_call=lambda m, s: "nope",
        now=NOW, skills_root=skills, state_path=tmp_path / "s.json", organization={"id": "org-1"},
    )
    assert log["skipped_reason"].startswith("agent_output_rejected") and sent == []


# ---------------------------------------------------------------------------
# share flow: no publish without confirmation
# ---------------------------------------------------------------------------

def _package_json(name: str, content_hash: str, *, leak: bool = False) -> str:
    body = "# Notes\n" + ("token = ghp_" + "b" * 36 if leak else "Generalized body.")
    return json.dumps(
        {
            "schema_version": 1, "skill_name": name, "source_content_hash": content_hash,
            "editorial_name": "Release Notes", "plain_description": "Drafts release notes.",
            "files": [{"path": "SKILL.md", "content": body, "generalized_from_original": True}],
            "requirements": [{"kind": "command", "name": "gh", "purpose": "read merged PRs", "handoff": "install GitHub CLI"}],
            "setup_instructions": ["Install gh", "Authenticate gh"],
            "credential_handoff": ["A GitHub token with repo read scope, from your own account"],
            "compatibility_limits": ["Requires GitHub-hosted repositories"],
            "verification_step": "gh --version",
            "removed_or_generalized": ["Replaced a home directory path with $HOME"],
            "related_skills": [],
        }
    )


def test_share_flow_never_publishes_without_approval(tmp_path):
    skill = _skill(tmp_path / "skills", "notes")
    (skill / "SKILL.md").write_text(
        (skill / "SKILL.md").read_text() + "\nSee /home/alice/notes and token = ghp_" + "c" * 36 + "\n"
    )
    flow = ShareFlow(root=tmp_path / "flows")
    summary = flow.start(skill)
    assert summary["status"] == "prepass"
    assert any("Credential-shaped" in p for p in summary["portability_problems"])
    assert any("Organization-specific" in p for p in summary["portability_problems"])
    published = {"n": 0}

    def submit(package, _state):
        published["n"] += 1
        return {"draft_id": "d1", "published": False}

    with pytest.raises(RuntimeError):
        flow.approve(submit=submit)  # not packaged yet
    assert published["n"] == 0
    content_hash = summary["source_content_hash"]
    flow.package(model_call=lambda m, s: _package_json("notes", content_hash))
    assert flow.status == "awaiting_approval"
    assert flow.summary()["next_actions"] == ["approve", "request_changes", "cancel"]
    assert published["n"] == 0  # packaging alone publishes nothing
    resumed = ShareFlow(flow.flow_id, root=tmp_path / "flows")
    assert resumed.status == "awaiting_approval"
    resumed.request_changes("mention rate limits")
    assert resumed.status == "changes_requested"
    resumed.package(model_call=lambda m, s: _package_json("notes", content_hash))
    resumed.approve(submit=submit)
    assert published["n"] == 1 and resumed.status == "submitted"
    assert resumed.summary()["published"] is False  # draft only; owner review still required


def test_share_flow_rejects_leaked_credentials_in_package(tmp_path):
    skill = _skill(tmp_path / "skills", "notes")
    flow = ShareFlow(root=tmp_path / "flows")
    content_hash = flow.start(skill)["source_content_hash"]
    with pytest.raises(SchemaRejected):
        flow.package(model_call=lambda m, s: _package_json("notes", content_hash, leak=True))
    assert flow.status == "prepass"
    assert scan_credentials([{"path": "a", "content": "AKIA" + "A" * 16}])[0]["kind"] == "aws_access_key"


# ---------------------------------------------------------------------------
# install flow: success only after verification, resumable
# ---------------------------------------------------------------------------

def test_install_flow_reports_installed_only_after_verification(tmp_path):
    package = json.loads(_package_json("notes", "abcdef1234"))
    package["requirements"].append({"kind": "env_var", "name": "NOTES_TOKEN", "purpose": "auth"})
    flow = InstallFlow(root=tmp_path / "installs")
    flow.start(skill_id="sk-1", version=2, package=package)
    summary = flow.check_prerequisites(env={})
    assert summary["step"] == "setup" and any(m["name"] == "NOTES_TOKEN" for m in summary["missing"])
    assert summary["installed"] is False
    resumed = InstallFlow(flow.flow_id, root=tmp_path / "installs")
    assert resumed.step == "setup"
    resumed.mark_setup_complete("NOTES_TOKEN")
    for item in resumed.state["prerequisites"]:
        item["status"] = "present"
    resumed.state["step"] = "apply"
    resumed.apply(lambda _state: {"installed": True, "managed_path": "x"})
    assert resumed.step == "verify" and resumed.installed is False
    failed = resumed.verify(runner=lambda cmd: (False, "boom"))
    assert failed["installed"] is False and failed["files_installed"] is True
    assert failed["message"] == "Files installed; setup verification is incomplete."
    passed = resumed.verify(runner=lambda cmd: (True, "ok"))
    assert passed["installed"] is True and passed["message"] == "Installed and verified."
    assert detect_prerequisites([{"kind": "command", "name": "definitely-not-a-real-binary-xyz"}])[0]["status"] == "missing"
