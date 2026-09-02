import json
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from hermes_wisdom.qualification import (
    HIGH_USAGE_CONSECUTIVE_BUSINESS_DAYS,
    RETENTION_DAYS,
    _emit_candidate,
    _classify_ambiguous,
    process_due_stability_jobs,
    record_mutation,
    record_successful_use,
    snapshot_tree,
)
from hermes_wisdom.store import WisdomStore


def _configured_store(tmp_path: Path) -> WisdomStore:
    store = WisdomStore(tmp_path / "state")
    store.installation_identity()
    store.verify_installation_identity("org-1")
    return store


def _skill(tmp_path: Path) -> Path:
    path = tmp_path / "skills" / "learned-skill"
    path.mkdir(parents=True)
    (path / "SKILL.md").write_text(
        "---\n"
        "name: learned-skill\n"
        "description: Use when retaining a learned workflow.\n"
        "metadata:\n"
        "  hermes:\n"
        "    editorial_name: Learned Workflow\n"
        "    editorial_description: Reuse a workflow refined through practice.\n"
        "---\n# One\n",
        encoding="utf-8",
    )
    return path


def _eligible(monkeypatch, skill: Path) -> None:
    monkeypatch.setattr(
        "hermes_wisdom.qualification.get_skills_dir", lambda: skill.parent
    )
    monkeypatch.setattr(
        "hermes_wisdom.qualification._find_skill_dir", lambda _name: skill
    )
    monkeypatch.setattr("hermes_wisdom.qualification.is_bundled", lambda _name: False)
    monkeypatch.setattr(
        "hermes_wisdom.qualification.is_hub_installed", lambda _name: False
    )


def test_high_usage_threshold_uses_consecutive_business_days_and_deduplicates(
    monkeypatch, tmp_path: Path
):
    skill = _skill(tmp_path)
    _eligible(monkeypatch, skill)
    store = _configured_store(tmp_path)
    monkeypatch.setattr(
        "hermes_wisdom.qualification.get_timezone",
        lambda: ZoneInfo("Australia/Brisbane"),
    )
    start = datetime(2026, 8, 3, 9, tzinfo=ZoneInfo("Australia/Brisbane"))
    business_offsets = (0, 1, 2, 3, 4, 7, 8)

    for offset in business_offsets[:-1]:
        assert (
            record_successful_use(
                "learned-skill", at=start + timedelta(days=offset), store=store
            )
            is None
        )
    event_id = record_successful_use(
        "learned-skill",
        at=start + timedelta(days=business_offsets[-1]),
        session_id="session-1",
        task_id="task-1",
        store=store,
    )
    assert event_id
    assert (
        record_successful_use(
            "learned-skill",
            at=start + timedelta(days=business_offsets[-1] + 1),
            store=store,
        )
        is None
    )
    events = store.local_events(kind="wisdom.candidate")
    assert len(events) == 1
    assert events[0]["session_id"] == "session-1"
    assert events[0]["payload"]["networked"] is False
    assert events[0]["payload"]["consent_required"] is True
    assert events[0]["payload"]["editorial_name"] == "Learned Workflow"
    assert (
        events[0]["payload"]["editorial_description"]
        == "Reuse a workflow refined through practice."
    )
    assert events[0]["payload"]["local_reasons"] == {
        "consecutive_business_days": HIGH_USAGE_CONSECUTIVE_BUSINESS_DAYS,
        "business_day_timezone": "Australia/Brisbane",
        "business_week": "monday_friday",
    }
    with store.transaction() as db:
        reviews = db.execute(
            "SELECT skill_id,content_hash,state FROM professionalism_review"
        ).fetchall()
    assert [tuple(row) for row in reviews] == [
        (events[0]["skill_id"], events[0]["content_hash"], "pending")
    ]


def test_candidate_hash_tracks_editorial_enrichment(monkeypatch, tmp_path: Path):
    skill = _skill(tmp_path)
    _eligible(monkeypatch, skill)
    store = _configured_store(tmp_path)
    old_hash, old_tree = snapshot_tree(skill)
    skill_id = store.register_skill(
        skill,
        content_hash=old_hash,
        source_kind="local",
        tree=old_tree,
        snapshot_text="workflow",
    )

    def enrich(path: Path):
        skill_md = path / "SKILL.md"
        skill_md.write_text(
            skill_md.read_text(encoding="utf-8").replace(
                "editorial_name: Learned Workflow",
                "editorial_name: Human Friendly Workflow",
            ),
            encoding="utf-8",
        )
        return {
            "editorial_name": "Human Friendly Workflow",
            "editorial_description": "Reuse a workflow refined through practice.",
            "changed": True,
        }

    monkeypatch.setattr(
        "hermes_wisdom.qualification.ensure_skill_editorial_metadata", enrich
    )
    event_id = _emit_candidate(
        store,
        skill_id=skill_id,
        skill_name="learned-skill",
        content_hash=old_hash,
        qualification="high_usage",
        local_reasons={},
        session_id=None,
        task_id=None,
    )

    assert event_id
    new_hash, _tree = snapshot_tree(skill)
    event = store.local_events(kind="wisdom.candidate")[0]
    assert new_hash != old_hash
    assert event["content_hash"] == new_hash
    assert store.local_skill(skill_id)["current_hash"] == new_hash


def test_weekend_usage_neither_advances_nor_breaks_business_day_streak(
    monkeypatch, tmp_path: Path
):
    skill = _skill(tmp_path)
    _eligible(monkeypatch, skill)
    store = _configured_store(tmp_path)
    monkeypatch.setattr(
        "hermes_wisdom.qualification.get_timezone",
        lambda: ZoneInfo("Australia/Brisbane"),
    )
    start = datetime(2026, 8, 3, 9, tzinfo=ZoneInfo("Australia/Brisbane"))

    for offset in (0, 1, 2, 3, 4, 5, 6, 7):
        assert (
            record_successful_use(
                "learned-skill", at=start + timedelta(days=offset), store=store
            )
            is None
        )
    assert record_successful_use(
        "learned-skill", at=start + timedelta(days=8), store=store
    )


def test_midweek_gap_resets_then_rebuilds_business_day_streak(
    monkeypatch, tmp_path: Path
):
    skill = _skill(tmp_path)
    _eligible(monkeypatch, skill)
    store = _configured_store(tmp_path)
    monkeypatch.setattr(
        "hermes_wisdom.qualification.get_timezone",
        lambda: ZoneInfo("Australia/Brisbane"),
    )
    start = datetime(2026, 8, 3, 9, tzinfo=ZoneInfo("Australia/Brisbane"))

    for offset in (0, 1, 3, 4, 7, 8, 9, 10):
        assert (
            record_successful_use(
                "learned-skill", at=start + timedelta(days=offset), store=store
            )
            is None
        )
    assert record_successful_use(
        "learned-skill", at=start + timedelta(days=11), store=store
    )


def test_profile_timezone_controls_the_qualification_day(monkeypatch, tmp_path: Path):
    skill = _skill(tmp_path)
    _eligible(monkeypatch, skill)
    store = _configured_store(tmp_path)
    monkeypatch.setattr(
        "hermes_wisdom.qualification.get_timezone",
        lambda: ZoneInfo("Australia/Brisbane"),
    )

    # Sunday in UTC is already Monday in the configured profile timezone.
    record_successful_use(
        "learned-skill",
        at=datetime(2026, 8, 2, 16, tzinfo=timezone.utc),
        store=store,
    )
    with store.transaction() as db:
        row = db.execute(
            "SELECT day_local,timezone_name FROM usage_day"
        ).fetchone()
    assert tuple(row) == ("2026-08-03", "Australia/Brisbane")


def test_usage_retention_is_bounded(monkeypatch, tmp_path: Path):
    skill = _skill(tmp_path)
    _eligible(monkeypatch, skill)
    store = _configured_store(tmp_path)
    monkeypatch.setattr(
        "hermes_wisdom.qualification.get_timezone", lambda: ZoneInfo("UTC")
    )
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for offset in range(RETENTION_DAYS + 8):
        record_successful_use(
            "learned-skill", at=start + timedelta(days=offset * 2), store=store
        )
    with store.transaction() as db:
        rows = db.execute(
            "SELECT day_local FROM usage_day ORDER BY day_local"
        ).fetchall()
    assert len(rows) <= (RETENTION_DAYS + 1) // 2
    assert rows[0][0] >= (start.date() + timedelta(days=50)).isoformat()


def test_structural_refinements_schedule_restart_safe_stability(
    monkeypatch, tmp_path: Path
):
    skill = _skill(tmp_path)
    _eligible(monkeypatch, skill)
    store = _configured_store(tmp_path)
    start = datetime(2026, 8, 1, tzinfo=timezone.utc)
    record_successful_use("learned-skill", at=start, store=store)

    for index in range(3):
        refs = skill / "refs"
        refs.mkdir(exist_ok=True)
        (refs / f"decision-{index}.md").write_text(
            f"decision {index}", encoding="utf-8"
        )
        record_mutation(
            "learned-skill", at=start + timedelta(days=index + 1), store=store
        )

    restarted = WisdomStore(store.root)
    event_id = record_successful_use(
        "learned-skill",
        at=start + timedelta(days=10),
        session_id="session-stable",
        store=restarted,
    )
    assert event_id
    event = restarted.local_events(kind="wisdom.candidate")[0]
    assert event["qualification"] == "refinement"
    assert event["payload"]["local_reasons"]["meaningful_refinements"] == 3


def test_due_stability_runs_without_another_use_and_keeps_session(
    monkeypatch, tmp_path: Path
):
    skill = _skill(tmp_path)
    _eligible(monkeypatch, skill)
    store = _configured_store(tmp_path)
    start = datetime(2026, 8, 1, tzinfo=timezone.utc)
    record_successful_use("learned-skill", at=start, store=store)
    for index in range(3):
        refs = skill / "refs"
        refs.mkdir(exist_ok=True)
        (refs / f"step-{index}.md").write_text(str(index), encoding="utf-8")
        record_mutation(
            "learned-skill",
            at=start + timedelta(days=index + 1),
            session_id="origin-session",
            task_id="origin-task",
            store=store,
        )

    emitted = process_due_stability_jobs(
        store=WisdomStore(store.root), at=start + timedelta(days=10)
    )

    assert len(emitted) == 1
    event = store.local_events(kind="wisdom.candidate")[0]
    assert event["session_id"] == "origin-session"
    assert event["task_id"] == "origin-task"
    assert process_due_stability_jobs(store=store, at=start + timedelta(days=11)) == []


def test_due_stability_waits_for_use_within_the_qualification_window(
    monkeypatch, tmp_path: Path
):
    skill = _skill(tmp_path)
    _eligible(monkeypatch, skill)
    store = _configured_store(tmp_path)
    start = datetime(2026, 8, 1, tzinfo=timezone.utc)
    record_mutation("learned-skill", at=start, store=store)
    for index in range(3):
        refs = skill / "refs"
        refs.mkdir(exist_ok=True)
        (refs / f"late-use-{index}.md").write_text(str(index), encoding="utf-8")
        record_mutation(
            "learned-skill", at=start + timedelta(days=index + 1), store=store
        )

    assert process_due_stability_jobs(store=store, at=start + timedelta(days=10)) == []
    assert len(store.due_stability_jobs((start + timedelta(days=10)).isoformat())) == 1

    event_id = record_successful_use(
        "learned-skill", at=start + timedelta(days=11), store=store
    )
    assert event_id
    assert store.local_events(kind="wisdom.candidate")[0]["qualification"] == (
        "refinement"
    )


def test_ambiguous_classifier_receives_only_a_bounded_before_after_diff(
    monkeypatch,
):
    captured = {}
    auxiliary = types.ModuleType("agent.auxiliary_client")

    def call_llm(**kwargs):
        captured.update(kwargs)
        return {"content": "meaningful"}

    auxiliary.call_llm = call_llm
    auxiliary.extract_content_or_reasoning = lambda response: response["content"]
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", auxiliary)

    assert (
        _classify_ambiguous(
            "# Procedure\nUse the old endpoint.\n",
            "# Procedure\nUse the new endpoint and verify the receipt.\n",
            {"added": [], "removed": [], "changed": ["SKILL.md"]},
        )
        == "meaningful"
    )
    payload = json.loads(captured["messages"][1]["content"])
    semantic = payload["untrusted_semantic_diff"]
    assert "Use the old endpoint" in semantic
    assert "Use the new endpoint" in semantic
    assert "usage" not in payload
    assert len(semantic) <= 16000


def test_ambiguous_classifier_failure_is_conservative(monkeypatch, tmp_path: Path):
    skill = _skill(tmp_path)
    _eligible(monkeypatch, skill)
    store = _configured_store(tmp_path)
    now = datetime(2026, 8, 1, tzinfo=timezone.utc)
    record_successful_use("learned-skill", at=now, store=store)
    (skill / "SKILL.md").write_text(
        "---\nname: learned-skill\n---\n# Typo only\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        "hermes_wisdom.qualification._classify_ambiguous",
        lambda *_args: "non_meaningful",
    )
    record_mutation("learned-skill", at=now + timedelta(days=1), store=store)
    with store.transaction() as db:
        row = db.execute("SELECT classification FROM refinement").fetchone()
    assert row[0] == "non_meaningful"
    assert store.due_stability_jobs((now + timedelta(days=30)).isoformat()) == []


def test_dismissal_suppresses_exact_content_but_stronger_path_can_resuggest(
    tmp_path: Path,
):
    store = _configured_store(tmp_path)
    skill = _skill(tmp_path)
    skill_id = store.register_skill(
        skill, content_hash="sha256:one", source_kind="local"
    )
    first = _emit_candidate(
        store,
        skill_id=skill_id,
        skill_name="learned-skill",
        content_hash="sha256:one",
        qualification="refinement",
        local_reasons={},
        session_id=None,
        task_id=None,
    )
    stronger = _emit_candidate(
        store,
        skill_id=skill_id,
        skill_name="learned-skill",
        content_hash="sha256:one",
        qualification="high_usage",
        local_reasons={},
        session_id=None,
        task_id=None,
    )
    assert first and stronger
    store.dismiss_candidate(skill_id, "sha256:one")
    assert store.local_events(kind="wisdom.candidate") == []
    assert (
        _emit_candidate(
            store,
            skill_id=skill_id,
            skill_name="learned-skill",
            content_hash="sha256:one",
            qualification="high_usage",
            local_reasons={},
            session_id=None,
            task_id=None,
        )
        is None
    )
    assert _emit_candidate(
        store,
        skill_id=skill_id,
        skill_name="learned-skill",
        content_hash="sha256:two",
        qualification="high_usage",
        local_reasons={},
        session_id=None,
        task_id=None,
    )
