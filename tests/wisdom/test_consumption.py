from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_wisdom.client import Feed, WisdomConflict
from hermes_wisdom.compatibility import CompatibilityResult
from hermes_wisdom.consumption import WisdomConsumption, _tree_hashes
from hermes_wisdom.contract import (
    PackageManifest,
    SystemSpecification,
    canonical_json_bytes,
)
from hermes_wisdom.package import PackagePolicyError, verify_content_files
from hermes_wisdom.store import WisdomStore


def _spec(*, credentials: list[str] | None = None) -> SystemSpecification:
    return SystemSpecification.model_validate({
        "hermes": {"minimum_version": "0.1.0"},
        "credentials": credentials or [],
    })


def _files(version: int, *, credentials: list[str] | None = None):
    manifest = PackageManifest(
        name="managed-skill", requirements=_spec(credentials=credentials)
    )
    return [
        ("SKILL.md", "file", f"# Managed v{version}\n".encode()),
        (
            "skill.manifest.json",
            "file",
            canonical_json_bytes(manifest.model_dump(mode="json")),
        ),
    ]


class Client:
    def __init__(self, files, *, mode="REQUIRED", fail_record=False, record_error=None):
        self.files = files
        self.mode = mode
        self.fail_record = fail_record
        self.record_error = record_error
        self.skill_state = "active"
        self.takedown_generation = 0
        self.latest_version = 2
        self.installed_version = 1
        self.recorded: list[dict] = []
        self.deactivated: list[tuple[str, str]] = []
        self.feed_pages: list[Feed] = []
        self.drafts = []
        self.discovery = []

    def installations(self, _identity):
        return [
            {
                "skill_id": "skill-1",
                "installed_version": self.installed_version,
                "latest_version": self.latest_version,
                "update_mode": self.mode,
                "skill_state": self.skill_state,
                "takedown_generation": self.takedown_generation,
            }
        ]

    def content(
        self,
        _skill_id,
        _version,
        *,
        installation_id,
        takedown_generation,
    ):
        assert installation_id.startswith("hwi_")
        assert takedown_generation == self.takedown_generation
        _records, content_hash = verify_content_files(self.files)
        return SimpleNamespace(content_hash=content_hash), self.files

    def record_install(self, **kwargs):
        if self.record_error is not None:
            raise self.record_error
        if self.fail_record:
            self.fail_record = False
            raise RuntimeError("network down")
        self.recorded.append(kwargs)
        return SimpleNamespace(effective_update_mode=self.mode)

    def list_drafts(self):
        return self.drafts

    def list_skills(self, *, cursor=None):
        assert cursor is None
        return SimpleNamespace(skills=self.discovery, next_cursor=None)

    def deactivate_install(self, installation_id, skill_id):
        self.deactivated.append((installation_id, skill_id))
        return SimpleNamespace(state="inactive")

    def feed(self, _cursor, *, installation_id):
        assert installation_id.startswith("hwi_")
        if self.feed_pages:
            return self.feed_pages.pop(0)
        return Feed(events=[], next_cursor="cursor-1", has_more=False)


def _manager(monkeypatch, tmp_path: Path, *, client: Client):
    skills = tmp_path / "skills"
    target = skills / "_wisdom" / "org-1" / "managed-skill"
    target.mkdir(parents=True)
    for name, _mode, body in _files(1):
        destination = target / name
        destination.write_bytes(body)
    monkeypatch.setattr("hermes_wisdom.consumption.get_skills_dir", lambda: skills)
    monkeypatch.setattr(
        "hermes_wisdom.consumption.evaluate",
        lambda _specification, _local: CompatibilityResult(
            "compatible", (), (), (), ()
        ),
    )
    store = WisdomStore(tmp_path / "wisdom")
    store.installation_identity()
    store.verify_installation_identity("org-1")
    store.record_install({
        "skill_id": "skill-1",
        "org_id": "org-1",
        "slug": "managed-skill",
        "version": 1,
        "content_hash": "sha256:old",
        "baseline": _tree_hashes(target),
        "target_path": str(target),
        "update_mode": client.mode,
    })
    scan = lambda _path: {
        "guard": {"allowed": True, "findings": [], "reason": None},
        "skill_evaluator": {"status": "disabled", "findings": []},
    }
    return WisdomConsumption(store=store, client=client, scan=scan, config={}), target


def _telegram_home(monkeypatch, chat_id: str) -> None:
    from gateway.config import Platform

    home = SimpleNamespace(chat_id=chat_id, thread_id=None)
    config = SimpleNamespace(
        get_home_channel=lambda platform: (
            home if platform == Platform.TELEGRAM else None
        )
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)


def test_required_update_preserves_modified_bytes_before_converging(
    monkeypatch, tmp_path: Path
):
    client = Client(_files(2), mode="REQUIRED")
    manager, target = _manager(monkeypatch, tmp_path, client=client)
    (target / "SKILL.md").write_text("# locally changed\n", encoding="utf-8")

    plan = manager.update_plan("skill-1")
    assert plan["modified"] is True
    assert plan["auto_allowed"] is True
    result = manager.update_apply(plan["receipt"], automatic=True)

    fork = Path(result["preserved_fork"])
    assert (fork / "SKILL.md").read_text() == "# locally changed\n"
    assert (target / "SKILL.md").read_text() == "# Managed v2\n"
    assert manager.store.installation("skill-1")["version"] == 2


def test_required_update_with_scan_findings_is_never_applied_automatically(
    monkeypatch, tmp_path: Path
):
    client = Client(_files(2), mode="REQUIRED")
    manager, target = _manager(monkeypatch, tmp_path, client=client)
    manager.scan = lambda _path: {
        "guard": {
            "allowed": True,
            "findings": [{"severity": "medium", "message": "review this"}],
            "reason": None,
        },
        "skill_evaluator": {"status": "disabled", "findings": []},
    }

    result = manager.check(apply_automatic=True)

    assert result["installations"][0]["state"] == "update_available"
    assert result["installations"][0]["plan"]["auto_allowed"] is False
    assert (target / "SKILL.md").read_text(encoding="utf-8") == "# Managed v1\n"


def test_repeated_update_checks_reuse_the_exact_unapplied_plan(
    monkeypatch, tmp_path: Path
):
    client = Client(_files(2), mode="MANUAL")
    manager, _target = _manager(monkeypatch, tmp_path, client=client)

    first = manager.update_plan("skill-1")
    second = manager.update_plan("skill-1")

    assert second["receipt"] == first["receipt"]
    plans = manager.store.root / "update-plans"
    assert len(list(plans.glob("wup_*.json"))) == 1
    assert len([path for path in plans.iterdir() if path.is_dir()]) == 1


def test_required_fork_recovery_reuses_the_same_durable_name(
    monkeypatch, tmp_path: Path
):
    client = Client(_files(2), mode="REQUIRED")
    manager, target = _manager(monkeypatch, tmp_path, client=client)
    (target / "SKILL.md").write_text("# locally changed\n", encoding="utf-8")
    plan = manager.update_plan("skill-1")
    real_advance = manager.store.advance
    failed = False

    def advance_once(operation_id, phase, *, done=False):
        nonlocal failed
        if not failed and phase == "fork_preserved":
            failed = True
            raise OSError("injected post-fork journal failure")
        return real_advance(operation_id, phase, done=done)

    monkeypatch.setattr(manager.store, "advance", advance_once)
    with pytest.raises(OSError, match="post-fork"):
        manager.update_apply(plan["receipt"], automatic=True)
    payload = json.loads(manager.store.pending_operations()[0]["payload_json"])
    fork = Path(payload["fork_path"])
    assert fork.is_dir()
    assert (fork / "SKILL.md").read_text() == "# locally changed\n"

    monkeypatch.setattr(manager.store, "advance", real_advance)
    result = manager.update_apply(plan["receipt"], automatic=True)
    assert result["updated"] is True
    assert [path for path in fork.parent.glob("managed-skill-local-fork*")] == [fork]


def test_sensitive_expansion_pauses_every_update_mode(monkeypatch, tmp_path: Path):
    client = Client(_files(2, credentials=["production-token"]), mode="REQUIRED")
    manager, _target = _manager(monkeypatch, tmp_path, client=client)
    plan = manager.update_plan("skill-1")
    assert plan["auto_allowed"] is False
    assert plan["sensitive_expansion"] == ["new credentials: production-token"]
    with pytest.raises(PackagePolicyError, match="sensitive requirements"):
        manager.update_apply(plan["receipt"])


def test_update_resumes_after_gateway_record_failure(monkeypatch, tmp_path: Path):
    client = Client(_files(2), fail_record=True)
    manager, target = _manager(monkeypatch, tmp_path, client=client)
    plan = manager.update_plan("skill-1")
    with pytest.raises(RuntimeError, match="network down"):
        manager.update_apply(plan["receipt"])
    assert (target / "SKILL.md").read_text() == "# Managed v2\n"
    assert manager.store.pending_operations()[0]["phase"] == "local_ledger_committed"

    assert manager.recover() == ["skill-1"]
    assert manager.store.pending_operations() == []
    assert client.recorded[0]["version"] == 2


def test_apply_rejects_policy_change_before_touching_managed_files(
    monkeypatch, tmp_path: Path
):
    client = Client(_files(2), mode="MANUAL")
    manager, target = _manager(monkeypatch, tmp_path, client=client)
    plan = manager.update_plan("skill-1")
    client.mode = "REQUIRED"

    with pytest.raises(PackagePolicyError, match="policy changed"):
        manager.update_apply(plan["receipt"])

    assert (target / "SKILL.md").read_text() == "# Managed v1\n"
    assert manager.store.pending_operations() == []


def test_apply_rechecks_current_compatibility_and_scan(monkeypatch, tmp_path: Path):
    client = Client(_files(2), mode="MANUAL")
    manager, target = _manager(monkeypatch, tmp_path, client=client)
    plan = manager.update_plan("skill-1")
    monkeypatch.setattr(
        "hermes_wisdom.consumption.evaluate",
        lambda _specification, _local: CompatibilityResult(
            "blocked_pending_action", (), (), (), ("new local gap",)
        ),
    )

    with pytest.raises(PackagePolicyError, match="blocked compatibility"):
        manager.update_apply(plan["receipt"])

    assert (target / "SKILL.md").read_text() == "# Managed v1\n"
    assert manager.store.pending_operations() == []


def test_permanent_nonterminal_gateway_rejection_restores_previous_install(
    monkeypatch, tmp_path: Path
):
    client = Client(
        _files(2),
        mode="MANUAL",
        record_error=WisdomConflict(
            "policy missing", status=409, code="policy_not_configured"
        ),
    )
    manager, target = _manager(monkeypatch, tmp_path, client=client)
    plan = manager.update_plan("skill-1")

    with pytest.raises(WisdomConflict, match="policy missing"):
        manager.update_apply(plan["receipt"])

    assert (target / "SKILL.md").read_text() == "# Managed v1\n"
    restored = manager.store.installation("skill-1")
    assert restored["version"] == 1
    assert restored["state"] == "active"
    assert manager.store.pending_operations() == []


def test_terminal_gateway_rejection_quarantines_and_deactivates(
    monkeypatch, tmp_path: Path
):
    client = Client(
        _files(2),
        mode="MANUAL",
        record_error=WisdomConflict("taken down", status=409, code="skill_taken_down"),
    )
    manager, target = _manager(monkeypatch, tmp_path, client=client)
    plan = manager.update_plan("skill-1")

    with pytest.raises(WisdomConflict, match="taken down"):
        manager.update_apply(plan["receipt"])

    assert not target.exists()
    assert manager.store.installation("skill-1")["state"] == "inactive"
    assert manager.store.pending_operations() == []


def test_returned_draft_invalidates_receipt_and_preserves_moderator_note(
    monkeypatch, tmp_path: Path
):
    client = Client(_files(2))
    manager, _target = _manager(monkeypatch, tmp_path, client=client)
    local_skill = tmp_path / "owner-skill"
    local_skill.mkdir()
    skill_id = manager.store.register_skill(
        local_skill, content_hash="sha256:source", source_kind="local"
    )
    manager.store.record_draft({
        "id": "draft-1",
        "skill_id": skill_id,
        "source_hash": "sha256:source",
        "overlay_path": str(local_skill),
        "state": "pending_moderation",
        "description": "Owner copy",
        "content_hash": "sha256:content",
        "description_hash": "sha256:description",
        "manifest_hash": "sha256:manifest",
    })
    manager.store.save_receipt(
        draft_id="draft-1",
        server_revision="revision-1",
        content_hash="sha256:content",
        description_hash="sha256:description",
        manifest_hash="sha256:manifest",
    )
    client.drafts = [
        SimpleNamespace(
            id="draft-1",
            slug="owner-skill",
            state="changes_requested",
            moderationNote="Remove the environment-specific hostname.",
            moderationDeciderUserId="moderator-1",
            moderationDecidedAt="2026-08-25T00:00:00Z",
        )
    ]

    assert manager.poll_owner_decisions() == {"inserted": 1}
    assert manager.store.receipt("draft-1") is None
    notice = manager.store.feed_events()[0]
    assert notice["payload"]["moderation_note"] == (
        "Remove the environment-specific hostname."
    )
    notification = manager.notifications()["events"][0]
    assert notification["category"] == "publication_decision"
    assert notification["skill_name"] == "owner-skill"
    assert notification["portal_url"].endswith("/orgs/org-1/wisdom/review/draft-1")


def test_portal_submission_retires_exact_candidate_without_final_notice(
    monkeypatch, tmp_path: Path
):
    client = Client(_files(2))
    manager, _target = _manager(monkeypatch, tmp_path, client=client)
    local_skill = tmp_path / "owner-skill"
    local_skill.mkdir()
    skill_id = manager.store.register_skill(
        local_skill, content_hash="sha256:source", source_kind="local"
    )
    event_id = manager.store.emit_local_event(
        kind="wisdom.candidate",
        skill_id=skill_id,
        content_hash="sha256:source",
        payload={"skill_name": "owner-skill"},
        session_id="telegram-session",
        task_id="task-1",
        qualification="high_usage",
    )
    assert event_id is not None
    manager.store.record_draft({
        "id": "draft-1",
        "skill_id": skill_id,
        "source_hash": "sha256:source",
        "overlay_path": str(local_skill),
        "state": "ready",
        "description": "Owner copy",
        "content_hash": "sha256:content",
        "description_hash": "sha256:description",
        "manifest_hash": "sha256:manifest",
    })
    client.drafts = [
        SimpleNamespace(
            id="draft-1",
            slug="owner-skill",
            state="pending_moderation",
            moderationNote=None,
            moderationDeciderUserId=None,
            moderationDecidedAt=None,
        )
    ]

    assert manager.poll_owner_decisions() == {"inserted": 0}
    assert manager.store.draft("draft-1")["state"] == "pending_moderation"
    assert manager.store.local_event(event_id)["state"] == "handled"
    assert manager.notifications() == {"events": []}


def test_update_recovers_when_swap_won_before_journal_advance(
    monkeypatch, tmp_path: Path
):
    client = Client(_files(2))
    manager, target = _manager(monkeypatch, tmp_path, client=client)
    plan = manager.update_plan("skill-1")
    real_advance = manager.store.advance
    failed = False

    def advance_once(operation_id, phase, *, done=False):
        nonlocal failed
        if not failed and phase == "files_committed":
            failed = True
            raise OSError("injected journal failure")
        return real_advance(operation_id, phase, done=done)

    monkeypatch.setattr(manager.store, "advance", advance_once)
    with pytest.raises(OSError, match="journal"):
        manager.update_apply(plan["receipt"])
    assert (target / "SKILL.md").read_text() == "# Managed v2\n"
    assert manager.store.pending_operations()[0]["phase"] == "fork_preserved"

    monkeypatch.setattr(manager.store, "advance", real_advance)
    assert manager.recover() == ["skill-1"]
    assert manager.store.installation("skill-1")["version"] == 2


def test_update_recovers_when_old_tree_moved_before_new_tree_swap(
    monkeypatch, tmp_path: Path
):
    client = Client(_files(2), mode="MANUAL")
    manager, target = _manager(monkeypatch, tmp_path, client=client)
    plan = manager.update_plan("skill-1")
    real_replace = __import__("os").replace
    failed = False

    def replace_once(source, destination):
        nonlocal failed
        if (
            not failed
            and Path(source).name.startswith(".wup_")
            and Path(destination) == target
        ):
            failed = True
            raise OSError("injected second-swap failure")
        return real_replace(source, destination)

    monkeypatch.setattr("hermes_wisdom.consumption.os.replace", replace_once)
    with pytest.raises(OSError, match="second-swap"):
        manager.update_apply(plan["receipt"])
    assert not target.exists()
    assert manager.store.pending_operations()[0]["phase"] == "fork_preserved"

    monkeypatch.setattr("hermes_wisdom.consumption.os.replace", real_replace)
    result = manager.update_apply(plan["receipt"])
    assert result["updated"] is True
    assert (target / "SKILL.md").read_text() == "# Managed v2\n"


def test_uninstall_validates_target_and_preserves_recoverable_trash(
    monkeypatch, tmp_path: Path
):
    client = Client(_files(2))
    manager, target = _manager(monkeypatch, tmp_path, client=client)
    result = manager.uninstall("skill-1")
    assert not target.exists()
    assert Path(result["recoverable_path"]).is_dir()
    assert manager.store.installation("skill-1")["state"] == "inactive"
    assert client.deactivated == [(manager.store.installation_identity(), "skill-1")]


def test_uninstall_recovery_rejects_a_traversal_slug(monkeypatch, tmp_path: Path):
    client = Client(_files(2))
    manager, target = _manager(monkeypatch, tmp_path, client=client)
    payload = {
        "skill_id": "skill-1",
        "slug": "../../outside",
        "target_path": str(target),
    }
    operation = manager.store.journal("uninstall", "skill-1", "validated", payload)

    with pytest.raises(PackagePolicyError, match="recovery path is invalid"):
        manager._resume_uninstall(operation, payload)

    assert target.is_dir()


def test_takedown_check_preserves_existing_local_installation(
    monkeypatch, tmp_path: Path
):
    client = Client(_files(2))
    client.skill_state = "taken_down"
    manager, target = _manager(monkeypatch, tmp_path, client=client)

    result = manager.check()

    assert result["installations"] == [
        {
            "skill_id": "skill-1",
            "state": "taken_down",
            "local_installation_preserved": True,
        }
    ]
    assert target.is_dir()
    assert manager.store.installation("skill-1")["state"] == "active"


def test_feed_cursor_is_durable_deduplicated_and_telegram_uses_home_target(
    monkeypatch, tmp_path: Path
):
    client = Client(_files(2))
    manager, _target = _manager(monkeypatch, tmp_path, client=client)
    event = {
        "event_id": "event-1",
        "kind": "installation_updated",
        "skill_id": "skill-1",
        "version": 2,
        "takedown_generation": 0,
        "installation_id": manager.store.installation_identity(),
        "update_mode": "REQUIRED",
        "occurred_at": "2026-08-24T00:00:00+00:00",
    }
    client.feed_pages = [
        Feed.model_validate({
            "events": [event],
            "next_cursor": "cursor-2",
            "has_more": False,
        })
    ]
    assert manager.poll_feed()["inserted"] == 1
    client.feed_pages = [
        Feed.model_validate({
            "events": [event],
            "next_cursor": "cursor-2",
            "has_more": False,
        })
    ]
    assert manager.poll_feed()["inserted"] == 0
    assert WisdomStore(manager.store.root).feed_cursor() == "cursor-2"
    _telegram_home(monkeypatch, "123456")

    calls = []
    monkeypatch.setattr(
        "tools.send_message_tool.send_telegram_notification_pane",
        lambda **kwargs: calls.append(kwargs) or {"success": True},
    )
    delivered = manager.dispatch_telegram()
    assert delivered == {"attempted": True, "delivered": 1}
    assert "managed-skill" in calls[0]["message"]
    assert "skill-1" not in calls[0]["message"]
    assert calls[0]["button_rows"] == [
        [
            {
                "label": "View ↗",
                "url": "https://portal.nousresearch.com/orgs/org-1/wisdom/skills/skill-1?version=2",
            }
        ]
    ]
    assert calls[0]["items"] == [
        {
            "heading": "✅ Updated on this device",
            "detail": (
                "managed-skill · v2\n"
                "Security check: Unavailable\n"
                "No known matches detected is not a security certification.\n\n"
                "Professionalism check (agent-assessed, advisory): Unavailable"
            ),
        }
    ]


def test_telegram_update_available_offers_verified_update_action(
    monkeypatch, tmp_path: Path
):
    client = Client(_files(2))
    manager, _target = _manager(monkeypatch, tmp_path, client=client)
    client.feed_pages = [
        Feed.model_validate({
            "events": [
                {
                    "event_id": "event-update",
                    "kind": "updated",
                    "skill_id": "skill-1",
                    "version": 2,
                    "takedown_generation": 0,
                    "installation_id": None,
                    "update_mode": "MANUAL",
                    "occurred_at": "2026-08-24T00:00:00+00:00",
                }
            ],
            "next_cursor": "cursor-update",
            "has_more": False,
        })
    ]
    assert manager.poll_feed()["inserted"] == 1
    _telegram_home(monkeypatch, "123456")

    calls = []
    monkeypatch.setattr(
        "tools.send_message_tool.send_telegram_notification_pane",
        lambda **kwargs: calls.append(kwargs) or {"success": True},
    )

    assert manager.dispatch_telegram() == {"attempted": True, "delivered": 1}
    assert calls[0]["button_rows"] == [
        [
            {
                "label": "View ↗",
                "url": "https://portal.nousresearch.com/orgs/org-1/wisdom/skills/skill-1?version=2",
            },
            {
                "label": "Update",
                "callback_data": "wi:plan:update:skill-1",
            },
        ]
    ]
    assert calls[0]["items"] == [
        {
            "heading": "⬆️ Update available",
            "detail": (
                "managed-skill · v2\n"
                "Security check: Unavailable\n"
                "No known matches detected is not a security certification.\n\n"
                "Professionalism check (agent-assessed, advisory): Unavailable"
            ),
        }
    ]


def test_notifications_resolve_org_skill_names_filter_noise_and_deep_link(
    monkeypatch, tmp_path: Path
):
    client = Client(_files(2))
    client.discovery = [
        SimpleNamespace(
            id="remote-skill",
            slug="team-runbook",
            latest_version=3,
            model_dump=lambda **_kwargs: {
                "id": "remote-skill",
                "slug": "team-runbook",
                "latest_version": 3,
            },
        )
    ]
    manager, _target = _manager(monkeypatch, tmp_path, client=client)
    manager.config = {
        "portal_url": "http://127.0.0.1:3111",
        "notifications": {"new_skills": "immediate"},
    }
    client.feed_pages = [
        Feed.model_validate({
            "events": [
                {
                    "event_id": "event-new",
                    "kind": "new",
                    "skill_id": "remote-skill",
                    "version": 3,
                    "takedown_generation": 0,
                    "installation_id": None,
                    "update_mode": None,
                    "occurred_at": "2026-08-24T00:00:00+00:00",
                },
                {
                    "event_id": "event-restored",
                    "kind": "restored",
                    "skill_id": "remote-skill",
                    "version": 3,
                    "takedown_generation": 0,
                    "installation_id": None,
                    "update_mode": None,
                    "occurred_at": "2026-08-24T00:00:01+00:00",
                },
            ],
            "next_cursor": "cursor-notifications",
            "has_more": False,
        })
    ]

    assert manager.poll_feed()["inserted"] == 2
    notifications = manager.notifications()["events"]
    assert notifications == [
        {
            "event_id": "event-new",
            "source_event_ids": ["event-new"],
            "category": "new_skill",
            "kind": "new",
            "skill_id": "remote-skill",
            "skill_name": "team-runbook",
            "version": 3,
            "state": None,
            "moderation_note": None,
            "portal_url": "http://127.0.0.1:3111/orgs/org-1/wisdom/skills/remote-skill?version=3",
            "occurred_at": "2026-08-24T00:00:00Z",
            "security_check": None,
            "professionalism_check": None,
        }
    ]
    assert [
        item["event_id"] for item in manager.store.feed_events(unseen_only=True)
    ] == ["event-new"]
    _telegram_home(monkeypatch, "123456")

    calls = []
    monkeypatch.setattr(
        "tools.send_message_tool.send_telegram_notification_pane",
        lambda **kwargs: calls.append(kwargs) or {"success": True},
    )
    assert manager.dispatch_telegram() == {"attempted": True, "delivered": 1}
    assert calls[0]["button_rows"] == [
        [
            {
                "label": "View ↗",
                "url": "http://127.0.0.1:3111/orgs/org-1/wisdom/skills/remote-skill?version=3",
            },
            {
                "label": "Install",
                "callback_data": "wi:plan:install:remote-skill",
            },
        ]
    ]
    assert calls[0]["items"] == [
        {
            "heading": "🆕 New skill from your team",
            "detail": (
                "team-runbook · v3\n"
                "Security check: Unavailable\n"
                "No known matches detected is not a security certification.\n\n"
                "Professionalism check (agent-assessed, advisory): Unavailable"
            ),
        }
    ]

    manager.notifications(mark_seen=True)
    assert manager.notifications()["events"] == []


def test_telegram_public_home_excludes_device_state_and_mutation_controls(
    monkeypatch, tmp_path: Path
):
    client = Client(_files(2))
    client.discovery = [
        SimpleNamespace(
            id="remote-skill",
            slug="team-runbook",
            latest_version=3,
            model_dump=lambda **_kwargs: {
                "id": "remote-skill",
                "slug": "team-runbook",
                "latest_version": 3,
            },
        )
    ]
    manager, _target = _manager(monkeypatch, tmp_path, client=client)
    manager.store.persist_local_notice(
        event_id="event-new-public",
        kind="new",
        skill_id="remote-skill",
        payload={"version": 3},
    )
    manager.store.persist_local_notice(
        event_id="event-update-private",
        kind="updated",
        skill_id="skill-1",
        payload={"version": 2, "update_mode": "MANUAL"},
    )
    _telegram_home(monkeypatch, "-100123456")
    calls = []
    monkeypatch.setattr(
        "tools.send_message_tool.send_telegram_notification_pane",
        lambda **kwargs: calls.append(kwargs) or {"success": True},
    )

    assert manager.dispatch_telegram() == {"attempted": True, "delivered": 1}
    assert calls[0]["button_rows"] == [
        [
            {
                "label": "View ↗",
                "url": "https://portal.nousresearch.com/orgs/org-1/wisdom/skills/remote-skill?version=3",
            }
        ]
    ]
    assert manager.dispatch_telegram() == {"attempted": False, "delivered": 0}


def test_install_notifications_are_limited_to_this_installation(
    monkeypatch, tmp_path: Path
):
    client = Client(_files(2))
    manager, _target = _manager(monkeypatch, tmp_path, client=client)
    local_identity = manager.store.installation_identity()
    client.feed_pages = [
        Feed.model_validate({
            "events": [
                {
                    "event_id": "event-local-install",
                    "kind": "installed",
                    "skill_id": "skill-1",
                    "version": 1,
                    "takedown_generation": 0,
                    "installation_id": local_identity,
                    "update_mode": "MANUAL",
                    "occurred_at": "2026-08-24T00:00:00+00:00",
                },
                {
                    "event_id": "event-other-install",
                    "kind": "installed",
                    "skill_id": "skill-1",
                    "version": 1,
                    "takedown_generation": 0,
                    "installation_id": "hwi_other-device",
                    "update_mode": "MANUAL",
                    "occurred_at": "2026-08-24T00:00:01+00:00",
                },
            ],
            "next_cursor": "cursor-installs",
            "has_more": False,
        })
    ]

    manager.poll_feed()
    events = manager.notifications()["events"]
    assert len(events) == 1
    assert events[0]["category"] == "installed"
    assert events[0]["source_event_ids"] == ["event-local-install"]


def test_off_cadence_suppresses_local_and_telegram_delivery(
    monkeypatch, tmp_path: Path
):
    client = Client(_files(2))
    manager, _target = _manager(monkeypatch, tmp_path, client=client)
    manager.config = {"notifications": {"installed_updates": "off"}}
    client.feed_pages = [
        Feed.model_validate({
            "events": [
                {
                    "event_id": "event-off",
                    "kind": "installation_updated",
                    "skill_id": "skill-1",
                    "version": 2,
                    "takedown_generation": 0,
                    "installation_id": manager.store.installation_identity(),
                    "update_mode": "MANUAL",
                    "occurred_at": "2026-08-24T00:00:00+00:00",
                }
            ],
            "next_cursor": "cursor-off",
            "has_more": False,
        })
    ]
    assert manager.poll_feed()["inserted"] == 1
    assert manager.notifications()["events"] == []
    assert manager.dispatch_telegram() == {"attempted": False, "delivered": 0}


def test_slack_public_home_only_emits_collective_publication_links(
    monkeypatch, tmp_path: Path
):
    from gateway.config import Platform, PlatformConfig

    client = Client(_files(2))
    client.discovery = [
        SimpleNamespace(
            id="remote-skill",
            slug="team-runbook",
            latest_version=3,
            model_dump=lambda **_kwargs: {
                "id": "remote-skill",
                "slug": "team-runbook",
                "latest_version": 3,
            },
        )
    ]
    manager, _target = _manager(monkeypatch, tmp_path, client=client)
    manager.store.persist_local_notice(
        event_id="event-new-slack",
        kind="new",
        skill_id="remote-skill",
        payload={"version": 3},
    )
    home = SimpleNamespace(chat_id="C_PUBLIC", thread_id=None)
    config = SimpleNamespace(
        platforms={Platform.SLACK: PlatformConfig(enabled=True, token="xoxb")},
        get_home_channel=lambda platform: home if platform == Platform.SLACK else None,
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    calls = []
    monkeypatch.setattr(
        "tools.send_message_tool.send_slack_wisdom_notification_pane",
        lambda **kwargs: calls.append(kwargs) or {"success": True},
    )

    assert manager.dispatch_slack() == {"attempted": True, "delivered": 1}
    assert calls[0]["button_rows"] == [
        [
            {
                "label": "View in Portal ↗",
                "url": "https://portal.nousresearch.com/orgs/org-1/wisdom/skills/remote-skill?version=3",
            }
        ]
    ]


def test_slack_dm_home_offers_verified_update_action(monkeypatch, tmp_path: Path):
    from gateway.config import Platform, PlatformConfig

    client = Client(_files(2))
    manager, _target = _manager(monkeypatch, tmp_path, client=client)
    manager.store.persist_local_notice(
        event_id="event-update-slack",
        kind="updated",
        skill_id="skill-1",
        payload={"version": 2, "update_mode": "MANUAL"},
    )
    home = SimpleNamespace(chat_id="D_PRIVATE", thread_id=None)
    config = SimpleNamespace(
        platforms={Platform.SLACK: PlatformConfig(enabled=True, token="xoxb")},
        get_home_channel=lambda platform: home if platform == Platform.SLACK else None,
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    calls = []
    monkeypatch.setattr(
        "tools.send_message_tool.send_slack_wisdom_notification_pane",
        lambda **kwargs: calls.append(kwargs) or {"success": True},
    )

    assert manager.dispatch_slack() == {"attempted": True, "delivered": 1}
    assert calls[0]["button_rows"][0][-1] == {
        "label": "Update",
        "callback_data": "wi:plan:update:skill-1",
    }
