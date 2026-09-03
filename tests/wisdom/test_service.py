import json
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.skill_utils import parse_frontmatter
from hermes_wisdom.client import (
    Draft,
    WisdomAuthError,
    WisdomConflict,
    WisdomValidationError,
)
from hermes_wisdom.contract import (
    PackageManifest,
    SystemSpecification,
    author_description_hash,
    canonical_json_bytes,
    sha256_address,
)
from hermes_wisdom.package import (
    PackagePolicyError,
    prepare_package,
    verify_content_files,
)
from hermes_wisdom.notice import qualification_notice
from hermes_wisdom.service import WisdomService
from hermes_wisdom.store import WisdomStore


class FakeClient:
    identity = {"owner": "user-1"}

    def __init__(self):
        self.uploaded = 0
        self.submissions = []
        self.drafts = {}
        self.declines = 0

    def upload_private_objects(self, objects):
        self.uploaded += len(objects)

    def submit_draft(self, **payload):
        self.submissions.append(payload)
        draft = Draft(
            id="draft-1",
            orgId="org-1",
            ownerUserId="user-1",
            slug=payload["slug"],
            draftCommit=payload["commit"],
            contentHash=payload["content_hash"],
            authorDescription=payload["description"],
            authorDescriptionHash=None,
            state="ready",
            packageManifestHash=None,
            packageManifestSchemaVersion=1,
            systemSpec=None,
            scan=None,
            scanVerdict="pass",
            explanation=None,
            updatedAt="revision-1",
        )
        self.drafts[draft.id] = draft
        return draft

    def draft(self, draft_id):
        return SimpleNamespace(
            draft=self.drafts[draft_id],
            effective_policy={"publication_policy": "moderated"},
        )

    def decline(self, draft_id):
        self.declines += 1
        self.drafts[draft_id] = self.drafts[draft_id].model_copy(
            update={"state": "declined"}
        )
        return {"state": "declined"}


class InstallClient:
    def __init__(self, *, fail_record: bool = False):
        manifest = PackageManifest(
            name="managed-skill",
            requirements=SystemSpecification.model_validate({
                "hermes": {"minimum_version": "0.1.0"}
            }),
        )
        self.files = [
            ("SKILL.md", "file", b"# Managed\n"),
            (
                "skill.manifest.json",
                "file",
                canonical_json_bytes(manifest.model_dump(mode="json")),
            ),
        ]
        self.fail_record = fail_record

    def skill(self, _skill_id):
        return SimpleNamespace(
            skill={
                "state": "active",
                "slug": "managed-skill",
                "takedown_generation": 0,
            },
            versions=[{"version": 1}],
        )

    def version(self, _skill_id, _version):
        manifest = json.loads(self.files[1][2])
        return SimpleNamespace(version={"system_spec": manifest["requirements"]})

    def content(
        self,
        _skill_id,
        _version,
        *,
        installation_id,
        takedown_generation,
    ):
        assert installation_id.startswith("hwi_")
        assert takedown_generation == 0
        _records, content_hash = verify_content_files(self.files)
        return SimpleNamespace(content_hash=content_hash), self.files

    def record_install(self, **_kwargs):
        if self.fail_record:
            self.fail_record = False
            raise RuntimeError("network down")
        return SimpleNamespace(effective_update_mode="MANUAL")


class SetupClient:
    display_scopes = ("wisdom:read", "wisdom:install")
    identity = {"claims": {"tool_gateway_admin": True}}

    def __init__(self, org_id: str = "org-1"):
        self.display_org_id = org_id

    def capability(self):
        return {"features": ["wisdom"]}

    def register_identity(self, installation_id):
        return {"installation_id": installation_id, "state": "active"}


class ReviewClient:
    identity = {"owner": "user-1"}

    def __init__(self):
        manifest = canonical_json_bytes(
            PackageManifest(
                name="reviewed-skill",
                requirements=SystemSpecification.model_validate({
                    "hermes": {"minimum_version": "0.1.0"}
                }),
            ).model_dump(mode="json")
        )
        self.files = [
            ("SKILL.md", "file", b"# Reviewed skill\n"),
            ("skill.manifest.json", "file", manifest),
        ]
        self.description = "Resolve incidents safely."
        self.revision = "revision-1"
        self.approvals: list[dict] = []
        self.publications = 0
        self.uploaded = 0
        self.revisions: list[dict] = []
        self.state = "ready"

    def _draft(self):
        _records, content_hash = verify_content_files(self.files)
        manifest = next(
            body for path, _, body in self.files if path == "skill.manifest.json"
        )
        return Draft(
            id="draft-review",
            orgId="org-1",
            ownerUserId="user-1",
            slug="reviewed-skill",
            draftCommit="sha256:" + "a" * 64,
            contentHash=content_hash,
            authorDescription=self.description,
            authorDescriptionHash=author_description_hash(self.description),
            state=self.state,
            packageManifestHash=sha256_address(manifest),
            packageManifestSchemaVersion=1,
            systemSpec=None,
            scan={"verdict": "pass", "findings": []},
            scanVerdict="pass",
            explanation="Mechanical facts.",
            updatedAt=self.revision,
        )

    def reconstruct_draft(self, _draft_id):
        draft = self._draft()
        records, content_hash = verify_content_files(self.files)
        return SimpleNamespace(
            detail=SimpleNamespace(
                draft=draft,
                effective_policy={"publication_policy": "open", "policy_version": 1},
            ),
            files=self.files,
            content_files=records,
            content_hash=content_hash,
        )

    def draft(self, _draft_id):
        return SimpleNamespace(
            draft=self._draft(),
            effective_policy={"publication_policy": "open", "policy_version": 1},
        )

    def upload_private_objects(self, objects):
        self.uploaded += len(objects)

    def revise_draft(self, _draft_id, **payload):
        self.revisions.append(payload)
        return self._draft().model_copy(
            update={
                "id": "draft-revised",
                "draftCommit": payload["commit"],
                "contentHash": payload["content_hash"],
                "authorDescription": payload["description"],
                "authorDescriptionHash": author_description_hash(
                    payload["description"]
                ),
                "packageManifestHash": None,
                "updatedAt": "revision-2",
            }
        )

    def approve(self, _draft_id, **hashes):
        self.approvals.append(hashes)
        self.state = "owner_approved"
        return self._draft().model_copy(update={"state": "owner_approved"})

    def publish(self, _draft_id, *, content_hash):
        self.publications += 1
        self.state = "published"
        return {"state": "published", "content_hash": content_hash}

    def decline(self, _draft_id):
        self.state = "declined"
        return {"state": "declined"}


def _review_service(tmp_path: Path, *, client: ReviewClient):
    store = WisdomStore(tmp_path / "state")
    skill_path = tmp_path / "skills" / "reviewed-skill"
    skill_path.mkdir(parents=True)
    skill_id = store.register_skill(
        skill_path,
        content_hash=client._draft().contentHash,
        source_kind="local",
    )
    draft = client._draft()
    store.record_draft({
        "id": draft.id,
        "skill_id": skill_id,
        "source_hash": draft.contentHash,
        "overlay_path": str(skill_path),
        "draft_commit": draft.draftCommit,
        "server_revision": draft.updatedAt,
        "state": draft.state,
        "description": draft.authorDescription or "",
        "content_hash": draft.contentHash,
        "description_hash": draft.authorDescriptionHash or "",
        "manifest_hash": draft.packageManifestHash or "",
    })
    return WisdomService(store=store, client=client)


def test_owner_draft_approval_reconciles_a_portal_won_race(monkeypatch, tmp_path: Path):
    client = ReviewClient()
    service = _review_service(tmp_path, client=client)
    monkeypatch.setattr(service, "require_setup", lambda: None)
    client.state = "pending_moderation"

    result = service.approve_owner_draft("draft-review")

    assert result["publication_state"] == "pending_moderation"
    assert result["already_advanced"] is True
    assert client.approvals == []
    assert client.publications == 0


def test_owner_draft_approval_uses_hash_bound_review_and_publication(
    monkeypatch, tmp_path: Path
):
    client = ReviewClient()
    service = _review_service(tmp_path, client=client)
    monkeypatch.setattr(service, "require_setup", lambda: None)

    result = service.approve_owner_draft("draft-review")

    assert result["publication_state"] == "published"
    assert len(client.approvals) == 1
    assert client.publications == 1
    assert service.store.receipt("draft-review") is None


def test_owner_publication_resume_revalidates_review_receipt(
    monkeypatch, tmp_path: Path
):
    client = ReviewClient()
    service = _review_service(tmp_path, client=client)
    monkeypatch.setattr(service, "require_setup", lambda: None)
    service.review("draft-review", acknowledge=True)
    client.state = "owner_approved"

    result = service.approve_owner_draft("draft-review")

    assert result["publication_state"] == "published"
    assert client.publications == 1
    assert service.store.receipt("draft-review") is None


def test_owner_publication_resume_rejects_stale_review_receipt(
    monkeypatch, tmp_path: Path
):
    client = ReviewClient()
    service = _review_service(tmp_path, client=client)
    monkeypatch.setattr(service, "require_setup", lambda: None)
    service.review("draft-review", acknowledge=True)
    client.state = "owner_approved"
    client.description = "Changed after owner review."

    with pytest.raises(PackagePolicyError, match="receipt is stale"):
        service.approve_owner_draft("draft-review")

    assert client.publications == 0
    assert service.store.receipt("draft-review") is not None


def test_owner_draft_decline_reports_already_published_without_mutating(
    monkeypatch, tmp_path: Path
):
    client = ReviewClient()
    service = _review_service(tmp_path, client=client)
    monkeypatch.setattr(service, "require_setup", lambda: None)
    client.state = "published"

    result = service.decline_owner_draft("draft-review")

    assert result["state"] == "published"
    assert result["already_advanced"] is True
    assert client.state == "published"


def _install_service(monkeypatch, tmp_path: Path, *, client: InstallClient):
    skills = tmp_path / "skills"
    monkeypatch.setattr("hermes_wisdom.service.get_skills_dir", lambda: skills)
    monkeypatch.setattr(
        "hermes_wisdom.service._scan_summary",
        lambda _path: {
            "guard": {"allowed": True, "findings": [], "reason": None},
            "skill_evaluator": {"status": "disabled", "findings": []},
        },
    )
    store = WisdomStore(tmp_path / "state")
    store.installation_identity()
    store.verify_installation_identity("org-1")
    return WisdomService(store=store, client=client)


def test_prepare_requires_local_owner_edit_before_any_network(
    monkeypatch, tmp_path: Path
):
    skill = tmp_path / "skills" / "my-skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\n"
        "name: my-skill\n"
        "description: Test.\n"
        "metadata:\n"
        "  hermes:\n"
        "    editorial_name: My Skill\n"
        "    editorial_description: A useful skill for people.\n"
        "---\n# Test\n",
        encoding="utf-8",
    )
    fake = FakeClient()
    service = WisdomService(store=WisdomStore(tmp_path / "state"), client=fake)
    monkeypatch.setattr(service, "_eligible_paths", lambda: [skill])
    monkeypatch.setattr(
        "hermes_wisdom.service.draft_description",
        lambda _body: pytest.fail("existing author copy should avoid a model call"),
    )
    monkeypatch.setattr(
        "hermes_wisdom.professionalism.run_review",
        lambda job: {
            "schema_version": 1,
            "content_hash": job["content_hash"],
            "author_description_hash": job["author_description_hash"],
            "status": "pass",
            "summary": "No professionalism concerns detected.",
            "checks": [
                {
                    "key": key,
                    "status": "pass",
                    "finding_count": 0,
                    "details": [],
                }
                for key in (
                    "profanity_or_abuse",
                    "hate_or_harassment",
                    "sexual_or_graphic_language",
                    "manipulative_or_spam",
                )
            ],
            "provenance": {
                "kind": "agent_assessed",
                "provider": "codex",
                "model": "gpt-5.6-sol",
            },
            "assessed_at": "2026-09-02T00:00:00+00:00",
        },
    )

    prepared = service.suggest("my-skill")
    assert prepared["network_submission"] is False
    assert prepared["drafted_description"] == "A useful skill for people."
    assert fake.uploaded == 0
    assert {item["path"] for item in prepared["files"]} == {
        "SKILL.md",
        "skill.manifest.json",
    }
    manifest = json.loads(
        next(
            item["content_utf8"]
            for item in prepared["files"]
            if item["path"] == "skill.manifest.json"
        )
    )
    manifest["requirements"]["known_limitations"] = ["Owner reviewed limitation"]
    edited_files = [
        {
            "path": item["path"],
            "content_utf8": (
                item["content_utf8"].replace("# Test", "# Owner edited test")
                if item["path"] == "SKILL.md"
                else json.dumps(manifest, sort_keys=True, separators=(",", ":"))
            ),
        }
        for item in prepared["files"]
    ]
    saved = service.save_prepared(
        prepared["local_draft_id"],
        author_description="Owner-approved copy.",
        files=edited_files,
    )
    assert fake.uploaded == 0
    assert saved["drafted_description"] == "Owner-approved copy."
    assert "# Owner edited test" in next(
        item["content_utf8"] for item in saved["files"] if item["path"] == "SKILL.md"
    )
    assert "# Owner edited test" not in (skill / "SKILL.md").read_text()

    submitted = service.suggest(
        "my-skill",
        description="Owner-approved copy.",
        system_specification=manifest["requirements"],
    )
    assert submitted["draft"]["id"] == "draft-1"
    assert fake.uploaded > 0
    assert fake.submissions[0].keys() == {
        "slug",
        "commit",
        "content_hash",
        "description",
        "professionalism_review",
    }
    assert fake.submissions[0]["professionalism_review"]["status"] == "pass"
    serialized = json.dumps(fake.submissions[0])
    for forbidden in ("usage", "refinement", "candidate", "ranking", "stability"):
        assert forbidden not in serialized


def test_local_candidate_decline_suppresses_exact_content_without_network(
    monkeypatch, tmp_path: Path
):
    skill = tmp_path / "skills" / "declined-skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# Declined\n", encoding="utf-8")
    fake = FakeClient()
    service = WisdomService(store=WisdomStore(tmp_path / "state"), client=fake)
    monkeypatch.setattr(service, "_eligible_paths", lambda: [skill])
    monkeypatch.setattr(
        "hermes_wisdom.service.draft_description", lambda _body: "Local draft."
    )

    candidate = service.scan_candidates()[0]
    prepared = service.suggest(
        "declined-skill", local_skill_id=candidate["local_skill_id"]
    )
    result = service.dismiss_local_candidate(
        candidate["local_skill_id"], candidate["content_hash"]
    )

    assert result == {"dismissed": True}
    local = service.store.draft(prepared["local_draft_id"])
    assert local is not None
    assert local["state"] == "declined"
    assert fake.uploaded == 0
    assert service.scan_candidates() == []


def test_candidate_scan_hides_an_exact_contributed_version_until_content_changes(
    monkeypatch, tmp_path: Path
):
    skills = tmp_path / "skills"
    skill = skills / "incident-handoff"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\n"
        "name: incident-handoff\n"
        "description: Use when handing off an incident.\n"
        "metadata:\n"
        "  hermes:\n"
        "    editorial_name: Incident Handoff\n"
        "    editorial_description: Transfer incident context between responders.\n"
        "---\n# Incident handoff\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("hermes_wisdom.service.get_skills_dir", lambda: skills)
    service = WisdomService(store=WisdomStore(tmp_path / "state"), client=FakeClient())

    candidate = service.scan_candidates()[0]
    assert candidate["editorial_name"] == "Incident Handoff"
    assert (
        candidate["editorial_description"]
        == "Transfer incident context between responders."
    )
    service.store.record_draft({
        "id": "draft-1",
        "skill_id": candidate["local_skill_id"],
        "source_hash": candidate["content_hash"],
        "overlay_path": str(skill),
        "state": "published",
        "description": "Share incident context.",
        "content_hash": "sha256:" + "1" * 64,
        "description_hash": "sha256:" + "2" * 64,
        "manifest_hash": "sha256:" + "3" * 64,
    })

    assert service.scan_candidates() == []

    (skill / "SKILL.md").write_text(
        (skill / "SKILL.md").read_text(encoding="utf-8") + "\nNew material guidance.\n",
        encoding="utf-8",
    )
    changed = service.scan_candidates()
    assert [item["name"] for item in changed] == ["incident-handoff"]
    assert changed[0]["contribution_state"] == "new"


def test_suggest_uses_candidate_identity_and_rejects_a_stale_duplicate_action(
    monkeypatch, tmp_path: Path
):
    skills = tmp_path / "skills"
    local = skills / "deployment-checklist"
    managed = skills / "_wisdom" / "org-1" / "deployment-checklist"
    local.mkdir(parents=True)
    managed.mkdir(parents=True)
    (local / "SKILL.md").write_text("# Local deployment checklist\n", encoding="utf-8")
    (managed / "SKILL.md").write_text("# Managed copy\n", encoding="utf-8")
    monkeypatch.setattr("hermes_wisdom.service.get_skills_dir", lambda: skills)
    monkeypatch.setattr(
        "hermes_wisdom.service.draft_description", lambda _body: "Deploy safely."
    )
    service = WisdomService(store=WisdomStore(tmp_path / "state"), client=FakeClient())
    candidate = service.scan_candidates()[0]

    prepared = service.suggest(
        "deployment-checklist", local_skill_id=candidate["local_skill_id"]
    )
    assert Path(prepared["overlay_path"], "SKILL.md").read_text() == (
        "# Local deployment checklist\n"
    )
    service.store.set_draft_state(prepared["local_draft_id"], "submitted")

    with pytest.raises(WisdomConflict, match="already in the contribution flow"):
        service.suggest(
            "deployment-checklist", local_skill_id=candidate["local_skill_id"]
        )


def _qualified_candidate_event(
    service: WisdomService,
    skill: Path,
    *,
    editorial_name: str | None = None,
    editorial_description: str | None = None,
) -> str:
    candidate = service.scan_candidates()[0]
    payload = {
        "skill_name": skill.name,
        "local_reasons": {"high_usage": True},
    }
    if editorial_name is not None:
        payload["editorial_name"] = editorial_name
    if editorial_description is not None:
        payload["editorial_description"] = editorial_description
    event_id = service.store.emit_local_event(
        kind="wisdom.candidate",
        skill_id=candidate["local_skill_id"],
        content_hash=candidate["content_hash"],
        payload=payload,
        session_id="telegram-session",
        task_id="task-1",
        qualification="high_usage",
    )
    assert event_id is not None
    return event_id


def test_candidate_notice_projection_is_stable_across_surfaces_and_uses_verified_org_name(
    monkeypatch, tmp_path: Path
):
    store = WisdomStore(tmp_path / "state")
    store.installation_identity()
    store.verify_installation_identity("org-1")
    calls = 0

    def account_info(**_kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(org_id="org-1", org_name="Nous Research")

    monkeypatch.setattr(
        "hermes_cli.nous_account.get_nous_portal_account_info", account_info
    )
    service = WisdomService(store=store, client=FakeClient())
    for index, name in enumerate(("first-skill", "second-skill"), start=1):
        path = tmp_path / name
        path.mkdir()
        (path / "SKILL.md").write_text(name, encoding="utf-8")
        skill_id = store.register_skill(
            path, content_hash=f"sha256:{name}", source_kind="local"
        )
        assert store.emit_local_event(
            kind="wisdom.candidate",
            skill_id=skill_id,
            content_hash=f"sha256:{name}",
            payload={"skill_name": name, "local_reasons": {}},
            session_id="session-1",
            task_id=f"task-{index}",
            qualification="high_usage",
        )

    events = list(reversed(service.local_candidate_events(session_id="session-1")))
    pending = service.pending_candidate_events(
        session_id="session-1", surface="telegram"
    )

    assert calls == 1
    assert [event["notice_variant"] for event in events] == ["first", "returning"]
    assert [event["qualification_sequence"] for event in events] == [1, 2]
    assert {event["organization_name"] for event in events} == {"Nous Research"}
    assert {event["id"]: event["notice_variant"] for event in pending} == {
        event["id"]: event["notice_variant"] for event in events
    }
    assert qualification_notice(events[0]).startswith(
        "Your organisation (Nous Research) has enabled Collective Wisdom"
    )
    assert qualification_notice(events[1]) == (
        "Hermes detected another skill that could be useful to your team."
    )


def test_defer_candidate_prompt_hides_only_the_selected_surface(tmp_path: Path):
    store = WisdomStore(tmp_path / "state")
    store.installation_identity()
    store.verify_installation_identity("org-1")
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("# Useful\n", encoding="utf-8")
    skill_id = store.register_skill(
        skill, content_hash="sha256:useful", source_kind="local"
    )
    event_id = store.emit_local_event(
        kind="wisdom.candidate",
        skill_id=skill_id,
        content_hash="sha256:useful",
        payload={"skill_name": "skill", "local_reasons": {}},
        session_id="session-1",
        task_id="task-1",
        qualification="high_usage",
    )
    assert event_id is not None
    service = WisdomService(store=store, client=FakeClient())

    result = service.defer_candidate_prompt(event_id, surface="desktop")

    assert result == {
        "event_id": event_id,
        "skill_name": "skill",
        "qualification": "high_usage",
        "state": "deferred",
    }
    assert (
        service.pending_candidate_events(session_id="session-1", surface="desktop")
        == []
    )
    assert [
        event["id"]
        for event in service.pending_candidate_events(
            session_id="session-1", surface="telegram"
        )
    ] == [event_id]
    assert store.local_event(event_id)["state"] == "unread"


def test_organization_name_mismatch_and_failure_are_negative_cached(
    monkeypatch, tmp_path: Path
):
    store = WisdomStore(tmp_path / "state")
    store.installation_identity()
    store.verify_installation_identity("org-1")
    calls = 0

    def mismatched(**_kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(org_id="org-2", org_name="Wrong Organization")

    monkeypatch.setattr(
        "hermes_cli.nous_account.get_nous_portal_account_info", mismatched
    )
    service = WisdomService(store=store, client=FakeClient())

    assert service.organization_display_name() is None
    assert service.organization_display_name() is None
    assert calls == 1
    cached = store.organization_display_name("org-1")
    assert cached is not None and cached["display_name"] is None
    assert cached["resolved"] == 0

    def unavailable(**_kwargs):
        raise RuntimeError("portal unavailable")

    monkeypatch.setattr(
        "hermes_cli.nous_account.get_nous_portal_account_info", unavailable
    )
    assert service.organization_display_name(force=True) is None
    assert qualification_notice({"notice_variant": "first"}).startswith(
        "Your organisation has enabled Collective Wisdom"
    )


def test_telegram_candidate_creates_an_owner_private_draft_and_portal_link(
    monkeypatch, tmp_path: Path
):
    skills = tmp_path / "skills"
    skill = skills / "telegram-skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: telegram-skill\ndescription: Share a safe runbook.\n---\n"
        "# Telegram skill\n",
        encoding="utf-8",
    )
    fake = FakeClient()
    store = WisdomStore(tmp_path / "state")
    store.installation_identity()
    store.verify_installation_identity("nas_organisation:wisdom-local")
    service = WisdomService(store=store, client=fake)
    monkeypatch.setattr(service, "_eligible_paths", lambda: [skill])
    monkeypatch.setattr(
        "hermes_wisdom.service._config",
        lambda: {
            "enabled": True,
            "disclosure_acknowledged_at": "2026-08-31T00:00:00+00:00",
        },
    )
    monkeypatch.setattr(
        "hermes_wisdom.service.portal_base_url", lambda: "https://portal.test"
    )
    monkeypatch.setattr(
        "hermes_wisdom.service._scan_summary",
        lambda _path: {
            "guard": {"allowed": True, "findings": [], "reason": None},
            "skill_evaluator": {"status": "disabled", "findings": []},
        },
    )
    source_before = (skill / "SKILL.md").read_text(encoding="utf-8")
    event_id = _qualified_candidate_event(
        service,
        skill,
        editorial_name="Telegram Runbook",
        editorial_description="A clear team runbook for Telegram workflows.",
    )

    result = service.draft_candidate(event_id)

    assert result == {
        "draft_id": "draft-1",
        "skill_name": "telegram-skill",
        "qualification": "high_usage",
        "state": "ready",
        "portal_url": ("https://portal.test/orgs/wisdom-local/wisdom/review/draft-1"),
        "created": True,
    }
    assert fake.uploaded > 0
    assert fake.submissions[0]["description"] == "Share a safe runbook."
    assert "local_reasons" not in json.dumps(fake.submissions)
    assert (skill / "SKILL.md").read_text(encoding="utf-8") == source_before
    local_draft = store.draft("draft-1")
    assert local_draft is not None
    overlay_frontmatter, _body = parse_frontmatter(
        (Path(str(local_draft["overlay_path"])) / "SKILL.md").read_text(
            encoding="utf-8"
        )
    )
    assert overlay_frontmatter["metadata"]["hermes"] == {
        "editorial_name": "Telegram Runbook",
        "editorial_description": "A clear team runbook for Telegram workflows.",
    }

    resumed = service.draft_candidate(event_id)
    assert resumed["created"] is False
    assert len(fake.submissions) == 1


def test_telegram_candidate_publish_uses_normal_review_and_approval(
    monkeypatch, tmp_path: Path
):
    service = WisdomService(store=WisdomStore(tmp_path / "state"), client=FakeClient())
    monkeypatch.setattr(
        service,
        "draft_candidate",
        lambda _event_id: {
            "draft_id": "draft-1",
            "skill_name": "telegram-skill",
            "state": "ready",
            "portal_url": "https://portal.test/review/draft-1",
            "created": True,
        },
    )
    reviewed: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        service,
        "review",
        lambda draft_id, *, acknowledge, portal=False: reviewed.append((
            draft_id,
            acknowledge,
        )),
    )
    monkeypatch.setattr(
        service,
        "approve",
        lambda _draft_id: {"publication": {"state": "pending_moderation"}},
    )

    result = service.approve_candidate("event-1")

    assert reviewed == [("draft-1", True)]
    assert result["publication_state"] == "pending_moderation"


def test_telegram_candidate_reconciles_portal_submission_without_duplicate_publish(
    monkeypatch, tmp_path: Path
):
    skill = tmp_path / "skills" / "portal-approved-skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: portal-approved-skill\ndescription: Portal test.\n---\n"
        "# Portal approved\n",
        encoding="utf-8",
    )
    fake = FakeClient()
    store = WisdomStore(tmp_path / "state")
    store.installation_identity()
    store.verify_installation_identity("nas_organisation:wisdom-local")
    service = WisdomService(store=store, client=fake)
    monkeypatch.setattr(service, "_eligible_paths", lambda: [skill])
    monkeypatch.setattr(
        "hermes_wisdom.service._config",
        lambda: {
            "enabled": True,
            "disclosure_acknowledged_at": "2026-08-31T00:00:00+00:00",
        },
    )
    monkeypatch.setattr(
        "hermes_wisdom.service._scan_summary",
        lambda _path: {
            "guard": {"allowed": True, "findings": [], "reason": None},
            "skill_evaluator": {"status": "disabled", "findings": []},
        },
    )
    event_id = _qualified_candidate_event(service, skill)
    drafted = service.draft_candidate(event_id)
    fake.drafts[drafted["draft_id"]] = fake.drafts[drafted["draft_id"]].model_copy(
        update={"state": "pending_moderation"}
    )

    result = service.approve_candidate(event_id)

    assert result["publication_state"] == "pending_moderation"
    assert result["already_advanced"] is True
    assert len(fake.submissions) == 1
    assert store.local_event(event_id)["state"] == "handled"
    assert store.draft(drafted["draft_id"])["state"] == "pending_moderation"


def test_telegram_candidate_reports_portal_publication_even_if_local_source_changed(
    monkeypatch, tmp_path: Path
):
    skill = tmp_path / "skills" / "published-elsewhere"
    skill.mkdir(parents=True)
    source = skill / "SKILL.md"
    source.write_text(
        "---\nname: published-elsewhere\ndescription: Publication test.\n---\n"
        "# Original\n",
        encoding="utf-8",
    )
    fake = FakeClient()
    store = WisdomStore(tmp_path / "state")
    store.installation_identity()
    store.verify_installation_identity("nas_organisation:wisdom-local")
    service = WisdomService(store=store, client=fake)
    monkeypatch.setattr(service, "_eligible_paths", lambda: [skill])
    monkeypatch.setattr(
        "hermes_wisdom.service._config",
        lambda: {
            "enabled": True,
            "disclosure_acknowledged_at": "2026-08-31T00:00:00+00:00",
        },
    )
    monkeypatch.setattr(
        "hermes_wisdom.service._scan_summary",
        lambda _path: {
            "guard": {"allowed": True, "findings": [], "reason": None},
            "skill_evaluator": {"status": "disabled", "findings": []},
        },
    )
    event_id = _qualified_candidate_event(service, skill)
    drafted = service.draft_candidate(event_id)
    fake.drafts[drafted["draft_id"]] = fake.drafts[drafted["draft_id"]].model_copy(
        update={"state": "published"}
    )
    source.write_text("# A newer local version\n", encoding="utf-8")

    result = service.approve_candidate(event_id)

    assert result["publication_state"] == "published"
    assert result["already_advanced"] is True
    assert len(fake.submissions) == 1


def test_telegram_decline_withdraws_portal_submission(monkeypatch, tmp_path: Path):
    skill = tmp_path / "skills" / "withdraw-from-telegram"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: withdraw-from-telegram\ndescription: Withdrawal test.\n---\n"
        "# Withdraw\n",
        encoding="utf-8",
    )
    fake = FakeClient()
    store = WisdomStore(tmp_path / "state")
    store.installation_identity()
    store.verify_installation_identity("nas_organisation:wisdom-local")
    service = WisdomService(store=store, client=fake)
    monkeypatch.setattr(service, "_eligible_paths", lambda: [skill])
    monkeypatch.setattr(
        "hermes_wisdom.service._config",
        lambda: {
            "enabled": True,
            "disclosure_acknowledged_at": "2026-08-31T00:00:00+00:00",
        },
    )
    monkeypatch.setattr(
        "hermes_wisdom.service._scan_summary",
        lambda _path: {
            "guard": {"allowed": True, "findings": [], "reason": None},
            "skill_evaluator": {"status": "disabled", "findings": []},
        },
    )
    event_id = _qualified_candidate_event(service, skill)
    drafted = service.draft_candidate(event_id)
    fake.drafts[drafted["draft_id"]] = fake.drafts[drafted["draft_id"]].model_copy(
        update={"state": "pending_moderation"}
    )

    result = service.decline_candidate(event_id)

    assert result["state"] == "declined"
    assert result["withdrawn"] is True
    assert fake.declines == 1
    assert store.local_event(event_id)["state"] == "dismissed"


def test_telegram_candidate_rejects_changed_source_before_any_upload(
    monkeypatch, tmp_path: Path
):
    skill = tmp_path / "skills" / "changing-skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# First\n", encoding="utf-8")
    fake = FakeClient()
    service = WisdomService(store=WisdomStore(tmp_path / "state"), client=fake)
    monkeypatch.setattr(service, "_eligible_paths", lambda: [skill])
    monkeypatch.setattr(service, "require_setup", lambda: None)
    event_id = _qualified_candidate_event(service, skill)
    (skill / "SKILL.md").write_text("# Changed\n", encoding="utf-8")

    with pytest.raises(WisdomConflict, match="changed after qualification"):
        service.draft_candidate(event_id)

    assert fake.uploaded == 0


def test_telegram_candidate_decline_suppresses_only_the_qualified_bytes(
    monkeypatch, tmp_path: Path
):
    skill = tmp_path / "skills" / "decline-from-telegram"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# Decline\n", encoding="utf-8")
    service = WisdomService(store=WisdomStore(tmp_path / "state"), client=FakeClient())
    monkeypatch.setattr(service, "_eligible_paths", lambda: [skill])
    event_id = _qualified_candidate_event(service, skill)

    result = service.decline_candidate(event_id)

    assert result["state"] == "declined"
    assert result["skill_name"] == "decline-from-telegram"
    assert service.store.local_event(event_id)["state"] == "dismissed"


def test_setup_persists_explicit_disclosure_and_enables_the_profile(
    monkeypatch, tmp_path: Path
):
    skills = tmp_path / "skills"
    monkeypatch.setattr("hermes_wisdom.service.get_skills_dir", lambda: skills)
    config = {"wisdom": {"enabled": False}}

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: copy.deepcopy(config))

    def save_config(value):
        config.clear()
        config.update(copy.deepcopy(value))

    monkeypatch.setattr("hermes_cli.config.save_config", save_config)
    service = WisdomService(store=WisdomStore(tmp_path / "state"), client=SetupClient())

    with pytest.raises(PackagePolicyError, match="explicit acceptance"):
        service.setup()
    first = service.setup(disclosure_accepted=True)
    second = service.setup(disclosure_accepted=True)

    assert config["wisdom"]["enabled"] is True
    assert (
        config["wisdom"]["disclosure_acknowledged_at"]
        == first["disclosure_acknowledged_at"]
    )
    assert second["disclosure_acknowledged_at"] == first["disclosure_acknowledged_at"]
    assert second["installation_id"] == first["installation_id"]

    service._client = SetupClient("org-2")
    third = service.setup(disclosure_accepted=True)
    assert third["installation_id"] != first["installation_id"]
    assert service.store.active_org_id() == "org-2"


def test_setup_accepts_opaque_nas_org_id_with_portable_managed_path(
    monkeypatch, tmp_path: Path
):
    from hermes_wisdom.contract import org_directory_name

    org_id = "nas_organisation:wisdom-local"
    skills = tmp_path / "skills"
    monkeypatch.setattr("hermes_wisdom.service.get_skills_dir", lambda: skills)
    config = {"wisdom": {"enabled": False}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: copy.deepcopy(config))
    monkeypatch.setattr("hermes_cli.config.save_config", lambda value: None)
    service = WisdomService(
        store=WisdomStore(tmp_path / "state"), client=SetupClient(org_id)
    )

    result = service.setup(disclosure_accepted=True)

    segment = org_directory_name(org_id)
    assert result["organization_id"] == org_id
    assert result["managed_directory"] == str(skills / "_wisdom" / segment)
    assert (skills / "_wisdom" / ".active_org").read_text() == segment + "\n"
    assert service.store.active_org_id() == org_id


def test_status_does_not_enroll_an_unconfigured_profile(monkeypatch, tmp_path: Path):
    store = WisdomStore(tmp_path / "state")
    service = WisdomService(store=store, client=SetupClient())
    monkeypatch.setattr("hermes_wisdom.service._config", lambda: {})

    status = service.status()

    assert status["configured"] is False
    assert status["setup_required_reason"] == "not_configured"
    assert status["installation_id"] is None
    assert store.existing_installation_identity() is None
    with pytest.raises(PackagePolicyError, match="wisdom setup"):
        service.require_setup()


def test_setup_guard_rejects_a_changed_authenticated_org(monkeypatch, tmp_path: Path):
    store = WisdomStore(tmp_path / "state")
    store.installation_identity()
    store.verify_installation_identity("org-1")
    service = WisdomService(store=store, client=SetupClient("org-2"))
    monkeypatch.setattr(
        "hermes_wisdom.service._config",
        lambda: {
            "enabled": True,
            "disclosure_acknowledged_at": "2026-08-25T00:00:00+00:00",
        },
    )

    with pytest.raises(PackagePolicyError, match="organization changed"):
        service.require_setup()

    status = service.status()
    assert status["configured"] is False
    assert status["setup_required_reason"] == "organization_changed"
    assert status["verified_org_id"] == "org-1"
    assert status["authenticated_org_id"] == "org-2"


def test_org_change_does_not_rotate_identity_before_gateway_accepts(
    monkeypatch, tmp_path: Path
):
    class RejectingSetupClient(SetupClient):
        def register_identity(self, installation_id):
            raise RuntimeError("gateway rejected identity")

    skills = tmp_path / "skills"
    monkeypatch.setattr("hermes_wisdom.service.get_skills_dir", lambda: skills)
    store = WisdomStore(tmp_path / "state")
    old_identity = store.installation_identity()
    store.verify_installation_identity("org-1")
    service = WisdomService(store=store, client=RejectingSetupClient("org-2"))

    with pytest.raises(RuntimeError, match="gateway rejected"):
        service.setup(disclosure_accepted=True)

    assert store.existing_installation_identity() == old_identity
    assert store.active_org_id() == "org-1"


def test_org_change_switches_marker_before_local_ledger(monkeypatch, tmp_path: Path):
    skills = tmp_path / "skills"
    monkeypatch.setattr("hermes_wisdom.service.get_skills_dir", lambda: skills)
    store = WisdomStore(tmp_path / "state")
    store.installation_identity()
    store.verify_installation_identity("org-1")
    service = WisdomService(store=store, client=SetupClient("org-2"))
    monkeypatch.setattr(
        store,
        "activate_installation_identity",
        lambda *_args: (_ for _ in ()).throw(OSError("injected ledger failure")),
    )

    with pytest.raises(OSError, match="injected ledger"):
        service.setup(disclosure_accepted=True)

    assert (skills / "_wisdom" / ".active_org").read_text() == "org-2\n"
    assert store.active_org_id() == "org-1"


def test_approval_requires_a_complete_review_receipt(tmp_path: Path):
    service = _review_service(tmp_path, client=ReviewClient())

    with pytest.raises(PackagePolicyError, match="fresh complete-package review"):
        service.approve("draft-review")


def test_edit_creates_a_rescanned_successor_and_invalidates_old_receipt(
    monkeypatch, tmp_path: Path
):
    client = ReviewClient()
    service = _review_service(tmp_path, client=client)
    reviewed = service.review("draft-review", acknowledge=True)
    edited_files = [
        {
            "path": path,
            "content_utf8": (
                "# Updated reviewed skill\n" if path == "SKILL.md" else body.decode()
            ),
        }
        for path, _mode, body in client.files
    ]
    monkeypatch.setattr(
        "hermes_wisdom.service._scan_summary",
        lambda _path: {
            "guard": {"allowed": True, "findings": [], "reason": None},
            "skill_evaluator": {"status": "disabled", "findings": []},
        },
    )

    result = service.revise(
        "draft-review",
        author_description="Updated owner copy.",
        files=edited_files,
        expected_content_hash=reviewed["hashes"]["content"],
        expected_description_hash=reviewed["hashes"]["author_description"],
        expected_manifest_hash=reviewed["hashes"]["package_manifest"],
    )

    assert result["draft"]["id"] == "draft-revised"
    assert client.uploaded > 0
    assert client.revisions[0]["description"] == "Updated owner copy."
    assert client.revisions[0]["expected_content_hash"] == reviewed["hashes"]["content"]
    assert service.store.draft("draft-review")["state"] == "invalidated"
    assert service.store.draft("draft-revised")["state"] == "ready"
    assert service.store.receipt("draft-review") is None


def test_edit_rejects_stale_hashes_and_incomplete_package(tmp_path: Path):
    client = ReviewClient()
    service = _review_service(tmp_path, client=client)
    reviewed = service.review("draft-review", acknowledge=False)
    complete_files = [
        {"path": path, "content_utf8": body.decode()}
        for path, _mode, body in client.files
    ]

    with pytest.raises(WisdomConflict, match="changed after it was opened"):
        service.revise(
            "draft-review",
            author_description=client.description,
            files=complete_files,
            expected_content_hash="sha256:" + "0" * 64,
            expected_description_hash=reviewed["hashes"]["author_description"],
            expected_manifest_hash=reviewed["hashes"]["package_manifest"],
        )
    with pytest.raises(PackagePolicyError, match="complete existing package"):
        service.revise(
            "draft-review",
            author_description=client.description,
            files=complete_files[:1],
            expected_content_hash=reviewed["hashes"]["content"],
            expected_description_hash=reviewed["hashes"]["author_description"],
            expected_manifest_hash=reviewed["hashes"]["package_manifest"],
        )
    assert client.uploaded == 0


def test_edit_never_uses_the_browser_draft_id_as_a_filesystem_segment(
    monkeypatch, tmp_path: Path
):
    client = ReviewClient()
    store = WisdomStore(tmp_path / "state")
    service = WisdomService(store=store, client=client)
    reviewed = service.review("../outside", acknowledge=False)
    edited_files = [
        {"path": path, "content_utf8": body.decode()}
        for path, _mode, body in client.files
    ]
    overlay_roots: list[Path] = []
    real_prepare = prepare_package

    def capture_overlay(*args, **kwargs):
        overlay_roots.append(Path(kwargs["overlay_root"]))
        return real_prepare(*args, **kwargs)

    monkeypatch.setattr("hermes_wisdom.service.prepare_package", capture_overlay)
    monkeypatch.setattr(
        "hermes_wisdom.service._scan_summary",
        lambda _path: {
            "guard": {"allowed": True, "findings": [], "reason": None},
            "skill_evaluator": {"status": "disabled", "findings": []},
        },
    )

    service.revise(
        "../outside",
        author_description=client.description,
        files=edited_files,
        expected_content_hash=reviewed["hashes"]["content"],
        expected_description_hash=reviewed["hashes"]["author_description"],
        expected_manifest_hash=reviewed["hashes"]["package_manifest"],
    )

    assert overlay_roots[0].parent == store.root / "revisions"
    assert len(overlay_roots[0].name) == 64


@pytest.mark.parametrize("mutation", ["content", "description", "manifest", "revision"])
def test_approval_rejects_each_stale_receipt_binding(tmp_path: Path, mutation: str):
    client = ReviewClient()
    service = _review_service(tmp_path, client=client)
    service.review("draft-review", acknowledge=True)

    if mutation == "content":
        client.files[0] = ("SKILL.md", "file", b"# Mutated content\n")
    elif mutation == "description":
        client.description = "Changed owner copy."
    elif mutation == "manifest":
        changed = json.loads(client.files[1][2])
        changed["name"] = "changed-manifest"
        client.files[1] = (
            "skill.manifest.json",
            "file",
            canonical_json_bytes(changed),
        )
    else:
        client.revision = "revision-2"

    with pytest.raises(PackagePolicyError, match="receipt is stale"):
        service.approve("draft-review")
    assert client.approvals == []
    assert client.publications == 0


def test_successful_approval_consumes_receipt_and_replay_is_denied(tmp_path: Path):
    client = ReviewClient()
    service = _review_service(tmp_path, client=client)
    store = service.store
    local = store.draft("draft-review")
    assert local is not None
    event_id = store.emit_local_event(
        kind="wisdom.candidate",
        skill_id=str(local["skill_id"]),
        content_hash=str(local["source_hash"]),
        payload={"skill_name": "reviewed-skill", "local_reasons": {}},
        session_id="session-1",
        task_id="task-1",
        qualification="manual_selection",
    )
    assert event_id is not None
    reviewed = service.review("draft-review", acknowledge=True)

    result = service.approve("draft-review")

    assert result["publication"]["state"] == "published"
    assert client.approvals == [
        {
            "content_hash": reviewed["hashes"]["content"],
            "description_hash": reviewed["hashes"]["author_description"],
            "manifest_hash": reviewed["hashes"]["package_manifest"],
        }
    ]
    assert client.publications == 1
    assert store.receipt("draft-review") is None
    assert store.draft("draft-review")["state"] == "published"
    assert store.local_events(kind="wisdom.candidate", session_id="session-1") == []
    with store.transaction() as db:
        candidate_state = db.execute(
            "SELECT state FROM candidate WHERE skill_id=? AND content_hash=?",
            (str(local["skill_id"]), str(local["source_hash"])),
        ).fetchone()[0]
    assert candidate_state == "contributed"
    with pytest.raises(PackagePolicyError, match="fresh complete-package review"):
        service.approve("draft-review")
    assert client.publications == 1


def test_setup_rejects_an_unsafe_org_path_before_writing_marker(
    monkeypatch, tmp_path: Path
):
    skills = tmp_path / "skills"
    monkeypatch.setattr("hermes_wisdom.service.get_skills_dir", lambda: skills)
    service = WisdomService(
        store=WisdomStore(tmp_path / "state"), client=SetupClient("../other-org")
    )

    with pytest.raises(WisdomValidationError, match="malformed"):
        service.setup(disclosure_accepted=True)

    assert not (skills / "_wisdom" / ".active_org").exists()


def test_install_retries_from_staged_bytes_after_directory_swap_failure(
    monkeypatch, tmp_path: Path
):
    client = InstallClient()
    service = _install_service(monkeypatch, tmp_path, client=client)
    plan = service.install_plan("skill-1")
    real_replace = __import__("os").replace
    failed = False

    def replace_once(source, destination):
        nonlocal failed
        if not failed and Path(destination).name == "managed-skill":
            failed = True
            raise OSError("injected swap failure")
        return real_replace(source, destination)

    monkeypatch.setattr("hermes_wisdom.service.os.replace", replace_once)
    with pytest.raises(OSError, match="injected"):
        service.install_apply(plan["receipt"])
    assert service.store.pending_operations()[0]["phase"] == "staged"

    monkeypatch.setattr("hermes_wisdom.service.os.replace", real_replace)
    assert service.reconcile_pending_install_records() == ["skill-1"]
    installation = service.store.installation("skill-1")
    assert installation["state"] == "active"
    assert Path(installation["target_path"], "SKILL.md").read_text() == "# Managed\n"


def test_portal_install_url_preserves_explicit_version_selector():
    service = object.__new__(WisdomService)

    assert service._resolve_install_ref(
        "https://portal.example/orgs/team/wisdom/skills/skill-1?version=7"
    ) == ("skill-1", 7)
    with pytest.raises(PackagePolicyError, match="invalid Wisdom version"):
        service._resolve_install_ref(
            "https://portal.example/orgs/team/wisdom/skills/skill-1?version=latest"
        )


def test_version_detail_resolves_metadata_and_profile_portal_url(
    monkeypatch, tmp_path: Path
):
    class VersionClient:
        def skill(self, skill_id):
            value = {
                "skill": {"id": skill_id, "slug": "managed-skill"},
                "versions": [{"version": 2}, {"version": 1}],
            }
            return SimpleNamespace(
                **value, model_dump=lambda mode: copy.deepcopy(value)
            )

        def version(self, skill_id, version):
            value = {
                "skill": {"id": skill_id, "slug": "managed-skill"},
                "version": {
                    "version": version,
                    "author_description": f"Release {version}",
                    "content_hash": f"sha256:content-{version}",
                    "system_spec": {},
                },
            }
            return SimpleNamespace(
                **value, model_dump=lambda mode: copy.deepcopy(value)
            )

    store = WisdomStore(tmp_path / "state")
    store.activate_installation_identity("hwi_test", "nas_organisation:wisdom-local")
    service = WisdomService(store=store, client=VersionClient())
    monkeypatch.setattr(
        "hermes_wisdom.service.portal_base_url", lambda: "http://127.0.0.1:3111"
    )

    detail = service.version_detail("skill-1", 1, include_compatibility=False)

    assert detail["version"]["author_description"] == "Release 1"
    assert detail["portal_url"] == (
        "http://127.0.0.1:3111/orgs/wisdom-local/wisdom/skills/skill-1?version=1"
    )


def test_command_home_does_not_read_remote_collections_when_status_is_degraded(
    monkeypatch, tmp_path: Path
):
    service = WisdomService(
        store=WisdomStore(tmp_path / "state"),
        client=SimpleNamespace(),
    )
    degraded = {
        "configured": True,
        "gateway_available": False,
        "capability_advertised": False,
        "entitled": False,
        "dogfood_admin_claim": False,
    }
    monkeypatch.setattr(service, "status", lambda: degraded)
    monkeypatch.setattr(
        service,
        "search_skills",
        lambda *_args, **_kwargs: pytest.fail("must not query the unavailable plane"),
    )

    assert service.command_home() == {"status": degraded}


def test_status_distinguishes_authentication_failure_from_plane_unavailability(
    tmp_path: Path,
):
    class AuthenticationFailureClient:
        def capability(self):
            raise WisdomAuthError("expired local credential")

    service = WisdomService(
        store=WisdomStore(tmp_path / "state"),
        client=AuthenticationFailureClient(),
    )

    status = service.status()

    assert status["gateway_available"] is False
    assert status["error_kind"] == "authentication"
    assert status["error"] == "expired local credential"


def test_install_recovery_verifies_target_when_swap_won_before_journal_advance(
    monkeypatch, tmp_path: Path
):
    client = InstallClient()
    service = _install_service(monkeypatch, tmp_path, client=client)
    plan = service.install_plan("skill-1")
    real_advance = service.store.advance
    failed = False

    def advance_once(operation_id, phase, *, done=False):
        nonlocal failed
        if not failed and phase == "files_committed":
            failed = True
            raise OSError("injected journal failure")
        return real_advance(operation_id, phase, done=done)

    monkeypatch.setattr(service.store, "advance", advance_once)
    with pytest.raises(OSError, match="journal"):
        service.install_apply(plan["receipt"])
    pending = service.store.pending_operations()[0]
    payload = json.loads(pending["payload_json"])
    assert pending["phase"] == "staged"
    assert not Path(payload["staging_path"]).exists()
    assert Path(payload["target_path"]).is_dir()

    monkeypatch.setattr(service.store, "advance", real_advance)
    result = service.install_apply(plan["receipt"])
    assert result["installed"] is True


def test_install_retries_only_gateway_record_after_local_commit(
    monkeypatch, tmp_path: Path
):
    client = InstallClient(fail_record=True)
    service = _install_service(monkeypatch, tmp_path, client=client)
    plan = service.install_plan("skill-1")
    with pytest.raises(RuntimeError, match="network down"):
        service.install_apply(plan["receipt"])
    assert service.store.pending_operations()[0]["phase"] == "local_ledger_committed"
    result = service.install_apply(plan["receipt"])
    assert result["installed"] is True
    assert service.store.pending_operations() == []
