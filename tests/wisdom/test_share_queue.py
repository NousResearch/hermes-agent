import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_wisdom.client import WisdomConflict, WisdomNotFound
from hermes_wisdom.consent import ConsentActor
from hermes_wisdom.delivery import DeliveryReceipt
from hermes_wisdom.mediation import WisdomMediation
from hermes_wisdom.mediation_view import advice_view, interaction_view
from tests.wisdom.test_share_staging import staged  # noqa: F401
from tests.wisdom.test_service import FakeClient

RECEIPT = DeliveryReceipt(
    platform="telegram",
    destination="chat",
    thread_id="thread",
    message_id="1",
    acknowledgement="provider_accepted",
)


class PublishingClient(FakeClient):
    """In-memory remote using the production CAS reconstruction path."""

    def __init__(self):
        super().__init__()
        self.objects = {}
        self.sync = SimpleNamespace(get_object=self.objects.__getitem__)
        self.publications = 0

    def upload_private_objects(self, objects):
        self.objects.update(objects.objects)
        self.uploaded += 1

    def submit_draft(self, **payload):
        from hermes_wisdom.client import _walk_private_commit
        from hermes_wisdom.contract import author_description_hash, sha256_address

        draft = super().submit_draft(**payload)
        files = _walk_private_commit(self.sync, draft.draftCommit)
        manifest = next(
            body for path, _, body in files if path == "skill.manifest.json"
        )
        self.drafts[draft.id] = draft.model_copy(
            update={
                "orgId": "org",
                "authorDescriptionHash": author_description_hash(
                    payload["description"]
                ),
                "packageManifestHash": sha256_address(manifest),
            }
        )
        return self.drafts[draft.id]

    def reconstruct_draft(self, identity):
        from hermes_wisdom.client import WisdomClient

        return WisdomClient.reconstruct_draft(self, identity)

    def approve(self, identity, **hashes):
        draft = self.drafts[identity]
        assert hashes == {
            "content_hash": draft.contentHash,
            "description_hash": draft.authorDescriptionHash,
            "manifest_hash": draft.packageManifestHash,
        }
        self.drafts[identity] = draft.model_copy(update={"state": "owner_approved"})
        return self.drafts[identity]

    def publish(self, identity, *, content_hash):
        assert self.drafts[identity].contentHash == content_hash
        self.publications += 1
        self.drafts[identity] = self.drafts[identity].model_copy(
            update={"state": "published"}
        )
        return {"state": "published"}


@pytest.fixture
def sharing(staged, monkeypatch):
    service, package, source, _ = staged
    service._client = PublishingClient()
    monkeypatch.setattr("hermes_wisdom.mediation.delivery_mode", lambda: "agent")
    service.store.activate_installation_identity("installation", "org")
    monkeypatch.setattr(service, "require_setup", Mock())
    skill = service.store.register_skill(
        source, content_hash=package.source_content_hash, source_kind="local"
    )
    event = service.store.emit_local_event(
        kind="wisdom.candidate",
        skill_id=skill,
        content_hash=package.source_content_hash,
        payload={"skill_name": "notes"},
        session_id="session",
        task_id=None,
        qualification="weekly_usage",
    )
    now = [1000.0]
    mediation = WisdomMediation(service, clock=lambda: now[0])
    actor = ConsentActor("session", "telegram", "owner", "chat", "thread")
    register(mediation, actor)
    identity = mediation.queue.enqueue(
        "org",
        f"candidate:{event}",
        {
            "kind": "candidate",
            "event_id": event,
            "content_hash": package.source_content_hash,
        },
        origin_session=actor.session_key,
    )
    job = mediation.queue.claim("org", actor.session_key)[0]
    mediation.queue.save_advice(
        "org",
        identity,
        job["lease_token"],
        {
            "title": "Useful team skill",
            "explanation": "This may help your team.",
            "relevance": "recommend",
        },
    )
    shown = mediation.consent.present("org", identity, actor)
    assert mediation.queue.begin_delivery("org", identity, job["lease_token"])
    assert mediation.queue.complete_delivery(
        "org", identity, job["lease_token"], receipt=RECEIPT
    )
    model = Mock(return_value=package)
    monkeypatch.setattr("hermes_wisdom.share_queue.package_for_share", model)
    return service, mediation, actor, shown, model, source, now


def register(mediation, actor, *, available=True):
    mediation.queue.register_session(
        "org",
        session_key=actor.session_key,
        session_id=actor.session_key,
        actor_id=actor.actor_id,
        platform=actor.platform,
        private=True,
        available=available,
        user_activity=True,
        address=actor.address,
    )


def test_share_is_native_local_preparation_then_separate_exact_publication(sharing):
    service, mediation, actor, shown, model, source, _ = sharing
    assert shown["operation"] == "share"
    assert (
        mediation.consent.resolve("org", shown["id"], actor, "inspect")["operation"]
        == "share"
    )
    model.assert_not_called()
    assert (
        service.store.latest_draft_for_source(
            shown["facts"]["skill_id"],
            service._candidate_event_context(
                mediation.queue.assessments("org")[0]["reference"]["event_id"]
            )[2],
        )
        is None
    )
    first = mediation.consent.resolve("org", shown["id"], actor, "confirm")
    repeat = mediation.consent.resolve("org", shown["id"], actor, "confirm")
    assert first["result"] == repeat["result"]
    assert first["result"]["packaging_state"] == "queued"
    receipt = interaction_view(first)
    assert receipt.summary == "Preparing to share"
    assert "Security check" not in receipt.to_text()
    assert "Professionalism" not in receipt.to_text()
    assert not receipt.actions
    assert len(mediation.queue.assessments("org")) == 2
    model.assert_not_called()
    assert service.client.uploaded == 0
    no_assessment = Mock(side_effect=AssertionError("must reuse packaging result"))
    items = mediation.prepare(
        "org",
        actor,
        runtime={"model": "selected", "provider": "chosen"},
        history=[],
        assessor=no_assessment,
    )
    assert len(items) == 1
    final = items[0]["interaction"]
    assert final["operation"] == "publish" and final["id"] != shown["id"]
    assert "refs/wisdom-setup.md" in final["facts"]["file_names"]
    assert service.client.uploaded == 0 and model.call_count == 1
    assert "ORIGINAL_PRIVATE_SETUP" in (source / "SKILL.md").read_text()
    assert "package" not in mediation.activity().get("assessments", [{}])[-1].get(
        "reference", {}
    )
    result = mediation.consent.resolve("org", final["id"], actor, "confirm")
    assert result["state"] == "completed", result
    assert service.client.uploaded == 1
    assert service.client.publications == 1
    receipt = interaction_view(result)
    assert receipt.summary == "Shared"
    assert "confirmation control" not in receipt.to_text()
    assert "check" not in receipt.to_text().lower()
    assert (
        mediation.consent.resolve("org", final["id"], actor, "confirm")["state"]
        == "completed"
    )
    assert service.client.publications == 1
    assert "ORIGINAL_PRIVATE_SETUP" not in json.dumps(service.client.submissions)


@pytest.mark.parametrize("field", ["actor_id", "session_key", "chat_id", "platform"])
def test_wrong_actor_cannot_queue_packaging(sharing, field):
    _, mediation, actor, shown, model, _, _ = sharing
    wrong = ConsentActor(**{**actor.__dict__, field: "other"})
    with pytest.raises(WisdomNotFound):
        mediation.consent.resolve("org", shown["id"], wrong, "confirm")
    assert len(mediation.queue.assessments("org")) == 1
    model.assert_not_called()


def test_packaging_waits_for_owning_session_safe_boundary(sharing):
    service, mediation, actor, shown, model, _, _ = sharing
    mediation.consent.resolve("org", shown["id"], actor, "confirm")
    register(mediation, actor, available=False)
    assert mediation.prepare("org", actor, runtime={}, history=[]) == []
    other = ConsentActor("other", "slack", "other-user", "different")
    register(mediation, other)
    assert mediation.prepare("org", other, runtime={}, history=[]) == []
    assert service.client.uploaded == 0
    model.assert_not_called()


def test_packaging_retries_resume_persisted_model_output(sharing, monkeypatch):
    service, mediation, actor, shown, model, _, now = sharing
    mediation.consent.resolve("org", shown["id"], actor, "confirm")
    prepare = service.prepare_share_package
    monkeypatch.setattr(
        service, "prepare_share_package", Mock(side_effect=OSError("disk busy"))
    )
    assert mediation.prepare("org", actor, runtime={}, history=[]) == []
    assert model.call_count == 1
    monkeypatch.setattr(service, "prepare_share_package", prepare)
    now[0] += 61
    register(mediation, actor)
    items = mediation.prepare("org", actor, runtime={}, history=[])
    assert items[0]["interaction"]["operation"] == "publish"
    assert model.call_count == 1 and service.client.uploaded == 0


def test_source_change_prevents_packaging_and_upload(sharing):
    service, mediation, actor, shown, model, source, _ = sharing
    mediation.consent.resolve("org", shown["id"], actor, "confirm")
    (source / "SKILL.md").write_text("different")
    assert mediation.prepare("org", actor, runtime={}, history=[]) == []
    model.assert_not_called()
    assert service.client.uploaded == 0


def test_superseded_model_owner_cannot_write_a_proposal(sharing):
    service, mediation, actor, shown, model, _, now = sharing
    mediation.consent.resolve("org", shown["id"], actor, "confirm")
    package = model.return_value

    def expire(*args, **kwargs):
        now[0] += 181
        return package

    model.side_effect = expire
    assert mediation.prepare("org", actor, runtime={}, history=[]) == []
    row = mediation.queue.assessments("org")[-1]
    assert "package" not in row["reference"]
    assert not (service.store.root / "share-staging").exists()
    assert service.client.uploaded == 0


def test_changed_prepared_bytes_are_not_uploaded_from_old_consent(sharing):
    service, mediation, actor, shown, _, _, _ = sharing
    mediation.consent.resolve("org", shown["id"], actor, "confirm")
    item = mediation.prepare("org", actor, runtime={}, history=[])[0]
    draft = service.store.draft(item["assessment"]["reference"]["prepared_draft_id"])
    from pathlib import Path

    (Path(draft["overlay_path"]) / "SKILL.md").write_text("different")
    result = mediation.consent.resolve(
        "org", item["interaction"]["id"], actor, "confirm"
    )
    assert result["state"] in {"stale", "needs_review"}
    assert service.client.uploaded == 0


def test_share_copy_and_controls_keep_publication_separate(sharing):
    _, _, _, shown, _, _, _ = sharing
    view = interaction_view(shown)
    assert [action.label for action in view.actions] == [
        "Not Now",
        "Review first",
        "Share",
    ]
    assert "Nothing is shared without your approval" in view.to_text()
    view = advice_view([
        {
            "advice": {
                "title": "Skill",
                "explanation": "Useful",
                "relevance": "recommend",
            },
            "interaction": shown,
        }
    ])
    assert "You can review the skill before publishing" in view.to_text()
    assert "handoff package" not in view.to_text()


def test_checks_toggle_is_read_only_and_preserves_consent(sharing):
    from hermes_wisdom.mediation_view import resolve_surface_action
    from hermes_wisdom.client import WisdomNotFound

    service, mediation, actor, shown, model, _, _ = sharing
    with service.store.transaction() as db:
        row = db.execute(
            "SELECT plan_json FROM wisdom_consent WHERE id=?", (shown["id"],)
        ).fetchone()
        plan = json.loads(row[0])
        plan["professionalism_check"] = {
            "status": "advisory",
            "summary": "Check the wording.",
            "checks": [
                {
                    "key": "profanity_or_abuse",
                    "status": "pass",
                    "finding_count": 0,
                    "details": [],
                }
            ],
        }
        db.execute(
            "UPDATE wisdom_consent SET plan_json=? WHERE id=?",
            (json.dumps(plan), shown["id"]),
        )

    def toggle(action, user=actor.actor_id):
        return resolve_surface_action(
            service,
            f"wi:agent:checks.{action}:{shown['id']}",
            platform=actor.platform,
            actor_id=user,
            chat_id=actor.chat_id,
            thread_id=actor.thread_id,
        )

    expanded = toggle("show")
    assert "Profanity or abusive language" in expanded.to_text()
    assert expanded.items[0].actions[0].label == "Hide checks"
    collapsed = toggle("hide")
    assert "Profanity or abusive language" not in collapsed.to_text()
    assert (
        "Advisory" in collapsed.to_text()
        and "Check the wording." in collapsed.to_text()
    )
    assert [a.label for a in collapsed.items[0].actions] == [
        "Show checks",
        "Not Now",
        "Review first",
        "Share",
    ]
    with pytest.raises(WisdomNotFound):
        toggle("show", "different-user")
    model.assert_not_called()
    assert (
        mediation.consent.resolve("org", shown["id"], actor, "inspect")["state"]
        == "pending"
    )


def test_private_review_pages_cover_exact_files_without_consuming_consent(sharing):
    service, mediation, actor, shown, _, _, _ = sharing
    mediation.consent.resolve("org", shown["id"], actor, "confirm")
    item = mediation.prepare("org", actor, runtime={}, history=[])[0]
    identity = item["interaction"]["id"]
    first = mediation.consent.resolve("org", identity, actor, "inspect")
    assert first["facts"]["editorial_name"] == "Release Notes"
    seen = {}
    for page in range(first["inspection"]["page_count"]):
        result = mediation.consent.resolve("org", identity, actor, f"inspect.{page}")
        inspection = result["inspection"]
        seen[inspection["path"]] = (
            seen.get(inspection["path"], "") + inspection["content"]
        )
        assert result["state"] == "pending"
        assert len(inspection["content"]) <= 1000
        view = interaction_view(result)
        assert view.actions[-1].primary
        assert view.actions[-1].callback_data == f"wi:agent:confirm:{identity}"
        assert f"/wisdom consent {identity} confirm" in view.to_local_text()
    prepared = service.prepare_candidate(item["assessment"]["reference"]["event_id"])[
        "prepared"
    ]
    assert seen.pop("Author description") == prepared["drafted_description"]
    assert seen == {file["path"]: file["content_utf8"] for file in prepared["files"]}
    assert service.client.uploaded == service.client.publications == 0
    assert all(
        "inspection" not in interaction
        for interaction in mediation.activity()["interactions"]
    )
    wrong = ConsentActor(**{**actor.__dict__, "actor_id": "stranger"})
    with pytest.raises(WisdomNotFound):
        mediation.consent.resolve("org", identity, wrong, "inspect.1")
    with pytest.raises(WisdomNotFound):
        mediation.consent.resolve("org", identity, actor, "inspect.999")


@pytest.mark.parametrize(
    "publication,title",
    [("published", "Shared"), ("pending_moderation", "Submitted for review")],
)
def test_publication_receipt_links_to_portal_without_expanding_checks(
    publication, title
):
    view = interaction_view({
        "state": "completed",
        "operation": "publish",
        "facts": {"editorial_name": "Skill"},
        "result": {
            "publication_state": publication,
            "portal_url": "https://portal.example/review/draft",
        },
    })
    assert view.summary == title
    assert len(view.actions) == 1
    assert view.actions[0].label == "View in Portal"
    assert view.actions[0].url == "https://portal.example/review/draft"


def test_packaging_prompt_contains_full_schema(sharing):
    from hermes_wisdom.agent_led.agent import package_for_share

    service, _, _, shown, model, _, _ = sharing
    package = model.return_value

    def call(messages, schema):
        assert (
            json.loads(messages[0]["content"].split("this exact schema:\n", 1)[1])
            == schema
        )
        assert "PackagedFile" in messages[0]["content"]
        return package.model_dump_json()

    assert package_for_share({}, model_call=call) == package


def test_packaging_failure_eventually_offers_manual_review_not_publication(sharing):
    service, mediation, actor, shown, model, _, now = sharing
    model.side_effect = TimeoutError
    mediation.consent.resolve("org", shown["id"], actor, "confirm")
    for _ in range(3):
        register(mediation, actor)
        assert mediation.prepare("org", actor, runtime={}, history=[]) == []
        now[0] += 61
    register(mediation, actor)
    items = mediation.prepare("org", actor, runtime={}, history=[])
    assert len(items) == 1 and items[0]["interaction"] is None
    assert "Nothing was uploaded or published" in items[0]["advice"]["explanation"]
    assert model.call_count == 3 and service.client.uploaded == 0
    job = items[0]["assessment"]
    assert mediation.queue.begin_delivery("org", job["id"], job["lease_token"])
    assert mediation.queue.complete_delivery(
        "org", job["id"], job["lease_token"], receipt=RECEIPT
    )
    assert mediation.prepare("org", actor, runtime={}, history=[]) == []


def test_private_review_escapes_terminal_controls_before_pagination(sharing):
    service, mediation, actor, shown, model, _, _ = sharing
    package = model.return_value
    package.files[0].content += "\n" + "\x1b[2J" * 350
    mediation.consent.resolve("org", shown["id"], actor, "confirm")
    item = mediation.prepare("org", actor, runtime={}, history=[])[0]
    identity = item["interaction"]["id"]
    first = mediation.consent.resolve("org", identity, actor, "inspect")
    contents = []
    for page in range(first["inspection"]["page_count"]):
        result = mediation.consent.resolve("org", identity, actor, f"inspect.{page}")
        content = result["inspection"]["content"]
        assert len(content) <= 1000
        assert "\x1b" not in content
        contents.append(content)
    assert "\\u001b[2J" in "".join(contents)
    assert service.client.uploaded == 0


def test_share_model_cannot_return_a_tool_call(monkeypatch):
    from hermes_wisdom.agent_led.agent import session_model_call

    call = Mock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[{}]))]
        )
    )
    monkeypatch.setattr("agent.auxiliary_client.call_llm", call)
    runtime = {"provider": "chosen", "model": "session-model"}
    with pytest.raises(ValueError, match="disallowed tools"):
        session_model_call(runtime, name="wisdom_share_packaging")([], {})
    assert call.call_args.kwargs["main_runtime"] is runtime
    assert call.call_args.kwargs["tools"] == []
    assert "task" not in call.call_args.kwargs
    with pytest.raises(ValueError, match="active session model"):
        session_model_call({}, name="wisdom_share_packaging")([], {})


def test_consent_expiring_during_review_cannot_upload(sharing, monkeypatch):
    service, mediation, actor, shown, _, _, now = sharing
    mediation.consent.resolve("org", shown["id"], actor, "confirm")
    item = mediation.prepare("org", actor, runtime={}, history=[])[0]
    final = item["interaction"]

    def slow_review(**kwargs):
        now[0] = final["expires_at"] + 1
        return {"status": "unavailable"}

    monkeypatch.setattr(service, "_require_professionalism_review", slow_review)
    result = mediation.consent.resolve("org", final["id"], actor, "confirm")
    assert result["state"] == "stale"
    assert service.client.uploaded == 0
