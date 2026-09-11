import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_wisdom.publication_cards import PublicationCards, has_card
from tests.wisdom.test_share_staging import staged  # noqa: F401
from tests.wisdom.test_share_queue import sharing  # noqa: F401


@pytest.fixture
def cards(sharing):
    service, mediation, actor, shown, _, _, _ = sharing
    owner = service.client.identity["owner"]
    draft = SimpleNamespace(
        id="draft",
        orgId="org",
        ownerUserId=owner,
        contentHash="content",
        authorDescriptionHash="description",
        packageManifestHash="manifest",
        state="pending_moderation",
    )
    service.client.list_drafts = Mock(return_value=[draft])
    with service.store.transaction() as db:
        plan = json.loads(
            db.execute(
                "SELECT plan_json FROM wisdom_consent WHERE id=?", (shown["id"],)
            ).fetchone()[0]
        )
        plan["hashes"] = {
            "content": "content",
            "author_description": "description",
            "package_manifest": "manifest",
        }
        result = {
            "draft_id": "draft",
            "owner_user_id": owner,
            "publication_state": "pending_moderation",
            "portal_url": "https://portal.example/orgs/org/wisdom/review/draft",
        }
        db.execute(
            "UPDATE wisdom_consent SET operation='publish',state='completed',plan_json=?,result_json=? WHERE id=?",
            (json.dumps(plan), json.dumps(result), shown["id"]),
        )
    return PublicationCards(service), draft, shown["id"]


def test_pending_then_published_updates_one_receipt_without_repeated_edits(cards):
    updater, draft, identity = cards
    (job,) = updater.claim("telegram")
    assert job["view"].summary == "Pending moderation"
    assert job["receipt"]["message_id"] == "1"
    assert updater.claim("telegram") == []  # leased across workers
    updater.finish(job, success=True)
    assert updater.claim("telegram") == []
    draft.state = "published"
    (published,) = updater.claim("telegram")
    assert published["id"] == identity and published["receipt"] == job["receipt"]
    assert published["view"].summary == "Published"
    assert published["view"].actions[0].label == "View in Portal"
    updater.finish(published, success=True)
    assert updater.claim("telegram") == []
    assert has_card(updater.store, "org", "draft")
    assert not has_card(updater.store, "other-org", "draft")
    assert updater.service.client.publications == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("orgId", "other"),
        ("ownerUserId", "other"),
        ("contentHash", "changed"),
        ("authorDescriptionHash", "changed"),
        ("packageManifestHash", "changed"),
    ],
)
def test_wrong_identity_or_changed_package_cannot_update_card(cards, field, value):
    updater, draft, _ = cards
    setattr(draft, field, value)
    assert updater.claim("telegram") == []


def test_failed_edit_is_retried_on_same_message_after_backoff(cards, monkeypatch):
    updater, draft, _ = cards
    monkeypatch.setattr("hermes_wisdom.publication_cards.time.time", lambda: 1000)
    (job,) = updater.claim("telegram")
    updater.finish(job, success=False)
    assert updater.claim("telegram") == []
    monkeypatch.setattr("hermes_wisdom.publication_cards.time.time", lambda: 5000)
    (retry,) = updater.claim("telegram")
    assert retry["receipt"] == job["receipt"]
    updater.finish(retry, success=True)


def test_returned_draft_updates_to_changes_requested(cards):
    updater, draft, _ = cards
    draft.state = "changes_requested"
    (job,) = updater.claim("telegram")
    assert job["view"].summary == "Changes requested"
    assert all(not action.callback_data for action in job["view"].actions)
    updater.finish(job, success=True)


def test_does_not_edit_other_platform_or_unverified_receipt(cards):
    updater, _, identity = cards
    assert updater.claim("slack") == []
    with updater.store.transaction() as db:
        db.execute(
            "UPDATE wisdom_delivery_receipt SET receipt_json=json_set(receipt_json,'$.destination','wrong-chat')"
        )
    assert updater.claim("telegram") == []


def test_gateway_failure_preserves_pending_card(cards):
    updater, _, _ = cards
    updater.service.client.list_drafts.side_effect = TimeoutError
    with pytest.raises(TimeoutError):
        updater.claim("telegram")
    with updater.store.transaction() as db:
        result = json.loads(
            db.execute(
                "SELECT result_json FROM wisdom_consent WHERE operation='publish'"
            ).fetchone()[0]
        )
    assert result["publication_state"] == "pending_moderation"
    assert "card_displayed_state" not in result


def test_older_unchanged_cards_do_not_starve_later_decisions(cards):
    updater, _, identity = cards
    with updater.store.transaction() as db:
        row = dict(
            db.execute(
                "SELECT * FROM wisdom_consent WHERE id=?", (identity,)
            ).fetchone()
        )
        result = json.loads(row["result_json"])
        result["card_displayed_state"] = "pending_moderation"
        for index in range(4):
            older = {
                **row,
                "id": f"older-{index}",
                "created_at": row["created_at"] - 1,
                "result_json": json.dumps(result),
            }
            db.execute(
                f"INSERT INTO wisdom_consent ({','.join(older)}) VALUES ({','.join('?' for _ in older)})",
                tuple(older.values()),
            )
    (job,) = updater.claim("telegram")
    assert job["id"] == identity
    updater.finish(job, success=True)


def test_tracked_decision_stays_in_activity_without_second_assessment(cards):
    from hermes_wisdom.mediation import WisdomMediation

    updater, _, _ = cards
    updater.service.client.display_org_id = "org"
    event = {
        "event_id": "decision",
        "category": "publication_decision",
        "draft_id": "draft",
    }
    updater.service.notifications = Mock(return_value={"events": [event]})
    mediation = WisdomMediation(updater.service)
    mediation.ingest()
    assert all(
        row["event_key"] != "feed:decision"
        for row in mediation.queue.assessments("org")
    )
    assert updater.service.notifications(mark_seen=False)["events"] == [event]


@pytest.mark.parametrize("state", ["pending", "expired", "stale"])
def test_portal_publication_retires_original_consent_controls(cards, state):
    updater, draft, identity = cards
    with updater.store.transaction() as db:
        db.execute("UPDATE wisdom_consent SET state=? WHERE id=?", (state, identity))
    draft.state = "published"
    (job,) = updater.claim("telegram")
    assert job["view"].summary == "Published"
    assert job["receipt"]["message_id"] == "1"
    assert [a.label for a in job["view"].actions] == ["View in Portal"]
    with updater.store.transaction() as db:
        assert (
            db.execute(
                "SELECT state FROM wisdom_consent WHERE id=?", (identity,)
            ).fetchone()[0]
            == "completed"
        )
    updater.finish(job, success=True)
    assert updater.claim("telegram") == []
    assert updater.service.client.publications == 0


def test_portal_revision_chain_tracks_moderation_then_publication(cards):
    updater, original, identity = cards
    with updater.store.transaction() as db:
        db.execute("UPDATE wisdom_consent SET state='pending' WHERE id=?", (identity,))
    original.state = "invalidated"
    revision = SimpleNamespace(**{
        **vars(original),
        "id": "revision",
        "supersedesDraftId": original.id,
        "contentHash": "edited",
        "state": "invalidated",
    })
    latest = SimpleNamespace(**{
        **vars(revision),
        "id": "latest",
        "supersedesDraftId": revision.id,
        "state": "pending_moderation",
    })
    updater.service.client.list_drafts.return_value = [latest, original, revision]
    updater.service.portal_review_url = Mock(
        side_effect=lambda id: f"https://portal.example/review/{id}"
    )
    (job,) = updater.claim("telegram")
    assert job["view"].summary == "Pending moderation"
    assert job["view"].actions[0].url == "https://portal.example/review/latest"
    updater.finish(job, success=True)
    assert updater.claim("telegram") == []
    latest.state = "published"
    (published,) = updater.claim("telegram")
    assert published["receipt"] == job["receipt"]
    assert published["view"].summary == "Published"
    updater.finish(published, success=True)
    assert updater.claim("telegram") == []
    assert has_card(updater.store, "org", "latest")
    assert updater.service.client.publications == 0


@pytest.mark.parametrize("field,value", [("orgId", "other"), ("ownerUserId", "other")])
def test_portal_revision_cannot_cross_authority(cards, field, value):
    updater, original, _ = cards
    original.state = "invalidated"
    revision = SimpleNamespace(**{
        **vars(original),
        "id": "revision",
        "supersedesDraftId": original.id,
        "state": "published",
        field: value,
    })
    updater.service.client.list_drafts.return_value = [original, revision]
    assert updater.claim("telegram") == []


def test_unrelated_publication_does_not_complete_private_review(cards):
    updater, original, identity = cards
    original.state = "ready"
    original.slug = "same-slug"
    other = SimpleNamespace(**{**vars(original), "id": "other", "state": "published"})
    updater.service.client.list_drafts.return_value = [original, other]
    with updater.store.transaction() as db:
        db.execute("UPDATE wisdom_consent SET state='pending' WHERE id=?", (identity,))
    assert updater.claim("telegram") == []


def test_reviewed_draft_revision_still_preparing_does_not_show_invalidated(cards):
    updater, original, _ = cards
    original.state = "invalidated"
    revision = SimpleNamespace(**{
        **vars(original),
        "id": "revision",
        "supersedesDraftId": original.id,
        "state": "vetting",
    })
    updater.service.client.list_drafts.return_value = [original, revision]
    assert updater.claim("telegram") == []
