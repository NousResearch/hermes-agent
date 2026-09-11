import json
import sqlite3
from unittest.mock import Mock

import pytest

from hermes_wisdom.mediation import WisdomMediation
from hermes_wisdom.operation_outbox import OperationOutbox
from tests.wisdom.test_operation_outbox import attach_shared_delivery
from tests.wisdom.test_share_queue import sharing  # noqa: F401
from tests.wisdom.test_share_staging import staged  # noqa: F401


def publication(sharing):
    attach_shared_delivery(sharing)
    service, mediation, actor, shown, _, _, now = sharing
    mediation.consent.resolve("org", shown["id"], actor, "confirm")
    final = mediation.prepare(
        "org", actor, runtime={"model": "test-model", "provider": "test-provider"},
        history=[], assessor=Mock(),
    )[0]["interaction"]
    return service, mediation, actor, final, now


def saved(service, identity):
    with service.store.transaction() as db:
        row = dict(db.execute("SELECT * FROM wisdom_consent WHERE id=?", (identity,)).fetchone())
        row["plan"] = json.loads(row["plan_json"])
        row["result"] = json.loads(row["result_json"]) if row["result_json"] else None
        return row


def test_publication_intent_must_commit_before_approval(sharing):
    service, mediation, actor, final, _ = publication(sharing)
    service.client.approve = Mock(wraps=service.client.approve)
    service.client.publish = Mock(wraps=service.client.publish)
    with service.store.transaction() as db:
        db.execute("CREATE TRIGGER fail_intent BEFORE UPDATE OF plan_json ON wisdom_consent WHEN NEW.plan_json LIKE '%publication_intent%' BEGIN SELECT RAISE(ABORT,'intent unavailable'); END")
    result = mediation.consent.resolve("org", final["id"], actor, "confirm")
    assert result["state"] == "needs_review"
    service.client.approve.assert_not_called()
    service.client.publish.assert_not_called()


@pytest.mark.parametrize("state", ["published", "pending_moderation"])
@pytest.mark.parametrize("failure", ["lost_response", "process_exit", "local_commit"])
def test_publish_recovers_exact_accepted_intent_without_republishing(sharing, state, failure):
    service, mediation, actor, final, now = publication(sharing)
    approve = service.client.approve
    publish = service.client.publish

    def checked_approve(identity, **hashes):
        row = saved(service, final["id"])
        assert row["state"] == "applying"
        intent = row["plan"]["publication_intent"]
        assert intent["draft_id"] == identity
        assert intent["server_revision"] == "revision-1"
        assert intent["hashes"] == row["plan"]["hashes"]
        assert intent["org_id"] == "org" and intent["owner_user_id"] == "account-user"
        return approve(identity, **hashes)

    def interrupted_publish(identity, **kwargs):
        publish(identity, **kwargs)
        service.client.drafts[identity] = service.client.drafts[identity].model_copy(
            update={"state": state, "updatedAt": "revision-after-publication"},
        )
        if failure == "lost_response":
            raise TimeoutError("lost Gateway reply")
        if failure == "process_exit":
            raise SystemExit("interrupted after Gateway commit")
        return {"state": state}

    service.client.approve = Mock(side_effect=checked_approve)
    service.client.publish = Mock(side_effect=interrupted_publish)
    if failure == "local_commit":
        with service.store.transaction() as db:
            db.execute("CREATE TRIGGER fail_publish_report BEFORE INSERT ON wisdom_operation_outbox BEGIN SELECT RAISE(ABORT,'injected'); END")
    if failure == "lost_response":
        assert mediation.consent.resolve("org", final["id"], actor, "confirm")["state"] == "applying"
    else:
        with pytest.raises(SystemExit if failure == "process_exit" else sqlite3.IntegrityError):
            mediation.consent.resolve("org", final["id"], actor, "confirm")
    if failure == "local_commit":
        with service.store.transaction() as db:
            db.execute("DROP TRIGGER fail_publish_report")
    assert saved(service, final["id"])["state"] == "applying"
    now[0] += 901
    reopened = WisdomMediation(service, clock=lambda: now[0])
    reopened.consent.recover("org")
    row = saved(service, final["id"])
    assert row["state"] == "completed"
    assert row["result"]["publication_state"] == state
    assert row["result"]["reason"] == "gateway_publication_reconciled"
    assert service.store.draft(row["result"]["draft_id"])["state"] == state
    assert service.store.receipt(row["result"]["draft_id"]) is None
    reopened.consent.recover("org")
    OperationOutbox(service, clock=lambda: now[0]).flush("org")
    service.client.approve.assert_called_once()
    service.client.publish.assert_called_once()
    assert service.client.publications == 1
    with service.store.transaction() as db:
        assert db.execute("SELECT count(*) FROM wisdom_operation_outbox WHERE interaction_id=?", (final["id"],)).fetchone()[0] == 1


@pytest.mark.parametrize("change", ["owner", "draft_commit", "content", "unpublished", "offline"])
def test_publish_recovery_requires_authoritative_matching_evidence(sharing, change):
    service, mediation, actor, final, now = publication(sharing)
    publish = service.client.publish

    def crash(identity, **kwargs):
        publish(identity, **kwargs)
        raise SystemExit()

    service.client.publish = Mock(side_effect=crash)
    with pytest.raises(SystemExit):
        mediation.consent.resolve("org", final["id"], actor, "confirm")
    if change == "offline":
        service.client.reconstruct_draft = Mock(side_effect=TimeoutError())
    elif change == "owner":
        service.client.identity = {"owner": "other-account"}
    else:
        updates = {"draft_commit": {"draftCommit": "different"},
                   "content": {"contentHash": "sha256:different"},
                   "unpublished": {"state": "ready"}}[change]
        service.client.drafts["draft-1"] = service.client.drafts["draft-1"].model_copy(update=updates)
    now[0] += 901
    mediation.consent.recover("org")
    assert saved(service, final["id"])["state"] != "completed"
    service.client.publish.assert_called_once()
    with service.store.transaction() as db:
        assert db.execute("SELECT count(*) FROM wisdom_operation_outbox WHERE interaction_id=? AND report_state='completed'", (final["id"],)).fetchone()[0] == 0


@pytest.mark.parametrize("operation", ["share", "publish"])
def test_local_final_confirmation_persists_acceptance_and_recovers(sharing, operation):
    from dataclasses import replace
    from types import SimpleNamespace
    from tests.wisdom.test_share_queue import register

    service, mediation, actor, final, now = publication(sharing)
    actor = replace(actor, platform="local", chat_id="local:session", thread_id="")
    register(mediation, actor)
    with service.store.transaction() as db:
        saved_plan = json.loads(db.execute("SELECT plan_json FROM wisdom_consent WHERE id=?", (final["id"],)).fetchone()[0])
        saved_plan["origin_address"] = actor.address
        db.execute("UPDATE wisdom_consent SET platform='local',operation=?,plan_json=? WHERE id=?", (operation, json.dumps(saved_plan), final["id"]))
    service.client.agent_led_policy = lambda: SimpleNamespace(publication_mode="moderated")
    prepared = mediation.consent.prepare_local_publication("org", final["id"], actor)
    review = service.publication_review(prepared["draft_id"])
    publish = service.client.publish

    def interrupted(identity, **kwargs):
        row = saved(service, final["id"])
        assert row["state"] == "applying" and row["operation"] == "publish"
        assert row["plan"]["publication_intent"]["draft_id"] == identity
        publish(identity, **kwargs)
        raise SystemExit()

    service.client.publish = Mock(side_effect=interrupted)
    with pytest.raises(SystemExit):
        mediation.consent.submit_local_publication("org", final["id"], actor, draft_id=prepared["draft_id"], expected_hashes=review["hashes"], publication_mode="moderated")
    now[0] += 901
    WisdomMediation(service, clock=lambda: now[0]).consent.recover("org")
    assert saved(service, final["id"])["state"] == "completed"
    service.client.publish.assert_called_once()
