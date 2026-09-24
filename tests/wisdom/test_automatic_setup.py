import json
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_wisdom.agent_led.schemas import SharePackage
from hermes_wisdom.agent_led.share_flow import normalize_generated_package
from hermes_wisdom.client import WisdomConflict, WisdomNotFound
from hermes_wisdom.delivery import DeliveryReceipt
from hermes_wisdom.installed_setup import inspect_installed_setup
from hermes_wisdom.mediation import WisdomMediation
from hermes_wisdom.mediation_store import MediationStore
from hermes_wisdom.setup_handoff import finish_automatic_update
from tests.wisdom.test_consumption import Client, _files, _manager
from tests.wisdom.test_setup_execution import _python, _settle, setup  # noqa: F401


@pytest.mark.parametrize("mode", ["AUTO_WITH_NOTICE", "REQUIRED"])
@pytest.mark.parametrize("copy_mode", ["agent", "fixed"])
@pytest.mark.parametrize("case", ["complete", "defer", "changed", "address_changed", "missing_model", "pending_offer"])
def test_automatic_update_handoff_requires_fresh_native_verification(setup, monkeypatch, mode, copy_mode, case):
    service, actor, root = setup
    package = normalize_generated_package(SharePackage.model_validate({
        "skill_name": "managed-skill", "source_content_hash": "sha256:source",
        "editorial_name": "Managed skill", "plain_description": "Verify an existing configuration after update.",
        "files": [{"path": "SKILL.md", "content": "# Managed v2\n"}],
        "verification_step": "Verify that the existing configuration is unchanged.",
    }))
    files = [(file.path, "file", file.content.encode()) for file in package.files]
    files.extend(file for file in _files(2) if file[0] == "skill.manifest.json")
    client = Client(files, mode=mode)
    client.identity, client.display_org_id = {"owner": "account-user"}, "org-1"
    service._client = client
    monkeypatch.setattr("hermes_wisdom.service._config", lambda: {
        "enabled": True, "disclosure_acknowledged_at": "test", "notifications": {"delivery_mode": copy_mode},
    })
    private_config = root / "existing-config"
    private_config.write_text("existing local preferences")
    verified = root / "verification-marker"
    now = [time.time()]
    mediation = WisdomMediation(service, clock=lambda: now[0])
    runtime = {"model": "session-model", "provider": "session-provider"}
    model_calls = []

    def register(who=actor):
        mediation.queue.register_session(
            "org-1", session_key=who.session_key, session_id=who.session_key,
            platform=who.platform, actor_id=who.actor_id, private=True, available=True,
            user_activity=True, address=who.address,
        )

    def model(**kwargs):
        assert kwargs["main_runtime"] == runtime and kwargs["tools"] == []
        evidence = json.loads(kwargs["messages"][-1]["content"])
        assert "existing local preferences" not in kwargs["messages"][-1]["content"]
        assert str(root) not in kwargs["messages"][-1]["content"]
        model_calls.append(evidence["phase"])
        if case == "changed":
            target = Path(service.store.installation("skill-1")["target_path"])
            (target / "SKILL.md").write_text("changed during assessment")
        elif case == "address_changed":
            register(replace(actor, chat_id="different-chat"))
        command = _python(f"from pathlib import Path; assert Path({str(private_config)!r}).read_text() == 'existing local preferences'; Path({str(verified)!r}).open('a').write('v2')")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            tool_calls=None, content=json.dumps({"command": command, "explanation": "Verify without replacing local configuration."}),
        ))])

    monkeypatch.setattr("agent.auxiliary_client.call_llm", model)
    if case == "pending_offer":
        register()
        monkeypatch.setattr(service, "version_detail", lambda *a: {"version": {}})
        offer = mediation.consent.request(
            "org-1", {"kind": "skill", "skill_id": "skill-1", "version": 2}, actor,
            title="Update available", explanation="Review the new version.",
        )
        assert offer["state"] == "pending" and offer["operation"] == "update"
        # Model an earlier delivered offer whose update was never confirmed.
        with service.store.transaction() as db:
            db.execute("UPDATE wisdom_assessment SET state='delivered' WHERE id=?", (offer["assessment_id"],))
    manager = service.consumption
    checked = manager.check(apply_automatic=True)
    now[0] = time.time()
    assert checked["installations"][0]["state"] == "updated"
    assert service.store.installation("skill-1")["version"] == 2
    assert private_config.read_text() == "existing local preferences"
    assert not model_calls and not verified.exists()
    with service.store.transaction() as db:
        assert db.execute("SELECT count(*) FROM wisdom_consent").fetchone()[0] == (1 if case == "pending_offer" else 0)
        assert db.execute("SELECT count(*) FROM wisdom_assessment WHERE event_key LIKE 'setup-update:%'").fetchone()[0] == 1
    assert inspect_installed_setup(service.store, "skill-1")["ready_to_use"] is not True
    if case != "pending_offer":
        assert mediation.prepare("org-1", actor, runtime=runtime, history=[]) == []  # no active private session
    # Re-create the mediator before any step is presented, as after a gateway restart.
    mediation = WisdomMediation(service, clock=lambda: now[0])
    if case == "missing_model":
        runtime.clear()
        for _ in range(3):
            register()
            assert mediation.prepare("org-1", actor, runtime=runtime, history=[]) == []
            job = next(row for row in mediation.queue.assessments("org-1") if row["event_key"].startswith("setup-update:"))
            assert job["state"] == "pending" and job["attempts"] == 0
            assert not model_calls and not verified.exists()
            now[0] += 61
        runtime.update({"model": "session-model", "provider": "session-provider"})
    register()
    items = mediation.prepare("org-1", actor, runtime=runtime, history=[],
                              assessor=lambda *a, **k: pytest.fail("setup is not a relevance recommendation"))
    if case in {"changed", "address_changed"}:
        assert items == [] and model_calls == ["verify"] and not verified.exists()
        return
    assert len(items) == 1 and model_calls == ["verify"]
    step = items[0]["interaction"]
    assert step["operation"] == "setup" and step["facts"]["version"] == 2
    assert step["facts"]["step"]["phase"] == "verify"
    assert not verified.exists()
    selected = mediation.begin_delivery("org-1", items)
    assert selected == items and mediation.delivery_ready("org-1", selected)
    job = items[0]["assessment"]
    assert mediation.queue.complete_delivery("org-1", job["id"], job["lease_token"], receipt=DeliveryReceipt(
        platform=actor.platform, destination=actor.chat_id, thread_id="", scope_id="",
        message_id=job["id"], acknowledgement="provider_accepted",
    ))
    with service.store.transaction() as db:
        assert db.execute("SELECT origin_session FROM wisdom_assessment WHERE id=?", (job["id"],)).fetchone()[0] == actor.session_key
    consent = mediation.consent
    if case == "defer":
        monkeypatch.setattr("hermes_wisdom.preferences.WisdomPreferences.identity", lambda *a: "owner")
        assert consent.resolve("org-1", step["id"], actor, "defer")["deferred"]
        other = replace(actor, session_key="other-session", actor_id="other")
        register(other)
        assert mediation.prepare("org-1", other, runtime=runtime, history=[]) == []
        assert not verified.exists() and model_calls == ["verify"]
        return
    with pytest.raises(WisdomNotFound):
        consent.resolve("org-1", step["id"], replace(actor, actor_id="other"), "confirm")
    consent.resolve("org-1", step["id"], actor, "confirm")
    assert _settle(consent, actor, step["id"])["state"] == "completed"
    consent.resolve("org-1", step["id"], actor, "confirm")
    assert private_config.read_text() == "existing local preferences"
    assert verified.read_text() == "v2"
    assert inspect_installed_setup(service.store, "skill-1", version=2)["ready_to_use"] is True


@pytest.mark.parametrize("mode", ["AUTO_WITH_NOTICE", "REQUIRED"])
@pytest.mark.parametrize("failure", ["record", "queue", "rejected", "manual"])
def test_automatic_update_receipt_and_setup_handoff_commit_together(monkeypatch, tmp_path, mode, failure):
    client = Client(_files(2), mode=mode, fail_record=failure == "record")
    manager, target = _manager(monkeypatch, tmp_path, client=client)
    MediationStore(manager.store)
    real_enqueue = MediationStore.enqueue
    failed = False

    def enqueue(queue, org, event, reference, **kwargs):
        nonlocal failed
        value = real_enqueue(queue, org, event, reference, **kwargs)
        if failure == "queue" and not failed and event.startswith("setup-update:"):
            failed = True
            raise RuntimeError("handoff transaction interrupted")
        return value

    monkeypatch.setattr(MediationStore, "enqueue", enqueue)
    if failure == "rejected":
        client.record_error = WisdomConflict("policy changed")
    if failure == "manual":
        client.mode = "MANUAL"
    plan = manager.update_plan("skill-1")
    with pytest.raises((RuntimeError, WisdomConflict, ValueError)):
        manager.update_apply(plan["receipt"], automatic=True)
    with manager.store.transaction() as db:
        assert db.execute("SELECT count(*) FROM wisdom_assessment").fetchone()[0] == 0
    if failure in {"rejected", "manual"}:
        assert manager.store.installation("skill-1")["version"] == 1
        assert (target / "SKILL.md").read_text() == "# Managed v1\n"
        return
    pending = manager.store.pending_operations()
    assert len(pending) == 1 and pending[0]["phase"] == "local_ledger_committed"
    manager.recover()
    assert manager.store.pending_operations() == []
    assert manager.store.installation("skill-1")["version"] == 2
    operation = manager.store.operation(pending[0]["id"])
    finish_automatic_update(manager.store, operation["id"])
    with manager.store.transaction() as db:
        assert db.execute("SELECT count(*) FROM wisdom_assessment WHERE event_key=?", (f"setup-update:{operation['id']}",)).fetchone()[0] == 1
