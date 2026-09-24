import json
import os
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_wisdom.agent_led.schemas import SharePackage
from hermes_wisdom.agent_led.setup_document import SETUP_PATH
from hermes_wisdom.agent_led.share_flow import normalize_generated_package
from hermes_wisdom.client import WisdomNotFound
from hermes_wisdom.delivery import DeliveryReceipt
from hermes_wisdom.installed_setup import inspect_installed_setup
from hermes_wisdom.mediation import WisdomMediation
from hermes_wisdom.mediation_view import desktop_interaction, interaction_view
from tests.wisdom.test_setup_execution import _python, _settle, setup  # noqa: F401


@pytest.mark.parametrize("setup", [False], indirect=True)
@pytest.mark.parametrize("copy_mode", ["agent", "fixed"])
@pytest.mark.parametrize("kind", ["env_var", "command"])
@pytest.mark.parametrize("case", ["complete", "disappeared", "deferred", "changed"])
def test_missing_prerequisite_can_be_rechecked_without_secret_or_command(setup, monkeypatch, copy_mode, kind, case):
    from hermes_cli.config import remove_env_value, save_env_value

    service, actor, root = setup
    name = "WISDOM_SETUP_TEST_VALUE" if kind == "env_var" else "wisdom-setup-test" + (".cmd" if os.name == "nt" else "")
    monkeypatch.delenv(name, raising=False)
    binary_dir = root / "bin"
    binary_dir.mkdir()
    executable = binary_dir / name
    monkeypatch.setenv("PATH", str(binary_dir) + os.pathsep + os.environ.get("PATH", ""))
    package = normalize_generated_package(SharePackage.model_validate({
        "skill_name": "managed-skill", "source_content_hash": "sha256:source",
        "editorial_name": "Managed skill", "plain_description": "Verify a private prerequisite.",
        "files": [{"path": "SKILL.md", "content": "# Managed\n"}],
        "requirements": [{"kind": kind, "name": name, "purpose": "Connect the test integration.",
                          "handoff": "Obtain access from your integration administrator."}],
        "verification_step": "Verify that the declared prerequisite is available to Hermes.",
    }))
    guide = next(file.content.encode() for file in package.files if file.path == SETUP_PATH)
    service.client.files = [row for row in service.client.files if row[0] != SETUP_PATH] + [(SETUP_PATH, "file", guide)]
    service.client.identity, service.client.display_org_id = {"owner": "account-user"}, "org-1"
    original_version = service.client.version

    def version_detail(skill, version):
        value = original_version(skill, version).version
        value["security_check"] = {"status": "pass"}
        return SimpleNamespace(version=value, model_dump=lambda **kwargs: {"version": {**value, "version": version}})

    monkeypatch.setattr(service.client, "version", version_detail)
    monkeypatch.setattr("hermes_wisdom.service._config", lambda: {
        "enabled": True, "disclosure_acknowledged_at": "test", "notifications": {"delivery_mode": copy_mode},
    })
    now = [time.time()]
    mediation = WisdomMediation(service, clock=lambda: now[0])
    consent = mediation.consent
    runtime = {"model": "test-model", "provider": "test-provider"}
    marker = root / "verified"
    private_value = "private-test-value-not-for-chat"
    model_calls = []

    def model(**kwargs):
        evidence = json.loads(kwargs["messages"][-1]["content"])
        assert private_value not in json.dumps(kwargs, default=str)
        assert evidence["phase"] == "verify"
        model_calls.append(evidence)
        check = f"import os; assert os.environ.get({name!r})" if kind == "env_var" else f"import shutil; assert shutil.which({name!r})"
        command = _python(check + f"; from pathlib import Path; Path({str(marker)!r}).write_text('verified')")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None, content=json.dumps({
            "command": command, "explanation": "Verify availability without displaying its value.",
        })))])

    monkeypatch.setattr("agent.auxiliary_client.call_llm", model)

    def prepare():
        mediation.queue.register_session("org-1", session_key=actor.session_key, session_id=actor.session_key,
                                         platform=actor.platform, actor_id=actor.actor_id, private=True,
                                         available=True, user_activity=True, address=actor.address)
        return mediation.prepare("org-1", actor, runtime=runtime, history=[],
                                 assessor=lambda *a, **k: pytest.fail("explicit setup is not a relevance assessment"))

    def deliver(items):
        selected = mediation.begin_delivery("org-1", items)
        assert selected == items and mediation.delivery_ready("org-1", selected)
        for item in items:
            job = item["assessment"]
            assert mediation.queue.complete_delivery("org-1", job["id"], job["lease_token"], receipt=DeliveryReceipt(
                platform=actor.platform, destination=actor.chat_id, thread_id="", scope_id="",
                message_id=job["id"], acknowledgement="provider_accepted",
            ))

    prepare()
    install = consent.request("org-1", {"kind": "skill", "skill_id": "skill-1", "version": 1}, actor,
                              title="Review installation", explanation="Install the test fixture.")
    deliver(prepare())
    assert consent.resolve("org-1", install["id"], actor, "confirm")["state"] == "completed"
    items = prepare()
    assert len(items) == 1
    blocked = items[0]["interaction"]
    assert blocked["facts"]["setup_requirement"]["name"] == name
    assert blocked["facts"]["allowed"] is False
    assert "confirm" not in blocked["actions"]
    assert [item.label for item in interaction_view(blocked).actions] == ["Not Now", "Recheck"]
    desktop = desktop_interaction(blocked)["setup_review"]
    assert [item["action"] for item in desktop["actions"]] == ["defer", "recheck"]
    assert "integration administrator" in desktop["detail"] and name in desktop["detail"]
    assert not model_calls and not marker.exists()
    deliver(items)
    with pytest.raises(WisdomNotFound):
        consent.resolve("org-1", blocked["id"], replace(actor, actor_id="other"), "recheck")
    refused = consent.resolve("org-1", blocked["id"], actor, "confirm")
    assert refused["state"] == "pending" and "confirm" not in refused["actions"]
    with service.store.transaction() as db:
        assert db.execute("SELECT count(*) FROM operation_journal WHERE kind='wisdom_setup'").fetchone()[0] == 0
    if case == "deferred":
        assert consent.resolve("org-1", blocked["id"], actor, "defer")["deferred"]
        assert prepare() == []
        blocked = consent.resolve("org-1", install["id"], actor, "setup.status")
        assert blocked["facts"]["allowed"] is False
    again = consent.resolve("org-1", blocked["id"], actor, "recheck")
    assert again["id"] != blocked["id"] and again["facts"]["allowed"] is False
    assert not model_calls
    if case == "changed":
        target = Path(service.store.installation("skill-1")["target_path"])
        (target / "SKILL.md").write_text("changed since prerequisite review")
        with pytest.raises(ValueError):
            consent.resolve("org-1", again["id"], actor, "recheck")
        assert not model_calls and not marker.exists()
        return
    if kind == "env_var":
        save_env_value(name, private_value)
    else:
        executable.write_text("@exit /b 0\n" if os.name == "nt" else "#!/bin/sh\nexit 0\n")
        executable.chmod(0o755)
    mediation = WisdomMediation(service, clock=lambda: now[0])
    consent = mediation.consent
    current = consent.resolve("org-1", again["id"], actor, "recheck")
    assert current["facts"]["allowed"] is True and current["facts"]["setup_requirement"]["status"] == "present"
    assert private_value not in json.dumps(current) and not model_calls and not marker.exists()
    if case == "disappeared":
        if kind == "env_var":
            remove_env_value(name)
        else:
            executable.unlink()
        result = consent.resolve("org-1", current["id"], actor, "confirm")
        assert result["state"] != "completed" and not marker.exists()
        return
    assert consent.resolve("org-1", current["id"], actor, "confirm")["state"] == "completed"
    steps = prepare()
    assert len(steps) == 1 and len(model_calls) == 1 and not marker.exists()
    verification = steps[0]["interaction"]
    deliver(steps)
    consent.resolve("org-1", verification["id"], actor, "confirm")
    assert _settle(consent, actor, verification["id"])["state"] == "completed"
    assert marker.read_text() == "verified"
    assert inspect_installed_setup(service.store, "skill-1")["ready_to_use"] is True
    with service.store.transaction() as db:
        for table in ("wisdom_consent", "wisdom_assessment", "operation_journal"):
            assert private_value not in json.dumps([dict(row) for row in db.execute(f"SELECT * FROM {table}")])
