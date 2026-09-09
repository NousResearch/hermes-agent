import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from gateway.session_context import clear_session_vars, set_session_vars
from hermes_wisdom.agent_led.schemas import SharePackage
from hermes_wisdom.agent_led.setup_document import SETUP_PATH
from hermes_wisdom.agent_led.share_flow import normalize_generated_package
from hermes_wisdom.client import WisdomNotFound
from hermes_wisdom.consent import ConsentActor, WisdomConsent
from hermes_wisdom.installed_setup import inspect_installed_setup
from hermes_wisdom.mediation_view import advice_view, interaction_view, resolve_surface_action
from hermes_wisdom.service import WisdomService
from hermes_wisdom.store import WisdomStore
from tests.wisdom.test_service import InstallClient, _install_service
from tools import wisdom_tool  # noqa: F401 - real tool registration
from tools.registry import registry


@pytest.fixture
def setup(tmp_path, monkeypatch):
    import hermes_cli.config as config
    from tools.approval import register_gateway_notify, resolve_gateway_approval, unregister_gateway_notify

    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text("approvals:\n  mode: manual\nterminal:\n  env: local\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    config._LOAD_CONFIG_CACHE.clear()
    client = InstallClient()
    service = _install_service(monkeypatch, tmp_path, client=client)
    monkeypatch.setattr("hermes_wisdom.consumption.get_skills_dir", lambda: tmp_path / "skills")
    package = SharePackage.model_validate({
        "skill_name": "managed-skill", "source_content_hash": "sha256:source",
        "editorial_name": "Managed skill", "plain_description": "A setup test fixture.",
        "files": [{"path": "SKILL.md", "content": "# Managed\n"}],
        "requirements": [{"kind": "account", "name": "example", "purpose": "Example account"}],
        "setup_instructions": ["Create the local test marker once."],
        "verification_step": "Check the marker exists and contains a single entry.",
    })
    normalized = normalize_generated_package(package)
    guide = next(file.content for file in normalized.files if file.path == SETUP_PATH)
    client.files.append((SETUP_PATH, "file", guide.encode()))
    service.install_apply(service.install_plan("skill-1")["receipt"])
    monkeypatch.setattr("hermes_wisdom.service._config", lambda: {
        "enabled": True, "disclosure_acknowledged_at": "test", "delivery_mode": "agent",
    })
    monkeypatch.setattr("hermes_wisdom.service.WisdomService", lambda: service)
    monkeypatch.setenv("TERMINAL_ENV", "local")
    actor = ConsentActor("setup-session", "telegram", "owner", "chat")
    register_gateway_notify(actor.session_key, lambda request: resolve_gateway_approval(
        actor.session_key, "once", request_id=request["request_id"],
    ))
    tokens = set_session_vars(platform="telegram", session_key=actor.session_key,
                              chat_type="dm", chat_id="chat", user_id="owner")
    yield service, actor, tmp_path
    clear_session_vars(tokens)
    unregister_gateway_notify(actor.session_key)
    config._LOAD_CONFIG_CACHE.clear()


def _present(phase, command="", *, index=0):
    result = json.loads(registry.dispatch("present_wisdom_consent", {
        "kind": "setup", "identity": "skill-1", "version": 1,
        "step": {"phase": phase, "index": index, "command": command},
        "title": "Review setup", "explanation": "Complete the installed skill's declared setup.",
    }))
    assert "interaction" in result, result
    return result["interaction"]


def _python(source):
    args = [sys.executable, "-c", source]
    return subprocess.list2cmdline(args) if os.name == "nt" else shlex.join(args)


def _settle(consent, actor, identity):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        consent.recover("org-1")
        result = consent.resolve("org-1", identity, actor, "inspect")
        if result["state"] != "applying":
            return result
        time.sleep(0.05)
    pytest.fail(f"setup did not finish: {result}")


def test_native_setup_runs_once_and_verifies_after_prerequisites(setup):
    service, actor, root = setup
    consent = WisdomConsent(service)
    marker = root / "marker"
    command = _python(f"from pathlib import Path; p=Path({str(marker)!r}); p.open('a').write('one')")
    card = _present("setup", command)
    assert not marker.exists()
    from hermes_wisdom.mediation import WisdomMediation
    from hermes_wisdom.delivery import DeliveryReceipt

    mediation = WisdomMediation(service)
    mediation.queue.register_session(
        "org-1", session_key=actor.session_key, session_id=actor.session_key,
        platform=actor.platform, actor_id=actor.actor_id, private=True,
        available=True, user_activity=True, address=actor.address,
    )
    def no_model(*args, **kwargs):
        pytest.fail("an explicit setup proposal must not generate unsolicited advice")
    items = mediation.prepare("org-1", actor, runtime={}, history=[], assessor=no_model)
    assert len(items) == 1 and items[0]["interaction"]["id"] == card["id"]
    assert mediation.begin_delivery("org-1", items) == items
    job = items[0]["assessment"]
    assert mediation.queue.complete_delivery("org-1", job["id"], job["lease_token"], receipt=DeliveryReceipt(
        platform=actor.platform, destination=actor.chat_id, thread_id="", scope_id="",
        message_id="setup-card", acknowledgement="provider_accepted",
    ))
    rendered = advice_view(items)
    assert command in rendered.items[0].detail
    assert rendered.actions[-1].label == "Run this step"
    from plugins.platforms.telegram.adapter import TelegramAdapter
    import html
    assert command in html.unescape(TelegramAdapter._wisdom_command_html(rendered, full_details=True))
    with pytest.raises(WisdomNotFound):
        consent.resolve("org-1", card["id"], replace(actor, actor_id="someone-else"), "confirm")
    assert not marker.exists()
    resolve_surface_action(
        service, rendered.actions[-1].callback_data, platform=actor.platform,
        actor_id=actor.actor_id, chat_id=actor.chat_id,
    )
    consent.resolve("org-1", card["id"], actor, "confirm")
    completed = _settle(consent, actor, card["id"])
    assert completed["state"] == "completed", completed
    assert marker.read_text() == "one"
    assert interaction_view(completed).summary == "Command completed"
    with service.store.transaction() as db:
        assert db.execute("SELECT count(*) FROM wisdom_operation_outbox").fetchone()[0] == 0

    verify = _python(f"from pathlib import Path; assert Path({str(marker)!r}).read_text() == 'one'")
    rejected = json.loads(registry.dispatch("present_wisdom_consent", {
        "kind": "setup", "identity": "skill-1", "version": 1,
        "step": {"phase": "verify", "command": verify},
        "title": "Verify", "explanation": "Verify setup.",
    }))
    assert "prerequisites" in str(rejected)
    acknowledgement = _present("prerequisite")
    consent.resolve("org-1", acknowledgement["id"], actor, "confirm")
    verification = _present("verify", verify)
    consent.resolve("org-1", verification["id"], actor, "confirm")
    assert _settle(consent, actor, verification["id"])["state"] == "completed"
    restarted = WisdomService(store=WisdomStore(root / "state"), client=service.client)
    inspected = inspect_installed_setup(restarted.store, "skill-1", version=1)
    assert inspected["verification"]["state"] == "passed"
    assert inspected["ready_to_use"] is True
    assert marker.read_text() == "one"
    from hermes_wisdom.consumption import WisdomConsumption
    from tests.wisdom.test_consumption import Client

    target = Path(service.store.installation("skill-1")["target_path"])
    files = [(name, "file", (target / name).read_bytes()) for name in service.store.installation("skill-1")["baseline"]]
    manager = WisdomConsumption(store=service.store, client=Client(files, mode="MANUAL"), scan=lambda path: {
        "guard": {"allowed": True, "findings": [], "reason": None},
        "skill_evaluator": {"status": "disabled", "findings": []},
    }, config={})
    manager.update_apply(manager.update_plan("skill-1")["receipt"])
    updated = inspect_installed_setup(service.store, "skill-1", version=2)
    assert updated["ready_to_use"] is None
    assert updated["verification"] == {"state": "not_recorded"}


@pytest.mark.parametrize("failure", ["changed", "expired", "spawn_unknown", "lost_handle", "exit_failure", "sandbox", "secret", "defer", "denied"])
def test_setup_authority_failure_and_uncertainty_never_replay(setup, monkeypatch, failure):
    service, actor, root = setup
    consent = WisdomConsent(service)
    marker = root / "marker"
    command = _python(f"from pathlib import Path; Path({str(marker)!r}).write_text('once')")
    if failure == "exit_failure":
        command = _python("raise SystemExit(3)")
    if failure == "secret":
        response = registry.dispatch("present_wisdom_consent", {
            "kind": "setup", "identity": "skill-1", "version": 1,
            "step": {"phase": "setup", "command": "echo sk-aaaaaaaaaaaaaaaaaaaaaaaa"},
            "title": "Setup", "explanation": "Configure setup.",
        })
        assert "credential-shaped" in response
        assert "sk-aaaaaaaaaaaaaaaaaaaaaaaa" not in response
        assert not service.store.pending_operations()
        return
    card = _present("setup", command)
    if failure == "changed":
        (Path(service.store.installation("skill-1")["target_path"]) / "SKILL.md").write_text("changed")
    elif failure == "expired":
        consent.queue.clock = lambda: card["expires_at"] + 1
    elif failure == "spawn_unknown":
        def interrupted(**kwargs):
            raise RuntimeError("interrupted before a spawn acknowledgement")
        monkeypatch.setattr("tools.terminal_tool.terminal_tool", interrupted)
    elif failure == "sandbox":
        monkeypatch.setattr("tools.terminal_tool._get_env_config", lambda: {"env_type": "docker"})
    elif failure == "denied":
        from tools.approval import register_gateway_notify, resolve_gateway_approval
        register_gateway_notify(actor.session_key, lambda request: resolve_gateway_approval(
            actor.session_key, "deny", request_id=request["request_id"],
        ))
    elif failure == "defer":
        # Setup deferral is local; it must not suppress future team recommendations.
        monkeypatch.setattr("hermes_wisdom.preferences.WisdomPreferences.identity", lambda *args: "owner")
        assert consent.resolve("org-1", card["id"], actor, "defer")["deferred"]
        successor = _present("setup", command)
        assert successor["id"] != card["id"]
        assert not marker.exists()
        return
    result = consent.resolve("org-1", card["id"], actor, "confirm")
    if failure == "lost_handle":
        journal = next(row for row in service.store.pending_operations() if row["kind"] == "wisdom_setup")
        process_id = json.loads(journal["payload_json"])["process_id"]
        from tools.process_registry import process_registry
        process_registry.wait(process_id, timeout=10)
        monkeypatch.setattr(process_registry, "get", lambda identity: None)
        result = consent.resolve("org-1", card["id"], actor, "inspect")
        assert result["result"]["setup"]["state"] == "unknown"
        assert service.store.pending_operations()
    elif failure == "exit_failure":
        result = _settle(consent, actor, card["id"])
        assert result["state"] == "failed"
    elif failure == "denied":
        assert result["state"] == "failed"
        assert result["result"]["setup"]["state"] == "blocked"
        assert not marker.exists()
        assert not service.store.pending_operations()
    else:
        assert result["state"] in {"needs_review", "expired", "stale"}, result
        assert not marker.exists()
    consent.resolve("org-1", card["id"], actor, "confirm")
    if failure in {"spawn_unknown", "lost_handle"}:
        inspected = inspect_installed_setup(service.store, "skill-1", version=1)
        assert inspected["setup_progress"][-1]["state"] == "unknown"
        assert inspected["ready_to_use"] is not True
        response = registry.dispatch("present_wisdom_consent", {
            "kind": "setup", "identity": "skill-1", "version": 1,
            "step": {"phase": "setup", "command": command},
            "title": "Setup", "explanation": "Retry setup.",
        })
        assert "unfinished" in response
