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
from hermes_wisdom.client import WisdomConflict, WisdomNotFound
from hermes_wisdom.consent import ConsentActor, WisdomConsent
from hermes_wisdom.installed_setup import inspect_installed_setup
from hermes_wisdom.mediation_view import advice_view, interaction_view, resolve_surface_action
from hermes_wisdom.service import WisdomService
from hermes_wisdom.store import WisdomStore
from tests.wisdom.test_service import InstallClient, _install_service
from tests.wisdom.entitlement_fixtures import authorized_wisdom_token_fixture
from tools import wisdom_tool  # noqa: F401 - real tool registration
from tools.registry import registry


@pytest.fixture
def setup(tmp_path, monkeypatch, request):
    authorized_wisdom_token_fixture(monkeypatch)
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
    if getattr(request, "param", True):
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


@pytest.mark.parametrize("path", ["complete", "expired", "interrupted"])
def test_desktop_api_uses_native_setup_review_and_authority(setup, monkeypatch, path):
    import asyncio
    from hermes_cli.web_models import WisdomConsentRequest
    from hermes_cli.web_routers import wisdom as routes
    from tools.terminal_tool import terminal_tool

    service, original_actor, root = setup
    actor = ConsentActor(original_actor.session_key, "local", "local-user", f"local:{original_actor.session_key}")
    tokens = set_session_vars(platform="desktop", session_key=actor.session_key)
    marker = root / "desktop-marker"
    command = _python(f"from pathlib import Path; Path({str(marker)!r}).open('a').write('once')")
    async def run(profile, fn):
        assert profile == "research"
        return fn(service)
    monkeypatch.setattr(routes, "_run_wisdom", run)

    def action(card, verb, session_id=actor.session_key):
        return asyncio.run(routes.post_wisdom_consent(WisdomConsentRequest(
            interaction_id=card["id"], session_id=session_id, action=verb, profile="research",
        )))

    def codes(card):
        return [item["action"] for item in card["setup_review"]["actions"]]

    try:
        card = _present("setup", command)
        activity = asyncio.run(routes.get_wisdom_mediation("research"))
        displayed = next(item for item in activity["interactions"] if item["id"] == card["id"])
        assert displayed["setup_review"]["command"] == command
        assert card["facts"]["setup_instruction"] in displayed["setup_review"]["detail"]
        assert codes(displayed) == ["defer", "confirm"]
        assert not marker.exists()
        with pytest.raises(WisdomNotFound):
            action(card, "confirm", "another-session")

        if path == "expired":
            with service.store.transaction() as db:
                db.execute("UPDATE wisdom_consent SET expires_at=? WHERE id=?", (time.time() - 1, card["id"]))
            activity = asyncio.run(routes.get_wisdom_mediation("research"))
            expired = next(item for item in activity["interactions"] if item["id"] == card["id"])
            assert codes(expired) == ["recheck"]
            assert expired["state"] == "expired"
            successor = action(card, "recheck")
            assert successor["id"] != card["id"]
            assert codes(successor) == ["defer", "confirm"]
            assert not marker.exists()
            action(card, "confirm")
            assert not marker.exists()
            card = successor
        elif path == "interrupted":
            def interrupted(**kwargs):
                raise RuntimeError("spawn acknowledgement lost")
            monkeypatch.setattr("tools.terminal_tool.terminal_tool", interrupted)
            unknown = action(card, "confirm")
            assert codes(unknown) == ["inspect", "setup.recover"]
            preview = action(card, "setup.recover")
            assert "child processes" in preview["setup_review"]["detail"]
            assert codes(preview) == ["inspect", "setup.clear"]
            cleared = action(card, "setup.clear")
            assert codes(cleared) == ["recheck"]
            card = action(card, "recheck")
            assert not marker.exists()
            monkeypatch.setattr("tools.terminal_tool.terminal_tool", terminal_tool)

        action(card, "confirm")
        consent = WisdomConsent(service)
        _settle(consent, actor, card["id"])
        completed = action(card, "inspect")
        assert codes(completed) == ["setup.status"]
        assert marker.read_text() == "once"
        prerequisite = _present("prerequisite")
        linked = action(card, "setup.status")
        assert linked["id"] == prerequisite["id"]
        assert linked["setup_review"]["actions"][-1]["label"] == "Confirm prerequisite"
        assert linked["setup_review"]["command"] == ""
        assert marker.read_text() == "once"
        with pytest.raises(WisdomNotFound):
            action(card, "setup.status", "another-session")
    finally:
        clear_session_vars(tokens)


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
    from tools.terminal_tool import terminal_tool
    from tools.process_registry import process_registry

    real_get = process_registry.get
    service, actor, root = setup
    consent = WisdomConsent(service)
    marker = root / "marker"
    command = _python(f"from pathlib import Path; Path({str(marker)!r}).open('a').write('once')")
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
        before = marker.read_text() if marker.exists() else ""
        view = interaction_view(consent.resolve("org-1", card["id"], actor, "inspect"))
        recovery = view.actions[-1].callback_data
        assert recovery == f"wi:agent:setup.recover:{card['id']}"
        with pytest.raises(WisdomNotFound):
            resolve_surface_action(service, recovery, platform=actor.platform, actor_id="other", chat_id=actor.chat_id)
        preview = resolve_surface_action(service, recovery, platform=actor.platform, actor_id=actor.actor_id, chat_id=actor.chat_id)
        assert "child processes" in preview.items[0].detail
        assert service.store.pending_operations()
        cleared = resolve_surface_action(service, preview.actions[-1].callback_data,
                                         platform=actor.platform, actor_id=actor.actor_id, chat_id=actor.chat_id)
        assert cleared.summary == "Interrupted step cleared"
        assert not service.store.pending_operations()
        assert (marker.read_text() if marker.exists() else "") == before
        # Neither old confirmation nor repeated clearing authorizes another run.
        consent.resolve("org-1", card["id"], actor, "confirm")
        consent.resolve("org-1", card["id"], actor, "setup.clear")
        successor = consent.resolve("org-1", card["id"], actor, "recheck")
        assert successor["id"] != card["id"] and successor["state"] == "pending"
        assert inspect_installed_setup(service.store, "skill-1", version=1)["ready_to_use"] is not True
        assert (marker.read_text() if marker.exists() else "") == before
        monkeypatch.setattr("tools.terminal_tool.terminal_tool", terminal_tool)
        monkeypatch.setattr(process_registry, "get", real_get)
        consent.resolve("org-1", successor["id"], actor, "confirm")
        assert _settle(consent, actor, successor["id"])["state"] == "completed"
        assert marker.read_text() == before + "once"


@pytest.mark.parametrize("phase", ["before_start", "starting", "running", "returned"])
def test_recovery_cannot_clear_live_execution_even_after_lease_expiry(setup, monkeypatch, phase):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from tools.terminal_tool import terminal_tool
    from hermes_wisdom.setup_execution import execute_step

    service, actor, root = setup
    consent = WisdomConsent(service)
    marker, release_file = root / "marker", root / "release"
    source = f"from pathlib import Path; Path({str(marker)!r}).write_text('once')"
    if phase == "running":
        source = f"import time; from pathlib import Path; exec(\"while not Path({str(release_file)!r}).exists(): time.sleep(0.02)\"); " + source
    card = _present("setup", _python(source))
    entered, release = Event(), Event()

    def pause_before_spawn(**kwargs):
        entered.set()
        assert release.wait(15), "test did not release the starter"
        return terminal_tool(**kwargs)

    def pause_execution(*args):
        if phase == "returned":
            outcome = execute_step(*args)
        entered.set()
        assert release.wait(15), "test did not release the handler"
        return outcome if phase == "returned" else execute_step(*args)

    if phase == "starting":
        monkeypatch.setattr("tools.terminal_tool.terminal_tool", pause_before_spawn)
    elif phase in {"before_start", "returned"}:
        monkeypatch.setattr("hermes_wisdom.setup_execution.execute_step", pause_execution)
        if phase == "returned":
            monkeypatch.setattr("tools.terminal_tool.terminal_tool", lambda **kwargs: json.dumps({"status": "unknown"}))
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(consent.resolve, "org-1", card["id"], actor, "confirm")
        try:
            if phase != "running":
                assert entered.wait(15)
            else:
                assert future.result(timeout=15)["result"]["setup"]["state"] == "running"
            with service.store.transaction() as db:
                db.execute("UPDATE operation_lock SET expires_at='1970-01-01T00:00:00+00:00'")
            if phase == "starting":
                with pytest.raises(WisdomConflict, match="busy"):
                    consent.resolve("org-1", card["id"], actor, "setup.clear")
            elif phase == "running":
                status = consent.resolve("org-1", card["id"], actor, "setup.clear")
                assert status["result"]["setup"]["state"] == "running"
            else:
                preview = consent.resolve("org-1", card["id"], actor, "setup.recover")
                assert preview["result"]["setup"]["recovery_review"]
                cleared = consent.resolve("org-1", card["id"], actor, "setup.clear")
                assert cleared["result"]["setup"]["state"] == "abandoned"
            assert not marker.exists()
            assert bool(service.store.pending_operations()) == (phase in {"starting", "running"})
        finally:
            release.set()
            release_file.write_text("continue")
        result = future.result(timeout=15)
    if phase in {"before_start", "returned"}:
        assert result["result"]["setup"]["state"] == "abandoned"
        assert not marker.exists()
        inspected = consent.resolve("org-1", card["id"], actor, "inspect")
        assert inspected["result"]["setup"]["state"] == "abandoned"
    else:
        assert _settle(consent, actor, card["id"])["state"] == "completed"
        assert marker.read_text() == "once"


def test_execution_lock_survives_contention_and_releases_after_process_exit(setup):
    from hermes_wisdom.setup_execution import _execution_lock

    service, _, root = setup
    ready = root / "lock-ready"
    source = (
        "import sys, time; from pathlib import Path; "
        "from hermes_wisdom.store import WisdomStore; "
        "from hermes_wisdom.setup_execution import _execution_lock; "
        "store=WisdomStore(Path(sys.argv[1])); "
        "lock=_execution_lock(store, 'skill-1'); lock.__enter__(); "
        "Path(sys.argv[2]).touch(); time.sleep(30)"
    )
    child = subprocess.Popen([sys.executable, "-c", source, str(service.store.root), str(ready)])
    try:
        deadline = time.monotonic() + 15
        while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists(), "child did not acquire the execution lock"
        with pytest.raises(WisdomConflict, match="busy"):
            with _execution_lock(service.store, "skill-1"):
                pytest.fail("live child owns the execution lock")
        with _execution_lock(service.store, "another-skill"):
            pass
    finally:
        if child.poll() is None:
            child.terminate()
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=15)
    with _execution_lock(service.store, "skill-1"):
        pass


@pytest.mark.parametrize("setup", [False], indirect=True)
@pytest.mark.parametrize("copy_mode", ["agent", "fixed"])
@pytest.mark.parametrize("case", ["complete", "defer", "changed", "secret", "missing_guide", "other_owner", "expired_lease", "new_control", "address_changed", "no_command", "missing_model"])
def test_native_install_hands_off_owned_setup_without_implicit_execution(setup, monkeypatch, case, copy_mode):
    from types import SimpleNamespace
    from hermes_wisdom.delivery import DeliveryReceipt
    from hermes_wisdom.mediation import WisdomMediation
    from tests.wisdom.test_consumption import Client

    service, actor, root = setup
    service.client.identity = {"owner": "account-user"}
    service.client.display_org_id = "org-1"
    monkeypatch.setattr("hermes_wisdom.service._config", lambda: {
        "enabled": True, "disclosure_acknowledged_at": "test",
        "notifications": {"delivery_mode": copy_mode},
    })
    assert wisdom_tool.available()
    if case == "missing_guide":
        service.client.files = [row for row in service.client.files if row[0] != SETUP_PATH]
    files = service.client.files
    original_version = service.client.version

    def version_detail(skill, version):
        value = original_version(skill, version).version
        value["security_check"] = {"status": "pass"}
        return SimpleNamespace(version=value, model_dump=lambda **kwargs: {"version": {**value, "version": version}})

    monkeypatch.setattr(service.client, "version", version_detail)
    now = [time.time()]
    mediation = WisdomMediation(service, clock=lambda: now[0])
    consent = mediation.consent
    runtime = {"model": "test-session-model", "provider": "test-provider"}
    marker = root / "setup-marker"
    model_calls = []

    def model(**kwargs):
        assert kwargs["main_runtime"] == runtime and kwargs["tools"] == []
        assert kwargs["model"] == runtime["model"] and kwargs["provider"] == runtime["provider"]
        evidence = json.loads(kwargs["messages"][-1]["content"])
        assert str(root) not in kwargs["messages"][-1]["content"]
        model_calls.append(evidence)
        source = (f"from pathlib import Path; Path({str(marker)!r}).open('a').write('once')"
                  if evidence["phase"] == "setup" else
                  f"from pathlib import Path; assert Path({str(marker)!r}).read_text().endswith('once')")
        command = "echo sk-aaaaaaaaaaaaaaaaaaaaaaaa" if case == "secret" else _python(source)
        if case == "expired_lease":
            now[0] += 200
        elif case == "new_control":
            consent.request("org-1", {"kind": "setup", "skill_id": "skill-1", "version": 1,
                                      "step": {"phase": "setup", "index": 0, "command": _python("print('review only')")}},
                            actor, title="User-requested setup", explanation="Review a different command.")
        elif case == "address_changed":
            register(replace(actor, chat_id="another-private-chat"))
        elif case == "no_command":
            command = None
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            tool_calls=None, content=json.dumps({"command": command, "explanation": "Review the declared step."}),
        ))])

    monkeypatch.setattr("agent.auxiliary_client.call_llm", model)

    def register(who=actor):
        mediation.queue.register_session(
            "org-1", session_key=who.session_key, session_id=who.session_key,
            platform=who.platform, actor_id=who.actor_id, private=True, available=True,
            user_activity=True, address=who.address,
        )

    def deliver(items):
        selected = mediation.begin_delivery("org-1", items)
        assert selected == items
        assert mediation.delivery_ready("org-1", selected)
        for item in items:
            job = item["assessment"]
            assert mediation.queue.complete_delivery("org-1", job["id"], job["lease_token"], receipt=DeliveryReceipt(
                platform=actor.platform, destination=actor.chat_id, thread_id="", scope_id="",
                message_id=job["id"], acknowledgement="provider_accepted",
            ))

    def prepare():
        register()
        return mediation.prepare("org-1", actor, runtime=runtime, history=[],
                                 assessor=lambda *args, **kwargs: pytest.fail("setup must not be assessed for relevance"))

    register()
    unsolicited = None
    if copy_mode == "fixed":
        unsolicited = mediation.queue.enqueue("org-1", "feed:unrequested", {"kind": "notice"})
        assert prepare() == []
    install = consent.request("org-1", {"kind": "skill", "skill_id": "skill-1", "version": 1}, actor,
                              title="Review installation", explanation="Install this test skill.")
    initial = prepare()
    if copy_mode == "agent":
        from tests.wisdom.test_operation_outbox import reserve

        delivery, acknowledge = reserve(service, mediation.queue, initial[0]["assessment"], now, actor)
        acknowledge()
        delivery.flush("org-1")
    else:
        deliver(initial)
    assert service.store.installation("skill-1") is None
    installed = consent.resolve("org-1", install["id"], actor, "confirm")
    assert installed["state"] == "completed"
    assert interaction_view(installed).summary == "Files installed"
    assert not marker.exists() and not model_calls
    def reported_states():
        with service.store.transaction() as db:
            return {row[0] for row in db.execute(
                "SELECT report_state FROM wisdom_operation_outbox WHERE interaction_id=?", (install["id"],),
            )}

    assert reported_states() == ({"files_installed"} if copy_mode == "agent" else set())
    consent.resolve("org-1", install["id"], actor, "confirm")
    with service.store.transaction() as db:
        assert db.execute("SELECT count(*) FROM wisdom_assessment WHERE event_key=?", (f"setup-handoff:{install['id']}",)).fetchone()[0] == 1
    if case == "other_owner":
        other = replace(actor, session_key="other-session", actor_id="other")
        register(other)
        assert mediation.prepare("org-1", other, runtime=runtime, history=[]) == []
        return
    if case == "changed":
        target = Path(service.store.installation("skill-1")["target_path"])
        (target / "SKILL.md").write_text("edited since installation")
    first = prepare()
    assert len(first) == 1
    if case in {"changed", "missing_guide"}:
        assert first[0]["interaction"] is None
        assert "attention" in first[0]["advice"]["title"]
        assert not model_calls and not marker.exists()
        return
    prerequisite = first[0]["interaction"]
    assert prerequisite["facts"]["step"]["phase"] == "prerequisite"
    activity = mediation.activity()
    assert activity["mode"] == copy_mode
    assert prerequisite["id"] in {item["id"] for item in activity["interactions"]}
    assert first[0]["assessment"]["id"] in {item["id"] for item in activity["assessments"]}
    if unsolicited:
        assert unsolicited not in {item["id"] for item in activity["assessments"]}
        untouched = next(item for item in mediation.queue.assessments("org-1") if item["id"] == unsolicited)
        assert untouched["state"] == "pending" and untouched["attempts"] == 0
    deliver(first)
    if case == "defer":
        assert consent.resolve("org-1", prerequisite["id"], actor, "defer")["deferred"]
        assert prepare() == []
        assert not model_calls and not marker.exists()
        reopened = resolve_surface_action(service, f"wi:agent:setup.status:{install['id']}",
                                          platform=actor.platform, actor_id=actor.actor_id, chat_id=actor.chat_id)
        assert any(action.callback_data == f"wi:agent:confirm:{prerequisite['id']}" for action in reopened.actions)
        assert not model_calls and not marker.exists()
        now[0] += 86401
        expired = consent.resolve("org-1", install["id"], actor, "setup.status")
        assert expired["id"] == prerequisite["id"] and expired["state"] == "expired"
        assert [action.label for action in interaction_view(expired).actions] == ["Recheck"]
        return
    consent.resolve("org-1", prerequisite["id"], actor, "confirm")
    if case == "missing_model":
        runtime.clear()
        for _ in range(5):
            assert prepare() == []
            handoff = next(row for row in mediation.queue.assessments("org-1") if row["event_key"] == f"setup-handoff:{prerequisite['id']}")
            assert handoff["state"] == "pending" and handoff["attempts"] == 0
            assert handoff["last_error"] == "session_model_unavailable"
            waiting = resolve_surface_action(service, f"wi:agent:setup.status:{install['id']}",
                                             platform=actor.platform, actor_id=actor.actor_id, chat_id=actor.chat_id)
            assert waiting.summary == "Setup waiting for model"
            assert not model_calls and not marker.exists()
            now[0] += 61
        # Reopening the database with an available model resumes the same handoff.
        mediation = WisdomMediation(service, clock=lambda: now[0])
        consent = mediation.consent
        runtime.update({"model": "test-session-model", "provider": "test-provider"})
    steps = prepare()
    if case in {"secret", "expired_lease", "new_control", "address_changed"}:
        assert steps == [] and not marker.exists()
        assert len(model_calls) == 1
        with service.store.transaction() as db:
            assert "sk-aaaaaaaaaaaaaaaaaaaaaaaa" not in json.dumps([dict(row) for row in db.execute("SELECT * FROM wisdom_assessment")])
        return
    if case == "no_command":
        assert len(steps) == 1 and steps[0]["interaction"] is None
        assert steps[0]["advice"]["assessment_kind"] == "setup_status"
        assert not marker.exists()
        return
    assert len(steps) == 1 and not marker.exists()
    step = steps[0]["interaction"]
    assert step["facts"]["step"]["phase"] == "setup"
    assert "Review the declared step." in advice_view(steps).items[0].detail
    reopened = resolve_surface_action(service, f"wi:agent:inspect:{step['id']}",
                                      platform=actor.platform, actor_id=actor.actor_id, chat_id=actor.chat_id)
    assert step["facts"]["step"]["command"] in reopened.items[0].detail
    assert "Review the declared step." in reopened.items[0].detail
    deliver(steps)
    consent.resolve("org-1", step["id"], actor, "confirm")
    assert _settle(consent, actor, step["id"])["state"] == "completed"
    assert marker.read_text() == "once"
    # A fresh mediator resumes from committed state, not the prior in-memory worker.
    mediation = WisdomMediation(service, clock=lambda: now[0])
    consent = mediation.consent
    verification_items = prepare()
    verification = verification_items[0]["interaction"]
    assert verification["facts"]["step"]["phase"] == "verify"
    assert inspect_installed_setup(service.store, "skill-1")["ready_to_use"] is not True
    assert "completed" not in reported_states()
    deliver(verification_items)
    consent.resolve("org-1", verification["id"], actor, "confirm")
    assert _settle(consent, actor, verification["id"])["state"] == "completed"
    ready = prepare()
    assert ready[0]["advice"]["title"].endswith("Ready")
    assert inspect_installed_setup(service.store, "skill-1")["ready_to_use"] is True
    assert reported_states() == ({"files_installed", "completed"} if copy_mode == "agent" else set())
    verified = resolve_surface_action(service, f"wi:agent:setup.status:{install['id']}",
                                      platform=actor.platform, actor_id=actor.actor_id, chat_id=actor.chat_id)
    assert verified.summary == "Ready"
    with pytest.raises(WisdomNotFound):
        resolve_surface_action(service, f"wi:agent:setup.status:{install['id']}",
                               platform=actor.platform, actor_id="someone-else", chat_id=actor.chat_id)
    deliver(ready)
    assert prepare() == [] and len(model_calls) == 2

    update_client = Client(files, mode="MANUAL")
    update_client.identity = service.client.identity
    update_client.display_org_id = "org-1"
    update_client.version = version_detail
    update_client.skill = service.client.skill
    service._client = update_client
    update = consent.request("org-1", {"kind": "skill", "skill_id": "skill-1", "version": 2}, actor,
                             title="Review update", explanation="Update this installed test skill.")
    deliver(prepare())
    updated = consent.resolve("org-1", update["id"], actor, "confirm")
    assert updated["state"] == "completed"
    assert interaction_view(updated).summary == "Files updated"
    assert inspect_installed_setup(service.store, "skill-1", version=2)["ready_to_use"] is not True
    assert not mediation.delivery_ready("org-1", ready)
    previous = resolve_surface_action(service, f"wi:agent:setup.status:{install['id']}",
                                      platform=actor.platform, actor_id=actor.actor_id, chat_id=actor.chat_id)
    assert previous.summary == "Setup needs attention"
    next_steps = prepare()
    assert next_steps[0]["interaction"]["facts"]["version"] == 2
    assert next_steps[0]["interaction"]["facts"]["step"]["phase"] == "prerequisite"
    assert marker.read_text() == "once" and len(model_calls) == 2
