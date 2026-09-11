import argparse
import json
import os
import sys
from pathlib import Path

import pytest

from gateway.session_context import clear_session_vars, set_session_vars
from hermes_cli.subcommands.wisdom import build_wisdom_parser, cmd_wisdom
from hermes_wisdom.agent_led.schemas import SharePackage
from hermes_wisdom.agent_led.setup_document import SETUP_PATH, parse_setup_document
from hermes_wisdom.agent_led.share_flow import normalize_generated_package
from hermes_wisdom.installed_setup import inspect_installed_setup
from hermes_wisdom.mediation import WisdomMediation
from hermes_wisdom.package import PackagePolicyError
from hermes_wisdom.service import WisdomService
from hermes_wisdom.store import WisdomStore
from tests.wisdom.test_service import InstallClient, _install_service
from tests.wisdom.entitlement_fixtures import authorized_wisdom_token_fixture
from tools import wisdom_tool
from tools.registry import registry


@pytest.fixture
def installed(tmp_path, monkeypatch):
    authorized_wisdom_token_fixture(monkeypatch)
    client = InstallClient()
    service = _install_service(monkeypatch, tmp_path, client=client)
    monkeypatch.setattr("hermes_wisdom.consumption.get_skills_dir", lambda: tmp_path / "skills")
    sentinel = tmp_path / "must-not-execute"
    package = SharePackage.model_validate({
        "skill_name": "managed-skill",
        "source_content_hash": "sha256:source",
        "editorial_name": "Managed skill",
        "plain_description": "A setup inspection fixture.",
        "files": [{"path": "SKILL.md", "content": "# Managed\n"}],
        "requirements": [
            {"kind": "env_var", "name": "WISDOM_FIXTURE_CREDENTIAL", "purpose": "Authenticate the example service"},
            {"kind": "command", "name": sys.executable, "purpose": "Run the example"},
            {"kind": "account", "name": "example-account", "purpose": "User-managed example account"},
        ],
        "setup_instructions": ["Connect your example account."],
        "credential_handoff": ["Configure WISDOM_FIXTURE_CREDENTIAL privately."],
        "verification_step": f"touch {sentinel}",
    })
    normalized = normalize_generated_package(package)
    setup = next(file.content for file in normalized.files if file.path == SETUP_PATH)
    client.files.append((SETUP_PATH, "file", setup.encode()))
    plan = service.install_plan("skill-1")
    outcome = service.install_apply(plan["receipt"])
    assert outcome["installed"] is True
    monkeypatch.setattr("hermes_wisdom.service._config", lambda: {
        "enabled": True, "disclosure_acknowledged_at": "test", "delivery_mode": "agent",
    })
    monkeypatch.setattr("hermes_wisdom.mediation.delivery_mode", lambda *args: "agent")
    monkeypatch.setattr("hermes_cli.config.get_env_value", lambda name: None)
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: pytest.fail("inspection must not execute commands"))
    for method in ("skill", "version", "content", "record_install"):
        monkeypatch.setattr(client, method, lambda *args, **kwargs: pytest.fail("inspection must not contact Gateway"))
    return service, tmp_path, sentinel, package


def test_installed_guidance_survives_restart_without_executing_or_claiming_ready(installed, monkeypatch, capsys):
    service, root, sentinel, package = installed
    tokens = set_session_vars(platform="telegram", chat_type="dm", chat_id="chat", user_id="owner", session_key="private")
    try:
        monkeypatch.setattr("hermes_wisdom.service.WisdomService", lambda: service)
        result = json.loads(registry.dispatch("wisdom_inspect", {"kind": "installed", "identity": "skill-1", "version": 1}))
        assert result["files_installed"] is True
        assert result["ready_to_use"] is None and result["execution_authorized"] is False
        assert result["state"] == "setup_required"
        assert [item["status"] for item in result["prerequisites"]] == ["missing", "present", "manual"]
        assert result["guidance"]["verification_step"] == package.verification_step
        assert result["guidance"]["setup_instructions"] == package.setup_instructions
        inbox = WisdomMediation(service).activity()
        assert inbox["installed_setup"][0]["version"] == 1
        assert "WISDOM_FIXTURE_CREDENTIAL" not in json.dumps(inbox)

        restarted = WisdomService(store=WisdomStore(root / "state"), client=service.client)
        monkeypatch.setattr("hermes_wisdom.service.WisdomService", lambda: restarted)
        secret = "a-private-value-that-must-not-appear"
        monkeypatch.setattr("hermes_cli.config.get_env_value", lambda name: secret if name == "WISDOM_FIXTURE_CREDENTIAL" else None)
        parser = argparse.ArgumentParser()
        build_wisdom_parser(parser.add_subparsers())
        args = parser.parse_args(["wisdom", "installed-setup", "skill-1", "--version", "1", "--json"])
        assert cmd_wisdom(args) == 0
        stdout = capsys.readouterr().out
        resumed = json.loads(stdout)
        assert resumed["content_hash"] == result["content_hash"]
        assert resumed["prerequisites"][0]["status"] == "present"
        assert resumed["verification"] == {"state": "not_recorded"}
        assert secret not in stdout and not sentinel.exists()
        assert service.store.pending_operations() == []
    finally:
        clear_session_vars(tokens)


@pytest.mark.parametrize("change", ["bytes", "version", "org", "inactive", "pending", "symlink", "hardlink", "extra_directory", "group"])
def test_installed_setup_refuses_stale_or_unowned_context(installed, monkeypatch, change):
    service, root, sentinel, _package = installed
    row = service.store.installation("skill-1")
    target = Path(row["target_path"])
    if change == "bytes":
        (target / "SKILL.md").write_text("# Edited\n")
    elif change == "version":
        service.store.record_install({**row, "version": 2})
    elif change == "org":
        service.store.verify_installation_identity("another-org")
    elif change == "inactive":
        service.store.deactivate_install("skill-1")
    elif change == "pending":
        service.store.journal("install", "skill-1", "local_ledger_committed", {})
    elif change == "symlink":
        file = target / SETUP_PATH
        outside = root / "outside.md"
        file.rename(outside)
        file.symlink_to(outside)
    elif change == "extra_directory":
        (target / "refs" / "not-in-the-installed-package").mkdir()
    elif change == "hardlink":
        os.link(target / SETUP_PATH, root / "linked-setup.md")
    if change == "group":
        tokens = set_session_vars(platform="slack", chat_type="group", chat_id="group", user_id="member", session_key="shared")
        try:
            monkeypatch.setattr("hermes_wisdom.service.WisdomService", lambda: service)
            result = registry.dispatch("wisdom_inspect", {"kind": "installed", "identity": "skill-1", "version": 1})
            assert "private conversation" in result and "WISDOM_FIXTURE_CREDENTIAL" not in result
            inbox = registry.dispatch("wisdom_inbox", {})
            assert "private conversation" in inbox and "installed_setup" not in inbox
        finally:
            clear_session_vars(tokens)
    else:
        with pytest.raises(PackagePolicyError):
            inspect_installed_setup(service.store, "skill-1", version=1)
    assert not sentinel.exists()


@pytest.mark.macos_only
def test_setup_inspection_rejects_executable_mode_on_macos(installed):
    _assert_executable_setup_rejected(installed)


@pytest.mark.linux_only
def test_setup_inspection_rejects_executable_mode_on_linux(installed):
    _assert_executable_setup_rejected(installed)


def _assert_executable_setup_rejected(installed):
    service, _root, sentinel, _package = installed
    setup = Path(service.store.installation("skill-1")["target_path"]) / SETUP_PATH
    setup.chmod(setup.stat().st_mode | 0o100)
    with pytest.raises(PackagePolicyError, match="executable"):
        inspect_installed_setup(service.store, "skill-1", version=1)
    assert not sentinel.exists()


def test_setup_inspection_rejects_file_replaced_before_open(installed, monkeypatch):
    service, root, sentinel, _package = installed
    setup = Path(service.store.installation("skill-1")["target_path"]) / SETUP_PATH
    replacement = root / "replacement.md"
    replacement.write_bytes(setup.read_bytes())
    original_open = os.open

    def swapped_open(path, flags, *args, **kwargs):
        return original_open(replacement if path == setup else path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", swapped_open)
    with pytest.raises(PackagePolicyError, match="changed during inspection"):
        inspect_installed_setup(service.store, "skill-1", version=1)
    assert not sentinel.exists()


def test_setup_document_preserves_existing_wire_bytes_and_rejects_repair(installed):
    _service, _root, _sentinel, package = installed
    from hermes_wisdom.contract import canonical_json_bytes

    keys = {"requirements", "setup_instructions", "credential_handoff", "compatibility_limits", "verification_step", "removed_or_generalized", "related_skills"}
    original = (
        "# Setup and portability\n\nInstalling files does not authorize executing these instructions.\n\n```json\n"
        + canonical_json_bytes({"schema_version": 1, "execution_requires_user_approval": True, **package.model_dump(mode="json", include=keys)}).decode()
        + "\n```\n"
    ).encode()
    normalized = normalize_generated_package(package)
    encoded = next(file.content.encode() for file in normalized.files if file.path == SETUP_PATH)
    assert encoded == original
    assert parse_setup_document(original).verification_step == package.verification_step
    for bad in (original + b"execute extra", original.replace(b'"schema_version":1', b'"schema_version":2'), original.replace(b'"execution_requires_user_approval":true', b'"execution_requires_user_approval":false')):
        with pytest.raises(ValueError):
            parse_setup_document(bad)
    with pytest.raises(ValueError):
        wisdom_tool.Presentation.model_validate({"kind": "installed", "identity": "skill-1", "title": "Run", "explanation": "Not an installation consent"})


@pytest.mark.parametrize("guidance", ["valid", "missing", "invalid"])
def test_real_update_replaces_setup_reference_without_inventing_verification(installed, guidance):
    from hermes_wisdom.consumption import WisdomConsumption
    from tests.wisdom.test_consumption import Client

    service, _root, sentinel, package = installed
    before = inspect_installed_setup(service.store, "skill-1", version=1)
    target = Path(service.store.installation("skill-1")["target_path"])
    revised = package.model_copy(update={
        "requirements": [], "setup_instructions": [], "credential_handoff": [],
        "verification_step": "Review the second version with the user.",
    })
    setup = next(file.content.encode() for file in normalize_generated_package(revised).files if file.path == SETUP_PATH)
    if guidance == "invalid":
        setup = setup.replace(b'"execution_requires_user_approval":true', b'"execution_requires_user_approval":false')
    files = [
        ("SKILL.md", "file", b"# Second version\n"),
        ("skill.manifest.json", "file", (target / "skill.manifest.json").read_bytes()),
    ]
    if guidance != "missing":
        files.append((SETUP_PATH, "file", setup))
    manager = WisdomConsumption(store=service.store, client=Client(files, mode="MANUAL"), scan=lambda path: {
        "guard": {"allowed": True, "findings": [], "reason": None},
        "skill_evaluator": {"status": "disabled", "findings": []},
    }, config={})
    plan = manager.update_plan("skill-1")
    manager.update_apply(plan["receipt"])
    with pytest.raises(PackagePolicyError, match="version changed"):
        inspect_installed_setup(service.store, "skill-1", version=1)
    after = inspect_installed_setup(service.store, "skill-1", version=2)
    assert after["content_hash"] != before["content_hash"]
    assert after["ready_to_use"] is None and after["execution_authorized"] is False
    assert after["verification"] == {"state": "not_recorded"}
    if guidance == "valid":
        assert after["state"] == "verification_required"
        assert after["guidance"]["verification_step"] == revised.verification_step
    else:
        assert after["state"] == "guidance_unavailable" and after["guidance"] is None
    assert not sentinel.exists()
