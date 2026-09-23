"""Browser vault opt-in is profile-local, not inferred from stored items or managers."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml


@pytest.fixture
def browser_available(monkeypatch):
    # Only external installation probes are faked; config, discovery, rewrites and dispatch are real.
    import model_tools
    from tools import browser_tool_install, browser_use_cli
    from tools.registry import invalidate_check_fn_cache

    monkeypatch.setattr(browser_tool_install, "_find_agent_browser", lambda **kw: ["agent-browser"])
    monkeypatch.setattr(browser_tool_install, "_chromium_installed", lambda: True)
    monkeypatch.setattr(browser_use_cli, "_find_cli", lambda: ["browser-use"])
    invalidate_check_fn_cache()
    model_tools._clear_tool_defs_cache()


def _schemas():
    from model_tools import get_tool_definitions
    return {t["function"]["name"]: t["function"] for t in get_tool_definitions(
        enabled_toolsets=["browser", "browser-use", "terminal"],
        quiet_mode=True, skip_tool_search_assembly=True,
    )}


@pytest.mark.parametrize("stack,input_tool", [("off", "browser_type"), ("browser-use", "browser_exec")])
@pytest.mark.parametrize("vault_config,enabled", [
    (None, False), ({}, False), ({"enabled": False}, False),
    ({"enabled": "true"}, False), ({"enabled": 1}, False),
    ({"enabled": True}, True),
])
@pytest.mark.parametrize("boundary", ["schemas", "dispatch"])
def test_only_explicit_boolean_opt_in_exposes_vault(
    tmp_path, monkeypatch, browser_available, stack, input_tool, vault_config, enabled, boundary,
):
    from agent.vault_backends import base
    from agent.vault_store import get_vault_store
    from hermes_constants import get_hermes_home
    from tools import browser_vault_tool, browser_use_cli
    from tools.registry import registry

    home = get_hermes_home()
    config = {"browser": {"backend": stack}}
    if vault_config is not None:
        config["vault"] = vault_config
    (home / "config.yaml").write_text(yaml.safe_dump(config))
    # Pre-existing data and detected managers must not opt the user in.
    item = get_vault_store().add_item("login", "Test", {
        "identifier": "test@example.test", "identifier_type": "email", "password": "synthetic-test-only",
    }, origin="https://example.test")
    monkeypatch.setattr(base, "is_installed", lambda name: True)
    backend_probe = Mock(side_effect=AssertionError("must not probe managers while disabled"))
    monkeypatch.setattr(base, "external_backend_classes", backend_probe if not enabled else lambda: ())
    names = {e.name for e in registry.get_all_entries() if e.name.startswith("browser_vault_")}

    if boundary == "schemas":
        schemas = _schemas()
        assert input_tool in schemas  # ordinary browser/custom login workflows remain available
        assert (names & schemas.keys()) == (names if enabled else set())
        description = schemas[input_tool]["description"]
        assert ("Never type a password" in description) is enabled
        if not enabled:
            assert "vault" not in json.dumps(schemas).lower()
            assert "never ask for or accept" not in description.lower()
        return

    if enabled:
        listed = json.loads(registry.dispatch("browser_vault_list", {}))
        assert listed["success"] is True
        assert listed["items"][0]["handle"] == item.id
        assert "synthetic-test-only" not in json.dumps(listed)
        return

    # Registry.dispatch does not enforce check_fn. Stale schemas and direct calls still must refuse.
    calls = {
        "browser_vault_list": ({}, ()),
        "browser_vault_fill": ({"handle": item.id}, (item.id,)),
        "browser_vault_unlock": ({"backend": "bitwarden"}, ("bitwarden",)),
        "browser_vault_save_login": ({}, ()),
        "browser_vault_enter_code": ({}, ()),
    }
    assert names == calls.keys()
    page_probe = Mock(side_effect=AssertionError("must not touch a page while disabled"))
    monkeypatch.setattr(browser_vault_tool, "_current_page_origin", page_probe)
    monkeypatch.setattr(browser_vault_tool, "_ensure_supervisor", page_probe)
    for name, (args, positional) in calls.items():
        for result in (registry.dispatch(name, args), getattr(browser_vault_tool, name)(*positional)):
            result = json.loads(result)
            assert result["success"] is False
            assert result["error_type"] == "vault_disabled"
    backend_probe.assert_not_called()
    page_probe.assert_not_called()

    from tools.browser_supervisor import SUPERVISOR_REGISTRY
    attach = Mock(side_effect=AssertionError("vault supervisor must stay off"))
    monkeypatch.setattr(SUPERVISOR_REGISTRY, "get_or_start", attach)
    browser_use_cli._attach_vault_supervisor({"BU_CDP_URL": "http://localhost:9222"}, "test")
    attach.assert_not_called()


@pytest.mark.parametrize("stack,input_tool", [("off", "browser_type"), ("browser-use", "browser_exec")])
def test_profile_opt_in_setup_and_new_session_schemas(
    tmp_path, monkeypatch, browser_available, stack, input_tool,
):
    from agent import secret_scope
    from agent.vault_backends import base
    from agent.vault_store import get_vault_store
    from hermes_cli.config import _validate_config_key, load_config_readonly, set_config_value
    from hermes_cli.vault import vault_command
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.registry import registry

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(base, "is_installed", lambda name: False)
    a, b = tmp_path / "a", tmp_path / "b"
    for home in (a, b):
        home.mkdir()
        (home / "config.yaml").write_text(yaml.safe_dump({"browser": {"backend": stack}}))
    monkeypatch.setenv("HERMES_HOME", str(a))  # the process env stays A while the activity changes
    secret_scope.set_multiplex_active(True)
    scope = secret_scope.set_secret_scope({})
    snapshots = []
    try:
        for home in (a, b, a):
            token = set_hermes_home_override(home)
            try:
                if home == a and not snapshots:
                    assert load_config_readonly()["vault"]["enabled"] is False
                    old_schemas = _schemas()
                    old_snapshot = json.dumps(old_schemas, sort_keys=True)
                    assert _validate_config_key("vault.enabled") == (True, None)
                    set_config_value("vault.enabled", "true")
                expected = home == a
                assert load_config_readonly()["vault"]["enabled"] is expected
                current = _schemas()
                snapshots.append(current)
                assert ("browser_vault_fill" in current) is expected
                assert ("Never type a password" in current[input_tool]["description"]) is expected
                result = json.loads(registry.dispatch("browser_vault_list", {}))
                assert result["success"] is expected
                # Setup/management remains usable even in B; it must not enable browser tools.
                vault_command(SimpleNamespace(vault_action="list", _vault_handler=None))
                get_vault_store().add_item("address", "Setup", {
                    "address_line1": "Test street", "city": "Test city", "postal_code": "12345", "country": "US",
                })
                assert load_config_readonly()["vault"]["enabled"] is expected
            finally:
                reset_hermes_home_override(token)
        assert snapshots[0] == snapshots[2]
        assert json.dumps(old_schemas, sort_keys=True) == old_snapshot  # no cached prompt mutation
        token = set_hermes_home_override(a)
        try:
            enabled_snapshot = json.dumps(snapshots[0], sort_keys=True)
            set_config_value("vault.enabled", "false")
            assert "browser_vault_fill" not in _schemas()  # no cached last-good/grace opt-in
            assert json.loads(registry.dispatch("browser_vault_list", {}))["error_type"] == "vault_disabled"
            assert json.dumps(snapshots[0], sort_keys=True) == enabled_snapshot
        finally:
            reset_hermes_home_override(token)
    finally:
        secret_scope.reset_secret_scope(scope)
        secret_scope.set_multiplex_active(False)


@pytest.mark.parametrize("stack,input_tool", [("off", "browser_type"), ("browser-use", "browser_exec")])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("rebuild", ["resume", "refresh"])
def test_saved_vault_tools_respect_opt_out_on_session_rebuild(
    monkeypatch, browser_available, stack, input_tool, enabled, rebuild,
):
    from hermes_constants import get_hermes_home
    from hermes_cli.config import set_config_value
    from tools.mcp_tool_agent import agent_tool_names, restore_agent_tool_prefix, refresh_agent_mcp_tools

    (get_hermes_home() / "config.yaml").write_text(yaml.safe_dump({
        "browser": {"backend": stack}, "vault": {"enabled": True},
    }))
    before = [{"type": "function", "function": schema} for schema in _schemas().values()]
    saved_names = [t["function"]["name"] for t in before]
    original = json.dumps(before, sort_keys=True)
    set_config_value("vault.enabled", "true" if enabled else "false")
    fresh = [{"type": "function", "function": schema} for schema in _schemas().values()]
    persisted = []
    agent = SimpleNamespace(
        tools=fresh if rebuild == "resume" else before,
        valid_tool_names={t["function"]["name"] for t in (fresh if rebuild == "resume" else before)},
        enabled_toolsets=["browser", "browser-use", "terminal"], disabled_toolsets=None,
        session_id="resumed-session",
        _session_db=SimpleNamespace(update_session_tool_names=lambda sid, names: persisted.append(names)),
    )
    if rebuild == "resume":
        restore_agent_tool_prefix(agent, saved_names)
    else:
        refresh_agent_mcp_tools(agent, preserve_prefix=True)
    names = agent_tool_names(agent)
    assert input_tool in names
    assert any(n.startswith("browser_vault_") for n in names) is enabled
    assert ("Passwords are typed ONLY" in json.dumps(agent.tools)) is enabled
    assert ("Never type a password" in json.dumps(agent.tools)) is enabled
    assert json.dumps(before, sort_keys=True) == original
    if not enabled:
        assert persisted and persisted[-1] == names  # don't resurrect the old persisted pin again


@pytest.mark.parametrize("input_tool", ["browser_type", "browser_exec"])
def test_opt_out_removes_vault_note_when_browser_probe_flaps(browser_available, input_tool):
    from model_tools import _VAULT_NO_PASSWORD_NOTE
    from hermes_constants import get_hermes_home
    from tools.mcp_tool_agent import _merge_preserving_prefix

    (get_hermes_home() / "config.yaml").write_text("vault:\n  enabled: false\n")
    current = [{"type": "function", "function": {
        "name": input_tool, "description": "Ordinary browser input." + _VAULT_NO_PASSWORD_NOTE,
        "parameters": {"type": "object", "properties": {}},
    }}]
    original = json.dumps(current, sort_keys=True)
    merged, names = _merge_preserving_prefix(current, [], {input_tool})
    assert names == {input_tool}  # retain an ordinary tool on a transient availability failure
    assert merged[0]["function"]["description"] == "Ordinary browser input."
    assert json.dumps(current, sort_keys=True) == original
