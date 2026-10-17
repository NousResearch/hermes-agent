"""Agent-facing credential tool: metadata-only reads, honest request path."""

import json

import pytest

import agent.credential_store as cs
from tools.credential_tool import CREDENTIAL_SCHEMA, credential_tool
from tools.registry import registry


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "get_hermes_home", lambda: tmp_path)
    cs.reset_store()
    yield
    cs.reset_store()


def test_registered_under_the_credential_toolset():
    assert registry.get_entry("credentials") is not None
    assert registry.get_toolset_for_tool("credentials") == "credential"
    assert "credentials" in registry.get_tool_names_for_toolset("credential")
    assert CREDENTIAL_SCHEMA["name"] == "credentials"
    assert set(CREDENTIAL_SCHEMA["parameters"]["properties"]["action"]["enum"]) == {
        "request", "list", "status", "delete"}


def test_list_and_status_never_return_values():
    cs.get_store().set("supabase_service_key", "sk-live-Value42")

    listing = json.loads(credential_tool(action="list"))
    assert listing["success"] and listing["count"] == 1
    assert listing["entries"][0]["name"] == "supabase_service_key"
    assert "sk-live-Value42" not in json.dumps(listing)

    status = json.loads(credential_tool(action="status",
                                        names=["supabase_service_key", "missing_one"]))
    assert status["configured"] == ["supabase_service_key"]
    assert status["missing"] == ["missing_one"]
    assert "sk-live-Value42" not in json.dumps(status)


def test_request_on_a_non_interactive_surface_tells_the_user_what_to_do(monkeypatch):
    monkeypatch.setattr("tools.credential_tool._interactive_terminal", lambda: False)
    result = json.loads(credential_tool(action="request", name="stripe_secret_key",
                                        description="Stripe"))
    assert result["success"] is False
    assert "/creds set stripe_secret_key" in result["error"]


def test_request_prompts_and_stores_on_an_interactive_terminal(monkeypatch):
    monkeypatch.setattr("tools.credential_tool._interactive_terminal", lambda: True)
    monkeypatch.setattr("hermes_cli.secret_prompt.masked_secret_prompt",
                        lambda prompt: "sk-live-Typed42")
    result = json.loads(credential_tool(action="request", name="stripe_secret_key"))
    assert result["success"] is True and result["stored"] is True
    assert cs.get_store().get("stripe_secret_key") == "sk-live-Typed42"


def test_request_reports_existing_unless_overwrite(monkeypatch):
    cs.get_store().set("stripe_secret_key", "original_value")
    monkeypatch.setattr("tools.credential_tool._interactive_terminal", lambda: False)
    result = json.loads(credential_tool(action="request", name="stripe_secret_key"))
    assert result["already_exists"] is True


def test_request_rejects_generic_names_before_prompting(monkeypatch):
    prompted = []
    monkeypatch.setattr("tools.credential_tool._interactive_terminal", lambda: True)
    monkeypatch.setattr("hermes_cli.secret_prompt.masked_secret_prompt",
                        lambda prompt: prompted.append(prompt) or "value1")
    result = json.loads(credential_tool(action="request", name="key"))
    assert result["success"] is False and "too generic" in result["error"]
    assert prompted == []  # never reached the prompt


def test_delete_and_missing_action_errors():
    cs.get_store().set("alpha_token_one", "value1")
    assert json.loads(credential_tool(action="delete", name="alpha_token_one"))["success"] is True
    assert json.loads(credential_tool(action="delete", name="alpha_token_one"))["success"] is False
    assert "name is required" in json.loads(credential_tool(action="request"))["error"]
    assert "names list is required" in json.loads(credential_tool(action="status"))["error"]
    assert "Unknown action" in json.loads(credential_tool(action="frobnicate"))["error"]