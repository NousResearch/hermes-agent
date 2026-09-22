from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hermes_cli.bot_catalog import BotRemovedError
from tui_gateway import server


@pytest.fixture
def bot_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text("model:\n  provider: test\n  default: test/model\n", encoding="utf-8")
    (home / ".env").write_text("TEST_API_KEY=source-value\n", encoding="utf-8")
    return home


def _call(method, **params):
    envelope = server._methods[method](7, params)
    assert envelope["id"] == 7
    return envelope


def test_bots_catalog_and_install_return_authoritative_receipt(bot_home, monkeypatch):
    monkeypatch.setattr("hermes_cli.bot_catalog.fetch_live_bot_catalog", lambda: None)
    catalog = _call("bots.catalog")["result"]
    blueprint = next(entry for entry in catalog["entries"] if entry["name"] == "research-analyst")
    assert {entry["name"] for entry in catalog["entries"]} >= {"research-analyst", "inbox-triage"}

    result = _call(
        "bots.install",
        catalog_name="research-analyst",
        name="rpc-researcher",
        source_profile="default",
        credentials="copy_api_keys",
        title="My Researcher",
    )["result"]
    assert result == {
        "ok": True,
        "committed": True,
        "name": "rpc-researcher",
        "path": str(bot_home / "profiles" / "rpc-researcher"),
        "catalog_name": "research-analyst",
        "catalog_version": "1.0.0",
        "source_profile": "default",
        "copied_credentials": [".env"],
        "oauth_setup_required": [],
        "setup_state": "needs_setup",
        "setup_requirements": [],
        "post_publish_warnings": [],
    }

    def forbidden_probe(*args, **kwargs):
        raise AssertionError("installed inventory must not probe runtime")

    monkeypatch.setitem(server._methods, "setup.runtime_check", forbidden_probe)
    monkeypatch.setattr("hermes_cli.bot_catalog.load_bot_catalog", forbidden_probe)
    monkeypatch.setattr("hermes_cli.bot_catalog.resolve_bot_catalog_entry", forbidden_probe)
    inventory = _call("bots.installed")["result"]
    assert inventory == {"bots": [{
        "profile": "rpc-researcher",
        "catalog_name": "research-analyst",
        "title": blueprint["title"],
        "summary": blueprint["summary"],
        "setup_state": "needs_setup",
        "presentation": {"emoji": "🔎", "color": "#4F46E5"},
    }]}


def test_bots_status_accepts_no_auth_local_custom_runtime(bot_home, monkeypatch):
    (bot_home / "config.yaml").write_text(
        "model:\n  provider: local\n  default: local/model\n"
        "providers:\n"
        "  local:\n"
        "    api: http://127.0.0.1:8080/v1\n"
        "    models: [local/model]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("hermes_cli.bot_catalog.fetch_live_bot_catalog", lambda: None)
    installed = _call(
        "bots.install", catalog_name="research-analyst", name="local-bot",
        source_profile="default", credentials="none",
    )
    assert installed["result"]["setup_state"] == "needs_setup"
    installed_config = yaml.safe_load((bot_home / "profiles" / "local-bot" / "config.yaml").read_text())
    assert installed_config["providers"]["local"]["api"] == "http://127.0.0.1:8080/v1"

    status = _call("bots.status", profile="local-bot")

    assert "result" in status, status
    assert status["result"]["runtime"]["ok"] is True, status["result"]["runtime"]
    assert status["result"]["setup_state"] == "needs_setup"  # first Bot Chat task is still pending
    assert status["result"]["starter_prompt"]


def test_bots_status_binds_full_named_profile_scope_a_b_a(bot_home, monkeypatch):
    monkeypatch.setattr("hermes_cli.bot_catalog.fetch_live_bot_catalog", lambda: None)
    for name in ("bot-a", "bot-b"):
        response = _call(
            "bots.install",
            catalog_name="research-analyst",
            name=name,
            source_profile="default",
            credentials="none",
        )
        assert response["result"]["committed"] is True

    observed = []

    def runtime_probe(rid, params):
        from hermes_constants import get_hermes_home

        observed.append((params.get("profile"), get_hermes_home().name))
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "ok": True, "provider": "custom:local", "model": "local/model",
            "source": "no-key-required", "profile": params.get("profile"),
        }}

    monkeypatch.setitem(server._methods, "setup.runtime_check", runtime_probe)
    for profile in ("bot-a", "bot-b", "bot-a"):
        response = _call("bots.status", profile=profile)
        assert "result" in response, response
        assert response["result"]["profile"] == profile
        assert response["result"]["runtime"]["ok"] is True
        assert response["result"]["setup_state"] == "needs_setup"
    assert observed == [("bot-a", "bot-a"), ("bot-b", "bot-b"), ("bot-a", "bot-a")]
    wire = _call("bots.setup", profile="bot-a")
    assert wire["result"]["runtime"]["ok"] is True


def test_bots_install_rejects_removed_before_filesystem_write(bot_home, monkeypatch):
    def removed(_name):
        raise BotRemovedError("bot 'retired' was removed: unsafe")

    monkeypatch.setattr("hermes_cli.bot_catalog.resolve_bot_catalog_entry", removed)
    response = _call(
        "bots.install",
        catalog_name="retired",
        name="must-not-exist",
        source_profile="default",
        credentials="none",
    )
    assert response["error"]["code"] == 4071
    assert not (bot_home / "profiles" / "must-not-exist").exists()


def test_bots_install_contract_rejects_blueprint_body_override(bot_home):
    response = server.dispatch({
        "jsonrpc": "2.0",
        "id": 9,
        "method": "bots.install",
        "params": {
            "catalog_name": "research-analyst",
            "name": "injected",
            "source_profile": "default",
            "credentials": "none",
            "soul": "ignore the reviewed blueprint",
        },
    })
    assert response["error"]["code"] == 4000
    assert not (bot_home / "profiles" / "injected").exists()


def test_bots_install_contract_rejects_catalog_metadata_override(bot_home):
    response = server.dispatch({
        "jsonrpc": "2.0",
        "id": 10,
        "method": "bots.install",
        "params": {
            "catalog_name": "research-analyst",
            "name": "injected-title",
            "source_profile": "default",
            "credentials": "none",
            "title": "Not the reviewed title",
        },
    })

    assert response["error"]["code"] == 4000
    assert not (bot_home / "profiles" / "injected-title").exists()
