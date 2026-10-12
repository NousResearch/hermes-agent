"""Creation model validation hydrates the staged profile's external credentials."""
import os
from pathlib import Path

from hermes_cli import profiles
from hermes_cli.config import DEFAULT_CONFIG, atomic_config_write, read_user_config_raw
from tui_gateway import server


def test_creation_pin_external_source_matches_published_pin(tmp_path, monkeypatch):
    launch = tmp_path / ".hermes"
    launch.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(server, "_hermes_home", launch)
    monkeypatch.setattr(profiles, "_maybe_register_gateway_service", lambda _name: None)
    monkeypatch.setattr(profiles, "_notify_multiplexer", lambda _name: None)
    atomic_config_write(launch / "config.yaml", {
        "_config_version": DEFAULT_CONFIG["_config_version"],
    })
    source = launch / "profiles" / "source"
    source.mkdir(parents=True)
    provider, model = "anthropic", "claude-sonnet-4-6"
    atomic_config_write(source / "config.yaml", {
        "_config_version": DEFAULT_CONFIG["_config_version"],
        "model": {"provider": provider, "default": model},
        "secrets": {"command": {
            "enabled": True,
            "command": "printf 'ANTHROPIC_API_KEY=fake-independent-provider-key\\n'",
        }},
    })
    (source / ".env").write_text("# no stored provider key\n", encoding="utf-8")
    source_bytes = (source / "config.yaml").read_bytes()
    before_env = dict(os.environ)
    published = profiles.create_profile(
        "published", clone_from="source", clone_config=True, no_alias=True)
    server._pin_profile_model(published, provider, model)
    assert read_user_config_raw(published / "config.yaml")["model"]["provider"] == provider

    response = server.handle_request({"id": "new", "method": "profiles.create", "params": {
        "name": "hidden-created", "clone_from": "source", "no_alias": True,
        "mirror_credentials": False, "provider": provider, "model": model,
    }})

    assert "error" not in response, response
    assert dict(os.environ) == before_env
    assert (source / "config.yaml").read_bytes() == source_bytes
    assert response["result"]["model_set"] is True, response
    created = Path(response["result"]["path"])
    cfg = read_user_config_raw(created / "config.yaml")
    assert cfg["model"]["provider"] == provider
    assert cfg["model"]["default"] == model
    assert "fake-independent-provider-key" not in (created / ".env").read_text()
