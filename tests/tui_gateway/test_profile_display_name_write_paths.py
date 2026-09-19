# SPDX-License-Identifier: MIT
"""Named-profile display_name write paths: RPC, CLI and REST plumbing over
``set_profile_display_name``. Regression for #113941."""

from pathlib import Path

import pytest


@pytest.fixture
def profile_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    named = home / "profiles" / "demo"
    named.mkdir(parents=True)
    (named / "config.yaml").write_text("model: gpt\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_configure_sets_named_profile_display_name(profile_env):
    import tui_gateway.server as srv

    result = srv._methods["profiles.configure"](
        "configure", {"name": "demo", "display_name": "My Demo"})
    assert result["result"]["applied"]["display_name"] is True

    from hermes_cli.profiles import read_profile_meta
    meta = read_profile_meta(profile_env / "profiles" / "demo")
    assert meta.get("display_name") == "My Demo"


def test_configure_clears_display_name_with_empty_string(profile_env):
    import tui_gateway.server as srv

    srv._methods["profiles.configure"](
        "configure", {"name": "demo", "display_name": "First"})
    result = srv._methods["profiles.configure"](
        "configure", {"name": "demo", "display_name": ""})
    assert result["result"]["applied"]["display_name"] is True

    from hermes_cli.profiles import read_profile_meta
    meta = read_profile_meta(profile_env / "profiles" / "demo")
    assert not meta.get("display_name")


def test_cli_display_name_flag_sets_meta_without_rename(profile_env, capsys):
    from hermes_cli.profile_cmd import _profile_rename

    class Args:
        old_name = "demo"
        new_name = None
        display_name = "Renamed Demo"

    _profile_rename(Args())

    from hermes_cli.profiles import read_profile_meta
    meta = read_profile_meta(profile_env / "profiles" / "demo")
    assert meta.get("display_name") == "Renamed Demo"
    # Directory was NOT renamed.
    assert (profile_env / "profiles" / "demo").is_dir()
    assert not (profile_env / "profiles" / "renamed-demo").exists()
    assert "Display name set" in capsys.readouterr().out


def test_rest_patch_display_name_only(profile_env):
    from fastapi.testclient import TestClient

    from hermes_cli import web_server

    with TestClient(web_server.app, raise_server_exceptions=False) as client:
        client.headers["Authorization"] = f"Bearer {web_server._SESSION_TOKEN}"
        resp = client.patch(
            "/api/profiles/demo",
            json={"display_name": "REST Name"},
        )
    assert resp.status_code == 200, resp.text
    assert resp.json()["name"] == "demo"
    assert resp.json()["display_name"] == "REST Name"

    from hermes_cli.profiles import read_profile_meta
    meta = read_profile_meta(profile_env / "profiles" / "demo")
    assert meta.get("display_name") == "REST Name"
    # Directory was NOT renamed.
    assert (profile_env / "profiles" / "demo").is_dir()
