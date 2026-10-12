"""Desktop's real profiles.create RPC shares the saved workspace creation policy."""

from pathlib import Path

import pytest

from hermes_cli import profiles
from hermes_cli.config import DEFAULT_CONFIG, atomic_config_write, read_user_config_raw
from tui_gateway import server


@pytest.fixture
def launch_home(tmp_path, monkeypatch):
    launch = tmp_path / ".hermes"
    launch.mkdir()
    atomic_config_write(launch / "config.yaml", {
        "_config_version": DEFAULT_CONFIG["_config_version"],
        "model": {"default": "test-model", "provider": "openrouter"},
    })
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(server, "_hermes_home", launch)
    monkeypatch.setattr(profiles, "_maybe_register_gateway_service", lambda _name: None)
    monkeypatch.setattr(profiles, "_notify_multiplexer", lambda _name: None)
    return launch


def test_desktop_new_profile_rpc_persists_destination_workspace(launch_home):
    response = server.handle_request({
        "id": "new-profile", "method": "profiles.create",
        "params": {"name": "desktop-new", "no_alias": True, "no_skills": True, "mirror_credentials": False},
    })

    assert "error" not in response, response
    home = launch_home / "profiles" / "desktop-new"
    assert (home / "workspace").is_dir()
    assert read_user_config_raw(home / "config.yaml")["terminal"]["cwd"] == str(home / "workspace")


@pytest.mark.parametrize("clone_all", [False, True], ids=["config-clone", "full-clone"])
@pytest.mark.parametrize("backend", ["local", "ssh", "docker"])
@pytest.mark.parametrize("canonical_null", [False, True], ids=["env-only", "canonical-null"])
def test_desktop_clone_rpc_uses_destination_backend(launch_home, monkeypatch, clone_all, backend, canonical_null):
    config = read_user_config_raw(launch_home / "config.yaml")
    config["terminal"] = {"cwd": ".", "timeout": 77}
    if canonical_null:
        config["terminal"].update(backend=None, env_type="ssh" if backend == "local" else "local")
    atomic_config_write(launch_home / "config.yaml", config)
    env_path = launch_home / ".env"
    env_path.write_text(f"TERMINAL_ENV={backend}\n", encoding="utf-8")
    original_yaml = (launch_home / "config.yaml").read_bytes()
    original_env = env_path.read_bytes()
    monkeypatch.setenv("TERMINAL_ENV", "local" if backend != "local" else "ssh")

    response = server.handle_request({
        "id": "clone-profile", "method": "profiles.create",
        "params": {"name": "desktop-clone", "clone_from": "default", "clone_all": clone_all,
                   "no_alias": True, "mirror_credentials": False},
    })

    assert "error" not in response, response
    home = launch_home / "profiles" / "desktop-clone"
    expected_cwd = str(home / "workspace") if backend == "local" else "."
    assert read_user_config_raw(home / "config.yaml")["terminal"] == {**config["terminal"], "cwd": expected_cwd}
    assert (home / ".env").read_bytes() == original_env
    assert (launch_home / "config.yaml").read_bytes() == original_yaml
    assert env_path.read_bytes() == original_env
