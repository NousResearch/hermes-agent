"""Credential mirroring completes before workspace policy and profile publication."""
import json
import os
import shutil
from pathlib import Path

import pytest

from hermes_cli import profiles
from hermes_cli.config import DEFAULT_CONFIG, atomic_config_write, read_user_config_raw
from tools.terminal_scope import build_profile_terminal_scope
from tui_gateway import server


@pytest.mark.parametrize("mirror", [None, True, False], ids=["omitted", "true", "false"])
@pytest.mark.parametrize("backend", ["local", "ssh", "docker"])
def test_fresh_rpc_default_mirroring_preserves_nonlocal_cwd_policy(tmp_path, monkeypatch, backend, mirror):
    assert Path(profiles.__file__).resolve().parents[1] == Path(__file__).resolve().parents[2]
    assert Path(server.__file__).resolve().parents[1] == Path(__file__).resolve().parents[2]
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
        "model": {"default": "test-model", "provider": "openrouter"},
    })
    (launch / ".env").write_text(f"TERMINAL_ENV={backend}\n", encoding="utf-8")
    original_yaml = (launch / "config.yaml").read_bytes()
    original_env = (launch / ".env").read_bytes()

    params = {"name": "mirrored-new", "no_alias": True, "no_skills": True}
    if mirror is not None:
        params["mirror_credentials"] = mirror
    response = server.handle_request({
        "id": "new-profile", "method": "profiles.create",
        "params": params,
    })

    assert "error" not in response, response
    home = launch / "profiles" / "mirrored-new"
    scope = build_profile_terminal_scope(home)
    terminal = read_user_config_raw(home / "config.yaml").get("terminal", {})
    expected_backend = "local" if mirror is False else backend
    assert scope["TERMINAL_ENV"] == expected_backend
    assert response["result"]["mirrored"]["env"] is (mirror is not False)
    assert (launch / "config.yaml").read_bytes() == original_yaml
    assert (launch / ".env").read_bytes() == original_env
    assert (terminal.get("cwd") == str(home / "workspace")) is (expected_backend == "local"), {
        "persisted_cwd": terminal.get("cwd"),
        "final_backend": scope["TERMINAL_ENV"],
        "final_terminal_cwd": scope.get("TERMINAL_CWD"),
        "mirrored": response["result"]["mirrored"],
    }


@pytest.mark.parametrize("backend", ["local", "ssh", "docker"])
def test_rpc_publishes_and_notifies_only_after_final_files(tmp_path, monkeypatch, backend):
    launch = tmp_path / ".hermes"
    launch.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(server, "_hermes_home", launch)
    atomic_config_write(launch / "config.yaml", {
        "_config_version": DEFAULT_CONFIG["_config_version"],
        "model": {"default": "test-model", "provider": "fixture-gateway"},
        "providers": {"fixture-gateway": {"api": "https://llm.invalid/v1", "api_key": "${GW_KEY}"}},
        "tts": {"provider": "elevenlabs", "elevenlabs": {"api_key": "${VOICE_KEY}"}},
    })
    (launch / ".env").write_text(
        f"TERMINAL_ENV={backend}\nVOICE_KEY=fake-voice\nGW_KEY=fake-key\n", encoding="utf-8")
    destination = launch / "profiles" / "atomic-new"
    published = []
    notified = []

    rename = os.rename

    def snapshot(home):
        return {name: (home / name).read_bytes() for name in ("config.yaml", ".env", "SOUL.md")}

    def publish(src, dst, *args, **kwargs):
        if Path(dst) == destination:
            assert not destination.exists()
            assert not any(p.name == destination.name for p in profiles._iter_named_profile_dirs())
            assert build_profile_terminal_scope(src)["TERMINAL_ENV"] == backend
            cfg = read_user_config_raw(Path(src) / "config.yaml")
            assert cfg["model"]["default"] == "chosen-model"
            assert cfg["tts"]["elevenlabs"]["api_key"] == "${VOICE_KEY}"
            assert (cfg.get("terminal", {}).get("cwd") == str(destination / "workspace")) is (backend == "local")
            assert (Path(src) / "SOUL.md").read_text() == "RPC soul"
            published.append(snapshot(Path(src)))
        return rename(src, dst, *args, **kwargs)

    def notify(name):
        assert name == "atomic-new"
        notified.append(snapshot(destination))

    monkeypatch.setattr(profiles.os, "rename", publish)
    monkeypatch.setattr(profiles, "_maybe_register_gateway_service", notify)
    monkeypatch.setattr(profiles, "_notify_multiplexer", notify)
    response = server.handle_request({
        "id": "atomic-create", "method": "profiles.create",
        "params": {"name": "atomic-new", "no_alias": True, "no_skills": True,
                   "soul": "RPC soul", "model": "chosen-model", "provider": "fixture-gateway"},
    })

    assert "error" not in response, response
    assert response["result"]["model_set"] is True
    assert response["result"]["soul_written"] is True
    assert response["result"]["mirrored"]["voice"] is True
    assert published == [snapshot(destination)]
    assert notified == published * 2


@pytest.fixture
def rpc_launch(tmp_path, monkeypatch):
    launch = tmp_path / ".hermes"
    launch.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(server, "_hermes_home", launch)
    published, notifications = [], []
    rename = os.rename

    def snapshot(home):
        return {name: (home / name).read_bytes() if (home / name).exists() else None
                for name in ("config.yaml", ".env", "auth.json", "SOUL.md")}

    def observe_publish(src, dst, *args, **kwargs):
        if Path(src).name.startswith(".") and ".staging-" in Path(src).name:
            assert not Path(dst).exists()
            published.append((Path(dst), snapshot(Path(src))))
        return rename(src, dst, *args, **kwargs)

    def observe_notify(name):
        home = launch / "profiles" / name
        notifications.append((home, snapshot(home)))

    monkeypatch.setattr(profiles.os, "rename", observe_publish)
    monkeypatch.setattr(profiles, "_maybe_register_gateway_service", observe_notify)
    monkeypatch.setattr(profiles, "_notify_multiplexer", observe_notify)
    atomic_config_write(launch / "config.yaml", {
        "_config_version": DEFAULT_CONFIG["_config_version"],
        "model": {"provider": "fixture-gateway", "default": "launch-model"},
        "providers": {"fixture-gateway": {"api": "https://llm.invalid/v1", "api_key": "${GW_KEY}"}},
        "tts": {"provider": "elevenlabs", "elevenlabs": {"api_key": "${VOICE_KEY}"}},
        "stt": {"provider": "openai", "openai": {"api_key": "${VOICE_KEY}"}},
        "voice": {"auto_tts": True},
    })
    (launch / ".env").write_text(
        "TERMINAL_ENV=ssh\nGW_KEY=fake-gateway\nVOICE_KEY=fake-voice\n"
        "TELEGRAM_BOT_TOKEN=123:fake-channel\nTELEGRAM_ALLOWED_USERS=42\n", encoding="utf-8")
    yield launch
    assert notifications == [entry for entry in published for _ in range(2)]
    assert all(snapshot(home) == files for home, files in published)


def _create_rpc(params):
    response = server.handle_request({"id": "create", "method": "profiles.create", "params": {
        "no_alias": True, **params,
    }})
    assert "error" not in response, response
    return response["result"]


_CLONE_POLICIES = [
    ({}, "local", "local"), ({}, "ssh", "ssh"), ({}, "docker", "docker"),
    ({"backend": "local", "env_type": "ssh"}, "docker", "local"),
    ({"backend": "ssh", "env_type": "local"}, "local", "ssh"),
    ({"backend": None, "env_type": "local"}, "ssh", "ssh"),
    ({"backend": None, "env_type": "ssh"}, "local", "local"),
    ({"env_type": "ssh"}, "local", "ssh"),
    ({"env_type": "local"}, "docker", "local"),
    ({"env_type": None}, "docker", "docker"),
]


@pytest.mark.parametrize("mirror", [None, True, False], ids=["omitted", "true", "false"])
@pytest.mark.parametrize("clone_all", [False, True], ids=["config", "full"])
@pytest.mark.parametrize("policy,env_backend,creation_backend", _CLONE_POLICIES)
def test_rpc_clone_keeps_destination_secrets_and_final_policy(
        rpc_launch, tmp_path, monkeypatch, mirror, clone_all, policy, env_backend, creation_backend):
    source = rpc_launch / "profiles" / "source"
    source.mkdir(parents=True)
    config_target, env_target = tmp_path / "source.yaml", tmp_path / "source.env"
    atomic_config_write(config_target, {
        "_config_version": DEFAULT_CONFIG["_config_version"],
        "model": {"provider": "fixture-gateway", "default": "source-model"},
        "terminal": {"cwd": str(source / "workspace" / ".."), "timeout": 77, **policy},
        "tts": {"provider": "source-voice"},
    })
    env_target.write_text(f"TERMINAL_ENV={env_backend}\nGW_KEY=source-secret\n", encoding="utf-8")
    (source / "config.yaml").symlink_to(config_target)
    (source / ".env").symlink_to(env_target)
    (source / "auth.json").write_text('{"source_auth": true}', encoding="utf-8")
    (rpc_launch / "auth.json").write_text('{"launch_auth": true}', encoding="utf-8")
    originals = {p: p.read_bytes() for p in (config_target, env_target, source / "auth.json",
                                            rpc_launch / "config.yaml", rpc_launch / ".env", rpc_launch / "auth.json")}
    monkeypatch.setenv("TERMINAL_ENV", "docker" if env_backend == "local" else "local")
    params = {"name": "clone-policy", "clone_from": "source", "clone_all": clone_all}
    if mirror is not None:
        params["mirror_credentials"] = mirror

    result = _create_rpc(params)

    home = Path(result["path"])
    expected = str(home / "workspace") if creation_backend == "local" else str(source / "workspace" / "..")
    cfg = read_user_config_raw(home / "config.yaml")
    assert cfg["terminal"] == {"cwd": expected, "timeout": 77, **policy}
    assert cfg["tts"] == {"provider": "source-voice"}
    assert cfg["model"]["default"] == "source-model"
    assert result["mirrored"]["env"] is False
    assert result["mirrored"]["auth"] is (not clone_all and mirror is not False)
    assert result["mirrored"]["model_inherited"] is False
    assert (home / ".env").read_bytes() == originals[env_target]
    if clone_all:
        assert (home / "auth.json").read_bytes() == originals[source / "auth.json"]
    elif mirror is not False:
        assert (home / "auth.json").read_bytes() == originals[rpc_launch / "auth.json"]
    else:
        assert not (home / "auth.json").exists()
    assert not (home / "config.yaml").is_symlink()
    assert not (home / ".env").is_symlink()
    assert (source / "config.yaml").readlink() == config_target
    assert (source / ".env").readlink() == env_target
    assert {p: p.read_bytes() for p in originals} == originals


@pytest.mark.parametrize("clone_all", [False, True], ids=["config", "full"])
@pytest.mark.parametrize("backend", ["local", "ssh", "docker"])
def test_rpc_clone_stub_inherits_model_before_final_workspace_policy(rpc_launch, clone_all, backend):
    (rpc_launch / ".env").write_text(f"TERMINAL_ENV={backend}\nGW_KEY=fake-gateway\n", encoding="utf-8")
    source = rpc_launch / "profiles" / "source"
    source.mkdir(parents=True)
    atomic_config_write(source / "config.yaml", {
        "_config_version": DEFAULT_CONFIG["_config_version"],
        "terminal": {"backend": None, "env_type": "ssh" if backend == "local" else "local", "cwd": "."},
    })
    (source / ".env").write_text("# empty clone credentials\n", encoding="utf-8")
    originals = {p: p.read_bytes() for p in (source / "config.yaml", source / ".env",
                                            rpc_launch / "config.yaml", rpc_launch / ".env")}

    result = _create_rpc({"name": "inherit-policy", "clone_from": "source", "clone_all": clone_all})

    home = Path(result["path"])
    cfg = read_user_config_raw(home / "config.yaml")
    scope = build_profile_terminal_scope(home)
    assert result["mirrored"] == {"env": True, "auth": False, "voice": True, "model_inherited": True}
    assert cfg["model"]["default"] == "launch-model"
    assert cfg["providers"]["fixture-gateway"]["api_key"] == "${GW_KEY}"
    assert cfg["tts"]["elevenlabs"]["api_key"] == "${VOICE_KEY}"
    assert scope["TERMINAL_ENV"] == backend
    assert cfg["terminal"]["backend"] is None
    assert cfg["terminal"]["cwd"] == (str(home / "workspace") if backend == "local" else ".")
    assert {p: p.read_bytes() for p in originals} == originals


@pytest.mark.parametrize("mirror", [None, True, False], ids=["omitted", "true", "false"])
@pytest.mark.parametrize("share_auth", [False, True])
def test_rpc_mirroring_preserves_credential_voice_and_oauth_contract(rpc_launch, mirror, share_auth):
    auth = {"providers": {"openai-codex": {"refresh_token": "fake-single-use", "access_token": "fake-access"}},
            "credential_pool": {"anthropic": [
                {"source": "anthropic_oauth", "refresh_token": "fake-single-use"},
                {"source": "manual", "api_key": "fake-static-key"},
            ]}}
    (rpc_launch / "auth.json").write_text(json.dumps(auth), encoding="utf-8")
    originals = {p: p.read_bytes() for p in (rpc_launch / "config.yaml", rpc_launch / ".env", rpc_launch / "auth.json")}
    params = {"name": "credentials", "no_skills": True, "share_auth": share_auth}
    if mirror is not None:
        params["mirror_credentials"] = mirror

    result = _create_rpc(params)

    home = Path(result["path"])
    enabled = mirror is not False
    assert result["mirrored"] == {"env": enabled, "auth": "shared" if share_auth else enabled,
                                  "voice": enabled, "model_inherited": False}
    env = (home / ".env").read_text()
    assert ("GW_KEY=fake-gateway" in env) is enabled
    assert "TELEGRAM_BOT_TOKEN" not in env
    assert "TELEGRAM_ALLOWED_USERS" not in env
    assert (home / "auth.json").exists() is (enabled and not share_auth)
    if enabled and not share_auth:
        copied = json.loads((home / "auth.json").read_text())
        assert "openai-codex" not in copied["providers"]
        assert copied["credential_pool"]["anthropic"] == [{"source": "manual", "api_key": "fake-static-key"}]
        assert (home / "auth.json").stat().st_mode & 0o777 == 0o600
    assert (home / ".env").stat().st_mode & 0o777 == 0o600
    cfg = read_user_config_raw(home / "config.yaml")
    assert ("tts" in cfg) is enabled
    if enabled:
        assert cfg["tts"]["elevenlabs"]["api_key"] == "${VOICE_KEY}"
        assert cfg["stt"]["openai"]["api_key"] == "${VOICE_KEY}"
        assert cfg["voice"] == {"auto_tts": True}
    assert {p: p.read_bytes() for p in originals} == originals


def test_rpc_mirror_copy_failure_is_best_effort_with_final_local_policy(rpc_launch, monkeypatch):
    copy = shutil.copy2
    original = (rpc_launch / ".env").read_bytes()
    attempts = []

    def fail_env_copy(src, dst, *args, **kwargs):
        if Path(src) == rpc_launch / ".env":
            attempts.append(Path(dst))
            raise OSError("injected mirror copy failure")
        return copy(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, "copy2", fail_env_copy)
    result = _create_rpc({"name": "mirror-failed", "no_skills": True})

    home = Path(result["path"])
    assert len(attempts) == 1
    assert attempts[0].parent.name.startswith(".mirror-failed.staging-")
    assert result["mirrored"]["env"] is False
    assert result["mirrored"]["voice"] is True
    assert build_profile_terminal_scope(home)["TERMINAL_ENV"] == "local"
    assert read_user_config_raw(home / "config.yaml")["terminal"]["cwd"] == str(home / "workspace")
    assert (rpc_launch / ".env").read_bytes() == original


def test_rpc_final_workspace_write_failure_discards_hidden_tree(rpc_launch, monkeypatch):
    from hermes_cli import config

    write = config.atomic_config_write
    notifications = []
    attempted = []
    originals = {p: p.read_bytes() for p in (rpc_launch / "config.yaml", rpc_launch / ".env")}
    (rpc_launch / ".env").write_text("GW_KEY=fake-gateway\n", encoding="utf-8")
    originals[rpc_launch / ".env"] = (rpc_launch / ".env").read_bytes()

    def fail_workspace_write(path, payload, *args, **kwargs):
        if "cwd" in (payload.get("terminal") or {}):
            attempted.append(Path(path).parent)
            raise OSError("injected final workspace write failure")
        return write(path, payload, *args, **kwargs)

    monkeypatch.setattr(config, "atomic_config_write", fail_workspace_write)
    monkeypatch.setattr(profiles, "_maybe_register_gateway_service", notifications.append)
    monkeypatch.setattr(profiles, "_notify_multiplexer", notifications.append)
    response = server.handle_request({"id": "fail", "method": "profiles.create", "params": {
        "name": "unpublished", "no_alias": True, "no_skills": True,
    }})

    assert response["error"]["code"] == 5062
    assert "injected final workspace write failure" in response["error"]["message"]
    assert attempted and all(not home.exists() for home in attempted)
    assert not (rpc_launch / "profiles" / "unpublished").exists()
    assert notifications == []
    assert {p: p.read_bytes() for p in originals} == originals


@pytest.mark.parametrize("clone_channels", [False, True])
def test_rpc_clone_stub_mirroring_respects_channel_opt_in(rpc_launch, clone_channels):
    source = rpc_launch / "profiles" / "source"
    source.mkdir(parents=True)
    atomic_config_write(source / "config.yaml", {"_config_version": DEFAULT_CONFIG["_config_version"]})
    (source / ".env").write_text("# stub\n", encoding="utf-8")
    original = (rpc_launch / ".env").read_bytes()

    result = _create_rpc({"name": "channel-copy", "clone_from": "source", "clone_channels": clone_channels})

    env = (Path(result["path"]) / ".env").read_text()
    assert result["mirrored"]["env"] is True
    assert ("TELEGRAM_BOT_TOKEN=123:fake-channel" in env) is clone_channels
    assert ("TELEGRAM_ALLOWED_USERS=42" in env) is clone_channels
    assert "GW_KEY=fake-gateway" in env
    assert (rpc_launch / ".env").read_bytes() == original


def test_rpc_model_rejection_is_best_effort_before_policy_and_publication(rpc_launch):
    originals = {p: p.read_bytes() for p in (rpc_launch / "config.yaml", rpc_launch / ".env")}

    result = _create_rpc({"name": "rejected-model", "no_skills": True,
                          "model": "chosen-model", "provider": "unconfigured-fixture-provider"})

    home = Path(result["path"])
    assert result["model_set"] is False
    assert result["mirrored"]["model_inherited"] is False
    assert result["mirrored"]["env"] is True
    assert read_user_config_raw(home / "config.yaml")["model"]["default"] == "launch-model"
    assert build_profile_terminal_scope(home)["TERMINAL_ENV"] == "ssh"
    assert read_user_config_raw(home / "config.yaml").get("terminal", {}).get("cwd") != str(home / "workspace")
    assert {p: p.read_bytes() for p in originals} == originals
