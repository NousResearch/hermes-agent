"""Behaviour tests for the ``dashboard.auth`` config setter (#146266).

The setter is the server-side half of Desktop Settings → Gateways → local
dashboard sign-in: it accepts a username + plaintext password over the
already-authenticated local control channel, hashes the password server-side,
and writes only the hash to the profile's config. The pure plan is tested
with a stubbed hash; the handler tests go through the bound gateway server
(the split module's handlers are rebound onto the server's globals, so
patching the split module directly would not reach the handler that runs).
They assert outcomes — what lands in config, what is rejected — never
source text.
"""

import sys
import types

import pytest
import hermes_yaml as yaml

import tui_gateway.methods_config_set as cfg_set
from tui_gateway import server


@pytest.fixture
def fake_hash(monkeypatch):
    """Stand-in for plugins.dashboard_auth.basic.hash_password, recording its input."""
    seen = []

    def fake(password):
        seen.append(password)
        return "scrypt$16384$8$1$c2FsdA==$aGFzaA=="

    module = types.ModuleType("plugins.dashboard_auth.basic")
    module.hash_password = fake
    monkeypatch.setitem(sys.modules, "plugins.dashboard_auth.basic", module)
    return seen


def test_plan_hashes_and_drops_plaintext(fake_hash):
    plan = cfg_set.dashboard_auth_plan(
        section={"username": "old", "password": "old-plain", "password_hash": "old-hash", "secret": "keepme"},
        username="  ben  ", password="correct horse", env={})
    assert plan["username"] == "ben"
    assert plan["sets"]["dashboard.basic_auth.username"] == "ben"
    assert plan["sets"]["dashboard.basic_auth.password_hash"] == "scrypt$16384$8$1$c2FsdA==$aGFzaA=="
    assert "dashboard.basic_auth.secret" not in plan["sets"]  # existing secret preserved
    assert plan["unsets"] == ("dashboard.basic_auth.password",)
    assert "correct horse" not in str(plan["sets"].values())
    assert fake_hash == ["correct horse"]


def test_plan_generates_secret_only_when_none_configured(fake_hash):
    plan = cfg_set.dashboard_auth_plan(section={}, username="ben", password="correct horse", env={})
    assert plan["secret_generated"] is True
    assert len(plan["sets"]["dashboard.basic_auth.secret"]) == 64  # token_hex(32)
    plan_env_secret = cfg_set.dashboard_auth_plan(
        section={}, username="ben", password="correct horse",
        env={"HERMES_DASHBOARD_BASIC_AUTH_SECRET": "env-secret"})
    assert plan_env_secret["secret_generated"] is False
    assert "dashboard.basic_auth.secret" not in plan_env_secret["sets"]


@pytest.mark.parametrize("username,password", [
    ("", "correct horse"),
    ("   ", "correct horse"),
    ("ben", ""),
    ("ben", "short"),
])
def test_plan_rejects_invalid_input(username, password, fake_hash):
    with pytest.raises(ValueError):
        cfg_set.dashboard_auth_plan(section={}, username=username, password=password, env={})
    assert fake_hash == []  # nothing hashed on a rejected input


def test_plan_refuses_env_override(fake_hash):
    with pytest.raises(ValueError, match="environment variables"):
        cfg_set.dashboard_auth_plan(
            section={}, username="ben", password="correct horse",
            env={"HERMES_DASHBOARD_BASIC_AUTH_PASSWORD": "env-password"})
    assert fake_hash == []


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    """Point the server's config read/write at a temp file."""
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    server._cfg_cache = server._cfg_sig = server._cfg_path = None
    for key in ("HERMES_DASHBOARD_BASIC_AUTH_USERNAME", "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD",
                "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH", "HERMES_DASHBOARD_BASIC_AUTH_SECRET"):
        monkeypatch.delenv(key, raising=False)
    yield tmp_path / "config.yaml"
    server._cfg_cache = server._cfg_sig = server._cfg_path = None


def _set_dashboard_auth(username, password):
    return server._methods["config.set"](1, {
        "key": "dashboard.auth", "value": "", "username": username, "password": password})


def _get_dashboard_auth():
    return server._methods["config.get"](1, {"key": "dashboard.auth"})


def test_handler_writes_hash_and_removes_plaintext(config_home):
    config_home.write_text(yaml.safe_dump({
        "dashboard": {"basic_auth": {
            "username": "old", "password": "old-plain", "password_hash": "old-hash"}}}))

    out = _set_dashboard_auth("ben", "correct horse")

    assert out["result"]["value"] == "ben"
    assert out["result"]["auth_configured"] is True
    assert out["result"]["restart_required"] is True
    assert "correct horse" not in str(out)
    basic = yaml.safe_load(config_home.read_text())["dashboard"]["basic_auth"]
    assert basic["username"] == "ben"
    assert basic["password_hash"].startswith("scrypt$")  # real scrypt hash, never plaintext
    assert "password" not in basic  # the old plaintext key is gone, not just emptied
    assert len(basic["secret"]) == 64  # generated, since the old section had none

    # The read side reports only the configured username, never a password or hash.
    assert _get_dashboard_auth()["result"] == {"value": "ben"}


def test_handler_rejects_bad_input_without_writing(config_home):
    out = _set_dashboard_auth("", "correct horse")
    assert out["error"]["code"] == 4002
    assert not config_home.exists()
