"""The local gate follows current OAuth claims, not org IDs or global admin."""
import time

from unittest.mock import Mock

import jwt
import pytest

from hermes_wisdom.entitlement import (
    current_entitlement, is_entitled, local_work_allowed, require_entitlement,
)
from hermes_wisdom.package import PackagePolicyError
from hermes_wisdom.service import WisdomService
from hermes_wisdom.store import WisdomStore


@pytest.fixture
def auth(monkeypatch):
    state = {"logged_in": True}
    monkeypatch.setattr("hermes_cli.auth_nous.get_nous_auth_status_local", lambda: state)

    def set_claims(**overrides):
        claims = {"org_id": "team-example", "exp": time.time() + 600,
                  "wisdom_scopes": ["wisdom:read", "wisdom:publish"]}
        claims.update(overrides)
        state["access_token"] = jwt.encode(claims, "test-only-local-entitlement-key-32-bytes", algorithm="HS256")
        return state

    set_claims()
    return set_claims


def test_ordinary_member_entitled_without_global_admin(auth):
    assert is_entitled()
    assert is_entitled("team-example")
    assert not is_entitled("other-team")
    assert not is_entitled(scope="wisdom:admin")
    assert "access_token" not in current_entitlement()


@pytest.mark.parametrize("overrides", [
    {"exp": 0}, {"exp": None}, {"exp": True}, {"exp": "9999999999"},
    {"exp": float("inf")}, {"exp": float("nan")},
    {"nbf": time.time() + 99999}, {"nbf": True},
    {"org_id": None}, {"org_id": ""}, {"org_id": "   "},
    {"wisdom_scopes": None}, {"wisdom_scopes": []},
    {"wisdom_scopes": "wisdom:read"}, {"wisdom_scopes": [1, "wisdom:read"]},
    {"wisdom_scopes": ["wisdom:admin"], "tool_gateway_admin": True},
])
def test_missing_malformed_expired_or_no_read_fails_closed(auth, overrides):
    auth(**overrides)
    assert not is_entitled()
    with pytest.raises(PackagePolicyError, match="unavailable"):
        require_entitlement()


def test_logout_and_token_replacement_have_no_positive_cache(auth):
    state = auth()
    assert is_entitled()
    state.clear()
    assert not is_entitled()
    state.update(logged_in=True, access_token="bad-token")
    assert not is_entitled()
    auth(org_id="second-team")
    assert is_entitled("second-team")
    assert not is_entitled("team-example")
    state["relogin_required"] = True
    assert not is_entitled()


def test_no_refresh_or_runtime_resolution_for_gate(auth, monkeypatch):
    refresh = Mock(side_effect=AssertionError("must not refresh for visibility"))
    monkeypatch.setattr("hermes_cli.auth.resolve_nous_runtime_credentials", refresh)
    assert is_entitled()
    refresh.assert_not_called()


def test_local_work_requires_opt_in_and_current_matching_org(auth, monkeypatch, tmp_path):
    store = WisdomStore(tmp_path / "state")
    store.installation_identity()
    store.verify_installation_identity("team-example")
    config = {"enabled": True, "disclosure_acknowledged_at": "accepted"}
    monkeypatch.setattr("hermes_wisdom.service._config", lambda: config)
    assert local_work_allowed(store)
    service = WisdomService(store=store, client=Mock())
    service.require_setup()
    config["enabled"] = False
    assert not local_work_allowed(store)
    config["enabled"] = True
    auth(wisdom_scopes=[])
    assert not local_work_allowed(store)
    with pytest.raises(PackagePolicyError, match="unavailable"):
        service.require_setup()
    auth(org_id="new-team")
    assert not local_work_allowed(store)
    with pytest.raises(PackagePolicyError, match="organization changed"):
        service.require_setup()


def test_tools_need_entitlement_and_config(auth, monkeypatch):
    from tools.wisdom_tool import available
    from tools.registry import _check_fn_cached
    config = {"enabled": True}
    monkeypatch.setattr("hermes_wisdom.service._config", lambda: config)
    assert available()
    assert _check_fn_cached(available)
    auth(wisdom_scopes=[])
    assert not available()
    assert not _check_fn_cached(available), "registry TTL/grace must not retain entitlement"
    auth()
    config["enabled"] = False
    assert not available()


def test_setup_denied_before_network_for_ineligible_account(auth, tmp_path):
    auth(wisdom_scopes=[])
    client = Mock()
    service = WisdomService(store=WisdomStore(tmp_path / "state"), client=client)
    with pytest.raises(PackagePolicyError, match="unavailable"):
        service.setup(disclosure_accepted=True)
    client.capability.assert_not_called()
    client.register_identity.assert_not_called()


def test_unentitled_status_does_not_probe_gateway(auth, tmp_path):
    auth(wisdom_scopes=[])
    client = Mock()
    service = WisdomService(store=WisdomStore(tmp_path / "state"), client=client)
    assert service.status()["entitled"] is False
    client.capability.assert_not_called()


def test_member_home_no_longer_requires_global_admin(auth, monkeypatch, tmp_path):
    service = WisdomService(store=WisdomStore(tmp_path / "state"))
    monkeypatch.setattr(service, "status", lambda: {
        "configured": True, "gateway_available": True, "capability_advertised": True,
        "entitled": True, "dogfood_admin_claim": False,
    })
    monkeypatch.setattr(service, "require_setup", lambda: None)
    for name in ("search_skills", "list_owner_drafts", "list_installations"):
        monkeypatch.setattr(service, name, lambda: [])
    monkeypatch.setattr(service, "list_candidates", lambda **kw: [])
    monkeypatch.setattr(service, "notifications", lambda **kw: {"events": []})
    assert service.command_home()["counts"]["published"] == 0


def test_profile_context_selects_its_own_claims(monkeypatch, tmp_path):
    from hermes_constants import (
        get_hermes_home, set_hermes_home_override, reset_hermes_home_override,
    )
    tokens = {
        "eligible": jwt.encode({"org_id": "team", "exp": time.time() + 600,
                                "wisdom_scopes": ["wisdom:read"]},
                               "test-only-local-entitlement-key-32-bytes", algorithm="HS256"),
        "unrelated": jwt.encode({"org_id": "other", "exp": time.time() + 600},
                                "test-only-local-entitlement-key-32-bytes", algorithm="HS256"),
    }
    monkeypatch.setattr("hermes_cli.auth_nous.get_nous_auth_status_local", lambda: {
        "logged_in": True, "access_token": tokens[get_hermes_home().name],
    })
    for profile, expected in [("eligible", True), ("unrelated", False), ("eligible", True)]:
        token = set_hermes_home_override(str(tmp_path / profile))
        try:
            assert is_entitled() is expected
        finally:
            reset_hermes_home_override(token)


def test_expiry_rechecked_without_token_replacement(auth, monkeypatch):
    auth(exp=2000)
    monkeypatch.setattr("hermes_wisdom.entitlement.time.time", lambda: 1999)
    assert is_entitled()
    monkeypatch.setattr("hermes_wisdom.entitlement.time.time", lambda: 2000)
    assert not is_entitled()


def test_real_local_auth_snapshot_never_refreshes(monkeypatch):
    token = jwt.encode({"org_id": "team", "exp": time.time() + 600,
                        "wisdom_scopes": ["wisdom:read"]},
                       "test-only-local-entitlement-key-32-bytes", algorithm="HS256")
    state = {"access_token": token, "refresh_token": "unused"}
    monkeypatch.setattr("hermes_cli.auth.get_provider_auth_state", lambda provider: state)
    refresh = Mock(side_effect=AssertionError("local gate must not refresh"))
    monkeypatch.setattr("hermes_cli.auth.resolve_nous_runtime_credentials", refresh)
    assert is_entitled("team")
    state["access_token"] = jwt.encode({"org_id": "team", "exp": 1,
                                        "wisdom_scopes": ["wisdom:read"]},
                                       "test-only-local-entitlement-key-32-bytes", algorithm="HS256")
    assert not is_entitled()
    refresh.assert_not_called()
