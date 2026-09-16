"""Skills authority remains profile-owned as upstream shares ordinary transports."""
from types import SimpleNamespace

import pytest


def test_skills_transport_never_adopted_across_profile_homes():
    from tools.mcp_tool_registration import _same_server_route

    ordinary = {"url": "https://fixture.invalid/mcp"}
    skills = {**ordinary, "skills": {"enabled": True}}
    for owner, adopter in ((skills, ordinary), (ordinary, skills), (skills, skills)):
        server = SimpleNamespace(_config=owner)
        assert not _same_server_route(server, adopter, cross_profile=True)
        if owner == adopter:
            assert _same_server_route(server, adopter, cross_profile=False)
    assert _same_server_route(SimpleNamespace(_config=ordinary), ordinary, cross_profile=True)


def test_revoke_catalog_uses_source_home_not_calling_home(tmp_path, monkeypatch):
    from tools.mcp_tool import MCPServerTask
    from tools import mcp_skills_registry as registry
    from tests.tools.test_mcp_skills_transport import _entry

    owner, caller = tmp_path / "owner", tmp_path / "caller"
    owner.mkdir()
    caller.mkdir()
    registry.clear_runtime_state()
    try:
        for home in (owner, caller):
            registry.publish_live_catalog(home, "fixture", "config", [_entry()])
        server = MCPServerTask("fixture")
        server._skills_home = str(owner)
        monkeypatch.setenv("HERMES_HOME", str(caller))
        server._revoke_skills_readiness()
        assert registry.pin_session(owner, "new-owner") == ()
        assert len(registry.pin_session(caller, "new-caller")) == 1
        # Repeated teardown without a publication cannot erase an ambient owner.
        server._revoke_skills_readiness()
        assert len(registry.pin_session(caller, "another-caller")) == 1
    finally:
        registry.clear_runtime_state()


def test_issuer_bookkeeping_is_bound_to_adopted_provider(oauth_skill):
    from tools.mcp_skills_auth import authorization_context, source_authorization_context
    import json

    s = oauth_skill
    original = source_authorization_context("fixture.team", s.server, s.home)
    assert original == authorization_context("fixture.team", s.home)
    from tools.mcp_oauth import _safe_filename
    token_file = s.home / "mcp-tokens" / f"{_safe_filename('fixture.team')}.json"
    payload = json.loads(token_file.read_text())
    assert payload["hermes_issuer"] == "https://auth-probe.invalid"
    payload["hermes_issuer"] = "https://different-issuer.invalid"
    token_file.write_text(json.dumps(payload))
    assert authorization_context("fixture.team", s.home) != original
    with pytest.raises(RuntimeError, match="not adopted"):
        source_authorization_context("fixture.team", s.server, s.home)


from tests.tools.test_mcp_skills_auth_context import oauth_skill  # noqa: E402,F401
