"""Held Skills authority follows real OAuth storage, not a cached verification bit."""
import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest


@pytest.fixture
def oauth_skill(tmp_path, monkeypatch):
    from mcp.shared.auth import OAuthToken, OAuthClientMetadata, OAuthMetadata
    from tools import mcp_skills_registry as registry, mcp_skills_cache as cache
    from tools import mcp_tool_discovery as discovery, mcp_tool_loop as loop
    from tools.mcp_skills_protocol import SkillEntry
    from tools.mcp_skills_view import serve_remote_skill
    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import MCPOAuthManager, HermesMCPOAuthProvider, _ProviderEntry

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    registry.clear_runtime_state()
    cfg = {"url": "https://auth-probe.invalid/mcp", "auth": "oauth", "skills": {"enabled": True}}
    fp = registry.server_config_fingerprint(cfg)
    body = b"---\nname: auth-probe\ndescription: Auth context probe\n---\n\n# Synthetic account A document\n"
    uri = "skill://fixture/auth-probe/SKILL.md"
    entry = SkillEntry.model_validate({"uri": uri,
        "frontmatter": {"name": "auth-probe", "description": "Auth context probe"},
        "resources": [{"uri": uri, "digest": "sha256:" + hashlib.sha256(body).hexdigest(), "size": len(body)}]})
    storage = HermesTokenStorage("fixture.team", hermes_home=home)
    storage.save_oauth_metadata(OAuthMetadata.model_validate({"issuer": "https://auth-probe.invalid",
        "authorization_endpoint": "https://auth-probe.invalid/authorize", "token_endpoint": "https://auth-probe.invalid/token",
        "response_types_supported": ["code"]}))

    async def no_browser(*args):
        raise AssertionError("No OAuth network/browser flow permitted")

    provider = HermesMCPOAuthProvider(server_name="fixture.team", server_url=cfg["url"],
        client_metadata=OAuthClientMetadata.model_validate({"redirect_uris": ["http://127.0.0.1:1/callback"]}),
        storage=storage, redirect_handler=no_browser, callback_handler=no_browser)
    provider._hermes_home = str(home)
    manager = MCPOAuthManager()
    from tools import mcp_oauth_manager
    monkeypatch.setattr(mcp_oauth_manager, "get_manager", lambda: manager)
    manager._entries[manager._key("fixture.team", home)] = _ProviderEntry(cfg["url"], {}, provider)

    async def switch(kind):
        # Rotation is deliberately still A-authorized remotely: invalidation is
        # conservative, not a claim that an opaque refreshed token changed user.
        token = "synthetic-account-B" if kind == "account" else "synthetic-account-A"
        if kind == "refresh":
            token = "synthetic-account-A-rotated"
        scope = "skills:none" if kind == "scope" else "skills:read"
        await storage.set_tokens(OAuthToken(access_token=token, token_type="Bearer", scope=scope,
            refresh_token="synthetic-refresh-secret", expires_in=3600))
        assert await manager.invalidate_if_disk_changed("fixture.team", hermes_home=home)
        await provider._initialize()
        assert provider.context.current_tokens.access_token == token

    asyncio.run(switch("unchanged"))
    calls = []
    state = SimpleNamespace(hook=None)

    def authorize():
        tokens = provider.context.current_tokens
        if tokens.access_token == "synthetic-account-B" or tokens.scope == "skills:none":
            raise PermissionError("Synthetic current account/scope is not authorized")

    class Remote:
        async def read_resource(self, requested_uri):
            calls.append("read")
            authorize()
            if state.hook == "read":
                await switch("account")
            return SimpleNamespace(contents=[SimpleNamespace(uri=requested_uri, text=body.decode(),
                mimeType="text/markdown", blob=None)])

    remote = Remote()
    server = SimpleNamespace(session=remote, _skills_epoch=1, _skills_ready_epoch=1,
        _skills_ready_session=remote, _skills_home=str(home), _skills_config_fingerprint=fp,
        _skills_local_opt_in=True, _skills_advertised=True, _skills_list_completed=True,
        _skills_connected=True, _config=cfg, _rpc_lock=asyncio.Lock(), tool_timeout=10,
        _oauth_provider=provider)

    async def remote_get(session, requested_uri):
        calls.append("get")
        authorize()
        if state.hook == "get":
            await switch("account")
        return entry

    monkeypatch.setattr(discovery, "_get_connected_server_for_call", lambda name: server)
    monkeypatch.setattr(loop, "_run_on_mcp_loop", lambda factory, **kw: asyncio.run(factory()))
    monkeypatch.setattr(cache, "get_skill", remote_get)
    registry.publish_live_catalog(home, "fixture.team", fp, [entry])
    record = registry.pin_session(home, "conversation")[0]

    def view():
        return json.loads(serve_remote_skill("mcp:fixture.team:" + uri, file_path=None, task_id=None,
            session_id="conversation", materialize=False, destination=None))

    yield SimpleNamespace(**locals())
    registry.clear_runtime_state()


@pytest.mark.parametrize("kind", ["account", "scope", "refresh", "unchanged", "legacy", "disk_only", "logout", "other_profile"])
@pytest.mark.parametrize("resumed", [False, True])
@pytest.mark.parametrize("cached", [False, True])
def test_held_authority_cannot_cross_oauth_context(oauth_skill, kind, resumed, cached, caplog):
    s = oauth_skill
    raw, _, _, path, _ = s.cache.get_verified_resource(s.record, s.record["resources"][0], s.home, "conversation")
    assert raw == s.body
    s.registry.mark_active(s.record, s.home, "conversation")
    original_manifest = s.record["manifest_fingerprint"]
    if not cached:
        path.unlink()
    if kind == "legacy":
        snapshot = s.home / s.registry._snapshot_rel("conversation")
        data = json.loads(snapshot.read_text())
        for row in data["entries"]:
            row.pop("authorization_context", None)
        snapshot.write_text(json.dumps(data))
    elif kind == "disk_only":
        from mcp.shared.auth import OAuthToken
        # No manager reload/HTTP request: the cached path must see disk changes.
        asyncio.run(s.storage.set_tokens(OAuthToken(access_token="synthetic-account-B", token_type="Bearer")))
    elif kind == "logout":
        s.storage.remove()
    elif kind == "other_profile":
        from tools.mcp_oauth import HermesTokenStorage
        from mcp.shared.auth import OAuthToken
        other_home = s.home.parent / "other-profile"
        other_home.mkdir()
        other = HermesTokenStorage("fixture.team", hermes_home=other_home)
        asyncio.run(other.set_tokens(OAuthToken(access_token="synthetic-account-B", token_type="Bearer")))
    else:
        asyncio.run(s.switch(kind))
    if resumed:
        s.registry.clear_runtime_state()
    # A fresh catalog may omit it; it must not repin the held conversation.
    s.registry.publish_live_catalog(s.home, "fixture.team", s.fp, [])
    before = list(s.calls)
    result = s.view()
    held = s.registry._pin_session_all(s.home, "conversation")[0]
    assert held["manifest_fingerprint"] == original_manifest
    if kind in ("unchanged", "other_profile"):
        assert result["success"] is True and result["raw_content"].encode() == s.body
        assert s.calls == before + ([] if cached else ["read"])
    else:
        assert result["success"] is False
        assert "authorization context" in result["error"].lower()
        assert "raw_content" not in result
        assert s.calls == before
        invalid = json.loads(s.serve_remote_skill("mcp:fixture.team:" + s.uri,
            file_path="references/missing.md", task_id=None, session_id="conversation",
            materialize=False, destination=None))
        assert invalid["success"] is False and "authorization context" in invalid["error"]
        assert "available_files" not in invalid
    # Only credential digests can escape into Skills durable state/diagnostics.
    exposed = json.dumps(result) + caplog.text
    for file in (s.home / "cache" / "mcp-skills").rglob("*.json"):
        exposed += file.read_text()
    for secret in ("synthetic-account-A", "synthetic-account-B", "synthetic-refresh-secret", "skills:none", "skills:read"):
        assert secret not in exposed


@pytest.mark.parametrize("stage", ["get", "read", "scan", "cached_scan", "lazy_get", "list", "logout_lazy", "logout_list"])
def test_oauth_change_during_operation_cannot_publish_or_deliver(oauth_skill, monkeypatch, stage):
    s = oauth_skill
    if stage in ("logout_lazy", "logout_list"):
        from tools.mcp_tool import sdk_httpx

        async def sent_token():
            request = sdk_httpx().Request("POST", s.cfg["url"])
            flow = s.provider.async_auth_flow(request)
            outgoing = await flow.__anext__()
            try:
                return outgoing.headers.get("Authorization")
            finally:
                await flow.aclose()

        assert asyncio.run(sent_token()) == "Bearer synthetic-account-A"
        s.registry.publish_live_catalog(s.home, "fixture.team", s.fp, [])
        s.storage.remove()
        assert asyncio.run(s.manager.invalidate_if_disk_changed("fixture.team", hermes_home=s.home)) is False
        assert asyncio.run(sent_token()) == "Bearer synthetic-account-A"
        if stage == "logout_lazy":
            with pytest.raises(RuntimeError, match="authorization context"):
                s.cache.register_skill_uri("fixture.team", s.uri, s.home, "after-logout")
            assert not (s.home / s.registry._snapshot_rel("after-logout")).exists()
            assert not s.calls
            return
    if stage in ("list", "logout_list"):
        from tools.mcp_tool import MCPServerTask
        from tools.mcp_skills_protocol import SkillsListResult

        class Listing:
            async def send_request(self, request, adapter):
                assert stage != "logout_list", "stale OAuth source must not enter skills/list"
                await s.switch("account")
                return SkillsListResult(skills=[s.entry])

        async def discover():
            server = MCPServerTask("fixture.team")
            server._config = s.cfg
            server._auth_type = "oauth"
            assert server._build_oauth_auth(s.cfg["url"], s.cfg) is s.provider
            server.initialize_result = SimpleNamespace(capabilities=SimpleNamespace(
                extensions={"io.modelcontextprotocol/skills": {}}))
            server.session = Listing()
            await server._discover_skills()
            return server

        server = asyncio.run(discover())
        assert not server._skills_connected
        assert "authorization context" in (server._skills_diagnostic or "")
        assert s.registry.pin_session(s.home, "new-conversation") == ()
        return
    if stage == "lazy_get":
        s.state.hook = "get"
        with pytest.raises(RuntimeError, match="authorization context"):
            s.cache.register_skill_uri("fixture.team", s.uri, s.home, "new-conversation")
        assert not (s.home / s.registry._snapshot_rel("new-conversation")).exists()
        return
    if stage == "cached_scan":
        s.cache.get_verified_resource(s.record, s.record["resources"][0], s.home, "conversation")
    if stage in ("scan", "cached_scan"):
        scan = s.cache.scan_resource_bytes

        def changing_scan(*args, **kwargs):
            result = scan(*args, **kwargs)
            asyncio.run(s.switch("account"))
            return result

        monkeypatch.setattr(s.cache, "scan_resource_bytes", changing_scan)
    else:
        s.state.hook = stage
    with pytest.raises(RuntimeError, match="authorization context"):
        s.cache.get_verified_resource(s.record, s.record["resources"][0], s.home, "conversation")
    if stage != "cached_scan":
        assert not list((s.home / "cache" / "mcp-skills").rglob("content"))
    if stage == "get":
        assert not s.registry._pin_session_all(s.home, "conversation")[0]["get_verified"]

@pytest.mark.parametrize("kind", ["account", "legacy", "unchanged_offline"])
@pytest.mark.parametrize("resumed", [False, True])
@pytest.mark.parametrize("surface", ["list", "pin", "ambiguous"])
def test_fresh_metadata_requires_current_held_authority(oauth_skill, monkeypatch, kind, resumed, surface):
    s = oauth_skill
    from tools.skills_tool import skills_list
    raw = s.entry.model_dump(mode="json")
    raw["uri"] = raw["uri"].replace("/auth-probe/", "/second-private-root/auth-probe/")
    raw["resources"][0]["uri"] = raw["uri"]
    s.registry.publish_live_catalog(s.home, "fixture.team", s.fp, [s.entry, raw])
    sid = "metadata"
    pinned = s.registry.pin_session(s.home, sid)
    snapshot = s.home / s.registry._snapshot_rel(sid)
    if kind == "legacy":
        data = json.loads(snapshot.read_text())
        for row in data["entries"]:
            row.pop("authorization_context", None)
        snapshot.write_text(json.dumps(data))
    else:
        asyncio.run(s.switch("account" if kind == "account" else "unchanged"))
    frozen = snapshot.read_bytes()
    if resumed:
        s.registry.clear_runtime_state()
    s.registry.publish_live_catalog(s.home, "fixture.team", s.fp, [])
    if kind == "unchanged_offline":
        monkeypatch.setattr(s.discovery, "_get_connected_server_for_call", lambda name: None)
    before = list(s.calls)
    allowed = kind == "unchanged_offline"
    if surface == "list":
        result = json.loads(skills_list(category="mcp:fixture.team", session_id=sid))
        assert bool(result.get("skills")) is allowed
        if not allowed:
            assert "second-private-root" not in json.dumps(result)
            assert "Auth context probe" not in json.dumps(result)
    elif surface == "pin":
        assert bool(s.registry.pin_session(s.home, sid)) is allowed
    else:
        result = json.loads(s.serve_remote_skill("mcp:fixture.team:auth-probe", file_path=None,
            task_id=None, session_id=sid, materialize=False, destination=None))
        assert ("second-private-root" in result["error"]) is allowed
        assert result["success"] is False
    assert s.calls == before
    assert snapshot.read_bytes() == frozen
    assert pinned[0]["authorization_context"] == s.record["authorization_context"]
    # Ownership must survive filtering: generic resource dispatch must not fall through.
    assert s.registry.resolve_catalog_resource("fixture.team", s.uri, s.home, sid) is not None


@pytest.mark.parametrize("surface", ["execution", "consented_execution", "generic_list", "generic_read", "view_read", "view_approval"])
def test_fresh_sibling_responses_do_not_disclose_ineligible_origins(oauth_skill, monkeypatch, surface):
    s = oauth_skill
    from tools import mcp_skills_consent as consent, mcp_tool_handlers as handlers
    from tools import approval_prompt
    prompts = []

    def decline(*args, **kwargs):
        prompts.append(args)
        if surface == "view_approval":
            asyncio.run(s.switch("account"))
        return "decline"

    monkeypatch.setattr(approval_prompt, "request_elicitation_consent", decline)
    if surface in ("view_read", "view_approval"):
        if surface == "view_read":
            s.registry.mark_active(s.record, s.home, "conversation")
            s.state.hook = "read"
        result = s.view()
        assert result["success"] is False
        assert "origin" not in result
        assert s.uri not in json.dumps(result)
        return
    s.registry.mark_active(s.record, s.home, "conversation")
    if surface == "consented_execution":
        s.registry.record_execution_consent(s.record, s.home, "conversation")
    asyncio.run(s.switch("account"))
    if surface.endswith("execution"):
        result = consent.enforce_remote_skill_gate("terminal", {"command": "true"}, session_id="conversation")
        assert result is not None
        assert s.uri not in result
        assert not prompts
        assert consent.enforce_remote_skill_gate("skills_list", {}, session_id="conversation") is None
    elif surface == "generic_list":
        row = SimpleNamespace(uri=s.uri, name="private-name", description="private-description")
        result = handlers._render_catalog_resource_list([row], "fixture.team", {}, {"session_id": "conversation"})
        assert "private-name" not in json.dumps(result)
        assert s.uri not in json.dumps(result)
    else:
        result = SimpleNamespace(contents=[SimpleNamespace(uri=s.uri, text=s.body.decode(), mimeType="text/markdown", blob=None)])
        with pytest.raises(RuntimeError, match="authorization context"):
            handlers._render_catalog_read_resource(result, "fixture.team", {"uri": s.uri}, {"session_id": "conversation"})
