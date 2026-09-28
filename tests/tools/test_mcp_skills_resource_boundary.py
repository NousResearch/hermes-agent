"""Catalog-owned utility output must be current at whole-response publication."""
import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

_spec = importlib.util.spec_from_file_location(
    "resource_boundary_fixture", Path(__file__).with_name("test_mcp_skills_auth_context.py"))
assert _spec is not None and _spec.loader is not None
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)
oauth_skill = _fixture.oauth_skill


@pytest.mark.parametrize("change", [False, True])
def test_resource_list_drops_prior_rows_on_late_auth_change(oauth_skill, monkeypatch, change):
    s = oauth_skill
    from tools import mcp_tool_handlers as handlers, mcp_skills_scan as scan
    original = scan.scan_resource_listing_metadata
    count = 0

    def scanning(*args, **kwargs):
        nonlocal count
        count += 1
        result = original(*args, **kwargs)
        if count == 2 and change:
            asyncio.run(s.switch("account"))
        return result

    monkeypatch.setattr(scan, "scan_resource_listing_metadata", scanning)
    second = s.entry.model_dump(mode="json")
    second["uri"] = second["uri"].replace("/auth-probe/", "/second-root/auth-probe/")
    second["resources"][0]["uri"] = second["uri"]
    s.registry.publish_live_catalog(s.home, "fixture.team", s.fp, [s.entry, second])
    s.registry.pin_session(s.home, "two-rows")
    snapshot = s.home / s.registry._snapshot_rel("two-rows")
    before = snapshot.read_bytes()
    rows = [SimpleNamespace(uri=uri, name="PRIVATE_A_NAME", description="Private A description")
            for uri in (s.uri, second["uri"])]
    ordinary = SimpleNamespace(uri="resource://ordinary/doc", name="ordinary")
    result = handlers._render_catalog_resource_list(
        [*rows, ordinary], "fixture.team", {}, {"session_id": "two-rows"})
    assert snapshot.read_bytes() == before
    assert ("PRIVATE_A_NAME" in json.dumps(result)) is not change
    assert any(row["uri"] == ordinary.uri for row in result["resources"])
    assert count == 2


@pytest.mark.parametrize("operation", ["read", "list"])
@pytest.mark.parametrize("stage", ["rpc_error", "recovery_error", "rpc_success", "scan_error", "return_success"])
@pytest.mark.parametrize("owned,change", [(True, True), (True, False), (False, True)])
def test_generic_resource_completion_fences_held_authority(
        oauth_skill, monkeypatch, operation, stage, owned, change):
    s = oauth_skill
    from tools import mcp_tool_handlers as handlers, mcp_tool_loop as loop
    private = "ACCOUNT_A_PRIVATE_DESCRIPTION"
    uri = s.uri if owned else "resource://ordinary/doc"
    sid = "conversation" if owned else "ordinary-session"
    if not owned:
        s.registry.publish_live_catalog(s.home, "fixture.team", s.fp, [])
        s.registry.pin_session(s.home, sid)

    async def rpc(*args, **kwargs):
        if stage == "rpc_error":
            if change:
                await s.switch("account")
            raise RuntimeError("Resource failed: " + private)
        if stage == "recovery_error":
            raise RuntimeError("retryable")
        if stage == "rpc_success" and change:
            await s.switch("account")
        if operation == "read":
            return SimpleNamespace(contents=[SimpleNamespace(
                uri=uri, text=s.body.decode(), mimeType="text/markdown", blob=None)])
        return SimpleNamespace(resources=[SimpleNamespace(uri=uri, name=private)], nextCursor=None)

    monkeypatch.setattr(s.remote, "read_resource", rpc)
    s.remote.list_resources = rpc

    if stage == "scan_error" and owned:
        from tools import mcp_skills_scan as scan
        original = (scan.scan_resource_bytes if operation == "read"
                    else scan.scan_resource_listing_metadata)

        def scanning(*args, **kwargs):
            result = original(*args, **kwargs)
            if change:
                s.storage.remove()
                raise RuntimeError(private)
            return result

        monkeypatch.setattr(scan, "scan_resource_bytes" if operation == "read"
                            else "scan_resource_listing_metadata", scanning)

    def recover(*args):
        if change:
            asyncio.run(s.switch("account"))
        return json.dumps({"error": private})

    if stage == "recovery_error":
        monkeypatch.setattr(handlers, "_handle_auth_error_and_retry", recover)
    if stage == "return_success":
        def run(factory, **kwargs):
            result = asyncio.run(factory())
            if change:
                asyncio.run(s.switch("account"))
            return result
        monkeypatch.setattr(loop, "_run_on_mcp_loop", run)
    handler = (handlers._make_read_resource_handler if operation == "read"
               else handlers._make_list_resources_handler)("fixture.team", 10)
    result = json.loads(handler({"uri": uri}, session_id=sid))
    denied = owned and change
    if denied:
        assert private not in json.dumps(result)
        if operation == "list" and stage in {"return_success", "rpc_success", "scan_error"}:
            assert result["resources"] == []
        else:
            assert "authorization context" in result["error"]
        assert s.body.decode() not in json.dumps(result)
    elif stage in {"return_success", "rpc_success", "scan_error"}:
        if operation == "read":
            assert result["result"] == s.body.decode()
        else:
            assert result["resources"][0]["name"] == private
    else:
        assert private in result["error"]
