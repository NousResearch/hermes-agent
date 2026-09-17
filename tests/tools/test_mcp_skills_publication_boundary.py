"""Observed authority invalidation removes the whole accumulated projection."""
import asyncio
import json
from types import SimpleNamespace

import pytest
from tests.tools.test_mcp_skills_auth_context import oauth_skill  # noqa: F401


@pytest.mark.parametrize("route", ["native", "generic"])
@pytest.mark.parametrize("change", [False, True])
def test_later_check_invalidates_earlier_rows(oauth_skill, monkeypatch, route, change):
    s = oauth_skill
    from tools import mcp_tool_handlers as handlers
    second = s.entry.model_dump(mode="json")
    second["uri"] = second["uri"].replace("/auth-probe/", "/second-root/auth-probe/")
    second["resources"][0]["uri"] = second["uri"]
    s.registry.publish_live_catalog(s.home, "fixture.team", s.fp, [s.entry, second])
    s.registry.pin_session(s.home, "two-rows")
    snapshot = s.home / s.registry._snapshot_rel("two-rows")
    before = snapshot.read_bytes()
    original = s.cache.require_skill_eligibility
    checks = 0
    returned = False

    async def listing(*args):
        return SimpleNamespace(resources=[SimpleNamespace(uri=uri, name="PRIVATE_A")
            for uri in (s.uri, second["uri"], "resource://ordinary/doc")], nextCursor=None)

    def require(record, home=None):
        nonlocal checks
        if route == "native" or returned:
            checks += 1
            if checks == 2 and change:
                asyncio.run(s.switch("account"))
        return original(record, home)

    def run(factory, **kwargs):
        nonlocal returned
        result = asyncio.run(factory())
        returned = True
        return result

    monkeypatch.setattr(s.cache, "require_skill_eligibility", require)
    if route == "generic":
        from tools import mcp_tool_loop
        monkeypatch.setattr(mcp_tool_loop, "_run_on_mcp_loop", run)
        s.remote.list_resources = listing
        result = json.loads(handlers._make_list_resources_handler("fixture.team", 10)(
            {}, session_id="two-rows"))["resources"]
        assert any(row["uri"] == "resource://ordinary/doc" for row in result)
        result = [row for row in result if row["uri"] != "resource://ordinary/doc"]
    else:
        result = s.registry.list_remote_skills(s.home, "two-rows")
    assert snapshot.read_bytes() == before
    assert checks >= 2
    assert bool(result) is not change, result


@pytest.mark.parametrize("change", ["none", "auth", "source"])
@pytest.mark.parametrize("route", ["lazy", "held_get", "held_read"])
def test_lazy_get_exception_fences_captured_source(oauth_skill, monkeypatch, change, route):
    s = oauth_skill
    private = "ACCOUNT_A_PRIVATE_LAZY_GET_ERROR"
    sid = "conversation"
    if route == "lazy":
        s.registry.publish_live_catalog(s.home, "fixture.team", s.fp, [])
        sid = "lazy-empty"
        s.registry.pin_session(s.home, sid)
    else:
        s.registry.mark_active(s.record, s.home, sid)

    async def fail(*args):
        if change == "auth":
            await s.switch("account")
        elif change == "source":
            s.server.session = SimpleNamespace()
            s.server._skills_ready_session = s.server.session
        raise RuntimeError(private)

    if route == "held_read":
        monkeypatch.setattr(s.remote, "read_resource", fail)
    else:
        monkeypatch.setattr(s.cache, "get_skill", fail)
    result = json.loads(s.serve_remote_skill("mcp:fixture.team:" + s.uri,
        file_path=None, task_id=None, session_id=sid, materialize=False, destination=None))
    assert (private in json.dumps(result)) is (change == "none"), result
