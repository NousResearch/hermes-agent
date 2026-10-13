"""Behaviour contracts for the skills-over-MCP instance manager.

The claims under test are the ones the design rests on: nothing is fetched before
it is needed, a half-listing is never published as if it were whole, and a result
that belongs to a connection the host has already left is discarded rather than
merged. Each is asserted through the fake server's call log, so "no read happened"
is proven by the absence of a call rather than by inspecting our own bookkeeping.
"""

import asyncio

import pytest

from agent.skills_contracts import BackendState, CatalogState, SkillError, SkillErrorCode, SkillsSupport
from tests.fakes.fake_mcp_skills import FakeSkill, FakeSkillsServer, FakeSkillsSender
from tools.mcp_skills_manager import SkillsOverMcpManager

SERVER = "acme"
SCOPE = "profile-a"


def _skills(count: int = 3, **kw):
    return [FakeSkill(uri=f"skill://team/topic{i}/SKILL.md", name=f"topic{i}",
                      description=f"skill {i}", files={"SKILL.md": f"---\nname: topic{i}\n---\nbody {i}\n"},
                      **kw)
            for i in range(count)]


def _manager(server: FakeSkillsServer, *, connected: bool = True):
    sender = FakeSkillsSender(server, connected=connected)
    manager = SkillsOverMcpManager(sender=sender)
    session = object()
    manager.observe_connection(SERVER, SCOPE, initialize_result=server.initialize_result(), session=session)
    return manager, sender, session


def _seen(server: FakeSkillsServer, method: str) -> int:
    return server.calls.count(method)


# ── capability gate ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_server_without_the_extension_never_receives_a_skills_request():
    server = FakeSkillsServer(_skills(), advertise=False)
    manager, _sender, _s = _manager(server)
    assert manager.instance_view(SERVER, SCOPE).support is SkillsSupport.UNSUPPORTED
    with pytest.raises(SkillError) as exc:
        await manager.discover(SERVER, SCOPE)
    assert exc.value.code is SkillErrorCode.UNSUPPORTED_EXTENSION
    assert server.calls == [], "an unsupported server must not be probed"


@pytest.mark.asyncio
async def test_an_empty_but_legal_catalog_is_still_a_supported_server():
    """``[]`` is a valid listing, not a failed one — the instance does not lose its channel."""
    manager, _sender, _s = _manager(FakeSkillsServer([]))
    snapshot = await manager.discover(SERVER, SCOPE)
    assert snapshot.entries == {}
    assert manager.instance_view(SERVER, SCOPE).support is SkillsSupport.SUPPORTED


@pytest.mark.asyncio
async def test_an_advertised_but_unimplemented_method_disables_only_this_channel():
    server = FakeSkillsServer(_skills())
    server.unimplemented.add("skills/list")
    manager, _sender, _s = _manager(server)
    with pytest.raises(SkillError) as exc:
        await manager.discover(SERVER, SCOPE)
    assert exc.value.code is SkillErrorCode.PROTOCOL_MISMATCH
    view = manager.instance_view(SERVER, SCOPE)
    assert view.support is SkillsSupport.PROTOCOL_MISMATCH
    assert view.catalog_state is CatalogState.ABSENT


# ── discovery ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_discovery_reads_no_resource_content():
    """The whole point of the metadata pass: connect and enumerate, fetch nothing."""
    server = FakeSkillsServer(_skills(5), page_size=2)
    manager, _sender, _s = _manager(server)
    snapshot = await manager.discover(SERVER, SCOPE)
    assert len(snapshot.entries) == 5
    assert _seen(server, "resources/read") == 0
    assert _seen(server, "skills/get") == 0
    assert _seen(server, "skills/list") == 3, "5 skills at 2 per page = 3 pages"


@pytest.mark.asyncio
async def test_discovery_follows_every_page_and_reports_completion():
    server = FakeSkillsServer(_skills(7), page_size=3)
    manager, _sender, _s = _manager(server)
    snapshot = await manager.discover(SERVER, SCOPE)
    assert snapshot.pagination_finished
    assert {e.name for e in snapshot.entries.values()} == {f"topic{i}" for i in range(7)}


@pytest.mark.asyncio
async def test_a_repeated_cursor_fails_the_whole_refresh():
    """A server looping on a cursor must not be published as if the listing had ended."""
    server = FakeSkillsServer(_skills(5), page_size=2)
    server.repeat_cursor = "skill://team/topic1/SKILL.md"
    manager, _sender, _s = _manager(server)
    with pytest.raises(SkillError) as exc:
        await manager.discover(SERVER, SCOPE)
    assert exc.value.code is SkillErrorCode.INCOMPLETE_CATALOG
    assert manager.published(SCOPE) == [], "a partial catalog is never published"


@pytest.mark.asyncio
async def test_a_later_page_failure_keeps_the_previous_batch_but_marks_it_failed():
    server = FakeSkillsServer(_skills(4), page_size=2)
    manager, _sender, _s = _manager(server)
    first = await manager.discover(SERVER, SCOPE)
    assert len(first.entries) == 4

    server.fail_next_list = True
    server.skills = _skills(6)
    with pytest.raises(SkillError):
        await manager.discover(SERVER, SCOPE)

    held = manager.snapshot(SERVER, SCOPE)
    assert held is not None, "the last good batch stays available as a diagnostic"
    assert len(held.entries) == 4
    assert held.snapshot_revision == first.snapshot_revision
    assert held.refresh_status.startswith("failed")


@pytest.mark.asyncio
async def test_one_invalid_entry_is_quarantined_without_losing_the_rest():
    server = FakeSkillsServer(_skills(3))
    server.digest_overrides = {server.skills[1].uri: {"SKILL.md": "sha256:not-a-digest"}}
    manager, _sender, _s = _manager(server)
    snapshot = await manager.discover(SERVER, SCOPE)
    assert len(snapshot.entries) == 2
    assert "topic1" not in {e.name for e in snapshot.entries.values()}
    assert "quarantined" in snapshot.diagnostics
    assert manager.instance_view(SERVER, SCOPE).quarantined


@pytest.mark.asyncio
async def test_partial_scope_keys_keep_profiles_apart():
    server = FakeSkillsServer(_skills(2))
    manager, _sender, _s = _manager(server)
    await manager.discover(SERVER, SCOPE)
    assert manager.entries("profile-b") == [], "another profile sees none of this catalog"
    assert len(manager.entries(SCOPE)) == 2


# ── connection generation ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_new_session_retires_the_old_catalog():
    server = FakeSkillsServer(_skills(3))
    manager, _sender, _session = _manager(server)
    await manager.discover(SERVER, SCOPE)
    assert len(manager.entries(SCOPE)) == 3

    manager.observe_connection(SERVER, SCOPE, initialize_result=server.initialize_result(),
                               session=object())
    assert manager.entries(SCOPE) == []
    assert manager.instance_view(SERVER, SCOPE).generation == 2


@pytest.mark.asyncio
async def test_a_result_from_a_superseded_connection_is_discarded():
    """A late page from the old session must not be published onto the new one."""
    server = FakeSkillsServer(_skills(3))
    manager, _sender, _s = _manager(server)

    original = manager._list_all

    async def _steal_a_reconnect(inst):
        manager.observe_connection(SERVER, SCOPE, initialize_result=server.initialize_result(),
                                   session=object())
        return await original(inst)

    manager._list_all = _steal_a_reconnect
    with pytest.raises(SkillError) as exc:
        await manager.discover(SERVER, SCOPE)
    assert exc.value.code is SkillErrorCode.VERSION_CHANGED
    assert manager.published(SCOPE) == []


def test_a_confirmed_disconnect_removes_the_catalog_entirely():
    """No on-disk copy stands in for the connection the approval was bound to."""
    server = FakeSkillsServer(_skills(2))
    manager, _sender, _s = _manager(server)
    asyncio.run(manager.discover(SERVER, SCOPE))
    assert len(manager.entries(SCOPE)) == 2
    manager.mark_disconnected(SERVER, SCOPE, "transport closed")
    assert manager.entries(SCOPE) == []
    assert manager.instance_view(SERVER, SCOPE).catalog_state is CatalogState.ABSENT


# ── reads ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_read_verifies_against_the_held_manifest_and_fetches_only_that_file():
    server = FakeSkillsServer(_skills(2))
    manager, _sender, _s = _manager(server)
    await manager.discover(SERVER, SCOPE)
    entry = manager.entries(SCOPE)[0]
    before = _seen(server, "resources/read")
    read = manager.read_resource(SERVER, SCOPE, entry, entry.identity.locator)
    assert read.verification.ok
    assert read.text.startswith("---")
    assert _seen(server, "resources/read") == before + 1
    assert read.skill_root_uri == "skill://team/topic0"


@pytest.mark.asyncio
async def test_bytes_that_no_longer_match_the_manifest_are_refused_and_retire_the_catalog():
    server = FakeSkillsServer(_skills(2))
    manager, _sender, _s = _manager(server)
    await manager.discover(SERVER, SCOPE)
    entry = manager.entries(SCOPE)[0]
    server.skills[0].files = {"SKILL.md": "---\nname: topic0\n---\ntampered body\n"}
    with pytest.raises(SkillError) as exc:
        manager.read_resource(SERVER, SCOPE, entry, entry.identity.locator)
    assert exc.value.code is SkillErrorCode.VERIFICATION_FAILED
    assert manager.snapshot(SERVER, SCOPE) is None, "the entry whose binding failed is not reusable"


@pytest.mark.asyncio
async def test_a_resource_outside_the_held_manifest_is_never_requested():
    server = FakeSkillsServer(_skills(2))
    manager, _sender, _s = _manager(server)
    await manager.discover(SERVER, SCOPE)
    entry = manager.entries(SCOPE)[0]
    with pytest.raises(SkillError) as exc:
        manager.read_resource(SERVER, SCOPE, entry, "skill://team/other/SKILL.md")
    assert exc.value.code is SkillErrorCode.NOT_PROVIDED
    assert _seen(server, "resources/read") == 0


@pytest.mark.asyncio
async def test_a_dynamic_manifest_is_refused_rather_than_loaded_unverified():
    server = FakeSkillsServer(_skills(1, dynamic=True))
    manager, _sender, _s = _manager(server)
    await manager.discover(SERVER, SCOPE)
    entry = manager.entries(SCOPE)[0]
    with pytest.raises(SkillError) as exc:
        manager.read_resource(SERVER, SCOPE, entry, entry.identity.locator)
    assert exc.value.code is SkillErrorCode.DYNAMIC_DISABLED


@pytest.mark.asyncio
async def test_an_explicit_uri_resolves_even_when_the_listing_never_offered_it():
    """A listing is not a claim of completeness, so absence from it proves nothing."""
    hidden = FakeSkill(uri="skill://team/hidden/SKILL.md", name="hidden",
                       files={"SKILL.md": "---\nname: hidden\n---\nx\n"})
    server = FakeSkillsServer(_skills(2))
    server.skills.append(hidden)
    server.unlisted.add(hidden.uri)
    manager, _sender, _s = _manager(server)
    snapshot = await manager.discover(SERVER, SCOPE)
    assert "hidden" not in {e.name for e in snapshot.entries.values()}
    assert snapshot.pagination_finished, "a partial listing is still a complete pagination"

    entry = manager.resolve_entry(SERVER, SCOPE, hidden.uri, refresh=True)
    assert entry.name == "hidden"


@pytest.mark.asyncio
async def test_an_unknown_uri_is_not_provided_rather_than_unavailable():
    server = FakeSkillsServer(_skills(2))
    manager, _sender, _s = _manager(server)
    await manager.discover(SERVER, SCOPE)
    with pytest.raises(SkillError) as exc:
        manager.resolve_entry(SERVER, SCOPE, "skill://team/nope/SKILL.md", refresh=True)
    assert exc.value.code is SkillErrorCode.NOT_PROVIDED


@pytest.mark.asyncio
async def test_a_zero_ttl_makes_metadata_expire_immediately_and_reconfirms_on_load():
    """ttlMs=0 is the server saying "do not treat this as current", not "cache forever"."""
    server = FakeSkillsServer(_skills(2), ttl_ms=0)
    manager, _sender, _s = _manager(server)
    snapshot = await manager.discover(SERVER, SCOPE)
    assert not snapshot.is_fresh()
    assert len(manager.entries(SCOPE)) == 2, "stale metadata still routes"

    entry = manager.entries(SCOPE)[0]
    before = _seen(server, "skills/get")
    refreshed = manager.resolve_entry(SERVER, SCOPE, entry.identity.locator)
    assert refreshed.name == entry.name
    assert _seen(server, "skills/get") == before + 1, "the load re-confirmed its own entry"
    assert _seen(server, "skills/list") == 1, "and did not re-list the whole catalog"


@pytest.mark.asyncio
async def test_a_healthy_ttl_serves_a_load_from_the_held_snapshot_without_an_extra_round_trip():
    server = FakeSkillsServer(_skills(2), ttl_ms=600_000)
    manager, _sender, _s = _manager(server)
    await manager.discover(SERVER, SCOPE)
    entry = manager.entries(SCOPE)[0]
    assert manager.resolve_entry(SERVER, SCOPE, entry.identity.locator) is entry
    assert _seen(server, "skills/get") == 0


def test_a_background_refresh_is_not_scheduled_when_no_loop_is_running():
    server = FakeSkillsServer(_skills(1))
    manager, _sender, _s = _manager(server)
    assert manager.request_refresh(SERVER, SCOPE) is False


def test_a_disconnected_sender_reports_unavailable():
    server = FakeSkillsServer(_skills(1))
    manager, _sender, _s = _manager(server, connected=False)
    with pytest.raises(SkillError) as exc:
        asyncio.run(manager.discover(SERVER, SCOPE))
    assert exc.value.code is SkillErrorCode.UNAVAILABLE


def test_transport_state_is_reported_separately_from_catalog_state():
    server = FakeSkillsServer(_skills(1), advertise=False)
    manager, _sender, _s = _manager(server)
    view = manager.instance_view(SERVER, SCOPE)
    assert view.transport_state is BackendState.READY
    assert view.support is SkillsSupport.UNSUPPORTED
    assert view.catalog_state is CatalogState.ABSENT
